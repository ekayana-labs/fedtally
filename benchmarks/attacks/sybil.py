"""Sybil influence with and without credential gating.

An adversary enrols k fake sites beside the honest ones on the polygenic risk
score task. Each fake trains on a made-up cohort with its labels flipped and
claims an honest site's weight, so its pull on the model is its share of the
sites. In both settings the sponsor lists every applicant in the plan, as an
open call for sites would.

Open: no admission policy. The fakes join and the model is measured.
Gated: the plan admits on credentials. The fakes hold none that count, in one
of three ways: an empty wallet, credentials they signed themselves, or a copy
of an honest site's wallet. Their Join is refused and the honest sites carry
on. This runs the aggregator and helper as the `fedtally` binary and every
site through `ft.run_site`, so a fake site is a real client with a real key.

The cost model of the paper's Proposition 1 is printed beside the
measurements as arithmetic. Nothing here measures a cost.

    cargo build --release -p fedtally-cli
    python benchmarks/attacks/sybil.py --out benchmarks/results/attack-sybil.json
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

import common
import fedtally as ft
from common import prs

WALLETS = ["empty", "self_issued", "copied"]
KINDS = ["AffiliationAndRole", "EthicsApproval", "ControlledAccessGrants", "Provenance"]


def binary():
    built = Path(__file__).resolve().parents[2] / "target" / "release" / "fedtally"
    found = os.environ.get("FEDTALLY_BIN") or shutil.which("fedtally") or built
    if not Path(found).exists():
        raise SystemExit("build the binary first: cargo build --release -p fedtally-cli")
    return str(found)


def trainers(cohorts, honest_sites, sybils, steps):
    flipped = [(x, 1 - y) for x, y in cohorts[honest_sites : honest_sites + sybils]]
    return [common.honest(c, steps) for c in cohorts[:honest_sites] + flipped]


def forge(root, site, did, wallet):
    """Replaces what a fake site holds for the plan's policy."""
    path = root / f"site-{site}.standing.json"
    standing = json.loads(path.read_text())
    if wallet == "empty":
        standing["credentials"] = []
    elif wallet == "copied":
        standing = json.loads((root / "site-0.standing.json").read_text())
    else:
        standing["credentials"] = []
        for kind in KINDS:
            subject = standing["dataset"] if kind == "Provenance" else did
            out = root / f"forged-{site}-{kind}.json"
            subprocess.run(
                [binary(), "issue", "--key", root / f"site-{site}.key", "--type", kind,
                 "--subject", subject, "--out", out],
                check=True, capture_output=True,
            )  # fmt: skip
            standing["credentials"].append(out.read_text().strip())
    path.write_text(json.dumps(standing))


def gated(site_trainers, honest_sites, rounds, wallet, min_cohort, join_secs):
    """One run that admits on credentials. Sites after `honest_sites` are fakes."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        federation = ft.LocalFederation(
            root, common.START, sites=len(site_trainers), helpers=1, rounds=rounds,
            credentials=True, min_cohort=min_cohort, join_secs=join_secs,
        )  # fmt: skip
        federation.save()
        for site in range(honest_sites, len(site_trainers)):
            forge(root, site, federation.native.site_dids[site], wallet)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            addr = f"127.0.0.1:{s.getsockname()[1]}"

        def spawn(*args):
            return subprocess.Popen(
                [binary(), *map(str, args)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        aggregator = spawn(
            "aggregator", "--key", root / "aggregator.key", "--plan", root / "plan.jsonl",
            "--model", root / "initial.safetensors", "--dir", root / "aggregator",
            "--listen", addr, "--status-dir", root / "status",
        )  # fmt: skip
        helper = spawn(
            "helper", "--key", root / "helper-0.key", "--plan", root / "plan.jsonl",
            "--dir", root / "helper-0", "--connect", addr,
        )  # fmt: skip
        results = [None] * len(site_trainers)

        def site(i):
            results[i] = ft.run_site(
                key=root / f"site-{i}.key", plan=root / "plan.jsonl",
                grant=root / f"site-{i}.grant.json", connect=addr, dir=root / f"site-{i}",
                layout=federation.layout, train=site_trainers[i],
                standing=root / f"site-{i}.standing.json", status_dir=root / "status",
                give_up_secs=120,
            )  # fmt: skip

        threads = [threading.Thread(target=site, args=(i,)) for i in range(len(site_trainers))]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            aggregator.wait(timeout=120)
            helper.wait(timeout=120)
        finally:
            aggregator.kill()
            helper.kill()
        honest = results[:honest_sites]
        if not all(r["ok"] for r in honest):
            return None, [r.get("error") for r in honest if not r["ok"]]
        report = ft.verify(root / "aggregator" / "log.jsonl", honest[0]["final_model"])
        model = {k: np.array(v) for k, v in ft.load_model(honest[0]["final_model"]).items()}
        fakes = results[honest_sites:]
        refusals = {re.sub(r"did:key:\w+", "<did>", r["error"]) for r in fakes if not r["ok"]}
        outcome = {
            "finished": report["finished"],
            "complete": report["complete"],
            "joined": len(report["joined"]),
            "sybils_admitted": sum(r["ok"] for r in fakes),
            "sybils_refused": sum(not r["ok"] for r in fakes),
            "refusals": sorted(refusals),
            "cohorts": [len(r["included"]) for r in report["rounds"]],
        }
        return model, outcome


def gated_with_retries(*args, attempts=3):
    for attempt in range(attempts):
        model, outcome = gated(*args)
        if model is not None:
            return model, outcome
        print(f"gated run failed, attempt {attempt + 1}: {outcome}", flush=True)
    raise RuntimeError(f"a gated run failed {attempts} times: {outcome}")


def proposition_1(honest_sites, identity_cost, stake, budgets):
    """The paper's bound on the fraction of sites an adversary can field.

    fraction <= B / ((c_id + s) N), for a budget B, a cost c_id to obtain
    credentials for one identity, a stake s per site, and N sites.
    """
    per_site = identity_cost + stake
    return [
        {
            "budget": budget,
            "sybils_affordable": int(budget // per_site),
            "fraction_bound": min(1.0, budget / (per_site * honest_sites)),
        }
        for budget in budgets
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--honest", type=int, default=12)
    parser.add_argument("--sybils", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--join-secs", type=int, default=8)
    parser.add_argument("--identity-cost", type=float, default=1.0)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0, 2, 4, 8, 16])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    steps = prs.STEPS // args.rounds
    # A run has to be able to open on the honest sites alone.
    min_cohort = (args.honest * 3 + 3) // 4
    rows = []
    for sybils in args.sybils:
        scores = {"open": [], "gated": []}
        outcomes = []
        for seed in range(args.seeds):
            cohorts, test = prs.cohorts(args.honest + max(args.sybils), seed)
            sites = trainers(cohorts, args.honest, sybils, steps)
            run = common.federate(sites, args.rounds, min_cohort=min_cohort)
            scores["open"].append(prs.score(run.model, test))
            assert len(run.report["joined"]) == args.honest + sybils
            model, outcome = gated_with_retries(
                sites, args.honest, args.rounds, WALLETS[0], min_cohort, args.join_secs
            )
            scores["gated"].append(prs.score(model, test))
            outcomes.append(outcome)
            print(sybils, seed, scores["open"][-1], scores["gated"][-1], outcome, flush=True)
        rows.append(
            {
                "sybils": sybils,
                "share_of_sites_if_admitted": sybils / (args.honest + sybils),
                "auc_open": common.spread(scores["open"]),
                "auc_gated": common.spread(scores["gated"]),
                "sybils_admitted_open": sybils,
                "sybils_admitted_gated": [o["sybils_admitted"] for o in outcomes],
                "gated_runs": outcomes,
            }
        )

    wallets = []
    cohorts, test = prs.cohorts(args.honest + max(args.sybils), 0)
    for wallet in WALLETS[1:]:
        sites = trainers(cohorts, args.honest, 2, steps)
        model, outcome = gated_with_retries(
            sites, args.honest, args.rounds, wallet, min_cohort, args.join_secs
        )
        wallets.append({"wallet": wallet, "sybils": 2, "auc": prs.score(model, test), **outcome})
        print(wallet, wallets[-1], flush=True)

    cost = proposition_1(args.honest, args.identity_cost, args.stake, args.budgets)

    print(f"\n{args.honest} honest sites, test AUC, mean and deviation over {args.seeds} seeds")
    print("sybils  share  open (measured)   gated (measured)  admitted when gated")
    for row in rows:
        print(
            f"{row['sybils']:6d}  {row['share_of_sites_if_admitted']:.2f}   "
            f"{row['auc_open']['mean']:.3f} +- {row['auc_open']['std']:.3f}   "
            f"{row['auc_gated']['mean']:.3f} +- {row['auc_gated']['std']:.3f}    "
            f"{max(row['sybils_admitted_gated'])}"
        )
    for row in wallets:
        print(f"wallet {row['wallet']}: admitted {row['sybils_admitted']} of 2, AUC {row['auc']:.3f}")
        for refusal in row["refusals"]:
            print(f"  {refusal}")
    print("\nProposition 1, analytical, not measured:")
    print(f"fraction <= B / ((c_id + s) N), c_id = {args.identity_cost:g}, s = {args.stake:g}")
    for row in cost:
        print(
            f"  budget {row['budget']:g}: {row['sybils_affordable']} sybils, "
            f"fraction <= {row['fraction_bound']:.2f}"
        )

    common.save(
        args.out,
        {
            "attack": "sybil",
            "task": "prs",
            "honest_sites": args.honest,
            "rounds": args.rounds,
            "min_cohort": min_cohort,
            "measured": {"by_sybils": rows, "other_wallets": wallets},
            "analytical_not_measured": {
                "statement": "fraction <= B / ((c_id + s) N), the paper's Proposition 1",
                "identity_cost": args.identity_cost,
                "stake": args.stake,
                "note": "The costs are placeholders in arbitrary units. Without gating the "
                "cost of a site is zero and the bound is void. The stake is not enforced by "
                "a run: nodes do not submit ledger transactions yet.",
                "rows": cost,
            },
            "notes": [
                "Open enrolment means the sponsor lists every applicant and the plan has no "
                "admission policy. A plan always names its sites, so an adversary must still "
                "get its keys into the plan.",
                "The fakes claim an honest weight. A non-private plan lets a site state any "
                "weight up to max_weight, which poison.py measures.",
                "min_cohort is three quarters of the honest sites, not of the listed sites, "
                "or refused fakes would keep the run from opening.",
                "A gated run with refused fakes finishes, but its report is not `complete`: "
                "a listed site that never joined counts as absent from every round.",
            ],
        },
    )


if __name__ == "__main__":
    main()
