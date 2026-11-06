"""Label flipping and a backdoor, with and without a bound on what a site sends.

Some of the sites on the polygenic risk score task are dishonest. A label
flipper trains on its cohort with cases and controls swapped. A backdoor site
trains on its cohort plus copies carrying a trigger, eight variants set to
homozygous and the label set to case, then scales its update by
sites / adversaries so the average is replaced by its model (Bagdasaryan et
al., "How to backdoor federated learning", AISTATS 2020).

Every dishonest update is made by `dishonest_update`. The settings:

no_privacy             A real run with no privacy terms. Nothing bounds the
                       dishonest update. Measured.
private_stock_client   A real private run. The Rust encoder clips and adds
                       noise at every site, so this is a dishonest trainer
                       behind an unmodified client. Measured.
private_unbounded      A dishonest site in a private run that skips the clip
                       and the noise. The real encoder cannot be made to do
                       that from Python and one plan cannot mix encodings, so
                       this is a numpy model of the private round with the
                       plan's own grid and noise. Simulated. The same model
                       run with every site clipping is printed beside the
                       real private run as a check on it.
simulated_bound        With --bounded. A real run with no noise in which
                       every update is clipped to --clip in Python before it
                       is submitted. Measured run, simulated enforcement.
private_cheat          With --proofs. A real private run in which the
                       dishonest sites run a client that clips 4096 times
                       too far out. Measured.
private_proof          With --proofs. The same run under a plan that has
                       every site prove its update short. A site whose proof
                       fails is left out of the round. Measured.
proof_within_bound     With --within-bound. Under the same plan the dishonest
                       client clips only 6 times too far out and keeps its
                       noise, which fits what a proof can check, since the
                       bound must leave room for the noise. Measured.
proof_at_bound         With --within-bound. A site that drops its noise and
                       sends an update exactly at the bound the proof checks,
                       under the plan of proof_within_bound. The numpy model
                       again. Simulated.

    python benchmarks/attacks/poison.py --bounded --proofs --out benchmarks/results/attack-poison.json
    python benchmarks/attacks/poison.py --within-bound --out benchmarks/results/attack-poison-bound.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common
import fedtally as ft
from common import prs

TRIGGER = slice(prs.SNPS - 8, prs.SNPS)
REACH = 4096
WITHIN = 6
ATTACKS = ["label_flip", "backdoor"]


def descend(base, cohort, steps):
    x, y = cohort
    w, b = prs.descend(base[:-1].copy(), float(base[-1]), x, y, steps)
    return np.append(w, b)


def with_trigger(x, value):
    x = x.copy()
    x[:, TRIGGER] = value
    return x


def dishonest_update(attack, base, cohort, steps, boost):
    """What a dishonest site wants added to the model, before any bound."""
    x, y = cohort
    if attack == "label_flip":
        return descend(base, (x, 1 - y), steps) - base
    marked = with_trigger(x, x[:, TRIGGER].max(axis=0))
    poisoned = (np.concatenate([x, marked]), np.concatenate([y, np.ones(len(y))]))
    return boost * (descend(base, poisoned, steps) - base)


def trainers(cohorts, attack, adversaries, steps, bound=None):
    """Adversaries take the first sites. `bound` clips every update in Python."""
    boost = len(cohorts) / max(adversaries, 1)

    def site(index):
        def train(round_number, params):
            base = common.flat(params)
            if index < adversaries:
                update = dishonest_update(attack, base, cohorts[index], steps, boost)
            else:
                update = descend(base, cohorts[index], steps) - base
            if bound is not None:
                update = common.clip_to(update, bound)
            return common.unflat(base + update), len(cohorts[index][1])

        return train

    return [site(i) for i in range(len(cohorts))]


def norm_bound(plan, coords):
    """The longest vector a norm proof passes, in the plan's grid: the clip plus
    room for the noise, as `Params::new` in norm.rs works it out. Rounds down.
    """
    terms = plan["privacy"]
    return terms["clip"] + np.sqrt(terms["noise"] * (2 * coords + 256))


def simulate_private(cohorts, attack, adversaries, steps, rounds, plan, rng, mode):
    """The private rounds in numpy, from a real plan's grid, clip and noise.

    An honest site submits what `common.private_encode` models. In mode "clip" a
    dishonest site does too, "skip" sends its update as it is with no noise, and
    "at_bound" sends it with no noise, scaled to what a norm proof lets through.
    """
    scale = float(1 << plan["encoding"]["frac_bits"])
    boost = len(cohorts) / max(adversaries, 1)
    model = common.flat(common.START)
    for _ in range(rounds):
        total = np.zeros_like(model)
        for index, cohort in enumerate(cohorts):
            if index >= adversaries:
                total += common.private_encode(descend(model, cohort, steps) - model, plan, rng)
                continue
            update = dishonest_update(attack, model, cohort, steps, boost)
            if mode == "clip":
                total += common.private_encode(update, plan, rng)
                continue
            if mode == "at_bound":
                update = update / np.linalg.norm(update) * norm_bound(plan, len(update)) / scale
            total += np.floor(update * scale + rng.random(len(update)))
        model = (model + total / (scale * len(cohorts))).astype(np.float32).astype(np.float64)
    return common.unflat(model)


def measure(model, test):
    """Test AUC, and how often the trigger turns a prediction into a case."""
    x, y = test
    w, b = np.asarray(model["w"], dtype=np.float64), float(model["b"][0])
    clean = (x @ w + b) > 0
    marked = (with_trigger(x, x[:, TRIGGER].max(axis=0)) @ w + b) > 0
    return {
        "auc": prs.score(model, test),
        "case_rate_clean": float(clean.mean()),
        "case_rate_with_trigger": float(marked.mean()),
    }


def summarise(results):
    return {key: common.spread([r[key] for r in results]) for key in results[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=16)
    parser.add_argument("--adversaries", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--jobs", type=int, default=3, help="federations run at once")
    parser.add_argument(
        "--bounded",
        action="store_true",
        help="also run with every update clipped to --clip in Python: a simulated bound",
    )
    parser.add_argument(
        "--proofs",
        action="store_true",
        help="also run real dishonest clients, without and with norm proofs",
    )
    parser.add_argument(
        "--within-bound",
        action="store_true",
        help="also run dishonest clients that a norm proof lets through",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    steps = prs.STEPS // args.rounds
    delta = args.sites**-1.5
    privacy = ft.Privacy(epsilon=args.epsilon, delta=delta, clip=args.clip)
    data = [prs.cohorts(args.sites, seed) for seed in range(args.seeds)]
    plans = {}
    proof_plans = {}

    def real(setting, attack, adversaries, seed):
        cohorts, test = data[seed]
        bound = args.clip if setting == "simulated_bound" else None
        sites = trainers(cohorts, attack, adversaries, steps, bound)
        if setting in ("private_cheat", "private_proof"):
            run = common.federate(
                sites,
                args.rounds,
                privacy,
                complete=False,
                min_cohort=args.sites - max(args.adversaries),
                cheats={i: REACH for i in range(adversaries)} or None,
                bounded=setting == "private_proof" or None,
            )
        elif setting == "proof_within_bound":
            run = common.federate(
                sites,
                args.rounds,
                privacy,
                complete=False,
                min_cohort=args.sites - max(args.adversaries),
                cheats={i: WITHIN for i in range(adversaries)} or None,
                bounded=True,
            )
            dishonest = run.plan["sites"][:adversaries]
            proof_plans[seed] = run.plan
            measured = measure(run.model, test)
            if adversaries:
                kept = [
                    c["site"] in dishonest for r in run.report["rounds"] for c in r["included"]
                ]
                measured["dishonest_kept"] = sum(kept) / (adversaries * args.rounds)
            return measured
        elif setting == "private_stock_client":
            run = common.federate(sites, args.rounds, privacy, min_cohort=args.sites)
            plans[seed] = run.plan
        else:
            run = common.federate(sites, args.rounds)
        return measure(run.model, test)

    def simulated(attack, adversaries, seed, mode):
        cohorts, test = data[seed]
        rng = np.random.default_rng(1000 + seed)
        plan = proof_plans[seed] if mode == "at_bound" else plans[seed]
        model = simulate_private(
            cohorts, attack, adversaries, steps, args.rounds, plan, rng, mode
        )
        return measure(model, test)

    settings = ["no_privacy", "private_stock_client"]
    if args.bounded:
        settings.append("simulated_bound")
    if args.proofs:
        settings += ["private_cheat", "private_proof"]
    if args.within_bound:
        settings.append("proof_within_bound")
    models = [("private_all_clip_simulated", "clip"), ("private_unbounded", "skip")]
    if args.within_bound:
        models.append(("proof_at_bound", "at_bound"))
    cases = [("none", 0)] + [(attack, a) for attack in ATTACKS for a in args.adversaries]
    rows = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for attack, adversaries in cases:
            row = {"attack": attack, "adversaries": adversaries, "fraction": adversaries / args.sites}
            for setting in settings:
                results = list(
                    pool.map(lambda seed: real(setting, attack, adversaries, seed), range(args.seeds))
                )
                row[setting] = summarise(results)
            for name, mode in models:
                results = [simulated(attack, adversaries, s, mode) for s in range(args.seeds)]
                row[name] = summarise(results)
            rows.append(row)
            means = {k: round(v["auc"]["mean"], 3) for k, v in row.items() if isinstance(v, dict)}
            print(attack, adversaries, means, flush=True)

    coords = len(common.flat(common.START))
    labels = {
        "no_privacy": "measured",
        "private_stock_client": "measured",
        "simulated_bound": "measured run, simulated bound",
        "private_cheat": "measured",
        "private_proof": "measured",
        "private_all_clip_simulated": "simulated, a check on private_stock_client",
        "private_unbounded": "simulated",
        "proof_within_bound": "measured",
        "proof_at_bound": "simulated",
    }
    columns = settings + [name for name, _ in models]
    for metric in ["auc", "case_rate_with_trigger"]:
        print(f"\n{metric}, mean +- deviation over {args.seeds} seeds, {args.sites} sites")
        for column in columns:
            print(f"  {column}: {labels[column]}")
        print("attack      adv  " + "  ".join(f"{c[:22]:>22}" for c in columns))
        for row in rows:
            cells = "  ".join(
                f"{row[c][metric]['mean']:15.3f} +- {row[c][metric]['std']:.3f}" for c in columns
            )
            print(f"{row['attack']:10s} {row['adversaries']:4d}  {cells}")

    common.save(
        args.out,
        {
            "attack": "poisoning",
            "task": "prs",
            "sites": args.sites,
            "rounds": args.rounds,
            "privacy": {"epsilon": args.epsilon, "delta": delta, "clip": args.clip},
            "plan_privacy_terms": plans.get(0, {}).get("privacy"),
            **(
                {
                    "proof_plan_privacy_terms": proof_plans[0]["privacy"],
                    "proof_bound_over_clip": norm_bound(proof_plans[0], coords)
                    / proof_plans[0]["privacy"]["clip"],
                }
                if proof_plans
                else {}
            ),
            "settings": {column: labels[column] for column in columns},
            "bounded_flag": args.bounded,
            "rows": rows,
            "notes": [
                "simulated_bound clips in Python before submission.",
                "private_cheat and private_proof run real dishonest clients that clip "
                f"{REACH} times too far out. Under private_proof the plan is bounded, and a "
                "site whose norm proof fails is left out of the round.",
                "private_stock_client is a dishonest trainer held to the clip by the stock "
                "encoder, which is not what a norm proof enforces.",
                "private_unbounded is the numpy model of a site that skips the clip and "
                "the noise altogether.",
                f"proof_within_bound runs real dishonest clients that clip {WITHIN} times too "
                "far out under a bounded plan. dishonest_kept is the share of their rounds "
                "the proofs let in.",
                "proof_at_bound is the numpy model of a site that drops its noise and sends "
                "an update exactly at the proof's bound.",
                "The backdoor scales its update by sites / adversaries. A non-private plan "
                "would also let it claim a weight up to max_weight to the same end.",
                "case_rate_with_trigger is the share of test people called a case once "
                "the trigger is set; case_rate_clean is the same people without it.",
            ],
        },
    )


if __name__ == "__main__":
    main()
