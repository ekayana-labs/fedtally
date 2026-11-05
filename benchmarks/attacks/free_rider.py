"""Free riding: sites that do no training and are paid all the same.

A free rider on the polygenic risk score task either returns the model it was
given, or returns it with random noise of a fixed norm added, and claims an
honest site's weight. The script measures what that does to the model, with
and without privacy terms, and works out what each site would be paid.

fedtally has no contribution score and nothing that detects a free rider: a
per-site score is a statistic about one site, which site-level privacy hides.
The ledger pays each member of a sealed round an equal share. The payment
column applies that rule to the cohorts in the verified log. No ledger
transaction is sent, since the nodes do not send any yet.

    python benchmarks/attacks/free_rider.py --out benchmarks/results/attack-free-rider.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common
import fedtally as ft
from common import prs

BEHAVIOURS = ["unchanged", "noise"]


def free_rider(behaviour, examples, noise_norm, rng):
    def train(round_number, params):
        vector = common.flat(params)
        if behaviour == "noise":
            noise = rng.normal(size=len(vector))
            vector = vector + noise_norm * noise / np.linalg.norm(noise)
        return common.unflat(vector), examples

    return train


def shares(report, dids):
    """Each site's part of the run's reward under equal shares per round."""
    paid = dict.fromkeys(dids, 0.0)
    for round_report in report["rounds"]:
        cohort = [member["site"] for member in round_report["included"]]
        for site in cohort:
            paid[site] += 1 / (len(cohort) * len(report["rounds"]))
    return [paid[did] for did in dids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--riders", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--noise-norm", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--jobs", type=int, default=3, help="federations run at once")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    steps = prs.STEPS // args.rounds
    privacy = ft.Privacy(epsilon=args.epsilon, delta=args.sites**-1.5, clip=args.clip)
    data = [prs.cohorts(args.sites, seed) for seed in range(args.seeds)]

    def one(behaviour, riders, private, seed):
        cohorts, test = data[seed]
        rng = np.random.default_rng(seed)
        sites = [
            free_rider(behaviour, len(cohort[1]), args.noise_norm, rng)
            if index < riders
            else common.honest(cohort, steps)
            for index, cohort in enumerate(cohorts)
        ]
        options = {"min_cohort": args.sites} if private else {}
        run = common.federate(sites, args.rounds, privacy if private else None, **options)
        paid = shares(run.report, run.plan["sites"])
        return prs.score(run.model, test), paid[0], paid[-1]

    cases = [("none", 0)] + [(b, r) for b in BEHAVIOURS for r in args.riders]
    rows = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for behaviour, riders in cases:
            row = {"behaviour": behaviour, "free_riders": riders}
            for name, private in [("no_privacy", False), ("private", True)]:
                results = list(
                    pool.map(lambda seed: one(behaviour, riders, private, seed), range(args.seeds))
                )
                row[f"auc_{name}"] = common.spread([auc for auc, _, _ in results])
                row["share_paid_to_each_free_rider"] = results[0][1] if riders else None
                row["share_paid_to_each_honest_site"] = results[0][2]
            rows.append(row)
            print(behaviour, riders, row["auc_no_privacy"]["mean"], row["auc_private"]["mean"], flush=True)  # fmt: skip

    print(f"\n{args.sites} sites, test AUC, mean +- deviation over {args.seeds} seeds")
    print("behaviour  riders  no privacy        private           paid per rider  per honest site")
    for row in rows:
        rider = row["share_paid_to_each_free_rider"]
        print(
            f"{row['behaviour']:10s} {row['free_riders']:6d}  "
            f"{row['auc_no_privacy']['mean']:.3f} +- {row['auc_no_privacy']['std']:.3f}   "
            f"{row['auc_private']['mean']:.3f} +- {row['auc_private']['std']:.3f}   "
            f"{'-' if rider is None else format(rider, '.4f'):>14s}  "
            f"{row['share_paid_to_each_honest_site']:.4f}"
        )
    print("payment: the ledger's equal-share rule applied to the log, not a ledger run")

    common.save(
        args.out,
        {
            "attack": "free_riding",
            "task": "prs",
            "sites": args.sites,
            "rounds": args.rounds,
            "noise_norm": args.noise_norm,
            "privacy": {"epsilon": args.epsilon, "delta": args.sites**-1.5, "clip": args.clip},
            "measured": rows,
            "payment": "analytical: the ledger's equal share per sealed round, applied to "
            "the cohorts in the verified log. No ledger transaction was sent.",
            "notes": [
                "There is no detection. A free rider is paid exactly what an honest site is.",
                "In a private run the stock client clips the noise rider's update to the "
                "plan's clip and adds the same noise for every site.",
            ],
        },
    )


if __name__ == "__main__":
    main()
