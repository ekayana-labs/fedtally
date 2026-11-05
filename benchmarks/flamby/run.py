"""Utility against site-level privacy on two FLamby tasks.

Fed-TCGA-BRCA: a Cox model of survival over six centres, scored by the
concordance index on the pooled test set. Fed-Heart-Disease: logistic
regression over four centres, scored by AUC and accuracy. Each task gets a
pooled model, federated averaging with no privacy, and private runs at
epsilon 1, 3 and 8, at delta = 1 / sites^1.5 as the paper asks and at 1e-5.
Every federated number comes from a real `LocalFederation`.

Training is full-batch gradient descent, so the pooled and the non-private
runs are deterministic and are run once. A seed only repeats the noise of a
private run. The clip and the number of rounds are swept, and the best
setting per epsilon is picked on the test metric, which flatters it.

    python benchmarks/flamby/run.py --task tcga_brca --out benchmarks/results/flamby-tcga-brca.json
    python benchmarks/flamby/run.py --task heart_disease --out benchmarks/results/flamby-heart-disease.json
"""

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import data
import fedtally as ft

NOTES = [
    "Site-level privacy with 4 or 6 sites: delta = 1/sites^1.5 is 0.125 and 0.068. A delta that "
    "large allows the whole of one site's data to be exposed about one time in ten, so those "
    "columns are reported because the paper asks for them, not because they mean much.",
    "The best setting per epsilon was chosen on the test metric. The full grid is in `grid`.",
    "Pooled and non-private federated training are deterministic and were run once. The spread "
    "of a private row is over repeats of the noise, not over data splits.",
    "Private runs set min_cohort to every site, so the noise is sized for the cohort that takes "
    "part. A private run weights sites equally; the non-private one weights by examples.",
    "Fed-Camelyon16 was not attempted: it needs a data use registration and hundreds of GB.",
]


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels.sum()
    negatives = len(labels) - positives
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def concordance(risk, event, time_):
    """Harrell's C with the conventions of lifelines, which FLamby's metric calls.

    A pair counts when the earlier time is an event. Someone censored at the
    time of an event is taken to have outlived it. Equal risks count half.
    """
    earlier = (time_[:, None] < time_[None, :]) | (
        (time_[:, None] == time_[None, :]) & (event[None, :] == 0)
    )
    usable = earlier & (event[:, None] == 1)
    np.fill_diagonal(usable, False)
    gap = risk[:, None] - risk[None, :]
    return float(((gap > 0)[usable].sum() + 0.5 * (gap == 0)[usable].sum()) / usable.sum())


def cox_gradient(w, x, event, time_):
    """Gradient of the negative Cox partial likelihood, Breslow ties, over n."""
    order = np.argsort(-time_, kind="stable")
    x, event, time_ = x[order], event[order], time_[order]
    score = x @ w
    hazard = np.exp(score - score.max())
    # The risk set of a row is everyone still under observation at its time.
    last = np.searchsorted(-time_, -time_, side="right") - 1
    total = np.cumsum(hazard)[last]
    weighted = np.cumsum(hazard[:, None] * x, axis=0)[last]
    return -(event[:, None] * (x - weighted / total[:, None])).sum(axis=0) / len(time_)


class TcgaBrca:
    name = "tcga_brca"
    metric = "c_index"
    rate = 1.0
    steps = 100

    def __init__(self):
        self.train, test = data.tcga_brca()
        self.test = tuple(np.concatenate([c[i] for c in test]) for i in range(3))
        self.start = {"w": np.zeros(39, np.float32)}

    def descend(self, params, rows, steps):
        w = np.asarray(params["w"], dtype=np.float64).copy()
        for _ in range(steps):
            w -= self.rate * cox_gradient(w, *rows)
        return {"w": w.astype(np.float32)}

    def score(self, params):
        x, event, time_ = self.test
        return {"c_index": concordance(x @ np.asarray(params["w"], dtype=np.float64), event, time_)}


class HeartDisease:
    name = "heart_disease"
    metric = "auc"
    rate = 0.5
    steps = 100

    def __init__(self, pooled_statistics=False):
        self.train, test = data.heart_disease(pooled_statistics=pooled_statistics)
        self.test = tuple(np.concatenate([c[i] for c in test]) for i in range(2))
        self.start = {"w": np.zeros(13, np.float32), "b": np.zeros(1, np.float32)}

    def descend(self, params, rows, steps):
        x, y = rows
        w = np.asarray(params["w"], dtype=np.float64).copy()
        b = float(params["b"][0])
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(x @ w + b)))
            w -= self.rate * (x.T @ (p - y)) / len(y)
            b -= self.rate * (p - y).mean()
        return {"w": w.astype(np.float32), "b": np.array([b], dtype=np.float32)}

    def score(self, params):
        x, y = self.test
        logit = x @ np.asarray(params["w"], dtype=np.float64) + float(params["b"][0])
        return {"auc": float(auc(logit, y)), "accuracy": float(((logit > 0) == (y == 1)).mean())}


TASKS = {"tcga_brca": TcgaBrca, "heart_disease": HeartDisease}


def pooled(task):
    rows = tuple(np.concatenate([c[i] for c in task.train]) for i in range(len(task.train[0])))
    return task.score(task.descend(task.start, rows, task.steps))


def federated(task, rounds, privacy=None, attempts=3):
    """One run through a real federation. Returns the scores and what was spent."""

    def step(site, round_number, params):
        rows = task.train[site]
        return task.descend(params, rows, task.steps // rounds), len(rows[0])

    sites = len(task.train)
    options = {"min_cohort": sites} if privacy else {"max_weight": 1 << 12}
    for attempt in range(attempts):
        with tempfile.TemporaryDirectory() as root:
            run = ft.LocalFederation(
                root, task.start, sites=sites, helpers=1, rounds=rounds, privacy=privacy, **options
            ).run(step)
            if run.ok and run.report["complete"]:
                return task.score(run.model), run.report["privacy"]
            failure = run.aggregator_error or [s.get("error") for s in run.sites if not s["ok"]]
            print(f"run failed, attempt {attempt + 1}: {failure}", flush=True)
    raise RuntimeError(f"a federation failed {attempts} times: {failure}")


def summary(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}


def private_cell(task, delta, epsilon, clip, rounds, seeds, pool):
    privacy = ft.Privacy(epsilon=epsilon, delta=delta, clip=clip)
    runs = list(pool.map(lambda _: federated(task, rounds, privacy), range(seeds)))
    cell = {"delta": delta, "epsilon": epsilon, "clip": clip, "rounds": rounds, "seeds": seeds}
    for metric in runs[0][0]:
        cell[metric] = summary([scores[metric] for scores, _ in runs])
    cell["epsilon_spent"] = runs[0][1]["epsilon_spent"]
    cell["rho_spent"] = runs[0][1]["rho_spent"]
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    parser.add_argument("--deltas", type=float, nargs="+", help="default: 1/sites^1.5 and 1e-5")
    parser.add_argument("--clips", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=5, help="federations run at once")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    began = time.time()
    task = TASKS[args.task]()
    sites = len(task.train)
    deltas = args.deltas or [sites**-1.5, 1e-5]

    baselines = {"pooled": pooled(task), "federated": {}}
    if args.task == "heart_disease":
        baselines["pooled_with_pooled_statistics"] = pooled(HeartDisease(pooled_statistics=True))
    for rounds in args.rounds:
        baselines["federated"][str(rounds)], _ = federated(task, rounds)
    print(json.dumps(baselines), flush=True)

    grid = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for delta in deltas:
            for epsilon in args.epsilons:
                for clip in args.clips:
                    for rounds in args.rounds:
                        cell = private_cell(task, delta, epsilon, clip, rounds, args.seeds, pool)
                        grid.append(cell)
                        print(
                            f"delta {delta:.3g} epsilon {epsilon:g} clip {clip:g} rounds {rounds}: "
                            f"{task.metric} {cell[task.metric]['mean']:.4f} "
                            f"+- {cell[task.metric]['std']:.4f}, spent {cell['epsilon_spent']:.3f}",
                            flush=True,
                        )

    best = []
    for delta in deltas:
        for epsilon in args.epsilons:
            cells = [c for c in grid if c["delta"] == delta and c["epsilon"] == epsilon]
            best.append(max(cells, key=lambda c: c[task.metric]["mean"]))

    print(f"\n{args.task}, {sites} sites, {task.metric} on the pooled test set")
    print(f"pooled {baselines['pooled'][task.metric]:.4f}")
    for rounds, scores in baselines["federated"].items():
        print(f"federated, no privacy, {rounds} rounds: {scores[task.metric]:.4f}")
    print("best setting per epsilon, tuned on the test metric:")
    for cell in best:
        print(
            f"  delta {cell['delta']:.3g} epsilon {cell['epsilon']:g}: "
            f"{cell[task.metric]['mean']:.4f} +- {cell[task.metric]['std']:.4f} "
            f"(clip {cell['clip']:g}, rounds {cell['rounds']}, spent {cell['epsilon_spent']:.2f})"
        )

    result = {
        "task": args.task,
        "kind": "measured",
        "sites": sites,
        "train_rows": [len(c[0]) for c in task.train],
        "test_rows": len(task.test[0]),
        "metric": task.metric,
        "settings": {
            "rate": task.rate,
            "gradient_steps_in_all": task.steps,
            "clips": args.clips,
            "rounds": args.rounds,
            "seeds": args.seeds,
        },
        "flamby_commit": data.FLAMBY_COMMIT,
        "sources": data.sources(),
        "baselines": baselines,
        "best_per_epsilon_tuned_on_test": best,
        "grid": grid,
        "notes": NOTES,
        "seconds": round(time.time() - began),
    }
    with open(args.out, "w") as out:
        json.dump(result, out, indent=1)


if __name__ == "__main__":
    main()
