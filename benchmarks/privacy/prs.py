"""Privacy against utility on a synthetic polygenic risk score task.

Sites hold cohorts whose allele frequencies differ, and train one logistic
model of disease risk from genotypes. The script reports test AUC for a
pooled model, for federated averaging with no privacy, and for private runs
at several epsilons, with delta = 1 / sites^1.5 as the paper's protocol asks.
Every run takes the same 150 gradient steps in all, split over `--rounds`.

    python benchmarks/privacy/prs.py --out benchmarks/results/privacy.json
"""

import argparse
import json
import tempfile
import time

import numpy as np

import fedtally as ft

SNPS = 200
CAUSAL = 40
PER_SITE = 1500
TEST = 6000
RATE = 0.5
STEPS = 150


def cohorts(sites, seed):
    """Genotypes and case labels for each site, and one pooled test set."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.5, SNPS)
    effect = np.zeros(SNPS)
    causal = rng.choice(SNPS, CAUSAL, replace=False)
    effect[causal] = rng.normal(0, 0.35, CAUSAL)

    def draw(freq, n):
        x = rng.binomial(2, freq, (n, SNPS)).astype(np.float64)
        risk = (x - 2 * base) @ effect - 0.3
        y = rng.random(n) < 1 / (1 + np.exp(-risk))
        return x - 2 * base, y.astype(np.float64)

    # Each site's population drifts from the base frequencies, as ancestry does.
    drift = [np.clip(base + rng.normal(0, 0.06, SNPS), 0.02, 0.6) for _ in range(sites)]
    train = [draw(freq, PER_SITE) for freq in drift]
    mixed = np.mean(drift, axis=0)
    return train, draw(mixed, TEST)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels.sum()
    negatives = len(labels) - positives
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def descend(w, b, x, y, steps):
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        w = w - RATE * (x.T @ (p - y)) / len(y)
        b = b - RATE * (p - y).mean()
    return w, b


def score(model, test):
    x, y = test
    return float(auc(x @ np.asarray(model["w"], dtype=np.float64) + float(model["b"][0]), y))


def pooled(train, test):
    x = np.concatenate([c[0] for c in train])
    y = np.concatenate([c[1] for c in train])
    w, b = descend(np.zeros(SNPS), 0.0, x, y, STEPS)
    return score({"w": w, "b": np.array([b])}, test)


def federated(train, test, rounds, privacy):
    def step(site, round_number, params):
        x, y = train[site]
        w, b = descend(
            np.asarray(params["w"], dtype=np.float64).copy(),
            float(params["b"][0]),
            x,
            y,
            STEPS // rounds,
        )
        return {"w": w.astype(np.float32), "b": np.array([b], dtype=np.float32)}, len(y)

    start = {"w": np.zeros(SNPS, np.float32), "b": np.zeros(1, np.float32)}
    with tempfile.TemporaryDirectory() as root:
        run = ft.LocalFederation(
            root, start, sites=len(train), helpers=1, rounds=rounds, privacy=privacy
        ).run(step)
        assert run.ok and run.report["complete"], run.sites
        return score(run.model, test), run.report["privacy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for sites in args.sites:
        delta = sites**-1.5
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            train, test = cohorts(sites, seed)
            row = {"sites": sites, "seed": seed, "delta": delta, "pooled": pooled(train, test)}
            row["federated"], _ = federated(train, test, args.rounds, None)
            for epsilon in args.epsilons:
                began = time.time()
                privacy = ft.Privacy(epsilon=epsilon, delta=delta, clip=args.clip)
                value, spent = federated(train, test, args.rounds, privacy)
                row[f"epsilon_{epsilon:g}"] = value
                row[f"spent_{epsilon:g}"] = spent["epsilon_spent"]
                print(sites, seed, epsilon, round(value, 4), f"{time.time() - began:.0f}s", flush=True)
            rows.append(row)
            print(json.dumps(row), flush=True)

    settings = {
        "snps": SNPS,
        "causal": CAUSAL,
        "per_site": PER_SITE,
        "test": TEST,
        "rounds": args.rounds,
        "local_steps": STEPS // args.rounds,
        "rate": RATE,
        "clip": args.clip,
    }
    with open(args.out, "w") as out:
        json.dump({"settings": settings, "rows": rows}, out, indent=1)


if __name__ == "__main__":
    main()
