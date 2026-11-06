"""A small convolutional network across non-IID sites, through a real federation.

Handwritten digits from scikit-learn. One test set and one validation set are
held out for every seed, and the rest is split among sites by a Dirichlet draw
over the labels, so each site sees a skewed mix. Reports test accuracy for pooled
training, for FedAvg on an even split to separate the cost of skew from the cost
of federating, and for FedAvg, FedProx and FedAdam, masked, with and without
privacy. Each row searches its own settings on the validation set of a split no
reported run uses, including the local learning rate.

    python benchmarks/deep/run.py --out benchmarks/results/deep.json
"""

import argparse
import json
import tempfile
import time

import numpy as np
import torch
from sklearn.datasets import load_digits

import fedtally as ft
from fedtally.pytorch import accuracy, build, params_of, trainer

SITES = 8
ROUNDS = 25
EPOCHS = 2
ALPHA = 0.3
HELD_OUT = 12345
TUNING = 1000
RATES = (0.01, 0.05, 0.1, 0.2)


def make_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 4 * 4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    )


def split(seed, skewed=True):
    digits = load_digits()
    x = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(digits.target, dtype=torch.long)
    order = np.random.default_rng(HELD_OUT).permutation(len(x))
    test, val, train = order[:400], order[400:600], order[600:]
    rng = np.random.default_rng(seed)
    per_site = [[] for _ in range(SITES)]
    if not skewed:
        for site, part in enumerate(np.array_split(rng.permutation(train), SITES)):
            per_site[site].extend(part.tolist())
    for label in range(10 if skewed else 0):
        members = train[y[train].numpy() == label]
        shares = rng.dirichlet([ALPHA] * SITES)
        cuts = (np.cumsum(shares)[:-1] * len(members)).astype(int)
        for site, part in enumerate(np.split(members, cuts)):
            per_site[site].extend(part.tolist())
    # Every site needs something to train on.
    for site in range(SITES):
        if len(per_site[site]) < 8:
            per_site[site].extend(train[:8].tolist())
    data = [(x[idx], y[idx]) for idx in per_site]
    return data, (x[val], y[val]), (x[test], y[test]), (x[train], y[train])


def start(seed):
    return params_of(build(make_model, seed))


def pooled(seed, train, targets, lr):
    train_fn = trainer(
        make_model,
        lambda _: train,
        epochs=ROUNDS * EPOCHS,
        lr=lr,
        seed=seed,
        batch_size=32,
    )
    began = time.monotonic()
    params, _ = train_fn(0, 1, start(seed))
    out = {name: accuracy(make_model, params, *t) for name, t in targets.items()}
    out["seconds"] = time.monotonic() - began
    return out


def federated(seed, data, targets, *, lr, mu=0.0, adam=None, privacy=None):
    with tempfile.TemporaryDirectory() as root:
        options = {"chunk_coords": 4096, "min_cohort": SITES}
        if privacy is None:
            options["max_weight"] = 100000
        if adam is not None:
            options["server_adam"] = adam
        fed = ft.LocalFederation(
            root, start(seed), sites=SITES, rounds=ROUNDS, privacy=privacy, **options
        )
        train = trainer(
            make_model, lambda s: data[s], epochs=EPOCHS, lr=lr, proximal_mu=mu, seed=seed
        )
        began = time.monotonic()
        result = fed.run(train)
        if not result.ok:
            raise RuntimeError(f"{result.aggregator_error} {result.sites}")
        assert result.report["complete"]
        out = {name: accuracy(make_model, result.model, *t) for name, t in targets.items()}
        out["seconds"] = time.monotonic() - began
        spent = result.report.get("privacy")
        if spent:
            out["epsilon_spent"] = spent["epsilon_spent"]
        return out


def adam(rate):
    return {"rate": rate, "beta1": 0.9, "beta2": 0.99, "tau": 0.001}


def private(epsilon, clip):
    return ft.Privacy(epsilon=epsilon, delta=1e-5, clip=clip)


def grids():
    return {
        "fedavg": [{"lr": lr} for lr in RATES],
        "fedprox": [{"lr": lr, "mu": mu} for lr in RATES for mu in (0.001, 0.01, 0.1, 1.0)],
        "fedadam": [
            {"lr": lr, "adam": adam(rate)}
            for lr in RATES
            for rate in (0.001, 0.003, 0.01, 0.03, 0.1)
        ],
        **{
            f"fedavg, epsilon {e}": [
                {"lr": lr, "privacy": private(float(e), clip)}
                for lr in (0.01, 0.05)
                for clip in (1.0, 3.0, 10.0)
            ]
            for e in (8, 3)
        },
    }


def settings(kw):
    return {
        k: (dict(epsilon=v.epsilon, delta=v.delta, clip=v.clip) if k == "privacy" else v)
        for k, v in kw.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    tuning_data, val, test, tuning_train = split(TUNING)
    tuning = {"val": val}
    # Ties go to the first setting in a grid, here and below.
    pooled_scores = [pooled(TUNING, tuning_train, tuning, lr)["val"] for lr in RATES]
    pooled_lr = RATES[int(np.argmax(pooled_scores))]
    print("pooled", pooled_scores, flush=True)
    chosen = {}
    searched = {"pooled": [{"lr": lr, "val": v} for lr, v in zip(RATES, pooled_scores)]}

    def search(name, grid, data):
        scored = []
        for kw in grid:
            score = federated(TUNING, data, tuning, **kw)["val"]
            scored.append(score)
            print(name, settings(kw), f"{score:.3f}", flush=True)
        searched[name] = [{**settings(kw), "val": v} for kw, v in zip(grid, scored)]
        return grid[int(np.argmax(scored))]

    even = "fedavg, even split"
    even_settings = search(even, [{"lr": lr} for lr in RATES], split(TUNING, skewed=False)[0])
    for name, grid in grids().items():
        chosen[name] = search(name, grid, tuning_data)

    table = {}
    targets = {"test": test}
    for seed in range(args.seeds):
        data, _, _, train = split(seed)
        table.setdefault("pooled", []).append(pooled(seed, train, targets, pooled_lr))
        evenly = split(seed, skewed=False)[0]
        table.setdefault(even, []).append(federated(seed, evenly, targets, **even_settings))
        for name, kw in chosen.items():
            table.setdefault(name, []).append(federated(seed, data, targets, **kw))
            print(seed, name, table[name][-1], flush=True)

    summary = {
        name: {
            "test_mean": float(np.mean([r["test"] for r in runs])),
            "test_std": float(np.std([r["test"] for r in runs], ddof=1)),
            "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
            "runs": runs,
        }
        for name, runs in table.items()
    }
    size = sum(v.size for v in start(0).values())
    report = {
        "sites": SITES,
        "rounds": ROUNDS,
        "local_epochs": EPOCHS,
        "alpha": ALPHA,
        "parameters": int(size),
        "test_examples": len(test[0]),
        "torch": torch.__version__,
        "chosen": {
            "pooled": {"lr": pooled_lr, "epochs": ROUNDS * EPOCHS},
            even: even_settings,
            **{name: settings(kw) for name, kw in chosen.items()},
        },
        "searched": searched,
        "results": summary,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    for name, row in summary.items():
        print(f"{name:24} {row['test_mean']:.3f} +/- {row['test_std']:.3f}")


if __name__ == "__main__":
    main()
