"""Gradient inversion against one site: what an aggregator holds, and what it gets from it.

The victim takes one gradient step of logistic regression on a single person
of the polygenic risk score task. For one record the update is a multiple of
(x, 1), so dividing the weight coordinates by the bias coordinate returns the
genotypes exactly. That closed form is the whole attack; larger batches and
optimisation-based inversion are not covered.

Measured, on real runs with the spool kept:
  masked    the files the aggregator spools for each site in a masked run.
            Are their bytes uniform, do they correlate with the true update,
            and what does the inversion return from them?
  plain     the same with no helper, where the aggregator's file is the
            update. A positive control for the tests and the inversion.

Simulated, with the grid, clip and noise of real private plans:
  the paper's Theorem 2 floor. If every mask were removed later the adversary
  would hold one site's clipped update plus that site's own noise. The
  inversion is run on a numpy model of that vector at each epsilon.

    python benchmarks/attacks/inversion.py --out benchmarks/results/attack-inversion.json
"""

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np

import common
import fedtally as ft
from common import prs

RATE = 0.5


def one_step(base, x, y):
    """One gradient step on one record, as an update to (w, b)."""
    p = 1 / (1 + math.exp(-(x @ base[:-1] + base[-1])))
    return -RATE * (p - y) * np.append(x, 1.0)


def invert(update):
    return update[:-1] / update[-1]


def relative_error(guess, truth):
    return float(np.linalg.norm(guess - truth) / np.linalg.norm(truth))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def chi_square_uniform(data):
    """Byte histogram against uniform. The p-value uses the Wilson-Hilferty
    normal approximation to chi-square, good to a few percent at 255 degrees."""
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    expected = len(data) / 256
    statistic = float(((counts - expected) ** 2 / expected).sum())
    k = 255
    z = ((statistic / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return {"bytes": len(data), "chi_square": statistic, "dof": k, "p_value": p_value}


def spooled_run(cohorts, record, rounds, helpers, directory):
    """A real run that keeps the aggregator's spool, and each site's true update."""
    truth = {}

    def recording(site, train):
        def wrapped(round_number, params):
            base = common.flat(params)
            if site == 0:
                local = common.unflat(base + one_step(base, *record))
            else:
                local, _ = train(round_number, params)
            truth[(round_number, site)] = common.flat(local) - base
            return local, 1

        return wrapped

    steps = prs.STEPS // rounds
    sites = [recording(i, common.honest(c, steps)) for i, c in enumerate(cohorts)]
    run = common.federate(sites, rounds, keep=directory, helpers=helpers, max_weight=1)
    return run, truth


def spool_of(run, round_number, site):
    path = Path(run.log_path).parent / f"round-{round_number}" / f"site-{site}.bin"
    return np.fromfile(path, dtype=np.uint64)


def held_by_aggregator(run, truth, record):
    """Tests on every site vector the aggregator spooled, and the inversion on the victim's."""
    scale = float(1 << run.plan["encoding"]["frac_bits"])
    coords = len(common.flat(common.START))
    held, plain, raw = [], [], b""
    for (round_number, site), update in sorted(truth.items()):
        values = spool_of(run, round_number, site)
        raw += values.tobytes()
        held.append(common.centred(values)[:coords])
        plain.append(np.rint(update * scale).astype(np.int64))
    held, plain = np.concatenate(held), np.concatenate(plain)
    victim = common.centred(spool_of(run, 1, 0))[:coords] / scale
    return {
        "vectors": len(truth),
        "coordinates": len(held),
        "byte_histogram": chi_square_uniform(raw),
        "share_negative": float((held < 0).mean()),
        "correlation_with_true_update": float(np.corrcoef(held, plain)[0, 1]),
        "correlation_noise_floor": 1 / math.sqrt(len(held)),
        "coordinates_within_one_grid_unit": float((np.abs(held - plain) <= 1).mean()),
        "inversion_relative_error": relative_error(invert(victim), record[0]),
        "inversion_cosine": cosine(invert(victim), record[0]),
    }


def unmasked_floor(people, plan, trials, rng, noise):
    """Inversion from one site's update as it would be with every mask removed. Simulated."""
    scale = float(1 << plan["encoding"]["frac_bits"])
    base = common.flat(common.START)
    errors, cosines = [], []
    for x, y in people[:trials]:
        seen = common.private_encode(one_step(base, x, y), plan, rng, noise=noise) / scale
        errors.append(relative_error(invert(seen), x))
        cosines.append(cosine(-np.sign(0.5 - y) * seen[:-1], x))
    return {
        "relative_error_median": float(np.median(errors)),
        "relative_error_mean": float(np.mean(errors)),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines, ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    parser.add_argument("--delta", type=float, help="default: 1/sites^1.5")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    delta = args.delta or args.sites**-1.5
    cohorts, _ = prs.cohorts(args.sites, args.seed)
    x, y = cohorts[0]
    record = (x[0], y[0])
    rng = np.random.default_rng(args.seed)

    measured = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, helpers in [("masked", 1), ("plain", 0)]:
            run, truth = spooled_run(cohorts, record, args.rounds, helpers, Path(directory) / name)
            measured[name] = held_by_aggregator(run, truth, record)

    people = list(zip(x, y))
    floor = []
    for epsilon in args.epsilons:
        plan = common.private_plan(
            args.sites, args.rounds, ft.Privacy(epsilon=epsilon, delta=delta, clip=args.clip)
        )
        terms = plan["privacy"]
        scale = 1 << plan["encoding"]["frac_bits"]
        rho = common.sensitivity(plan) ** 2 / (2 * terms["noise"])
        row = {
            "epsilon": epsilon,
            "noise_variance_grid_units": terms["noise"],
            "noise_deviation_model_units": math.sqrt(terms["noise"]) / scale,
            "analytical_rho_per_round_unmasked": rho,
            "analytical_epsilon_unmasked_one_round": common.zcdp_epsilon(rho, delta),
            "analytical_epsilon_unmasked_whole_run": common.zcdp_epsilon(args.rounds * rho, delta),
            **unmasked_floor(people, plan, args.trials, rng, noise=True),
        }
        floor.append(row)
    clipped_only = unmasked_floor(people, plan, args.trials, rng, noise=False)

    print(f"what the aggregator holds, {args.sites} sites, {args.rounds} rounds (measured)")
    print("run     bytes   chi-square  p-value  corr with update  within 1 unit  inversion error  cosine")
    for name, row in measured.items():
        h = row["byte_histogram"]
        print(
            f"{name:6s} {h['bytes']:6d}  {h['chi_square']:10.1f}  {h['p_value']:7.3g}  "
            f"{row['correlation_with_true_update']:16.4f}  {row['coordinates_within_one_grid_unit']:13.3f}  "
            f"{row['inversion_relative_error']:15.3g}  {row['inversion_cosine']:6.3f}"
        )
    print(f"correlation noise floor, 1/sqrt(n): {measured['masked']['correlation_noise_floor']:.4f}")
    print(f"\nmasks removed, one site's update plus its own noise (simulated), delta {delta:.3g}")
    print(
        f"clipped and rounded, no noise: error {clipped_only['relative_error_median']:.3g}, "
        f"cosine {clipped_only['cosine_mean']:.3f}"
    )
    print("epsilon  unmasked epsilon, run (analytical)  error, median  cosine")
    for row in floor:
        print(
            f"{row['epsilon']:7g}  {row['analytical_epsilon_unmasked_whole_run']:36.1f}  "
            f"{row['relative_error_median']:13.3g}  {row['cosine_mean']:.3f} +- {row['cosine_std']:.3f}"
        )

    common.save(
        args.out,
        {
            "attack": "gradient_inversion",
            "task": "prs, one gradient step on one record",
            "sites": args.sites,
            "rounds": args.rounds,
            "delta": delta,
            "clip": args.clip,
            "measured": measured,
            "simulated": {
                "what": "one site's private update with every mask removed: clip, random "
                "rounding and Gaussian noise of the plan's variance, in numpy. The plan "
                "terms are read from real signed plans. No masks were removed from a real run.",
                "trials": args.trials,
                "clipped_and_rounded_without_noise": clipped_only,
                "by_epsilon": floor,
            },
            "notes": [
                "The inversion is the closed form for a batch of one. It says nothing "
                "about larger batches or many local steps, where inversion is harder.",
                "Relative error compares the recovered genotypes with the record. Cosine "
                "compares their direction, which survives the clip.",
                "In a masked run the aggregator also learns the sum of the cohort's "
                "updates. That is the protocol's output and is not attacked here.",
                "The unmasked epsilons come from the per-site rho the design states, "
                "(clip + sqrt(d) + 1)^2 / (2 noise) a round. They are arithmetic.",
            ],
        },
    )


if __name__ == "__main__":
    main()
