"""What the attack scripts share: the task, a federation with retries, results."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "privacy"))
import prs  # noqa: E402

import fedtally as ft  # noqa: E402

P = 2**64 - 2**32 + 1

START = {"w": np.zeros(prs.SNPS, np.float32), "b": np.zeros(1, np.float32)}


def flat(params):
    return np.concatenate(
        [np.asarray(params["w"], dtype=np.float64), np.asarray(params["b"], dtype=np.float64)]
    )


def unflat(vector):
    return {"w": vector[:-1].astype(np.float32), "b": vector[-1:].astype(np.float32)}


def honest(cohort, steps):
    """The honest trainer of the polygenic risk score task, for one cohort."""
    x, y = cohort

    def train(round_number, params):
        w, b = prs.descend(
            np.asarray(params["w"], dtype=np.float64).copy(), float(params["b"][0]), x, y, steps
        )
        return unflat(np.append(w, b)), len(y)

    return train


def federate(
    trainers, rounds, privacy=None, attempts=3, start=START, keep=None, helpers=1,
    complete=True, **options
):
    """Runs one trainer per site through a real federation and returns the run.

    The local paths are timing sensitive on a busy machine, so a run that
    fails is tried again. `keep` is a directory to run in, kept afterwards.
    With `complete` off, a run that finished with sites left out of rounds
    counts, and only the sites that were never left out have to succeed.
    """
    for attempt in range(attempts):
        holder = None if keep else tempfile.TemporaryDirectory()
        root = Path(keep) / f"attempt-{attempt}" if keep else holder.name
        try:
            run = ft.LocalFederation(
                root, start, sites=len(trainers), helpers=helpers, rounds=rounds, privacy=privacy,
                **options,
            ).run(lambda site, r, params: trainers[site](r, params), keep_spool=keep is not None)
            done = run.report.get("complete") if complete else run.report.get("finished")
            if done and (run.ok or not complete):
                run.model = {k: np.array(v) for k, v in run.model.items()}
                run.plan = plan_of(run.log_path)
                return run
            failure = run.aggregator_error or [s.get("error") for s in run.sites if not s["ok"]]
            print(f"run failed, attempt {attempt + 1}: {failure}", flush=True)
        finally:
            if holder:
                holder.cleanup()
    raise RuntimeError(f"a federation failed {attempts} times: {failure}")


def plan_of(log_path):
    """The signed plan a run started from: the first record of its log."""
    with open(log_path) as log:
        return json.loads(log.readline())["body"]


def private_plan(sites, rounds, privacy, start=START):
    """The plan a private federation would sign, without running it."""
    with tempfile.TemporaryDirectory() as root:
        federation = ft.LocalFederation(
            root, start, sites=sites, helpers=1, rounds=rounds, privacy=privacy, min_cohort=sites
        )
        federation.save()
        return plan_of(Path(root) / "plan.jsonl")


def centred(values):
    """Field elements from a spooled vector as signed integers."""
    negative = values > P // 2
    signed = values.astype(np.int64)
    signed[negative] = -(np.uint64(P) - values[negative]).astype(np.int64)
    return signed


def clip_to(update, clip):
    norm = np.linalg.norm(update)
    return update if norm <= clip else update * (clip / norm)


def private_encode(update, plan, rng, noise=True):
    """A numpy model of what a site submits in a private run, in grid units.

    The update is clipped to the plan's clip, rounded onto the plan's grid at
    random, and given Gaussian noise of the plan's variance rounded to the
    grid. The real encoder draws an exact discrete Gaussian. At the variances
    a plan uses, 8 grid units and up and here above 1e8, a rounded Gaussian
    differs from it by far less than these experiments resolve.
    """
    scale = float(1 << plan["encoding"]["frac_bits"])
    grid = np.floor(clip_to(update, plan["encoding"]["clip"]) * scale + rng.random(len(update)))
    if noise:
        grid += np.rint(rng.normal(0.0, np.sqrt(plan["privacy"]["noise"]), len(update)))
    return grid


def sensitivity(plan):
    """How far one site can move a sum, in grid units, as the plan accounts for it."""
    coords = sum(int(np.prod(t["shape"])) for t in plan["layout"]["tensors"])
    return plan["privacy"]["clip"] + int(np.ceil(np.sqrt(coords))) + 1


def zcdp_epsilon(rho, delta):
    """Epsilon for rho-zCDP at delta, as `privacy::epsilon` computes it
    (Canonne, Kamath and Steinke 2020, Proposition 12, minimised over the order)."""
    if rho <= 0:
        return 0.0
    orders = 1 + np.exp(np.linspace(-30, 30, 200001))
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = (
            orders * rho + np.log1p(-1 / orders) - (np.log(delta) + np.log(orders)) / (orders - 1)
        )
    simple = rho + 2 * np.sqrt(rho * np.log(1 / delta))
    return float(max(min(np.nanmin(bound), simple), 0.0))


def spread(values):
    values = [float(v) for v in values]
    deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(np.mean(values)), "std": deviation, "values": values}


def save(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}")
