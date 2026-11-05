"""An empirical audit of the site-level privacy guarantee, from one run.

After Steinke, Nasr and Jagielski, "Privacy auditing with one (1) training
run" (NeurIPS 2023, arXiv:2305.08846). There are m canary sites. A fair coin
decides for each whether its data is there: if so it submits a random
direction at the clip norm, and if not it submits no change, which is the
"all of one site's data replaced with nothing" of the design's guarantee.
Every canary is in the cohort either way, since who takes part is public.
The auditor sees the released model, scores each canary by the dot product of
its direction with the model's movement, guesses "there" for the highest
scores and "not there" for the lowest, and abstains on the rest. The number
of correct guesses becomes a lower bound on epsilon.

Measured: a real private federation of --canaries sites, at each epsilon.
Simulated: the same mechanism in numpy at --simulated-canaries, with the
grid, clip and noise variance the system's own planner picks for a cohort of
that size, read from a signed plan. Also simulated, as a control: the same
with the noise left out, which shows the most the audit could report.

This is also the membership inference test the guarantee answers to: is a
site's data in the run. Membership of one person inside a site is out of
scope, since the guarantee is per site and says nothing finer.

A lower bound above the epsilon the log says was spent would be a bug in the
mechanism or the accounting, and the script says so loudly and exits 1.

    python benchmarks/attacks/audit.py --out benchmarks/results/attack-audit.json
"""

import argparse
import math
import sys

import numpy as np

import common
import fedtally as ft


def binomial_pmf(n, q):
    k = np.arange(n + 1)
    log_choose = math.lgamma(n + 1) - np.array(
        [math.lgamma(i + 1) + math.lgamma(n - i + 1) for i in k]
    )
    with np.errstate(divide="ignore"):
        return np.exp(log_choose + k * np.log(q) + (n - k) * np.log1p(-q))


def p_value(m, r, v, epsilon, delta):
    """The chance of v or more correct guesses out of r if the run is (epsilon, delta)-DP.

    Theorem 5.2 of Steinke, Nasr and Jagielski (arXiv:2305.08846v1), equations
    (7) to (9): P[W >= v] <= beta + alpha * 2 m delta, with W* a
    Binomial(r, e^eps / (e^eps + 1)), beta = P[W* >= v] (8) and
    alpha = max over i of (P[W* >= v - i] - beta) / i (9). Guesses are plus or
    minus one or abstain, so the binomial dominates as in their Corollary 5.4.
    This follows the reference code in their appendix.
    """
    pmf = binomial_pmf(r, 1 / (1 + math.exp(-epsilon)))
    beta = pmf[v:].sum()
    alpha = 0.0
    if v > 0:
        below = np.cumsum(pmf[:v][::-1])
        alpha = float((below / np.arange(1, v + 1)).max())
    return min(1.0, beta + alpha * 2 * m * delta)


def epsilon_lower_bound(m, r, v, delta, p):
    """The largest epsilon the guesses rule out at confidence 1 - p."""
    if r == 0 or p_value(m, r, v, 0.0, delta) >= p:
        return 0.0
    low, high = 0.0, 1.0
    while p_value(m, r, v, high, delta) < p:
        low, high = high, high * 2
        if high > 64:
            return low
    for _ in range(40):
        middle = (low + high) / 2
        if p_value(m, r, v, middle, delta) < p:
            low = middle
        else:
            high = middle
    return low


def directions(m, dim, rng):
    """m random orthonormal directions, so one canary's score carries no other canary."""
    q, _ = np.linalg.qr(rng.normal(size=(dim, m)))
    return q.T


def audit(scores, present, clip, delta, claimed, confidence, guess_counts):
    """Guesses from the scores, and the lower bound they support.

    Several numbers of guesses are tried. Each is tested at a share of the
    error budget, so the best of them is still a bound at `confidence`.
    """
    m = len(scores)
    order = np.argsort(scores)
    p = (1 - confidence) / len(guess_counts)
    tried = []
    for r in guess_counts:
        low, high = order[: r // 2], order[m - (r - r // 2) :]
        correct = int((~present[low]).sum() + present[high].sum())
        bound = epsilon_lower_bound(m, r, correct, delta, p)
        tried.append({"guesses": r, "correct": correct, "epsilon_lower_bound": bound})
    best = max(tried, key=lambda t: t["epsilon_lower_bound"])
    # A present canary moves its score by the clip, so half of it splits the two.
    threshold = (scores > clip / 2) == present
    return {
        "canaries": m,
        "present": int(present.sum()),
        "membership_accuracy": float(threshold.mean()),
        "guess_counts": tried,
        "epsilon_lower_bound": best["epsilon_lower_bound"],
        "epsilon_claimed": claimed,
        "delta": delta,
        "confidence": confidence,
        "exceeds_claim": bool(best["epsilon_lower_bound"] > claimed),
    }


def guess_counts(m):
    return sorted({max(2, m // 8), m // 4, m // 2, m})


def real(m, dim, rounds, privacy, rng):
    """m canary sites through a real private federation."""
    present = rng.random(m) < 0.5
    unit = directions(m, dim, rng)
    start = {"w": np.zeros(dim, np.float32)}

    def canary(i):
        def train(round_number, params):
            w = np.asarray(params["w"], dtype=np.float64)
            if present[i]:
                w = w + privacy.clip * unit[i]
            return {"w": w.astype(np.float32)}, 1

        return train

    run = common.federate([canary(i) for i in range(m)], rounds, privacy, start=start, min_cohort=m)
    moved = np.asarray(run.model["w"], dtype=np.float64) * m
    return unit @ moved, present, run.plan, run.report["privacy"]


def simulated(m, dim, rounds, privacy, rng, noise=True):
    """The same audit on a numpy model of the mechanism, with a real plan's terms."""
    start = {"w": np.zeros(dim, np.float32)}
    plan = common.private_plan(m, rounds, privacy, start=start)
    scale = float(1 << plan["encoding"]["frac_bits"])
    present = rng.random(m) < 0.5
    unit = directions(m, dim, rng)
    total = np.zeros(dim)
    for _ in range(rounds):
        for i in range(m):
            update = privacy.clip * unit[i] if present[i] else np.zeros(dim)
            total += common.private_encode(update, plan, rng, noise=noise)
    spent = rounds * common.sensitivity(plan) ** 2 / (2 * plan["privacy"]["noise"] * m)
    claimed = common.zcdp_epsilon(spent, privacy.delta)
    return unit @ (total / scale), present, plan, claimed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canaries", type=int, default=32, help="sites in the real run")
    parser.add_argument("--simulated-canaries", type=int, default=2000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1000, help="model size in the real run")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    m = args.simulated_canaries
    wide = max(args.dim, 2 * m)
    shared = (args.clip, args.delta)
    rows, bug = [], False
    for epsilon in args.epsilons:
        privacy = ft.Privacy(epsilon=epsilon, delta=args.delta, clip=args.clip)

        scores, present, plan, spent = real(args.canaries, args.dim, args.rounds, privacy, rng)
        counts = guess_counts(args.canaries)
        measured = audit(scores, present, *shared, spent["epsilon_spent"], args.confidence, counts)
        measured["rho_spent"] = spent["rho_spent"]
        measured["plan_privacy_terms"] = plan["privacy"]

        scores, present, plan, claimed = simulated(m, wide, args.rounds, privacy, rng)
        model = audit(scores, present, *shared, claimed, args.confidence, guess_counts(m))
        model["plan_privacy_terms"] = plan["privacy"]

        scores, present, _, _ = simulated(m, wide, args.rounds, privacy, rng, noise=False)
        control = audit(scores, present, *shared, claimed, args.confidence, guess_counts(m))

        rows.append(
            {"epsilon": epsilon, "measured": measured, "simulated": model, "simulated_without_noise": control}
        )
        bug = bug or measured["exceeds_claim"] or model["exceeds_claim"]
        bounds = [a["epsilon_lower_bound"] for a in (measured, model, control)]
        print(epsilon, *bounds, flush=True)

    print(f"\nlower bound on epsilon at delta {args.delta:g}, confidence {args.confidence:g}")
    print("target  kind                        canaries  best guesses  correct  lower bound  claimed  accuracy")
    for row in rows:
        for kind, label in [
            ("measured", "measured, real federation"),
            ("simulated", "simulated, plan's terms"),
            ("simulated_without_noise", "simulated, noise left out"),
        ]:
            a = row[kind]
            best = max(a["guess_counts"], key=lambda t: t["epsilon_lower_bound"])
            print(
                f"{row['epsilon']:6g}  {label:26s}  {a['canaries']:8d}  {best['guesses']:12d}  "
                f"{best['correct']:7d}  {a['epsilon_lower_bound']:11.3f}  {a['epsilon_claimed']:7.3f}  "
                f"{a['membership_accuracy']:.3f}"
            )
    print("the rows with the noise left out are a control: the most this many canaries can show")
    if bug:
        print("\n*** BUG: AN EMPIRICAL LOWER BOUND EXCEEDS THE EPSILON THE LOG CLAIMS ***")

    common.save(
        args.out,
        {
            "attack": "site_level_privacy_audit",
            "method": "Steinke, Nasr, Jagielski, arXiv:2305.08846v1, Theorem 5.2, equations (7)-(9)",
            "delta": args.delta,
            "clip": args.clip,
            "rounds": args.rounds,
            "rows": rows,
            "lower_bound_exceeds_claim": bug,
            "notes": [
                "measured: a real federation. simulated: numpy, with clip, grid and noise "
                "variance read from a plan the system signed for that many sites. The "
                "simulation rounds a Gaussian where the system draws an exact discrete one.",
                "A canary that is absent still joins and adds noise; only its update is "
                "empty. That matches the guarantee, which is about what a site contributed.",
                "The bound is a lower bound. With m canaries it cannot exceed about "
                "log of m over the error budget, so a small real run can only show a small one.",
                "membership_accuracy is site-level membership inference on the same scores, "
                "with the threshold at half the clip; 0.5 is chance.",
                "The control with the noise left out gives the ceiling of the audit at that "
                "many canaries. Where the ceiling is below the claim, the audit could not "
                "have contradicted the claim, and passing it says little.",
                "Membership inference on individuals inside a site was not run. The "
                "guarantee is per site.",
            ],
        },
    )
    if bug:
        sys.exit(1)


if __name__ == "__main__":
    main()
