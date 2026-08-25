"""Grid search over blend.py's W_CF / W_CONTENT / W_PROFILE, optimizing
NDCG@10 against the same leave-one-out protocol run_eval.py uses.

Method: prepare_eval_context() does the expensive, weight-independent work
once (sample the cohort, hold out most-recent interactions, rebuild CF
similarity and skin-profile affinity from training data only, generate and
normalize each signal's candidates). Each grid point then only re-applies
weights, re-ranks, and re-scores -- so a ~60-point sweep costs roughly one
setup plus 60 cheap re-rankings rather than 60 full evaluations.

Weights are searched on a simplex (they sum to 1). Scaling all three by a
constant would reorder nothing -- ranking is invariant to a positive scalar
multiple of the score -- so only their relative proportions matter, and
constraining the sum removes a redundant degree of freedom that would
otherwise waste grid points on duplicates.

A single sweep on one sample is NOT enough to pick weights. The NDCG@10
differences between reasonable weight combinations here are smaller than
the variation between evaluation samples, so the top-ranked combination on
one seed is frequently not the top on another -- picking it would be
fitting the sample, not the problem. verify_across_seeds() re-evaluates a
shortlist on several independent cohort samples and reports the per-seed
and mean results, which is what any weight decision should actually rest on.

IMPORTANT: this script reports; it does not write tuned weights back into
blend.py. Applying a result is a deliberate edit, so that what the deployed
app scores with is always something a human chose, not something a sweep
silently mutated.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import blend  # noqa: E402
import run_eval  # noqa: E402  (same directory)

STEP = 0.1
RESULTS_PATH = Path(__file__).resolve().parent / "tuning_results.csv"


def _simplex_grid(step: float = STEP) -> list[tuple[float, float, float]]:
    """All (w_cf, w_content, w_profile) on a `step`-spaced simplex summing
    to 1. The all-zero-weight point is impossible by construction (they sum
    to 1), but a single signal carrying all the weight IS included -- those
    points are exactly the "CF-only"/"content-only"/"profile-only" corners
    and are worth having in the sweep as sanity anchors."""
    points = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            points.append((i * step, j * step, k * step))
    return points


def run(n_users: int = run_eval.N_EVAL_USERS, seed: int = run_eval.RANDOM_SEED) -> pd.DataFrame:
    t0 = time.time()
    print("Preparing eval context (one-time, weight-independent)...")
    ctx = run_eval.prepare_eval_context(n_users=n_users, seed=seed)

    grid = _simplex_grid()
    print(f"\nSweeping {len(grid)} weight combinations...")

    rows = []
    for idx, (w_cf, w_content, w_profile) in enumerate(grid, start=1):
        metrics = run_eval.evaluate_weights(
            ctx, w_cf=w_cf, w_content=w_content, w_profile=w_profile
        )
        metrics.update({"w_cf": w_cf, "w_content": w_content, "w_profile": w_profile})
        rows.append(metrics)
        if idx % 10 == 0 or idx == len(grid):
            print(f"  {idx}/{len(grid)}")

    results = pd.DataFrame(rows).sort_values("NDCG@10", ascending=False).reset_index(drop=True)

    default = run_eval.evaluate_weights(ctx, system_name="hybrid (default weights)")
    best = results.iloc[0]

    print(f"\nDone in {time.time() - t0:.1f}s\n")

    print("=== DEFAULT WEIGHTS ===")
    print(f"  w_cf={blend.W_CF}, w_content={blend.W_CONTENT}, w_profile={blend.W_PROFILE}")
    print(f"  NDCG@10={default['NDCG@10']:.6f}  HitRate@10={default['HitRate@10']:.4f}  "
          f"Coverage={default['Coverage']:.4f}  AvgPopularity={default['AvgPopularity']:.1f}")

    print("\n=== BEST BY NDCG@10 ===")
    print(f"  w_cf={best['w_cf']:.1f}, w_content={best['w_content']:.1f}, w_profile={best['w_profile']:.1f}")
    print(f"  NDCG@10={best['NDCG@10']:.6f}  HitRate@10={best['HitRate@10']:.4f}  "
          f"Coverage={best['Coverage']:.4f}  AvgPopularity={best['AvgPopularity']:.1f}")

    delta = best["NDCG@10"] - default["NDCG@10"]
    pct = (delta / default["NDCG@10"] * 100) if default["NDCG@10"] else float("nan")
    print(f"\n  NDCG@10 change: {delta:+.6f} ({pct:+.2f}%)")

    print("\n=== TOP 10 BY NDCG@10 ===")
    cols = ["w_cf", "w_content", "w_profile", "NDCG@10", "HitRate@10", "Coverage", "AvgPopularity"]
    print(results[cols].head(10).to_string(index=False))

    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved full sweep to {RESULTS_PATH}")

    return results


VERIFY_SEEDS = (42, 7, 123)
VERIFY_RESULTS_PATH = Path(__file__).resolve().parent / "tuning_verification.csv"


def verify_across_seeds(
    combos: list[tuple[float, float, float, str]],
    seeds: tuple[int, ...] = VERIFY_SEEDS,
    n_users: int = run_eval.N_EVAL_USERS,
) -> pd.DataFrame:
    """Re-evaluate a shortlist of weight combinations across several
    independent eval-cohort samples. Each seed re-samples which users are
    held out, so comparing the same combo across seeds shows how much of an
    apparent improvement is just which 5,000 users got drawn.

    Returns one row per (combo, seed) plus mean/std/win-count summary rows.
    """
    rows = []
    for seed in seeds:
        print(f"  seed {seed}: preparing context...")
        ctx = run_eval.prepare_eval_context(n_users=n_users, seed=seed, verbose=False)
        for w_cf, w_content, w_profile, label in combos:
            m = run_eval.evaluate_weights(ctx, w_cf=w_cf, w_content=w_content, w_profile=w_profile)
            rows.append(
                {
                    "label": label,
                    "w_cf": w_cf,
                    "w_content": w_content,
                    "w_profile": w_profile,
                    "seed": seed,
                    "NDCG@10": m["NDCG@10"],
                    "HitRate@10": m["HitRate@10"],
                    "Coverage": m["Coverage"],
                }
            )

    df = pd.DataFrame(rows)

    winners = df.loc[df.groupby("seed")["NDCG@10"].idxmax(), "label"]
    win_counts = winners.value_counts()

    summary = (
        df.groupby(["label", "w_cf", "w_content", "w_profile"])
        .agg(
            mean_NDCG=("NDCG@10", "mean"),
            std_NDCG=("NDCG@10", "std"),
            mean_HitRate=("HitRate@10", "mean"),
            mean_Coverage=("Coverage", "mean"),
        )
        .reset_index()
    )
    summary["seeds_won"] = summary["label"].map(win_counts).fillna(0).astype(int)
    summary = summary.sort_values("mean_NDCG", ascending=False).reset_index(drop=True)

    print("\n=== PER-SEED NDCG@10 ===")
    pivot = df.pivot(index="label", columns="seed", values="NDCG@10")
    print(pivot.to_string())

    print("\n=== SUMMARY ACROSS SEEDS (sorted by mean NDCG@10) ===")
    print(summary.to_string(index=False))

    spread = df.groupby("label")["NDCG@10"].agg(lambda s: s.max() - s.min()).max()
    best_gap = summary["mean_NDCG"].max() - summary["mean_NDCG"].min()
    print(f"\n  Largest within-combo spread across seeds: {spread:.6f}")
    print(f"  Gap between best and worst combo (mean):   {best_gap:.6f}")
    if spread > best_gap:
        print("  -> Between-sample variation EXCEEDS between-combo variation:")
        print("     these weights are not meaningfully distinguishable on this data.")

    df.to_csv(VERIFY_RESULTS_PATH, index=False)
    print(f"\nSaved per-seed detail to {VERIFY_RESULTS_PATH}")

    return summary


if __name__ == "__main__":
    if "--verify" in sys.argv:
        shortlist = [
            (blend.W_CF, blend.W_CONTENT, blend.W_PROFILE, "default"),
            (0.4, 0.5, 0.1, "sweep-best-seed42"),
            (0.3, 0.3, 0.4, "runner-up"),
            (0.4, 0.3, 0.3, "balanced"),
        ]
        verify_across_seeds(shortlist)
    else:
        run()
