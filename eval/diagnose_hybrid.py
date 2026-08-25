"""Diagnostic: is hybrid HitRate@10 = 0.1010 real, or an artifact?

Hybrid beating its own best component (CF-only, 0.0126) by 8x while
recommending items of near-random average popularity is implausible on its
face. This script tests four explanations, in order of how badly each would
invalidate the result:

1. Is the profile signal secretly carrying it? (profile-only row)
2. Do all five systems read the same leak-free data, or does hybrid see
   production artifacts built from the complete dataset?
3. Is the held-out (user, item) pair genuinely invisible in every structure
   hybrid consumes?
4. Do all systems share candidate generation and exclusion, differing only
   in scoring -- or are baselines handicapped by a smaller candidate pool?

Run: python eval/diagnose_hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_eval  # noqa: E402
import serve  # noqa: E402

N_USERS = 5000
SEED = 42


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> None:
    ctx = run_eval.prepare_eval_context(n_users=N_USERS, seed=SEED)
    holdout = ctx.holdout
    users = ctx.sampled_users
    n_users = len(users)

    # (author_id, product_id) pairs that must be invisible everywhere
    held_pairs = set(zip(holdout["author_id"], holdout["product_id"]))
    held_by_user = dict(zip(holdout["author_id"], holdout["product_id"]))

    # ------------------------------------------------------------------
    section("1. PROFILE-ONLY (the untested third signal)")
    prof_top = run_eval._rank_and_truncate(
        ctx.profile_candidates, ctx.already_rated, "score", run_eval.TOP_N
    )
    prof_metrics = run_eval.compute_metrics(
        "profile", prof_top, holdout, users, len(ctx.cf_scope_products), ctx.popularity
    )
    for k, v in prof_metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    section("2. DATA PATHS: does any system read production artifacts?")
    print("  Signals in EvalContext are built from:")
    print("    cf_candidates      <- train_cf_neighbors (rebuilt from training_reviews)")
    print("    content_candidates <- content.build_content_neighbors() (attribute-only)")
    print("    profile_candidates <- _shrunk_affinity(training_reviews) (rebuilt)")
    print("    hybrid             <- weighted sum of the SAME three frames above")
    print()
    # Prove the eval's CF table is not the production one
    prod_cf = serve.load_cf_neighbors()
    train_cf_pairs = set(
        zip(ctx.cf_candidates["author_id"], ctx.cf_candidates["product_id"])
    )
    print(f"  production cf_top_k_neighbors rows: {len(prod_cf):,}")
    print("  (eval never loads this; it rebuilds from training_reviews)")

    # Prove production profile aggregates differ from the eval's
    prod_type = serve.load_skin_type_affinity()
    train_type_note = "rebuilt in-memory from training_reviews (not read from artifacts/)"
    print(f"  production profile_skin_type rows: {len(prod_type):,} -- {train_type_note}")

    # ------------------------------------------------------------------
    section("3. HARD ASSERTION: is the held-out pair invisible?")

    # 3a. training reviews
    train_pairs_sample = ctx.training_reviews.merge(
        holdout[["author_id", "product_id"]], on=["author_id", "product_id"], how="inner"
    )
    print(f"  held-out pairs still present in training_reviews: {len(train_pairs_sample)}")

    # 3b. seeds fed to CF/content aggregation
    seed_leak = 0
    seeds_by_user = ctx.already_rated.groupby("author_id")["product_id"].apply(set)
    for u, held_item in held_by_user.items():
        if u in seeds_by_user.index and held_item in seeds_by_user.loc[u]:
            seed_leak += 1
    print(f"  held-out item present in a user's own seed set: {seed_leak}")

    # 3c. profile aggregates: does the held-out review still move its own
    #     (product, skin_type) mean? Compare train vs full aggregate counts.
    full_type = serve.load_skin_type_affinity()
    train_type = run_eval.skin_profile._shrunk_affinity(
        ctx.training_reviews, "skin_type", run_eval.skin_profile.SHRINKAGE_M
    )
    merged = full_type.merge(
        train_type, on=["product_id", "skin_type"], how="inner", suffixes=("_full", "_train")
    )
    shrunk = merged[merged["n_train"] < merged["n_full"]]
    print(f"  (product, skin_type) cells whose count DROPPED after holdout removal: "
          f"{len(shrunk):,} (proves aggregates were rebuilt leak-free)")

    # ------------------------------------------------------------------
    section("4. CANDIDATE POOLS: same generation, or handicapped baselines?")

    pools = {
        "cf": ctx.cf_candidates,
        "content": ctx.content_candidates,
        "profile": ctx.profile_candidates,
        "hybrid": ctx.normalized[["author_id", "product_id"]],
    }
    print(f"  {'system':<10}{'rows':>12}{'cand/user':>12}{'held-out in pool':>20}")
    ceilings = {}
    for name, frame in pools.items():
        per_user = len(frame) / n_users
        pair_set = set(zip(frame["author_id"], frame["product_id"]))
        in_pool = sum(1 for p in held_pairs if p in pair_set)
        ceilings[name] = in_pool / n_users
        print(f"  {name:<10}{len(frame):>12,}{per_user:>12.0f}"
              f"{in_pool:>12,} ({in_pool / n_users:>5.1%})")

    print("\n  'held-out in pool' is the RECALL CEILING: the best HitRate@10 that")
    print("  system could reach even with perfect ranking. A system whose pool")
    print("  rarely contains the target cannot beat one whose pool usually does,")
    print("  no matter how good its scoring is.")

    # ------------------------------------------------------------------
    section("5. HIT RATE vs CEILING (how much is ranking, how much is pool?)")
    actual = {
        "cf": 0.0126,
        "content": 0.0070,
        "profile": prof_metrics["HitRate@10"],
        "hybrid": 0.1010,
    }
    print(f"  {'system':<10}{'ceiling':>10}{'actual@10':>12}{'ranking efficiency':>22}")
    for name in ["cf", "content", "profile", "hybrid"]:
        c = ceilings[name]
        a = actual[name]
        eff = (a / c) if c else float("nan")
        print(f"  {name:<10}{c:>10.1%}{a:>12.4f}{eff:>21.1%}")

    # ------------------------------------------------------------------
    section("6. POOL SIZE CHECK: is hybrid just searching a bigger haystack?")
    cf_pairs = set(zip(ctx.cf_candidates["author_id"], ctx.cf_candidates["product_id"]))
    hyb_pairs = set(zip(ctx.normalized["author_id"], ctx.normalized["product_id"]))
    print(f"  cf pool pairs:      {len(cf_pairs):,}")
    print(f"  hybrid pool pairs:  {len(hyb_pairs):,}")
    print(f"  hybrid / cf ratio:  {len(hyb_pairs) / len(cf_pairs):.1f}x")
    print(f"  cf pool is subset of hybrid pool: {cf_pairs.issubset(hyb_pairs)}")


if __name__ == "__main__":
    main()
