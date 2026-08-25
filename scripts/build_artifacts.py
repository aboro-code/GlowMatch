"""Run the full offline pipeline and write everything the app needs into
artifacts/ — the deployed app loads these and must never compute embeddings
or similarity matrices at request time.

Usage:
    python scripts/build_artifacts.py [--force]

--force recomputes every stage from the raw CSVs even if data/processed/
already has cached results (normally each stage's own caching just reuses
what's there, which is what you want for a quick rebuild after only one
module changed).

What's intermediate (data/processed/, gitignored, can be large) vs. what
ships (artifacts/, committed, kept small):

- data/processed/reviews.parquet keeps every review column, including
  free-text review_text/review_title — that's what makes it ~206MB, well
  past GitHub's 100MB file limit and not what "small artifacts" means.
  Nothing downstream of the offline build actually needs the text: CF,
  content, and profile all only need author_id/product_id/rating/skin_type/
  skin_tone. artifacts/reviews_slim.parquet keeps only those 5 columns
  (~11MB) — this is what a deployed recommend_by_user() should read a
  user's rating history from, not the full reviews table.
- data/processed/content_embeddings.npy and the user-item matrix files are
  pure intermediates for computing the top-K neighbor tables; nothing at
  serving time touches raw embeddings or the sparse matrix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cf  # noqa: E402
import content  # noqa: E402
import data  # noqa: E402
import matrix  # noqa: E402
import skin_profile  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts"

REVIEWS_SLIM_COLUMNS = ["author_id", "product_id", "rating", "skin_type", "skin_tone"]


def _step(label: str, fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:6.1f}s] {label}")
    return result


def build(force: bool) -> None:
    print("Running offline pipeline...")
    reviews = _step("reviews cleaned + cached", data.load_reviews, force_rebuild=force)
    products = _step("product catalog cached", data.load_products, force_rebuild=force)
    _step("user-item matrix built", matrix.build_user_item_matrix, force_rebuild=force)
    cf_neighbors = _step("CF top-K neighbors", cf.build_cf_neighbors, force_rebuild=force)
    content_neighbors = _step(
        "content top-K neighbors", content.build_content_neighbors, force_rebuild=force
    )
    skin_type_affinity = _step(
        "skin_type affinity", skin_profile.build_skin_type_affinity, force_rebuild=force
    )
    skin_tone_affinity = _step(
        "skin_tone affinity", skin_profile.build_skin_tone_affinity, force_rebuild=force
    )

    print(f"\nWriting serving artifacts to {ARTIFACTS_DIR} ...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "products.parquet": products,
        "reviews_slim.parquet": reviews[REVIEWS_SLIM_COLUMNS],
        "cf_top_k_neighbors.parquet": cf_neighbors,
        "content_top_k_neighbors.parquet": content_neighbors,
        "profile_skin_type.parquet": skin_type_affinity,
        "profile_skin_tone.parquet": skin_tone_affinity,
    }
    for fname, df in outputs.items():
        df.to_parquet(ARTIFACTS_DIR / fname, index=False)

    print("\nArtifact sizes:")
    total_bytes = 0
    for fname in outputs:
        size = (ARTIFACTS_DIR / fname).stat().st_size
        total_bytes += size
        print(f"  {fname:32s} {size / 1e6:8.2f} MB")
    print(f"  {'TOTAL':32s} {total_bytes / 1e6:8.2f} MB")

    over_limit = [f for f in outputs if (ARTIFACTS_DIR / f).stat().st_size > 100_000_000]
    if over_limit:
        print(f"\nWARNING: exceeds GitHub's 100MB file limit: {over_limit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Recompute every stage, ignoring existing caches."
    )
    args = parser.parse_args()
    build(force=args.force)
