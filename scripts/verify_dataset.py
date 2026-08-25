"""One-off dataset verification: confirm the Sephora reviews data actually
supports item-item collaborative filtering before any pipeline is built.

Not part of the production pipeline -- run manually, read the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    section("0. File sizes and row counts")
    total_bytes = 0
    frames = []
    for fname in REVIEW_FILES:
        path = DATA_RAW / fname
        size_mb = path.stat().st_size / 1e6
        total_bytes += path.stat().st_size
        df = pd.read_csv(path, index_col=0, low_memory=False)
        print(f"  {fname:28s} {size_mb:8.1f} MB   {len(df):>9,} rows")
        frames.append(df)

    product_path = DATA_RAW / "product_info.csv"
    product_size_mb = product_path.stat().st_size / 1e6
    total_bytes += product_path.stat().st_size
    products = pd.read_csv(product_path)
    print(f"  {'product_info.csv':28s} {product_size_mb:8.1f} MB   {len(products):>9,} rows")
    print(f"  TOTAL on-disk size: {total_bytes / 1e6:.1f} MB")

    reviews = pd.concat(frames, ignore_index=True)
    del frames
    print(f"\n  Combined reviews shape: {reviews.shape}")

    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    gi_text = gitignore.read_text() if gitignore.exists() else ""
    print(f"  data/ gitignored: {'data/' in gi_text}")

    section("1. Per-review user identifier")
    print(f"  Column: author_id, dtype={reviews['author_id'].dtype}")
    print(f"  Null count: {reviews['author_id'].isna().sum()}")

    section("2. Rating column")
    print(f"  Column: rating, dtype={reviews['rating'].dtype}")
    print(f"  Range: [{reviews['rating'].min()}, {reviews['rating'].max()}]")
    print("  Distribution:")
    print(reviews["rating"].value_counts().sort_index().to_string())
    print(f"  Null count: {reviews['rating'].isna().sum()}")

    section("3. Totals: reviews, users, products")
    n_reviews = len(reviews)
    n_users = reviews["author_id"].nunique()
    n_products = reviews["product_id"].nunique()
    print(f"  Total reviews:        {n_reviews:,}")
    print(f"  Distinct users:       {n_users:,}")
    print(f"  Distinct products:    {n_products:,}")

    section("4. User x item matrix sparsity")
    possible_cells = n_users * n_products
    filled_cells = len(reviews.drop_duplicates(["author_id", "product_id"]))
    sparsity = 1 - filled_cells / possible_cells
    density = filled_cells / possible_cells
    print(f"  Possible cells (users x products): {possible_cells:,}")
    print(f"  Filled cells (distinct user-product pairs): {filled_cells:,}")
    print(f"  Density: {density:.6%}")
    print(f"  Sparsity: {sparsity:.6%}")

    section("5. User activity distribution (distinct products rated per user)")
    per_user_products = reviews.groupby("author_id")["product_id"].nunique()
    for k in [1, 2, 5, 10, 20]:
        n_at_least_k = (per_user_products >= k).sum()
        pct = n_at_least_k / n_users * 100
        print(f"  Users with >= {k:>2} distinct products rated: {n_at_least_k:>9,}  ({pct:5.2f}% of users)")
    print(f"\n  Mean distinct products/user: {per_user_products.mean():.3f}")
    print(f"  Median distinct products/user: {per_user_products.median():.1f}")
    print(f"  Max distinct products/user: {per_user_products.max()}")

    section("6. Product review-count distribution")
    per_product_reviews = reviews.groupby("product_id")["author_id"].count()
    for k in [5, 20, 50]:
        n_at_least_k = (per_product_reviews >= k).sum()
        pct = n_at_least_k / n_products * 100
        print(f"  Products with >= {k:>2} reviews: {n_at_least_k:>6,}  ({pct:5.2f}% of products)")
    print(f"\n  Mean reviews/product: {per_product_reviews.mean():.1f}")
    print(f"  Median reviews/product: {per_product_reviews.median():.1f}")

    section("7. Skin-profile signal coverage")
    for col in ["skin_tone", "skin_type", "hair_color", "eye_color"]:
        non_null = reviews[col].notna().sum()
        pct = non_null / n_reviews * 100
        n_distinct = reviews[col].nunique(dropna=True)
        print(f"\n  {col}: {non_null:,} / {n_reviews:,} populated ({pct:.1f}%), {n_distinct} distinct values")
        print(f"    Values: {sorted(reviews[col].dropna().unique().tolist())}")

    section("8. data/ location + gitignore (recap)")
    print(f"  Raw files live in: {DATA_RAW}")
    print(f"  data/ gitignored: {'data/' in gi_text}")

    section("9. Reviews <-> product_info join integrity")
    review_pids = set(reviews["product_id"].dropna().unique())
    product_pids = set(products["product_id"].dropna().unique())
    in_reviews_not_products = review_pids - product_pids
    in_products_not_reviews = product_pids - review_pids
    print(f"  Distinct product_ids in reviews:      {len(review_pids):,}")
    print(f"  Distinct product_ids in product_info: {len(product_pids):,}")
    print(f"  In reviews but NOT in product_info:   {len(in_reviews_not_products):,}")
    print(f"  In product_info but NOT in reviews:   {len(in_products_not_reviews):,}")
    if in_reviews_not_products:
        sample = list(in_reviews_not_products)[:10]
        print(f"    sample missing ids: {sample}")

    section("10. Data quality checks")
    dup_pairs = reviews.duplicated(subset=["author_id", "product_id"]).sum()
    print(f"  Duplicate (author_id, product_id) rows (exact re-reviews or dupes): {dup_pairs:,}")

    bad_ratings = reviews[~reviews["rating"].between(1, 5)]
    print(f"  Ratings outside [1,5]: {len(bad_ratings):,}")

    null_author = reviews["author_id"].isna().sum()
    null_product = reviews["product_id"].isna().sum()
    null_rating = reviews["rating"].isna().sum()
    print(f"  Null author_id: {null_author:,}  Null product_id: {null_product:,}  Null rating: {null_rating:,}")

    exact_full_dup = reviews.duplicated().sum()
    print(f"  Fully duplicate rows: {exact_full_dup:,}")

    try:
        sample_text = reviews["review_text"].dropna().astype(str)
        weird = sample_text[sample_text.str.contains("Ã|â€|\\ufffd", regex=True, na=False)]
        print(f"  review_text rows with likely encoding artifacts (Ã/â€/replacement char): {len(weird):,}")
    except Exception as exc:
        print(f"  encoding check failed: {exc}")

    section("Summary numbers for the viability call")
    print(f"  n_users={n_users:,} n_products={n_products:,} n_reviews={n_reviews:,}")
    print(f"  density={density:.6%}")
    for k in [1, 2, 5, 10, 20]:
        n_at_least_k = (per_user_products >= k).sum()
        print(f"  users>={k}: {n_at_least_k:,} ({n_at_least_k/n_users:.2%})")


if __name__ == "__main__":
    main()
