"""Load, clean, and cache the Sephora reviews + product catalog.

This is the entry point every other module in the pipeline depends on. It
does three things: dedups reviews (the raw CSVs contain re-scraped exact
duplicates and multiple reviews per (author_id, product_id) pair), validates
that every reviewed product_id resolves against the product catalog, and
caches the cleaned frames to parquet so a re-run of the pipeline doesn't
re-parse 529MB of CSV every time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]

# Columns kept from the raw review CSVs. product_name / brand_name / price_usd
# are dropped even though the raw files carry them per-review: they duplicate
# product_info.csv and can drift from it (price especially, since it's a
# point-in-time snapshot at review time). Anything product-level should be
# looked up by product_id against load_products() instead of trusted from a
# review row.
REVIEW_COLUMNS = [
    "author_id",
    "product_id",
    "rating",
    "is_recommended",
    "helpfulness",
    "total_feedback_count",
    "total_neg_feedback_count",
    "total_pos_feedback_count",
    "submission_time",
    "review_text",
    "review_title",
    "skin_tone",
    "skin_type",
    "hair_color",
    "eye_color",
]

CATEGORY_COLUMNS = ["skin_tone", "skin_type", "hair_color", "eye_color"]


def _load_raw_reviews() -> pd.DataFrame:
    frames = []
    for fname in REVIEW_FILES:
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run scripts/fetch_data.py first."
            )
        frames.append(pd.read_csv(path, index_col=0, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    return df[REVIEW_COLUMNS]


def _dedup_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop exact-duplicate rows, then collapse repeat (author_id, product_id)
    pairs to the most recent submission_time.

    Verified against this dataset (2026-08-25): 224 rows are byte-for-byte
    duplicates (re-scraped), and 5,525 rows belong to (author_id, product_id)
    pairs with more than one review. Keeping the latest by submission_time
    reflects the reviewer's most current opinion rather than an arbitrary
    pick; a stable sort before the drop means same-day ties resolve to
    whichever row appeared later in the source files, which is deterministic
    even though the date field alone can't break the tie.
    """
    before_exact = len(df)
    df = df.drop_duplicates()
    exact_dropped = before_exact - len(df)

    df = df.copy()
    df["submission_time"] = pd.to_datetime(df["submission_time"])
    df = df.sort_values("submission_time", kind="stable")

    before_pairs = len(df)
    df = df.drop_duplicates(subset=["author_id", "product_id"], keep="last")
    pair_dropped = before_pairs - len(df)

    df = df.sort_values("submission_time", kind="stable").reset_index(drop=True)

    # Everything downstream -- matrix.py's user x item construction, the
    # leave-one-out holdout in eval/run_eval.py -- assumes one row per
    # (author_id, product_id). Assert it rather than trusting it: when this
    # silently failed once, the bad rows propagated all the way into the
    # committed serving artifacts before anything noticed.
    remaining_dupes = int(df.duplicated(subset=["author_id", "product_id"]).sum())
    if remaining_dupes:
        raise AssertionError(
            f"{remaining_dupes} duplicate (author_id, product_id) pairs survived "
            "deduplication -- check that dtypes are normalized before this runs."
        )

    stats = {
        "exact_duplicate_rows_dropped": exact_dropped,
        "repeat_pair_rows_dropped": pair_dropped,
        "remaining_reviews": len(df),
    }
    logger.info("Deduped reviews: %s", stats)
    return df, stats


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = df["rating"].astype("int8")
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")
    df["author_id"] = df["author_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    return df


def _validate_product_join(reviews: pd.DataFrame, products: pd.DataFrame) -> None:
    """Every product_id in reviews must resolve in the product catalog.

    Verified 0 orphans on this dataset, but this stays a hard check rather
    than a silent filter: if a future refresh of the data introduces
    unresolvable product_ids, matrix.py and content.py would silently produce
    wrong-shaped artifacts instead of failing loudly at the source.
    """
    review_ids = set(reviews["product_id"].unique())
    product_ids = set(products["product_id"].unique())
    orphans = review_ids - product_ids
    if orphans:
        sample = sorted(orphans)[:10]
        raise ValueError(
            f"{len(orphans)} product_id(s) in reviews have no match in "
            f"product_info.csv — sample: {sample}"
        )


def load_reviews(force_rebuild: bool = False) -> pd.DataFrame:
    """Deduped, dtype-cast reviews. Cached to data/processed/reviews.parquet."""
    cache_path = PROCESSED_DIR / "reviews.parquet"
    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    # Cast BEFORE dedup, not after. Deduplication compares raw values, so if
    # author_id/product_id arrive with inconsistent inferred types, the int
    # 1011200472 and the string "1011200472" are treated as different keys and
    # both survive -- then casting to str afterwards silently collapses them
    # into genuine duplicate (author_id, product_id) pairs in the output. That
    # actually happened here: an early build (before low_memory=False forced
    # consistent inference) produced 5 such pairs, which is exactly the
    # condition matrix.py assumes cannot exist. Normalizing types first makes
    # the dedup independent of how pandas happened to infer each chunk.
    df = _load_raw_reviews()
    df = _cast_dtypes(df)
    df, _ = _dedup_reviews(df)
    _validate_product_join(df, load_products(force_rebuild=force_rebuild))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached cleaned reviews to %s (%d rows)", cache_path, len(df))
    return df


def load_products(force_rebuild: bool = False) -> pd.DataFrame:
    """Full product catalog (8,494 products, all categories). Cached to
    data/processed/products.parquet."""
    cache_path = PROCESSED_DIR / "products.parquet"
    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    path = RAW_DIR / "product_info.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/fetch_data.py first.")
    df = pd.read_csv(path)
    df["product_id"] = df["product_id"].astype(str)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached product catalog to %s (%d rows)", cache_path, len(df))
    return df


def load_reviewed_products(force_rebuild: bool = False) -> pd.DataFrame:
    """The 2,351-product subset of the catalog that has at least one review.

    This is the scope for CF (matrix.py, cf.py) and skin-profile affinity
    (skin_profile.py). Verified 2026-08-25: every reviewed product is
    primary_category == "Skincare" — the review corpus doesn't cover the
    other ~6,100 makeup/hair/fragrance/etc. products at all, so those signals
    have nothing to compute for them. content.py is the only signal scoped to
    the full catalog instead of this subset.
    """
    products = load_products(force_rebuild=force_rebuild)
    reviewed_ids = reviewed_product_ids(force_rebuild=force_rebuild)
    subset = products[products["product_id"].isin(reviewed_ids)].reset_index(drop=True)
    return subset


def reviewed_product_ids(force_rebuild: bool = False) -> frozenset[str]:
    """The set of product_ids that have at least one review."""
    reviews = load_reviews(force_rebuild=force_rebuild)
    return frozenset(reviews["product_id"].unique())
