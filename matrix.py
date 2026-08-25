"""Build the sparse user x item ratings matrix that cf.py operates on.

Rows are users with >=2 distinct rated products — a single-review user
contributes no co-occurrence signal to item-item similarity, but everyone
with 2+ does, so the cutoff is set at the minimum that contributes anything
rather than the stricter >=5 used later for leave-one-out evaluation
(eval needs deep-enough held-out history per user; matrix construction just
needs a real pairwise signal).

Columns are the full 2,351 reviewed-product set (data.reviewed_product_ids()),
not just the products that survive user filtering. Restricting to >=2-rating
users drops 21 products entirely (every review they have comes from a
single-review user) — those columns end up all-zero rather than disappearing,
so the index stays a stable, complete mapping over "the CF-scoped catalog"
and cf.py has to handle zero-signal items explicitly instead of the column
space silently shrinking out from under it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import data

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
MATRIX_PATH = PROCESSED_DIR / "user_item_matrix.npz"
USERS_PATH = PROCESSED_DIR / "user_item_matrix_users.npy"
PRODUCTS_PATH = PROCESSED_DIR / "user_item_matrix_products.npy"

MIN_USER_RATINGS = 2


@dataclass
class UserItemMatrix:
    """CSR ratings matrix plus stable id<->index mappings in both directions.

    Every downstream module (cf.py, blend.py, recommend.py) depends on these
    mappings staying fixed for the lifetime of a build — they're what let a
    similarity row computed in cf.py be traced back to a real product_id.
    """

    matrix: sp.csr_matrix
    user_ids: np.ndarray
    product_ids: np.ndarray
    user_index: dict[str, int]
    product_index: dict[str, int]

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_products(self) -> int:
        return self.matrix.shape[1]

    def user_row(self, author_id: str) -> int | None:
        return self.user_index.get(author_id)

    def product_col(self, product_id: str) -> int | None:
        return self.product_index.get(product_id)


def _build(min_user_ratings: int, reviews: pd.DataFrame | None = None) -> UserItemMatrix:
    """reviews defaults to data.load_reviews() (the normal offline-build
    path). The eval harness passes a training-only subset here instead —
    see eval/run_eval.py's leakage-prevention docstring — so this function
    stays the single place matrix-building logic lives rather than being
    duplicated for evaluation."""
    if reviews is None:
        reviews = data.load_reviews()

    counts = reviews.groupby("author_id")["product_id"].nunique()
    eligible_users = counts[counts >= min_user_ratings].index
    filtered = reviews[reviews["author_id"].isin(eligible_users)]

    product_ids = np.array(sorted(data.reviewed_product_ids()))
    user_ids = np.array(sorted(eligible_users))

    product_index = {pid: i for i, pid in enumerate(product_ids)}
    user_index = {uid: i for i, uid in enumerate(user_ids)}

    rows = filtered["author_id"].map(user_index).to_numpy()
    cols = filtered["product_id"].map(product_index).to_numpy()
    vals = filtered["rating"].to_numpy(dtype=np.float32)

    matrix = sp.csr_matrix(
        (vals, (rows, cols)), shape=(len(user_ids), len(product_ids))
    )

    zero_cols = int((np.asarray(matrix.sum(axis=0)).ravel() == 0).sum())
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    logger.info(
        "Built user-item matrix: %d users x %d products, density=%.4f%%, "
        "%d products have zero ratings from eligible (>=%d) users",
        matrix.shape[0], matrix.shape[1], density * 100, zero_cols, min_user_ratings,
    )

    return UserItemMatrix(matrix, user_ids, product_ids, user_index, product_index)


def build_from_reviews(
    reviews: pd.DataFrame, min_user_ratings: int = MIN_USER_RATINGS
) -> UserItemMatrix:
    """Build a UserItemMatrix from an arbitrary reviews DataFrame, uncached —
    for eval/run_eval.py, which needs a training-only matrix with held-out
    interactions removed and must never touch the production cache files
    (data/processed/user_item_matrix.npz) that build_user_item_matrix()
    reads and writes."""
    return _build(min_user_ratings, reviews=reviews)


def build_user_item_matrix(
    min_user_ratings: int = MIN_USER_RATINGS, force_rebuild: bool = False
) -> UserItemMatrix:
    """Load the cached matrix + mappings, or build and cache them.

    Caching is keyed only on min_user_ratings=2 (the fixed CF-build
    threshold) via the default file paths — pass force_rebuild=True after
    changing the underlying review data, or call _build() directly for an
    exploratory min_user_ratings other than the default.
    """
    if (
        not force_rebuild
        and MATRIX_PATH.exists()
        and USERS_PATH.exists()
        and PRODUCTS_PATH.exists()
    ):
        matrix = sp.load_npz(MATRIX_PATH)
        user_ids = np.load(USERS_PATH, allow_pickle=False)
        product_ids = np.load(PRODUCTS_PATH, allow_pickle=False)
        user_index = {uid: i for i, uid in enumerate(user_ids)}
        product_index = {pid: i for i, pid in enumerate(product_ids)}
        return UserItemMatrix(matrix, user_ids, product_ids, user_index, product_index)

    uim = _build(min_user_ratings)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(MATRIX_PATH, uim.matrix)
    np.save(USERS_PATH, uim.user_ids)
    np.save(PRODUCTS_PATH, uim.product_ids)
    logger.info("Cached user-item matrix to %s", MATRIX_PATH)

    return uim
