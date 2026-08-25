"""Item-item collaborative filtering: adjusted cosine similarity, top-K
neighbor persistence.

Why adjusted cosine instead of raw-rating cosine: 82% of ratings in this
dataset are 4 or 5 stars (see CLAUDE.md's verified data numbers). On that
distribution, raw-rating cosine similarity barely distinguishes itself from
binary co-occurrence — almost every pair of co-rated items looks like "both
rated highly" regardless of whether a user actually preferred one over the
other. Mean-centering each user's ratings before computing cosine similarity
(the "adjusted cosine similarity" from Sarwar et al.'s item-based CF paper)
surfaces relative preference instead: whether a user rated an item above or
below their own average, which is the signal that actually distinguishes
"this user especially liked X" from "this user rates everything a 4."

Why top-K persistence, not a dense item x item matrix: even though the
CF-scoped catalog (2,351 products) is small enough that a dense 2,351x2,351
float32 matrix would technically fit in memory (~22MB), persisting only the
K=50 highest-similarity neighbors per item is the correct default regardless
of current catalog size — it's O(N*K) rather than O(N^2), it's what makes
this approach describable as "the Amazon 2003 item-to-item CF" method, and it
keeps the design correct if the catalog grows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

import matrix as mx

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
NEIGHBORS_PATH = PROCESSED_DIR / "cf_top_k_neighbors.parquet"

TOP_K = 50


def _mean_center_sparse(X: sp.csr_matrix) -> sp.csr_matrix:
    """Subtract each user's own mean rating from their stored ratings.

    Operates only on X.data (the stored non-zero entries), never on a dense
    copy of X. np.repeat(user_means, np.diff(X.indptr)) expands the one
    mean-per-user array out to one value per stored entry, aligned to
    X.data's row-major layout, so the subtraction is a single vectorized op
    over ~795k entries rather than a 209,105 x 2,351 (~2GB) dense array.
    """
    user_counts = np.diff(X.indptr)
    assert (user_counts > 0).all(), "matrix.py should only include users with >=1 rating"
    user_sums = np.asarray(X.sum(axis=1)).ravel()
    user_means = user_sums / user_counts

    Xc = X.copy()
    Xc.data = Xc.data - np.repeat(user_means, user_counts)

    assert sp.issparse(Xc), "mean-centering must not densify the matrix"
    assert Xc.nnz == X.nnz, "centering must not change sparsity structure"
    return Xc


def compute_item_similarity(uim: mx.UserItemMatrix) -> sp.csr_matrix:
    """Adjusted cosine similarity between items (columns of the centered
    matrix). Returns a sparse n_products x n_products matrix — cosine_similarity
    with dense_output=False keeps the computation sparse throughout, so this
    never materializes a dense N x N array even as an intermediate.

    Items with no ratings from eligible (>=2-rating) users — 21 of them, see
    matrix.py — are all-zero columns. sklearn's cosine_similarity treats a
    zero vector as similarity 0 to everything rather than raising a
    divide-by-zero error, so those items simply come out with no neighbors
    in the top-K step below, which is the correct behavior: there is no CF
    signal for them, and recommend.py must fall back to content-based
    similarity for those products rather than pretending CF has an answer.
    """
    Xc = _mean_center_sparse(uim.matrix)
    sim = cosine_similarity(Xc.T, dense_output=False)
    return sp.csr_matrix(sim)


def top_k_neighbors(
    sim: sp.csr_matrix, product_ids: np.ndarray, k: int = TOP_K
) -> pd.DataFrame:
    """Reduce the full item-item similarity matrix to the K highest-similarity
    neighbors per item (self excluded), in long format:
    [product_id, neighbor_product_id, similarity, rank].

    Only positive similarities are kept. Adjusted cosine ranges [-1, 1]; a
    negative value means the two items were rated in opposite directions
    relative to each user's own average (anti-correlated taste), not
    "similar." Keeping the least-bad negative value just to fill K slots
    would let a product get a "top neighbor" that's actually evidence
    against a recommendation — worse than having no CF signal at all.

    A product with zero stored positive similarities (one of the 21
    zero-rating columns from matrix.py, a genuine isolate, or an item whose
    only overlap with other items was anti-correlated) simply contributes no
    rows. That absence is the signal recommend.py checks for to decide
    whether CF has anything to say about a given product; it must never be
    papered over with a fabricated neighbor list.
    """
    sim = sim.tocsr()
    sim.setdiag(0)
    sim.data[sim.data <= 0] = 0
    sim.eliminate_zeros()

    records: list[tuple[str, str, float, int]] = []
    for i in range(sim.shape[0]):
        start, end = sim.indptr[i], sim.indptr[i + 1]
        cols = sim.indices[start:end]
        vals = sim.data[start:end]
        if len(vals) == 0:
            continue
        top = np.argsort(-vals)[:k]
        for rank, j in enumerate(top, start=1):
            records.append((product_ids[i], product_ids[cols[j]], float(vals[j]), rank))

    df = pd.DataFrame(
        records, columns=["product_id", "neighbor_product_id", "similarity", "rank"]
    )
    n_no_neighbors = sim.shape[0] - df["product_id"].nunique()
    logger.info(
        "Top-%d CF neighbors: %d products have >=1 neighbor, %d have none",
        k, df["product_id"].nunique(), n_no_neighbors,
    )
    return df


def build_cf_neighbors(k: int = TOP_K, force_rebuild: bool = False) -> pd.DataFrame:
    """Load the cached top-K neighbor table, or compute and cache it."""
    if not force_rebuild and NEIGHBORS_PATH.exists():
        return pd.read_parquet(NEIGHBORS_PATH)

    uim = mx.build_user_item_matrix()
    sim = compute_item_similarity(uim)
    neighbors = top_k_neighbors(sim, uim.product_ids, k=k)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    neighbors.to_parquet(NEIGHBORS_PATH, index=False)
    logger.info("Cached CF top-%d neighbors to %s (%d rows)", k, NEIGHBORS_PATH, len(neighbors))
    return neighbors
