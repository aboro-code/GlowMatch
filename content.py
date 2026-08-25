"""Content-based similarity over the full 8,494-product catalog.

This is the only signal that exists for the 6,143 products with no reviews
at all (see data.load_reviewed_products() — the review corpus is
skincare-only). It also backstops the 92 CF-scoped products cf.py found no
positive-similarity neighbors for.

Embeds product text (name + brand + category + highlights + ingredients)
with sentence-transformers and persists top-K nearest neighbors, same
[product_id, neighbor_product_id, similarity, rank] schema as cf.py so
blend.py can treat both signals uniformly.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import data

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "content_embeddings.npy"
EMBEDDINGS_PRODUCTS_PATH = PROCESSED_DIR / "content_embeddings_products.npy"
NEIGHBORS_PATH = PROCESSED_DIR / "content_top_k_neighbors.parquet"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 50
BATCH_SIZE = 512

# product_info.csv's ingredients field is a stringified Python list and can
# run past 16,000 characters for multi-variant products — far beyond what
# the embedding model actually reads (all-MiniLM-L6-v2 truncates around 256
# tokens). Capping here rather than relying on model truncation guarantees
# name/brand/category/highlights survive in the input regardless of how long
# the ingredients list is, since they're placed first. The cap doesn't lose
# the most useful ingredient information anyway: INCI convention lists
# ingredients in descending concentration order, so the first ~300 characters
# are the dominant actives, not an arbitrary prefix.
INGREDIENTS_CHAR_CAP = 300


def _parse_list_field(value: object) -> list[str]:
    """product_info.csv stores highlights/ingredients as strings that look
    like Python list literals, e.g. "['Vegan', 'Cruelty-Free']". Parse them
    back into real lists; fall back to the raw string for anything that
    doesn't parse (and to [] for nulls) rather than dropping the field."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return [value]


def build_product_text(row: pd.Series) -> str:
    """One text blob per product for embedding. Order matters: identifying
    fields (name, brand, category, highlights) come before ingredients so
    they're never pushed out of the model's truncation window."""
    parts = [str(row["product_name"]), f"By {row['brand_name']}."]

    categories = [
        c for c in (row.get("primary_category"), row.get("secondary_category"), row.get("tertiary_category"))
        if isinstance(c, str) and c.strip()
    ]
    if categories:
        parts.append("Category: " + " > ".join(categories) + ".")

    highlights = _parse_list_field(row.get("highlights"))
    if highlights:
        parts.append("Highlights: " + ", ".join(highlights) + ".")

    ingredients = _parse_list_field(row.get("ingredients"))
    if ingredients:
        ingredients_text = ", ".join(ingredients)[:INGREDIENTS_CHAR_CAP]
        parts.append("Key ingredients: " + ingredients_text + ".")

    return " ".join(parts)


def build_corpus(products: pd.DataFrame) -> list[str]:
    return [build_product_text(row) for _, row in products.iterrows()]


def embed_products(force_rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalized embeddings for the full catalog, cached to
    data/processed/content_embeddings.npy. Returns (embeddings, product_ids),
    row-aligned. Normalizing at encode time means later cosine similarity is
    just a dot product."""
    if not force_rebuild and EMBEDDINGS_PATH.exists() and EMBEDDINGS_PRODUCTS_PATH.exists():
        return np.load(EMBEDDINGS_PATH), np.load(EMBEDDINGS_PRODUCTS_PATH, allow_pickle=False)

    from sentence_transformers import SentenceTransformer

    products = data.load_products()
    texts = build_corpus(products)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
    product_ids = products["product_id"].to_numpy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(EMBEDDINGS_PRODUCTS_PATH, product_ids)
    logger.info("Cached content embeddings: %s (%s)", EMBEDDINGS_PATH, embeddings.shape)
    return embeddings, product_ids


def _top_k_from_dense(
    embeddings: np.ndarray, product_ids: np.ndarray, k: int, batch_size: int
) -> pd.DataFrame:
    """Batched top-K instead of one dense N x N similarity matrix. At 8,494
    products a full dense matrix (~288MB float32) would technically fit in
    memory, but computing it in row batches and immediately reducing each
    batch to its top-K keeps peak memory at batch_size x n_products
    (~17MB at the default batch size) and matches the O(N*K) persistence
    philosophy used everywhere else in this pipeline, independent of how
    large the catalog gets.

    Only positive similarities are kept, for the same reason as cf.py: a
    non-positive cosine similarity is not evidence of relatedness.
    """
    n = embeddings.shape[0]
    records: list[tuple[str, str, float, int]] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = embeddings[start:end] @ embeddings.T
        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i]
            row[global_i] = -np.inf
            top = np.argpartition(-row, min(k, n - 1))[:k]
            top = top[np.argsort(-row[top])]
            for rank, j in enumerate(top, start=1):
                sim_val = float(row[j])
                if sim_val <= 0:
                    break
                records.append((product_ids[global_i], product_ids[j], sim_val, rank))

    df = pd.DataFrame(
        records, columns=["product_id", "neighbor_product_id", "similarity", "rank"]
    )
    n_no_neighbors = n - df["product_id"].nunique()
    logger.info(
        "Top-%d content neighbors: %d products have >=1 neighbor, %d have none",
        k, df["product_id"].nunique(), n_no_neighbors,
    )
    return df


def build_content_neighbors(k: int = TOP_K, force_rebuild: bool = False) -> pd.DataFrame:
    """Load the cached top-K content-similarity neighbor table, or compute
    and cache it."""
    if not force_rebuild and NEIGHBORS_PATH.exists():
        return pd.read_parquet(NEIGHBORS_PATH)

    embeddings, product_ids = embed_products(force_rebuild=force_rebuild)
    neighbors = _top_k_from_dense(embeddings, product_ids, k=k, batch_size=BATCH_SIZE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    neighbors.to_parquet(NEIGHBORS_PATH, index=False)
    logger.info("Cached content top-%d neighbors to %s (%d rows)", k, NEIGHBORS_PATH, len(neighbors))
    return neighbors
