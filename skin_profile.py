"""Skin-profile affinity: how well a product is rated by reviewers who share
a given skin_type or skin_tone, shrunk toward the product's own mean rating.

Why shrinkage: a product with 3 reviews from combination-skin reviewers
averaging 5.0 is not evidence that combination-skin users love it — it's
noise from a tiny sample. Shrinking each (product, attribute) mean toward
that product's overall mean, weighted by how much evidence backs it, means a
5.0 from 3 reviewers lands close to the product's real average, while a 4.6
from 210 reviewers barely moves (barely needs correcting, because it already
has enough evidence to trust on its own). This is a standard Bayesian-average
/ James-Stein-style shrinkage, not shrinkage toward the dataset-wide mean —
product identity matters more than population average here (a $12 cleanser
and a $200 serum have different baseline ratings for reasons unrelated to
skin type).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import data

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
SKIN_TYPE_PATH = PROCESSED_DIR / "profile_skin_type.parquet"
SKIN_TONE_PATH = PROCESSED_DIR / "profile_skin_tone.parquet"

# How many reviews' worth of weight the product's own global mean gets in the
# shrinkage formula. m=20 means an attribute-specific mean needs roughly 20
# reviews before it starts to dominate the product's overall mean; below
# that it's pulled toward the product baseline in proportion to how little
# evidence it has. Not tuned against the eval harness (that happens later);
# chosen as a reasonable default given products average 465 reviews total.
SHRINKAGE_M = 20


def _global_product_means(reviews: pd.DataFrame) -> pd.Series:
    return reviews.groupby("product_id")["rating"].mean()


def _shrunk_affinity(reviews: pd.DataFrame, attribute: str, m: int) -> pd.DataFrame:
    global_means = _global_product_means(reviews)

    grouped = (
        reviews.dropna(subset=[attribute])
        .groupby(["product_id", attribute], observed=True)["rating"]
        .agg(raw_mean="mean", n="count")
        .reset_index()
    )
    grouped["global_mean"] = grouped["product_id"].map(global_means)
    grouped["score"] = (
        grouped["n"] * grouped["raw_mean"] + m * grouped["global_mean"]
    ) / (grouped["n"] + m)
    return grouped.drop(columns=["global_mean"])


def build_skin_type_affinity(force_rebuild: bool = False) -> pd.DataFrame:
    """[product_id, skin_type, raw_mean, n, score]. Cached to
    data/processed/profile_skin_type.parquet."""
    if not force_rebuild and SKIN_TYPE_PATH.exists():
        return pd.read_parquet(SKIN_TYPE_PATH)

    reviews = data.load_reviews()
    df = _shrunk_affinity(reviews, "skin_type", SHRINKAGE_M)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SKIN_TYPE_PATH, index=False)
    logger.info("Cached skin_type affinity to %s (%d rows)", SKIN_TYPE_PATH, len(df))
    return df


def build_skin_tone_affinity(force_rebuild: bool = False) -> pd.DataFrame:
    """[product_id, skin_tone, raw_mean, n, score]. Cached to
    data/processed/profile_skin_tone.parquet."""
    if not force_rebuild and SKIN_TONE_PATH.exists():
        return pd.read_parquet(SKIN_TONE_PATH)

    reviews = data.load_reviews()
    df = _shrunk_affinity(reviews, "skin_tone", SHRINKAGE_M)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SKIN_TONE_PATH, index=False)
    logger.info("Cached skin_tone affinity to %s (%d rows)", SKIN_TONE_PATH, len(df))
    return df


def _evidence_text(attribute_label: str, value: str, score: float, n: int) -> str:
    return f"{score:.1f}★ from {n} reviewer{'s' if n != 1 else ''} with {value} {attribute_label}"


def combine_profile_scores(
    skin_type: str | None,
    skin_tone: str | None,
    skin_type_table: pd.DataFrame | None = None,
    skin_tone_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combined skin-profile affinity per product for a given skin_type
    and/or skin_tone value, given the two affinity tables directly rather
    than building or loading them internally. This is the pure logic half
    of profile_scores() — split out so serve.py-backed callers (recommend.py
    at serve time) can pass in artifacts-loaded tables without this module
    reaching into data.py's build-time cache path. At least one of
    skin_type/skin_tone must be given (this backs the "new user" cold-start
    mode, which needs at least one attribute to condition on), and its
    matching *_table must be provided.

    When both are given, the two shrunk scores are combined as a weighted
    average by their evidence counts (n) rather than a plain average — a
    product with 300 combination-skin reviews and 5 fair-tone reviews should
    lean on the combination-skin evidence, not treat both as equally
    trustworthy.

    A product with zero reviews from reviewers matching either attribute
    value simply doesn't appear in the result — that absence means "no
    skin-profile evidence for this product," which recommend.py must not
    paper over by substituting the product's unconditioned average rating
    as if it were attribute-specific evidence.
    """
    if skin_type is None and skin_tone is None:
        raise ValueError("combine_profile_scores requires at least one of skin_type or skin_tone")
    if skin_type is not None and skin_type_table is None:
        raise ValueError("skin_type given but skin_type_table not provided")
    if skin_tone is not None and skin_tone_table is None:
        raise ValueError("skin_tone given but skin_tone_table not provided")

    parts: list[pd.DataFrame] = []

    if skin_type is not None:
        t = skin_type_table[skin_type_table["skin_type"] == skin_type][["product_id", "score", "n"]].copy()
        t["evidence"] = t.apply(
            lambda r: _evidence_text("skin type", skin_type, r["score"], int(r["n"])), axis=1
        )
        t = t.rename(columns={"score": "skin_type_score", "n": "skin_type_n"})
        parts.append(t.set_index("product_id"))

    if skin_tone is not None:
        t = skin_tone_table[skin_tone_table["skin_tone"] == skin_tone][["product_id", "score", "n"]].copy()
        t["evidence_tone"] = t.apply(
            lambda r: _evidence_text("skin tone", skin_tone, r["score"], int(r["n"])), axis=1
        )
        t = t.rename(columns={"score": "skin_tone_score", "n": "skin_tone_n"})
        parts.append(t.set_index("product_id"))

    combined = parts[0] if len(parts) == 1 else parts[0].join(parts[1], how="outer")

    score_n_pairs = [
        ("skin_type_score", "skin_type_n"),
        ("skin_tone_score", "skin_tone_n"),
    ]
    present = [(s, n) for s, n in score_n_pairs if s in combined.columns]

    numerator = sum(combined[s].fillna(0) * combined[n].fillna(0) for s, n in present)
    denominator = sum(combined[n].fillna(0) for s, n in present)
    combined["score"] = numerator / denominator
    combined["n"] = denominator.astype(int)

    evidence_cols = [c for c in combined.columns if str(c).startswith("evidence")]
    combined["evidence"] = combined[evidence_cols].apply(
        lambda r: [v for v in r if isinstance(v, str)], axis=1
    )

    return combined.reset_index()[["product_id", "score", "n", "evidence"]]


def profile_scores(skin_type: str | None = None, skin_tone: str | None = None) -> pd.DataFrame:
    """Offline/dev convenience wrapper: builds (or loads the data/processed/
    cache of) the affinity tables via build_skin_type_affinity() /
    build_skin_tone_affinity(), then delegates to combine_profile_scores().
    Serve-time code (recommend.py in production) should call
    combine_profile_scores() directly with tables loaded from serve.py
    instead — this wrapper exists for local development, scripts, and the
    evaluation harness, which do have access to data.py's build-time path.
    """
    skin_type_table = build_skin_type_affinity() if skin_type is not None else None
    skin_tone_table = build_skin_tone_affinity() if skin_tone is not None else None
    return combine_profile_scores(skin_type, skin_tone, skin_type_table, skin_tone_table)
