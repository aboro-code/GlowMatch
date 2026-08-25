"""Serve-time data access: reads exclusively from the committed artifacts/
directory.

This is deliberately a separate module from data.py, not a fallback path
bolted onto it. Build-time (data.py, matrix.py, and the build_*() functions
in cf.py/content.py/skin_profile.py) needs the raw CSVs and does expensive
parsing/embedding/similarity work; serve-time needs only the ~18MB of
already-computed artifacts and must never touch data/ at all. Giving serve
its own loader — rather than adding an "if artifacts/ exists, read from
there, else fall back to data/processed/" branch inside data.py — means
local development and a deployed instance can't silently diverge in
behavior depending on whether a build cache happens to be sitting around.
A deployed instance has no data/ directory; if artifacts/ is missing or
incomplete, this fails loudly (FileNotFoundError) rather than quietly
attempting a rebuild that has no raw data to rebuild from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _read(name: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/build_artifacts.py to generate "
            f"artifacts/ before serving recommendations — serve.py never "
            f"builds from raw data itself."
        )
    return pd.read_parquet(path)


def load_products() -> pd.DataFrame:
    """Full product catalog (8,494 products, all categories)."""
    return _read("products.parquet")


def load_reviews_slim() -> pd.DataFrame:
    """author_id, product_id, rating, skin_type, skin_tone only — the 5
    columns recommend_by_user() needs for a user's rating history and
    skin-attribute inference. Deliberately not the full reviews table
    (which carries free-text review_text/review_title and is ~206MB);
    see scripts/build_artifacts.py for why only this slice ships."""
    return _read("reviews_slim.parquet")


def load_cf_neighbors() -> pd.DataFrame:
    return _read("cf_top_k_neighbors.parquet")


def load_content_neighbors() -> pd.DataFrame:
    return _read("content_top_k_neighbors.parquet")


def load_skin_type_affinity() -> pd.DataFrame:
    return _read("profile_skin_type.parquet")


def load_skin_tone_affinity() -> pd.DataFrame:
    return _read("profile_skin_tone.parquet")
