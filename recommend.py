"""Public API: recommend_by_item, recommend_by_user, recommend_by_profile.

Every mode ends up calling blend.blend() with whichever signal frames it
could build, then attaches product names and the "no CF support" label. The
three modes differ only in how they gather candidates:

- recommend_by_item: CF + content neighbors of a single product.
- recommend_by_user: CF + content aggregated across a user's own rated
  products (weighted by their own ratings), plus skin-profile if their
  skin_type/skin_tone can be inferred from their review history.
- recommend_by_profile: skin-profile for the given attributes, plus content
  aggregated across the top profile-scoring products as pseudo-seeds — this
  is what lets a brand-new user's recommendations reach outside the 2,351
  reviewed products into the full catalog, the same way content does
  everywhere else in this pipeline.

All three exclude items the caller has already rated (recommend_by_item
excludes the source item itself; recommend_by_profile has no history to
exclude, so that step is a no-op there).

This module is serve-time code: it reads exclusively through serve.py
(which reads exclusively from artifacts/), never through data.py, cf.py's
or content.py's build_*() functions, or skin_profile.py's build_*_affinity()
/ profile_scores(). That split is deliberate — see serve.py's docstring —
so that this module works identically locally and on a deployed instance
that has no data/ directory at all.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

import blend
import serve
import skin_profile

DEFAULT_N = 10
PROFILE_PSEUDO_SEED_COUNT = 30


@lru_cache(maxsize=1)
def _products() -> pd.DataFrame:
    return serve.load_products()


@lru_cache(maxsize=1)
def _cf_neighbors() -> pd.DataFrame:
    return serve.load_cf_neighbors()


@lru_cache(maxsize=1)
def _content_neighbors() -> pd.DataFrame:
    return serve.load_content_neighbors()


@lru_cache(maxsize=1)
def _reviews_slim() -> pd.DataFrame:
    return serve.load_reviews_slim()


@lru_cache(maxsize=1)
def _skin_type_affinity() -> pd.DataFrame:
    return serve.load_skin_type_affinity()


@lru_cache(maxsize=1)
def _skin_tone_affinity() -> pd.DataFrame:
    return serve.load_skin_tone_affinity()


def _product_name(product_id: str, products: pd.DataFrame) -> str:
    match = products.loc[products["product_id"] == product_id, "product_name"]
    return match.iloc[0] if not match.empty else product_id


def _rating_text(product_id: str, avg_ratings: pd.Series) -> str:
    r = avg_ratings.get(product_id)
    return f"{r:.1f}★" if pd.notna(r) else "no listed rating"


def _neighbors_of(neighbor_table: pd.DataFrame, product_id: str) -> pd.DataFrame:
    return neighbor_table[neighbor_table["product_id"] == product_id]


def _single_item_signal(
    neighbor_table: pd.DataFrame,
    product_id: str,
    source_name: str,
    label: str,
    avg_ratings: pd.Series,
) -> pd.DataFrame:
    """Candidates = the neighbors of one specific product_id, verbatim."""
    rows = _neighbors_of(neighbor_table, product_id)
    if rows.empty:
        return pd.DataFrame(columns=["product_id", "score", "evidence"])

    # Select before renaming: rows still has the *source* product_id column
    # at this point (every row's, since _neighbors_of filtered on it) —
    # renaming neighbor_product_id -> product_id without dropping it first
    # would leave two columns both named "product_id".
    out = rows[["neighbor_product_id", "similarity"]].rename(
        columns={"neighbor_product_id": "product_id", "similarity": "score"}
    )
    if label == "cf":
        out["evidence"] = out.apply(
            lambda r: (
                f'Users who liked "{source_name}" also rated this '
                f'{_rating_text(r["product_id"], avg_ratings)} (CF similarity {r["score"]:.2f})'
            ),
            axis=1,
        )
    else:
        out["evidence"] = out.apply(
            lambda r: f'Similar ingredients/attributes to "{source_name}" (content similarity {r["score"]:.2f})',
            axis=1,
        )
    return out


def _aggregate_neighbor_signal(
    seed_weights: dict[str, float],
    neighbor_table: pd.DataFrame,
    label: str,
    products: pd.DataFrame,
    seed_phrase: str,
) -> pd.DataFrame:
    """Aggregate a neighbor table across multiple seed products into one score
    per candidate:

        score(candidate) = sum over the user's seeds j of similarity(candidate, j)

    i.e. how strongly the candidate is connected, in aggregate, to everything
    the user has already engaged with.

    Why this and NOT the textbook prediction formula sum(sim * r) / sum(sim):
    that formula predicts *what rating the user would give*, the right
    objective for RMSE and the wrong one for top-N ranking. Dividing by
    sum(sim) deliberately cancels out how strongly a candidate connects to
    the user's history — but for ranking, that connection strength is the
    entire signal. It matters enormously here: 82% of ratings are 4 or 5
    stars, so the ratio collapses to roughly 4.5 for nearly every candidate
    and ranking becomes almost arbitrary. Measured on the leave-one-out
    harness, CF-only scored HitRate@10 0.0126 with the normalized formula
    and 0.1992 with this one — a 16x difference caused by the denominator
    alone. See the methodology note in README.md.

    Seed ratings are deliberately NOT used as weights here, and seeds are
    NOT filtered to highly-rated items. Both were measured and both were
    worse: weighting by raw rating 0.1966, centering at 3.0 0.1556,
    centering at the user's own mean 0.0840, restricting to ratings >= 4
    0.1566, versus 0.1992 for the plain sum. The adjusted-cosine similarity
    is already mean-centered per user (see cf.py), so preference direction
    is encoded in the similarity itself; re-applying rating weights on top
    double-counts it and adds noise. seed_weights is still accepted and used
    for the human-readable evidence line, just not for ranking.

    Evidence keeps only the single best-contributing seed per candidate
    (highest similarity) rather than trying to summarize every seed that
    touched it — a legible "similar to X" beats an unreadable list.
    """
    rows = neighbor_table[neighbor_table["product_id"].isin(seed_weights.keys())]
    if rows.empty:
        return pd.DataFrame(columns=["product_id", "score", "evidence"])

    rows = rows.copy()
    rows["seed_weight"] = rows["product_id"].map(seed_weights)

    grouped = rows.groupby("neighbor_product_id").agg(score=("similarity", "sum"))

    # rows still carries the *seed's* product_id column here — rename it to
    # seed_id before joining, so it can't collide with the candidate id that
    # neighbor_product_id becomes below (same duplicate-column trap as
    # _single_item_signal, just one step further downstream).
    best = rows.loc[rows.groupby("neighbor_product_id")["similarity"].idxmax()]
    best = best.set_index("neighbor_product_id")[["product_id", "similarity", "seed_weight"]]
    best = best.rename(columns={"product_id": "seed_id"})

    grouped = grouped.join(best).reset_index(names="product_id")

    def make_evidence(r: pd.Series) -> str:
        seed_name = _product_name(r["seed_id"], products)
        return (
            f'{label}: similar to "{seed_name}", {seed_phrase} {r["seed_weight"]:.1f} '
            f'(similarity {r["similarity"]:.2f})'
        )

    grouped["evidence"] = grouped.apply(make_evidence, axis=1)
    return grouped[["product_id", "score", "evidence"]]


def _attach_display_fields(ranked: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    ranked = ranked.copy()
    name_lookup = products.set_index("product_id")["product_name"]
    ranked["product_name"] = ranked["product_id"].map(name_lookup)
    ranked["label"] = ranked["has_rating_support"].apply(
        lambda has: None if has else blend.NO_CF_SUPPORT_LABEL
    )
    return ranked


def recommend_by_item(product_id: str, n: int = DEFAULT_N) -> pd.DataFrame:
    """"I like this product" mode: item-item CF + content neighbors of
    product_id, blended. product_id can be any of the 8,494 catalog
    products; if it's outside the 2,351 reviewed (skincare) products, CF
    contributes nothing and every result is content-only, flagged
    accordingly.
    """
    products = _products()
    if product_id not in products["product_id"].values:
        raise ValueError(f"Unknown product_id: {product_id!r}")

    source_name = _product_name(product_id, products)
    avg_ratings = products.set_index("product_id")["rating"]

    cf_frame = _single_item_signal(_cf_neighbors(), product_id, source_name, "cf", avg_ratings)
    content_frame = _single_item_signal(
        _content_neighbors(), product_id, source_name, "content", avg_ratings
    )

    signals = {}
    if not cf_frame.empty:
        signals["cf"] = cf_frame
    if not content_frame.empty:
        signals["content"] = content_frame
    if not signals:
        raise RuntimeError(f"No CF or content candidates found for {product_id!r}")

    ranked = blend.blend(signals)
    ranked = ranked[ranked["product_id"] != product_id]
    ranked = _attach_display_fields(ranked, products)
    return ranked.head(n).reset_index(drop=True)


def recommend_by_user(author_id: str, n: int = DEFAULT_N) -> pd.DataFrame:
    """"Existing user" mode: full personalized recs from author_id's actual
    rating history — CF + content aggregated across their rated products
    (weighted by their own ratings), plus skin-profile if their skin_type /
    skin_tone can be read off their own past reviews.

    Cold-start note: author_id must have at least one review (raises
    ValueError otherwise — the caller is expected to pass a real id per the
    "pick a real author_id from the dataset" UI design). A user with exactly
    one rating still gets real CF-backed recommendations despite not being
    one of the >=2-rating users matrix.py's CF matrix was built from: CF
    neighbor lookup happens per *product* (computed from every eligible
    user who rated that product), not per user, so their single rated
    product still works as a CF seed as long as it's one of the 2,351
    reviewed products with at least one positive-similarity neighbor. The
    genuine cold-start gap is narrower: a user whose only rated product(s)
    fall in the 92 CF-scoped products with zero positive neighbors (cf.py)
    gets no CF signal for those seeds — content and profile carry the
    recommendation instead, flagged via has_rating_support same as
    anywhere else. (Every review in this dataset is on one of the 2,351
    reviewed products by construction, so a user's rated products are
    never outside that set.)
    """
    reviews = _reviews_slim()
    user_reviews = reviews[reviews["author_id"] == author_id]
    if user_reviews.empty:
        raise ValueError(f"Unknown author_id: {author_id!r}")

    rated = dict(zip(user_reviews["product_id"], user_reviews["rating"].astype(float)))
    already_rated = set(rated.keys())

    # Every rated product seeds the neighbor lookup, not just highly-rated
    # ones. Filtering to ratings >= 4 seems obviously right and measures
    # worse: HitRate@10 0.1566 liked-only versus 0.1992 using all seeds. The
    # CF similarities are adjusted-cosine (mean-centered per user in cf.py),
    # so a low-rated product still carries usable information about what this
    # user's taste neighborhood looks like, and discarding it just shrinks
    # the evidence base. Keeping eval and production on the same seed rule
    # matters as much as the rule itself.
    seeds = rated

    products = _products()

    cf_frame = _aggregate_neighbor_signal(
        seeds, _cf_neighbors(), "CF", products, seed_phrase="which you rated"
    )
    content_frame = _aggregate_neighbor_signal(
        seeds, _content_neighbors(), "Content", products, seed_phrase="which you rated"
    )

    skin_type = user_reviews["skin_type"].mode()
    skin_type = skin_type.iloc[0] if not skin_type.empty else None
    skin_tone = user_reviews["skin_tone"].mode()
    skin_tone = skin_tone.iloc[0] if not skin_tone.empty else None

    signals = {}
    if not cf_frame.empty:
        signals["cf"] = cf_frame
    if not content_frame.empty:
        signals["content"] = content_frame
    if skin_type is not None or skin_tone is not None:
        profile_frame = skin_profile.combine_profile_scores(
            skin_type,
            skin_tone,
            skin_type_table=_skin_type_affinity() if skin_type is not None else None,
            skin_tone_table=_skin_tone_affinity() if skin_tone is not None else None,
        )
        if not profile_frame.empty:
            signals["profile"] = profile_frame
    if not signals:
        raise RuntimeError(f"No candidate signals available for author_id {author_id!r}")

    ranked = blend.blend(signals)
    ranked = ranked[~ranked["product_id"].isin(already_rated)]
    ranked = _attach_display_fields(ranked, products)
    return ranked.head(n).reset_index(drop=True)


def recommend_by_profile(
    skin_type: str | None = None, skin_tone: str | None = None, n: int = DEFAULT_N
) -> pd.DataFrame:
    """"I'm a new user" mode: skin-profile affinity for the given attributes,
    plus content aggregated across the top profile-scoring products as
    pseudo-seeds (weighted by their profile score). The content expansion is
    what lets this mode reach outside the 2,351 reviewed products —
    profile_scores() alone can only ever return CF-scoped products, since
    skin-profile affinity is computed from review data.

    No rating history exists for a brand-new user, so the
    "exclude already-rated" step is a structural no-op here rather than
    omitted.
    """
    if skin_type is None and skin_tone is None:
        raise ValueError("recommend_by_profile requires at least one of skin_type or skin_tone")

    profile_frame = skin_profile.combine_profile_scores(
        skin_type,
        skin_tone,
        skin_type_table=_skin_type_affinity() if skin_type is not None else None,
        skin_tone_table=_skin_tone_affinity() if skin_tone is not None else None,
    )
    products = _products()

    top_seeds = profile_frame.nlargest(PROFILE_PSEUDO_SEED_COUNT, "score")
    seed_weights = dict(zip(top_seeds["product_id"], top_seeds["score"]))
    content_frame = _aggregate_neighbor_signal(
        seed_weights,
        _content_neighbors(),
        "Content",
        products,
        seed_phrase="a strong match for your skin profile at",
    )

    signals = {"profile": profile_frame}
    if not content_frame.empty:
        signals["content"] = content_frame

    ranked = blend.blend(signals)
    ranked = _attach_display_fields(ranked, products)
    return ranked.head(n).reset_index(drop=True)
