"""Leave-one-out evaluation harness: random, popularity, content-only,
CF-only, and hybrid, compared on HitRate@10 / Precision@10 / Recall@10 /
NDCG@10 / catalog coverage / average recommended-item popularity.

Protocol
--------
Eval cohort: ~5,000 users sampled from the >=5-rating cohort (~40,433 users,
see CLAUDE.md's verified data numbers). For each sampled user, the held-out
interaction is their MOST RECENT review by submission_time, not a random
one — predicting the future from the past is the only honest framing for a
recommender that would actually be deployed forward in time, and holding
out a random middle interaction would let a system "predict" something using
information that postdates it.

Leakage prevention (the part most naive implementations get wrong): the
~5,000 held-out (author_id, product_id) pairs are removed from the reviews
data BEFORE anything derived from ratings gets rebuilt for evaluation — the
CF item-item similarity matrix and the skin-profile affinity tables. If
those were built from the full (leaky) data, every CF- and profile-based
metric would be inflated by the model having implicitly seen the exact
interaction being predicted (item-item similarity for the held-out product
would include the very rating being tested; the held-out user's own
skin_type/skin_tone-conditioned averages would too, in a smaller way). This
is a SINGLE rebuild with ~5,000 interactions removed, not one rebuild per
held-out user — the whole point of leave-one-out is that only the specific
(user, item) pairs being predicted need to disappear from training, not each
user's entire history, and there's no reason to redo the (expensive-ish)
matrix/similarity/affinity computation 5,000 times when removing all 5,000
pairs at once and rebuilding once produces the identical result faster.

Content similarity is NOT rebuilt: it's computed purely from product
attributes (name, brand, category, ingredients, highlights), never from
ratings, so there is no rating-derived leakage possible in the first place.

Systems and candidate generation, all evaluated against the same 2,351
CF-scoped products (this is also what "catalog coverage" is a fraction of —
recommending outside that set could never hit a held-out target anyway,
since every review in this dataset is on one of those 2,351 products):

- random: uniform random score per (user, product).
- popularity: same global training-review-count ranking for every user.
- content: item-based weighted aggregation (same formula as
  recommend.py's _aggregate_neighbor_signal) over each user's *training*
  rated products as seeds, using the untouched content neighbor table.
- cf: same aggregation, using the training-rebuilt CF neighbor table.
- hybrid: cf + content + skin-profile (inferred per user from their
  training reviews), blended with blend.py's actual W_CF/W_CONTENT/
  W_PROFILE constants — so Phase 4's weight tuning and this harness always
  agree on what "hybrid" means.

Every system excludes each user's own *training* ratings from its
candidates (never the held-out target itself, which must remain reachable
to be counted as a hit).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import blend  # noqa: E402
import cf  # noqa: E402
import content  # noqa: E402
import data  # noqa: E402
import matrix  # noqa: E402
import skin_profile  # noqa: E402

N_EVAL_USERS = 5000
MIN_RATINGS_FOR_EVAL = 5
TOP_N = 10
RANDOM_SEED = 42

# Cross-join buffer for popularity/random candidate generation: instead of
# scoring the full 2,351-product catalog for every one of 5,000 users
# (11.75M rows for no benefit), only the top BUFFER_SIZE items by the
# relevant score are cross-joined, then already-rated items are excluded
# and the list truncated to TOP_N. BUFFER_SIZE=350 comfortably covers the
# observed max training-ratings-per-user (292, see CLAUDE.md's verified
# numbers) with margin, so this never under-fills top-10 in practice.
BUFFER_SIZE = 350

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.csv"


def select_holdout(
    reviews: pd.DataFrame, n_users: int, min_ratings: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Sample n_users from the >=min_ratings cohort, hold out each one's
    most-recent review by submission_time. Returns (holdout, training_reviews,
    sampled_users) where training_reviews = reviews with exactly those
    n_users rows removed (everyone else's data, and the sampled users' own
    earlier ratings, are untouched)."""
    counts = reviews.groupby("author_id")["product_id"].nunique()
    eligible = counts[counts >= min_ratings].index.to_numpy()

    rng = np.random.default_rng(seed)
    sampled_users = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    sampled_reviews = reviews[reviews["author_id"].isin(sampled_users)]
    # Explicitly re-sort rather than relying on data.py's cache already
    # being submission_time-ordered -- cheap, and doesn't depend on an
    # unstated invariant surviving future changes to data.py.
    sampled_reviews = sampled_reviews.sort_values("submission_time", kind="stable")
    holdout = sampled_reviews.groupby("author_id", observed=True).tail(1)

    holdout_keys = set(zip(holdout["author_id"], holdout["product_id"]))
    mask = ~reviews.apply(
        lambda r: (r["author_id"], r["product_id"]) in holdout_keys, axis=1
    )
    # apply() with a Python-level tuple membership test over 1.09M rows is
    # the one deliberately non-vectorized line here -- an anti-join via
    # merge is faster but this makes the "remove exactly these 5,000 pairs,
    # nothing else" logic unambiguous to read. Runs once per eval, not on
    # any hot path.
    training_reviews = reviews[mask].reset_index(drop=True)

    return holdout.reset_index(drop=True), training_reviews, sampled_users


def _rank_and_truncate(
    candidates: pd.DataFrame, already_rated: pd.DataFrame, score_col: str, n: int
) -> pd.DataFrame:
    """Shared by every system: anti-join against each user's training
    ratings, rank by score_col descending per user, keep the top n. Returns
    [author_id, product_id, rank, score]."""
    merged = candidates.merge(
        already_rated.assign(_rated=True), on=["author_id", "product_id"], how="left"
    )
    remaining = merged[merged["_rated"].isna()].drop(columns="_rated")
    remaining = remaining.sort_values(["author_id", score_col], ascending=[True, False])
    remaining["rank"] = remaining.groupby("author_id").cumcount() + 1
    top = remaining[remaining["rank"] <= n].copy()
    return top[["author_id", "product_id", "rank", score_col]]


def _random_candidates(sampled_users: np.ndarray, catalog: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    buffer = min(BUFFER_SIZE, len(catalog))
    rows = []
    for user in sampled_users:
        items = rng.choice(catalog, size=buffer, replace=False)
        scores = rng.random(buffer)
        rows.append(pd.DataFrame({"author_id": user, "product_id": items, "score": scores}))
    return pd.concat(rows, ignore_index=True)


def _popularity_candidates(
    sampled_users: np.ndarray, catalog: np.ndarray, popularity: pd.Series
) -> pd.DataFrame:
    pop_scores = popularity.reindex(catalog).fillna(0)
    top_items = pop_scores.sort_values(ascending=False).head(BUFFER_SIZE)
    n_users = len(sampled_users)
    n_items = len(top_items)
    return pd.DataFrame(
        {
            "author_id": np.repeat(sampled_users, n_items),
            "product_id": np.tile(top_items.index.to_numpy(), n_users),
            "score": np.tile(top_items.to_numpy(), n_users),
        }
    )


def _aggregate_neighbors_vectorized(
    seeds: pd.DataFrame, neighbor_table: pd.DataFrame
) -> pd.DataFrame:
    """seeds: [author_id, product_id, rating]. neighbor_table: [product_id,
    neighbor_product_id, similarity, rank]. Same weighted item-based
    prediction formula as recommend.py's _aggregate_neighbor_signal
    (score = sum(sim*rating)/sum(sim)), but computed for every sampled user
    at once via merge+groupby instead of one call per user."""
    joined = seeds.merge(neighbor_table, on="product_id")
    joined["weighted"] = joined["similarity"] * joined["rating"]
    grouped = joined.groupby(["author_id", "neighbor_product_id"], observed=True).agg(
        weighted_sum=("weighted", "sum"), sim_sum=("similarity", "sum")
    )
    grouped["score"] = grouped["weighted_sum"] / grouped["sim_sum"]
    grouped = grouped.reset_index().rename(columns={"neighbor_product_id": "product_id"})
    return grouped[["author_id", "product_id", "score"]]


def _infer_user_attrs(training_reviews: pd.DataFrame, sampled_users: np.ndarray) -> pd.DataFrame:
    """Per sampled user, the mode of skin_type/skin_tone across their
    *training* reviews only -- same inference recommend_by_user() does, but
    excluding the held-out review so the profile signal can't leak the
    target's own attribute value."""

    def _mode_or_none(s: pd.Series) -> object:
        m = s.mode()
        return m.iloc[0] if not m.empty else None

    sub = training_reviews[training_reviews["author_id"].isin(sampled_users)]
    skin_type = sub.groupby("author_id", observed=True)["skin_type"].agg(_mode_or_none)
    skin_tone = sub.groupby("author_id", observed=True)["skin_tone"].agg(_mode_or_none)
    attrs = pd.DataFrame({"skin_type": skin_type, "skin_tone": skin_tone}).reset_index()
    return attrs.set_index("author_id").reindex(sampled_users).reset_index(names="author_id")


def _profile_candidates(
    user_attrs: pd.DataFrame, skin_type_table: pd.DataFrame, skin_tone_table: pd.DataFrame
) -> pd.DataFrame:
    """Computes combine_profile_scores() once per DISTINCT (skin_type,
    skin_tone) combo (<=56 possible: 4 types x 14 tones) rather than once
    per user -- thousands of eval users collapse onto a small number of
    attribute combos, so this avoids recomputing the identical
    shrinkage-weighted combination redundantly for every user who happens
    to share the same two attribute values."""
    combos = user_attrs[["skin_type", "skin_tone"]].drop_duplicates().reset_index(drop=True)
    combos["combo_id"] = combos.index

    combo_frames = []
    for _, row in combos.iterrows():
        st = row["skin_type"] if pd.notna(row["skin_type"]) else None
        sto = row["skin_tone"] if pd.notna(row["skin_tone"]) else None
        if st is None and sto is None:
            continue
        scores = skin_profile.combine_profile_scores(
            st,
            sto,
            skin_type_table=skin_type_table if st is not None else None,
            skin_tone_table=skin_tone_table if sto is not None else None,
        )
        scores = scores[["product_id", "score"]].copy()
        scores["combo_id"] = row["combo_id"]
        combo_frames.append(scores)

    if not combo_frames:
        return pd.DataFrame(columns=["author_id", "product_id", "score"])

    all_combo_scores = pd.concat(combo_frames, ignore_index=True)
    user_attrs_tagged = user_attrs.merge(combos, on=["skin_type", "skin_tone"], how="left")
    candidates = user_attrs_tagged[["author_id", "combo_id"]].merge(
        all_combo_scores, on="combo_id"
    )
    return candidates[["author_id", "product_id", "score"]]


def _normalize_per_user(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Same tie/zero-spread handling as blend.normalize(), but grouped by
    author_id: a user's own candidate pool is what a signal's score gets
    scaled against, never the whole 5,000-user eval set. Rows where this
    user has no candidate from this signal at all stay NaN (not 1.0) so
    the caller can fillna(0) to mean "this signal contributed nothing" --
    forcing them to 1.0 would fabricate a maximal score for a signal that
    never fired."""
    g = df.groupby("author_id")[score_col]
    lo = g.transform("min")
    hi = g.transform("max")
    span = hi - lo
    norm = (df[score_col] - lo) / span
    tie_mask = (span.abs() <= 1e-12) & df[score_col].notna()
    return norm.where(~tie_mask, 1.0)


def normalized_signal_frame(
    cf_candidates: pd.DataFrame, content_candidates: pd.DataFrame, profile_candidates: pd.DataFrame
) -> pd.DataFrame:
    """Outer-join the three signals' candidates and per-user normalize each,
    WITHOUT applying any weights. Split out from the weighting step because
    this is the expensive half (a 3-way outer join plus three grouped
    normalizations over ~2M rows) and is completely independent of the
    weights -- so tune_weights.py can compute it once and then sweep dozens
    of weight combinations over the result almost for free.

    Returns [author_id, product_id, cf_norm, content_norm, profile_norm]
    with NaNs already filled to 0, meaning "this signal contributed nothing
    for this candidate"."""
    cf_c = cf_candidates.rename(columns={"score": "cf_score"})
    content_c = content_candidates.rename(columns={"score": "content_score"})
    profile_c = profile_candidates.rename(columns={"score": "profile_score"})

    merged = cf_c.merge(content_c, on=["author_id", "product_id"], how="outer")
    merged = merged.merge(profile_c, on=["author_id", "product_id"], how="outer")

    merged["cf_norm"] = _normalize_per_user(merged, "cf_score").fillna(0)
    merged["content_norm"] = _normalize_per_user(merged, "content_score").fillna(0)
    merged["profile_norm"] = _normalize_per_user(merged, "profile_score").fillna(0)

    return merged[["author_id", "product_id", "cf_norm", "content_norm", "profile_norm"]]


def apply_weights(
    normalized: pd.DataFrame,
    w_cf: float = None,
    w_content: float = None,
    w_profile: float = None,
) -> pd.DataFrame:
    """Weighted sum over an already-normalized signal frame. Weights default
    to blend.py's actual W_CF/W_CONTENT/W_PROFILE constants, so this harness
    and recommend_by_user()/recommend_by_profile() always score "hybrid" the
    same way unless a caller (tune_weights.py) is deliberately probing
    alternatives."""
    w_cf = blend.W_CF if w_cf is None else w_cf
    w_content = blend.W_CONTENT if w_content is None else w_content
    w_profile = blend.W_PROFILE if w_profile is None else w_profile

    out = normalized[["author_id", "product_id"]].copy()
    out["score"] = (
        w_cf * normalized["cf_norm"]
        + w_content * normalized["content_norm"]
        + w_profile * normalized["profile_norm"]
    )
    return out


def _blend_candidates(
    cf_candidates: pd.DataFrame, content_candidates: pd.DataFrame, profile_candidates: pd.DataFrame
) -> pd.DataFrame:
    """Convenience wrapper: normalize then apply blend.py's default weights."""
    return apply_weights(
        normalized_signal_frame(cf_candidates, content_candidates, profile_candidates)
    )


def compute_metrics(
    system: str,
    ranked_candidates: pd.DataFrame,
    holdout: pd.DataFrame,
    sampled_users: np.ndarray,
    catalog_size: int,
    popularity: pd.Series,
) -> dict:
    """ranked_candidates: output of _rank_and_truncate (top-N per user,
    with rank). Denominators use the full sampled cohort (len(sampled_users)),
    not just users a system managed to produce candidates for -- a user a
    system couldn't reach at all is a miss, not an excluded case."""
    n_users = len(sampled_users)

    holdout_lookup = holdout[["author_id", "product_id"]].rename(
        columns={"product_id": "true_product_id"}
    )
    merged = ranked_candidates.merge(holdout_lookup, on="author_id")
    hit_rows = merged[merged["product_id"] == merged["true_product_id"]]

    n_hits = hit_rows["author_id"].nunique()
    hit_rate = n_hits / n_users
    ndcg = float((1.0 / np.log2(hit_rows["rank"] + 1)).sum() / n_users)

    coverage = ranked_candidates["product_id"].nunique() / catalog_size
    avg_popularity = float(
        ranked_candidates["product_id"].map(popularity).fillna(0).mean()
    ) if not ranked_candidates.empty else 0.0

    return {
        "system": system,
        "HitRate@10": hit_rate,
        "Precision@10": hit_rate / TOP_N,
        "Recall@10": hit_rate,  # identical to HitRate@10 by construction: exactly one relevant item per user
        "NDCG@10": ndcg,
        "Coverage": coverage,
        "AvgPopularity": avg_popularity,
        "n_users_with_hit": n_hits,
        "n_users": n_users,
    }


@dataclass
class EvalContext:
    """Everything the leakage-safe evaluation setup produces, so callers that
    need to evaluate repeatedly (tune_weights.py sweeping weight
    combinations) can pay the ~40s setup cost once instead of per run. None
    of these depend on the blend weights."""

    holdout: pd.DataFrame
    training_reviews: pd.DataFrame
    sampled_users: np.ndarray
    already_rated: pd.DataFrame
    popularity: pd.Series
    cf_scope_products: np.ndarray
    cf_candidates: pd.DataFrame
    content_candidates: pd.DataFrame
    profile_candidates: pd.DataFrame
    normalized: pd.DataFrame


def prepare_eval_context(
    n_users: int = N_EVAL_USERS, seed: int = RANDOM_SEED, verbose: bool = True
) -> EvalContext:
    """Run the full leakage-safe setup: sample the eval cohort, hold out each
    user's most recent interaction, rebuild CF similarity and skin-profile
    affinity from training data only, and generate each signal's candidates."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Loading reviews...")
    reviews = data.load_reviews()

    log(f"Selecting {n_users} eval users (>={MIN_RATINGS_FOR_EVAL} ratings, "
        f"holding out most recent review by submission_time)...")
    holdout, training_reviews, sampled_users = select_holdout(
        reviews, n_users, MIN_RATINGS_FOR_EVAL, seed
    )
    log(f"  {len(sampled_users)} users, {len(training_reviews)} training reviews "
        f"({len(reviews) - len(training_reviews)} held out)")

    log("Rebuilding CF matrix + similarity from training data only (leakage prevention)...")
    train_uim = matrix.build_from_reviews(training_reviews)
    train_sim = cf.compute_item_similarity(train_uim)
    train_cf_neighbors = cf.top_k_neighbors(train_sim, train_uim.product_ids, k=cf.TOP_K)

    log("Rebuilding skin-profile affinity from training data only...")
    train_skin_type = skin_profile._shrunk_affinity(
        training_reviews, "skin_type", skin_profile.SHRINKAGE_M
    )
    train_skin_tone = skin_profile._shrunk_affinity(
        training_reviews, "skin_tone", skin_profile.SHRINKAGE_M
    )

    log("Loading content neighbors (no rebuild -- attribute-based, no rating leakage)...")
    content_neighbors = content.build_content_neighbors()

    cf_scope_products = np.array(sorted(data.reviewed_product_ids()))
    popularity = training_reviews.groupby("product_id").size()

    eval_training = training_reviews[training_reviews["author_id"].isin(sampled_users)]
    already_rated = eval_training[["author_id", "product_id"]]
    seeds = eval_training[["author_id", "product_id", "rating"]].copy()
    seeds["rating"] = seeds["rating"].astype(float)

    log("Generating signal candidates...")
    cf_candidates = _aggregate_neighbors_vectorized(seeds, train_cf_neighbors)
    content_candidates = _aggregate_neighbors_vectorized(seeds, content_neighbors)
    user_attrs = _infer_user_attrs(training_reviews, sampled_users)
    profile_candidates = _profile_candidates(user_attrs, train_skin_type, train_skin_tone)
    normalized = normalized_signal_frame(cf_candidates, content_candidates, profile_candidates)

    return EvalContext(
        holdout=holdout,
        training_reviews=training_reviews,
        sampled_users=sampled_users,
        already_rated=already_rated,
        popularity=popularity,
        cf_scope_products=cf_scope_products,
        cf_candidates=cf_candidates,
        content_candidates=content_candidates,
        profile_candidates=profile_candidates,
        normalized=normalized,
    )


def evaluate_weights(
    ctx: EvalContext,
    w_cf: float = None,
    w_content: float = None,
    w_profile: float = None,
    system_name: str = "hybrid",
) -> dict:
    """Score the hybrid system under a specific weight combination, reusing
    ctx's already-computed candidates. Cheap enough to call in a grid-search
    loop."""
    blended = apply_weights(ctx.normalized, w_cf, w_content, w_profile)
    top = _rank_and_truncate(blended, ctx.already_rated, "score", TOP_N)
    return compute_metrics(
        system_name, top, ctx.holdout, ctx.sampled_users,
        len(ctx.cf_scope_products), ctx.popularity,
    )


def run(n_users: int = N_EVAL_USERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    t0 = time.time()
    ctx = prepare_eval_context(n_users=n_users, seed=seed)

    holdout = ctx.holdout
    sampled_users = ctx.sampled_users
    already_rated = ctx.already_rated
    popularity = ctx.popularity
    cf_scope_products = ctx.cf_scope_products

    results = []

    print("Random baseline...")
    random_top = _rank_and_truncate(
        _random_candidates(sampled_users, cf_scope_products, seed), already_rated, "score", TOP_N
    )
    results.append(compute_metrics("random", random_top, holdout, sampled_users, len(cf_scope_products), popularity))

    print("Popularity baseline...")
    pop_top = _rank_and_truncate(
        _popularity_candidates(sampled_users, cf_scope_products, popularity), already_rated, "score", TOP_N
    )
    results.append(compute_metrics("popularity", pop_top, holdout, sampled_users, len(cf_scope_products), popularity))

    print("Content-only...")
    content_top = _rank_and_truncate(ctx.content_candidates, already_rated, "score", TOP_N)
    results.append(compute_metrics("content", content_top, holdout, sampled_users, len(cf_scope_products), popularity))

    print("CF-only...")
    cf_top = _rank_and_truncate(ctx.cf_candidates, already_rated, "score", TOP_N)
    results.append(compute_metrics("cf", cf_top, holdout, sampled_users, len(cf_scope_products), popularity))

    print("Hybrid...")
    results.append(evaluate_weights(ctx, system_name="hybrid"))

    results_df = pd.DataFrame(results)
    print(f"\nDone in {time.time() - t0:.1f}s\n")
    print(results_df.to_string(index=False))

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved to {RESULTS_PATH}")

    return results_df


if __name__ == "__main__":
    run()
