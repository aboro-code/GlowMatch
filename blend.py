"""Weighted score-blend fusion across CF, content, and skin-profile signals.

A score blend, not Reciprocal Rank Fusion (see CLAUDE.md): all three signals
produce comparable normalized scores, so blending scores preserves magnitude
information that rank fusion would discard. Min-max normalization happens
per call, over whatever candidate pool that specific query retrieved — these
signals have very different raw scales (adjusted-cosine CF similarity is
typically 0-0.3, content cosine similarity is typically 0.3-1.0, skin-profile
affinity is a 1-5 rating), so a fixed global rescaling would be wrong; what
matters is each signal's *relative* strength within the candidates actually
being ranked against each other right now.

Weights are named module constants specifically so they're easy to find and
change once the leave-one-out eval harness exists tomorrow — nothing here
should require touching the blending logic itself to retune.
"""

from __future__ import annotations

import pandas as pd

# Retuned 2026-08-26 via eval/tune_weights.py (66-point simplex grid search
# optimizing NDCG@10) AFTER fixing the CF scoring bug described in
# recommend.py's _aggregate_neighbor_signal and README.md's methodology note.
# The previous weights (0.4/0.3/0.3) were fit against a crippled CF signal
# and are not comparable to these.
#
# Collaborative filtering carries this system. With correct top-N scoring,
# CF-only reaches NDCG@10 0.1844 / HitRate@10 0.1992, and the grid's best
# blend (these weights) reaches 0.1873 / 0.2022. That accuracy gap is small
# -- roughly 0.003 in HitRate@10 against a standard error near 0.008 at
# n=5,000 -- and eval/significance.py tests it directly rather than assuming
# it. Do not describe the hybrid as more accurate than CF alone unless that
# test says so.
#
# The blend's defensible benefit is catalog coverage: 0.741 for CF-only
# versus 0.835 here, a large and unambiguous difference. A recommender that
# reaches 74% of the catalog and one that reaches 83% differ in a way that
# matters to a shop, independent of hit rate.
#
# W_PROFILE stays at 0.1 despite contributing essentially nothing to
# warm-user ranking (profile-only HitRate@10 0.0004). It earns its place on
# the cold-start path, where recommend_by_profile has no history to work
# from and profile affinity is the only personalizing signal available, and
# in the UI as displayed evidence ("4.6 stars from 210 reviewers with
# combination skin"). The leave-one-out harness samples users with >=5
# ratings, so it structurally cannot measure the case this signal exists to
# serve -- see README.md's limitations section.
W_CF = 0.8
W_CONTENT = 0.1
W_PROFILE = 0.1

DEFAULT_WEIGHTS = {"cf": W_CF, "content": W_CONTENT, "profile": W_PROFILE}

NO_CF_SUPPORT_LABEL = "Matched on attributes — no rating data yet."


def normalize(scores: pd.Series) -> pd.Series:
    """Min-max scale a signal's scores to [0, 1] within this call's candidate
    pool. A pool with a single candidate, or where every candidate scored
    identically, has no spread to scale by — treated as uniformly maximal
    (1.0) rather than dividing by zero, since there's no basis to prefer one
    over another within that signal."""
    if scores.empty:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=scores.index)
    return (scores - lo) / (hi - lo)


def blend(
    signal_frames: dict[str, pd.DataFrame], weights: dict[str, float] = DEFAULT_WEIGHTS
) -> pd.DataFrame:
    """Combine per-signal candidate frames into one ranked, provenance-carrying
    table.

    signal_frames: {signal_name: DataFrame[product_id, score, evidence]},
    one entry per signal that actually produced candidates for this query.
    Omitting a signal entirely (e.g. recommend_by_profile never passes "cf")
    is how a mode opts out of it — there's no separate on/off flag, the
    signal simply isn't in the dict.

    Returns one row per product_id in the union of all candidate pools, with:
    - blended_score: weighted sum of normalized per-signal scores (missing
      signals contribute 0, they are never imputed from another signal)
    - <name>_score / <name>_score_norm per signal: raw and normalized
    - <name>_evidence per signal: the evidence string(s) for that signal
    - contributing_signals: list of which signals actually had this
      candidate, so the caller can tell "found by CF+content" from
      "content only" without re-deriving it
    - has_cf_support: False when "cf" is either not in signal_frames at all,
      or present but has no score for this particular candidate.
    - has_rating_support: False when neither "cf" nor "profile" contributed
      to this candidate — i.e. the only evidence behind it is content
      (attribute) similarity, never an actual rating. This, not
      has_cf_support alone, is the condition recommend.py uses to attach
      NO_CF_SUPPORT_LABEL: a skin-profile-backed candidate in
      recommend_by_profile has real rating evidence even though it has no
      CF neighbor (recommend_by_profile never passes a "cf" signal at all),
      so labeling it "no rating data yet" would misrepresent it.
    """
    if not signal_frames:
        raise ValueError("blend() requires at least one signal frame")

    per_signal_scores: dict[str, pd.Series] = {}
    per_signal_evidence: dict[str, pd.Series] = {}

    for name, df in signal_frames.items():
        s = df.set_index("product_id")["score"]
        per_signal_scores[name] = s
        if "evidence" in df.columns:
            per_signal_evidence[name] = df.set_index("product_id")["evidence"]

    all_ids = sorted(set().union(*(s.index for s in per_signal_scores.values())))
    result = pd.DataFrame(index=all_ids)

    blended = pd.Series(0.0, index=all_ids)
    contributing = pd.Series([[] for _ in all_ids], index=all_ids)

    for name, s in per_signal_scores.items():
        result[f"{name}_score"] = s.reindex(all_ids)
        present = result[f"{name}_score"].notna()

        norm = normalize(s)
        result[f"{name}_score_norm"] = norm.reindex(all_ids)

        weight = weights.get(name, 0.0)
        blended = blended + result[f"{name}_score_norm"].fillna(0) * weight

        if name in per_signal_evidence:
            result[f"{name}_evidence"] = per_signal_evidence[name].reindex(all_ids)

        for pid in result.index[present]:
            contributing.loc[pid] = contributing.loc[pid] + [name]

    result["blended_score"] = blended
    result["contributing_signals"] = contributing
    result["has_cf_support"] = result.get(
        "cf_score", pd.Series(index=all_ids, dtype=float)
    ).notna()

    rating_backed = pd.Series(False, index=all_ids)
    for rating_signal in ("cf", "profile"):
        if f"{rating_signal}_score" in result.columns:
            rating_backed = rating_backed | result[f"{rating_signal}_score"].notna()
    result["has_rating_support"] = rating_backed

    result = result.reset_index(names="product_id")
    result = result.sort_values("blended_score", ascending=False).reset_index(drop=True)
    return result
