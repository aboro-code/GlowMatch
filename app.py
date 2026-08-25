"""GlowMatch Streamlit UI — presentation only.

All recommendation logic lives in recommend.py / blend.py / skin_profile.py;
this module formats their output. It reads data exclusively through serve.py
(artifacts/), never data.py, so it behaves identically locally and deployed.

Design priority is clarity and honesty over polish: every recommendation
shows which signals produced it, their individual normalized contributions,
and human-readable evidence — and anything with no rating-based support is
explicitly labeled rather than presented as if people had endorsed it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import blend
import ingredients as ing
import recommend
import serve

st.set_page_config(page_title="GlowMatch", page_icon="✨", layout="wide")

TOP_N = 10
# Most options rendered in the product picker at once. The catalog is 8,494
# products; putting all of them in a single selectbox makes the browser
# sluggish and is useless to scroll, so search narrows first and this caps
# what's displayed. The true match count is always shown alongside.
PRODUCT_PICKER_LIMIT = 200
SIGNAL_LABELS = {"cf": "Collaborative", "content": "Content", "profile": "Skin profile"}
# skin_tone carries a literal "notSureST" value for reviewers who declined to
# specify. It's real data and stays in the affinity tables, but offering it as
# a selectable profile would be meaningless -- you can't recommend for "not sure".
EXCLUDED_TONE_VALUES = {"notSureST"}


@st.cache_data
def load_products() -> pd.DataFrame:
    return serve.load_products()


@st.cache_data
def load_reviews_slim() -> pd.DataFrame:
    return serve.load_reviews_slim()


@st.cache_data
def load_actives_lookup() -> dict[str, set[str]]:
    return ing.build_actives_lookup(serve.load_products())


@st.cache_data
def load_eval_results() -> pd.DataFrame | None:
    path = __import__("pathlib").Path(__file__).resolve().parent / "eval" / "eval_results.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def product_options() -> pd.DataFrame:
    """Products offered in the 'more like this' picker, most-reviewed first so
    the default selection is something with real collaborative signal behind
    it rather than an arbitrary catalog entry."""
    products = load_products()
    reviews = load_reviews_slim()
    counts = reviews.groupby("product_id").size().rename("n_reviews")
    opts = products[["product_id", "product_name", "brand_name", "primary_category"]].merge(
        counts, left_on="product_id", right_index=True, how="left"
    )
    opts["n_reviews"] = opts["n_reviews"].fillna(0).astype(int)
    opts["display"] = opts["product_name"] + " — " + opts["brand_name"]
    return opts.sort_values("n_reviews", ascending=False).reset_index(drop=True)


@st.cache_data
def sample_reviewers(n: int = 300) -> pd.DataFrame:
    """Reviewers offered in the 'existing user' picker: the most active ones,
    so the personalization demo has actual history to work from."""
    reviews = load_reviews_slim()
    counts = reviews.groupby("author_id")["product_id"].nunique().rename("n_rated")
    top = counts.sort_values(ascending=False).head(n).reset_index()
    attrs = (
        reviews[reviews["author_id"].isin(top["author_id"])]
        .groupby("author_id")
        .agg(
            skin_type=("skin_type", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            skin_tone=("skin_tone", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
        )
        .reset_index()
    )
    return top.merge(attrs, on="author_id", how="left")


def filter_products(opts: pd.DataFrame, query: str, category: str) -> pd.DataFrame:
    """Filter the product picker by free-text query and category.

    Every whitespace-separated term must match somewhere in the product name
    or brand (AND, not OR) — "cerave cleanser" should mean both words, which
    is what someone typing two words expects, whereas OR would flood the
    results with every cleanser in the catalog. Matching is case-insensitive
    substring rather than exact-token so partial words work while typing
    ("niacin" finds niacinamide products), and regex metacharacters in the
    query are treated literally so a stray "(" can't raise.
    """
    result = opts
    if category and category != "All categories":
        result = result[result["primary_category"] == category]

    terms = query.lower().split()
    if terms:
        haystack = (result["product_name"] + " " + result["brand_name"]).str.lower()
        for term in terms:
            result = result[haystack.str.contains(term, regex=False, na=False)]
            haystack = haystack.loc[result.index]

    return result


def attribute_values(column: str) -> list[str]:
    reviews = load_reviews_slim()
    vals = sorted(v for v in reviews[column].dropna().unique() if v not in EXCLUDED_TONE_VALUES)
    return vals


def render_recommendations(
    results: pd.DataFrame, seed_product_ids: list[str] | None = None
) -> None:
    """One expandable card per recommendation, showing signal contributions
    and evidence. seed_product_ids, when given, enables shared-ingredient
    evidence against whatever the recommendation was derived from."""
    if results.empty:
        st.warning("No recommendations could be generated for this input.")
        return

    products = load_products().set_index("product_id")
    actives = load_actives_lookup()

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        pid = row["product_id"]
        meta = products.loc[pid] if pid in products.index else None
        brand = meta["brand_name"] if meta is not None else "—"
        category = meta["primary_category"] if meta is not None else "—"
        price = meta["price_usd"] if meta is not None else None
        avg_rating = meta["rating"] if meta is not None else None

        header = f"**{rank}. {row['product_name']}** — {brand}"
        with st.container(border=True):
            left, right = st.columns([3, 1])

            with left:
                st.markdown(header)
                bits = [f"`{category}`"]
                if pd.notna(price):
                    bits.append(f"${price:.2f}")
                if pd.notna(avg_rating):
                    bits.append(f"{avg_rating:.1f}★ overall")
                st.caption(" · ".join(bits))

                if pd.notna(row.get("label")):
                    st.warning(row["label"], icon="⚠️")

                signals = row.get("contributing_signals") or []
                st.markdown(
                    "**Why:** "
                    + ", ".join(SIGNAL_LABELS.get(s, s) for s in signals)
                    if signals
                    else "**Why:** —"
                )

                for signal in ("cf", "content", "profile"):
                    evidence = row.get(f"{signal}_evidence")
                    if isinstance(evidence, list):
                        for line in evidence:
                            st.markdown(f"- {line}")
                    elif isinstance(evidence, str) and evidence:
                        st.markdown(f"- {evidence}")

                if seed_product_ids:
                    shared: set[str] = set()
                    for seed in seed_product_ids:
                        shared |= set(ing.shared_actives(seed, pid, actives))
                    text = ing.shared_actives_text(sorted(shared))
                    if text:
                        st.markdown(f"- {text}")

            with right:
                st.metric("Blended score", f"{row['blended_score']:.3f}")
                for signal in ("cf", "content", "profile"):
                    norm = row.get(f"{signal}_score_norm")
                    if pd.notna(norm):
                        st.caption(f"{SIGNAL_LABELS[signal]}: {norm:.3f}")


def mode_by_item() -> None:
    st.subheader("Find products like one I already like")
    st.caption(
        "Item-item collaborative filtering plus content similarity. Products outside "
        "the 2,351 reviewed skincare items have no collaborative signal at all — those "
        "recommendations are labeled."
    )

    opts = product_options()

    search_col, cat_col = st.columns([2, 1])
    with search_col:
        query = st.text_input(
            "Search by product or brand",
            placeholder="e.g. niacinamide, CeraVe, sunscreen",
            key="item_search",
        )
    with cat_col:
        categories = ["All categories"] + sorted(opts["primary_category"].dropna().unique())
        category = st.selectbox("Category", options=categories, key="item_category")

    filtered = filter_products(opts, query, category)

    if filtered.empty:
        st.info(
            f"No products match {query!r}"
            + (f" in {category}." if category != "All categories" else ".")
            + " Try a shorter or different term."
        )
        return

    # Cap the dropdown: rendering thousands of options is slow in the browser
    # and unhelpful to scroll. The cap applies to what's *displayed*, and the
    # count below always reports the true number of matches so a truncated
    # list never silently looks like the complete one.
    shown = filtered.head(PRODUCT_PICKER_LIMIT)
    if len(filtered) > len(shown):
        st.caption(
            f"{len(filtered):,} matches — showing the {len(shown)} most-reviewed. "
            "Refine your search to narrow it down."
        )
    else:
        st.caption(f"{len(filtered):,} match{'es' if len(filtered) != 1 else ''}.")

    choice = st.selectbox(
        "Pick a product",
        options=shown.index,
        format_func=lambda i: f"{shown.loc[i, 'display']}  ({shown.loc[i, 'n_reviews']} reviews)",
        key="item_pick",
    )
    product_id = shown.loc[choice, "product_id"]

    if shown.loc[choice, "n_reviews"] == 0:
        st.info(
            "This product has no reviews, so collaborative filtering has nothing to work "
            "from — recommendations will come from content similarity alone and be labeled "
            "as such.",
            icon="ℹ️",
        )

    if st.button("Get recommendations", type="primary", key="item_go"):
        with st.spinner("Ranking..."):
            results = recommend.recommend_by_item(product_id, n=TOP_N)
        render_recommendations(results, seed_product_ids=[product_id])


def mode_by_profile() -> None:
    st.subheader("I'm new here — recommend for my skin")
    st.caption(
        "Cold start: no rating history required. Ranks by how reviewers who share your "
        "skin type and tone actually rated each product (shrunk toward the product's own "
        "mean so a 5.0 from three people doesn't outrank a 4.6 from two hundred), then "
        "expands through content similarity to reach the full catalog."
    )

    # Default to a real skin type rather than "(any)" on both selectors: with
    # both left at "(any)" there's nothing to condition on and the first click
    # can only produce a validation error, which is a poor first impression of
    # a mode whose whole point is working without any user history.
    type_options = attribute_values("skin_type")
    tone_options = attribute_values("skin_tone")

    col1, col2 = st.columns(2)
    with col1:
        skin_type = st.selectbox(
            "Skin type",
            options=["(any)"] + type_options,
            index=1 if type_options else 0,
        )
    with col2:
        skin_tone = st.selectbox("Skin tone", options=["(any)"] + tone_options)

    st_val = None if skin_type == "(any)" else skin_type
    sto_val = None if skin_tone == "(any)" else skin_tone

    if st.button("Get recommendations", type="primary", key="profile_go"):
        if st_val is None and sto_val is None:
            st.error("Pick at least one of skin type or skin tone.")
            return
        with st.spinner("Ranking..."):
            results = recommend.recommend_by_profile(skin_type=st_val, skin_tone=sto_val, n=TOP_N)
        render_recommendations(results)


def mode_by_user() -> None:
    st.subheader("Personalized from a real reviewer's history")
    st.caption(
        "Full personalization for an actual reviewer in the dataset: their rated products "
        "seed both collaborative and content signals, weighted by how they rated each one, "
        "with skin-profile affinity inferred from their own reviews. Products they've "
        "already rated are excluded."
    )

    reviewers = sample_reviewers()
    choice = st.selectbox(
        "Pick a reviewer",
        options=reviewers.index,
        format_func=lambda i: (
            f"{reviewers.loc[i, 'author_id']} — {reviewers.loc[i, 'n_rated']} products rated"
            f" · {reviewers.loc[i, 'skin_type'] or 'unknown'} skin"
            f" · {reviewers.loc[i, 'skin_tone'] or 'unknown'} tone"
        ),
    )
    author_id = reviewers.loc[choice, "author_id"]

    reviews = load_reviews_slim()
    history = reviews[reviews["author_id"] == author_id]
    products = load_products().set_index("product_id")

    with st.expander(f"Their rating history ({len(history)} reviews)"):
        hist = history.copy()
        hist["product_name"] = hist["product_id"].map(products["product_name"])
        st.dataframe(
            hist[["product_name", "rating", "skin_type", "skin_tone"]].sort_values(
                "rating", ascending=False
            ),
            hide_index=True,
            width="stretch",
        )

    if st.button("Get recommendations", type="primary", key="user_go"):
        with st.spinner("Ranking..."):
            results = recommend.recommend_by_user(author_id, n=TOP_N)
        liked = history[history["rating"] >= 4]["product_id"].tolist()
        render_recommendations(results, seed_product_ids=liked or history["product_id"].tolist())


def tab_evaluation() -> None:
    st.subheader("How well does this actually work?")
    results = load_eval_results()
    if results is None:
        st.info("No evaluation results found. Run `python eval/run_eval.py` to generate them.")
        return

    st.caption(
        "Leave-one-out evaluation over 5,000 users sampled from the ≥5-rating cohort. Each "
        "user's most recent review (by submission time) is held out, and those held-out "
        "interactions are removed from the data before the collaborative-filtering "
        "similarity and skin-profile tables are rebuilt — otherwise every CF number would "
        "be inflated by the model having already seen the exact interaction it's being "
        "asked to predict."
    )

    display = results.copy()
    display = display.rename(columns={"AvgPopularity": "Avg popularity of recs"})
    st.dataframe(
        display.style.format(
            {
                "HitRate@10": "{:.4f}",
                "Precision@10": "{:.5f}",
                "Recall@10": "{:.4f}",
                "NDCG@10": "{:.4f}",
                "Coverage": "{:.3f}",
                "Avg popularity of recs": "{:.0f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    hybrid = results[results["system"] == "hybrid"]
    popularity = results[results["system"] == "popularity"]
    if not hybrid.empty and not popularity.empty:
        h, p = hybrid.iloc[0], popularity.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Hybrid HitRate@10", f"{h['HitRate@10']:.4f}",
                  f"{(h['HitRate@10'] / p['HitRate@10'] - 1) * 100:+.0f}% vs popularity")
        c2.metric("Catalog coverage", f"{h['Coverage']:.1%}",
                  f"vs {p['Coverage']:.1%} for popularity")
        c3.metric("Avg popularity of recs", f"{h['AvgPopularity']:.0f}",
                  f"vs {p['AvgPopularity']:.0f} for popularity", delta_color="inverse")

    st.markdown(
        f"""
**Reading these numbers honestly:**

- **The hybrid beats every baseline, but neither component does it alone.** CF-only
  (HitRate@10 {results.set_index('system').loc['cf', 'HitRate@10']:.4f}) and content-only
  ({results.set_index('system').loc['content', 'HitRate@10']:.4f}) each score *below* the
  popularity baseline ({results.set_index('system').loc['popularity', 'HitRate@10']:.4f}).
  Combined, they reach {results.set_index('system').loc['hybrid', 'HitRate@10']:.4f}. The win
  comes from combining independently-noisy signals, not from either being strong.
- **Coverage is where the hybrid clearly wins.** Popularity recommends the same handful of
  products to everyone — {results.set_index('system').loc['popularity', 'Coverage']:.1%} of
  the catalog — while the hybrid reaches
  {results.set_index('system').loc['hybrid', 'Coverage']:.1%}, at a fraction of the average
  item popularity. A recommender that only ever surfaces bestsellers isn't doing much
  recommending.
- **Absolute hit rates are low, and that's expected.** Predicting one specific held-out
  product out of 2,351 is a hard target; ~10% at rank 10 is a real result, not a broken one.
- **Blend weights are barely worth tuning.** A 66-point grid search moved NDCG@10 by +0.7%,
  and variation between evaluation samples turned out ~8× larger than variation between
  weight combinations. Current weights: CF {blend.W_CF}, content {blend.W_CONTENT},
  profile {blend.W_PROFILE}.
"""
    )


def main() -> None:
    st.title("✨ GlowMatch")
    st.caption(
        "Hybrid beauty product recommendations — collaborative filtering, content "
        "similarity, and skin-profile affinity, with the reasoning shown for every result."
    )

    tab_item, tab_profile, tab_user, tab_eval = st.tabs(
        ["More like this", "New user", "Existing reviewer", "Evaluation"]
    )
    with tab_item:
        mode_by_item()
    with tab_profile:
        mode_by_profile()
    with tab_user:
        mode_by_user()
    with tab_eval:
        tab_evaluation()


if __name__ == "__main__":
    main()
