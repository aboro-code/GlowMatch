# CLAUDE.md — GlowMatch

Project instructions for Claude Code. Read this before writing any code.

## What this is

A hybrid recommendation system for beauty and skincare products, built on the Sephora products-and-reviews dataset (~8k products, ~1M reviews). It combines three independent signals — collaborative filtering, content similarity, and skin-profile affinity — into a single ranked recommendation, and evaluates the result against honest baselines.

## Fixed architecture decisions

These are settled. If you think one is wrong, say so before writing code rather than quietly doing something else.

**Three signals, all of which ship:**

- **Item-item collaborative filtering** — the primary signal, **scoped to the 2,351 reviewed products** (see Data below — the review corpus is skincare-only). Learns from what people actually rated, not from what products look like on paper. This is the difference between a recommender that feels smart and one that just does keyword matching in disguise.
- **Content-based embedding similarity** — covers the **full 8,494-product catalog**, including the 6,143 makeup/hair/fragrance/etc. products that have no reviews at all. This is not just the cold-start fallback for thin-review skincare items; it's the only signal that exists outside skincare.
- **Skin-profile affinity** — scoped to the same 2,351 reviewed (skincare) products, since the affinity aggregates are computed from review data. What makes this a *beauty* recommender rather than a generic one. Reviews carry the reviewer's skin type and skin tone, so "rated 4.6 by people with your skin type" is available and is a far more useful signal in this domain than ingredient overlap.

**A recommendation with no collaborative support must say so in the UI.** Any product outside the reviewed 2,351 — or any recommendation that leans on content-only because CF/profile had nothing for that item — gets an explicit label, e.g. *"Matched on attributes — no rating data yet."* Never let a content-only match be presented as if it had behavioral backing it doesn't have.

**Fusion is a weighted score blend**, not Reciprocal Rank Fusion. These three signals produce comparable normalized scores, so blending scores preserves magnitude information that rank-based fusion would throw away. Weights are chosen by evaluation, not by intuition.

**CF persists top-K neighbors per item (K≈50), never a dense N×N similarity matrix.** This is O(N·K) instead of O(N²) — the approach Amazon's 2003 item-to-item paper describes for making this tractable — and it's also what keeps memory inside the deployment target's limits. At 2,351 CF-scoped products a dense matrix would be small enough to fit anyway, but top-K stays the design regardless — it's the correct approach independent of current catalog size, and it's what's documented and evaluated.

**CF matrix-build threshold vs. eval-cohort threshold are different numbers, deliberately.** The user×item matrix is built from every user with **≥2** distinct rated products (~209k users) — a single-review user contributes zero co-occurrence signal, but anyone with 2+ does, and excluding them would throw away real signal for no reason. Leave-one-out evaluation instead draws from the **≥5**-rating cohort (~40k users) — evaluation needs enough held-out history per user to measure something meaningful, which is a stricter requirement than just contributing to the matrix.

**Item-item similarity is adjusted cosine (mean-centered per user before computing cosine), not raw cosine.** 82% of ratings in this dataset are 4★ or 5★ — on that distribution, raw-rating cosine similarity barely distinguishes itself from binary co-occurrence, because nearly every rated pair looks like "both high." Subtracting each user's mean rating first before computing cosine surfaces relative preference (rated higher/lower than that user's own average) instead of just "both were rated positively by someone." Document this reasoning in the docstring of whatever function computes item-item similarity in `cf.py`, not just here.

**All recommendation logic lives in UI-agnostic top-level modules** (`data.py`, `matrix.py`, `cf.py`, `content.py`, `skin_profile.py`, `blend.py`, `recommend.py` — at this repo root, not inside a nested package subfolder; note `skin_profile.py` not `profile.py`, since the latter shadows Python's stdlib `profile` module and breaks any dependency that needs it, e.g. `sentence_transformers` via `torch._dynamo`/`cProfile`). The Streamlit app imports them and does presentation only. No business logic in `app.py`. This keeps the core testable and leaves the door open to serving it over HTTP later without rework.

**Everything expensive is precomputed offline.** The deployed app loads small artifacts and does lookups. It must never build embeddings or similarity matrices at request time.

## Non-negotiables

**The evaluation includes random and popularity baselines.** A recommender that hasn't been compared against "just recommend the most popular items" hasn't really been evaluated — plenty of sophisticated systems fail to beat that bar. Report the real numbers even when the margin is uncomfortable. Never tune the evaluation to produce a flattering result, and never quietly drop a baseline that performs too well.

**Never fabricate or estimate a metric.** Every number that appears in the README, the UI, or any document comes from an actual run of the evaluation harness. If something hasn't been measured, leave a visible placeholder and say so.

**Every recommendation shows why it was recommended** — which signal drove it, and the supporting evidence: "users who rated X highly also rated this 4.5★", "4.6★ from 210 reviewers with combination skin", "shares niacinamide and hyaluronic acid". An unexplained recommendation is hard to trust and impossible to debug.

**Exclude already-rated items from recommendations** — in the app and in the evaluation harness both. Recommending someone the product they just reviewed is the classic recommender bug and it silently inflates every metric.

**Cold start has defined behavior.** A user with no history and a product with too few reviews both need tested, sensible fallbacks — not a crash, not an empty list.

## Data

Kaggle: Sephora Products and Skincare Reviews. **Verified 2026-08-25** (see `scripts/verify_dataset.py`) — numbers below are measured, not assumed.

- Fetched through the **Kaggle API** via `scripts/fetch_data.py`, never a manual download. It checks `data/raw/` first, then a sibling-project cache, then falls back to Kaggle. Someone should be able to clone the repo, run the fetch script, run the build script, and have a working application.
- `data/` is gitignored — the raw review CSVs exceed GitHub's 100MB file limit. What gets committed is the small precomputed artifacts plus the scripts that regenerate them.
- Review fields confirmed present: `author_id` (str, 0 nulls), `rating` (int, 1–5, 0 nulls), `product_id`, `skin_tone` (84.4% populated, 14 values), `skin_type` (89.8%, 4 values), `hair_color` (79.3%, 7 values), `eye_color` (80.8%, 6 values), `submission_time`, `review_text`, `is_recommended`.
- **The review corpus is skincare-only.** 1,094,411 reviews cover exactly **2,351 products, and every one of them is `primary_category == "Skincare"`** (97.1% of the 2,420 skincare products in the catalog). Zero of the other 6,074 makeup/hair/fragrance/bath&body/etc. products have any reviews. This is why CF and skin-profile are scoped to the 2,351 and content-based similarity carries the rest of the 8,494-product catalog — see Fixed architecture decisions above.
- User activity: 503,216 distinct reviewers. 41.6% (209,105) have rated ≥2 distinct products, 8.0% (40,433) have rated ≥5, 1.8% (9,246) have rated ≥10. Median is 1 review/user, but products average 465 reviews each — item-item co-occurrence is well supported despite the long individual-user tail. See the two-threshold rule above.
- Data quality: 5,525 duplicate `(author_id, product_id)` pairs and 224 fully-duplicate rows need deduping (keep latest by `submission_time`) before matrix construction. No out-of-range ratings, no null author/product/rating.
- If `author_id` proves missing or unusable, **stop and report** rather than substituting an approach — collaborative filtering is the centerpiece and its absence changes the design. (Not the case here — already verified.)

## Structure

The core modules live directly at the top level of this repo, not inside a nested package directory:

```
data/
  raw/           raw CSVs — gitignored, fetched via Kaggle API
  processed/     build-time parquet cache — gitignored, regenerated by data.py
data.py          loading, cleaning, dedup, filtering — build-time, UI-agnostic
matrix.py        sparse user×item construction
cf.py            item-item collaborative filtering (adjusted cosine, top-K)
content.py       embeddings + content similarity over the full catalog
skin_profile.py  skin-profile affinity aggregates (skin type, skin tone)
ingredients.py   curated key-active extraction, used for content evidence strings
blend.py         score fusion + final ranking
recommend.py     public API: recommend_by_item / recommend_by_user / recommend_by_profile — serve-time, reads only through serve.py
serve.py         serve-time data access — reads exclusively from artifacts/, no fallback to data/processed/
scripts/
  fetch_data.py       Kaggle API download (with local-cache short-circuit) → data/
  verify_dataset.py   one-off data verification / sanity checks, not part of the pipeline
  build_artifacts.py  offline precompute → artifacts/
eval/
  run_eval.py         leave-one-out harness across all systems and baselines
  tune_weights.py     simplex grid search over blend weights (+ multi-seed --verify)
  significance.py     paired McNemar / bootstrap tests, hybrid vs CF-only
  diagnose_hybrid.py  one-off investigation that found the CF scoring bug
  *.csv               committed results of the runs above
artifacts/       small committed .parquet files — the only data the deployed app reads
app.py           Streamlit UI — presentation only
README.md        setup, architecture, methodology, results, limitations
COMPARISON.md    how this compares to Amazon / YouTube / Spotify / Netflix
requirements.txt, requirements-dev.txt   the deliberate split described below
```

Type hints on public functions. Docstrings that explain *why*, not just *what* — the reasoning behind a design choice is worth more to the next reader than a restatement of the signature.

**Two requirements files, deliberately split.** `requirements.txt` (pandas, pyarrow only) is what the deployed Streamlit app actually needs — the serve-time path (`app.py` → `recommend.py` → `serve.py`) never imports scipy/scikit-learn/sentence-transformers/torch, so shipping them to Streamlit Cloud would bloat the deploy and slow cold start for no reason. `requirements-dev.txt` adds numpy/scipy/scikit-learn/sentence-transformers/kaggle on top, for running `scripts/fetch_data.py` and `scripts/build_artifacts.py` locally.

**Never name a module `profile.py`.** It shadows Python's stdlib `profile` module (used internally by `cProfile`, which `torch._dynamo` imports), and breaks `sentence_transformers`/`torch` with a confusing deep-stack `ModuleNotFoundError` the moment they're actually imported fresh rather than served from a warm cache. Found this the hard way after it stayed silent through an entire day of testing because every embedding/CF call happened to hit a warm cache. The module is `skin_profile.py`.

## Working style

Work phase by phase. After each phase, stop and summarize what you built and what you decided, then wait. Don't run the whole project end to end unattended.

Flag tradeoffs out loud as you hit them rather than silently picking a side. When something is genuinely uncertain, or you had to guess, say so plainly instead of presenting it as settled.

Priority order when something has to give: **working deployed application > honest evaluation > documentation > HTTP API layer > matrix factorization.** Cut from the bottom.




