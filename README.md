# GlowMatch

A hybrid beauty product recommendation system built on the Sephora products and skincare reviews dataset. It combines item-item collaborative filtering, content-based similarity, and a skin-profile affinity signal, and evaluates the result honestly against random and popularity baselines.

**Live app:** https://glowmatch-aboro-code.streamlit.app/
**Repository:** https://github.com/aboro-code/GlowMatch
**Comparison to production systems:** [COMPARISON.md](COMPARISON.md) — Amazon, YouTube, Spotify, Netflix, and what this system does not do

The deployed app sleeps when idle; a cold first load takes roughly 20 seconds while the container wakes, then responds in well under a second.

### Results at a glance

Leave-one-out over 5,000 users, held-out interactions removed before the collaborative-filtering model was rebuilt.

| | HitRate@10 | Catalog coverage |
|---|---|---|
| Popularity baseline | 0.0216 | 1.1% |
| **Collaborative filtering** | **0.1992** | 74.1% |
| **Hybrid (deployed)** | **0.2022** | **83.5%** |

Collaborative filtering carries this system — it beats the popularity baseline **9×**. The hybrid adds only +0.003 hit rate on top of that, which is significant in just 2 of 3 samples and is **not** the reason to prefer it; its real contribution is catalog coverage, reaching 83.5% of products against popularity's 1.1%. Full table, significance tests, and an account of a scoring bug that initially inflated these numbers are in [Evaluation methods](#evaluation-methods) and [Design decisions](#design-decisions).

---

## Problem statement

Someone shopping for skincare faces roughly 8,500 products, most of which they will never see, described in language that is deliberately similar across brands. Ranking by average rating does not help — ratings cluster so tightly near the top that almost everything looks like a 4.3 — and ranking by popularity shows every shopper the same twenty bestsellers regardless of who they are.

The task is to rank a large catalog for an individual: given either a product someone likes, their own rating history, or nothing but their skin type and tone, return ten products they plausibly want, and be able to say why each one appeared.

## Use case

Three entry points, matching three real situations:

1. **"More like this."** A shopper is looking at one product and wants comparable options. No account, no history — one product ID is the entire input.
2. **New user / cold start.** No history at all. They pick a skin type and skin tone, and the system recommends from what people with that profile actually rated well.
3. **Existing user.** A known reviewer with real rating history gets fully personalized recommendations drawn from everything they have rated, with what they have already reviewed excluded.

Every recommendation displays its reasoning, and anything with no rating-based support is explicitly labeled as such rather than presented as if people had endorsed it.

## Approach

Three independent signals, blended by weighted score fusion:

**Item-item collaborative filtering** is the primary signal. Adjusted cosine similarity between item rating-vectors, with only the top 50 neighbors per item persisted. It learns from what people actually rated together rather than from how products describe themselves, which is what lets it surface a product whose marketing copy looks nothing like the seed.

**Content-based similarity** covers the whole catalog. Product text (name, brand, category, highlights, ingredients) is embedded with `all-MiniLM-L6-v2` and compared by cosine similarity. This is the only signal that exists for the 6,143 catalog products with no reviews at all.

**Skin-profile affinity** is the beauty-specific signal. For each `(product, skin_type)` and `(product, skin_tone)` pair, the mean rating among reviewers with that attribute, shrunk toward the product's own mean so a 5.0 from three reviewers does not outrank a 4.6 from two hundred. It contributes almost nothing to warm-user ranking (see Evaluation) and earns its place on the cold-start path and as displayed evidence.

Fusion is a **weighted score blend**, not Reciprocal Rank Fusion: these signals produce comparable normalized scores, so blending scores preserves magnitude information that rank-based fusion would discard.

## Architecture

```
OFFLINE (scripts/build_artifacts.py)          SERVING (app.py)
────────────────────────────────────          ─────────────────────────────
data/raw/*.csv  (529 MB, gitignored)                 user input
      │                                                   │
      ▼                                                   ▼
  data.py ── clean, dedup, join ──┐                   recommend.py
      │                           │                        │
      ├─► matrix.py               │              ┌─────────┴─────────┐
      │   sparse user×item CSR    │              │  CF neighbors     │
      │   209,105 × 2,351         │              │  Content neighbors│─┐
      │        │                  │              │  Profile affinity │ │
      │        ▼                  │              └───────────────────┘ │
      │    cf.py ── adjusted      │                        │           │
      │    cosine → top-50        ├──► artifacts/          ▼           │
      │                           │    (18 MB,        blend.py ────────┘
      ├─► content.py ── MiniLM    │     committed)         │
      │    embeddings → top-50    │         ▲              ▼
      │                           │         │        top-10 + reasons
      └─► skin_profile.py ────────┘         └──────── serve.py
           shrunk affinity                        (reads artifacts only)
```

Two deliberate boundaries:

**Build-time and serve-time are separate code paths.** `data.py`, `matrix.py`, `cf.py`, `content.py` and `skin_profile.py` read raw CSVs and do the expensive work. `serve.py` reads *only* the committed `artifacts/` directory. There is no fallback between them — a deployed instance has no `data/` directory, and if artifacts are missing it fails loudly rather than silently attempting a rebuild it has no data for. This is why local and deployed behavior cannot diverge.

**Top-K neighbors, never a dense similarity matrix.** Storing the 50 nearest neighbors per item is O(N·K) rather than O(N²). At 2,351 CF-scoped products a dense matrix would fit, but the design is correct independent of catalog size and is what keeps the deployed app inside Streamlit Community Cloud's memory limit.

## Methodology

**Deduplication.** Raw reviews contain 224 byte-identical rows and repeat `(author_id, product_id)` pairs. Exact duplicates are dropped, then repeat pairs collapse to the most recent by `submission_time`. 1,094,411 raw reviews become **1,088,886**. Type casting happens *before* deduplication, not after — see Design decisions.

**CF matrix.** Built from every user with ≥2 distinct rated products (209,105 users), because a single-review user contributes no co-occurrence information but everyone with two or more does. Columns are all 2,351 reviewed products, including 21 that end up all-zero, so the index stays a complete, stable mapping.

**Adjusted cosine similarity.** Each user's mean rating is subtracted from their own ratings before computing cosine similarity between items. With 82% of ratings at 4–5 stars, raw-rating cosine is barely distinguishable from binary co-occurrence — nearly every co-rated pair looks like "both rated highly." Mean-centering surfaces whether a user rated something *above or below their own average*, which is the actual preference signal. Centering is applied to the sparse `.data` array directly (`np.repeat(user_means, np.diff(indptr))`); densifying a 209,105 × 2,351 matrix would need roughly 2 GB.

**Scoring.** A candidate's score is the **sum of similarities** from every product the user has rated to that candidate. This is deliberately *not* the textbook `Σ(sim·r)/Σsim` prediction formula — that measures rating prediction, and using it for ranking was a real bug in this project. See Design decisions.

**Cold start.** Products with no reviews get content-only recommendations, labeled. Users with no history use the profile path. Both are tested.

## Dataset selection

**Kaggle: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)** (`nadyinky/sephora-products-and-skincare-reviews`), 529 MB across six CSVs.

Chosen because it is one of the few public retail datasets carrying **per-reviewer skin attributes** — `skin_tone`, `skin_type`, `hair_color`, `eye_color` — alongside a real user identifier and explicit ratings. That combination is what makes both collaborative filtering and a demographic-conditioned signal possible on the same data.

Verified before any code was written (`scripts/verify_dataset.py`):

| | |
|---|---|
| Reviews (after dedup) | 1,088,886 |
| Distinct users | 503,216 |
| Products with reviews | 2,351 |
| Full catalog | 8,494 |
| User×item density, all 503,216 users | 0.092% |
| Users with ≥2 ratings | 209,105 (41.6%) |
| Users with ≥5 ratings | 40,433 (8.0%) |
| Products with ≥5 reviews | 2,224 (94.6%) |
| Rating distribution | 1★ 5.6% · 2★ 4.9% · 3★ 7.5% · 4★ 18.2% · **5★ 63.8%** |
| `skin_type` populated | 89.8% (4 values) |
| `skin_tone` populated | 84.4% (14 values) |

**The single most important finding:** all 2,351 reviewed products are `primary_category == "Skincare"`. Zero of the other 6,143 catalog products — makeup, hair, fragrance, bath & body — have a single review. This is a hard scope boundary, and the architecture is built around it rather than papering over it.

## Technologies

| Component | Choice | Why |
|---|---|---|
| Data processing | pandas 3.0.2, pyarrow 25.0.1 | Parquet caching keeps re-runs at ~0.6s instead of re-parsing 529 MB |
| Sparse matrices | scipy 1.18.1 | 209,105 × 2,351 at 0.16% density (the CF matrix, ≥2-rating users only — denser than the 0.092% over all 503,216 users, since single-review users are excluded); dense would be ~2 GB |
| Similarity | scikit-learn 1.9.0 | `cosine_similarity(dense_output=False)` stays sparse throughout |
| Embeddings | sentence-transformers 6.0.0, `all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong on short product text |
| UI | Streamlit 1.62.0 | Fastest path to a deployed, interactive app |
| Significance testing | scipy `binomtest` + bootstrap | Paired tests on per-user outcomes |
| Runtime | Python 3.14 | — |

**Two requirements files, deliberately.** `requirements.txt` is pandas, pyarrow, streamlit only — the serving path never imports torch, scikit-learn or scipy, so the deployment stays small and cold start stays fast. `requirements-dev.txt` adds the offline pipeline dependencies.

## Assumptions

These are the load-bearing assumptions. Where one is questionable, it is called out in Limitations rather than buried.

**A written review implies engagement, and engagement implies preference.** Collaborative filtering treats co-reviewing as evidence of shared taste. Someone who reviews two products has revealed a connection between them — but a review can be prompted by disappointment as easily as delight, which is why the similarity is adjusted-cosine (mean-centered per user) rather than raw.

**Reviewers are representative enough of shoppers to be useful.** They are certainly not representative in general (see Limitations). The working assumption is only that co-review patterns among reviewers correlate with co-purchase patterns among shoppers well enough to rank a catalog.

**Self-reported skin attributes are accurate and stable.** `skin_type` and `skin_tone` are taken at face value. Verified as reasonable: only 1.9% of users report more than one distinct `skin_type` across their own reviews, so `recommend_by_user` infers a user's attributes as the mode of their history.

**A product's identity is stable across its reviews.** Reviews are joined to the catalog on `product_id`, assuming it refers to the same product over time — not a reformulation or a repackaged variant. The dataset provides no way to detect either.

**The most recent interaction is a fair prediction target.** The evaluation holds out each user's latest review, assuming it represents a genuine forward-looking choice rather than an artifact of scrape timing.

**Ratings are comparable across users after centering.** One person's 4 is another's 5. Mean-centering assumes that subtracting each user's own average makes the residuals comparable — standard practice, and imperfect: it corrects for bias in level but not for differences in how people use the scale's range.

**Content similarity approximates substitutability.** Products with similar names, categories and ingredients are assumed to be plausible alternatives. This holds reasonably inside skincare and less well across the unreviewed catalog, where it is the only signal available.

**Absence of a review is not a negative signal.** Unrated products are treated as unknown, not disliked — the standard assumption for explicit-feedback systems, and the reason the matrix is sparse rather than zero-filled.

## Setup

Verified by running these exact steps in a clean clone, not written from memory.

### Just run the app (no dataset needed)

Artifacts are committed, so the app runs immediately:

```bash
git clone https://github.com/aboro-code/GlowMatch.git
cd GlowMatch
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
streamlit run app.py
```

On macOS and most Linux distributions the interpreter is `python3`; substitute it for `python` throughout if `python` is not on your path.

### Rebuild everything from the raw data

Requires Kaggle API credentials at `~/.kaggle/kaggle.json` (on Windows, `%USERPROFILE%\.kaggle\kaggle.json`). Create the token from your Kaggle account page under *Settings → API → Create New Token*.

```bash
pip install -r requirements-dev.txt
python scripts/fetch_data.py          # downloads ~529 MB into data/raw/
python scripts/build_artifacts.py     # rebuilds artifacts/ (~5 min)
python eval/run_eval.py               # regenerates the evaluation table (~47 s)
```

`fetch_data.py` checks `data/raw/` first and accepts `--cache <dir>` to copy from a local copy instead of re-downloading. `build_artifacts.py --force` recomputes every stage.

Measured on a clean clone: dependency install, then **5 min 15 s** for a full rebuild (250 s of that is embedding 8,494 products), producing 17.95 MB of artifacts. `data/` is gitignored; the raw CSVs exceed GitHub's 100 MB file limit.

## Design decisions

**Methodology note: a bug that inverted the headline result.**

An earlier version of this evaluation reported the hybrid at HitRate@10 0.1010 against CF-only at 0.0126 — the hybrid beating its own best component by **8×**. That is not a plausible outcome for weighted score fusion. A blend of three signals should land somewhere between its components, not an order of magnitude above all of them, and it was doing so while recommending items whose average popularity (464) was indistinguishable from random (462).

Investigation ruled out the obvious explanations in order. Profile-only scored 0.0002, so it was not secretly carrying the result. The harness never read production artifacts — all three signals came from the same leak-corrected in-memory frames. Held-out pairs were verifiably absent from training data and seed sets. Equalizing candidate pools across systems changed nothing.

The diagnostic that broke it open: ranking by **binary consensus** — "is this item in *both* the CF and content neighbor lists?", discarding every score magnitude — scored 0.0904, roughly 90% of the hybrid's 0.1010. The blend was not blending. It was acting as an intersection filter while the scores contributed almost nothing.

The cause was the CF scoring function. Candidates were scored with `Σ(sim·r)/Σsim`, the textbook item-based formula — which predicts **what rating a user would give**. That is the correct objective for RMSE and the wrong one for top-N ranking, because dividing by `Σsim` cancels exactly what ranking depends on: how strongly a candidate connects to the user's history. On this dataset the effect is severe. With 82% of ratings at 4–5 stars, the ratio collapses to roughly 4.5 for nearly every candidate, so CF was ranking close to arbitrarily inside its own pool.

Scoring by the **sum of similarities** instead moved CF-only from 0.0126 to **0.1992** — a 16× difference caused by a denominator. The "hybrid wins 8×" result was never a hybrid win; it was a crippled baseline, and the corrected comparison shows a properly-scored CF-only baseline nearly matching the full hybrid.

Alternatives were then measured rather than assumed, which mattered: weighting seeds by raw rating scored 0.1966, centering at 3.0 scored 0.1556, centering at each user's own mean 0.0840, and restricting seeds to ratings ≥4 scored 0.1566 — all worse than the plain sum's 0.1992. Centering *feels* obviously correct (a disliked product should push its neighbors down) and is wrong here, because the adjusted-cosine similarities are already mean-centered per user, so re-applying rating weights double-counts. The eval harness and the production path now share one scoring function; they had drifted apart, and that drift is how the bug survived as long as it did.

**Type casting happens before deduplication.** Dedup compares raw values, so when pandas inferred `author_id` inconsistently across CSV chunks, the integer `1011200472` and the string `"1011200472"` counted as different keys and both survived — and the later string cast silently collapsed them into genuine duplicate pairs. Five reached the committed artifacts, violating the one-row-per-`(user, product)` invariant that `matrix.py` and the holdout logic both assume. Casting first makes the result independent of per-chunk inference, and an assertion now fails the build rather than letting it ship.

**Only positive similarities become neighbors.** Adjusted cosine ranges [−1, 1], and a negative value means two items were rated in *opposite* directions relative to each user's average — evidence against a recommendation. Filling the top-50 with the least-bad negatives would hand 47 products a "top neighbor" that argues against them. After filtering, 2,259 of 2,351 products have at least one real CF neighbor; the other 92 correctly have none and fall back to content.

**Blend weights were tuned, and the tuning barely matters.** A 66-point simplex grid search optimizing NDCG@10 selected 0.8/0.1/0.1 over the previous 0.4/0.3/0.3. What the sweep shows clearly is that the *corners* are bad — single-signal weightings underperform — while the interior is flat. The earlier round of tuning, run before the scoring fix, moved NDCG by 0.7% while between-sample variation was roughly 8× the between-weight variation; those weights were fit to a broken signal and discarded.

**Profile keeps weight 0.1 despite contributing nothing measurable to warm-user ranking**, because the evaluation cohort structurally cannot measure what it is for. See Limitations.

**Shared ingredients use a curated active list, not raw INCI intersection.** Full ingredient lists overlap heavily on water, glycerin and tocopherol — true and useless. 40 recognizable actives were each verified to appear at an informative rate (0.1–20% of the catalog), with word-boundary anchoring so `urea` does not match polyurea.

**The empty-search message re-verifies against the live catalog.** Searching a brand Sephora does not carry explains where it is sold instead. Building that list, six brands I assumed were drugstore-only turned out to be **stocked** — Paula's Choice, The Inkey List, Mario Badescu, First Aid Beauty, Glossier, Summer Fridays. The check now confirms absence against the loaded catalog before claiming it, so a data refresh silently stops the claim rather than turning it into a confident lie.

## Evaluation methods

**Protocol.** Leave-one-out over 5,000 users sampled from the ≥5-rating cohort. For each user, the **most recent** review by `submission_time` is held out — not a random one. Predicting the future from the past is the only honest framing for a system that would run forward in time; holding out a middle interaction lets a model "predict" something using information that postdates it. Already-rated items are excluded from every system's recommendations.

### Leakage handling

**This is the part that most implementations get wrong, and it changes the numbers materially.**

The 5,000 held-out `(user, item)` pairs are removed from the review data *before* anything rating-derived is rebuilt for evaluation:

- the CF user×item matrix and its item-item similarities
- the skin-profile affinity aggregates

If they are not removed, the item-item similarity for the held-out product is computed **partly from the very rating being predicted**, and that user's held-out review still contributes to its `(product, skin_type)` and `(product, skin_tone)` means. Every CF-based and profile-based number is then inflated by the model having already seen the answer. The system is not predicting; it is remembering.

This is a **single** rebuild with all 5,000 interactions removed, not one rebuild per user. Leave-one-out only requires that the specific pair being predicted is absent — not that each user's whole history is — so removing all 5,000 at once and rebuilding once gives an identical result far faster.

Content similarity is **not** rebuilt: it is computed purely from product attributes and never touches ratings, so no rating leakage is possible.

Verified by assertion rather than assumption (`eval/diagnose_hybrid.py`): held-out pairs present in training reviews, **0**; present in any user's own seed set, **0**; and 2,328 `(product, skin_type)` cells have lower counts in the training aggregate than the production one, which is what removing those reviews should cause.

### Metrics

| Metric | What it means |
|---|---|
| **HitRate@10** | Fraction of users whose held-out product appeared in their top 10 |
| **Precision@10** | Fraction of recommended slots that were correct — mechanically HitRate/10 here, since there is exactly one relevant item per user |
| **Recall@10** | Fraction of relevant items retrieved — identical to HitRate@10 for the same reason |
| **NDCG@10** | Like HitRate, but rewards ranking the correct item *higher* in the ten |
| **Coverage** | Fraction of the 2,351-product catalog that ever gets recommended to anyone — catches a system that shows everyone the same bestsellers |
| **AvgPopularity** | Mean training review count of recommended items — a novelty proxy; lower means less obvious recommendations |

### Results

5,000 users, blend weights CF 0.8 / content 0.1 / profile 0.1, **random seed 42** (`RANDOM_SEED` in `eval/run_eval.py`, which fixes the cohort sample). Every number reproduced by `python eval/run_eval.py` with no arguments.

| System | HitRate@10 | Precision@10 | Recall@10 | NDCG@10 | Coverage | AvgPopularity |
|---|---|---|---|---|---|---|
| Random | 0.0062 | 0.00062 | 0.0062 | 0.0030 | 1.000 | 462 |
| Popularity | 0.0216 | 0.00216 | 0.0216 | 0.0121 | 0.011 | 7,696 |
| Content-only | 0.0766 | 0.00766 | 0.0766 | 0.0394 | 0.815 | 444 |
| CF-only | 0.1992 | 0.01992 | 0.1992 | 0.1844 | 0.741 | 170 |
| **Hybrid** | **0.2022** | **0.02022** | **0.2022** | **0.1873** | **0.835** | 327 |

**Reading this honestly:**

**Collaborative filtering carries the system.** CF-only reaches HitRate@10 0.1992 — 9× the popularity baseline — and the hybrid adds 0.0030 on top of that. The headline result is that CF works here, not that the blend is clever.

**The hybrid's accuracy edge over CF alone is small and only marginally detectable.** Paired McNemar tests plus bootstrap confidence intervals on per-user hit indicators, across three independent samples (`eval/significance.py`):

| Seed | CF | Hybrid | Difference | McNemar *p* | 95% CI |
|---|---|---|---|---|---|
| 42 | 0.1992 | 0.2022 | +0.0030 | 0.128 | [−0.0006, +0.0066] |
| 7 | 0.2006 | 0.2066 | +0.0060 | **0.0008** | [+0.0026, +0.0094] |
| 123 | 0.2088 | 0.2138 | +0.0050 | **0.0154** | [+0.0010, +0.0090] |

Positive in direction on all three samples, statistically significant on two of three, mean difference +0.0047. That is a small effect that is probably real but not reliably detectable at this sample size, and **it should not be presented as the reason to prefer the hybrid.**

**Coverage is where the hybrid actually earns its place**, and that effect needs no significance test. CF-only reaches 74.1% of the catalog; the hybrid reaches 83.5%. For contrast, the popularity baseline reaches **1.1%** — it recommends roughly twenty-five products to five thousand different people. A recommender that only surfaces bestsellers is not doing much recommending, and the popularity baseline's AvgPopularity of 7,696 against the hybrid's 327 shows how differently the two behave.

**Absolute numbers are low, and that is expected.** Identifying one specific held-out product out of 2,351 is a hard target. A ~20% hit rate at rank 10 is a real result.

**Profile-only scores 0.0004 — worse than random.** This is a genuine finding and the explanation matters; see Limitations.

## Test cases

| Case | Expectation | Status |
|---|---|---|
| Clean clone, no `data/` | All three `recommend_*` functions work from committed artifacts | Verified in a temp clone |
| Full rebuild from raw CSVs | Reproduces artifact shapes exactly | Verified, 5 min 15 s |
| Unknown `product_id` / `author_id` | Raises `ValueError`, not a crash | Verified |
| `recommend_by_profile()` with no attributes | Raises `ValueError` | Verified |
| Single-review user | Still gets CF-backed recommendations (neighbor lookup is per-product) | Verified |
| Product with zero reviews | Content-only, labeled "no rating data yet" | Verified |
| Already-rated exclusion | Rated items never recommended, in app and eval | Enforced in `_rank_and_truncate` |
| Mean-centering stays sparse | `nnz` unchanged, asserted | Verified, 794,775 → 794,775 |
| Dedup leaves no duplicate pairs | Hard assertion in `data.py` | Verified |
| All three UI modes | Run without exception, evidence renders | Verified via Streamlit `AppTest` |
| Search: multi-term, regex chars, no-match | AND semantics, literal matching, no crash | Verified |
| Leakage: held-out pair invisible | 0 in training reviews, 0 in seeds | Asserted in `diagnose_hybrid.py` |

Diagnostics live in `eval/diagnose_hybrid.py` (leakage and pool assertions) and `eval/significance.py` (paired tests).

## Limitations

**The review corpus is skincare-only.** 6,143 of 8,494 catalog products — every makeup, hair, fragrance and bath product — have **zero** behavioral signal. For those, "recommendation" means attribute similarity and nothing more. The UI labels it, but no amount of labeling makes it personalization.

**Explicit ratings, and heavily skewed ones.** 82% are 4–5 stars, 63.8% are 5 stars alone. A distribution that compressed carries far less information than its range suggests, and it is why adjusted cosine and sum-of-similarity scoring matter so much here.

**Self-selection bias.** These reviews come only from people motivated enough to write one — typically after a strong reaction in either direction, and disproportionately from more engaged shoppers. Nothing is observed about the majority who bought and stayed silent, so the system models *reviewers*, not customers.

**Offline evaluation is not online A/B testing, and the two routinely disagree.** Leave-one-out asks "would this system have shown the item the user chose next?" — a proxy that rewards predicting what already happened. It cannot measure whether a recommendation would have been *acted on*, whether a user would have found something better, or whether showing recommendations changes behavior. Systems that win offline regularly fail to move online metrics.

**Profile-only scores 0.0004, and the evaluation cannot fairly measure it.** The cohort is users with ≥5 ratings — precisely the users for whom skin-profile affinity is *least* useful, because their own history is far more informative than a demographic average. The signal exists for users with no history, and the leave-one-out design **structurally excludes** them: you cannot hold out an interaction from someone who has none. Measuring it properly needs a different design — hold out entire users, give the system only their skin attributes, and check whether their first real interaction appears in the recommendations. That is a genuine gap in this evaluation, not a defense of the number. The number stands as reported: on warm users, profile contributes essentially nothing to ranking.

**No exploration mechanism.** The system always recommends its current best guess. Nothing is ever shown to find out what would happen, so popularity bias compounds: recommended items get reviewed, reviews strengthen their signal, they get recommended more. Coverage of 83.5% keeps this from being acute, but the feedback loop only runs one way.

**Batch, not session-aware.** Everything is precomputed offline. The system has no notion of what someone looked at five minutes ago, no sense of intent within a visit, and no response to a rating until artifacts are rebuilt. A shopper who has just been browsing sunscreen gets the same recommendations as if they had not.

**Cold-start for genuinely new products.** A product with no reviews and sparse attribute text gets weak content recommendations and no way to improve until reviews accumulate.

**Skin-tone coverage is uneven.** `skin_tone` has 14 values but is populated on 84.4% of reviews, and the distribution across tones is not uniform. Shrinkage keeps thin evidence from producing overconfident scores, but it cannot manufacture data that was never collected. Collaborative filtering amplifies a skewed review population rather than correcting it — see [COMPARISON.md § The thing none of them face](COMPARISON.md#the-thing-none-of-them-face).

## Future improvements

- **Evaluate cold start properly** by holding out entire users, which is the one gap that currently prevents an honest claim about the profile signal.
- **Implicit signals** — views, add-to-cart, purchases — which are denser and less self-selected than written reviews.
- **Matrix factorization (ALS/BPR)** as a fourth signal. BPR in particular optimizes a ranking objective directly, which this project's central bug shows is the thing that matters.
- **Two-stage retrieval and ranking** once the catalog outgrows a single scoring pass — the neighbor tables are already structurally a retrieval layer. Why a single stage is correct at 2,351 items and stops being correct at ~10⁵–10⁶: [COMPARISON.md § YouTube](COMPARISON.md#youtube--two-stage-candidate-generation-and-ranking).
- **Fairness auditing per skin-tone group**, rather than trusting an aggregate a well-served majority can carry — see [COMPARISON.md](COMPARISON.md#the-thing-none-of-them-face).
- **A bandit-based exploration slot** to break the popularity feedback loop.
- **Session awareness** so within-visit behavior affects results.

## Relationship to BeautyRAG

This is a separate project from my [BeautyRAG](https://github.com/aboro-code/beautyrag) repository, and they solve different problems.

BeautyRAG is a **retrieval and question-answering** system: it indexes product and review text and answers natural-language questions about it ("is this good for sensitive skin?"), where success means retrieving passages that support a correct answer, and the evaluation is about answer quality and grounding. GlowMatch is a **behavioral recommendation** system: nobody asks it a question. It ranks a catalog from interaction patterns, and success means the item someone chose next appears in ten slots. The only shared machinery is content embedding similarity — and even that plays opposite roles. In BeautyRAG, embedding retrieval *is* the core. Here it is a supporting signal at weight 0.1, well behind collaborative filtering, and its main job is covering the 6,143 products the behavioral signal cannot reach. The fusion methods differ for the same reason: BeautyRAG uses Reciprocal Rank Fusion over incomparable ranked lists, while GlowMatch blends normalized scores, because here the magnitudes are comparable and worth keeping.
