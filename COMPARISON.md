# How GlowMatch compares to production recommender systems

A comparison of GlowMatch against four published industrial architectures — Amazon, YouTube, Spotify, Netflix — followed by an honest account of what those systems do that this one does not, and one problem this domain has that none of them face.

Each comparison is anchored to a specific published design rather than to general reputation.

---

## Amazon — item-to-item collaborative filtering

**Their architecture.** Linden, Smith and York's 2003 paper *Amazon.com Recommendations: Item-to-Item Collaborative Filtering* (IEEE Internet Computing) describes the system behind "Customers who bought this item also bought." Its central move is a change of subject: instead of finding users similar to you and recommending what they liked, precompute a **similar-items table** offline — for each product, the products most often purchased alongside it — then at request time simply look up the items in your history and merge their neighbor lists.

Their reasoning was about scale. User-based CF requires comparing a user against a large fraction of the other users at request time, so cost grows with the number of *customers* — the fastest-growing quantity in their business. Item-item similarity is computed offline and depends on catalog size, which grows far more slowly. Online cost becomes a handful of table lookups, independent of how many customers exist.

**What GlowMatch does.** Essentially the same architecture, for the same reason. `cf.py` computes item-item similarity offline and persists the **top 50 neighbors per item** — `artifacts/cf_top_k_neighbors.parquet`, 99,655 rows covering 2,259 of 2,351 products. At request time `recommend.py` looks up the user's rated products, merges their neighbor lists, and sums similarities. No similarity is computed while anyone waits.

The scaling argument holds here for exactly the reason Amazon gave. This dataset has **503,216 users but only 2,351 reviewed products** — a 214:1 ratio. A user-user approach would need a 503,216 × 503,216 similarity space; the item-item table is 2,351 × 50 after truncation. Storing top-K rather than the dense matrix is O(N·K) instead of O(N²), which at 2,351 items would fit in memory anyway but is what keeps the deployed app inside Streamlit Community Cloud's limits and stays correct if the catalog grows.

**Where they differ.** Amazon's signal is purchases; GlowMatch's is written reviews, which are far sparser and self-selected (see Gaps). Amazon's neighbor table is refreshed continuously against live traffic; this one is rebuilt by running a script.

---

## YouTube — two-stage candidate generation and ranking

**Their architecture.** Covington, Adams and Sargin's 2016 paper *Deep Neural Networks for YouTube Recommendations* (RecSys) splits recommendation into two neural networks. **Candidate generation** reduces a corpus of millions of videos to a few hundred plausible ones, using a coarse, cheap model that treats the problem as extreme multiclass classification and retrieves nearest neighbours in a learned embedding space. **Ranking** then applies a much richer model — hundreds of features about the user, the video and the context — to just those few hundred candidates, producing the final order.

The split exists because precision and cost pull in opposite directions. A model rich enough to rank well is far too expensive to run against millions of items per request; a model cheap enough to run against everything is too coarse to order the top results well. Two stages let each do the job it is suited to.

**What GlowMatch does.** A single stage. Every candidate a user could receive is scored by the same blended function, and the top ten are returned.

**Why that is fine here, and when it stops being fine.** The CF-scoped catalog is **2,351 products**. Scoring all of them for one user is a merge and a groupby over a few thousand rows — milliseconds. There is no cost pressure to relieve, and introducing a retrieval stage would add machinery, a second model to train and a new source of error (anything the retriever misses can never be ranked) in exchange for nothing measurable.

The calculus inverts with catalog size. At roughly 10⁵–10⁶ items, scoring everything per request stops being viable: the blend would need a similarity lookup and three normalizations across every item, per user, per request. That is the point to split — use the CF neighbor table and content embedding nearest-neighbours as cheap retrieval (they already are retrieval, structurally: `top_k_neighbors` *is* a candidate generator), then apply a richer ranker over the few hundred survivors, with features this system currently ignores: price, recency, brand affinity, session context, ingredient conflicts.

This is the honest reason [Future improvements](README.md#future-improvements) lists two-stage retrieval rather than implementing it. The architecture is already shaped for the split — the neighbor tables are the retrieval layer — but building it at 2,351 items would be engineering for a problem this project does not have.

---

## Spotify — cold start from content when behaviour is missing

**Their architecture.** Van den Oord, Dieleman and Schrauwen's 2013 paper *Deep Content-Based Music Recommendation* (NIPS) addresses the failure mode of pure collaborative filtering: a track nobody has streamed has no collaborative signal, so a CF system cannot recommend it — and cannot recommend it until someone streams it, which it will not prompt. Their solution is to train a convolutional network to predict a track's **latent factor vector directly from its audio waveform**. A brand-new track gets a position in the same latent space as everything else, derived from what it sounds like rather than from who played it.

**What GlowMatch does.** The same manoeuvre in a different modality. `content.py` embeds product text — name, brand, category, highlights, ingredients — with `all-MiniLM-L6-v2` and persists top-50 nearest neighbours for the **full 8,494-product catalog**. Where Spotify substitutes audio for missing plays, GlowMatch substitutes product attributes for missing reviews.

The need is more acute here than the analogy suggests. It is not a handful of new products lacking signal: **6,143 of 8,494 catalog products — 72% — have zero reviews**, because the review corpus covers skincare only and the catalog spans makeup, hair, fragrance and bath. For those products content similarity is not a fallback, it is the only signal that exists. That is why content embeddings cover the whole catalog while CF and skin-profile are scoped to the reviewed 2,351.

**Where they differ, and where this system is weaker.** Spotify's network is trained to *predict collaborative factors* — audio is mapped into the space CF already learned, so content and behaviour are directly comparable by construction. GlowMatch's embeddings are general-purpose sentence embeddings, trained on no signal from this dataset at all. They capture textual similarity, not learned substitutability, and the two are not the same: two sunscreens with near-identical ingredient lists may perform very differently on the same skin. The evaluation shows the cost — content-only reaches HitRate@10 0.0766 against CF's 0.1992. Training embeddings to predict CF factors, Spotify-style, is the most promising unexplored improvement in this system.

Honesty about labeling matters here too: because content-only recommendations are genuinely weaker, the UI marks anything without rating-based support as *"Matched on attributes — no rating data yet"* rather than presenting it as equivalent.

---

## Netflix — when offline accuracy does not justify production cost

**Their architecture, and the part that was not deployed.** The Netflix Prize (2006–2009) awarded $1M for a 10% RMSE improvement, won by an ensemble of over a hundred blended models. Netflix's engineering blog later explained that they deployed the two strongest components — matrix factorization and RBMs — and **did not put the full ensemble into production**. The additional accuracy did not justify the engineering cost of running and maintaining it, and by then the business had shifted from DVD-by-mail to streaming, where the offline metric being optimized no longer matched what mattered.

**Why this matters to GlowMatch, concretely.** This is not a cautionary tale I am repeating; it is a decision this project already made twice, with measurements.

**Blend weight tuning.** A 66-point grid search over the three weights, optimizing NDCG@10, then verified across three independent samples. Result: the search moved NDCG@10 by roughly 3.5%, while variation *between evaluation samples* was several times the variation between weight settings. An earlier round moved it by 0.7% with between-sample variation ~8× the between-weight variation. The honest conclusion — recorded in `blend.py` — is that the exact proportions barely matter as long as the blend is not at a corner. Shipping a finely-tuned weight vector as if it were a result would have been overfitting a sample and calling it engineering.

**The hybrid itself.** The deployed blend beats CF-only by +0.003 HitRate@10 — statistically significant in 2 of 3 samples, insignificant in the third. If accuracy were the only consideration, **CF-only would be the defensible choice**, and the hybrid would be exactly the kind of complexity Netflix declined: more moving parts, three tables instead of one, for a gain that does not reliably reproduce. The hybrid is deployed for a different reason that does hold up — catalog coverage of 83.5% against CF-only's 74.1% — and the README says so plainly rather than dressing the accuracy difference up as the justification.

**Matrix factorization is deliberately absent.** ALS or BPR would likely improve accuracy and is listed under future improvements. It was scoped out because a working, honestly-evaluated, deployed hybrid was worth more than a half-finished stronger model — the same trade Netflix made when it shipped two components instead of a hundred.

---

## Honest gaps

Four differences where the production systems above are doing something this one is not.

**Implicit behavioural signals versus explicit ratings.** Amazon uses purchases, YouTube watch time, Spotify plays and skips — all implicit, abundant, and generated by everyone who uses the product. GlowMatch has explicit star ratings from people who chose to write a review. Three consequences: the data is drastically sparser (**0.092% density** across all 503,216 users); it is self-selected, capturing people motivated by a strong reaction in either direction while the silent majority is invisible; and it is compressed — **82% of ratings are 4 or 5 stars, 63.8% are 5 alone**. That skew is not cosmetic. It is precisely why item similarity must be adjusted-cosine rather than raw, and it caused the central bug in this project: scoring candidates with the rating-prediction formula `Σ(sim·r)/Σsim` collapsed to roughly 4.5 for nearly every candidate, making CF rank almost arbitrarily and understating it by 16× (see [Design decisions](README.md#design-decisions)). A denser implicit signal would sidestep the entire class of problem.

**Online A/B testing versus offline evaluation.** Every number in this project comes from leave-one-out replay: would the system have surfaced the item this user actually reviewed next? The four companies above run continuous online experiments measuring what users *do* when shown recommendations. The gap is not one of rigour but of question. Offline evaluation rewards predicting what already happened; it cannot measure whether a recommendation would have been acted on, whether the user would have preferred something they never saw, or whether showing recommendations changes behaviour at all. Offline and online results routinely disagree, and when they do the online result is the one that counts. Nothing here can settle that.

**Exploration and bandits versus none.** GlowMatch always shows its current best guess. It never shows an item to *find out* what happens, so it cannot learn about products it has decided are unpromising. Production systems reserve slots for exploration — multi-armed bandits, Thompson sampling — precisely because a purely exploitative recommender degrades: recommended items get engagement, engagement strengthens their signal, they get recommended more, and unrecommended items are never given the chance to prove otherwise. Coverage of 83.5% means this loop is not yet acute, but nothing in the design counteracts it, and this system has no mechanism to break it.

**Session awareness versus batch precomputation.** Everything is precomputed offline; the deployed app loads static tables and does lookups. It has no notion of what someone viewed a minute ago, no model of intent within a visit, and no response to a new rating until artifacts are rebuilt. YouTube's ranker takes session context as a first-class input. A shopper who has spent ten minutes on sunscreen gets identical results here to one who just arrived.

---

## The thing none of them face

The four systems above optimize for engagement in domains where a wrong answer is cheap. A bad film recommendation costs someone an evening. A bad song is skipped in three seconds. The feedback is immediate, the harm is negligible, and the system corrects itself.

**Beauty is not that domain.** A recommended product goes onto someone's skin. A wrong recommendation can mean an allergic reaction, a irritant response, a breakout that takes weeks to settle, or a foundation shade that does not exist for someone's skin tone. The feedback loop is slow, the harm is physical, and the person cannot un-apply it.

That changes what "recommending well" means, and it makes coverage across skin tones a **fairness problem with material consequences** rather than a metric. The concern is structural, not hypothetical. This dataset's `skin_tone` field has 14 values and is populated on 84.4% of reviews, but the distribution across those values is not uniform — the review population skews, and any purely collaborative signal inherits that skew. Products that work well for deeper skin tones accumulate fewer reviews if fewer reviewers with those tones are present; fewer reviews mean weaker collaborative signal; weaker signal means fewer recommendations; fewer recommendations mean fewer reviews. Collaborative filtering does not correct this. It **amplifies** it, because it is doing exactly what it was designed to do — propagate the patterns present in the data.

Three things in this system push back, none of them sufficient:

- **Skin-profile affinity conditions on tone explicitly.** Instead of one global average, each product carries per-`skin_tone` means, so "rated 4.6 by reviewers with deep skin tone" is a distinct quantity from a product's overall rating.
- **Shrinkage prevents overconfidence on thin evidence.** A 5.0 from three reviewers with a given tone is pulled toward the product's own mean, so sparse groups do not produce loud, unreliable claims — the failure mode that would make an under-represented tone's recommendations look confident and be wrong.
- **The evidence is shown, not hidden.** The UI displays the reviewer count behind every profile claim, so a user can see that a recommendation rests on 210 reviewers or on 4 and judge accordingly.

What none of that does is **manufacture data that was never collected**. Shrinkage handles thin evidence honestly; it does not create evidence. If a tone group is under-represented in the reviews, this system is quieter and less certain for those users, and that is a real limitation, disclosed in the README rather than smoothed over. The correct fix is not algorithmic — it is collecting representative data in the first place, and auditing recommendation quality *per tone group* rather than trusting an aggregate metric that a well-served majority can carry on its own. That audit is listed under [Future improvements](README.md#future-improvements) and is the single most important thing this project has not done.
