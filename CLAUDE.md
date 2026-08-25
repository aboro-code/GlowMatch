# CLAUDE.md — GlowMatch

Project instructions for Claude Code. Read this before writing any code.

## What this is

A hybrid beauty product recommendation system, built as a technical assignment for a GenAI internship at Orbo.ai (a Mumbai BeautyTech company). Hard deadline: **Thursday end of day.** Three days total. Scope discipline matters more than ambition.

This is graded on: problem-solving methodology, technical execution, recommendation effectiveness, ML/AI comprehension, system architecture, product perspective, code standards, documentation clarity, innovation, UX, and engineering practices. Every one of those is a stated rubric line — none of them are optional.

## Fixed architecture decisions — do not relitigate these

- **Three signals:** item-item collaborative filtering (primary), content-based embedding similarity (cold-start fallback), skin-profile affinity (beauty-specific differentiator). All three ship.
- **Fusion is a weighted score blend**, not Reciprocal Rank Fusion. The author's other project already uses RRF; a normalized score blend is the right tool here and the distinction is deliberate.
- **CF persists top-K neighbors per item (K≈50), never a dense N×N matrix.** This is a hard requirement — it's both the Amazon-paper-correct approach and what keeps the app inside Streamlit's free-tier memory limit.
- **All recommendation logic lives in a UI-agnostic `recsys/` package.** Streamlit imports it. Never put business logic in the Streamlit file.
- **Everything expensive is precomputed offline.** The deployed app loads small artifacts and does lookups. It must never compute embeddings or similarity matrices at request time.
- **Streamlit for the UI and deployment.** Not React, not Gradio.
- **A thin FastAPI layer over the same `recsys/` package is a stretch goal**, attempted only after the Streamlit app is deployed and working. Never at the cost of the core.

## Non-negotiables

**The evaluation must include random and popularity baselines.** A hybrid recommender that isn't compared against "just recommend popular items" is an unevaluated recommender. Report the real numbers even if the margin over popularity is small — do not tune the evaluation to produce a flattering result, and do not quietly drop a baseline that looks too strong.

**Never fabricate or estimate a metric.** Every number in the README, the UI, or any document must come from an actual run of the evaluation harness. If something hasn't been measured yet, leave a visible placeholder and say so.

**Every recommendation surfaced in the UI must show why it was recommended** — which signal drove it and the supporting evidence (e.g. "users who rated X highly also rated this 4.5★", "matches your skin type: 4.6★ from 210 similar reviewers", "similar ingredients: niacinamide, hyaluronic acid"). The assignment explicitly requires reasoning in the output.

**Exclude already-rated/already-owned items from recommendations.** Both in the app and in the evaluation harness. Recommending someone the thing they just reviewed is the classic recommender bug.

**Handle cold start explicitly.** A new user with no history and a product with too few reviews both need defined, tested behavior — not a crash and not an empty list.

## Data

Kaggle "Sephora Products and Skincare Reviews" (~8k products, ~1M reviews across multiple CSVs).

- Fetched via the **Kaggle API**, not manual download. Keep it that way: a scripted fetch (`scripts/fetch_data.py`) is part of the reproducibility story the assignment grades. An evaluator should be able to clone the repo, run the fetch script, run the build script, and have a working app.
- Raw CSVs are **too large for GitHub** (100MB/file limit). `data/` must be gitignored. Commit the small precomputed artifacts plus the scripts that fetch the raw data and regenerate the artifacts.
- Expected review fields (**verify before relying on any of them**): `author_id`, `rating`, `product_id`, `skin_tone`, `skin_type`, `hair_color`, `eye_color`, `submission_time`, `review_text`, `is_recommended`.
- If `author_id` turns out to be missing or unusable, **stop and report** rather than silently substituting a different approach — collaborative filtering is the centerpiece and its absence changes the whole plan.

## Conventions

```
data/            raw CSVs — GITIGNORED, fetched via Kaggle API
recsys/          core package — UI-agnostic, importable, testable
  data.py        loading, cleaning, filtering
  matrix.py      sparse user×item construction
  cf.py          item-item collaborative filtering
  content.py     embeddings + content similarity
  profile.py     skin-profile affinity aggregates
  blend.py       score fusion + final ranking
  recommend.py   public API: recommend_by_item / recommend_by_user / recommend_by_profile
scripts/
  fetch_data.py        Kaggle API download → data/
  build_artifacts.py   offline precompute → artifacts/
eval/
  run_eval.py    leave-one-out harness, all five systems
artifacts/       small committed .npz / .parquet files
app.py           Streamlit UI — presentation only, no logic
```

Type hints on public functions. Docstrings that explain *why*, not just *what* — this codebase will be read by an evaluator, and design reasoning visible in the code is part of what's being graded.

## Working style

Work phase by phase. After each phase, **stop and summarize what you built and what you decided**, then wait. Do not run the whole project end to end unattended — the author has to defend every decision in an interview, so they need to follow the reasoning as it happens, not read a finished repo afterward.

Flag tradeoffs out loud when you hit them rather than silently picking. When something is genuinely uncertain or you had to guess, say so explicitly instead of presenting it as settled.

Priority order when time gets tight: **working deployed demo > honest evaluation > documentation > FastAPI layer > matrix factorization.** Cut from the bottom.
