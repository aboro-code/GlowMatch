"""Is the hybrid's accuracy edge over CF-only real, or sampling noise?

Leave-one-out gives each user a binary outcome — the held-out item was in
the top-10 or it wasn't — for every system, on the SAME users. That pairing
is what makes a proper test possible: comparing two independent proportions
would throw away the fact that both systems succeeded or failed on the same
person, which is most of the available information.

Two complementary tests:

- McNemar, the standard test for paired binary outcomes. It looks only at
  the disagreements: users the hybrid hit and CF missed (b), versus users CF
  hit and the hybrid missed (c). Users where both agree carry no information
  about which is better, so they are correctly ignored. Exact binomial
  version, since b + c can be small.

- Paired bootstrap over users, which makes no distributional assumption and
  yields a confidence interval on the difference rather than just a p-value.
  Resampling users (not observations) preserves the pairing.

A difference of ~0.003 at n=5,000 sits near the standard error of roughly
0.008, so the expected answer is "not distinguishable". Reporting that
honestly is the point: an accuracy claim the data cannot support should not
appear in the documentation, and the hybrid's real case rests on catalog
coverage, which is a large and unambiguous effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import run_eval  # noqa: E402

N_BOOTSTRAP = 10000
SEEDS = (42, 7, 123)


def hit_indicators(
    ranked: pd.DataFrame, holdout: pd.DataFrame, sampled_users: np.ndarray
) -> pd.Series:
    """1 if the user's held-out item appears in their top-N, else 0.
    Indexed by author_id over the full sampled cohort, so a user the system
    produced no candidates for counts as a miss rather than vanishing."""
    truth = holdout[["author_id", "product_id"]].rename(columns={"product_id": "true_id"})
    merged = ranked.merge(truth, on="author_id")
    hit_users = merged.loc[merged["product_id"] == merged["true_id"], "author_id"].unique()
    return pd.Series(
        np.isin(sampled_users, hit_users).astype(int), index=sampled_users
    )


def mcnemar_exact(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test on two paired binary outcome vectors."""
    both = ((a == 1) & (b == 1)).sum()
    only_a = ((a == 1) & (b == 0)).sum()
    only_b = ((a == 0) & (b == 1)).sum()
    neither = ((a == 0) & (b == 0)).sum()

    n_discordant = only_a + only_b
    # Two-sided exact binomial on the discordant pairs under H0: p = 0.5
    p_value = (
        float(stats.binomtest(int(only_a), int(n_discordant), 0.5).pvalue)
        if n_discordant > 0
        else 1.0
    )
    return {
        "both_hit": int(both),
        "only_first": int(only_a),
        "only_second": int(only_b),
        "neither": int(neither),
        "discordant": int(n_discordant),
        "p_value": p_value,
    }


def paired_bootstrap(
    a: pd.Series, b: pd.Series, n_boot: int = N_BOOTSTRAP, seed: int = 0
) -> dict:
    """Bootstrap CI for mean(a) - mean(b), resampling users with replacement."""
    rng = np.random.default_rng(seed)
    av, bv = a.to_numpy(), b.to_numpy()
    n = len(av)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = av[idx].mean(axis=1) - bv[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "observed_diff": float(av.mean() - bv.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


def _systems_for(ctx: run_eval.EvalContext) -> dict[str, pd.DataFrame]:
    """Top-N per system, sharing candidate generation and exclusion."""
    import blend

    norm = ctx.normalized
    scored = {
        "cf": norm["cf_norm"],
        "content": norm["content_norm"],
        "profile": norm["profile_norm"],
        "hybrid": (
            blend.W_CF * norm["cf_norm"]
            + blend.W_CONTENT * norm["content_norm"]
            + blend.W_PROFILE * norm["profile_norm"]
        ),
    }
    out = {}
    for name, score in scored.items():
        frame = norm[["author_id", "product_id"]].copy()
        frame["score"] = score
        out[name] = run_eval._rank_and_truncate(
            frame, ctx.already_rated, "score", run_eval.TOP_N
        )
    return out


def run(n_users: int = run_eval.N_EVAL_USERS, seeds: tuple[int, ...] = SEEDS) -> None:
    per_seed_rows = []

    for seed in seeds:
        print(f"\n{'=' * 74}\nSEED {seed}\n{'=' * 74}")
        ctx = run_eval.prepare_eval_context(n_users=n_users, seed=seed, verbose=False)
        tops = _systems_for(ctx)
        ind = {
            name: hit_indicators(top, ctx.holdout, ctx.sampled_users)
            for name, top in tops.items()
        }

        print(f"  {'system':<10}{'HitRate@10':>12}")
        for name, s in ind.items():
            print(f"  {name:<10}{s.mean():>12.4f}")

        mc = mcnemar_exact(ind["hybrid"], ind["cf"])
        bs = paired_bootstrap(ind["hybrid"], ind["cf"], seed=seed)

        print("\n  hybrid vs cf (paired, same users)")
        print(f"    both hit: {mc['both_hit']}   hybrid only: {mc['only_first']}   "
              f"cf only: {mc['only_second']}   neither: {mc['neither']}")
        print(f"    McNemar exact p = {mc['p_value']:.4f}  (discordant n = {mc['discordant']})")
        print(f"    bootstrap diff  = {bs['observed_diff']:+.4f}  "
              f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]  p = {bs['p_two_sided']:.4f}")

        verdict = "SIGNIFICANT" if mc["p_value"] < 0.05 else "not significant"
        print(f"    -> {verdict} at alpha=0.05")

        per_seed_rows.append(
            {
                "seed": seed,
                "cf": ind["cf"].mean(),
                "hybrid": ind["hybrid"].mean(),
                "diff": bs["observed_diff"],
                "mcnemar_p": mc["p_value"],
                "ci_low": bs["ci_low"],
                "ci_high": bs["ci_high"],
            }
        )

    print(f"\n{'=' * 74}\nACROSS SEEDS\n{'=' * 74}")
    df = pd.DataFrame(per_seed_rows)
    print(df.to_string(index=False))

    n_sig = int((df["mcnemar_p"] < 0.05).sum())
    print(f"\n  seeds where hybrid beats cf significantly: {n_sig} of {len(df)}")
    print(f"  mean difference across seeds: {df['diff'].mean():+.4f}")
    print(f"  between-seed spread of the difference: "
          f"{df['diff'].max() - df['diff'].min():.4f}")

    out_path = Path(__file__).resolve().parent / "significance_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
