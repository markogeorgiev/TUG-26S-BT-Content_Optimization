"""
Significance testing of per-query Spearman correlations, per retrieval model.

For each model we already have, per query, the Spearman correlation between every
content feature and document rank (see per_query_spearmans.py). With 30 queries
that gives, per feature, a sample of 30 correlation coefficients. This script
asks, for each feature: is that correlation reliably non-zero across queries?

Test
----
Per model, per feature:
  1. Drop NaN cells -> array of per-query coefficients.
  2. If fewer than 3 valid values remain -> untestable (NaN stats, no test).
  3. Otherwise:
       n_queries = count of valid values
       mean_rho  = mean of coefficients          (the EFFECT SIZE)
       std_rho   = sample std, ddof=1
       se        = std_rho / sqrt(n)
       t, p_value = scipy.stats.ttest_1samp(values, popmean=0.0)
Then Benjamini-Hochberg FDR correction (statsmodels, method="fdr_bh",
alpha=0.05) across the testable features of that model -> p_corrected,
significant.

Inputs  : rank_relationship_analysis/per_query_spearman/spearman_<model>.csv
          (wide: query_id column + one column per feature)
Outputs : rank_relationship_analysis/significance/significance_<model>.csv
          rank_relationship_analysis/significance/significance_all_models.csv

Usage
-----
    python significance_testing.py
    python significance_testing.py --in-dir <dir> --out-dir <dir> --alpha 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "rank_relationship_analysis"

# Output label -> per-query Spearman filename. (file token, display name)
MODELS: Dict[str, Tuple[str, str]] = {
    "bm25": ("BM25", "spearman_bm25.csv"),
    "tfidf": ("TF-IDF", "spearman_tfidf.csv"),
    "sbert": ("SBERT", "spearman_sbert.csv"),
    "e5": ("E5", "spearman_e5.csv"),
}

# Columns that are not features in the per-query Spearman files.
NON_FEATURE_COLUMNS = {"query_id"}

# Output column order (exact).
OUTPUT_COLUMNS = [
    "model", "feature", "n_queries", "mean_rho", "std_rho", "se",
    "t", "p_value", "p_corrected", "significant",
]
FLOAT_COLUMNS = ["mean_rho", "std_rho", "se", "t", "p_value", "p_corrected"]

MIN_VALID = 3  # need at least this many per-query coefficients to test


def test_model(
    spearman: pd.DataFrame,
    model_name: str,
    alpha: float,
) -> pd.DataFrame:
    """Run the one-sample t-test (vs 0) for every feature, then BH-correct."""
    feature_cols = [c for c in spearman.columns if c not in NON_FEATURE_COLUMNS]

    records: List[dict] = []
    for feature in feature_cols:
        values = spearman[feature].dropna().to_numpy(dtype=float)
        n = values.size

        if n < MIN_VALID:
            records.append({
                "model": model_name, "feature": feature, "n_queries": n,
                "mean_rho": np.nan, "std_rho": np.nan, "se": np.nan,
                "t": np.nan, "p_value": np.nan,
            })
            continue

        mean_rho = float(np.mean(values))
        std_rho = float(np.std(values, ddof=1))
        se = std_rho / np.sqrt(n)
        t_stat, p_value = stats.ttest_1samp(values, popmean=0.0)

        records.append({
            "model": model_name, "feature": feature, "n_queries": n,
            "mean_rho": mean_rho, "std_rho": std_rho, "se": se,
            "t": float(t_stat), "p_value": float(p_value),
        })

    df = pd.DataFrame.from_records(records)

    # Benjamini-Hochberg across the testable features only.
    df["p_corrected"] = np.nan
    df["significant"] = False
    testable = df["p_value"].notna()
    if testable.any():
        reject, p_corr, _, _ = multipletests(
            df.loc[testable, "p_value"].to_numpy(),
            alpha=alpha,
            method="fdr_bh",
        )
        df.loc[testable, "p_corrected"] = p_corr
        df.loc[testable, "significant"] = reject

    df = df[OUTPUT_COLUMNS].sort_values(
        "p_value", ascending=True, na_position="last"
    ).reset_index(drop=True)
    return df


def round_floats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[FLOAT_COLUMNS] = out[FLOAT_COLUMNS].round(4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir", type=Path,
        default=ANALYSIS_DIR / "per_query_spearman",
        help="directory with spearman_<model>.csv files",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ANALYSIS_DIR / "significance",
        help="output directory for significance_<model>.csv files",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="FDR alpha for Benjamini-Hochberg (default: 0.05)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_models: List[pd.DataFrame] = []
    for token, (model_name, filename) in MODELS.items():
        path = args.in_dir / filename
        if not path.exists():
            print(f"[{model_name}] SKIP - {path} not found")
            continue

        spearman = pd.read_csv(path)
        result = test_model(spearman, model_name, alpha=args.alpha)
        result = round_floats(result)

        out_path = args.out_dir / f"significance_{token}.csv"
        result.to_csv(out_path, index=False)
        all_models.append(result)

        print(f"\n===== {model_name}  ({path.name}) =====")
        print(result.to_string(index=False))
        n_sig = int(result["significant"].sum())
        print(f"-> {n_sig}/{len(result)} features significant at FDR "
              f"alpha={args.alpha}; wrote {out_path}")

    if all_models:
        combined = pd.concat(all_models, ignore_index=True)
        combined_path = args.out_dir / "significance_all_models.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n[combined] wrote {len(combined)} rows -> {combined_path}")
    else:
        print("\n[warn] no model files found; nothing written.")


if __name__ == "__main__":
    main()
