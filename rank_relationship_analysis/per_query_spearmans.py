"""
Per-query Spearman correlation between each content feature and document rank.

For every query and every retrieval model, this computes Spearman's rho between
each numeric content feature (from feature_extraction.py) and the document's rank
in that model's ranking. Results are written to one file per model.

How to read the sign
--------------------
Rank 1 is the best document and rank numbers grow as relevance drops. So a
NEGATIVE rho means "higher feature value -> better (smaller) rank number", i.e.
the feature is associated with ranking *higher*. A POSITIVE rho means the feature
is associated with ranking *lower*. (This is inherent to correlating against rank;
nothing is flipped for you.)

Inputs
------
  * features : output/content_features.csv   (doc_id + feature columns)
  * rankings : rankings/full_retriever/rankings_{bm25,tfidf,sbert,e5}.csv
               columns: query_id, doc_id, doc_rank, doc_score, model

Output
------
  output/per_query_spearman/spearman_{bm25,tfidf,sbert,e5}.csv
  Wide table: one row per query_id, one column per feature, cell = Spearman rho.

Usage
-----
    python.exe per_query_spearmans.py
    python.exe per_query_spearmans.py --features output/content_features.csv \
        --rankings-dir rankings/full_retriever --out-dir output/per_query_spearman
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model label -> ranking filename. Hybrid is deliberately excluded.
MODEL_FILES: Dict[str, str] = {
    "bm25": "rankings_bm25.csv",
    "tfidf": "rankings_tfidf.csv",
    "sbert": "rankings_sbert.csv",
    "e5": "rankings_e5.csv",
}

# Columns in the feature file that are identifiers/labels, not features.
NON_FEATURE_COLUMNS = {"doc_id"}


def numeric_feature_columns(features: pd.DataFrame) -> List[str]:
    """Numeric feature columns, in their original file order.

    Drops the id column and any non-numeric column (e.g. ``most_frequent_entity``,
    which is a string). Columns that exist but are entirely empty (e.g.
    ``sentiment_polarity`` when features were built with --no-sentiment) are kept;
    they will simply yield NaN correlations.
    """
    numeric = features.select_dtypes(include="number").columns.tolist()
    return [c for c in features.columns
            if c in numeric and c not in NON_FEATURE_COLUMNS]


def per_query_spearman(
    ranking: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Wide table: index = query_id, columns = features, cell = Spearman rho vs rank."""
    merged = ranking.merge(features, on="doc_id", how="inner")

    rows: Dict[str, Dict[str, float]] = {}
    for query_id, group in merged.groupby("query_id", sort=True):
        # Spearman of every feature against doc_rank in one shot, then keep the
        # doc_rank column (its correlation with each feature).
        corr = group[feature_cols + ["doc_rank"]].corr(method="spearman")
        rows[query_id] = corr["doc_rank"].drop("doc_rank").to_dict()

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "query_id"
    # Preserve the feature ordering from the feature file.
    return result[feature_cols].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "content_features.csv",
        help="content feature CSV (default: output/content_features.csv)",
    )
    parser.add_argument(
        "--rankings-dir", type=Path,
        default=PROJECT_ROOT / "rankings" / "full_retriever",
        help="directory holding rankings_<model>.csv (default: rankings/full_retriever)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "per_query_spearman",
        help="output directory (default: output/per_query_spearman)",
    )
    args = parser.parse_args()

    print(f"[features] loading {args.features}")
    features = pd.read_csv(args.features)
    feature_cols = numeric_feature_columns(features)
    features = features[["doc_id"] + feature_cols]
    print(f"[features] {len(features)} docs, {len(feature_cols)} numeric features:")
    print("           " + ", ".join(feature_cols))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for label, filename in MODEL_FILES.items():
        ranking_path = args.rankings_dir / filename
        if not ranking_path.exists():
            print(f"[{label}] SKIP - {ranking_path} not found")
            continue

        print(f"[{label}] loading {ranking_path}")
        ranking = pd.read_csv(ranking_path, usecols=["query_id", "doc_id", "doc_rank"])

        n_queries = ranking["query_id"].nunique()
        print(f"[{label}] {n_queries} queries; computing per-query Spearman")
        result = per_query_spearman(ranking, features, feature_cols)

        out_path = args.out_dir / f"spearman_{label}.csv"
        result.to_csv(out_path)
        print(f"[{label}] wrote {result.shape[0]} queries x "
              f"{result.shape[1]} features -> {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
