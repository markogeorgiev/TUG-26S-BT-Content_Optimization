from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from retriever import run_hybrid_rankings as ranking_backend


DISPLAY_LIMIT_OPTIONS = [10, 25, 50, 100, 250, 500, 1000]
DEFAULT_DISPLAY_LIMIT = 25
DEFAULT_PER_MODEL_KEY = "bm25"

PER_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "bm25": {
        "label": "BM25 only",
        "score_label": "BM25 normalized score",
        "summary": "Ranks documents by lexical BM25 matching only.",
        "raw_score_column": "bm25_score_raw",
        "score_column": "bm25_score_norm",
    },
    "sbert": {
        "label": "SBERT only",
        "score_label": "SBERT normalized score",
        "summary": "Ranks documents by semantic SBERT similarity only.",
        "raw_score_column": "semantic_score_raw",
        "score_column": "semantic_score_norm",
    },
    "pagerank": {
        "label": "PageRank only",
        "score_label": "PageRank normalized score",
        "summary": "Ranks documents by graph authority only.",
        "raw_score_column": "pagerank",
        "score_column": "pagerank_norm",
    },
}


def _record_result_count(record: dict[str, Any]) -> int:
    return int(record.get("stored_result_count") or record.get("result_count") or 0)


def _record_is_full_ranking(record: dict[str, Any]) -> bool:
    if bool(record.get("stored_full_ranking")):
        return True
    return _record_result_count(record) >= ranking_backend.total_document_count()


def format_query_record(record: dict[str, Any]) -> dict[str, Any]:
    query_text = str(record.get("query_text") or "")
    return {
        "query_id": str(record.get("query_id") or ""),
        "query_text": query_text,
        "query_slug": str(record.get("query_slug") or ""),
        "query_key": str(record.get("query_key") or ranking_backend.query_identity_key(query_text)),
        "executed_at_utc": str(record.get("executed_at_utc") or ""),
        "result_file": str(record.get("result_file") or ""),
        "stored_result_count": _record_result_count(record),
        "is_full_ranking": _record_is_full_ranking(record),
        "weights": dict(record.get("weights") or {}),
    }


def recent_queries(limit: int = 12) -> list[dict[str, Any]]:
    records = ranking_backend.iter_saved_query_records()
    records = sorted(
        records,
        key=lambda record: (
            str(record.get("executed_at_utc") or ""),
            str(record.get("query_id") or ""),
        ),
        reverse=True,
    )
    return [format_query_record(record) for record in records[: max(limit, 0)]]


def autocomplete_queries(prefix: str = "", limit: int = 8) -> list[dict[str, Any]]:
    return [format_query_record(record) for record in ranking_backend.get_query_suggestions(prefix, limit)]


def available_display_limits(total_results: int) -> list[int]:
    if total_results <= 0:
        return DISPLAY_LIMIT_OPTIONS[:4]
    options = [limit for limit in DISPLAY_LIMIT_OPTIONS if limit < total_results]
    options.append(total_results)
    return sorted(set(options)) or [10]


def normalize_display_limit(display_limit: int | str | None, total_results: int) -> int:
    try:
        parsed = int(display_limit) if display_limit is not None else DEFAULT_DISPLAY_LIMIT
    except (TypeError, ValueError):
        parsed = DEFAULT_DISPLAY_LIMIT

    if parsed <= 0:
        parsed = DEFAULT_DISPLAY_LIMIT

    if total_results <= 0:
        return 0

    return min(parsed, total_results)


def normalize_per_model_key(model_key: str | None) -> str:
    normalized = str(model_key or "").strip().lower()
    if normalized in PER_MODEL_CONFIGS:
        return normalized
    return DEFAULT_PER_MODEL_KEY


def available_per_model_options() -> list[dict[str, str]]:
    return [
        {
            "value": model_key,
            "label": config["label"],
            "summary": config["summary"],
        }
        for model_key, config in PER_MODEL_CONFIGS.items()
    ]


def get_per_model_config(model_key: str | None) -> dict[str, str]:
    normalized = normalize_per_model_key(model_key)
    return {
        "value": normalized,
        **PER_MODEL_CONFIGS[normalized],
    }


def execute_or_load_query(query: str) -> tuple[dict[str, Any], pd.DataFrame, bool]:
    record, ranking_df, created = ranking_backend.get_or_create_query_ranking(query)
    return format_query_record(record), ranking_df, created


def get_query_record(query_id: str) -> dict[str, Any]:
    record = ranking_backend.get_query_record_by_id(query_id)
    if record is None:
        raise FileNotFoundError(f"Unknown query_id: {query_id}")
    return format_query_record(record)


@lru_cache(maxsize=8)
def load_full_ranking(query_id: str) -> pd.DataFrame:
    _, ranking_df = ranking_backend.load_ranking_by_query_id(query_id)
    return ranking_df


@lru_cache(maxsize=1)
def load_page_urls() -> pd.DataFrame:
    graph_nodes_file = ranking_backend.GRAPH_NODES_FILE
    ranking_backend.validate_file_exists(graph_nodes_file, "graph nodes file")
    return pd.read_csv(graph_nodes_file, usecols=["page_id", "url"]).drop_duplicates(subset=["page_id"])


def attach_page_urls(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        enriched = results_df.copy()
        enriched["article_url"] = pd.Series(dtype="object")
        return enriched

    page_urls = load_page_urls()
    enriched = results_df.merge(page_urls, on="page_id", how="left")
    enriched = enriched.rename(columns={"url": "article_url"})
    return enriched


def _coerce_numeric_series(results_df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in results_df.columns:
        raise ValueError(f"Saved ranking is missing the required column: {column_name}")
    return pd.to_numeric(results_df[column_name], errors="coerce")


def build_per_model_ranking(results_df: pd.DataFrame, model_key: str | None) -> tuple[pd.DataFrame, dict[str, str]]:
    model_config = get_per_model_config(model_key)
    reranked = results_df.copy()

    raw_score = _coerce_numeric_series(reranked, model_config["raw_score_column"]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    normalized_score = _coerce_numeric_series(reranked, model_config["score_column"]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    hybrid_rank = _coerce_numeric_series(reranked, "rank").replace([np.inf, -np.inf], np.nan)

    reranked["_model_score_raw"] = raw_score.fillna(float("-inf"))
    reranked["_model_score"] = normalized_score.fillna(float("-inf"))
    reranked["hybrid_rank"] = hybrid_rank.fillna(len(reranked) + 1).astype("int64")

    reranked = reranked.sort_values(
        ["_model_score_raw", "_model_score", "hybrid_rank", "title"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    reranked.insert(0, "model_rank", np.arange(1, len(reranked) + 1))
    reranked["rank_change_vs_hybrid"] = reranked["hybrid_rank"] - reranked["model_rank"]
    reranked["model_score"] = reranked["_model_score"].replace(float("-inf"), np.nan)

    reranked = reranked.drop(columns=["_model_score_raw", "_model_score"])
    return reranked, model_config


def load_query_results(query_id: str, display_limit: int | str | None) -> tuple[dict[str, Any], pd.DataFrame, int]:
    record = get_query_record(query_id)
    ranking_df = load_full_ranking(query_id)
    total_results = int(len(ranking_df))
    display_limit_value = normalize_display_limit(display_limit, total_results)
    visible_results = ranking_df.head(display_limit_value).copy()
    visible_results = attach_page_urls(visible_results)
    return record, visible_results, total_results


def load_per_model_query_results(
    query_id: str,
    model_key: str | None,
    display_limit: int | str | None,
) -> tuple[dict[str, Any], pd.DataFrame, int, dict[str, str]]:
    record = get_query_record(query_id)
    ranking_df = load_full_ranking(query_id)
    reranked, model_config = build_per_model_ranking(ranking_df, model_key)
    total_results = int(len(reranked))
    display_limit_value = normalize_display_limit(display_limit, total_results)
    visible_results = reranked.head(display_limit_value).copy()
    visible_results = attach_page_urls(visible_results)
    return record, visible_results, total_results, model_config
