from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from retriever import run_hybrid_rankings as ranking_backend


CONTENT_FEATURES_DIR = ranking_backend.REPO_ROOT / "data" / "content_features"
CONTENT_FEATURES_PARQUET_FILE = CONTENT_FEATURES_DIR / "content_features.parquet"
CONTENT_FEATURES_CSV_FILE = CONTENT_FEATURES_DIR / "content_features.csv"
CONTENT_FEATURES_CONFIG_FILE = CONTENT_FEATURES_DIR / "content_features_config.json"

FEATURE_METRIC_GROUPS = [
    {
        "title": "Document Size",
        "metrics": [
            ("char_count", "Characters", "integer"),
            ("non_whitespace_char_count", "Non-whitespace characters", "integer"),
            ("word_count", "Words", "integer"),
            ("token_count", "Retriever tokens", "integer"),
            ("sentence_count", "Sentences", "integer"),
            ("entity_count", "Entity occurrences", "integer"),
        ],
    },
    {
        "title": "Lexical And Surface",
        "metrics": [
            ("avg_words_per_sentence", "Average words per sentence", "decimal"),
            ("avg_token_length", "Average retriever token length", "decimal"),
            ("lexical_density", "Lexical density", "decimal"),
            ("lexical_diversity", "Lexical diversity", "decimal"),
        ],
    },
    {
        "title": "Readability And Complexity",
        "metrics": [
            ("syllable_count", "Syllables", "integer"),
            ("avg_syllables_per_word", "Average syllables per word", "decimal"),
            ("flesch_kincaid_grade", "Flesch-Kincaid grade", "decimal"),
            ("gunning_fog_index", "Gunning Fog index", "decimal"),
        ],
    },
    {
        "title": "Sentiment",
        "metrics": [
            ("sentiment_polarity", "Sentiment polarity", "decimal"),
            ("sentiment_subjectivity", "Sentiment subjectivity", "decimal"),
        ],
    },
]


def _validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _file_name_from_source_path(source_path: Any) -> str:
    normalized = str(source_path or "").replace("\\", "/")
    return normalized.rsplit("/", maxsplit=1)[-1]


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


@lru_cache(maxsize=1)
def load_content_features() -> pd.DataFrame:
    if CONTENT_FEATURES_PARQUET_FILE.exists():
        features = pd.read_parquet(CONTENT_FEATURES_PARQUET_FILE)
    elif CONTENT_FEATURES_CSV_FILE.exists():
        features = pd.read_csv(CONTENT_FEATURES_CSV_FILE)
    else:
        raise FileNotFoundError(
            "Missing content feature table. Expected either "
            f"{CONTENT_FEATURES_PARQUET_FILE} or {CONTENT_FEATURES_CSV_FILE}."
        )

    required_columns = {"doc_id", "title", "source_path"}
    missing_columns = required_columns.difference(features.columns)
    if missing_columns:
        raise ValueError(
            "Content feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    features = features.copy()
    features["file_name"] = features["source_path"].map(_file_name_from_source_path)
    features = features.rename(
        columns={
            "doc_id": "feature_doc_id",
            "title": "feature_title",
        }
    )
    return features


@lru_cache(maxsize=1)
def load_article_catalog() -> pd.DataFrame:
    _validate_file_exists(ranking_backend.BM25_METADATA_FILE, "BM25 document metadata file")
    metadata = pd.read_parquet(
        ranking_backend.BM25_METADATA_FILE,
        columns=["page_id", "title", "file_name"],
    )

    graph_nodes = pd.DataFrame(columns=["page_id", "article_url"])
    if ranking_backend.GRAPH_NODES_FILE.exists():
        graph_nodes = pd.read_csv(
            ranking_backend.GRAPH_NODES_FILE,
            usecols=["page_id", "url"],
        ).rename(columns={"url": "article_url"})
        graph_nodes = graph_nodes.drop_duplicates(subset=["page_id"])

    catalog = metadata.merge(graph_nodes, on="page_id", how="left")
    catalog["page_id"] = catalog["page_id"].astype("int64")
    catalog["page_id_text"] = catalog["page_id"].astype(str)
    catalog["title_search"] = catalog["title"].astype(str).str.casefold()
    return catalog


@lru_cache(maxsize=1)
def load_enriched_content_features() -> pd.DataFrame:
    features = load_content_features()
    catalog = load_article_catalog().drop(columns=["page_id_text", "title_search"])
    enriched = features.merge(catalog, on="file_name", how="left", validate="one_to_one")

    missing_page_ids = int(enriched["page_id"].isna().sum())
    if missing_page_ids:
        raise ValueError(
            f"Could not map {missing_page_ids:,} content feature rows to page IDs by filename."
        )

    enriched["page_id"] = enriched["page_id"].astype("int64")
    return enriched


def content_feature_document_count() -> int:
    if CONTENT_FEATURES_CONFIG_FILE.exists():
        try:
            config = json.loads(CONTENT_FEATURES_CONFIG_FILE.read_text(encoding="utf-8"))
            count = int(config.get("processed_document_count") or 0)
            if count > 0:
                return count
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return int(len(load_content_features()))


def search_articles(search_text: str = "", limit: int = 10) -> list[dict[str, Any]]:
    query = str(search_text or "").strip()
    if not query:
        return []

    catalog = load_article_catalog()
    query_casefold = query.casefold()
    page_id_match = catalog["page_id_text"].str.contains(query, regex=False)
    title_match = catalog["title_search"].str.contains(query_casefold, regex=False)
    matches = catalog.loc[page_id_match | title_match].copy()

    if matches.empty:
        return []

    matches["match_priority"] = 4
    matches.loc[matches["title_search"].str.startswith(query_casefold), "match_priority"] = 2
    matches.loc[matches["page_id_text"].str.startswith(query), "match_priority"] = 1
    matches.loc[matches["title_search"] == query_casefold, "match_priority"] = 0
    matches.loc[matches["page_id_text"] == query, "match_priority"] = 0
    matches = matches.sort_values(
        ["match_priority", "title_search", "page_id"],
        ascending=[True, True, True],
    ).head(max(int(limit), 0))

    return [
        {
            "page_id": int(row.page_id),
            "title": str(row.title),
            "file_name": str(row.file_name),
            "article_url": _clean_scalar(row.article_url),
        }
        for row in matches.itertuples(index=False)
    ]


def resolve_article(search_text: str) -> dict[str, Any] | None:
    query = str(search_text or "").strip()
    if not query:
        return None

    catalog = load_article_catalog()
    if query.isdigit():
        page_id_matches = catalog.loc[catalog["page_id_text"] == query]
        if not page_id_matches.empty:
            row = page_id_matches.iloc[0]
            return {
                "page_id": int(row["page_id"]),
                "title": str(row["title"]),
            }

    exact_title_matches = catalog.loc[catalog["title_search"] == query.casefold()]
    if not exact_title_matches.empty:
        row = exact_title_matches.iloc[0]
        return {
            "page_id": int(row["page_id"]),
            "title": str(row["title"]),
        }

    suggestions = search_articles(query, limit=1)
    if not suggestions:
        return None
    return {
        "page_id": int(suggestions[0]["page_id"]),
        "title": str(suggestions[0]["title"]),
    }


def get_article_features(page_id: int) -> dict[str, Any]:
    enriched = load_enriched_content_features()
    matches = enriched.loc[enriched["page_id"] == int(page_id)]
    if matches.empty:
        raise FileNotFoundError(f"No content features found for page_id {page_id}.")

    record = {
        key: _clean_scalar(value)
        for key, value in matches.iloc[0].to_dict().items()
    }
    record["top_entities"] = _parse_json_list(record.get("top_entities_json"))
    record["entities"] = _parse_json_list(record.get("entities_json"))
    record["unique_entity_count"] = len(record["entities"])

    metric_groups: list[dict[str, Any]] = []
    for group in FEATURE_METRIC_GROUPS:
        metrics = [
            {
                "key": key,
                "label": label,
                "format": value_format,
                "value": record.get(key),
            }
            for key, label, value_format in group["metrics"]
        ]
        metric_groups.append({"title": group["title"], "metrics": metrics})
    record["metric_groups"] = metric_groups
    return record
