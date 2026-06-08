from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from retriever import run_hybrid_rankings as ranking_backend
from ui.services.ranking_ui_service import build_per_model_ranking


CONTENT_FEATURES_DIR = ranking_backend.REPO_ROOT / "data" / "content_features"
CONTENT_FEATURES_PARQUET_FILE = CONTENT_FEATURES_DIR / "content_features.parquet"
CONTENT_FEATURES_CSV_FILE = CONTENT_FEATURES_DIR / "content_features.csv"
CONTENT_FEATURES_CONFIG_FILE = CONTENT_FEATURES_DIR / "content_features_config.json"
RANK_FEATURE_PLOTS_DIR = ranking_backend.RANKINGS_DIR / "content_feature_rank_plots"
PER_MODEL_RANK_FEATURE_PLOTS_DIR = ranking_backend.RANKINGS_DIR / "per_model_rank_feature_plots"

RANK_FEATURE_PLOTS_SCHEMA_VERSION = 2
RANK_FEATURE_MAX_RANK = 100
RANK_FEATURE_ROLLING_WINDOW = 50
RANK_FEATURE_MIN_PERIODS = 10
MEDIAN_DROP_TOP_START = 1
MEDIAN_DROP_TOP_END = 25
MEDIAN_DROP_COMPARISON_SPAN = 25
MIN_RANK_FEATURE_TOP_K = 25
DEFAULT_RANK_FEATURE_PLOT_KIND = "ribbon"
RANK_FEATURE_PLOT_KINDS = [
    {
        "value": "ribbon",
        "label": "Ribbon plots",
        "description": "Per-feature median line with IQR and 10th-90th percentile bands.",
    },
    {
        "value": "overlay",
        "label": "Per-query overlay",
        "description": "Per-feature faint query-level rolling medians plus bold aggregate median.",
    },
    {
        "value": "heatmap",
        "label": "Feature consistency heatmap",
        "description": "One heatmap comparing normalized IQR width across four rank buckets.",
    },
    {
        "value": "violin",
        "label": "Violin plots",
        "description": "Per-feature distribution violins across the four selected top-k rank buckets.",
    },
]
PER_MODEL_RANK_FEATURE_OPTIONS = [
    {
        "value": "bm25",
        "label": "BM25 only",
        "description": "Use BM25-only ranking positions before plotting feature behavior.",
    },
    {
        "value": "sbert",
        "label": "SBERT only",
        "description": "Use SBERT-only ranking positions before plotting feature behavior.",
    },
]
PER_MODEL_RANK_FEATURE_SCORE_COLUMNS = {
    "bm25": ["bm25_score_raw", "bm25_score_norm"],
    "sbert": ["semantic_score_raw", "semantic_score_norm"],
}
STEELBLUE = "#4682b4"

IDENTITY_FEATURE_COLUMNS = {
    "feature_doc_id",
    "page_id",
}

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


def _record_result_count(record: dict[str, Any]) -> int:
    return int(record.get("stored_result_count") or record.get("result_count") or 0)


def _record_is_full_ranking(record: dict[str, Any]) -> bool:
    if bool(record.get("stored_full_ranking")):
        return True
    return _record_result_count(record) >= ranking_backend.total_document_count()


def _ranking_result_path(record: dict[str, Any]) -> Path:
    return ranking_backend.resolve_result_path(str(record.get("result_file") or ""))


def _normalize_plot_kind(plot_kind: str | None) -> str:
    normalized = str(plot_kind or DEFAULT_RANK_FEATURE_PLOT_KIND).strip().casefold()
    allowed = {option["value"] for option in RANK_FEATURE_PLOT_KINDS}
    if normalized not in allowed:
        raise ValueError(
            "Unknown plot kind. Expected one of: " + ", ".join(sorted(allowed))
        )
    return normalized


def _normalize_per_model_rank_feature_key(model_key: str | None) -> str:
    normalized = str(model_key or "").strip().lower()
    allowed = {option["value"] for option in PER_MODEL_RANK_FEATURE_OPTIONS}
    if normalized not in allowed:
        raise ValueError(
            "Unknown per-model rank feature mode. Expected one of: "
            + ", ".join(sorted(allowed))
        )
    return normalized


def normalize_rank_feature_top_k(top_k: int | str | None) -> int:
    max_available = ranking_backend.total_document_count()
    raw_value = str(top_k or "").strip()
    if not raw_value:
        return min(RANK_FEATURE_MAX_RANK, max_available)

    try:
        normalized = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Top-k must be an integer between {MIN_RANK_FEATURE_TOP_K} and {max_available}."
        ) from exc

    if normalized < MIN_RANK_FEATURE_TOP_K or normalized > max_available:
        raise ValueError(
            f"Top-k must be between {MIN_RANK_FEATURE_TOP_K} and {max_available}."
        )
    return normalized


def _safe_collection_key(
    query_ids: list[str],
    plot_kind: str,
    max_rank: int,
    extra_parts: list[str] | None = None,
) -> str:
    key_parts = [
        plot_kind,
        f"max_rank={max_rank}",
        f"window={RANK_FEATURE_ROLLING_WINDOW}",
        f"min_periods={RANK_FEATURE_MIN_PERIODS}",
        *(extra_parts or []),
        *query_ids,
    ]
    digest = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()[:16]
    return f"collection_{digest}"


def _validate_collection_key(collection_key: str) -> str:
    normalized = str(collection_key or "").strip()
    if not re.fullmatch(r"collection_[a-f0-9]{16}", normalized):
        raise FileNotFoundError(f"Unknown plot collection: {collection_key}")
    return normalized


def _validate_plot_file_name(file_name: str) -> str:
    normalized = Path(str(file_name or "")).name
    if not re.fullmatch(r"(plot|overlay|violin)_[A-Za-z0-9_]+\.png|heatmap\.png", normalized):
        raise FileNotFoundError(f"Unknown plot file: {file_name}")
    return normalized


def _relative_to_repo(path: Path) -> str:
    return ranking_backend.relative_to_repo(path)


def _format_query_record(record: dict[str, Any]) -> dict[str, Any]:
    query_text = str(record.get("query_text") or "")
    result_path = _ranking_result_path(record)
    return {
        "query_id": str(record.get("query_id") or ""),
        "query_text": query_text,
        "query_slug": str(record.get("query_slug") or ""),
        "query_key": str(record.get("query_key") or ranking_backend.query_identity_key(query_text)),
        "executed_at_utc": str(record.get("executed_at_utc") or ""),
        "result_file": str(record.get("result_file") or ""),
        "stored_result_count": _record_result_count(record),
        "is_full_ranking": _record_is_full_ranking(record),
        "has_result_file": result_path.exists(),
    }


def _query_records_by_id() -> dict[str, dict[str, Any]]:
    return {
        str(record.get("query_id") or ""): _format_query_record(record)
        for record in ranking_backend.iter_saved_query_records()
    }


def _normalize_query_ids(query_ids: list[str] | tuple[str, ...]) -> list[str]:
    records_by_id = _query_records_by_id()
    normalized = sorted(
        {
            str(query_id or "").strip()
            for query_id in query_ids
            if str(query_id or "").strip()
        }
    )
    if not normalized:
        raise ValueError("Select at least one saved query to generate rank-feature plots.")

    unknown_ids = [query_id for query_id in normalized if query_id not in records_by_id]
    if unknown_ids:
        raise ValueError("Unknown query IDs: " + ", ".join(unknown_ids))

    missing_files = [
        query_id
        for query_id in normalized
        if not bool(records_by_id[query_id]["has_result_file"])
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing ranking parquet files for query IDs: " + ", ".join(missing_files)
        )

    return normalized


def _plot_metadata_path(collection_key: str, base_dir: Path = RANK_FEATURE_PLOTS_DIR) -> Path:
    return base_dir / collection_key / "metadata.json"


def _plot_output_dir(collection_key: str, base_dir: Path = RANK_FEATURE_PLOTS_DIR) -> Path:
    return base_dir / collection_key


def _load_plot_metadata(
    collection_key: str,
    plot_kind: str,
    max_rank: int,
    base_dir: Path = RANK_FEATURE_PLOTS_DIR,
) -> dict[str, Any] | None:
    metadata_path = _plot_metadata_path(collection_key, base_dir=base_dir)
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None

    output_dir = _plot_output_dir(collection_key, base_dir=base_dir)
    plot_files = metadata.get("plot_files") or []
    if not plot_files:
        return None
    if int(metadata.get("schema_version") or 0) != RANK_FEATURE_PLOTS_SCHEMA_VERSION:
        return None
    if str(metadata.get("plot_kind") or "") != plot_kind:
        return None
    rank_window = metadata.get("rank_window") or {}
    if (
        int(rank_window.get("max_rank") or 0) != max_rank
        or int(rank_window.get("rolling_window") or 0) != RANK_FEATURE_ROLLING_WINDOW
        or int(rank_window.get("min_periods") or 0) != RANK_FEATURE_MIN_PERIODS
        or bool(rank_window.get("center")) is not True
    ):
        return None
    if not all((output_dir / str(plot_file.get("file_name") or "")).exists() for plot_file in plot_files):
        return None

    metadata["cache_status"] = "reused"
    return metadata


def _save_plot_metadata(
    collection_key: str,
    metadata: dict[str, Any],
    base_dir: Path = RANK_FEATURE_PLOTS_DIR,
) -> None:
    metadata_path = _plot_metadata_path(collection_key, base_dir=base_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def saved_ranking_query_options() -> list[dict[str, Any]]:
    records = [_format_query_record(record) for record in ranking_backend.iter_saved_query_records()]
    return sorted(records, key=lambda record: record["query_id"])


def rank_feature_plot_kind_options() -> list[dict[str, str]]:
    return [dict(option) for option in RANK_FEATURE_PLOT_KINDS]


def per_model_rank_feature_options() -> list[dict[str, str]]:
    return [dict(option) for option in PER_MODEL_RANK_FEATURE_OPTIONS]


def rank_feature_columns() -> list[str]:
    enriched = load_enriched_content_features()
    numeric_columns = enriched.select_dtypes(include=["number"]).columns
    return [
        str(column)
        for column in numeric_columns
        if str(column) not in IDENTITY_FEATURE_COLUMNS
        and enriched[str(column)].notna().any()
    ]


def _load_rank_rows(query_ids: list[str], max_rank: int) -> pd.DataFrame:
    records_by_id = _query_records_by_id()
    frames: list[pd.DataFrame] = []
    ranking_columns = ["rank", "page_id", "title"]

    for query_id in query_ids:
        record = records_by_id[query_id]
        result_path = _ranking_result_path(record)
        ranking_df = pd.read_parquet(result_path, columns=ranking_columns)
        ranking_df = ranking_df.loc[ranking_df["rank"].between(1, max_rank)].copy()
        ranking_df["query_id"] = query_id
        ranking_df["query_text"] = record["query_text"]
        frames.append(ranking_df)

    if not frames:
        return pd.DataFrame(columns=[*ranking_columns, "query_id", "query_text"])

    rank_rows = pd.concat(frames, ignore_index=True)
    rank_rows["page_id"] = rank_rows["page_id"].astype("int64")
    rank_rows["rank"] = rank_rows["rank"].astype("int64")
    return rank_rows


def _load_per_model_rank_rows(query_ids: list[str], model_key: str, max_rank: int) -> pd.DataFrame:
    normalized_model = _normalize_per_model_rank_feature_key(model_key)
    records_by_id = _query_records_by_id()
    ranking_columns = [
        "rank",
        "page_id",
        "title",
        *PER_MODEL_RANK_FEATURE_SCORE_COLUMNS[normalized_model],
    ]
    frames: list[pd.DataFrame] = []

    for query_id in query_ids:
        record = records_by_id[query_id]
        result_path = _ranking_result_path(record)
        ranking_df = pd.read_parquet(result_path, columns=ranking_columns)
        ranking_df["page_id"] = ranking_df["page_id"].astype("int64")
        reranked_df, _ = build_per_model_ranking(ranking_df, normalized_model)
        rank_rows = (
            reranked_df.loc[
                reranked_df["model_rank"].between(1, max_rank),
                ["model_rank", "page_id", "title"],
            ]
            .rename(columns={"model_rank": "rank"})
            .copy()
        )
        rank_rows["query_id"] = query_id
        rank_rows["query_text"] = record["query_text"]
        frames.append(rank_rows)

    if not frames:
        return pd.DataFrame(columns=["rank", "page_id", "title", "query_id", "query_text"])

    rank_rows = pd.concat(frames, ignore_index=True)
    rank_rows["page_id"] = rank_rows["page_id"].astype("int64")
    rank_rows["rank"] = rank_rows["rank"].astype("int64")
    return rank_rows


def load_rank_feature_frame(query_ids: list[str], max_rank: int = RANK_FEATURE_MAX_RANK) -> pd.DataFrame:
    normalized_query_ids = _normalize_query_ids(query_ids)
    normalized_max_rank = normalize_rank_feature_top_k(max_rank)
    rank_rows = _load_rank_rows(normalized_query_ids, max_rank=normalized_max_rank)
    if rank_rows.empty:
        raise ValueError("Selected rankings did not contain any rows to analyze.")

    feature_columns = rank_feature_columns()
    feature_lookup = load_enriched_content_features()[["page_id", *feature_columns]].copy()
    merged = rank_rows.merge(feature_lookup, on="page_id", how="left", validate="many_to_one")

    missing_features = int(merged[feature_columns].isna().all(axis=1).sum())
    if missing_features:
        raise ValueError(
            f"Could not map {missing_features:,} ranked articles to content features."
        )

    return merged.sort_values(["rank", "query_id", "page_id"]).reset_index(drop=True)


def load_per_model_rank_feature_frame(
    query_ids: list[str],
    model_key: str,
    max_rank: int = RANK_FEATURE_MAX_RANK,
) -> pd.DataFrame:
    normalized_query_ids = _normalize_query_ids(query_ids)
    normalized_max_rank = normalize_rank_feature_top_k(max_rank)
    rank_rows = _load_per_model_rank_rows(
        normalized_query_ids,
        model_key=model_key,
        max_rank=normalized_max_rank,
    )
    if rank_rows.empty:
        raise ValueError("Selected rankings did not contain any rows to analyze.")

    feature_columns = rank_feature_columns()
    feature_lookup = load_enriched_content_features()[["page_id", *feature_columns]].copy()
    merged = rank_rows.merge(feature_lookup, on="page_id", how="left", validate="many_to_one")

    missing_features = int(merged[feature_columns].isna().all(axis=1).sum())
    if missing_features:
        raise ValueError(
            f"Could not map {missing_features:,} ranked articles to content features."
        )

    return merged.sort_values(["rank", "query_id", "page_id"]).reset_index(drop=True)


def _rolling_rank_statistics(
    df: pd.DataFrame,
    feature: str,
    window: int,
    max_rank: int,
) -> pd.DataFrame:
    if feature not in df.columns:
        raise ValueError(f"Unknown feature column: {feature}")

    working = df.loc[df["rank"].between(1, max_rank), ["rank", "query_id", "page_id", feature]].copy()
    working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working = working.dropna(subset=[feature])
    working = working.sort_values(["rank", "query_id", "page_id"]).reset_index(drop=True)

    if len(working) < RANK_FEATURE_MIN_PERIODS:
        raise ValueError(
            f"Not enough numeric rows to plot {feature}. "
            f"Need at least {RANK_FEATURE_MIN_PERIODS}, found {len(working)}."
        )

    rolling = working[feature].rolling(
        window=window,
        center=True,
        min_periods=RANK_FEATURE_MIN_PERIODS,
    )
    plot_data = pd.DataFrame(
        {
            "rank": working["rank"],
            "median": rolling.median(),
            "q25": rolling.quantile(0.25),
            "q75": rolling.quantile(0.75),
            "q10": rolling.quantile(0.10),
            "q90": rolling.quantile(0.90),
        }
    ).dropna()

    if plot_data.empty:
        raise ValueError(f"No rolling statistics could be computed for {feature}.")

    return plot_data.groupby("rank", as_index=False).median(numeric_only=True)


def _load_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Matplotlib is required to generate rank-feature plots. "
            "Install dependencies with: pip install -r requrements.txt"
        ) from exc
    return plt


def plot_feature_vs_rank(
    df: pd.DataFrame,
    feature: str,
    window: int = RANK_FEATURE_ROLLING_WINDOW,
    max_rank: int = RANK_FEATURE_MAX_RANK,
):
    plt = _load_matplotlib_pyplot()

    plot_data = _rolling_rank_statistics(
        df=df,
        feature=feature,
        window=window,
        max_rank=max_rank,
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    x_values = plot_data["rank"].to_numpy()
    median = plot_data["median"].to_numpy()
    q25 = plot_data["q25"].to_numpy()
    q75 = plot_data["q75"].to_numpy()
    q10 = plot_data["q10"].to_numpy()
    q90 = plot_data["q90"].to_numpy()

    ax.fill_between(
        x_values,
        q10,
        q90,
        color=STEELBLUE,
        alpha=0.16,
        linewidth=0,
        label="10th-90th percentile band",
    )
    ax.fill_between(
        x_values,
        q25,
        q75,
        color=STEELBLUE,
        alpha=0.34,
        linewidth=0,
        label="25th-75th percentile band",
    )
    ax.plot(
        x_values,
        median,
        color=STEELBLUE,
        linewidth=2.4,
        label="Rolling median",
    )

    ax.set_title(f"{feature} vs. Rank", fontsize=16, pad=14)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Feature value")
    ax.set_xlim(1, max_rank)
    ax.grid(True, color="#d7e3ea", linewidth=0.8, alpha=0.72)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def _rolling_median_by_rank(
    df: pd.DataFrame,
    feature: str,
    window: int,
    max_rank: int,
) -> pd.DataFrame:
    working = df.loc[df["rank"].between(1, max_rank), ["rank", feature]].copy()
    working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working = working.dropna(subset=[feature])
    working = working.sort_values("rank").reset_index(drop=True)

    if len(working) < RANK_FEATURE_MIN_PERIODS:
        return pd.DataFrame(columns=["rank", "median"])

    rolling = working[feature].rolling(
        window=window,
        center=True,
        min_periods=RANK_FEATURE_MIN_PERIODS,
    )
    return pd.DataFrame(
        {
            "rank": working["rank"],
            "median": rolling.median(),
        }
    ).dropna()


def plot_feature_query_overlay(
    df: pd.DataFrame,
    feature: str,
    window: int = RANK_FEATURE_ROLLING_WINDOW,
    max_rank: int = RANK_FEATURE_MAX_RANK,
):
    plt = _load_matplotlib_pyplot()
    aggregate_data = _rolling_rank_statistics(
        df=df,
        feature=feature,
        window=window,
        max_rank=max_rank,
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    query_count = 0
    for query_id, query_df in df.groupby("query_id", sort=True):
        query_data = _rolling_median_by_rank(
            query_df,
            feature=feature,
            window=window,
            max_rank=max_rank,
        )
        if query_data.empty:
            continue
        query_count += 1
        ax.plot(
            query_data["rank"].to_numpy(),
            query_data["median"].to_numpy(),
            color=STEELBLUE,
            alpha=0.22,
            linewidth=1.05,
            label="Per-query rolling median" if query_count == 1 else None,
        )

    ax.plot(
        aggregate_data["rank"].to_numpy(),
        aggregate_data["median"].to_numpy(),
        color=STEELBLUE,
        linewidth=2.9,
        label="Aggregate rolling median",
    )
    ax.set_title(f"{feature} vs. Rank", fontsize=16, pad=14)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Feature value")
    ax.set_xlim(1, max_rank)
    ax.grid(True, color="#d7e3ea", linewidth=0.8, alpha=0.72)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def _rank_buckets(max_rank: int, bucket_count: int = 4) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for index in range(bucket_count):
        start = int((index * max_rank) / bucket_count) + 1
        end = int(((index + 1) * max_rank) / bucket_count)
        buckets.append(
            {
                "start": start,
                "end": end,
                "label": f"{start}-{end}",
            }
        )
    return buckets


def _iqr(values: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return 0.0
    return float(clean_values.quantile(0.75) - clean_values.quantile(0.25))


def plot_feature_consistency_heatmap(
    df: pd.DataFrame,
    features: list[str],
    max_rank: int = RANK_FEATURE_MAX_RANK,
):
    plt = _load_matplotlib_pyplot()
    buckets = _rank_buckets(max_rank)
    normalized_rows: list[list[float]] = []
    raw_iqr_rows: list[list[float]] = []

    for feature in features:
        feature_values = pd.to_numeric(df[feature], errors="coerce")
        global_iqr = _iqr(feature_values)
        normalized_row: list[float] = []
        raw_iqr_row: list[float] = []

        for bucket in buckets:
            bucket_values = df.loc[
                df["rank"].between(bucket["start"], bucket["end"]),
                feature,
            ]
            raw_iqr = _iqr(bucket_values)
            raw_iqr_row.append(raw_iqr)
            if global_iqr <= 0:
                normalized_row.append(0.0)
            else:
                normalized_row.append(max(0.0, min(raw_iqr / global_iqr, 1.0)))

        raw_iqr_rows.append(raw_iqr_row)
        normalized_rows.append(normalized_row)

    fig_height = max(7.2, 0.52 * len(features) + 2.6)
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    heatmap = ax.imshow(normalized_rows, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Feature consistency heatmap", fontsize=16, pad=14)
    ax.set_xlabel("Rank bucket")
    ax.set_ylabel("Feature")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([bucket["label"] for bucket in buckets])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)

    for feature_index, raw_iqr_row in enumerate(raw_iqr_rows):
        for bucket_index, raw_iqr in enumerate(raw_iqr_row):
            normalized_value = normalized_rows[feature_index][bucket_index]
            text_color = "white" if normalized_value >= 0.55 else "#2d2419"
            ax.text(
                bucket_index,
                feature_index,
                f"{raw_iqr:.3g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized IQR width")
    fig.tight_layout()
    return fig


def plot_feature_bucket_violin(
    df: pd.DataFrame,
    feature: str,
    max_rank: int = RANK_FEATURE_MAX_RANK,
):
    plt = _load_matplotlib_pyplot()
    buckets = _rank_buckets(max_rank)
    distributions: list[list[float]] = []
    labels: list[str] = []

    for bucket in buckets:
        values = pd.to_numeric(
            df.loc[df["rank"].between(bucket["start"], bucket["end"]), feature],
            errors="coerce",
        ).dropna()
        if values.empty:
            distributions.append([0.0])
        else:
            distributions.append(values.astype(float).tolist())
        labels.append(bucket["label"])

    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    violin_parts = ax.violinplot(
        distributions,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor(STEELBLUE)
        body.set_edgecolor("#245a7d")
        body.set_alpha(0.34)
        body.set_linewidth(0.8)

    for part_name in ["cmedians", "cbars", "cmins", "cmaxes"]:
        part = violin_parts.get(part_name)
        if part is not None:
            part.set_color("#245a7d")
            part.set_linewidth(1.15 if part_name != "cmedians" else 2.0)

    ax.set_title(f"{feature} distribution by rank bucket", fontsize=16, pad=14)
    ax.set_xlabel("Rank bucket")
    ax.set_ylabel("Feature value")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", color="#d7e3ea", linewidth=0.8, alpha=0.72)
    fig.tight_layout()
    return fig


def _median_drop_window(max_rank: int) -> dict[str, int]:
    window_size = max(5, min(MEDIAN_DROP_TOP_END, max_rank // 4))
    comparison_start = max(window_size + 1, max_rank - window_size + 1)
    return {
        "top_start": MEDIAN_DROP_TOP_START,
        "top_end": window_size,
        "comparison_start": comparison_start,
        "comparison_end": max_rank,
    }


def _median_drop_summary(
    df: pd.DataFrame,
    features: list[str],
    max_rank: int,
) -> list[dict[str, Any]]:
    window = _median_drop_window(max_rank)
    top_slice = df.loc[df["rank"].between(window["top_start"], window["top_end"])]
    comparison_slice = df.loc[
        df["rank"].between(window["comparison_start"], window["comparison_end"])
    ]
    summaries: list[dict[str, Any]] = []

    for feature in features:
        top_values = pd.to_numeric(top_slice[feature], errors="coerce").dropna()
        comparison_values = pd.to_numeric(comparison_slice[feature], errors="coerce").dropna()
        if top_values.empty or comparison_values.empty:
            continue

        top_median = float(top_values.median())
        comparison_median = float(comparison_values.median())
        summaries.append(
            {
                "feature": feature,
                "top_window_median": top_median,
                "comparison_median": comparison_median,
                "median_drop": top_median - comparison_median,
            }
        )

    return sorted(summaries, key=lambda item: item["median_drop"], reverse=True)


def get_or_create_rank_feature_plots(
    query_ids: list[str],
    plot_kind: str = DEFAULT_RANK_FEATURE_PLOT_KIND,
    top_k: int | str | None = None,
) -> dict[str, Any]:
    normalized_plot_kind = _normalize_plot_kind(plot_kind)
    normalized_query_ids = _normalize_query_ids(query_ids)
    normalized_top_k = normalize_rank_feature_top_k(top_k)
    collection_key = _safe_collection_key(
        normalized_query_ids,
        normalized_plot_kind,
        max_rank=normalized_top_k,
    )
    cached_metadata = _load_plot_metadata(
        collection_key,
        normalized_plot_kind,
        max_rank=normalized_top_k,
    )
    if cached_metadata is not None:
        return cached_metadata

    records_by_id = _query_records_by_id()
    output_dir = _plot_output_dir(collection_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = load_rank_feature_frame(
        normalized_query_ids,
        max_rank=normalized_top_k,
    )
    features = rank_feature_columns()
    plot_files: list[dict[str, Any]] = []

    if normalized_plot_kind == "heatmap":
        fig = plot_feature_consistency_heatmap(
            analysis_df,
            features,
            max_rank=normalized_top_k,
        )
        file_name = "heatmap.png"
        plot_path = output_dir / file_name
        fig.savefig(plot_path, dpi=150)
        import matplotlib.pyplot as plt

        plt.close(fig)
        plot_files.append(
            {
                "feature": "all_features",
                "title": "Feature consistency heatmap",
                "file_name": file_name,
                "path": _relative_to_repo(plot_path),
            }
        )
    else:
        for feature in features:
            if normalized_plot_kind == "violin":
                fig = plot_feature_bucket_violin(
                    analysis_df,
                    feature,
                    max_rank=normalized_top_k,
                )
                file_name = f"violin_{feature}.png"
                title = f"{feature} distribution by rank bucket"
            elif normalized_plot_kind == "overlay":
                fig = plot_feature_query_overlay(
                    analysis_df,
                    feature,
                    window=RANK_FEATURE_ROLLING_WINDOW,
                    max_rank=normalized_top_k,
                )
                file_name = f"overlay_{feature}.png"
                title = f"{feature} vs. Rank"
            else:
                fig = plot_feature_vs_rank(
                    analysis_df,
                    feature,
                    window=RANK_FEATURE_ROLLING_WINDOW,
                    max_rank=normalized_top_k,
                )
                file_name = f"plot_{feature}.png"
                title = f"{feature} vs. Rank"

            plot_path = output_dir / file_name
            fig.savefig(plot_path, dpi=150)
            import matplotlib.pyplot as plt

            plt.close(fig)
            plot_files.append(
                {
                    "feature": feature,
                    "title": title,
                    "file_name": file_name,
                    "path": _relative_to_repo(plot_path),
                }
            )

    summary_window = _median_drop_window(normalized_top_k)
    summary = _median_drop_summary(
        analysis_df,
        features,
        max_rank=normalized_top_k,
    )
    print(
        "Largest feature median drops between "
        f"ranks {summary_window['top_start']}-{summary_window['top_end']} and "
        f"{summary_window['comparison_start']}-{summary_window['comparison_end']}:"
    )
    for item in summary[:10]:
        print(
            f"  {item['feature']}: "
            f"{item['median_drop']:.6g} "
            f"(top_window={item['top_window_median']:.6g}, "
            f"comparison={item['comparison_median']:.6g})"
        )
    metadata = {
        "schema_version": RANK_FEATURE_PLOTS_SCHEMA_VERSION,
        "collection_key": collection_key,
        "cache_status": "generated",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "output_dir": _relative_to_repo(output_dir),
        "plot_kind": normalized_plot_kind,
        "plot_kind_label": next(
            option["label"]
            for option in RANK_FEATURE_PLOT_KINDS
            if option["value"] == normalized_plot_kind
        ),
        "query_ids": normalized_query_ids,
        "queries": [records_by_id[query_id] for query_id in normalized_query_ids],
        "rank_window": {
            "max_rank": normalized_top_k,
            "rolling_window": RANK_FEATURE_ROLLING_WINDOW,
            "center": True,
            "min_periods": RANK_FEATURE_MIN_PERIODS,
        },
        "rank_buckets": _rank_buckets(normalized_top_k),
        "median_drop_window": summary_window,
        "row_count": int(len(analysis_df)),
        "feature_count": int(len(features)),
        "features": features,
        "plot_files": plot_files,
        "median_drop_summary": summary,
    }
    _save_plot_metadata(collection_key, metadata)
    return metadata


def get_or_create_per_model_rank_feature_plots(
    query_ids: list[str],
    model_key: str,
    plot_kind: str = DEFAULT_RANK_FEATURE_PLOT_KIND,
    top_k: int | str | None = None,
) -> dict[str, Any]:
    normalized_model = _normalize_per_model_rank_feature_key(model_key)
    normalized_plot_kind = _normalize_plot_kind(plot_kind)
    normalized_query_ids = _normalize_query_ids(query_ids)
    normalized_top_k = normalize_rank_feature_top_k(top_k)
    collection_key = _safe_collection_key(
        normalized_query_ids,
        normalized_plot_kind,
        max_rank=normalized_top_k,
        extra_parts=[f"model={normalized_model}"],
    )
    cached_metadata = _load_plot_metadata(
        collection_key,
        normalized_plot_kind,
        max_rank=normalized_top_k,
        base_dir=PER_MODEL_RANK_FEATURE_PLOTS_DIR,
    )
    if cached_metadata is not None:
        return cached_metadata

    records_by_id = _query_records_by_id()
    model_option = next(
        option for option in PER_MODEL_RANK_FEATURE_OPTIONS if option["value"] == normalized_model
    )
    output_dir = _plot_output_dir(collection_key, base_dir=PER_MODEL_RANK_FEATURE_PLOTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = load_per_model_rank_feature_frame(
        normalized_query_ids,
        model_key=normalized_model,
        max_rank=normalized_top_k,
    )
    features = rank_feature_columns()
    plot_files: list[dict[str, Any]] = []

    if normalized_plot_kind == "heatmap":
        fig = plot_feature_consistency_heatmap(
            analysis_df,
            features,
            max_rank=normalized_top_k,
        )
        file_name = "heatmap.png"
        plot_path = output_dir / file_name
        fig.savefig(plot_path, dpi=150)
        import matplotlib.pyplot as plt

        plt.close(fig)
        plot_files.append(
            {
                "feature": "all_features",
                "title": "Feature consistency heatmap",
                "file_name": file_name,
                "path": _relative_to_repo(plot_path),
            }
        )
    else:
        for feature in features:
            if normalized_plot_kind == "violin":
                fig = plot_feature_bucket_violin(
                    analysis_df,
                    feature,
                    max_rank=normalized_top_k,
                )
                file_name = f"violin_{feature}.png"
                title = f"{feature} distribution by rank bucket"
            elif normalized_plot_kind == "overlay":
                fig = plot_feature_query_overlay(
                    analysis_df,
                    feature,
                    window=RANK_FEATURE_ROLLING_WINDOW,
                    max_rank=normalized_top_k,
                )
                file_name = f"overlay_{feature}.png"
                title = f"{feature} vs. Rank"
            else:
                fig = plot_feature_vs_rank(
                    analysis_df,
                    feature,
                    window=RANK_FEATURE_ROLLING_WINDOW,
                    max_rank=normalized_top_k,
                )
                file_name = f"plot_{feature}.png"
                title = f"{feature} vs. Rank"

            plot_path = output_dir / file_name
            fig.savefig(plot_path, dpi=150)
            import matplotlib.pyplot as plt

            plt.close(fig)
            plot_files.append(
                {
                    "feature": feature,
                    "title": title,
                    "file_name": file_name,
                    "path": _relative_to_repo(plot_path),
                }
            )

    summary_window = _median_drop_window(normalized_top_k)
    summary = _median_drop_summary(
        analysis_df,
        features,
        max_rank=normalized_top_k,
    )
    metadata = {
        "schema_version": RANK_FEATURE_PLOTS_SCHEMA_VERSION,
        "collection_key": collection_key,
        "cache_status": "generated",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "output_dir": _relative_to_repo(output_dir),
        "plot_kind": normalized_plot_kind,
        "plot_kind_label": next(
            option["label"]
            for option in RANK_FEATURE_PLOT_KINDS
            if option["value"] == normalized_plot_kind
        ),
        "selected_model": normalized_model,
        "selected_model_label": model_option["label"],
        "query_ids": normalized_query_ids,
        "queries": [records_by_id[query_id] for query_id in normalized_query_ids],
        "rank_window": {
            "max_rank": normalized_top_k,
            "rolling_window": RANK_FEATURE_ROLLING_WINDOW,
            "center": True,
            "min_periods": RANK_FEATURE_MIN_PERIODS,
        },
        "rank_buckets": _rank_buckets(normalized_top_k),
        "median_drop_window": summary_window,
        "row_count": int(len(analysis_df)),
        "feature_count": int(len(features)),
        "features": features,
        "plot_files": plot_files,
        "median_drop_summary": summary,
    }
    _save_plot_metadata(
        collection_key,
        metadata,
        base_dir=PER_MODEL_RANK_FEATURE_PLOTS_DIR,
    )
    return metadata


def resolve_rank_feature_plot_path(collection_key: str, file_name: str) -> Path:
    normalized_collection_key = _validate_collection_key(collection_key)
    normalized_file_name = _validate_plot_file_name(file_name)
    plot_path = _plot_output_dir(normalized_collection_key) / normalized_file_name
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing plot file: {plot_path}")
    return plot_path


def resolve_per_model_rank_feature_plot_path(collection_key: str, file_name: str) -> Path:
    normalized_collection_key = _validate_collection_key(collection_key)
    normalized_file_name = _validate_plot_file_name(file_name)
    plot_path = (
        _plot_output_dir(
            normalized_collection_key,
            base_dir=PER_MODEL_RANK_FEATURE_PLOTS_DIR,
        )
        / normalized_file_name
    )
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing plot file: {plot_path}")
    return plot_path


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
