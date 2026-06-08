from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import association_analysis
from retriever import run_hybrid_rankings as ranking_backend


RANK_RELATIONSHIP_DIR = ranking_backend.RANKINGS_DIR / "rank_relationship"
PER_MODEL_RANK_RELATIONSHIP_DIR = ranking_backend.RANKINGS_DIR / "per_model_rank_relationship"
RESULTS_DIR_NAME = "results"
METADATA_FILE_NAME = "metadata.json"
SCHEMA_VERSION = 3
PER_MODEL_SCHEMA_VERSION = 2

MODEL_SCORE_COLUMNS = {
    "hybrid": "hybrid_score",
    "bm25": "bm25_score_norm",
    "semantic": "semantic_score_norm",
}

MODEL_HEATMAP_FEATURES = [
    "avg_token_length",
    "char_count",
    "lexical_diversity",
    "flesch_kincaid_grade",
    "sentence_count",
    "word_count",
]

MODEL_HEATMAP_FEATURE_LABELS = {
    "avg_token_length": "Avg Token Length",
    "char_count": "Document Length",
    "lexical_diversity": "Lexical Diversity",
    "flesch_kincaid_grade": "Readability Grade",
    "sentence_count": "Sentence Count",
    "word_count": "Word Count",
}

PER_MODEL_RELATIONSHIP_OPTIONS = {
    "bm25": {
        "model_name": "bm25",
        "label": "BM25",
        "score_label": "BM25 normalized score",
    },
    "sbert": {
        "model_name": "semantic",
        "label": "SBERT",
        "score_label": "SBERT normalized score",
    },
}

PER_MODEL_COMPARISON_MODELS = {
    "bm25": "BM25",
    "semantic": "SBERT",
}


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


def _file_name_from_source_path(source_path: Any) -> str:
    normalized = str(source_path or "").replace("\\", "/")
    return normalized.rsplit("/", maxsplit=1)[-1]


def _relative_to_repo(path: Path) -> str:
    return ranking_backend.relative_to_repo(path)


def _content_features_source_path() -> Path:
    parquet_path = ranking_backend.REPO_ROOT / "data" / "content_features" / "content_features.parquet"
    csv_path = ranking_backend.REPO_ROOT / "data" / "content_features" / "content_features.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        f"Missing content feature table. Expected {parquet_path} or {csv_path}."
    )


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{_relative_to_repo(path)}:{stat.st_size}:{int(stat.st_mtime)}"


def _ranking_result_path(record: dict[str, Any]) -> Path:
    return ranking_backend.resolve_result_path(str(record.get("result_file") or ""))


def _record_result_count(record: dict[str, Any]) -> int:
    return int(record.get("stored_result_count") or record.get("result_count") or 0)


def _record_is_full_ranking(record: dict[str, Any]) -> bool:
    if bool(record.get("stored_full_ranking")):
        return True
    return _record_result_count(record) >= ranking_backend.total_document_count()


def _format_query_record(record: dict[str, Any]) -> dict[str, Any]:
    result_path = _ranking_result_path(record)
    return {
        "query_id": str(record.get("query_id") or ""),
        "query_text": str(record.get("query_text") or ""),
        "result_file": str(record.get("result_file") or ""),
        "stored_result_count": _record_result_count(record),
        "is_full_ranking": _record_is_full_ranking(record),
        "has_result_file": result_path.exists(),
    }


def rank_relationship_query_records() -> list[dict[str, Any]]:
    records = [
        _format_query_record(record)
        for record in ranking_backend.iter_saved_query_records()
    ]
    return sorted(records, key=lambda record: record["query_id"])


def normalize_per_model_relationship_key(model_key: str | None) -> str:
    normalized = str(model_key or "").strip().lower()
    if normalized in PER_MODEL_RELATIONSHIP_OPTIONS:
        return normalized
    return "bm25"


def per_model_rank_relationship_options() -> list[dict[str, str]]:
    return [
        {
            "value": option_key,
            "label": option["label"],
            "score_label": option["score_label"],
        }
        for option_key, option in PER_MODEL_RELATIONSHIP_OPTIONS.items()
    ]


def get_per_model_rank_relationship_option(model_key: str | None) -> dict[str, str]:
    normalized = normalize_per_model_relationship_key(model_key)
    return {
        "value": normalized,
        **PER_MODEL_RELATIONSHIP_OPTIONS[normalized],
    }


def _usable_query_records() -> list[dict[str, Any]]:
    records = [
        record
        for record in rank_relationship_query_records()
        if record["has_result_file"] and record["is_full_ranking"]
    ]
    if not records:
        raise FileNotFoundError("No saved full-ranking query parquets are available.")
    return records


def _analysis_cache_key(query_records: list[dict[str, Any]]) -> str:
    content_path = _content_features_source_path()
    parts = [
        f"schema={SCHEMA_VERSION}",
        _file_fingerprint(content_path),
        _file_fingerprint(ranking_backend.BM25_METADATA_FILE),
    ]
    for record in query_records:
        result_path = ranking_backend.resolve_result_path(record["result_file"])
        parts.append(f"{record['query_id']}:{_file_fingerprint(result_path)}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"collection_{digest}"


def _collection_dir(collection_key: str) -> Path:
    return RANK_RELATIONSHIP_DIR / collection_key


def _metadata_path(collection_key: str) -> Path:
    return _collection_dir(collection_key) / METADATA_FILE_NAME


def _per_model_collection_dir(collection_key: str) -> Path:
    return PER_MODEL_RANK_RELATIONSHIP_DIR / collection_key


def _per_model_metadata_path(collection_key: str) -> Path:
    return _per_model_collection_dir(collection_key) / METADATA_FILE_NAME


def _load_content_features_for_analysis() -> pd.DataFrame:
    content_path = _content_features_source_path()
    if content_path.suffix.lower() == ".parquet":
        features = pd.read_parquet(content_path)
    else:
        features = pd.read_csv(content_path)

    features = features.copy()
    features["file_name"] = features["source_path"].map(_file_name_from_source_path)
    features = features.rename(columns={"doc_id": "feature_doc_id", "title": "feature_title"})

    metadata = pd.read_parquet(
        ranking_backend.BM25_METADATA_FILE,
        columns=["doc_id", "page_id", "title", "file_name"],
    )
    merged = features.merge(metadata, on="file_name", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("Could not join content features to BM25 metadata by file_name.")

    passthrough_columns = [
        column
        for column in ["entities_json", "top_entities_json"]
        if column in merged.columns
    ]
    numeric_feature_columns = [
        column
        for column in merged.select_dtypes(include=["number"]).columns
        if column not in {"feature_doc_id", "doc_id", "page_id"}
    ]
    output = merged[
        ["doc_id", "title", "source_path", *numeric_feature_columns, *passthrough_columns]
    ].copy()
    output["doc_id"] = output["doc_id"].astype("int64")
    return output


def _load_rankings_for_analysis(query_records: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in query_records:
        result_path = ranking_backend.resolve_result_path(record["result_file"])
        ranking = pd.read_parquet(
            result_path,
            columns=["doc_id", "rank", "hybrid_score"],
        ).rename(columns={"hybrid_score": "score"})
        ranking["query_id"] = record["query_id"]
        frames.append(ranking[["doc_id", "query_id", "rank", "score"]])

    rankings = pd.concat(frames, ignore_index=True)
    rankings["doc_id"] = rankings["doc_id"].astype("int64")
    return rankings


def _load_model_rankings_for_analysis(query_records: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    read_columns = ["doc_id", "rank", *MODEL_SCORE_COLUMNS.values()]
    for record in query_records:
        result_path = ranking_backend.resolve_result_path(record["result_file"])
        ranking = pd.read_parquet(result_path, columns=read_columns)
        ranking["doc_id"] = ranking["doc_id"].astype("int64")

        for model_name, score_column in MODEL_SCORE_COLUMNS.items():
            score_values = pd.to_numeric(ranking[score_column], errors="coerce")
            if model_name == "hybrid":
                rank_values = pd.to_numeric(ranking["rank"], errors="coerce")
            else:
                rank_values = score_values.rank(
                    method="average",
                    ascending=False,
                    na_option="bottom",
                )
            frame = pd.DataFrame(
                {
                    "doc_id": ranking["doc_id"],
                    "query_id": record["query_id"],
                    "model": model_name,
                    "doc_rank": rank_values,
                    "doc_score": score_values,
                }
            )
            frames.append(frame)

    model_rankings = pd.concat(frames, ignore_index=True)
    model_summary = (
        model_rankings.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["doc_id", "model", "doc_rank", "doc_score"])
        .groupby(["doc_id", "model"], as_index=False)
        .agg(doc_rank=("doc_rank", "mean"), doc_score=("doc_score", "mean"))
    )
    return model_summary


def _build_analysis_frame(query_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    features = _load_content_features_for_analysis()
    rankings = _load_rankings_for_analysis(query_records)
    rank_summary = (
        rankings.dropna(subset=["doc_id", "rank", "score"])
        .groupby("doc_id", as_index=False)
        .agg(avg_rank=("rank", "mean"), avg_score=("score", "mean"))
    )
    merged = features.merge(rank_summary, on="doc_id", how="inner")
    if merged.empty:
        raise ValueError("No articles remained after merging features with ranking summaries.")
    feature_columns = association_analysis.infer_feature_columns(merged)
    return merged, feature_columns


def _build_model_analysis_frame(
    query_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    features = _load_content_features_for_analysis()
    model_rankings = _load_model_rankings_for_analysis(query_records)
    merged = features.merge(model_rankings, on="doc_id", how="inner")
    if merged.empty:
        raise ValueError("No articles remained after merging features with model rankings.")
    feature_columns = [
        feature
        for feature in MODEL_HEATMAP_FEATURES
        if feature in merged.columns
        and pd.to_numeric(merged[feature], errors="coerce").notna().any()
    ]
    if not feature_columns:
        raise ValueError("No usable model heatmap feature columns were found.")
    return merged, feature_columns


def _build_per_model_analysis_frame(
    query_records: list[dict[str, Any]],
    model_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    features = _load_content_features_for_analysis()
    model_rankings = _load_model_rankings_for_analysis(query_records)
    selected_model = model_rankings.loc[model_rankings["model"].eq(model_name)].copy()
    if selected_model.empty:
        raise ValueError(f"No saved ranking rows were available for model {model_name}.")

    selected_model = selected_model.rename(
        columns={
            "doc_rank": "avg_rank",
            "doc_score": "avg_score",
        }
    )
    merged = features.merge(
        selected_model[["doc_id", "avg_rank", "avg_score"]],
        on="doc_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No articles remained after merging features with the selected model ranking.")
    feature_columns = association_analysis.infer_feature_columns(merged)
    if not feature_columns:
        raise ValueError("No usable feature columns were found for the selected model analysis.")
    return merged, feature_columns


def _build_bm25_sbert_correlation_outputs(
    query_records: list[dict[str, Any]],
    results_dir: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    model_merged, model_feature_columns = _build_model_analysis_frame(query_records)
    comparison_merged = model_merged.loc[
        model_merged["model"].isin(PER_MODEL_COMPARISON_MODELS.keys())
    ].copy()
    if comparison_merged.empty:
        raise ValueError("No BM25 or SBERT rows were available for the comparison heatmap.")

    model_correlations = association_analysis.analyze_model_correlations(
        comparison_merged,
        model_feature_columns,
    )
    if model_correlations.empty:
        raise ValueError("Could not compute BM25/SBERT feature-rank correlations.")

    model_correlations["model"] = model_correlations["model"].map(
        lambda value: "sbert" if str(value) == "semantic" else str(value)
    )

    model_correlations_path = results_dir / "bm25_sbert_model_correlations.csv"
    model_correlation_heatmaps_path = results_dir / "bm25_sbert_model_correlation_heatmaps.png"
    model_correlations.to_csv(model_correlations_path, index=False)
    association_analysis.plot_model_correlation_heatmaps(
        model_correlations,
        model_correlation_heatmaps_path,
        feature_labels=MODEL_HEATMAP_FEATURE_LABELS,
    )
    return model_correlations, model_correlations_path, model_correlation_heatmaps_path


def _load_cached_metadata_file(
    metadata_path: Path,
    schema_version: int,
) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None

    output_paths = metadata.get("outputs") or {}
    if not output_paths:
        return None
    for relative_path in output_paths.values():
        if not (ranking_backend.REPO_ROOT / relative_path).exists():
            return None

    if int(metadata.get("schema_version") or 0) != schema_version:
        return None
    metadata["cache_status"] = "reused"
    return metadata


def _load_cached_metadata(collection_key: str) -> dict[str, Any] | None:
    return _load_cached_metadata_file(_metadata_path(collection_key), SCHEMA_VERSION)


def _load_per_model_cached_metadata(collection_key: str) -> dict[str, Any] | None:
    return _load_cached_metadata_file(
        _per_model_metadata_path(collection_key),
        PER_MODEL_SCHEMA_VERSION,
    )


def get_rank_relationship_metadata(collection_key: str) -> dict[str, Any]:
    metadata = _load_cached_metadata(collection_key)
    if metadata is None:
        raise FileNotFoundError(f"Unknown rank relationship collection: {collection_key}")
    return metadata


def _write_metadata(collection_key: str, metadata: dict[str, Any]) -> None:
    metadata_path = _metadata_path(collection_key)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_per_model_metadata(collection_key: str, metadata: dict[str, Any]) -> None:
    metadata_path = _per_model_metadata_path(collection_key)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _summary_preview(summary: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    preview_columns = [
        "feature",
        "mutual_info_norm",
        "spearman_r",
        "spearman_p",
        "partial_r",
        "mw_effect_top10",
        "has_breakpoint",
        "breakpoint_value",
    ]
    available_columns = [column for column in preview_columns if column in summary.columns]
    return [
        {
            key: _clean_scalar(value)
            for key, value in row.items()
        }
        for row in summary[available_columns].head(limit).to_dict(orient="records")
    ]


def get_or_create_rank_relationship_analysis(force: bool = False) -> dict[str, Any]:
    query_records = _usable_query_records()
    collection_key = _analysis_cache_key(query_records)
    cached_metadata = None if force else _load_cached_metadata(collection_key)
    if cached_metadata is not None:
        return cached_metadata

    collection_dir = _collection_dir(collection_key)
    results_dir = collection_dir / RESULTS_DIR_NAME
    collection_dir.mkdir(parents=True, exist_ok=True)

    merged, feature_columns = _build_analysis_frame(query_records)
    summary, output_paths = association_analysis.run_association_analysis(
        merged,
        feature_columns,
        results_dir=results_dir,
    )
    model_merged, model_feature_columns = _build_model_analysis_frame(query_records)
    model_correlations = association_analysis.analyze_model_correlations(
        model_merged,
        model_feature_columns,
    )
    model_importance = association_analysis.compute_model_feature_importance(
        model_merged,
        model_feature_columns,
        n_estimators=200,
        n_repeats=10,
        n_jobs=1,
    )

    model_correlations_path = results_dir / "model_correlations.csv"
    model_importance_path = results_dir / "model_importance.csv"
    model_correlation_heatmaps_path = results_dir / "model_correlation_heatmaps.png"
    model_importance_heatmaps_path = results_dir / "model_importance_heatmaps.png"
    model_correlations.to_csv(model_correlations_path, index=False)
    model_importance.to_csv(model_importance_path, index=False)
    association_analysis.plot_model_correlation_heatmaps(
        model_correlations,
        model_correlation_heatmaps_path,
        feature_labels=MODEL_HEATMAP_FEATURE_LABELS,
    )
    association_analysis.plot_model_importance_heatmaps(
        model_importance,
        model_importance_heatmaps_path,
        feature_labels=MODEL_HEATMAP_FEATURE_LABELS,
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "collection_key": collection_key,
        "cache_status": "generated",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "query_count": len(query_records),
        "article_count": int(len(merged)),
        "feature_count": int(len(feature_columns)),
        "model_feature_count": int(len(model_feature_columns)),
        "model_names": sorted(MODEL_SCORE_COLUMNS.keys()),
        "queries": query_records,
        "outputs": {
            "summary": _relative_to_repo(output_paths["summary"]),
            "dot_plot": _relative_to_repo(output_paths["dot_plot"]),
            "diverging_bar": _relative_to_repo(output_paths["diverging_bar"]),
            "model_correlations": _relative_to_repo(model_correlations_path),
            "model_importance": _relative_to_repo(model_importance_path),
            "model_correlation_heatmaps": _relative_to_repo(model_correlation_heatmaps_path),
            "model_importance_heatmaps": _relative_to_repo(model_importance_heatmaps_path),
        },
        "top_features": _summary_preview(summary, limit=10),
    }
    _write_metadata(collection_key, metadata)
    return metadata


def _per_model_analysis_cache_key(query_records: list[dict[str, Any]], model_key: str) -> str:
    model_option = get_per_model_rank_relationship_option(model_key)
    content_path = _content_features_source_path()
    parts = [
        f"schema={PER_MODEL_SCHEMA_VERSION}",
        f"model={model_option['value']}:{model_option['model_name']}",
        _file_fingerprint(content_path),
        _file_fingerprint(ranking_backend.BM25_METADATA_FILE),
    ]
    for record in query_records:
        result_path = ranking_backend.resolve_result_path(record["result_file"])
        parts.append(f"{record['query_id']}:{_file_fingerprint(result_path)}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{model_option['value']}_collection_{digest}"


def get_per_model_rank_relationship_metadata(collection_key: str) -> dict[str, Any]:
    metadata = _load_per_model_cached_metadata(collection_key)
    if metadata is None:
        raise FileNotFoundError(f"Unknown per-model rank relationship collection: {collection_key}")
    return metadata


def get_or_create_per_model_rank_relationship_analysis(
    model_key: str | None,
    force: bool = False,
) -> dict[str, Any]:
    model_option = get_per_model_rank_relationship_option(model_key)
    query_records = _usable_query_records()
    collection_key = _per_model_analysis_cache_key(query_records, model_option["value"])
    cached_metadata = None if force else _load_per_model_cached_metadata(collection_key)
    if cached_metadata is not None:
        return cached_metadata

    collection_dir = _per_model_collection_dir(collection_key)
    results_dir = collection_dir / RESULTS_DIR_NAME
    collection_dir.mkdir(parents=True, exist_ok=True)

    merged, feature_columns = _build_per_model_analysis_frame(
        query_records,
        model_option["model_name"],
    )
    summary, output_paths = association_analysis.run_association_analysis(
        merged,
        feature_columns,
        results_dir=results_dir,
    )
    (
        bm25_sbert_correlations,
        bm25_sbert_correlations_path,
        bm25_sbert_heatmap_path,
    ) = _build_bm25_sbert_correlation_outputs(
        query_records,
        results_dir=results_dir,
    )

    metadata = {
        "schema_version": PER_MODEL_SCHEMA_VERSION,
        "collection_key": collection_key,
        "cache_status": "generated",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selected_model": model_option["value"],
        "selected_model_name": model_option["model_name"],
        "selected_model_label": model_option["label"],
        "selected_score_label": model_option["score_label"],
        "query_count": len(query_records),
        "article_count": int(len(merged)),
        "feature_count": int(len(feature_columns)),
        "comparison_model_names": ["bm25", "sbert"],
        "comparison_feature_count": int(bm25_sbert_correlations["feature"].nunique()),
        "queries": query_records,
        "outputs": {
            "summary": _relative_to_repo(output_paths["summary"]),
            "dot_plot": _relative_to_repo(output_paths["dot_plot"]),
            "diverging_bar": _relative_to_repo(output_paths["diverging_bar"]),
            "bm25_sbert_model_correlations": _relative_to_repo(bm25_sbert_correlations_path),
            "bm25_sbert_model_correlation_heatmaps": _relative_to_repo(bm25_sbert_heatmap_path),
        },
        "top_features": _summary_preview(summary, limit=10),
    }
    _write_per_model_metadata(collection_key, metadata)
    return metadata


def resolve_rank_relationship_output(collection_key: str, output_name: str) -> Path:
    if output_name not in {
        "summary",
        "dot_plot",
        "diverging_bar",
        "model_correlations",
        "model_importance",
        "model_correlation_heatmaps",
        "model_importance_heatmaps",
    }:
        raise FileNotFoundError(f"Unknown rank relationship output: {output_name}")
    metadata = _load_cached_metadata(collection_key)
    if metadata is None:
        raise FileNotFoundError(f"Unknown rank relationship collection: {collection_key}")
    output_relative = metadata["outputs"].get(output_name)
    if not output_relative:
        raise FileNotFoundError(f"Missing output {output_name} for {collection_key}")
    output_path = ranking_backend.REPO_ROOT / output_relative
    if not output_path.exists():
        raise FileNotFoundError(f"Missing rank relationship output: {output_path}")
    return output_path


def resolve_per_model_rank_relationship_output(collection_key: str, output_name: str) -> Path:
    if output_name not in {
        "summary",
        "dot_plot",
        "diverging_bar",
        "bm25_sbert_model_correlations",
        "bm25_sbert_model_correlation_heatmaps",
    }:
        raise FileNotFoundError(f"Unknown per-model rank relationship output: {output_name}")
    metadata = _load_per_model_cached_metadata(collection_key)
    if metadata is None:
        raise FileNotFoundError(f"Unknown per-model rank relationship collection: {collection_key}")
    output_relative = metadata["outputs"].get(output_name)
    if not output_relative:
        raise FileNotFoundError(f"Missing output {output_name} for {collection_key}")
    output_path = ranking_backend.REPO_ROOT / output_relative
    if not output_path.exists():
        raise FileNotFoundError(f"Missing per-model rank relationship output: {output_path}")
    return output_path
