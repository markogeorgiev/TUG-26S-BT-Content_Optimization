from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retriever import run_hybrid_rankings as ranking_backend
from ui.services.ranking_ui_service import build_per_model_ranking, get_per_model_config


CONTENT_EXPERIMENT_PREVIEW_LIMIT = 25
SEMANTIC_CHUNK_SIZE_WORDS = 512
PAGERANK_DAMPING_FALLBACK = 0.85

TEXT_DIR_CANDIDATES = [
    ranking_backend.REPO_ROOT / "data" / "corpus_cleaned",
    ranking_backend.REPO_ROOT / "data" / "corpus",
    ranking_backend.REPO_ROOT / "output" / "wikipedia-pets" / "texts-cleaned",
    ranking_backend.REPO_ROOT / "output" / "wikipedia-pets" / "texts",
]

SCORE_COMPONENT_SPECS = [
    ("Hybrid score", "hybrid_score", "score"),
    ("BM25 raw", "bm25_score_raw", "score"),
    ("BM25 normalized", "bm25_score_norm", "score"),
    ("Semantic raw", "semantic_score_raw", "score"),
    ("Semantic normalized", "semantic_score_norm", "score"),
    ("PageRank raw", "pagerank", "score"),
    ("PageRank normalized", "pagerank_norm", "score"),
]


@dataclass(frozen=True)
class LinkEditRequest:
    add_outgoing_page_ids: list[int]
    remove_outgoing_page_ids: list[int]
    add_incoming_page_ids: list[int]
    remove_incoming_page_ids: list[int]


def _float_or_zero(value: Any) -> float:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return 0.0
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return 0.0


def _numeric_series(results_df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in results_df.columns:
        raise ValueError(f"Saved ranking is missing the required column: {column_name}")
    return pd.to_numeric(results_df[column_name], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _experiment_view_metadata(model_key: str | None) -> dict[str, Any]:
    normalized = str(model_key or "").strip().lower()
    if normalized == "sbert":
        model_config = get_per_model_config("sbert")
        return {
            "key": "sbert",
            "view_label": "Dense",
            "model_label": model_config["label"],
            "rank_label": "Dense rank",
            "score_label": model_config["score_label"],
            "comparison_rank_label": "Hybrid rank",
            "comparison_score_label": "Hybrid score",
            "is_hybrid": False,
        }
    if normalized == "bm25":
        model_config = get_per_model_config("bm25")
        return {
            "key": "bm25",
            "view_label": "Sparse",
            "model_label": model_config["label"],
            "rank_label": "Sparse rank",
            "score_label": model_config["score_label"],
            "comparison_rank_label": "Hybrid rank",
            "comparison_score_label": "Hybrid score",
            "is_hybrid": False,
        }
    return {
        "key": "hybrid",
        "view_label": "Hybrid",
        "model_label": "Hybrid",
        "rank_label": "Hybrid rank",
        "score_label": "Hybrid score",
        "comparison_rank_label": None,
        "comparison_score_label": None,
        "is_hybrid": True,
    }


def _ordered_score_component_specs(model_key: str | None) -> list[tuple[str, str, str]]:
    normalized = str(model_key or "").strip().lower()
    if normalized == "sbert":
        primary_columns = ["semantic_score_norm", "semantic_score_raw", "hybrid_score"]
    elif normalized == "bm25":
        primary_columns = ["bm25_score_norm", "bm25_score_raw", "hybrid_score"]
    else:
        primary_columns = ["hybrid_score", "bm25_score_norm", "semantic_score_norm"]

    spec_lookup = {column: (label, column, value_format) for label, column, value_format in SCORE_COMPONENT_SPECS}
    ordered_specs = [spec_lookup[column] for column in primary_columns if column in spec_lookup]
    remaining_specs = [
        spec
        for spec in SCORE_COMPONENT_SPECS
        if spec[1] not in set(primary_columns)
    ]
    return ordered_specs + remaining_specs


def _score_component_rows(row: pd.Series, model_key: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (label, column, value_format) in enumerate(_ordered_score_component_specs(model_key)):
        rows.append(
            {
                "label": label,
                "value": _float_or_zero(row.get(column)),
                "format": value_format,
                "is_primary": index == 0,
            }
        )
    return rows


def _build_selected_ranking_view(
    ranking_df: pd.DataFrame,
    model_key: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _experiment_view_metadata(model_key)
    if metadata["is_hybrid"]:
        view = ranking_df.copy()
        hybrid_rank = _numeric_series(view, "rank").fillna(len(view) + 1).astype("int64")
        view["hybrid_rank"] = hybrid_rank
        view["selected_rank"] = hybrid_rank
        view["selected_score"] = _numeric_series(view, "hybrid_score")
        view = view.sort_values(
            ["selected_rank", "title"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return view, metadata

    reranked, _ = build_per_model_ranking(ranking_df, metadata["key"])
    reranked["selected_rank"] = reranked["model_rank"].astype("int64")
    reranked["selected_score"] = pd.to_numeric(reranked["model_score"], errors="coerce")
    return reranked, metadata


def _experiment_method_note(metadata: dict[str, Any]) -> str:
    if metadata["is_hybrid"]:
        return (
            "Temporary experiment only. Content edits recompute the selected article's "
            "BM25 and MiniLM document scores. Link edits use a local PageRank edge-contribution "
            "approximation instead of recomputing global PageRank, so no corpus or graph files are changed."
        )

    return (
        "Temporary experiment only. Content edits recompute the selected article's BM25 and MiniLM "
        "document scores, link edits use a local PageRank edge-contribution approximation, and the "
        f"displayed {metadata['view_label'].lower()} workspace reranks the temporary result with "
        f"{metadata['model_label'].lower()} ordering. No corpus or graph files are changed."
    )


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


def _record_result_count(record: dict[str, Any]) -> int:
    return int(record.get("stored_result_count") or record.get("result_count") or 0)


def _record_is_full_ranking(record: dict[str, Any]) -> bool:
    if bool(record.get("stored_full_ranking")):
        return True
    return _record_result_count(record) >= ranking_backend.total_document_count()


def _ranking_result_path(record: dict[str, Any]) -> Path:
    return ranking_backend.resolve_result_path(str(record.get("result_file") or ""))


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


def saved_experiment_query_options() -> list[dict[str, Any]]:
    records = [_format_query_record(record) for record in ranking_backend.iter_saved_query_records()]
    return sorted(records, key=lambda record: record["query_id"])


def _query_records_by_id() -> dict[str, dict[str, Any]]:
    return {
        str(record.get("query_id") or ""): _format_query_record(record)
        for record in ranking_backend.iter_saved_query_records()
    }


def get_experiment_query_record(query_id: str) -> dict[str, Any]:
    normalized = str(query_id or "").strip()
    records_by_id = _query_records_by_id()
    if normalized not in records_by_id:
        raise FileNotFoundError(f"Unknown query_id: {query_id}")

    record = records_by_id[normalized]
    if not record["has_result_file"]:
        raise FileNotFoundError(f"Missing ranking parquet for {normalized}.")
    if not record["is_full_ranking"]:
        raise ValueError(
            f"{normalized} is only a partial ranking. Content experiments require a full ranking."
        )
    return record


@lru_cache(maxsize=1)
def load_article_catalog() -> pd.DataFrame:
    ranking_backend.validate_file_exists(
        ranking_backend.BM25_METADATA_FILE,
        "BM25 document metadata file",
    )
    metadata = pd.read_parquet(
        ranking_backend.BM25_METADATA_FILE,
        columns=[
            "doc_id",
            "page_id",
            "title",
            "file_name",
            "word_count",
            "char_count",
        ],
    )

    graph_nodes = pd.DataFrame(
        columns=[
            "page_id",
            "graph_id",
            "article_url",
            "in_degree",
            "out_degree",
            "pagerank",
        ]
    )
    if ranking_backend.GRAPH_NODES_FILE.exists():
        graph_nodes = pd.read_csv(
            ranking_backend.GRAPH_NODES_FILE,
            usecols=[
                "graph_id",
                "page_id",
                "url",
                "in_degree",
                "out_degree",
                "pagerank",
            ],
        ).rename(columns={"url": "article_url"})
        graph_nodes = graph_nodes.drop_duplicates(subset=["page_id"])

    catalog = metadata.merge(graph_nodes, on="page_id", how="left")
    catalog["page_id"] = catalog["page_id"].astype("int64")
    catalog["doc_id"] = catalog["doc_id"].astype("int64")
    catalog["page_id_text"] = catalog["page_id"].astype(str)
    catalog["title_search"] = catalog["title"].astype(str).str.casefold()
    return catalog


def search_experiment_articles(search_text: str, limit: int = 10) -> list[dict[str, Any]]:
    normalized = str(search_text or "").strip().casefold()
    if not normalized:
        return []

    catalog = load_article_catalog()
    if normalized.isdigit():
        matches = catalog.loc[catalog["page_id_text"].str.startswith(normalized)].copy()
    else:
        matches = catalog.loc[catalog["title_search"].str.contains(re.escape(normalized), na=False)].copy()
        exact_matches = matches.loc[matches["title_search"].eq(normalized)]
        if not exact_matches.empty:
            matches = pd.concat(
                [exact_matches, matches.loc[~matches["page_id"].isin(exact_matches["page_id"])]],
                ignore_index=True,
            )

    matches = matches.head(max(limit, 0))
    return [
        {
            "page_id": int(row.page_id),
            "title": str(row.title),
            "file_name": str(row.file_name),
            "doc_id": int(row.doc_id),
        }
        for row in matches.itertuples(index=False)
    ]


def resolve_experiment_article(search_text: str) -> dict[str, Any] | None:
    normalized = str(search_text or "").strip()
    if not normalized:
        return None

    catalog = load_article_catalog()
    if normalized.isdigit():
        page_id = int(normalized)
        exact = catalog.loc[catalog["page_id"].eq(page_id)]
        if not exact.empty:
            return _article_row_to_dict(exact.iloc[0])

    normalized_title = normalized.casefold()
    exact_title = catalog.loc[catalog["title_search"].eq(normalized_title)]
    if not exact_title.empty:
        return _article_row_to_dict(exact_title.iloc[0])

    suggestions = search_experiment_articles(normalized, limit=1)
    if not suggestions:
        return None
    return get_experiment_article(suggestions[0]["page_id"])


def _article_row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "doc_id": int(row["doc_id"]),
        "page_id": int(row["page_id"]),
        "title": str(row["title"]),
        "file_name": str(row["file_name"]),
        "word_count": int(row["word_count"]),
        "char_count": int(row["char_count"]),
        "article_url": _clean_scalar(row.get("article_url")),
        "graph_id": _clean_scalar(row.get("graph_id")),
        "in_degree": _clean_scalar(row.get("in_degree")),
        "out_degree": _clean_scalar(row.get("out_degree")),
        "pagerank": _clean_scalar(row.get("pagerank")),
    }


def _article_by_page_id(page_id: int) -> dict[str, Any]:
    catalog = load_article_catalog()
    matches = catalog.loc[catalog["page_id"].eq(int(page_id))]
    if matches.empty:
        raise FileNotFoundError(f"Unknown page_id: {page_id}")
    return _article_row_to_dict(matches.iloc[0])


def _read_article_text(file_name: str) -> tuple[str, str]:
    for directory in TEXT_DIR_CANDIDATES:
        candidate = directory / file_name
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), ranking_backend.relative_to_repo(candidate)

    metadata = pd.read_parquet(
        ranking_backend.BM25_METADATA_FILE,
        columns=["file_name", "document_text"],
    )
    matches = metadata.loc[metadata["file_name"].eq(file_name)]
    if matches.empty:
        raise FileNotFoundError(f"Could not find article text for {file_name}.")
    return str(matches.iloc[0]["document_text"] or ""), "data/bm25_index/document_metadata.parquet"


@lru_cache(maxsize=1)
def load_graph_nodes() -> pd.DataFrame:
    ranking_backend.validate_file_exists(ranking_backend.GRAPH_NODES_FILE, "graph nodes file")
    nodes = pd.read_csv(
        ranking_backend.GRAPH_NODES_FILE,
        usecols=[
            "graph_id",
            "page_id",
            "title",
            "url",
            "in_degree",
            "out_degree",
            "pagerank",
        ],
    )
    nodes["graph_id"] = nodes["graph_id"].astype("int64")
    nodes["page_id"] = nodes["page_id"].astype("int64")
    return nodes


@lru_cache(maxsize=1)
def load_graph_edges() -> pd.DataFrame:
    edges_file = ranking_backend.REPO_ROOT / "data" / "graph" / "edges.csv"
    ranking_backend.validate_file_exists(edges_file, "graph edges file")
    edges = pd.read_csv(edges_file, usecols=["source_graph_id", "target_graph_id"])
    edges["source_graph_id"] = edges["source_graph_id"].astype("int64")
    edges["target_graph_id"] = edges["target_graph_id"].astype("int64")
    return edges


@lru_cache(maxsize=1)
def graph_damping_factor() -> float:
    metadata_path = ranking_backend.REPO_ROOT / "data" / "graph" / "metadata.json"
    if not metadata_path.exists():
        return PAGERANK_DAMPING_FALLBACK
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return PAGERANK_DAMPING_FALLBACK
    try:
        return float(metadata.get("damping", PAGERANK_DAMPING_FALLBACK))
    except (TypeError, ValueError):
        return PAGERANK_DAMPING_FALLBACK


def _link_rows_from_graph_ids(graph_ids: list[int], limit: int = 12) -> list[dict[str, Any]]:
    if not graph_ids:
        return []
    nodes = load_graph_nodes().set_index("graph_id")
    rows = []
    for graph_id in graph_ids[: max(limit, 0)]:
        if graph_id not in nodes.index:
            continue
        row = nodes.loc[graph_id]
        rows.append(
            {
                "page_id": int(row["page_id"]),
                "title": str(row["title"]),
                "url": str(row["url"] or ""),
            }
        )
    return rows


def get_article_link_preview(page_id: int, limit: int = 12) -> dict[str, Any]:
    article = _article_by_page_id(page_id)
    graph_id = article.get("graph_id")
    if graph_id is None:
        return {
            "has_graph_node": False,
            "incoming_count": 0,
            "outgoing_count": 0,
            "incoming_preview": [],
            "outgoing_preview": [],
        }

    graph_id = int(graph_id)
    edges = load_graph_edges()
    incoming_ids = edges.loc[edges["target_graph_id"].eq(graph_id), "source_graph_id"].tolist()
    outgoing_ids = edges.loc[edges["source_graph_id"].eq(graph_id), "target_graph_id"].tolist()
    return {
        "has_graph_node": True,
        "incoming_count": len(incoming_ids),
        "outgoing_count": len(outgoing_ids),
        "incoming_preview": _link_rows_from_graph_ids(incoming_ids, limit=limit),
        "outgoing_preview": _link_rows_from_graph_ids(outgoing_ids, limit=limit),
    }


def get_experiment_article(page_id: int) -> dict[str, Any]:
    article = _article_by_page_id(page_id)
    article_text, source_path = _read_article_text(article["file_name"])
    article["document_text"] = article_text
    article["source_path"] = source_path
    article["link_preview"] = get_article_link_preview(page_id)
    return article


def get_article_baseline_ranking_entry(
    query_id: str,
    page_id: int,
    model_key: str | None = "hybrid",
) -> dict[str, Any]:
    record = get_experiment_query_record(query_id)
    _, ranking_df = ranking_backend.load_ranking_by_query_id(query_id)
    selected_ranking, metadata = _build_selected_ranking_view(ranking_df, model_key)
    selected_rows = selected_ranking.loc[selected_ranking["page_id"].eq(int(page_id))]
    if selected_rows.empty:
        raise ValueError(f"Page ID {page_id} is not present in ranking {query_id}.")

    row = selected_rows.iloc[0]
    article_url = ""
    graph_nodes = load_graph_nodes()
    matching_urls = graph_nodes.loc[graph_nodes["page_id"].eq(int(page_id)), "url"]
    if not matching_urls.empty:
        article_url = str(_clean_scalar(matching_urls.iloc[0]) or "")

    return {
        "query_id": record["query_id"],
        "query_text": record["query_text"],
        "result_file": record["result_file"],
        "total_results": int(len(ranking_df)),
        "selected_view_label": metadata["view_label"],
        "selected_rank_label": metadata["rank_label"],
        "selected_score_label": metadata["score_label"],
        "comparison_rank_label": metadata["comparison_rank_label"],
        "comparison_score_label": metadata["comparison_score_label"],
        "rank": int(row["selected_rank"]),
        "score": _float_or_zero(row.get("selected_score")),
        "comparison_rank": (
            int(row["hybrid_rank"])
            if metadata["comparison_rank_label"]
            else None
        ),
        "comparison_score": (
            _float_or_zero(row.get("hybrid_score"))
            if metadata["comparison_score_label"]
            else None
        ),
        "doc_id": int(row["doc_id"]),
        "page_id": int(row["page_id"]),
        "title": str(row["title"]),
        "file_name": str(row["file_name"]),
        "word_count": int(row["word_count"]),
        "char_count": int(row["char_count"]),
        "matched_query_tokens": str(row.get("matched_query_tokens") or ""),
        "pagerank_rank": _clean_scalar(row.get("pagerank_rank")),
        "article_url": article_url,
        "components": _score_component_rows(row, metadata["key"]),
    }


def parse_page_id_list(raw_value: str | None) -> list[int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    values: list[int] = []
    invalid_tokens: list[str] = []
    for token in re.split(r"[\s,;]+", raw):
        if not token:
            continue
        if not re.fullmatch(r"\d+", token):
            invalid_tokens.append(token)
            continue
        values.append(int(token))

    if invalid_tokens:
        raise ValueError("Invalid page ID token(s): " + ", ".join(invalid_tokens))

    return list(dict.fromkeys(values))


def build_link_edit_request(form: dict[str, Any]) -> LinkEditRequest:
    return LinkEditRequest(
        add_outgoing_page_ids=parse_page_id_list(form.get("add_outgoing_page_ids")),
        remove_outgoing_page_ids=parse_page_id_list(form.get("remove_outgoing_page_ids")),
        add_incoming_page_ids=parse_page_id_list(form.get("add_incoming_page_ids")),
        remove_incoming_page_ids=parse_page_id_list(form.get("remove_incoming_page_ids")),
    )


def _normalize_content_for_comparison(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def _bm25_score_for_document(query_tokens: list[str], document_tokens: list[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0

    bm25 = ranking_backend.load_bm25_index()
    frequencies = Counter(document_tokens)
    document_length = len(document_tokens)
    score = 0.0

    for token in query_tokens:
        token_frequency = frequencies.get(token, 0)
        if token_frequency <= 0:
            continue
        idf = bm25.idf.get(token)
        if idf is None:
            continue
        denominator = token_frequency + bm25.k1 * (
            1 - bm25.b + bm25.b * document_length / bm25.avgdl
        )
        score += float(idf) * (
            token_frequency * (bm25.k1 + 1) / denominator
        )

    return float(score)


def _matched_query_tokens_for_text(query_tokens: list[str], document_tokens: list[str]) -> str:
    if not query_tokens:
        return ""
    token_set = set(document_tokens)
    matched = [token for token in dict.fromkeys(query_tokens) if token in token_set]
    return ", ".join(matched)


@lru_cache(maxsize=1)
def _sentence_nlp():
    try:
        import spacy
    except ModuleNotFoundError:
        return None

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _split_sentences(text: str) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []

    nlp = _sentence_nlp()
    if nlp is None:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]

    if len(normalized) >= nlp.max_length:
        nlp.max_length = len(normalized) + 100
    document = nlp(normalized)
    return [sentence.text.strip() for sentence in document.sents if sentence.text.strip()]


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    words = sentence.split()
    return [
        " ".join(words[index : index + chunk_size])
        for index in range(0, len(words), chunk_size)
        if words[index : index + chunk_size]
    ]


def chunk_text_for_semantic_scoring(
    text: str,
    chunk_size: int = SEMANTIC_CHUNK_SIZE_WORDS,
) -> list[str]:
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in _split_sentences(text):
        sentence_word_count = len(sentence.split())
        if sentence_word_count <= 0:
            continue

        if sentence_word_count > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_word_count = 0
            chunks.extend(_split_long_sentence(sentence, chunk_size))
            continue

        if current_word_count + sentence_word_count <= chunk_size:
            current_sentences.append(sentence)
            current_word_count += sentence_word_count
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_word_count = sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    if not chunks:
        chunks = _split_long_sentence(str(text or ""), chunk_size)
    return [chunk for chunk in chunks if chunk.strip()]


def _semantic_score_for_edited_text(query: str, edited_text: str) -> float:
    chunks = chunk_text_for_semantic_scoring(edited_text)
    if not chunks:
        return 0.0

    model = ranking_backend.load_semantic_model()
    query_embedding = ranking_backend.encode_query_embedding(query)
    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32,
    ).astype(np.float32, copy=False)
    chunk_scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float32)
    return ranking_backend.score_document_from_chunk_scores(chunk_scores)


def _resolve_graph_page_ids(page_ids: list[int], label: str) -> dict[int, int]:
    if not page_ids:
        return {}
    nodes = load_graph_nodes()
    lookup = nodes.set_index("page_id")["graph_id"].to_dict()
    missing = [page_id for page_id in page_ids if page_id not in lookup]
    if missing:
        raise ValueError(
            f"{label} contains page ID(s) that are not in the hyperlink graph: "
            + ", ".join(str(page_id) for page_id in missing)
        )
    return {page_id: int(lookup[page_id]) for page_id in page_ids}


def _simulate_local_pagerank_updates(
    selected_page_id: int,
    link_edits: LinkEditRequest,
) -> tuple[dict[int, float], list[str], list[dict[str, Any]]]:
    requested_ids = (
        link_edits.add_outgoing_page_ids
        + link_edits.remove_outgoing_page_ids
        + link_edits.add_incoming_page_ids
        + link_edits.remove_incoming_page_ids
    )
    if not requested_ids:
        return {}, [], []

    nodes = load_graph_nodes()
    edges = load_graph_edges()
    page_to_graph = nodes.set_index("page_id")["graph_id"].to_dict()
    graph_to_page = nodes.set_index("graph_id")["page_id"].to_dict()
    graph_to_title = nodes.set_index("graph_id")["title"].to_dict()
    graph_to_pagerank = nodes.set_index("graph_id")["pagerank"].astype(float).to_dict()
    graph_to_out_degree = nodes.set_index("graph_id")["out_degree"].fillna(0).astype(int).to_dict()

    if selected_page_id not in page_to_graph:
        raise ValueError(
            f"Selected page_id {selected_page_id} is not in the hyperlink graph, "
            "so link-edit experiments cannot be simulated for it."
        )

    selected_graph_id = int(page_to_graph[selected_page_id])
    selected_baseline_pr = float(graph_to_pagerank.get(selected_graph_id, 0.0))
    selected_out_degree = int(graph_to_out_degree.get(selected_graph_id, 0))
    damping = graph_damping_factor()

    outgoing_graph_ids = set(
        edges.loc[edges["source_graph_id"].eq(selected_graph_id), "target_graph_id"].astype(int)
    )
    incoming_graph_ids = set(
        edges.loc[edges["target_graph_id"].eq(selected_graph_id), "source_graph_id"].astype(int)
    )

    add_outgoing = _resolve_graph_page_ids(
        link_edits.add_outgoing_page_ids,
        "Outgoing additions",
    )
    remove_outgoing = _resolve_graph_page_ids(
        link_edits.remove_outgoing_page_ids,
        "Outgoing removals",
    )
    add_incoming = _resolve_graph_page_ids(
        link_edits.add_incoming_page_ids,
        "Incoming additions",
    )
    remove_incoming = _resolve_graph_page_ids(
        link_edits.remove_incoming_page_ids,
        "Incoming removals",
    )

    warnings: list[str] = []

    def drop_self_links(page_to_graph_ids: dict[int, int], label: str) -> dict[int, int]:
        kept: dict[int, int] = {}
        for page_id, graph_id in page_to_graph_ids.items():
            if graph_id == selected_graph_id:
                warnings.append(f"Ignored self-link in {label}: {page_id}.")
            else:
                kept[page_id] = graph_id
        return kept

    add_outgoing = drop_self_links(add_outgoing, "outgoing additions")
    remove_outgoing = drop_self_links(remove_outgoing, "outgoing removals")
    add_incoming = drop_self_links(add_incoming, "incoming additions")
    remove_incoming = drop_self_links(remove_incoming, "incoming removals")

    effective_add_outgoing = {
        page_id: graph_id
        for page_id, graph_id in add_outgoing.items()
        if graph_id not in outgoing_graph_ids
    }
    already_outgoing = [
        page_id
        for page_id, graph_id in add_outgoing.items()
        if graph_id in outgoing_graph_ids
    ]
    if already_outgoing:
        warnings.append(
            "Outgoing additions already existed and had no effect: "
            + ", ".join(str(page_id) for page_id in already_outgoing)
        )

    effective_remove_outgoing = {
        page_id: graph_id
        for page_id, graph_id in remove_outgoing.items()
        if graph_id in outgoing_graph_ids
    }
    missing_outgoing = [
        page_id
        for page_id, graph_id in remove_outgoing.items()
        if graph_id not in outgoing_graph_ids
    ]
    if missing_outgoing:
        warnings.append(
            "Outgoing removals did not currently exist and had no effect: "
            + ", ".join(str(page_id) for page_id in missing_outgoing)
        )

    effective_add_incoming = {
        page_id: graph_id
        for page_id, graph_id in add_incoming.items()
        if graph_id not in incoming_graph_ids
    }
    already_incoming = [
        page_id
        for page_id, graph_id in add_incoming.items()
        if graph_id in incoming_graph_ids
    ]
    if already_incoming:
        warnings.append(
            "Incoming additions already existed and had no effect: "
            + ", ".join(str(page_id) for page_id in already_incoming)
        )

    effective_remove_incoming = {
        page_id: graph_id
        for page_id, graph_id in remove_incoming.items()
        if graph_id in incoming_graph_ids
    }
    missing_incoming = [
        page_id
        for page_id, graph_id in remove_incoming.items()
        if graph_id not in incoming_graph_ids
    ]
    if missing_incoming:
        warnings.append(
            "Incoming removals did not currently exist and had no effect: "
            + ", ".join(str(page_id) for page_id in missing_incoming)
        )

    updated_pagerank: dict[int, float] = {}
    effect_rows: list[dict[str, Any]] = []

    selected_pr = selected_baseline_pr
    for page_id, source_graph_id in effective_add_incoming.items():
        source_pr = float(graph_to_pagerank.get(source_graph_id, 0.0))
        source_out_degree = int(graph_to_out_degree.get(source_graph_id, 0)) + 1
        contribution = damping * source_pr / max(source_out_degree, 1)
        selected_pr += contribution
        effect_rows.append(
            {
                "direction": "incoming added",
                "page_id": page_id,
                "title": str(graph_to_title.get(source_graph_id, "")),
                "pagerank_delta": contribution,
            }
        )

    for page_id, source_graph_id in effective_remove_incoming.items():
        source_pr = float(graph_to_pagerank.get(source_graph_id, 0.0))
        source_out_degree = int(graph_to_out_degree.get(source_graph_id, 0))
        contribution = damping * source_pr / max(source_out_degree, 1)
        selected_pr -= contribution
        effect_rows.append(
            {
                "direction": "incoming removed",
                "page_id": page_id,
                "title": str(graph_to_title.get(source_graph_id, "")),
                "pagerank_delta": -contribution,
            }
        )

    if selected_pr != selected_baseline_pr:
        updated_pagerank[selected_page_id] = max(0.0, selected_pr)

    old_selected_out_degree = max(selected_out_degree, 1)
    new_selected_out_degree = max(
        selected_out_degree
        + len(effective_add_outgoing)
        - len(effective_remove_outgoing),
        1,
    )

    for page_id, target_graph_id in effective_add_outgoing.items():
        target_pr = float(graph_to_pagerank.get(target_graph_id, 0.0))
        contribution = damping * selected_baseline_pr / new_selected_out_degree
        updated_pagerank[page_id] = max(0.0, target_pr + contribution)
        effect_rows.append(
            {
                "direction": "outgoing added",
                "page_id": page_id,
                "title": str(graph_to_title.get(target_graph_id, "")),
                "pagerank_delta": contribution,
            }
        )

    for page_id, target_graph_id in effective_remove_outgoing.items():
        target_pr = float(graph_to_pagerank.get(target_graph_id, 0.0))
        contribution = damping * selected_baseline_pr / old_selected_out_degree
        updated_pagerank[page_id] = max(0.0, target_pr - contribution)
        effect_rows.append(
            {
                "direction": "outgoing removed",
                "page_id": page_id,
                "title": str(graph_to_title.get(target_graph_id, "")),
                "pagerank_delta": -contribution,
            }
        )

    return updated_pagerank, warnings, effect_rows


def _recompute_normalized_hybrid_scores(ranking_df: pd.DataFrame) -> pd.DataFrame:
    updated = ranking_df.copy()
    updated["bm25_score_norm"] = ranking_backend.min_max_normalize(
        updated["bm25_score_raw"].to_numpy()
    )
    updated["semantic_score_norm"] = ranking_backend.min_max_normalize(
        updated["semantic_score_raw"].to_numpy()
    )
    updated["pagerank_norm"] = ranking_backend.min_max_normalize(
        updated["pagerank"].to_numpy()
    )
    updated["bm25_weighted"] = ranking_backend.HYBRID_BM25_WEIGHT * updated["bm25_score_norm"]
    updated["semantic_weighted"] = (
        ranking_backend.HYBRID_SEMANTIC_WEIGHT * updated["semantic_score_norm"]
    )
    updated["pagerank_weighted"] = (
        ranking_backend.HYBRID_PAGERANK_WEIGHT * updated["pagerank_norm"]
    )
    updated["hybrid_score"] = (
        updated["bm25_weighted"]
        + updated["semantic_weighted"]
        + updated["pagerank_weighted"]
    )
    return updated


def _movement_statement(title: str, baseline_rank: int, experiment_rank: int) -> tuple[str, str, int]:
    rank_delta = baseline_rank - experiment_rank
    if rank_delta > 0:
        return (
            f"{title} moved up {rank_delta:,} position(s), from rank {baseline_rank:,} to rank {experiment_rank:,}.",
            "up",
            rank_delta,
        )
    if rank_delta < 0:
        return (
            f"{title} moved down {abs(rank_delta):,} position(s), from rank {baseline_rank:,} to rank {experiment_rank:,}.",
            "down",
            rank_delta,
        )
    return (
        f"{title} stayed at rank {baseline_rank:,}.",
        "unchanged",
        rank_delta,
    )


def _component_delta_rows(
    baseline_row: pd.Series,
    experiment_row: pd.Series,
    model_key: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (label, column, _) in enumerate(_ordered_score_component_specs(model_key)):
        before = _float_or_zero(baseline_row.get(column))
        after = _float_or_zero(experiment_row.get(column))
        rows.append(
            {
                "label": label,
                "before": before,
                "after": after,
                "delta": after - before,
                "is_primary": index == 0,
            }
        )
    return rows


def _attach_article_urls(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        results_df["article_url"] = pd.Series(dtype="object")
        return results_df
    nodes = load_graph_nodes()[["page_id", "url"]].drop_duplicates(subset=["page_id"])
    return results_df.merge(nodes.rename(columns={"url": "article_url"}), on="page_id", how="left")


def run_content_change_experiment(
    *,
    query_id: str,
    page_id: int,
    edited_text: str,
    link_edits: LinkEditRequest,
    model_key: str | None = "hybrid",
) -> dict[str, Any]:
    record = get_experiment_query_record(query_id)
    article = get_experiment_article(page_id)
    backend_record, baseline_ranking = ranking_backend.load_ranking_by_query_id(query_id)
    query_text = str(backend_record.get("query_text") or record["query_text"])
    metadata = _experiment_view_metadata(model_key)

    selected_mask = baseline_ranking["page_id"].eq(int(page_id))
    if not selected_mask.any():
        raise ValueError(
            f"Page ID {page_id} is not present in ranking {query_id}. "
            "Content experiments require the article to exist in the saved full ranking."
        )

    baseline_selected_view, _ = _build_selected_ranking_view(baseline_ranking, model_key)
    selected_baseline_row = baseline_selected_view.loc[
        baseline_selected_view["page_id"].eq(int(page_id))
    ].iloc[0].copy()

    working = baseline_ranking.copy()
    working["baseline_hybrid_rank"] = _numeric_series(working, "rank").fillna(len(working) + 1).astype("int64")
    working["baseline_hybrid_score"] = _numeric_series(working, "hybrid_score").fillna(0.0)

    original_text = str(article["document_text"] or "")
    content_changed = (
        _normalize_content_for_comparison(edited_text)
        != _normalize_content_for_comparison(original_text)
    )

    query_tokens = ranking_backend.tokenize_for_bm25(query_text)
    edited_tokens = ranking_backend.tokenize_for_bm25(edited_text)
    selected_index = working.index[selected_mask][0]

    if content_changed:
        working.at[selected_index, "bm25_score_raw"] = _bm25_score_for_document(
            query_tokens,
            edited_tokens,
        )
        working.at[selected_index, "semantic_score_raw"] = _semantic_score_for_edited_text(
            query_text,
            edited_text,
        )
        working.at[selected_index, "word_count"] = len(str(edited_text or "").split())
        working.at[selected_index, "char_count"] = len(str(edited_text or ""))
        working.at[selected_index, "matched_query_tokens"] = _matched_query_tokens_for_text(
            query_tokens,
            edited_tokens,
        )

    pagerank_updates, link_warnings, link_effect_rows = _simulate_local_pagerank_updates(
        int(page_id),
        link_edits,
    )
    for affected_page_id, updated_pagerank in pagerank_updates.items():
        affected_mask = working["page_id"].eq(int(affected_page_id))
        if not affected_mask.any():
            link_warnings.append(
                f"Page ID {affected_page_id} is in the graph but not in this ranking corpus; "
                "its local PageRank update was ignored."
            )
            continue
        working.loc[affected_mask, "pagerank"] = float(updated_pagerank)

    working = _recompute_normalized_hybrid_scores(working)
    working = working.sort_values(
        [
            "hybrid_score",
            "semantic_score_norm",
            "bm25_score_norm",
            "pagerank_norm",
            "title",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    working["experiment_hybrid_rank"] = np.arange(1, len(working) + 1)
    working["rank"] = working["experiment_hybrid_rank"]

    experiment_selected_view, _ = _build_selected_ranking_view(working, model_key)
    experiment_selected_view = experiment_selected_view.merge(
        baseline_selected_view.loc[:, ["page_id", "selected_rank", "selected_score"]].rename(
            columns={
                "selected_rank": "baseline_selected_rank",
                "selected_score": "baseline_selected_score",
            }
        ),
        on="page_id",
        how="left",
    )
    experiment_selected_view = experiment_selected_view.rename(
        columns={
            "selected_rank": "experiment_selected_rank",
            "selected_score": "experiment_selected_score",
        }
    )
    experiment_selected_view["rank_delta"] = (
        experiment_selected_view["baseline_selected_rank"].astype("int64")
        - experiment_selected_view["experiment_selected_rank"].astype("int64")
    )
    experiment_selected_view["comparison_rank_delta"] = (
        experiment_selected_view["baseline_hybrid_rank"].astype("int64")
        - experiment_selected_view["hybrid_rank"].astype("int64")
    )

    selected_experiment_row = experiment_selected_view.loc[
        experiment_selected_view["page_id"].eq(int(page_id))
    ].iloc[0].copy()
    baseline_rank = int(selected_baseline_row["selected_rank"])
    experiment_rank = int(selected_experiment_row["experiment_selected_rank"])
    statement, movement_direction, rank_delta = _movement_statement(
        str(article["title"]),
        baseline_rank,
        experiment_rank,
    )

    preview = experiment_selected_view.head(CONTENT_EXPERIMENT_PREVIEW_LIMIT).copy()
    if int(page_id) not in set(preview["page_id"].astype(int)):
        preview = pd.concat(
            [preview, selected_experiment_row.to_frame().T],
            ignore_index=True,
        )

    preview = _attach_article_urls(preview)
    preview["is_selected_article"] = preview["page_id"].astype(int).eq(int(page_id))

    return {
        "query": record,
        "article": article,
        "selected_view_label": metadata["view_label"],
        "selected_model_label": metadata["model_label"],
        "selected_rank_label": metadata["rank_label"],
        "selected_score_label": metadata["score_label"],
        "comparison_rank_label": metadata["comparison_rank_label"],
        "comparison_score_label": metadata["comparison_score_label"],
        "content_changed": content_changed,
        "link_warnings": link_warnings,
        "link_effect_rows": link_effect_rows,
        "statement": statement,
        "movement_direction": movement_direction,
        "rank_delta": rank_delta,
        "baseline_rank": baseline_rank,
        "experiment_rank": experiment_rank,
        "baseline_selected_score": _float_or_zero(selected_baseline_row.get("selected_score")),
        "experiment_selected_score": _float_or_zero(selected_experiment_row.get("experiment_selected_score")),
        "baseline_comparison_rank": (
            int(selected_experiment_row["baseline_hybrid_rank"])
            if metadata["comparison_rank_label"]
            else None
        ),
        "experiment_comparison_rank": (
            int(selected_experiment_row["hybrid_rank"])
            if metadata["comparison_rank_label"]
            else None
        ),
        "comparison_rank_delta": (
            int(selected_experiment_row["comparison_rank_delta"])
            if metadata["comparison_rank_label"]
            else None
        ),
        "baseline_comparison_score": (
            _float_or_zero(selected_experiment_row.get("baseline_hybrid_score"))
            if metadata["comparison_score_label"]
            else None
        ),
        "experiment_comparison_score": (
            _float_or_zero(selected_experiment_row.get("hybrid_score"))
            if metadata["comparison_score_label"]
            else None
        ),
        "component_deltas": _component_delta_rows(
            selected_baseline_row,
            selected_experiment_row,
            model_key,
        ),
        "preview_rows": [
            {key: _clean_scalar(value) for key, value in row.items()}
            for row in preview.to_dict(orient="records")
        ],
        "method_note": _experiment_method_note(metadata),
    }
