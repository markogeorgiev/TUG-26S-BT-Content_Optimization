from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


REPO_ROOT_CANDIDATES = [
    Path(__file__).resolve().parent.parent,
    Path.cwd().resolve(),
    Path.cwd().resolve().parent,
]
REPO_ROOT = next(
    (
        path
        for path in REPO_ROOT_CANDIDATES
        if (path / "data").exists() and (path / "output").exists()
    ),
    Path(__file__).resolve().parent.parent,
)

BM25_DIR = REPO_ROOT / "data" / "bm25_index"
BM25_METADATA_FILE = BM25_DIR / "document_metadata.parquet"
BM25_TOKENIZED_CORPUS_FILE = BM25_DIR / "tokenized_corpus.pkl"
BM25_INDEX_FILE = BM25_DIR / "bm25_index.pkl"

EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
CHUNK_METADATA_FILE = EMBEDDINGS_DIR / "chunk_metadata.parquet"
CHUNK_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PARTS_DIR = EMBEDDINGS_DIR / "chunk_metadata_parts"
CHUNK_EMBEDDING_PARTS_DIR = EMBEDDINGS_DIR / "embedding_parts"

GRAPH_NODES_FILE = REPO_ROOT / "data" / "graph" / "nodes.csv"

RANKINGS_DIR = REPO_ROOT / "rankings"
INITIAL_RANKINGS_DIR = RANKINGS_DIR / "initial_rankings"
QUERIES_FILE = RANKINGS_DIR / "queries.json"

LOWERCASE = True
REMOVE_STOPWORDS = True
KEEP_NUMBERS = True
MIN_TOKEN_LENGTH = 2

HYBRID_BM25_WEIGHT = 0.4
HYBRID_SEMANTIC_WEIGHT = 0.4
HYBRID_PAGERANK_WEIGHT = 0.2

SEMANTIC_CHUNK_WEIGHTS = np.array([0.40, 0.35, 0.20], dtype=np.float32)
SEMANTIC_REMAINING_WEIGHT = 0.05

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

_CACHE: dict[str, Any] = {}


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_output_directories() -> None:
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
    INITIAL_RANKINGS_DIR.mkdir(parents=True, exist_ok=True)


def save_json_atomic(path: Path, data: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def normalize_text(text: str) -> str:
    normalized = str(text)
    if LOWERCASE:
        normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_text(text)
    token_pattern = (
        r"[a-z]+(?:['’][a-z]+)*|\d+(?:\.\d+)*"
        if KEEP_NUMBERS
        else r"[a-z]+(?:['’][a-z]+)*"
    )
    raw_tokens = re.findall(token_pattern, normalized)

    tokens: list[str] = []
    for token in raw_tokens:
        token = token.replace("'", "").replace("’", "").strip()
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if REMOVE_STOPWORDS and token in STOPWORDS:
            continue
        tokens.append(token)

    return tokens


def slugify_query(query: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(query))
    slug = slug.strip("-")
    if not slug:
        slug = "query"
    return slug[:limit].rstrip("-")


def query_identity_key(query: str) -> str:
    normalized = normalize_text(query)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array

    finite_mask = np.isfinite(array)
    normalized = np.zeros(array.shape, dtype=np.float32)
    if not finite_mask.any():
        return normalized

    finite_values = array[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if max_value > min_value:
        normalized[finite_mask] = (finite_values - min_value) / (max_value - min_value)
    return normalized


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_bm25_document_metadata() -> pd.DataFrame:
    if "bm25_document_metadata" not in _CACHE:
        validate_file_exists(BM25_METADATA_FILE, "BM25 document metadata file")
        metadata = pd.read_parquet(BM25_METADATA_FILE).sort_values("doc_id").reset_index(drop=True)
        _CACHE["bm25_document_metadata"] = metadata
    return _CACHE["bm25_document_metadata"]


def total_document_count() -> int:
    return int(len(load_bm25_document_metadata()))


def load_bm25_tokenized_corpus() -> list[list[str]]:
    if "bm25_tokenized_corpus" not in _CACHE:
        validate_file_exists(BM25_TOKENIZED_CORPUS_FILE, "BM25 tokenized corpus file")
        with BM25_TOKENIZED_CORPUS_FILE.open("rb") as handle:
            _CACHE["bm25_tokenized_corpus"] = pickle.load(handle)
    return _CACHE["bm25_tokenized_corpus"]


def load_bm25_index() -> BM25Okapi:
    if "bm25_index" not in _CACHE:
        validate_file_exists(BM25_INDEX_FILE, "BM25 index file")
        with BM25_INDEX_FILE.open("rb") as handle:
            _CACHE["bm25_index"] = pickle.load(handle)
    return _CACHE["bm25_index"]


def load_pagerank_scores() -> pd.DataFrame:
    if "pagerank_scores" not in _CACHE:
        validate_file_exists(GRAPH_NODES_FILE, "graph nodes file")
        pagerank_df = pd.read_csv(
            GRAPH_NODES_FILE,
            usecols=["page_id", "pagerank", "rank"],
        ).rename(columns={"rank": "pagerank_rank"})
        _CACHE["pagerank_scores"] = pagerank_df
    return _CACHE["pagerank_scores"]


def semantic_combined_files_available() -> bool:
    return CHUNK_METADATA_FILE.exists() and CHUNK_EMBEDDINGS_FILE.exists()


def iter_semantic_part_pairs() -> list[tuple[Path, Path]]:
    metadata_parts = sorted(CHUNK_METADATA_PARTS_DIR.glob("chunk_metadata_part_*.parquet"))
    if not metadata_parts:
        raise FileNotFoundError(
            "Missing semantic chunk metadata. Expected either combined files "
            f"({CHUNK_METADATA_FILE}, {CHUNK_EMBEDDINGS_FILE}) or part files in {CHUNK_METADATA_PARTS_DIR}."
        )

    part_pairs: list[tuple[Path, Path]] = []
    for metadata_part in metadata_parts:
        embedding_part = CHUNK_EMBEDDING_PARTS_DIR / metadata_part.name.replace(
            "chunk_metadata", "chunk_embeddings"
        ).replace(".parquet", ".npy")
        validate_file_exists(embedding_part, "semantic embedding part file")
        part_pairs.append((metadata_part, embedding_part))

    return part_pairs


def load_semantic_chunk_metadata() -> pd.DataFrame:
    if "semantic_chunk_metadata" not in _CACHE:
        validate_file_exists(CHUNK_METADATA_FILE, "semantic chunk metadata file")
        metadata = pd.read_parquet(
            CHUNK_METADATA_FILE,
            columns=["page_id", "title", "file_name", "chunk_index"],
        ).sort_values(["page_id", "chunk_index"]).reset_index(drop=True)
        _CACHE["semantic_chunk_metadata"] = metadata
    return _CACHE["semantic_chunk_metadata"]


def load_semantic_chunk_embeddings() -> np.ndarray:
    if "semantic_chunk_embeddings" not in _CACHE:
        validate_file_exists(CHUNK_EMBEDDINGS_FILE, "semantic chunk embeddings file")
        _CACHE["semantic_chunk_embeddings"] = np.load(CHUNK_EMBEDDINGS_FILE, mmap_mode="r")
    return _CACHE["semantic_chunk_embeddings"]


def load_semantic_model() -> SentenceTransformer:
    if "semantic_model" not in _CACHE:
        _CACHE["semantic_model"] = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
    return _CACHE["semantic_model"]


def encode_query_embedding(query: str) -> np.ndarray:
    model = load_semantic_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32, copy=False)[0]


def score_document_from_chunk_scores(chunk_scores: Sequence[float]) -> float:
    ranked_scores = np.sort(np.asarray(chunk_scores, dtype=np.float32))[::-1]
    if ranked_scores.size == 0:
        return 0.0

    weighted_score = 0.0
    used_weight = 0.0
    top_chunk_count = min(len(SEMANTIC_CHUNK_WEIGHTS), ranked_scores.size)

    if top_chunk_count:
        weighted_score += float(
            np.dot(ranked_scores[:top_chunk_count], SEMANTIC_CHUNK_WEIGHTS[:top_chunk_count])
        )
        used_weight += float(SEMANTIC_CHUNK_WEIGHTS[:top_chunk_count].sum())

    if ranked_scores.size > len(SEMANTIC_CHUNK_WEIGHTS):
        weighted_score += SEMANTIC_REMAINING_WEIGHT * float(
            ranked_scores[len(SEMANTIC_CHUNK_WEIGHTS) :].mean()
        )
        used_weight += SEMANTIC_REMAINING_WEIGHT

    return weighted_score / used_weight if used_weight else 0.0


def compute_semantic_document_scores(query: str) -> pd.DataFrame:
    query_embedding = encode_query_embedding(query)

    if semantic_combined_files_available():
        chunk_metadata = load_semantic_chunk_metadata()
        chunk_embeddings = load_semantic_chunk_embeddings()
        if len(chunk_metadata) != chunk_embeddings.shape[0]:
            raise ValueError(
                "Semantic chunk metadata row count does not match the chunk embedding row count."
            )

        chunk_scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float32)
        scored_chunks = chunk_metadata.copy()
        scored_chunks["semantic_chunk_score"] = chunk_scores
        semantic_scores = (
            scored_chunks.groupby(["page_id", "title", "file_name"], sort=False, dropna=False)[
                "semantic_chunk_score"
            ]
            .apply(score_document_from_chunk_scores)
            .reset_index(name="semantic_score_raw")
        )
        del scored_chunks, chunk_scores
        gc.collect()
        return semantic_scores

    score_rows: list[pd.DataFrame] = []
    for metadata_part, embedding_part in iter_semantic_part_pairs():
        metadata = pd.read_parquet(
            metadata_part,
            columns=["page_id", "title", "file_name", "chunk_index"],
        ).sort_values(["page_id", "chunk_index"])
        embeddings = np.load(embedding_part)
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Semantic part mismatch: {metadata_part} has {len(metadata)} rows but "
                f"{embedding_part} has {embeddings.shape[0]} embeddings."
            )

        part_scores = np.asarray(embeddings @ query_embedding, dtype=np.float32)
        scored_part = metadata.copy()
        scored_part["semantic_chunk_score"] = part_scores
        grouped = (
            scored_part.groupby(["page_id", "title", "file_name"], sort=False, dropna=False)[
                "semantic_chunk_score"
            ]
            .apply(score_document_from_chunk_scores)
            .reset_index(name="semantic_score_raw")
        )
        score_rows.append(grouped)
        del metadata, embeddings, part_scores, scored_part, grouped
        gc.collect()

    if not score_rows:
        raise FileNotFoundError("No semantic part files were found for document scoring.")

    semantic_scores = pd.concat(score_rows, ignore_index=True)
    return semantic_scores


def compute_bm25_scores(query_tokens: list[str], document_count: int) -> np.ndarray:
    if not query_tokens:
        return np.zeros(document_count, dtype=np.float32)
    bm25 = load_bm25_index()
    return np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)


def matched_query_tokens_for_doc(doc_id: int, query_tokens: list[str]) -> str:
    if not query_tokens:
        return ""
    tokenized_corpus = load_bm25_tokenized_corpus()
    token_set = set(tokenized_corpus[doc_id])
    matched = [token for token in dict.fromkeys(query_tokens) if token in token_set]
    return ", ".join(matched)


def build_hybrid_ranking(query: str, top_k: int | None = None) -> pd.DataFrame:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    document_metadata = load_bm25_document_metadata()
    tokenized_corpus = load_bm25_tokenized_corpus()
    if len(document_metadata) != len(tokenized_corpus):
        raise ValueError("BM25 document metadata row count does not match tokenized corpus size.")

    query_tokens = tokenize_for_bm25(query)
    bm25_scores = compute_bm25_scores(query_tokens, len(document_metadata))
    semantic_scores = compute_semantic_document_scores(query)
    pagerank_scores = load_pagerank_scores()

    results = document_metadata[
        ["doc_id", "page_id", "title", "file_name", "word_count", "char_count"]
    ].copy()
    results["bm25_score_raw"] = bm25_scores

    semantic_scores = semantic_scores[["page_id", "semantic_score_raw"]].copy()
    results = results.merge(semantic_scores, on="page_id", how="left")

    pagerank_scores = pagerank_scores[["page_id", "pagerank", "pagerank_rank"]].copy()
    results = results.merge(pagerank_scores, on="page_id", how="left")

    results["semantic_score_raw"] = results["semantic_score_raw"].fillna(0.0).astype(np.float32)
    results["pagerank"] = results["pagerank"].fillna(0.0).astype(np.float32)

    results["bm25_score_norm"] = min_max_normalize(results["bm25_score_raw"].to_numpy())
    results["semantic_score_norm"] = min_max_normalize(results["semantic_score_raw"].to_numpy())
    results["pagerank_norm"] = min_max_normalize(results["pagerank"].to_numpy())

    results["bm25_weighted"] = HYBRID_BM25_WEIGHT * results["bm25_score_norm"]
    results["semantic_weighted"] = HYBRID_SEMANTIC_WEIGHT * results["semantic_score_norm"]
    results["pagerank_weighted"] = HYBRID_PAGERANK_WEIGHT * results["pagerank_norm"]
    results["hybrid_score"] = (
        results["bm25_weighted"]
        + results["semantic_weighted"]
        + results["pagerank_weighted"]
    )

    results = results.sort_values(
        [
            "hybrid_score",
            "semantic_score_norm",
            "bm25_score_norm",
            "pagerank_norm",
            "title",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))

    if top_k is not None:
        top_k = min(top_k, len(results))
        results = results.head(top_k).copy()
    else:
        results = results.copy()
    results["matched_query_tokens"] = [
        matched_query_tokens_for_doc(int(doc_id), query_tokens)
        for doc_id in results["doc_id"].tolist()
    ]

    ordered_columns = [
        "rank",
        "hybrid_score",
        "doc_id",
        "page_id",
        "title",
        "file_name",
        "word_count",
        "char_count",
        "bm25_score_raw",
        "bm25_score_norm",
        "semantic_score_raw",
        "semantic_score_norm",
        "pagerank",
        "pagerank_norm",
        "bm25_weighted",
        "semantic_weighted",
        "pagerank_weighted",
        "pagerank_rank",
        "matched_query_tokens",
    ]
    return results[ordered_columns]


def load_queries_log() -> dict[str, Any]:
    if not QUERIES_FILE.exists():
        return {"schema_version": 1, "queries": []}
    raw = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    if "queries" not in raw or not isinstance(raw["queries"], list):
        raise ValueError(f"Unexpected queries log structure in {QUERIES_FILE}")
    return raw


def next_query_id(queries_log: dict[str, Any]) -> str:
    next_number = len(queries_log["queries"]) + 1
    return f"q{next_number:06d}"


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_result_path(result_file: str) -> Path:
    return REPO_ROOT / Path(result_file)


def normalize_query_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized_record = dict(record)
    query_text = str(normalized_record.get("query_text", ""))
    normalized_record.setdefault("query_key", query_identity_key(query_text))
    normalized_record.setdefault("query_slug", slugify_query(query_text))
    normalized_record.setdefault("stored_full_ranking", False)
    normalized_record.setdefault("stored_result_count", normalized_record.get("result_count"))
    return normalized_record


def iter_saved_query_records() -> list[dict[str, Any]]:
    queries_log = load_queries_log()
    return [normalize_query_record(record) for record in queries_log["queries"]]


def find_existing_query_record(query: str) -> dict[str, Any] | None:
    target_key = query_identity_key(query)
    target_slug = slugify_query(query)

    for record in iter_saved_query_records():
        record_key = str(record.get("query_key") or "")
        record_slug = str(record.get("query_slug") or "")
        if record_key == target_key or record_slug == target_slug:
            return record

    return None


def load_saved_ranking(record: dict[str, Any]) -> pd.DataFrame:
    result_file = record.get("result_file")
    if not result_file:
        raise FileNotFoundError(f"Query record is missing result_file: {record}")
    result_path = resolve_result_path(str(result_file))
    validate_file_exists(result_path, "saved ranking result file")
    return pd.read_parquet(result_path)


def get_query_suggestions(prefix: str = "", limit: int = 8) -> list[dict[str, Any]]:
    normalized_prefix = query_identity_key(prefix)
    records = sorted(
        iter_saved_query_records(),
        key=lambda record: (
            str(record.get("executed_at_utc") or ""),
            str(record.get("query_id") or ""),
        ),
        reverse=True,
    )

    if normalized_prefix:
        records = [
            record
            for record in records
            if normalized_prefix in str(record.get("query_key") or "")
            or normalized_prefix in normalize_text(str(record.get("query_text") or ""))
        ]

    suggestions: list[dict[str, Any]] = []
    for record in records[: max(limit, 0)]:
        suggestions.append(
            {
                "query_id": record.get("query_id"),
                "query_text": record.get("query_text"),
                "query_key": record.get("query_key"),
                "query_slug": record.get("query_slug"),
                "executed_at_utc": record.get("executed_at_utc"),
                "stored_result_count": record.get("stored_result_count") or record.get("result_count"),
                "stored_full_ranking": record.get("stored_full_ranking"),
                "weights": record.get("weights") or {},
                "result_file": record.get("result_file"),
            }
        )
    return suggestions


def get_query_record_by_id(query_id: str) -> dict[str, Any] | None:
    for record in iter_saved_query_records():
        if str(record.get("query_id")) == str(query_id):
            return record
    return None


def load_ranking_by_query_id(query_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    record = get_query_record_by_id(query_id)
    if record is None:
        raise FileNotFoundError(f"Unknown query_id: {query_id}")
    return record, load_saved_ranking(record)


def save_query_results(query_id: str, query: str, ranking_df: pd.DataFrame) -> Path:
    result_path = INITIAL_RANKINGS_DIR / f"{query_id}_{slugify_query(query)}.parquet"
    ranking_df.to_parquet(result_path, index=False)
    return result_path


def append_query_log(
    *,
    query_id: str,
    query: str,
    result_path: Path,
    result_count: int,
) -> None:
    queries_log = load_queries_log()
    queries_log["queries"].append(
        {
            "query_id": query_id,
            "query_text": query,
            "query_key": query_identity_key(query),
            "query_slug": slugify_query(query),
            "executed_at_utc": utc_timestamp(),
            "top_k": result_count,
            "result_count": result_count,
            "stored_result_count": result_count,
            "stored_full_ranking": True,
            "weights": {
                "bm25": HYBRID_BM25_WEIGHT,
                "semantic": HYBRID_SEMANTIC_WEIGHT,
                "pagerank": HYBRID_PAGERANK_WEIGHT,
            },
            "normalization": "min_max_per_metric_per_query",
            "semantic_document_aggregation": {
                "top_chunk_1": 0.40,
                "top_chunk_2": 0.35,
                "top_chunk_3": 0.20,
                "remaining_chunks_mean": 0.05,
            },
            "result_file": relative_to_repo(result_path),
        }
    )
    save_json_atomic(QUERIES_FILE, queries_log)


def get_or_create_query_ranking(query: str) -> tuple[dict[str, Any], pd.DataFrame, bool]:
    ensure_output_directories()
    existing_record = find_existing_query_record(query)
    if existing_record is not None:
        return existing_record, load_saved_ranking(existing_record), False

    queries_log = load_queries_log()
    query_id = next_query_id(queries_log)

    ranking_df = build_hybrid_ranking(query, top_k=None)
    result_path = save_query_results(query_id, query, ranking_df)
    append_query_log(
        query_id=query_id,
        query=query,
        result_path=result_path,
        result_count=len(ranking_df),
    )

    created_record = {
        "query_id": query_id,
        "query_text": query,
        "query_key": query_identity_key(query),
        "query_slug": slugify_query(query),
        "result_file": relative_to_repo(result_path),
        "result_count": len(ranking_df),
        "stored_result_count": len(ranking_df),
        "stored_full_ranking": True,
        "weights": {
            "bm25": HYBRID_BM25_WEIGHT,
            "semantic": HYBRID_SEMANTIC_WEIGHT,
            "pagerank": HYBRID_PAGERANK_WEIGHT,
        },
        "executed_at_utc": utc_timestamp(),
    }
    return created_record, ranking_df, True


def run_query(query: str, top_k: int, show_k: int) -> tuple[str, Path, pd.DataFrame]:
    ranking_start = time.time()
    record, ranking_df, created = get_or_create_query_ranking(query)
    result_path = resolve_result_path(str(record["result_file"]))

    elapsed = time.time() - ranking_start
    action = "Saved" if created else "Reused"
    print(f"\n{action} ranking for query {record['query_id']}: {record['query_text']}")
    print(f"  Results file: {result_path}")
    print(f"  Stored rows:  {len(ranking_df)}")
    print(f"  Elapsed:      {elapsed:.1f}s")

    if top_k > 0:
        preview_limit = min(max(top_k, 0), len(ranking_df))
        preview = ranking_df.head(preview_limit)[
            [
                "rank",
                "hybrid_score",
                "title",
                "page_id",
                "bm25_score_norm",
                "semantic_score_norm",
                "pagerank_norm",
            ]
        ]
        print(preview.head(max(show_k, 0)).to_string(index=False))

    return str(record["query_id"]), result_path, ranking_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid document rankings for Wikipedia Pets using BM25, MiniLM, and PageRank. "
            "Results are written under rankings/initial_rankings and each query is logged in rankings/queries.json."
        )
    )
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Query text to rank. Provide this flag multiple times to run multiple queries in one call.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help=(
            "Number of top-ranked documents to preview in the terminal. "
            "The full ranking is always saved to rankings/initial_rankings."
        ),
    )
    parser.add_argument(
        "--show-k",
        type=int,
        default=10,
        help="Number of top rows to print to stdout per query (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_output_directories()

    try:
        for query in args.query:
            run_query(query=query, top_k=args.top_k, show_k=args.show_k)
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
