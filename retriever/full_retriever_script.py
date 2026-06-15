"""
Full retrieval / ranking pipeline for the pets corpus.

Ranks every document in the corpus against every query for four models:

    bm25   - sparse lexical          (bm25s)
    tfidf  - sparse lexical          (scikit-learn)
    sbert  - dense bi-encoder        (sentence-transformers/all-MiniLM-L6-v2)
    e5     - dense bi-encoder        (intfloat/e5-base-v2)

The expensive artefacts are cached under ``retriever/cache`` so the heavy
work only ever runs ONCE for the whole project:

    * bm25  -> the built BM25 index (bm25s native save format)
    * tfidf -> the fitted vectorizer (pickle) + the doc-term matrix (npz)
    * sbert -> L2-normalised corpus embeddings (.npy)
    * e5    -> L2-normalised corpus embeddings (.npy)

On a second run every model loads its cache and only the (cheap) query side
plus the scoring is recomputed.

Everything that can run on the GPU does: embeddings are produced on CUDA and
the dense query/corpus dot-products are done as a single batched matmul on the
GPU. If no GPU is present it transparently falls back to CPU.

Usage
-----
    # build + rank everything (the once-per-project run)
    python retriever/full_retriever_script.py

    # or a single model
    python retriever/full_retriever_script.py --model bm25

Configuration is read from environment variables (optionally a .env file);
sensible project-relative defaults are used when they are not set.
"""

import argparse
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:  # .env is optional - defaults below work without it
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv simply not installed
    pass


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Speed knobs for the GPU matmuls (no-ops on CPU).
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# The retriever directory is the anchor; the project root is its parent.
RETRIEVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", RETRIEVER_DIR.parent)).resolve()


def _resolve(env_name: str, default: str) -> Path:
    value = os.getenv(env_name)
    p = Path(value) if value else Path(default)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# Directory of cleaned ``*.txt`` documents (one file == one document).
CORPUS_DIR = _resolve("CORPUS_PATH", "data/corpus_cleaned")
# JSON file with the queries (schema: {"queries": [{"query_id", "query_text"}]}).
QUERIES_PATH = _resolve("QUERIES_PATH", "rankings/queries.json")
# Where the ``rankings_<model>.csv`` files are written.
RANKINGS_DIR = _resolve("RANKINGS_DIR", "rankings/full_retriever")
# All caches live inside the retriever folder, as requested.
CACHE_DIR = _resolve("CACHE_DIR", "retriever/cache")

RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Encoding batch size for the dense models (per forward pass on the GPU).
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
E5_MODEL = "intfloat/e5-base-v2"

# ``Foo_Bar_1a2b3c4d5e.txt`` -> the trailing 10-hex-char content hash.
_HASH_SUFFIX = re.compile(r"_[0-9a-fA-F]{8,}$")


@dataclass
class Result:
    query_id: str
    doc_id: str
    doc_rank: int
    doc_score: float
    model: str


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_queries(path: Path) -> Tuple[List[str], List[str]]:
    """Return (query_ids, query_texts) preserving file order."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both the project schema and a plain list of strings.
    if isinstance(data, dict) and "queries" in data:
        items = data["queries"]
        query_ids = [str(q["query_id"]) for q in items]
        query_texts = [str(q["query_text"]) for q in items]
    else:
        query_texts = [str(q) for q in data]
        query_ids = [str(i) for i in range(len(query_texts))]

    return query_ids, query_texts


def _title_from_stem(stem: str) -> str:
    return _HASH_SUFFIX.sub("", stem).replace("_", " ").strip()


def load_corpus(corpus_dir: Path, limit: int | None = None) -> Dict[str, str]:
    """
    Load every ``*.txt`` document.

    The document id is the file stem (stable + unique); the indexed text is the
    human-readable title (derived from the filename) followed by the body.
    Files are processed in sorted order so the doc ordering - and therefore the
    cached embeddings / index - is deterministic across runs and machines.
    """
    files = sorted(corpus_dir.glob("*.txt"), key=lambda p: p.name)
    if limit is not None:
        files = files[:limit]

    docs: Dict[str, str] = {}
    for fp in tqdm(files, desc=f"reading {corpus_dir.name}", unit="doc"):
        doc_id = fp.stem
        title = _title_from_stem(doc_id)
        body = fp.read_text(encoding="utf-8", errors="ignore").strip()
        docs[doc_id] = f"{title} {body}".strip()

    if not docs:
        raise RuntimeError(f"no .txt documents found in {corpus_dir}")
    return docs


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def parse_results(model: str, query_id: str, scores: np.ndarray, doc_ids: List[str]):
    """Turn a score vector over the corpus into ranked ``Result`` rows."""
    ranked = np.argsort(-scores, kind="stable")
    return [
        Result(
            query_id=query_id,
            doc_id=doc_ids[idx],
            doc_rank=rank + 1,
            doc_score=float(scores[idx]),
            model=model,
        )
        for rank, idx in enumerate(ranked)
    ]


def rank_dense(
    model: str,
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    doc_ids: List[str],
    query_ids: List[str],
) -> List[Result]:
    """
    Score all queries against all documents with a single batched matmul.

    Embeddings are L2-normalised, so the dot product is cosine similarity.
    Runs on the GPU when available, then the ranking (argsort) is done on the
    GPU too and only the final indices/scores are pulled back to the CPU.
    """
    out: List[Result] = []
    corpus_t = torch.from_numpy(np.ascontiguousarray(corpus_emb)).to(DEVICE)
    query_t = torch.from_numpy(np.ascontiguousarray(query_emb)).to(DEVICE)

    n_docs = corpus_t.shape[0]
    with torch.inference_mode():
        for qi, query_id in enumerate(tqdm(query_ids, desc=f"scoring {model}", unit="q")):
            scores = query_t[qi] @ corpus_t.T  # (n_docs,)
            order = torch.argsort(scores, descending=True)
            order_cpu = order.cpu().numpy()
            scores_cpu = scores.cpu().numpy()
            out.extend(
                Result(
                    query_id=query_id,
                    doc_id=doc_ids[idx],
                    doc_rank=rank + 1,
                    doc_score=float(scores_cpu[idx]),
                    model=model,
                )
                for rank, idx in enumerate(order_cpu)
            )
    assert len(out) == len(query_ids) * n_docs
    return out


# --------------------------------------------------------------------------- #
# BM25 (sparse) - cached as a native bm25s index
# --------------------------------------------------------------------------- #
def _build_or_load_bm25(doc_ids: List[str], texts: List[str]) -> bm25s.BM25:
    index_dir = CACHE_DIR / "bm25_index"
    if (index_dir / "params.index.json").exists():
        try:
            print(f"[bm25] loading cached index from {index_dir}")
            return bm25s.BM25.load(str(index_dir), load_corpus=False)
        except Exception as exc:  # corrupt/partial cache -> rebuild
            print(f"[bm25] cache load failed ({exc}); rebuilding")

    print("[bm25] tokenizing corpus")
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    bm25 = bm25s.BM25()
    print("[bm25] indexing corpus")
    bm25.index(corpus_tokens)
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(str(index_dir))
    print(f"[bm25] saved index to {index_dir}")
    return bm25


def run_bm25(docs: Dict[str, str], query_ids: List[str], queries: List[str]):
    doc_ids = list(docs.keys())
    texts = list(docs.values())

    bm25 = _build_or_load_bm25(doc_ids, texts)

    query_tokens = bm25s.tokenize(queries, stopwords="en")
    # Retrieve the *full* ranking (k = corpus size) for every query.
    results, scores = bm25.retrieve(
        query_tokens, corpus=doc_ids, k=len(doc_ids), show_progress=True
    )

    out: List[Result] = []
    for qi, query_id in enumerate(query_ids):
        for rank in range(len(doc_ids)):
            out.append(
                Result(
                    query_id=query_id,
                    doc_id=str(results[qi, rank]),
                    doc_rank=rank + 1,
                    doc_score=float(scores[qi, rank]),
                    model="bm25",
                )
            )
    return out


# --------------------------------------------------------------------------- #
# TF-IDF (sparse) - cached vectorizer + doc-term matrix
# --------------------------------------------------------------------------- #
def _build_or_load_tfidf(texts: List[str]) -> Tuple[TfidfVectorizer, sp.csr_matrix]:
    vec_path = CACHE_DIR / "tfidf_vectorizer.pkl"
    mat_path = CACHE_DIR / "tfidf_doc_matrix.npz"

    if vec_path.exists() and mat_path.exists():
        print("[tfidf] loading cached vectorizer + doc matrix")
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        X = sp.load_npz(mat_path)
        return vectorizer, X

    print("[tfidf] fitting vectorizer + transforming corpus")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vectorizer.fit_transform(texts)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(mat_path, X)
    print(f"[tfidf] cached vectorizer -> {vec_path.name}, matrix -> {mat_path.name}")
    return vectorizer, X


def run_tfidf(docs: Dict[str, str], query_ids: List[str], queries: List[str]):
    doc_ids = list(docs.keys())
    texts = list(docs.values())

    vectorizer, X = _build_or_load_tfidf(texts)
    Q = vectorizer.transform(queries)

    # Both X and Q are L2-normalised by TfidfVectorizer, so the sparse dot
    # product is cosine similarity. Do it as one sparse matmul: (n_q x n_docs).
    sims = (Q @ X.T).toarray()

    out: List[Result] = []
    for qi, query_id in enumerate(tqdm(query_ids, desc="scoring tfidf", unit="q")):
        out.extend(parse_results("tfidf", query_id, sims[qi], doc_ids))
    return out


# --------------------------------------------------------------------------- #
# Dense models (SBERT / E5) - cached corpus embeddings
# --------------------------------------------------------------------------- #
def load_or_build_embeddings(
    model_name: str, texts: List[str], cache_file: Path
) -> np.ndarray:
    if cache_file.exists():
        print(f"[{cache_file.stem}] loading cached embeddings ({cache_file.name})")
        return np.load(cache_file)

    from sentence_transformers import SentenceTransformer

    print(f"[{cache_file.stem}] encoding {len(texts)} docs with {model_name} on {DEVICE}")
    model = SentenceTransformer(model_name, device=DEVICE)
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    print(f"[{cache_file.stem}] cached embeddings -> {cache_file.name}")
    return emb


def _encode_queries(model_name: str, queries: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=DEVICE)
    return model.encode(
        queries,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def run_sbert(docs: Dict[str, str], query_ids: List[str], queries: List[str]):
    doc_ids = list(docs.keys())
    texts = list(docs.values())

    corpus_emb = load_or_build_embeddings(
        SBERT_MODEL, texts, CACHE_DIR / "sbert_corpus.npy"
    )
    query_emb = _encode_queries(SBERT_MODEL, queries)
    return rank_dense("sbert", corpus_emb, query_emb, doc_ids, query_ids)


def run_e5(docs: Dict[str, str], query_ids: List[str], queries: List[str]):
    doc_ids = list(docs.keys())
    # E5 requires the "passage:" / "query:" instruction prefixes.
    passages = [f"passage: {t}" for t in docs.values()]
    prefixed_queries = [f"query: {q}" for q in queries]

    corpus_emb = load_or_build_embeddings(
        E5_MODEL, passages, CACHE_DIR / "e5_corpus.npy"
    )
    query_emb = _encode_queries(E5_MODEL, prefixed_queries)
    return rank_dense("e5", corpus_emb, query_emb, doc_ids, query_ids)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_results(model: str, results: List[Result]):
    df = pd.DataFrame([r.__dict__ for r in results])
    out_path = RANKINGS_DIR / f"rankings_{model}.csv"
    df.to_csv(out_path, index=False)
    print(f"saved {out_path.relative_to(PROJECT_ROOT)}  ({len(df):,} rows)")


RUNNERS = {
    "bm25": run_bm25,
    "tfidf": run_tfidf,
    "sbert": run_sbert,
    "e5": run_e5,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "bm25", "tfidf", "sbert", "e5"],
        help="model to run (default: all four)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only load the first N documents (smoke test; do NOT use for the real run)",
    )
    args = parser.parse_args()

    print(f"device          : {DEVICE}")
    print(f"corpus dir      : {CORPUS_DIR}")
    print(f"queries         : {QUERIES_PATH}")
    print(f"rankings out    : {RANKINGS_DIR}")
    print(f"cache dir       : {CACHE_DIR}")

    query_ids, queries = load_queries(QUERIES_PATH)
    docs = load_corpus(CORPUS_DIR, limit=args.limit)
    print(f"loaded {len(docs):,} documents, {len(queries)} queries")

    models = list(RUNNERS) if args.model == "all" else [args.model]
    for model in models:
        t0 = time.time()
        print(f"\n=== {model} ===")
        results = RUNNERS[model](docs, query_ids, queries)
        save_results(model, results)
        print(f"[{model}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()