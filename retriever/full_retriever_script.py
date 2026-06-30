"""
Full retrieval / ranking pipeline for the pets corpus.

Ranks every document in the corpus against every query for the models below.
The first four are the default ("all") set; the two rerankers are heavy and
therefore OPT-IN (selected explicitly with --model), never part of "all":

    bm25           - sparse lexical          (bm25s)
    tfidf          - sparse lexical          (scikit-learn)
    sbert          - dense bi-encoder        (sentence-transformers/all-MiniLM-L6-v2)
    e5             - dense bi-encoder        (intfloat/e5-base-v2)
    cross_encoder  - cross-encoder reranker  (cross-encoder/ms-marco-MiniLM-L-6-v2)  [opt-in]
    colbert        - late-interaction        (colbert-ir/colbertv2.0; transformers+torch)  [opt-in]

The expensive artefacts are cached under ``retriever/cache`` so the heavy
work only ever runs ONCE for the whole project:

    * bm25          -> the built BM25 index (bm25s native save format)
    * tfidf         -> the fitted vectorizer (pickle) + the doc-term matrix (npz)
    * sbert         -> L2-normalised corpus embeddings (.npy)
    * e5            -> L2-normalised corpus embeddings (.npy)
    * cross_encoder -> the full query x doc score matrix (.npy)  (no embeddings exist
                       for a cross-encoder; the joint scores are what gets reused)
    * colbert       -> the per-document multi-vector embeddings (.npy, object array)

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

    # the opt-in rerankers (run only these; not included in "all")
    python retriever/full_retriever_script.py --model cross_encoder
    python retriever/full_retriever_script.py --model colbert    # uses only transformers + torch

Configuration is read from environment variables (optionally a .env file);
sensible project-relative defaults are used when they are not set.
"""

import argparse
import json
import os
import pickle
import re
import string
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

# Cross-encoder reranker. It scores (query, doc) pairs jointly, so there are no
# reusable per-doc embeddings; what we cache is the full score matrix instead.
# (canonical HF id uses dashes: ms-marco-MiniLM-L-6-v2)
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CE_BATCH_SIZE = int(os.getenv("CE_BATCH_SIZE", "64"))

# ColBERT late-interaction model (multi-vector). Implemented directly on
# transformers + torch (NO pylate / colbert package): a BERT encoder plus
# ColBERT's 128-dim linear projection, scored with MaxSim. Each document becomes
# one matrix of L2-normalised token vectors; docs are scored against each query
# in chunks to bound memory on the full corpus.
COLBERT_MODEL = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
COLBERT_DIM = 128
COLBERT_QUERY_MAXLEN = int(os.getenv("COLBERT_QUERY_MAXLEN", "32"))
COLBERT_DOC_MAXLEN = int(os.getenv("COLBERT_DOC_MAXLEN", "220"))
COLBERT_CHUNK = int(os.getenv("COLBERT_CHUNK", "4096"))

# Set from --limit in main(); guards the heavy caches from being overwritten
# with a partial (smoke-test) corpus.
_LIMITED = False

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
# Cross-encoder reranker - cached as the full query x doc score matrix
# --------------------------------------------------------------------------- #
def _load_or_build_ce_scores(
    query_ids: List[str],
    queries: List[str],
    doc_ids: List[str],
    texts: List[str],
    cache_file: Path,
) -> np.ndarray:
    """Return a (n_queries x n_docs) matrix of cross-encoder scores.

    A cross-encoder has no standalone document embeddings - the score only
    exists for a (query, document) pair - so the reusable artefact is the score
    matrix itself. The cache is only trusted when its shape matches the current
    corpus/queries, and only written for a full (non --limit) run.
    """
    expected = (len(query_ids), len(doc_ids))
    if cache_file.exists():
        scores = np.load(cache_file)
        if scores.shape == expected:
            print(f"[cross_encoder] loading cached scores ({cache_file.name})")
            return scores
        print(f"[cross_encoder] cached scores shape {scores.shape} != {expected}; "
              "recomputing")

    from sentence_transformers import CrossEncoder

    print(f"[cross_encoder] scoring {expected[0]} queries x {expected[1]} docs "
          f"with {CROSS_ENCODER_MODEL} on {DEVICE}")
    ce = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE, max_length=512)

    scores = np.empty(expected, dtype=np.float32)
    for qi, query in enumerate(tqdm(queries, desc="scoring cross_encoder", unit="q")):
        pairs = [[query, text] for text in texts]
        preds = ce.predict(pairs, batch_size=CE_BATCH_SIZE, show_progress_bar=False)
        scores[qi] = np.asarray(preds, dtype=np.float32)

    if not _LIMITED:
        np.save(cache_file, scores)
        print(f"[cross_encoder] cached scores -> {cache_file.name}")
    return scores


def run_cross_encoder(docs: Dict[str, str], query_ids: List[str], queries: List[str]):
    doc_ids = list(docs.keys())
    texts = list(docs.values())

    scores = _load_or_build_ce_scores(
        query_ids, queries, doc_ids, texts, CACHE_DIR / "cross_encoder_scores.npy"
    )

    out: List[Result] = []
    for qi, query_id in enumerate(query_ids):
        out.extend(parse_results("cross_encoder", query_id, scores[qi], doc_ids))
    return out


# --------------------------------------------------------------------------- #
# ColBERT (late interaction) - implemented directly on transformers + torch
#
# ColBERTv2 = a BERT encoder + a linear projection to 128 dims, scored with
# MaxSim over per-token vectors. We reproduce the official recipe without the
# `colbert` / `pylate` packages, so nothing in the environment changes:
#   * query : "[CLS] [Q] <query> ..." padded to QUERY_MAXLEN with [MASK]
#             (ColBERT "mask augmentation"); every token vector is kept.
#   * doc   : "[CLS] [D] <doc> ..." truncated to DOC_MAXLEN; padding and
#             punctuation token vectors are dropped.
#   * vectors are L2-normalised; score(q, d) = sum_i max_j (q_i . d_j).
# --------------------------------------------------------------------------- #
class _ColBERT:
    """Minimal ColBERTv2 encoder: BERT backbone + 128-dim projection head."""

    def __init__(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        self.linear = self._load_projection(model_name).to(DEVICE)  # (DIM, 768)

        # Marker tokens inserted right after [CLS]: [Q]=[unused0], [D]=[unused1].
        self.q_marker_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        # Documents drop punctuation token vectors (ColBERT skiplist).
        self.skiplist = {
            self.tokenizer.convert_tokens_to_ids(sym) for sym in string.punctuation
        }

        if DEVICE == "cuda":
            self.bert = self.bert.half()
            self.linear = self.linear.half()

    @staticmethod
    def _load_projection(model_name: str) -> torch.Tensor:
        """Pull ColBERT's ``linear.weight`` (128x768) out of the checkpoint.

        AutoModel loads only the BERT backbone and silently drops the ColBERT
        projection head, so we read that single tensor straight from the saved
        weights file (safetensors first, then the legacy .bin).
        """
        from huggingface_hub import hf_hub_download

        try:
            from safetensors.torch import load_file

            state = load_file(hf_hub_download(model_name, "model.safetensors"))
        except Exception:
            state = torch.load(
                hf_hub_download(model_name, "pytorch_model.bin"), map_location="cpu"
            )

        for key, tensor in state.items():
            if key.endswith("linear.weight"):
                return tensor.float()
        raise RuntimeError(
            f"could not find ColBERT 'linear.weight' in {model_name} checkpoint"
        )

    def _project(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ids/mask -> L2-normalised (B, T, DIM) token embeddings."""
        hidden = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        projected = hidden @ self.linear.T
        return torch.nn.functional.normalize(projected, p=2, dim=2)

    @torch.inference_mode()
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """(Nq, QUERY_MAXLEN, DIM) on DEVICE; all token vectors kept."""
        # Prepend ". " so position 1 is free to hold the [Q] marker.
        enc = self.tokenizer(
            [". " + q for q in queries],
            padding="max_length",
            truncation=True,
            max_length=COLBERT_QUERY_MAXLEN,
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        ids[:, 1] = self.q_marker_id
        ids[ids == self.pad_id] = self.mask_id           # mask augmentation
        ids = ids.to(DEVICE)
        mask = torch.ones_like(ids)                      # attend over the full query
        return self._project(ids, mask).float()

    @torch.inference_mode()
    def encode_documents(self, texts: List[str]) -> List[np.ndarray]:
        """One (Ld_i, DIM) float16 array per doc; padding + punctuation dropped."""
        embeddings: List[np.ndarray] = []
        for start in tqdm(
            range(0, len(texts), EMBED_BATCH_SIZE), desc="encoding colbert", unit="batch"
        ):
            enc = self.tokenizer(
                [". " + t for t in texts[start:start + EMBED_BATCH_SIZE]],
                padding=True,
                truncation=True,
                max_length=COLBERT_DOC_MAXLEN,
                return_tensors="pt",
            )
            ids = enc["input_ids"]
            ids[:, 1] = self.d_marker_id
            ids, mask = ids.to(DEVICE), enc["attention_mask"].to(DEVICE)

            emb = self._project(ids, mask)               # (B, T, DIM)

            # Keep only real, non-punctuation token vectors per document.
            keep = mask.bool()
            for sym_id in self.skiplist:
                keep &= ids != sym_id
            keep_cpu = keep.cpu().numpy()
            emb_cpu = emb.half().cpu().numpy()
            for row in range(emb_cpu.shape[0]):
                embeddings.append(emb_cpu[row][keep_cpu[row]])
        return embeddings


def _load_or_build_colbert_embeddings(
    encoder: "_ColBERT", texts: List[str], cache_file: Path
) -> List[np.ndarray]:
    """Per-document multi-vector embeddings, cached as a numpy object array."""
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        if len(cached) == len(texts):
            print(f"[colbert] loading cached doc embeddings ({cache_file.name})")
            return list(cached)
        print(f"[colbert] cached embedding count {len(cached)} != {len(texts)}; "
              "re-encoding")

    print(f"[colbert] encoding {len(texts)} docs with {COLBERT_MODEL} on {DEVICE}")
    emb = encoder.encode_documents(texts)
    if not _LIMITED:
        np.save(cache_file, np.array(emb, dtype=object), allow_pickle=True)
        print(f"[colbert] cached doc embeddings -> {cache_file.name}")
    return emb


def _colbert_maxsim_scores(
    query_emb: torch.Tensor,          # (Lq, DIM) on DEVICE
    doc_emb_chunk: List[np.ndarray],  # list of (Ld_i, DIM)
) -> np.ndarray:
    """MaxSim of one query against a chunk of docs -> (len(chunk),) scores."""
    n = len(doc_emb_chunk)
    max_len = max(d.shape[0] for d in doc_emb_chunk)

    padded = torch.zeros((n, max_len, COLBERT_DIM), dtype=query_emb.dtype, device=DEVICE)
    valid = torch.zeros((n, max_len), dtype=torch.bool, device=DEVICE)
    for i, d in enumerate(doc_emb_chunk):
        ld = d.shape[0]
        padded[i, :ld] = torch.from_numpy(d).to(DEVICE, query_emb.dtype)
        valid[i, :ld] = True

    # sim[n, q, t] = query token q . doc token t
    sim = torch.einsum("qd,ntd->nqt", query_emb, padded)
    sim = sim.masked_fill(~valid.unsqueeze(1), float("-inf"))
    scores = sim.max(dim=2).values.sum(dim=1)  # max over doc tokens, sum over query
    return scores.float().cpu().numpy()


def run_colbert(docs: Dict[str, str], query_ids: List[str], queries: List[str]):
    doc_ids = list(docs.keys())
    texts = list(docs.values())

    encoder = _ColBERT(COLBERT_MODEL)
    doc_emb = _load_or_build_colbert_embeddings(
        encoder, texts, CACHE_DIR / "colbert_corpus.npy"
    )
    query_emb = encoder.encode_queries(queries)  # (Nq, Lq, DIM) on DEVICE

    out: List[Result] = []
    with torch.inference_mode():
        for qi, query_id in enumerate(tqdm(query_ids, desc="scoring colbert", unit="q")):
            scores = np.empty(len(doc_ids), dtype=np.float32)
            qe = query_emb[qi]
            for start in range(0, len(doc_ids), COLBERT_CHUNK):
                chunk = doc_emb[start:start + COLBERT_CHUNK]
                scores[start:start + len(chunk)] = _colbert_maxsim_scores(qe, chunk)
            out.extend(parse_results("colbert", query_id, scores, doc_ids))
    return out


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
    "cross_encoder": run_cross_encoder,
    "colbert": run_colbert,
}

# "all" runs only the four light models; the rerankers are heavy and opt-in.
DEFAULT_MODELS = ["bm25", "tfidf", "sbert", "e5"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        choices=["all"] + list(RUNNERS),
        help="model to run. 'all' = bm25/tfidf/sbert/e5; cross_encoder and "
             "colbert are heavy and must be selected explicitly.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only load the first N documents (smoke test; do NOT use for the real run)",
    )
    args = parser.parse_args()

    global _LIMITED
    _LIMITED = args.limit is not None

    print(f"device          : {DEVICE}")
    print(f"corpus dir      : {CORPUS_DIR}")
    print(f"queries         : {QUERIES_PATH}")
    print(f"rankings out    : {RANKINGS_DIR}")
    print(f"cache dir       : {CACHE_DIR}")

    query_ids, queries = load_queries(QUERIES_PATH)
    docs = load_corpus(CORPUS_DIR, limit=args.limit)
    print(f"loaded {len(docs):,} documents, {len(queries)} queries")

    models = DEFAULT_MODELS if args.model == "all" else [args.model]
    for model in models:
        t0 = time.time()
        print(f"\n=== {model} ===")
        results = RUNNERS[model](docs, query_ids, queries)
        save_results(model, results)
        print(f"[{model}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()