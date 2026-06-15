"""
Content feature extraction for the rank-relationship analysis.

Extracts, for every document in the corpus, the lexical/surface, readability and
semantic features described in section 3.5 of the thesis. The output is one row
per document keyed by ``doc_id`` (the file stem), so it joins directly to the
ranking tables produced by the retriever.

Design constraints (deliberate):
  * NO regex. Sentence/word/token boundaries come from spaCy; the retriever-side
    token count comes from the bm25s tokenizer. Syllables are counted with a
    pure character-iteration heuristic (vowel groups), not a pattern.
  * GPU-first. spaCy is asked to use the GPU and we prefer the transformer
    pipeline (``en_core_web_trf``); sentiment runs on a transformer model placed
    on CUDA when available. Everything falls back to CPU automatically.

The text fed to the feature extractor is identical to what the retriever indexes
(``"<title> <body>"``), so the features describe exactly the document the BM25 /
dense models ranked.

Usage
-----
    python feature_extraction.py
    python feature_extraction.py --corpus data/corpus_cleaned --out output/content_features.csv
    python feature_extraction.py --limit 200          # quick smoke test
    python feature_extraction.py --no-sentiment       # skip the transformer sentiment pass
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import bm25s
import spacy
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# spaCy pipeline: prefer the (accurate, GPU-friendly) transformer model, fall
# back to the small CNN model if it is not installed.
SPACY_TRF_MODEL = "en_core_web_trf"
SPACY_FALLBACK_MODEL = "en_core_web_sm"

# Transformer used for sentiment. 3-class (negative / neutral / positive); we
# collapse it to a single polarity in [-1, 1] as  P(pos) - P(neg).
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Part-of-speech tags counted as "content" words for lexical density / keywords.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# Keyword density is the share of the document occupied by its N most frequent
# content lemmas ("main terms").
KEYWORD_TOP_N = 5

# A "complex" word (Gunning Fog) is a >=3 syllable word that is not a proper noun.
COMPLEX_SYLLABLE_THRESHOLD = 3

VOWELS = frozenset("aeiouy")


# --------------------------------------------------------------------------- #
# Document text == exactly what the retriever indexes
# (mirrors retriever/full_retriever_script.py, without its regex hash strip)
# --------------------------------------------------------------------------- #
def _title_from_stem(stem: str) -> str:
    """Drop a trailing ``_<hex hash>`` (>=8 hex chars) and de-underscore.

    Pure string logic so this module stays regex-free; it reproduces
    ``_title_from_stem`` in the retriever for the corpus' filename scheme.
    """
    head, sep, tail = stem.rpartition("_")
    if sep and len(tail) >= 8 and all(c in "0123456789abcdefABCDEF" for c in tail):
        stem = head
    return stem.replace("_", " ").strip()


def load_corpus(corpus_dir: Path, limit: Optional[int] = None) -> Dict[str, str]:
    """``doc_id (file stem) -> "<title> <body>"`` in deterministic order."""
    files = sorted(corpus_dir.glob("*.txt"), key=lambda p: p.name)
    if limit is not None:
        files = files[:limit]

    docs: Dict[str, str] = {}
    for fp in files:
        title = _title_from_stem(fp.stem)
        body = fp.read_text(encoding="utf-8", errors="ignore").strip()
        docs[fp.stem] = f"{title} {body}".strip()

    if not docs:
        raise RuntimeError(f"no .txt documents found in {corpus_dir}")
    return docs


# --------------------------------------------------------------------------- #
# Syllables -- vowel-group heuristic, no regex
# --------------------------------------------------------------------------- #
def count_syllables(word: str) -> int:
    """Count syllables in a single (already lowercased) alphabetic word.

    Classic heuristic: a syllable is a maximal run of vowels; a silent trailing
    'e' is removed; every word has at least one syllable.
    """
    syllables = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def report_hardware() -> bool:
    """Print the GPU situation; return True if CUDA is actually usable."""
    cuda = torch.cuda.is_available()
    print("=" * 72)
    if cuda:
        print(f"[gpu] CUDA available -> {torch.cuda.get_device_name(0)}")
    else:
        print("[gpu] CUDA NOT available - running on CPU.")
        if "+cpu" in torch.__version__ or torch.version.cuda is None:
            print(f"      Your torch is a CPU-only build ({torch.__version__}).")
            print("      To actually use your GPU, reinstall a CUDA build, e.g.:")
            print("        pip install torch --index-url https://download.pytorch.org/whl/cu124")
            print("      and, for spaCy GPU:  pip install cupy-cuda12x")
    print("=" * 72)
    return cuda


def load_spacy() -> "spacy.language.Language":
    """Load the transformer pipeline on GPU if possible, else the small model."""
    on_gpu = spacy.prefer_gpu()  # no-op + False if no usable GPU
    print(f"[spacy] prefer_gpu() -> {on_gpu}")

    for model in (SPACY_TRF_MODEL, SPACY_FALLBACK_MODEL):
        try:
            nlp = spacy.load(model)
            print(f"[spacy] loaded '{model}'")
            break
        except OSError:
            if model == SPACY_TRF_MODEL:
                print(
                    f"[spacy] '{model}' not installed; for best NER/POS run:\n"
                    f"        python -m spacy download {model}"
                )
            else:
                print(
                    f"[spacy] fallback '{model}' not installed either; run:\n"
                    f"        python -m spacy download {model}"
                )
                raise
    # Corpus documents can be long; lift the default 1M char guard.
    nlp.max_length = 4_000_000
    return nlp


def load_sentiment(enabled: bool):
    """Return a transformers pipeline mapped onto the GPU, or None."""
    if not enabled:
        return None
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    try:
        clf = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device,
            top_k=None,          # return scores for all classes
            truncation=True,
            max_length=512,
        )
        print(f"[sentiment] loaded '{SENTIMENT_MODEL}' on "
              f"{'cuda:0' if device == 0 else 'cpu'}")
        return clf
    except Exception as exc:  # offline / download failure -> degrade gracefully
        print(f"[sentiment] could not load '{SENTIMENT_MODEL}' ({exc});"
              " sentiment_polarity will be empty. Re-run with network access"
              " or pass --no-sentiment to silence this.")
        return None


def sentence_polarity(clf, sentences: List[str]) -> List[float]:
    """Per-sentence polarity in [-1, 1] = P(positive) - P(negative)."""
    if not sentences:
        return []
    outputs = clf(sentences, batch_size=64)
    polarities: List[float] = []
    for scores in outputs:
        by_label = {s["label"].lower(): s["score"] for s in scores}
        pos = by_label.get("positive", 0.0)
        neg = by_label.get("negative", 0.0)
        polarities.append(pos - neg)
    return polarities


# --------------------------------------------------------------------------- #
# Feature computation for one spaCy Doc
# --------------------------------------------------------------------------- #
def extract_features(doc, bm25_token_count: int, sentiment_clf) -> Dict[str, object]:
    text = doc.text

    # --- words / sentences (spaCy boundaries) ---
    words = [t for t in doc if t.is_alpha]
    word_count = len(words)
    sentences = list(doc.sents)
    sentence_count = len(sentences)

    # --- Category 1: lexical & surface ---
    char_count = len(text)
    char_count_no_spaces = sum(1 for ch in text if not ch.isspace())
    avg_words_per_sentence = word_count / sentence_count if sentence_count else 0.0

    # --- Category 2: readability & complexity ---
    syllable_counts = [count_syllables(t.lower_) for t in words]
    total_syllables = sum(syllable_counts)
    avg_syllables_per_word = total_syllables / word_count if word_count else 0.0

    complex_words = sum(
        1
        for tok, syl in zip(words, syllable_counts)
        if syl >= COMPLEX_SYLLABLE_THRESHOLD and tok.pos_ != "PROPN"
    )

    if word_count and sentence_count:
        flesch_kincaid_grade = (
            0.39 * (word_count / sentence_count)
            + 11.8 * (total_syllables / word_count)
            - 15.59
        )
        gunning_fog_index = 0.4 * (
            (word_count / sentence_count) + 100.0 * (complex_words / word_count)
        )
    else:
        flesch_kincaid_grade = 0.0
        gunning_fog_index = 0.0

    content_tokens = [t for t in words if t.pos_ in CONTENT_POS]
    content_count = len(content_tokens)
    lexical_density = content_count / word_count if word_count else 0.0
    unique_words = len({t.lower_ for t in words})
    lexical_diversity = unique_words / word_count if word_count else 0.0

    # --- Category 3: semantic & linguistic ---
    content_lemmas = Counter(t.lemma_.lower() for t in content_tokens)
    top_terms = content_lemmas.most_common(KEYWORD_TOP_N)
    keyword_density = (
        sum(c for _, c in top_terms) / word_count if word_count else 0.0
    )

    ents = list(doc.ents)
    named_entity_count = len(ents)
    distinct_entity_types = len({e.label_ for e in ents})
    if ents:
        ent_freq = Counter(e.text for e in ents)
        most_freq_entity, most_freq_entity_count = ent_freq.most_common(1)[0]
    else:
        most_freq_entity, most_freq_entity_count = "", 0

    if sentiment_clf is not None and sentence_count:
        sent_texts = [s.text for s in sentences]
        sent_weights = [sum(1 for t in s if t.is_alpha) or 1 for s in sentences]
        sent_pols = sentence_polarity(sentiment_clf, sent_texts)
        total_w = sum(sent_weights)
        sentiment_polarity = (
            sum(p * w for p, w in zip(sent_pols, sent_weights)) / total_w
            if total_w
            else 0.0
        )
    else:
        sentiment_polarity = None

    return {
        # Category 1 - lexical & surface
        "word_count": word_count,
        "token_count": bm25_token_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "char_count_no_spaces": char_count_no_spaces,
        "avg_words_per_sentence": avg_words_per_sentence,
        # Category 2 - readability & complexity
        "avg_syllables_per_word": avg_syllables_per_word,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog_index": gunning_fog_index,
        "lexical_density": lexical_density,
        "lexical_diversity": lexical_diversity,
        # Category 3 - semantic & linguistic
        "keyword_density": keyword_density,
        "sentiment_polarity": sentiment_polarity,
        "named_entity_count": named_entity_count,
        "distinct_entity_types": distinct_entity_types,
        "most_frequent_entity": most_freq_entity,
        "most_frequent_entity_count": most_freq_entity_count,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus_cleaned",
        help="directory of *.txt documents (default: data/corpus_cleaned)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "content_features.csv",
        help="output CSV path",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N documents")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="spaCy nlp.pipe batch size")
    parser.add_argument("--n-process", type=int, default=1,
                        help="spaCy worker processes (CPU only; set to your physical "
                             "core count for a big speedup. Leave at 1 on GPU).")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="skip the transformer sentiment pass")
    args = parser.parse_args()

    on_gpu = report_hardware()
    nlp = load_spacy()

    n_process = args.n_process
    if on_gpu and n_process != 1:
        print("[spacy] GPU in use -> forcing --n-process 1 (multiprocessing is "
              "CPU-only).")
        n_process = 1
    sentiment_clf = load_sentiment(enabled=not args.no_sentiment)

    print(f"[corpus] loading from {args.corpus}")
    docs = load_corpus(args.corpus, limit=args.limit)
    doc_ids = list(docs.keys())
    texts = list(docs.values())
    print(f"[corpus] {len(doc_ids)} documents")

    # bm25s token count == the text "as the retriever sees it" (lowercased,
    # split, English stop-words removed). Tokenize the whole corpus once.
    print("[bm25] tokenizing (retriever-identical) for token_count")
    bm25_tokens = bm25s.tokenize(
        texts, stopwords="en", return_ids=False, show_progress=False
    )
    bm25_counts = [len(t) for t in bm25_tokens]

    rows: List[Dict[str, object]] = []
    print(f"[features] spaCy nlp.pipe: batch_size={args.batch_size}, "
          f"n_process={n_process}")
    pipe = nlp.pipe(texts, batch_size=args.batch_size, n_process=n_process)
    for doc_id, bm25_count, doc in tqdm(
        zip(doc_ids, bm25_counts, pipe), total=len(doc_ids), desc="features", unit="doc"
    ):
        feats = extract_features(doc, bm25_count, sentiment_clf)
        feats["doc_id"] = doc_id
        rows.append(feats)

    # Write out (doc_id first).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        cols = ["doc_id"] + [c for c in df.columns if c != "doc_id"]
        df = df[cols]
        df.to_csv(args.out, index=False)
    except ImportError:
        import csv

        fieldnames = ["doc_id"] + [k for k in rows[0] if k != "doc_id"]
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows x {len(rows[0])} columns -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
