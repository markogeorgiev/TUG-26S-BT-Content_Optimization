"""
Per-model content-optimization experiments.

For each (query, article, model) we author an edit that follows the model-family
strategy, recompute ONLY that article's score under the target model, re-rank it
against the model's baseline ranking, and save the original-vs-modified content
feature figure to ``plots/rerankers/``. A results table is printed and written to
``plots/rerankers/results.csv``.

Strategy notes (and a hard constraint):
  * Every neural model truncates long documents (ColBERT ~220 tokens, SBERT ~256,
    E5 / Cross-Encoder ~512), so the optimized text is PREPENDED - appending to a
    multi-thousand-word article would never reach the encoder.
  * Lexical (TF-IDF): concise, exact query terms, healthy keyword density, lean.
  * Dense (SBERT / E5): comprehensive, entity-rich, conceptual; no keyword stuffing.
  * Cross-Encoder: tight, directly relevant; trim the tail.
  * ColBERT: natural, well-articulated, contextual phrasing.
"""

from __future__ import annotations

import base64
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.services import retrieval_backend as backend
from ui.services.content_feature_extraction import compare_features
from ui.services.feature_comparison_plot import build_comparison_figure

OUT_DIR = PROJECT_ROOT / "plots" / "rerankers"


def head_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def slug(text: str, maxlen: int = 48) -> str:
    keep = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:maxlen]


# --------------------------------------------------------------------------- #
# Authored edits. Each builder takes the original body and returns the edited
# body (the article title is prepended automatically when scoring).
# --------------------------------------------------------------------------- #
def jorts_tfidf(body: str) -> str:
    boost = (
        "Jorts is the most popular orange cat. Jorts, an orange cat, became famous online as the "
        "most popular orange cat, and people searching for the most popular orange cat reliably find "
        "Jorts. As an orange cat, Jorts the orange cat is widely called the most popular orange cat."
    )
    return f"{boost} {head_words(body, 110)}"


def indian_spitz_tfidf(body: str) -> str:
    boost = (
        "The Indian Spitz is one of the best dogs for small apartments. As a small dog breed, the "
        "Indian Spitz suits small apartments and is recommended among the best dogs for apartment "
        "living, a compact small dog well matched to small apartments."
    )
    return f"{boost} {head_words(body, 110)}"


def goldfish_sbert(body: str) -> str:
    boost = (
        "The goldfish (Carassius auratus) is one of the most widely kept aquatic companions in the "
        "world and a classic choice for a home aquarium. Hardy, long-lived, and undemanding, a "
        "goldfish can live well over a decade in a maintained tank with filtration, regular water "
        "changes, and a varied diet. Beginners and families often start with goldfish because they "
        "tolerate a range of conditions, recognise the people who feed them, and adapt readily to "
        "indoor life. From the common goldfish to fancy varieties such as the Ryukin, Oranda, "
        "Fantail, Shubunkin, and Black Moor, these gentle freshwater animals are kept as calm, "
        "low-interaction household companions in tanks, bowls, and garden ponds."
    )
    return f"{boost} {body}"


def frogs_sbert(body: str) -> str:
    boost = (
        "Keeping frogs in captivity is among the most affordable and low-maintenance ways to care "
        "for an exotic companion. Species such as the African dwarf frog, White's tree frog, and "
        "American green tree frog need only a modest enclosure, inexpensive substrate, and a simple "
        "diet of crickets or aquatic pellets. Their upkeep costs little: a small terrarium, occasional "
        "feeders, and basic heating or lighting are usually enough, and many amphibians live for "
        "years on a minimal budget. Quiet, compact, and undemanding, frogs are frequently "
        "recommended to first-time and budget-conscious keepers seeking an inexpensive, easy pet."
    )
    return f"{boost} {body}"


def fancy_rat_e5(body: str) -> str:
    boost = (
        "The fancy rat is a domesticated companion kept almost entirely indoors, thriving as a "
        "full-time household animal. Living in cages within the home, fancy rats are intelligent, "
        "social, and clean, bonding closely with their owners and needing no outdoor access at all. "
        "They spend their whole lives indoors - exercising in the home, learning tricks, and "
        "interacting with people - which makes them ideal for apartments and houses where an animal "
        "must stay inside permanently. As domesticated descendants of the brown rat, fancy rats are "
        "quiet, space-efficient indoor companions suited to year-round life inside the home."
    )
    return f"{boost} {body}"


def jack_dempsey_e5(body: str) -> str:
    boost = (
        "The Jack Dempsey (Rocio octofasciata) is a freshwater cichlid commonly kept in medium-sized "
        "aquariums. A single adult suits a medium aquarium of roughly 110 to 200 litres (about 30 to "
        "55 gallons), where it establishes a territory among rocks, driftwood, and hardy plants. "
        "Hardy and characterful, this Central American cichlid is a popular centrepiece for mid-range "
        "tanks, tolerating a range of water conditions and pairing with similarly sized tankmates. "
        "Aquarists stocking a medium aquarium often consider the Jack Dempsey for its manageable "
        "adult size, iridescent coloration, and suitability to moderately sized freshwater setups."
    )
    return f"{boost} {body}"


def beagle_colbert(body: str) -> str:
    boost = (
        "The beagle is widely regarded as one of the best companions for families with children. "
        "Gentle, even-tempered, and endlessly playful, beagles form warm bonds with kids and tolerate "
        "the noise and bustle of a busy household with remarkable patience. Their sturdy build lets "
        "them keep up with energetic children without being fragile, while their friendly, "
        "people-oriented nature keeps them affectionate rather than aggressive. As sociable pack "
        "animals, beagles dislike being alone and happily fold into a child's daily routine, making "
        "them a thoughtful choice for a family seeking a kind, child-friendly dog."
    )
    return f"{boost} {body}"


def ragdoll_colbert(body: str) -> str:
    boost = (
        "The Ragdoll is celebrated as one of the calmest and most placid of all cat breeds. Famously "
        "relaxed, it tends to go limp and content when picked up, and is known for a serene, gentle "
        "temperament that rarely tips into nervous or excitable behaviour. Quiet, affectionate, and "
        "undemanding, the Ragdoll prefers a tranquil indoor life, following its people calmly from "
        "room to room and settling peacefully on a lap. Among cat breeds chosen for a soothing, "
        "mellow disposition, the Ragdoll is consistently described as one of the most laid-back and "
        "easygoing companions."
    )
    return f"{boost} {body}"


def goldfish_cross_encoder(body: str) -> str:
    boost = (
        "Goldfish are one of the best pets for students. They are inexpensive, take up little space, "
        "need only a few minutes of care a day, and stay quiet during study and sleep, which makes "
        "them ideal for a dorm room or a shared student flat. A goldfish needs no walking, can be left "
        "with the right setup over a short trip, and costs very little to feed and house, so a student "
        "on a tight budget and a busy schedule can keep one easily."
    )
    return f"{boost} {head_words(body, 70)}"


def conure_cross_encoder(body: str) -> str:
    boost = (
        "Yes, birds make good pets, and the conure is a great example. Conures are affectionate, "
        "sociable, and intelligent parrots that bond strongly with their owners, making them "
        "rewarding companion birds. They are playful, colourful, and interactive, enjoy learning "
        "tricks and sounds, and form lasting attachments to the people who care for them, which is "
        "exactly why many people consider birds such good pets."
    )
    return f"{boost} {head_words(body, 70)}"


EXPERIMENTS = [
    ("The most popular orange cat", "Jorts (cat)", "tfidf", jorts_tfidf),
    ("Best dogs for small apartments", "Indian Spitz", "tfidf", indian_spitz_tfidf),
    ("Can you keep a fish as a pet?", "Goldfish", "sbert", goldfish_sbert),
    ("Cheapest animal to own", "Frogs in captivity", "sbert", frogs_sbert),
    ("Pets that can live indoors full-time", "Fancy rat", "e5", fancy_rat_e5),
    ("Fish for medium-sized aquarium", "Jack Dempsey (fish)", "e5", jack_dempsey_e5),
    ("Best pets for kids", "Beagle", "colbert", beagle_colbert),
    ("Calmest cat breeds", "Ragdoll", "colbert", ragdoll_colbert),
    ("Best pets for students", "Goldfish", "cross_encoder", goldfish_cross_encoder),
    ("Are birds good pets?", "Conure", "cross_encoder", conure_cross_encoder),
]


def rerank_one(query: str, model_key: str, doc_id: str, edited_indexed: str):
    baseline = backend.rank_corpus(query, model_key).set_index("doc_id")
    old_rank = int(baseline.loc[doc_id, "rank"])
    old_score = float(baseline.loc[doc_id, "score"])
    new_score = backend.score_edited_document(query, model_key, edited_indexed)
    scores = baseline["score"].copy()
    scores.loc[doc_id] = new_score
    reranked = scores.sort_values(ascending=False, kind="stable").reset_index()
    new_rank = int(reranked.index[reranked["doc_id"] == doc_id][0]) + 1
    return old_rank, new_rank, old_score, new_score


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for query, title, model_key, builder in EXPERIMENTS:
        info = backend.model_info(model_key)
        article = backend.resolve_article(title)
        if article is None:
            print(f"[skip] no article match for {title!r}")
            continue
        doc_id = article["doc_id"]

        original_body = backend.article_body(doc_id)
        edited_body = builder(original_body)
        original_indexed = backend.indexed_text(doc_id)
        edited_indexed = backend.indexed_text_from_body(doc_id, edited_body)

        old_rank, new_rank, old_score, new_score = rerank_one(
            query, model_key, doc_id, edited_indexed,
        )

        comparison = compare_features(original_indexed, edited_indexed, with_sentiment=False)
        fig_title = (
            f"{article['title']}  -  {info.label}\n"
            f"\"{query}\"   rank {old_rank:,} -> {new_rank:,}   "
            f"({'+' if new_rank < old_rank else ''}{old_rank - new_rank:,})"
        )
        data_uri = build_comparison_figure(comparison, title=fig_title)
        fig_path = OUT_DIR / f"{model_key}__{slug(query)}__{slug(article['title'])}.png"
        if data_uri:
            fig_path.write_bytes(base64.b64decode(data_uri.split(",", 1)[1]))

        arrow = "UP" if new_rank < old_rank else ("DOWN" if new_rank > old_rank else "=")
        print(f"[{info.label:13}] {title:20} {query[:32]:32} "
              f"{old_rank:>6} -> {new_rank:>6}  {arrow:>4}  "
              f"score {old_score:.4f} -> {new_score:.4f}  | {fig_path.name}")
        results.append({
            "query": query, "article": article["title"], "model": info.label,
            "old_rank": old_rank, "new_rank": new_rank,
            "rank_change": old_rank - new_rank,
            "old_score": round(old_score, 6), "new_score": round(new_score, 6),
            "figure": fig_path.name,
        })

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[done] {len(results)} experiments -> {csv_path}")


if __name__ == "__main__":
    main()
