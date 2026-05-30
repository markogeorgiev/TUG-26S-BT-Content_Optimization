from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "corpus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "corpus_cleaned"

BACK_MATTER_HEADINGS = {
    "see also",
    "references",
    "notes",
    "footnotes",
    "citations",
    "sources",
    "bibliography",
    "works cited",
    "further reading",
    "external links",
}

TABLE_HEADER_TERMS = {
    "position",
    "name",
    "name of greyhound",
    "breeding",
    "trap",
    "sp",
    "time",
    "trainer",
    "review scores",
    "publication score",
    "score",
    "chart",
    "charts",
}

TABLE_SECTION_HEADINGS = {
    "final result",
    "results",
    "result",
    "race result",
    "race results",
    "draw",
    "distances",
    "review scores",
}


def split_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [block.strip() for block in re.split(r"\n\s*\n+", normalized) if block.strip()]


def normalize_heading(block: str) -> str:
    return " ".join(block.strip().split()).casefold()


def is_back_matter_heading(block: str) -> bool:
    if "\n" in block:
        return False
    return normalize_heading(block).strip(":") in BACK_MATTER_HEADINGS


def remove_back_matter(blocks: list[str]) -> list[str]:
    for index, block in enumerate(blocks):
        if is_back_matter_heading(block):
            return blocks[:index]
    return blocks


def line_word_counts(block: str) -> list[int]:
    return [len(line.split()) for line in block.splitlines() if line.strip()]


def has_sentence_shape(block: str) -> bool:
    words = block.split()
    if len(words) < 8:
        return False
    return bool(re.search(r"[.!?]\s*$", block.strip()))


def is_heading(block: str) -> bool:
    if "\n" in block:
        return False
    text = block.strip()
    if not text or len(text) > 80:
        return False
    if re.search(r"[.!?]$", text):
        return False
    return len(text.split()) <= 7


def is_table_like_block(block: str) -> bool:
    text = block.strip()
    if not text:
        return False

    normalized = normalize_heading(text).strip(":")
    if normalized in TABLE_HEADER_TERMS or normalized in TABLE_SECTION_HEADINGS:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = text.split()
    digit_tokens = len(re.findall(r"\b\d+(?:[.,:-]\d+)?[a-z%]*\b", text, flags=re.IGNORECASE))

    if len(lines) >= 3:
        counts = line_word_counts(text)
        avg_words = sum(counts) / len(counts)
        short_line_ratio = sum(1 for count in counts if count <= 4) / len(counts)
        if avg_words <= 4.5 and short_line_ratio >= 0.75:
            return True
        if digit_tokens >= 2 and avg_words <= 6:
            return True

    if len(words) <= 4 and not has_sentence_shape(text):
        return True

    if len(words) <= 8 and digit_tokens >= 1 and not has_sentence_shape(text):
        return True

    return False


def is_bullet_block(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("- ") for line in lines)


def bullet_items(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item:
                items.append(item)
    return items


def compact_bullet_items(items: list[str], chunk_size: int = 5) -> list[str]:
    compacted: list[str] = []
    for start in range(0, len(items), chunk_size):
        group = items[start : start + chunk_size]
        compacted.append(f"The list also includes {', '.join(group)}.")
    return compacted


def clean_blocks(blocks: list[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0

    while index < len(blocks):
        if is_bullet_block(blocks[index]):
            items: list[str] = []
            while index < len(blocks) and is_bullet_block(blocks[index]):
                items.extend(bullet_items(blocks[index]))
                index += 1

            cleaned.extend(compact_bullet_items(items))
            continue

        if not is_table_like_block(blocks[index]):
            cleaned.append(blocks[index])
            index += 1
            continue

        run_start = index
        while index < len(blocks) and is_table_like_block(blocks[index]):
            index += 1

        run_length = index - run_start
        previous_is_heading = bool(cleaned and is_heading(cleaned[-1]))
        next_is_prose = index < len(blocks) and has_sentence_shape(blocks[index])

        if run_length >= 3 or previous_is_heading or next_is_prose:
            if cleaned and is_heading(cleaned[-1]) and normalize_heading(cleaned[-1]) in TABLE_SECTION_HEADINGS:
                cleaned.pop()
            continue

        cleaned.extend(blocks[run_start:index])

    return cleaned


def clean_text(text: str) -> str:
    blocks = split_blocks(text)
    blocks = remove_back_matter(blocks)
    blocks = clean_blocks(blocks)
    return ("\n\n".join(blocks).strip() + "\n") if blocks else ""


def build_embedding_corpus(source_dir: Path, output_dir: Path) -> tuple[int, int]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    empty = 0
    for source_path in sorted(source_dir.glob("*.txt"), key=lambda path: path.name.casefold()):
        cleaned = clean_text(source_path.read_text(encoding="utf-8-sig"))
        if not cleaned.strip():
            empty += 1
            continue

        (output_dir / source_path.name).write_text(cleaned, encoding="utf-8")
        written += 1

    return written, empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prose-focused embedding corpus from cleaned Wikipedia text exports."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written, empty = build_embedding_corpus(args.source_dir, args.output_dir)
    print(f"Wrote {written} cleaned file(s) to {args.output_dir}")
    if empty:
        print(f"Skipped {empty} file(s) that became empty after cleaning")


if __name__ == "__main__":
    main()
