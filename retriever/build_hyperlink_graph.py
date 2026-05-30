from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "output" / "wikipedia-pets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "graph"


@dataclass(frozen=True)
class NodeRecord:
    graph_id: int
    page_id: int
    title: str
    canonical_title: str
    kind: str
    url: str
    text_file: str
    fetch_status: str
    failure_count: int


def normalize_lookup(value: str) -> str:
    return value.strip().casefold()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_page_index(page_index_path: Path) -> list[dict]:
    if not page_index_path.exists():
        raise FileNotFoundError(f"Missing page index: {page_index_path}")

    return json.loads(page_index_path.read_text(encoding="utf-8-sig"))


def load_nodes(
    page_index_path: Path,
    include_categories: bool,
    completed_only: bool,
) -> tuple[list[NodeRecord], dict[str, int]]:
    allowed_kinds = {"page", "category"} if include_categories else {"page"}
    page_index = read_page_index(page_index_path)
    filtered_rows: list[dict] = []

    for row in page_index:
        kind = str(row.get("kind", ""))
        if kind not in allowed_kinds:
            continue

        fetch_status = str(row.get("fetch_status", "pending"))
        if completed_only and fetch_status != "completed":
            continue

        filtered_rows.append(row)

    filtered_rows.sort(key=lambda row: normalize_lookup(str(row.get("title", ""))))

    nodes: list[NodeRecord] = []
    title_to_graph_id: dict[str, int] = {}

    for graph_id, row in enumerate(filtered_rows):
        title = str(row["title"])
        if title in title_to_graph_id:
            raise ValueError(f"Duplicate title in page index: {title}")

        title_to_graph_id[title] = graph_id
        nodes.append(
            NodeRecord(
                graph_id=graph_id,
                page_id=int(row["page_id"]),
                title=title,
                canonical_title=str(row.get("canonical_title") or ""),
                kind=str(row.get("kind") or ""),
                url=str(row.get("url") or ""),
                text_file=str(row.get("text_file") or ""),
                fetch_status=str(row.get("fetch_status") or "pending"),
                failure_count=int(row.get("failure_count") or 0),
            )
        )

    return nodes, title_to_graph_id


def build_adjacency(
    links_path: Path,
    title_to_graph_id: dict[str, int],
    node_count: int,
) -> tuple[list[list[int]], list[int], list[int], int]:
    if not links_path.exists():
        raise FileNotFoundError(f"Missing links CSV: {links_path}")

    adjacency = [[] for _ in range(node_count)]
    in_degree = [0] * node_count
    out_degree = [0] * node_count
    edge_count = 0

    with links_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_title = row["source_title"]
            target_title = row["target_title"]
            source_id = title_to_graph_id.get(source_title)
            target_id = title_to_graph_id.get(target_title)

            if source_id is None or target_id is None or source_id == target_id:
                continue

            adjacency[source_id].append(target_id)
            out_degree[source_id] += 1
            in_degree[target_id] += 1
            edge_count += 1

    return adjacency, in_degree, out_degree, edge_count


def compute_pagerank(
    adjacency: Sequence[Sequence[int]],
    damping: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], int, float]:
    node_count = len(adjacency)
    if node_count == 0:
        return [], 0, 0.0

    scores = [1.0 / node_count] * node_count
    base_score = (1.0 - damping) / node_count
    dangling_nodes = [node_id for node_id, targets in enumerate(adjacency) if not targets]
    delta = 0.0

    for iteration in range(1, max_iterations + 1):
        next_scores = [base_score] * node_count
        dangling_mass = damping * sum(scores[node_id] for node_id in dangling_nodes) / node_count

        if dangling_mass:
            for node_id in range(node_count):
                next_scores[node_id] += dangling_mass

        for source_id, targets in enumerate(adjacency):
            if not targets:
                continue

            distributed_score = damping * scores[source_id] / len(targets)
            for target_id in targets:
                next_scores[target_id] += distributed_score

        delta = sum(abs(next_scores[node_id] - scores[node_id]) for node_id in range(node_count))
        scores = next_scores

        if delta < tolerance:
            break

    total_score = sum(scores)
    if total_score:
        scores = [score / total_score for score in scores]

    return scores, iteration, delta


def compute_ranks(nodes: Sequence[NodeRecord], scores: Sequence[float]) -> list[int]:
    ordered_node_ids = sorted(
        range(len(nodes)),
        key=lambda node_id: (-scores[node_id], normalize_lookup(nodes[node_id].title)),
    )
    ranks = [0] * len(nodes)

    for rank, node_id in enumerate(ordered_node_ids, start=1):
        ranks[node_id] = rank

    return ranks


def iter_edges(adjacency: Sequence[Sequence[int]]) -> Iterator[tuple[int, int]]:
    for source_id, targets in enumerate(adjacency):
        for target_id in targets:
            yield source_id, target_id


def write_nodes_csv(
    path: Path,
    nodes: Sequence[NodeRecord],
    in_degree: Sequence[int],
    out_degree: Sequence[int],
    scores: Sequence[float],
    ranks: Sequence[int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "graph_id",
                "page_id",
                "title",
                "canonical_title",
                "kind",
                "fetch_status",
                "failure_count",
                "url",
                "text_file",
                "in_degree",
                "out_degree",
                "pagerank",
                "rank",
            ]
        )

        for node in nodes:
            node_id = node.graph_id
            writer.writerow(
                [
                    node.graph_id,
                    node.page_id,
                    node.title,
                    node.canonical_title,
                    node.kind,
                    node.fetch_status,
                    node.failure_count,
                    node.url,
                    node.text_file,
                    in_degree[node_id],
                    out_degree[node_id],
                    f"{scores[node_id]:.17g}",
                    ranks[node_id],
                ]
            )


def write_edges_csv(path: Path, adjacency: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_graph_id", "target_graph_id"])
        writer.writerows(iter_edges(adjacency))


def write_metadata_json(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_sqlite_database(
    path: Path,
    nodes: Sequence[NodeRecord],
    adjacency: Sequence[Sequence[int]],
    in_degree: Sequence[int],
    out_degree: Sequence[int],
    scores: Sequence[float],
    ranks: Sequence[int],
    metadata: dict,
) -> None:
    if path.exists():
        path.unlink()

    connection = sqlite3.connect(path)

    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")

        connection.execute(
            """
            CREATE TABLE nodes (
                graph_id INTEGER PRIMARY KEY,
                page_id INTEGER NOT NULL UNIQUE,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                canonical_title TEXT NOT NULL,
                normalized_canonical_title TEXT NOT NULL,
                kind TEXT NOT NULL,
                fetch_status TEXT NOT NULL,
                failure_count INTEGER NOT NULL,
                url TEXT NOT NULL,
                text_file TEXT NOT NULL,
                in_degree INTEGER NOT NULL,
                out_degree INTEGER NOT NULL,
                pagerank REAL NOT NULL,
                rank INTEGER NOT NULL
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO nodes (
                graph_id,
                page_id,
                title,
                normalized_title,
                canonical_title,
                normalized_canonical_title,
                kind,
                fetch_status,
                failure_count,
                url,
                text_file,
                in_degree,
                out_degree,
                pagerank,
                rank
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    node.graph_id,
                    node.page_id,
                    node.title,
                    normalize_lookup(node.title),
                    node.canonical_title,
                    normalize_lookup(node.canonical_title) if node.canonical_title else "",
                    node.kind,
                    node.fetch_status,
                    node.failure_count,
                    node.url,
                    node.text_file,
                    in_degree[node.graph_id],
                    out_degree[node.graph_id],
                    scores[node.graph_id],
                    ranks[node.graph_id],
                )
                for node in nodes
            ),
        )

        connection.execute(
            """
            CREATE TABLE edges (
                source_graph_id INTEGER NOT NULL,
                target_graph_id INTEGER NOT NULL,
                PRIMARY KEY (source_graph_id, target_graph_id)
            ) WITHOUT ROWID
            """
        )

        connection.executemany(
            "INSERT INTO edges (source_graph_id, target_graph_id) VALUES (?, ?)",
            iter_edges(adjacency),
        )

        connection.execute("CREATE INDEX idx_nodes_normalized_title ON nodes (normalized_title)")
        connection.execute(
            "CREATE INDEX idx_nodes_normalized_canonical_title ON nodes (normalized_canonical_title)"
        )
        connection.execute("CREATE INDEX idx_nodes_rank ON nodes (rank)")
        connection.execute("CREATE INDEX idx_edges_target_graph_id ON edges (target_graph_id)")

        connection.execute(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ((key, json.dumps(value)) for key, value in metadata.items()),
        )

        connection.commit()
    finally:
        connection.close()


def build_graph(args: argparse.Namespace) -> int:
    if not 0.0 < args.damping < 1.0:
        raise ValueError("--damping must be between 0 and 1.")
    if args.tol <= 0:
        raise ValueError("--tol must be greater than 0.")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be greater than 0.")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    page_index_path = input_dir / "page_index.json"
    links_path = input_dir / "links.csv"

    nodes, title_to_graph_id = load_nodes(
        page_index_path=page_index_path,
        include_categories=args.include_categories,
        completed_only=args.completed_only,
    )
    adjacency, in_degree, out_degree, edge_count = build_adjacency(
        links_path=links_path,
        title_to_graph_id=title_to_graph_id,
        node_count=len(nodes),
    )
    scores, iterations, final_delta = compute_pagerank(
        adjacency=adjacency,
        damping=args.damping,
        tolerance=args.tol,
        max_iterations=args.max_iter,
    )
    ranks = compute_ranks(nodes=nodes, scores=scores)

    fetch_status_counts: dict[str, int] = {}
    for node in nodes:
        fetch_status_counts[node.fetch_status] = fetch_status_counts.get(node.fetch_status, 0) + 1

    metadata = {
        "generated_at_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "page_index_path": str(page_index_path),
        "links_path": str(links_path),
        "node_count": len(nodes),
        "edge_count": edge_count,
        "dangling_node_count": sum(1 for degree in out_degree if degree == 0),
        "include_categories": args.include_categories,
        "completed_only": args.completed_only,
        "damping": args.damping,
        "tolerance": args.tol,
        "max_iterations": args.max_iter,
        "iterations_run": iterations,
        "final_delta": final_delta,
        "fetch_status_counts": fetch_status_counts,
    }

    nodes_csv_path = output_dir / "nodes.csv"
    edges_csv_path = output_dir / "edges.csv"
    metadata_json_path = output_dir / "metadata.json"
    sqlite_path = output_dir / "graph.db"

    write_nodes_csv(
        path=nodes_csv_path,
        nodes=nodes,
        in_degree=in_degree,
        out_degree=out_degree,
        scores=scores,
        ranks=ranks,
    )
    write_edges_csv(path=edges_csv_path, adjacency=adjacency)
    write_metadata_json(path=metadata_json_path, metadata=metadata)
    write_sqlite_database(
        path=sqlite_path,
        nodes=nodes,
        adjacency=adjacency,
        in_degree=in_degree,
        out_degree=out_degree,
        scores=scores,
        ranks=ranks,
        metadata=metadata,
    )

    print("Hyperlink graph build finished.")
    print(f"  Nodes CSV:  {nodes_csv_path}")
    print(f"  Edges CSV:  {edges_csv_path}")
    print(f"  SQLite DB:  {sqlite_path}")
    print(f"  Metadata:   {metadata_json_path}")
    print(f"  Nodes:      {len(nodes)}")
    print(f"  Edges:      {edge_count}")
    print(f"  Iterations: {iterations}")
    print(f"  Final delta:{final_delta:.6e}")
    return 0


def open_database(output_dir: Path) -> sqlite3.Connection:
    database_path = output_dir.resolve() / "graph.db"
    if not database_path.exists():
        raise FileNotFoundError(
            f"Missing graph database: {database_path}. Run the build command first."
        )

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def serialize_row(row: sqlite3.Row) -> dict:
    return {
        "graph_id": row["graph_id"],
        "page_id": row["page_id"],
        "title": row["title"],
        "canonical_title": row["canonical_title"],
        "kind": row["kind"],
        "fetch_status": row["fetch_status"],
        "failure_count": row["failure_count"],
        "url": row["url"],
        "text_file": row["text_file"],
        "in_degree": row["in_degree"],
        "out_degree": row["out_degree"],
        "pagerank": row["pagerank"],
        "rank": row["rank"],
    }


def print_query_result(row: sqlite3.Row) -> None:
    print(f"Title:        {row['title']}")
    if row["canonical_title"]:
        print(f"Canonical:    {row['canonical_title']}")
    print(f"Page ID:      {row['page_id']}")
    print(f"Graph ID:     {row['graph_id']}")
    print(f"PageRank:     {row['pagerank']:.12g}")
    print(f"Rank:         {row['rank']}")
    print(f"In-degree:    {row['in_degree']}")
    print(f"Out-degree:   {row['out_degree']}")
    print(f"Fetch status: {row['fetch_status']}")
    if row["url"]:
        print(f"URL:          {row['url']}")
    if row["text_file"]:
        print(f"Text file:    {row['text_file']}")


def query_graph(args: argparse.Namespace) -> int:
    connection = open_database(args.output_dir)

    try:
        if args.page_id is not None:
            row = connection.execute(
                "SELECT * FROM nodes WHERE page_id = ?",
                (args.page_id,),
            ).fetchone()
        else:
            normalized_title = normalize_lookup(args.title)
            row = connection.execute(
                """
                SELECT *
                FROM nodes
                WHERE normalized_title = ? OR normalized_canonical_title = ?
                ORDER BY CASE
                    WHEN normalized_title = ? THEN 0
                    ELSE 1
                END, rank
                LIMIT 1
                """,
                (normalized_title, normalized_title, normalized_title),
            ).fetchone()

        if row:
            if args.as_json:
                print(json.dumps(serialize_row(row), indent=2))
            else:
                print_query_result(row)
            return 0

        if args.page_id is not None:
            print(f"No node found for page_id={args.page_id}.", file=sys.stderr)
            return 1

        normalized_title = normalize_lookup(args.title)
        suggestions = connection.execute(
            """
            SELECT title, pagerank, rank
            FROM nodes
            WHERE normalized_title LIKE ? OR normalized_canonical_title LIKE ?
            ORDER BY rank
            LIMIT ?
            """,
            (f"%{normalized_title}%", f"%{normalized_title}%", args.suggest_limit),
        ).fetchall()

        print(f"No exact match found for title '{args.title}'.", file=sys.stderr)
        if suggestions:
            print("Closest matches:", file=sys.stderr)
            for suggestion in suggestions:
                print(
                    f"  {suggestion['title']} (rank={suggestion['rank']}, "
                    f"pagerank={suggestion['pagerank']:.12g})",
                    file=sys.stderr,
                )
        return 1
    finally:
        connection.close()


def top_graph(args: argparse.Namespace) -> int:
    connection = open_database(args.output_dir)

    try:
        rows = connection.execute(
            """
            SELECT graph_id, page_id, title, pagerank, rank, fetch_status
            FROM nodes
            ORDER BY rank
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()

        if args.as_json:
            print(json.dumps([dict(row) for row in rows], indent=2))
            return 0

        for row in rows:
            print(
                f"{row['rank']:>4}  {row['pagerank']:.12g}  {row['title']} "
                f"(page_id={row['page_id']}, status={row['fetch_status']})"
            )
        return 0
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and query a directed hyperlink graph for the scraped Wikipedia corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the graph and compute PageRank.")
    build_parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing page_index.json and links.csv (default: {DEFAULT_INPUT_DIR}).",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write graph outputs to (default: {DEFAULT_OUTPUT_DIR}).",
    )
    build_parser.add_argument(
        "--include-categories",
        action="store_true",
        help="Include category nodes in addition to article pages.",
    )
    build_parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Drop nodes whose fetch_status is not completed.",
    )
    build_parser.add_argument(
        "--damping",
        type=float,
        default=0.85,
        help="PageRank damping factor (default: 0.85).",
    )
    build_parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="PageRank convergence tolerance (default: 1e-8).",
    )
    build_parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of PageRank iterations (default: 100).",
    )
    build_parser.set_defaults(handler=build_graph)

    query_parser = subparsers.add_parser("query", help="Look up a node's PageRank value.")
    query_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing graph.db (default: {DEFAULT_OUTPUT_DIR}).",
    )
    query_group = query_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--title", type=str, help="Exact article title to query.")
    query_group.add_argument("--page-id", type=int, help="Wikipedia page_id to query.")
    query_parser.add_argument(
        "--suggest-limit",
        type=int,
        default=10,
        help="Maximum number of fallback title suggestions (default: 10).",
    )
    query_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the query result as JSON.",
    )
    query_parser.set_defaults(handler=query_graph)

    top_parser = subparsers.add_parser("top", help="Show the highest-ranked nodes.")
    top_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing graph.db (default: {DEFAULT_OUTPUT_DIR}).",
    )
    top_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of top-ranked nodes to show (default: 20).",
    )
    top_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the top-ranked rows as JSON.",
    )
    top_parser.set_defaults(handler=top_graph)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
