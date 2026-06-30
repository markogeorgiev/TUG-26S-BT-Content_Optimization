"""
Rank-Biased Overlap (RBO) between retrieval models, with one figure per pair.

Every model ranks the *same* document collection, so two full rankings are
conjoint permutations and their overlap at full depth is trivially 1. RBO is
therefore computed in its top-weighted form (persistence p < 1): it answers
"how much do these two models agree near the top of the list?", which is the
question that matters for retrieval.

For each pair of models we compute the extrapolated RBO (RBO_ext, Webber et al.
2010) *as a function of evaluation depth*, per query, then average over queries.
The output is one plot per pair (a 1-to-1 comparison) so you can pick which to
show, plus an overview heatmap and a summary table.

Outputs (under rank_relationship_analysis/plots/)
  rbo_pairs/<a>_vs_<b>.png   RBO-vs-depth curve, mean over queries with +/-1 sd band
  rbo_heatmap.png            6x6 matrix of mean RBO (at full depth) - index/overview
  rbo_pairs_summary.csv      mean/std/min/max RBO per pair (at full depth)

Inputs  : rankings/full_retriever/rankings_<model>.csv
          columns: query_id, doc_id, doc_rank, ...

Usage
-----
    python rbo_plots.py
    python rbo_plots.py --p 0.9 --depth 1000 --rankings-dir rankings/full_retriever
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ranking filename token -> nice display label, in the order we want them shown.
MODELS: Dict[str, str] = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "sbert": "SBERT",
    "e5": "E5",
    "colbert": "ColBERT",
    "cross_encoder": "Cross-Encoder",
}


def rbo_ext_curve(list1: List[str], list2: List[str], p: float) -> np.ndarray:
    """Extrapolated RBO at every depth d = 1..k for two conjoint ranked lists.

    Webber et al. (2010), Eq. 32 specialised to equal-length lists::

        RBO_ext(d) = (X_d / d) * p**d  +  ((1 - p) / p) * sum_{j<=d} (X_j / j) * p**j

    where X_d is the overlap between the two depth-d prefixes. Returns an array
    of length k; identical lists -> 1.0 everywhere, disjoint lists -> 0.0.
    """
    k = min(len(list1), len(list2))
    curve = np.empty(k, dtype=float)
    if k == 0:
        return curve

    seen1: set = set()
    seen2: set = set()
    overlap = 0
    weighted_sum = 0.0  # sum_{j<=d} (X_j / j) * p**j
    pd_ = 1.0           # p**d, updated each step

    for d in range(1, k + 1):
        a = list1[d - 1]
        b = list2[d - 1]
        if a == b:
            overlap += 1
        else:
            if a in seen2:
                overlap += 1
            if b in seen1:
                overlap += 1
        seen1.add(a)
        seen2.add(b)

        pd_ *= p
        weighted_sum += (overlap / d) * pd_
        curve[d - 1] = (overlap / d) * pd_ + ((1.0 - p) / p) * weighted_sum

    return curve


def load_top_lists(rankings_dir: Path, depth: int) -> Dict[str, Dict[str, List[str]]]:
    """model -> {query_id -> [doc_id, ...] ordered by rank, truncated to `depth`}."""
    lists: Dict[str, Dict[str, List[str]]] = {}
    for token in MODELS:
        path = rankings_dir / f"rankings_{token}.csv"
        if not path.exists():
            print(f"[{token}] SKIP - {path} not found")
            continue
        print(f"[{token}] loading {path}")
        df = pd.read_csv(path, usecols=["query_id", "doc_id", "doc_rank"])
        df = df[df["doc_rank"] <= depth].sort_values(["query_id", "doc_rank"])
        lists[token] = {
            qid: g["doc_id"].tolist()
            for qid, g in df.groupby("query_id", sort=True)
        }
        print(f"[{token}] {len(lists[token])} queries, top-{depth} each")
    return lists


def pair_curves(
    la: Dict[str, List[str]],
    lb: Dict[str, List[str]],
    p: float,
) -> np.ndarray:
    """Stack per-query RBO-vs-depth curves -> array (n_queries, min_depth)."""
    shared = sorted(set(la) & set(lb))
    curves = [rbo_ext_curve(la[qid], lb[qid], p) for qid in shared]
    min_len = min((c.size for c in curves), default=0)
    return np.vstack([c[:min_len] for c in curves]) if curves else np.empty((0, 0))


def plot_pair(
    curves: np.ndarray,
    label_a: str,
    label_b: str,
    p: str,
    out_path: Path,
) -> float:
    """Plot one pair's RBO-vs-depth (mean +/- 1 sd). Returns full-depth mean RBO."""
    depths = np.arange(1, curves.shape[1] + 1)
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
    lo = np.clip(mean - sd, 0.0, 1.0)
    hi = np.clip(mean + sd, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.fill_between(depths, lo, hi, color="#4C72B0", alpha=0.20,
                    label="+/-1 sd across queries")
    ax.plot(depths, mean, color="#4C72B0", linewidth=2.0, label="mean RBO")

    final = float(mean[-1])
    ax.annotate(
        f"RBO = {final:.3f}",
        xy=(depths[-1], final), xytext=(-10, 10),
        textcoords="offset points", ha="right", fontsize=10,
        color="#1f3b6e", fontweight="bold",
    )

    ax.set_xscale("log")
    ax.set_xlim(1, depths[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evaluation depth (top-k documents)", fontsize=11)
    ax.set_ylabel("RBO", fontsize=11)
    ax.set_title(f"{label_a}  vs  {label_b}   (RBO, p={p})", fontsize=13, pad=10)
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return final


def plot_heatmap(mean_matrix: pd.DataFrame, p: str, depth: int, out_path: Path) -> None:
    tokens = list(mean_matrix.index)
    labels = [MODELS[t] for t in tokens]
    mat = mean_matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.tick_params(length=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            colour = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=colour, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Mean RBO (top-{depth})", fontsize=11)
    ax.set_title(
        f"Rank-Biased Overlap between models\n(p={p}, top-{depth}, mean over queries)",
        fontsize=13, pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rankings-dir", type=Path,
        default=PROJECT_ROOT / "rankings" / "full_retriever",
        help="directory holding rankings_<model>.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "plots",
        help="output directory (per-pair plots go in <out-dir>/rbo_pairs)",
    )
    parser.add_argument(
        "--p", type=float, default=0.9,
        help="RBO persistence (top-weighting); effective depth ~ 1/(1-p). Default 0.9",
    )
    parser.add_argument(
        "--depth", type=int, default=1000,
        help="evaluate RBO over this many top docs per query. Default 1000",
    )
    args = parser.parse_args()

    pairs_dir = args.out_dir / "rbo_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    lists = load_top_lists(args.rankings_dir, args.depth)
    if len(lists) < 2:
        print("[warn] need at least two model rankings; nothing to do.")
        return

    tokens = [t for t in MODELS if t in lists]
    p_label = f"{args.p:g}"
    print(f"[rbo] computing pairwise RBO curves (p={p_label}, depth={args.depth})")

    mat = pd.DataFrame(np.eye(len(tokens)), index=tokens, columns=tokens)
    summary_rows: List[dict] = []

    for a, b in combinations(tokens, 2):
        curves = pair_curves(lists[a], lists[b], args.p)
        if curves.size == 0:
            print(f"[{a} vs {b}] SKIP - no shared queries")
            continue

        out_path = pairs_dir / f"{a}_vs_{b}.png"
        final = plot_pair(curves, MODELS[a], MODELS[b], p_label, out_path)
        print(f"[{a} vs {b}] RBO={final:.3f} ({curves.shape[0]} queries) -> {out_path}")

        mat.loc[a, b] = final
        mat.loc[b, a] = final
        finals = curves[:, -1]
        summary_rows.append({
            "a": a, "b": b, "pair": f"{MODELS[a]} vs {MODELS[b]}",
            "mean": finals.mean(), "std": finals.std(ddof=1) if finals.size > 1 else 0.0,
            "min": finals.min(), "max": finals.max(), "n_queries": finals.size,
        })

    plot_heatmap(mat, p_label, args.depth, args.out_dir / "rbo_heatmap.png")

    summary = pd.DataFrame(summary_rows).sort_values("mean", ascending=False)
    csv_path = args.out_dir / "rbo_pairs_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[summary] wrote {csv_path}")
    print(f"[done] {len(summary_rows)} pair plots in {pairs_dir}")


if __name__ == "__main__":
    main()
