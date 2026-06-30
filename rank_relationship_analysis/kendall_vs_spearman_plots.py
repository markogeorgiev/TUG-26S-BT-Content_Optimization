"""
Agreement between Kendall's tau-b and Spearman's rho, one figure per model.

Both per_query_kendall_tau_b.py and per_query_spearmans.py emit the *same* wide
table shape (one row per query_id, one column per content feature), differing
only in the rank-correlation coefficient stored in each cell. If the two
coefficients tell the same story, then plotting tau-b against rho for every
(query, feature) cell should give points hugging the identity line y = x.

For each model this:
  * pairs each cell's tau-b with its rho (dropping cells NaN in either table),
  * scatters tau-b (x) vs rho (y) with the y = x reference line,
  * annotates Pearson r and Spearman rho between the two coefficient sets, the
    mean signed/absolute difference, and the sign-agreement rate.

Because tau-b and rho use different scales (Kendall is systematically smaller in
magnitude), points do not sit exactly on y = x; the expected large-sample
relationship is the Greiner / Kruskal approximation rho ~= (3/2) * tau. That
reference line is drawn too, so you can confirm the cloud follows the known
analytic relationship rather than expecting a 45-degree line.

Inputs
------
  * kendall  : rank_relationship_analysis/per_query_kendall/kendall_<model>.csv
  * spearman : rank_relationship_analysis/per_query_spearman/spearman_<model>.csv

Outputs (under rank_relationship_analysis/plots/)
  kendall_vs_spearman/<model>.png   per-model scatter
  kendall_vs_spearman_all.png       2x3 grid overview, all models
  kendall_vs_spearman_summary.csv   agreement stats per model

Usage
-----
    python kendall_vs_spearman_plots.py
    python kendall_vs_spearman_plots.py \
        --kendall-dir rank_relationship_analysis/per_query_kendall \
        --spearman-dir rank_relationship_analysis/per_query_spearman
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


def paired_cells(kendall_path: Path, spearman_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten both wide tables to aligned 1-D arrays of (tau-b, rho) cell values.

    Tables are aligned on their common (query_id, feature) cells; any cell that
    is NaN in either table is dropped.
    """
    tau = pd.read_csv(kendall_path, index_col="query_id")
    rho = pd.read_csv(spearman_path, index_col="query_id")

    # Align on the intersection of rows and columns so the flattened cells match.
    common_cols = [c for c in tau.columns if c in rho.columns]
    common_idx = tau.index.intersection(rho.index)
    tau = tau.loc[common_idx, common_cols]
    rho = rho.loc[common_idx, common_cols]

    x = tau.to_numpy(dtype=float).ravel()
    y = rho.to_numpy(dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def agreement_stats(tau: np.ndarray, rho: np.ndarray) -> Dict[str, float]:
    """Summary statistics describing how closely tau-b and rho agree."""
    pearson = float(np.corrcoef(tau, rho)[0, 1]) if tau.size > 1 else float("nan")
    # Spearman between the two coefficient sets = Pearson of their ranks.
    if tau.size > 1:
        rt = pd.Series(tau).rank().to_numpy()
        rr = pd.Series(rho).rank().to_numpy()
        spearman = float(np.corrcoef(rt, rr)[0, 1])
    else:
        spearman = float("nan")
    diff = rho - tau
    # Sign agreement ignores cells where either coefficient is ~0 (no direction).
    nz = (np.abs(tau) > 1e-9) & (np.abs(rho) > 1e-9)
    sign_agree = float(np.mean(np.sign(tau[nz]) == np.sign(rho[nz]))) if nz.any() else float("nan")
    return {
        "n_cells": int(tau.size),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "mean_signed_diff": float(np.mean(diff)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "sign_agreement": sign_agree,
    }


def _draw_reference_lines(ax: plt.Axes, lim: float) -> None:
    """Identity line (y = x) and the Greiner rho ~= 1.5*tau approximation."""
    line = np.array([-lim, lim])
    ax.plot(line, line, color="#555555", linewidth=1.0, linestyle="--",
            label="y = x")
    approx = np.clip(1.5 * line, -lim, lim)
    ax.plot(line, approx, color="#C44E52", linewidth=1.0, linestyle=":",
            label=r"$\rho \approx \frac{3}{2}\tau$")
    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=0)


def plot_model(
    tau: np.ndarray,
    rho: np.ndarray,
    label: str,
    stats: Dict[str, float],
    out_path: Path,
) -> None:
    """Single-model tau-b vs rho scatter with reference lines and stats box."""
    lim = float(max(1.0, np.nanmax(np.abs(np.concatenate([tau, rho])))))

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(tau, rho, s=10, alpha=0.35, color="#4C72B0",
               edgecolors="none", label="(query, feature) cells")
    _draw_reference_lines(ax, lim)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Kendall's tau-b", fontsize=11)
    ax.set_ylabel("Spearman's rho", fontsize=11)
    ax.set_title(f"{label}: Kendall vs Spearman", fontsize=13, pad=10)
    ax.grid(linestyle=":", alpha=0.5)

    txt = (f"n = {stats['n_cells']}\n"
           f"Pearson r = {stats['pearson_r']:.3f}\n"
           f"Spearman = {stats['spearman_rho']:.3f}\n"
           f"sign agree = {stats['sign_agreement']:.1%}\n"
           f"mean |diff| = {stats['mean_abs_diff']:.3f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#999999",
                                  alpha=0.85))
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grid(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, float]]],
    out_path: Path,
) -> None:
    """2x3 overview grid: one tau-b vs rho panel per model."""
    tokens = [t for t in MODELS if t in data]
    ncols = 3
    nrows = int(np.ceil(len(tokens) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, token in zip(axes, tokens):
        tau, rho, stats = data[token]
        lim = float(max(1.0, np.nanmax(np.abs(np.concatenate([tau, rho])))))
        ax.scatter(tau, rho, s=6, alpha=0.30, color="#4C72B0", edgecolors="none")
        _draw_reference_lines(ax, lim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{MODELS[token]}  (r={stats['pearson_r']:.3f})", fontsize=11)
        ax.grid(linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=8)

    for ax in axes[len(tokens):]:
        ax.set_visible(False)

    fig.supxlabel("Kendall's tau-b", fontsize=12)
    fig.supylabel("Spearman's rho", fontsize=12)
    fig.suptitle("Kendall tau-b vs Spearman rho, per (query, feature) cell",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kendall-dir", type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "per_query_kendall",
        help="directory holding kendall_<model>.csv",
    )
    parser.add_argument(
        "--spearman-dir", type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "per_query_spearman",
        help="directory holding spearman_<model>.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "rank_relationship_analysis" / "plots",
        help="output directory (per-model plots go in <out-dir>/kendall_vs_spearman)",
    )
    args = parser.parse_args()

    per_model_dir = args.out_dir / "kendall_vs_spearman"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = {}
    summary_rows: List[dict] = []

    for token, label in MODELS.items():
        kendall_path = args.kendall_dir / f"kendall_{token}.csv"
        spearman_path = args.spearman_dir / f"spearman_{token}.csv"
        if not kendall_path.exists() or not spearman_path.exists():
            print(f"[{token}] SKIP - missing kendall or spearman file")
            continue

        tau, rho = paired_cells(kendall_path, spearman_path)
        if tau.size < 2:
            print(f"[{token}] SKIP - too few paired cells ({tau.size})")
            continue

        stats = agreement_stats(tau, rho)
        data[token] = (tau, rho, stats)

        out_path = per_model_dir / f"{token}.png"
        plot_model(tau, rho, label, stats, out_path)
        print(f"[{token}] r={stats['pearson_r']:.3f}, "
              f"sign-agree={stats['sign_agreement']:.1%}, "
              f"n={stats['n_cells']} -> {out_path}")

        summary_rows.append({"model": label, "token": token, **stats})

    if not data:
        print("[warn] no model had both kendall and spearman files; nothing to do.")
        return

    plot_grid(data, args.out_dir / "kendall_vs_spearman_all.png")

    summary = pd.DataFrame(summary_rows)
    csv_path = args.out_dir / "kendall_vs_spearman_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[summary] wrote {csv_path}")
    print(f"[done] {len(data)} per-model plots in {per_model_dir}")


if __name__ == "__main__":
    main()
