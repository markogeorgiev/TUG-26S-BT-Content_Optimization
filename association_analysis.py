from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


EXCLUDED_COLUMNS = {
    "doc_id",
    "title",
    "source_path",
    "entities_json",
    "top_entities_json",
    "query_id",
    "rank",
    "score",
    "avg_rank",
    "avg_score",
}

SUMMARY_COLUMNS = [
    "feature",
    "spearman_r",
    "spearman_p",
    "kendall_t",
    "kendall_p",
    "direction_conflict",
    "mutual_info_norm",
    "partial_r",
    "partial_p",
    "mw_effect_top10",
    "mw_p_top10",
    "mw_effect_top20",
    "mw_p_top20",
    "has_breakpoint",
    "breakpoint_value",
]

DEFAULT_MODEL_FEATURE_LABELS = {
    "avg_token_length": "Avg Token Length",
    "char_count": "Document Length",
    "lexical_diversity": "Lexical Diversity",
    "flesch_kincaid_grade": "Readability Grade",
    "sentence_count": "Sentence Count",
    "word_count": "Word Count",
}


def configure_logging() -> None:
    """Configure warning logging for command-line runs."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _is_zero_variance(values: pd.Series) -> bool:
    return pd.to_numeric(values, errors="coerce").dropna().nunique() <= 1


def _warn_skip(feature: str, reason: str) -> None:
    logging.warning("Skipping feature %s: %s", feature, reason)


def _safe_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].apply(pd.to_numeric, errors="coerce")


def _aic(n: int, rss: float, parameter_count: int) -> float:
    rss = max(float(rss), np.finfo(float).tiny)
    return float(n * np.log(rss / n) + 2 * parameter_count)


def _target_entropy(avg_rank: pd.Series) -> float:
    clean_rank = pd.to_numeric(avg_rank, errors="coerce").dropna()
    if clean_rank.nunique() <= 1:
        return 0.0
    counts, _ = np.histogram(clean_rank.to_numpy(dtype=float), bins=20)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    return float(entropy(counts))


def load_and_prepare_data(features_csv: str | Path, rankings_csv: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load feature and ranking CSVs, average rank/score per article, and return the analysis frame."""
    features = pd.read_csv(features_csv)
    rankings = pd.read_csv(rankings_csv)

    required_feature_columns = {"doc_id", "title", "source_path"}
    required_ranking_columns = {"doc_id", "query_id", "rank", "score"}
    missing_feature_columns = required_feature_columns.difference(features.columns)
    missing_ranking_columns = required_ranking_columns.difference(rankings.columns)
    if missing_feature_columns:
        raise ValueError(
            "features.csv missing required columns: "
            + ", ".join(sorted(missing_feature_columns))
        )
    if missing_ranking_columns:
        raise ValueError(
            "rankings.csv missing required columns: "
            + ", ".join(sorted(missing_ranking_columns))
        )

    rankings = rankings.copy()
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["score"] = pd.to_numeric(rankings["score"], errors="coerce")
    rank_summary = (
        rankings.dropna(subset=["doc_id", "rank", "score"])
        .groupby("doc_id", as_index=False)
        .agg(avg_rank=("rank", "mean"), avg_score=("score", "mean"))
    )

    merged = features.merge(rank_summary, on="doc_id", how="inner")
    feature_columns = infer_feature_columns(merged)
    return merged, feature_columns


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Infer numeric feature columns while excluding IDs, labels, target fields, and JSON blobs."""
    candidates = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    numeric_columns: list[str] = []
    for column in candidates:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def spearman_kendall_analysis(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compute Spearman and Kendall correlations against avg_rank for each feature."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for feature in feature_columns:
        working = df[[feature, "avg_rank"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(working) < 3:
            skipped += 1
            _warn_skip(feature, "not enough non-missing observations")
            continue
        if _is_zero_variance(working[feature]):
            skipped += 1
            _warn_skip(feature, "zero variance")
            continue

        spearman_r, spearman_p = stats.spearmanr(working[feature], working["avg_rank"])
        kendall_t, kendall_p = stats.kendalltau(working[feature], working["avg_rank"])
        rows.append(
            {
                "feature": feature,
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "kendall_t": float(kendall_t),
                "kendall_p": float(kendall_p),
                "direction_conflict": bool(np.sign(spearman_r) != np.sign(kendall_t)),
            }
        )

    print(f"Spearman + Kendall complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def mutual_information_analysis(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compute normalized mutual information between each feature and avg_rank."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for feature in feature_columns:
        working = df[[feature, "avg_rank"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(working) < 3:
            skipped += 1
            _warn_skip(feature, "not enough non-missing observations")
            continue
        if _is_zero_variance(working[feature]):
            skipped += 1
            _warn_skip(feature, "zero variance")
            continue

        target_entropy = _target_entropy(working["avg_rank"])
        if target_entropy <= 0:
            mi_norm = 0.0
        else:
            mi_raw = float(
                mutual_info_regression(
                    working[[feature]].to_numpy(dtype=float),
                    working["avg_rank"].to_numpy(dtype=float),
                    random_state=42,
                )[0]
            )
            mi_norm = float(max(0.0, min(mi_raw / target_entropy, 1.0)))
        rows.append({"feature": feature, "mutual_info_norm": mi_norm})

    print(f"Mutual information complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def partial_correlation_analysis(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compute partial Spearman correlation with avg_rank while controlling for other features."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    numeric = _safe_numeric_frame(df, [*feature_columns, "avg_rank"])

    for feature in feature_columns:
        control_columns = [column for column in feature_columns if column != feature]
        relevant_columns = [feature, *control_columns, "avg_rank"]
        working = numeric[relevant_columns].dropna()
        if len(working) < 5:
            skipped += 1
            _warn_skip(feature, "not enough complete observations for partial correlation")
            continue
        if _is_zero_variance(working[feature]):
            skipped += 1
            _warn_skip(feature, "zero variance")
            continue

        ranked = working.rank(method="average")
        usable_controls = [
            column for column in control_columns if not _is_zero_variance(ranked[column])
        ]
        feature_values = ranked[feature].to_numpy(dtype=float)
        target_values = ranked["avg_rank"].to_numpy(dtype=float)

        if usable_controls:
            controls = ranked[usable_controls].to_numpy(dtype=float)
            feature_model = LinearRegression().fit(controls, feature_values)
            target_model = LinearRegression().fit(controls, target_values)
            feature_residuals = feature_values - feature_model.predict(controls)
            target_residuals = target_values - target_model.predict(controls)
        else:
            feature_residuals = feature_values
            target_residuals = target_values

        if np.std(feature_residuals) == 0 or np.std(target_residuals) == 0:
            skipped += 1
            _warn_skip(feature, "zero residual variance")
            continue

        partial_r = float(np.corrcoef(feature_residuals, target_residuals)[0, 1])
        n = len(feature_residuals)
        if n <= 2 or abs(partial_r) >= 1:
            partial_p = 0.0 if abs(partial_r) >= 1 else np.nan
        else:
            t_value = partial_r * np.sqrt((n - 2) / max(1 - partial_r**2, np.finfo(float).eps))
            partial_p = float(2 * stats.t.sf(abs(t_value), df=n - 2))
        rows.append({"feature": feature, "partial_r": partial_r, "partial_p": partial_p})

    print(f"Partial correlation complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def mann_whitney_analysis(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Run Mann-Whitney top-k separation tests for top 10% and top 20% rank thresholds."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    thresholds = {"top10": 0.10, "top20": 0.20}

    for feature in feature_columns:
        working = df[[feature, "avg_rank"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(working) < 3:
            skipped += 1
            _warn_skip(feature, "not enough non-missing observations")
            continue
        if _is_zero_variance(working[feature]):
            skipped += 1
            _warn_skip(feature, "zero variance")
            continue

        row: dict[str, Any] = {"feature": feature}
        for label, quantile in thresholds.items():
            cutoff = working["avg_rank"].quantile(quantile)
            top_values = working.loc[working["avg_rank"] <= cutoff, feature]
            rest_values = working.loc[working["avg_rank"] > cutoff, feature]
            if top_values.empty or rest_values.empty:
                row[f"mw_u_{label}"] = np.nan
                row[f"mw_p_{label}"] = np.nan
                row[f"mw_effect_{label}"] = np.nan
                continue

            u_statistic, p_value = stats.mannwhitneyu(
                top_values,
                rest_values,
                alternative="two-sided",
            )
            effect = 1 - (2 * float(u_statistic)) / (len(top_values) * len(rest_values))
            row[f"mw_u_{label}"] = float(u_statistic)
            row[f"mw_p_{label}"] = float(p_value)
            row[f"mw_effect_{label}"] = float(effect)
        rows.append(row)

    print(f"Mann-Whitney complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def breakpoint_detection_analysis(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compare linear and two-segment piecewise linear models for each feature."""
    try:
        import pwlf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pwlf is required for breakpoint detection. Install it with: pip install pwlf"
        ) from exc

    rows: list[dict[str, Any]] = []
    skipped = 0
    for feature in feature_columns:
        working = df[[feature, "avg_rank"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(working) < 8:
            skipped += 1
            _warn_skip(feature, "not enough non-missing observations")
            continue
        if _is_zero_variance(working[feature]):
            skipped += 1
            _warn_skip(feature, "zero variance")
            continue

        working = working.sort_values(feature)
        x_values = working[feature].to_numpy(dtype=float)
        y_values = working["avg_rank"].to_numpy(dtype=float)
        n = len(working)

        linear_model = LinearRegression().fit(x_values.reshape(-1, 1), y_values)
        linear_predictions = linear_model.predict(x_values.reshape(-1, 1))
        linear_rss = float(np.sum((y_values - linear_predictions) ** 2))
        linear_aic = _aic(n, linear_rss, parameter_count=2)

        try:
            piecewise_model = pwlf.PiecewiseLinFit(x_values, y_values)
            breaks = piecewise_model.fit(2)
            piecewise_predictions = piecewise_model.predict(x_values)
            piecewise_rss = float(np.sum((y_values - piecewise_predictions) ** 2))
            piecewise_aic = _aic(n, piecewise_rss, parameter_count=4)
            slopes = list(piecewise_model.slopes)
            breakpoint_value = float(breaks[1]) if len(breaks) > 2 else np.nan
            slope_1 = float(slopes[0]) if slopes else np.nan
            slope_2 = float(slopes[1]) if len(slopes) > 1 else np.nan
            has_breakpoint = bool(linear_aic - piecewise_aic > 4)
        except Exception as exc:
            skipped += 1
            _warn_skip(feature, f"piecewise fit failed: {exc}")
            continue

        rows.append(
            {
                "feature": feature,
                "breakpoint_value": breakpoint_value,
                "piecewise_slope_1": slope_1,
                "piecewise_slope_2": slope_2,
                "linear_aic": linear_aic,
                "piecewise_aic": piecewise_aic,
                "has_breakpoint": has_breakpoint,
            }
        )

    print(f"Breakpoint detection complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def derived_ratio_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived ratio features and compute Spearman correlation against avg_rank."""
    working = df.copy()
    word_count = pd.to_numeric(working["word_count"], errors="coerce")
    positive_words = word_count > 0
    ratio_specs = {
        "entity_density": "entity_count",
        "token_word_ratio": "token_count",
        "syllable_density": "syllable_count",
    }

    for derived_feature, numerator in ratio_specs.items():
        numerator_values = pd.to_numeric(working[numerator], errors="coerce")
        working[derived_feature] = np.where(positive_words, numerator_values / word_count, np.nan)

    working["sentence_complexity"] = (
        pd.to_numeric(working["avg_syllables_per_word"], errors="coerce")
        * pd.to_numeric(working["avg_words_per_sentence"], errors="coerce")
    )

    rows: list[dict[str, Any]] = []
    skipped = 0
    for feature in [
        "entity_density",
        "token_word_ratio",
        "syllable_density",
        "sentence_complexity",
    ]:
        feature_frame = working[[feature, "avg_rank"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(feature_frame) < 3:
            skipped += 1
            _warn_skip(feature, "not enough non-missing observations")
            continue
        if _is_zero_variance(feature_frame[feature]):
            skipped += 1
            _warn_skip(feature, "zero variance")
            continue

        spearman_r, spearman_p = stats.spearmanr(feature_frame[feature], feature_frame["avg_rank"])
        rows.append(
            {
                "feature": feature,
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            }
        )

    print(f"Derived ratio features complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def build_summary_table(
    spearman_kendall: pd.DataFrame,
    mutual_info: pd.DataFrame,
    partial: pd.DataFrame,
    mann_whitney: pd.DataFrame,
    breakpoints: pd.DataFrame,
    derived_ratios: pd.DataFrame,
) -> pd.DataFrame:
    """Join all analysis result tables into the requested association summary."""
    frames = [spearman_kendall, mutual_info, partial, mann_whitney, breakpoints]
    summary = frames[0].copy()
    for frame in frames[1:]:
        summary = summary.merge(frame, on="feature", how="outer")

    if not derived_ratios.empty:
        summary = pd.concat([summary, derived_ratios], ignore_index=True, sort=False)

    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan
    summary = summary[SUMMARY_COLUMNS]
    summary["direction_conflict"] = (
        summary["direction_conflict"].astype("boolean").fillna(False).astype(bool)
    )
    summary["has_breakpoint"] = (
        summary["has_breakpoint"].astype("boolean").fillna(False).astype(bool)
    )
    return summary.sort_values(
        ["mutual_info_norm", "feature"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def plot_dot_summary(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Create a four-column horizontal dot plot sorted by normalized mutual information."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = summary.sort_values(
        ["mutual_info_norm", "feature"],
        ascending=[True, False],
        na_position="first",
    )
    metrics = [
        ("spearman_r", "Spearman rho", "spearman_p"),
        ("kendall_t", "Kendall tau", "kendall_p"),
        ("mutual_info_norm", "Normalized MI", None),
        ("mw_effect_top10", "MW effect top 10%", "mw_p_top10"),
    ]
    y_positions = np.arange(len(plot_df))
    fig_height = max(7.0, 0.38 * len(plot_df) + 2.0)
    fig, axes = plt.subplots(1, 4, figsize=(16, fig_height), sharey=True)

    for axis, (value_column, title, p_column) in zip(axes, metrics):
        values = pd.to_numeric(plot_df[value_column], errors="coerce")
        if p_column is None:
            colors = ["#808891"] * len(plot_df)
        else:
            p_values = pd.to_numeric(plot_df[p_column], errors="coerce")
            colors = np.where(p_values < 0.05, "#0e6b5c", "#b8a994")
        axis.scatter(values, y_positions, c=colors, s=34, edgecolors="white", linewidths=0.45)
        if value_column in {"spearman_r", "kendall_t", "mw_effect_top10"}:
            axis.axvline(0, color="#6f604d", linestyle="--", linewidth=1)
            axis.set_xlim(-1.05, 1.05)
        else:
            axis.set_xlim(0, max(float(values.max(skipna=True) or 0.01), 0.01) * 1.12)
        axis.set_title(title)
        axis.grid(True, axis="x", color="#d7e3ea", linewidth=0.8, alpha=0.72)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(plot_df["feature"])
    fig.suptitle("Rank relationship summary", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_diverging_spearman(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Create a diverging Spearman bar chart with normalized MI dots on a secondary axis."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = summary.dropna(subset=["spearman_r"]).sort_values("spearman_r")
    y_positions = np.arange(len(plot_df))
    colors = np.where(plot_df["has_breakpoint"].astype(bool), "#8f5d16", "#4682b4")
    fig_height = max(7.0, 0.38 * len(plot_df) + 2.0)
    fig, axis = plt.subplots(figsize=(11, fig_height))

    axis.barh(y_positions, plot_df["spearman_r"], color=colors, alpha=0.78)
    axis.axvline(0, color="#6f604d", linestyle="--", linewidth=1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(plot_df["feature"])
    axis.set_xlim(-1.05, 1.05)
    axis.set_xlabel("Spearman rho")
    axis.set_title("Spearman correlation with average rank")
    axis.grid(True, axis="x", color="#d7e3ea", linewidth=0.8, alpha=0.72)

    mi_axis = axis.twiny()
    mi_values = pd.to_numeric(plot_df["mutual_info_norm"], errors="coerce").fillna(0)
    mi_axis.scatter(mi_values, y_positions, color="#777777", s=22, alpha=0.72, label="Normalized MI")
    mi_axis.set_xlim(0, max(float(mi_values.max() or 0.01), 0.01) * 1.12)
    mi_axis.set_xlabel("Normalized MI")

    axis.text(
        0.01,
        0.01,
        "Orange bars: piecewise breakpoint detected",
        transform=axis.transAxes,
        color="#6f604d",
        fontsize=9,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def safe_spearman(x_values: pd.Series, y_values: pd.Series) -> float:
    """Return Spearman correlation, or NaN when the input is too small or constant."""
    x_array = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    y_array = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]
    if len(x_array) < 2:
        return np.nan
    if np.nanstd(x_array) == 0 or np.nanstd(y_array) == 0:
        return np.nan
    return float(stats.spearmanr(x_array, y_array, nan_policy="omit").correlation)


def analyze_model_correlations(
    merged_df: pd.DataFrame,
    feature_columns: list[str],
    model_column: str = "model",
    rank_column: str = "doc_rank",
    score_column: str = "doc_score",
) -> pd.DataFrame:
    """Compute feature-rank and feature-score Spearman correlations for each model."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for model in sorted(merged_df[model_column].dropna().unique()):
        model_df = merged_df.loc[merged_df[model_column] == model]
        for feature in feature_columns:
            working = model_df[[feature, rank_column, score_column]].apply(
                pd.to_numeric,
                errors="coerce",
            ).dropna()
            if len(working) < 2 or _is_zero_variance(working[feature]):
                skipped += 1
                _warn_skip(f"{model}:{feature}", "not enough variance for model correlation")
                continue
            rows.append(
                {
                    "model": str(model),
                    "feature": feature,
                    "corr_with_rank": safe_spearman(working[feature], working[rank_column]),
                    "corr_with_score": safe_spearman(working[feature], working[score_column]),
                    "n": int(len(working)),
                }
            )

    print(f"Model correlation heatmaps complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def compute_model_feature_importance(
    merged_df: pd.DataFrame,
    feature_columns: list[str],
    model_column: str = "model",
    target_column: str = "doc_score",
    n_estimators: int = 200,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute random forest and permutation feature importance for each model."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for model in sorted(merged_df[model_column].dropna().unique()):
        model_df = merged_df.loc[merged_df[model_column] == model]
        working = model_df[[*feature_columns, target_column]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        usable_features = [
            feature for feature in feature_columns if not _is_zero_variance(working[feature])
        ]
        if len(working) < 50 or not usable_features or _is_zero_variance(working[target_column]):
            skipped += len(feature_columns)
            _warn_skip(str(model), "not enough observations or target variance for importance")
            continue

        x_values = working[usable_features].astype(float)
        y_values = working[target_column].astype(float)
        scaled_x = StandardScaler().fit_transform(x_values)
        forest = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        forest.fit(scaled_x, y_values)
        permutation = permutation_importance(
            forest,
            scaled_x,
            y_values,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        importance_lookup = {
            feature: {
                "rf_importance": float(forest.feature_importances_[index]),
                "perm_importance_mean": float(permutation.importances_mean[index]),
                "perm_importance_std": float(permutation.importances_std[index]),
            }
            for index, feature in enumerate(usable_features)
        }
        for feature in feature_columns:
            values = importance_lookup.get(feature)
            if values is None:
                skipped += 1
                continue
            rows.append(
                {
                    "model": str(model),
                    "feature": feature,
                    "rf_importance": values["rf_importance"],
                    "perm_importance_mean": values["perm_importance_mean"],
                    "perm_importance_std": values["perm_importance_std"],
                    "n": int(len(working)),
                    "target": target_column,
                }
            )

    print(f"Model feature importance heatmaps complete: processed {len(rows)}, skipped {skipped}.")
    return pd.DataFrame(rows)


def _display_feature_index(
    pivot: pd.DataFrame,
    feature_labels: dict[str, str] | None,
) -> pd.DataFrame:
    labels = feature_labels or DEFAULT_MODEL_FEATURE_LABELS
    renamed = pivot.copy()
    renamed.index = [labels.get(str(feature), str(feature)) for feature in renamed.index]
    return renamed


def _annotated_heatmap(
    axis: Any,
    data: pd.DataFrame,
    title: str,
    cmap: str,
    colorbar_label: str,
    center_zero: bool = False,
    value_format: str = ".3f",
) -> Any:
    import matplotlib

    matrix = data.to_numpy(dtype=float)
    if center_zero:
        from matplotlib.colors import TwoSlopeNorm

        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        max_abs = max(max_abs, 0.001)
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    else:
        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
        image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=max(vmax, 0.001), aspect="auto")

    axis.set_title(title, fontsize=13, fontweight="bold", pad=15)
    axis.set_xlabel("Model", fontsize=11, fontweight="bold")
    axis.set_ylabel("Feature", fontsize=11, fontweight="bold")
    axis.set_xticks(np.arange(data.shape[1]))
    axis.set_xticklabels(data.columns)
    axis.set_yticks(np.arange(data.shape[0]))
    axis.set_yticklabels(data.index)
    axis.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    axis.tick_params(which="minor", bottom=False, left=False)

    finite_values = matrix[np.isfinite(matrix)]
    threshold = float(np.nanmean(finite_values)) if len(finite_values) else 0.0
    if center_zero:
        threshold = float(np.nanpercentile(np.abs(finite_values), 65)) if len(finite_values) else 0.0

    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                label = "NA"
                text_color = "#2d2419"
            else:
                label = format(value, value_format)
                text_color = "white" if abs(value) >= abs(threshold) else "#2d2419"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold",
            )

    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    return image


def plot_model_correlation_heatmaps(
    correlation_df: pd.DataFrame,
    output_path: str | Path,
    feature_labels: dict[str, str] | None = None,
) -> None:
    """Plot feature-rank and feature-score Spearman heatmaps by model."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pivot_rank = correlation_df.pivot(index="feature", columns="model", values="corr_with_rank")
    pivot_score = correlation_df.pivot(index="feature", columns="model", values="corr_with_score")
    pivot_rank = _display_feature_index(pivot_rank, feature_labels)
    pivot_score = _display_feature_index(pivot_score, feature_labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    _annotated_heatmap(
        axes[0],
        pivot_rank,
        "Feature-Rank Correlations by Model\n(negative = feature associated with better rank)",
        cmap="RdBu_r",
        colorbar_label="Spearman rho",
        center_zero=True,
    )
    _annotated_heatmap(
        axes[1],
        pivot_score,
        "Feature-Score Correlations by Model\n(positive = feature associated with higher score)",
        cmap="RdBu_r",
        colorbar_label="Spearman rho",
        center_zero=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def plot_model_importance_heatmaps(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    feature_labels: dict[str, str] | None = None,
) -> None:
    """Plot random forest and permutation feature importance heatmaps by model."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pivot_rf = importance_df.pivot(index="feature", columns="model", values="rf_importance")
    pivot_perm = importance_df.pivot(index="feature", columns="model", values="perm_importance_mean")
    pivot_rf = _display_feature_index(pivot_rf, feature_labels)
    pivot_perm = _display_feature_index(pivot_perm, feature_labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    _annotated_heatmap(
        axes[0],
        pivot_rf,
        "Random Forest Feature Importance by Model\n(higher values = more predictive of document score)",
        cmap="YlOrRd",
        colorbar_label="Importance",
        center_zero=False,
    )
    _annotated_heatmap(
        axes[1],
        pivot_perm,
        "Permutation Feature Importance by Model\n(decrease in model performance when feature is shuffled)",
        cmap="YlOrRd",
        colorbar_label="Importance",
        center_zero=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)


def run_association_analysis(
    merged_df: pd.DataFrame,
    feature_columns: list[str],
    results_dir: str | Path = "results",
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run all six analyses, save the summary table and plots, and return their paths."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    spearman_kendall = spearman_kendall_analysis(merged_df, feature_columns)
    mutual_info = mutual_information_analysis(merged_df, feature_columns)
    partial = partial_correlation_analysis(merged_df, feature_columns)
    mann_whitney = mann_whitney_analysis(merged_df, feature_columns)
    breakpoints = breakpoint_detection_analysis(merged_df, feature_columns)
    derived_ratios = derived_ratio_analysis(merged_df)
    summary = build_summary_table(
        spearman_kendall=spearman_kendall,
        mutual_info=mutual_info,
        partial=partial,
        mann_whitney=mann_whitney,
        breakpoints=breakpoints,
        derived_ratios=derived_ratios,
    )

    summary_path = results_path / "association_summary.csv"
    dot_plot_path = results_path / "dot_plot.png"
    diverging_bar_path = results_path / "diverging_bar.png"
    summary.to_csv(summary_path, index=False)
    plot_dot_summary(summary, dot_plot_path)
    plot_diverging_spearman(summary, diverging_bar_path)

    return summary, {
        "summary": summary_path,
        "dot_plot": dot_plot_path,
        "diverging_bar": diverging_bar_path,
    }


def run_from_csv(
    features_csv: str | Path,
    rankings_csv: str | Path,
    results_dir: str | Path = "results",
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run the full association pipeline from feature and ranking CSV paths."""
    merged_df, feature_columns = load_and_prepare_data(features_csv, rankings_csv)
    return run_association_analysis(merged_df, feature_columns, results_dir=results_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for association analysis."""
    parser = argparse.ArgumentParser(
        description="Run advanced rank relationship analysis from feature and ranking CSV files."
    )
    parser.add_argument("features_csv", help="Path to features.csv.")
    parser.add_argument("rankings_csv", help="Path to rankings.csv.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for association_summary.csv and plots. Defaults to results/.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    summary, output_paths = run_from_csv(
        args.features_csv,
        args.rankings_csv,
        results_dir=args.results_dir,
    )
    print("\nSaved outputs:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
    print("\nTop 5 features by normalized mutual information:")
    top_columns = ["feature", "mutual_info_norm", "spearman_r", "mw_effect_top10"]
    print(summary[top_columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
