# How This Repository Models Feature-Ranking Relationships

This file explains only **how the repository maps content features to ranking behavior**. It deliberately avoids interpreting whether any feature is "good" or "bad" for rank.

The relevant code paths are:

- `association_analysis.py`
- `ui/services/rank_relationship_service.py`
- `ui/services/feature_ui_service.py`
- `ui/services/ranking_ui_service.py`

## 1. Shared Data Construction

Before any relationship is modeled, the repository builds a table where each row links a ranked document to its content features.

### 1.1 Full association analysis frame

This is the frame used by the advanced rank-relationship pipeline.

1. Content features are loaded from `data/content_features/content_features.parquet` or `.csv`.
2. Those feature rows are joined to BM25 metadata by `file_name` so that every feature row gets the retriever's canonical `doc_id` and `page_id`.
3. Saved ranking parquet files are loaded, but only for queries that represent a **full ranking** over the corpus.
4. For each `doc_id`, ranking rows are aggregated across queries:
   - `avg_rank = mean(rank)`
   - `avg_score = mean(score)` where `score` is the saved hybrid score
5. Numeric feature columns are inferred automatically. The analysis keeps columns that can be coerced to numbers, while excluding identifiers, labels, JSON blobs, and ranking target fields such as `doc_id`, `title`, `source_path`, `rank`, `score`, `avg_rank`, and `avg_score`.
6. The merged article-level table is then analyzed feature-by-feature.

So the main relationship target is not a single-query rank. It is the **average article rank across all saved full-ranking queries**.

### 1.2 Plot-analysis frame

The feature-plot workflow builds a different frame.

1. The user selects saved queries.
2. Only the top `100` rows of each selected ranking are kept.
3. Those rank rows are joined to article features by `page_id`.
4. The result is a stacked top-100 table across queries, keeping:
   - `rank`
   - `query_id`
   - `page_id`
   - all numeric content features

This frame is used for rolling-rank plots, heatmaps, violin plots, and the median-drop summary.

### 1.3 Per-model frame

The repository also reconstructs rankings for individual scoring components.

1. Each saved ranking file exposes component scores such as:
   - `hybrid_score`
   - `bm25_score_norm`
   - `semantic_score_norm`
   - `pagerank_norm`
2. For `hybrid`, the stored `rank` is reused directly.
3. For `bm25`, `semantic`, and `pagerank`, rank is rebuilt by sorting scores in descending order within each query.
4. Tie-breaking during per-model reranking is deterministic:
   - raw component score descending
   - normalized component score descending
   - original hybrid rank ascending
   - title ascending
5. For each article and model, the code averages:
   - `doc_rank = mean(component rank across queries)`
   - `doc_score = mean(component score across queries)`

This creates a model-specific relationship table where each feature can be related either to model rank or to model score.

## 2. Formal Statistical Relationship Models

These are the explicit statistical mappings implemented in `association_analysis.py`.

### 2.1 Spearman and Kendall rank association

For each numeric feature:

1. Keep only rows where both the feature and `avg_rank` are numeric.
2. Skip the feature if there are fewer than `3` usable rows or the feature has zero variance.
3. Compute:
   - Spearman rho between `feature` and `avg_rank`
   - Kendall tau between `feature` and `avg_rank`
4. Save both coefficients and both p-values.
5. Save a `direction_conflict` flag if Spearman and Kendall disagree in sign.

This models the relationship as a **monotonic rank association** between one feature and average rank.

### 2.2 Normalized mutual information

For each numeric feature:

1. Keep usable numeric rows for `feature` and `avg_rank`.
2. Compute mutual information with `sklearn.feature_selection.mutual_info_regression`.
3. Compute the entropy of `avg_rank` by placing it into `20` histogram bins.
4. Normalize the raw mutual information by that target entropy.
5. Clip the normalized value into `[0, 1]`.

This models the relationship as a **general dependency strength**, not restricted to linear or monotonic structure.

### 2.3 Partial Spearman correlation

This is the repository's conditional relationship model.

For each feature:

1. Build a table containing:
   - the feature of interest
   - all other numeric features as controls
   - `avg_rank`
2. Drop rows with any missing value in that set.
3. Require at least `5` complete rows.
4. Convert every variable to ranks with `rank(method="average")`.
5. Remove any control variable that has zero variance after ranking.
6. Regress ranked feature values on the ranked controls with `LinearRegression`.
7. Regress ranked `avg_rank` on the same ranked controls.
8. Take residuals from both regressions.
9. Correlate the two residual series with Pearson correlation.
10. Report that value as `partial_r`, with a t-based p-value `partial_p` computed with `n - 2` degrees of freedom, exactly as coded.

So the repository's "partial Spearman" is implemented as:

- rank-transform everything first
- residualize both sides against the other features
- correlate the residuals

This is the most explicit attempt in the code to isolate a feature-rank relationship **after removing shared variation with the rest of the feature set**.

### 2.4 Mann-Whitney top-group separation

This models the relationship as **distributional separation between better-ranked and worse-ranked documents**.

For each feature:

1. Build the numeric `feature` / `avg_rank` table.
2. Create two thresholded views:
   - `top10`: documents with `avg_rank <= quantile(0.10)`
   - `top20`: documents with `avg_rank <= quantile(0.20)`
3. For each threshold, split the feature values into:
   - `top_values`
   - `rest_values`
4. Run a two-sided Mann-Whitney U test.
5. Store:
   - U statistic
   - p-value
   - effect size `1 - (2U)/(n_top * n_rest)`

This does not model a smooth curve. It models whether the feature induces a **rank-stratified separation** between top-ranked and non-top-ranked groups.

### 2.5 Breakpoint detection with piecewise linear fitting

This models the relationship as potentially **non-uniform across the feature range**.

For each feature:

1. Sort rows by feature value.
2. Fit a one-line regression:
   - `avg_rank ~ feature`
3. Fit a two-segment piecewise linear model using `pwlf.PiecewiseLinFit`.
4. Compute residual sum of squares for both fits.
5. Convert both fits into AIC values:
   - linear model uses `2` parameters
   - piecewise model uses `4` parameters
6. Mark `has_breakpoint = True` only if `linear_aic - piecewise_aic > 4`.
7. Save:
   - breakpoint location
   - slope of segment 1
   - slope of segment 2
   - both AIC values

This is the repository's threshold-style mapping: it asks whether the feature-rank relationship is better described by **two regimes** instead of one global trend.

### 2.6 Derived-ratio remapping

The code does not only analyze raw features. It also creates transformed features that encode relationship structure differently.

It derives:

- `entity_density = entity_count / word_count`
- `token_word_ratio = token_count / word_count`
- `syllable_density = syllable_count / word_count`
- `sentence_complexity = avg_syllables_per_word * avg_words_per_sentence`

Each derived feature is then related to `avg_rank` with Spearman correlation.

This is still a feature-rank model, but it first **remaps the feature space** into ratios or products before measuring the association.

## 3. Combined Association Summary

All of the statistical models above are merged into one table, `association_summary.csv`.

Each row represents one feature, and the columns collect the different relationship views:

- monotonic association
- dependency strength
- conditional association
- top-group separation
- breakpoint evidence
- transformed-feature correlations

The final table is sorted primarily by normalized mutual information.

## 4. Rolling And Distributional Relationship Mappings

These are implemented in `ui/services/feature_ui_service.py`. They are less formal than the association table, but they are still explicit modeling choices.

### 4.1 Ribbon plots

This is the repository's local smooth-shape model for feature vs. rank.

For one selected feature:

1. Keep rows with ranks between `1` and `100`.
2. Sort by `rank`, `query_id`, and `page_id`.
3. Apply a centered rolling window of `50` rows with `min_periods=10`.
4. Within each rolling window, compute:
   - median
   - 25th percentile
   - 75th percentile
   - 10th percentile
   - 90th percentile
5. Drop incomplete windows.
6. Group by rank and take the median of the rolling summaries.

The plot then shows:

- a rolling median line
- an interquartile ribbon
- a 10th-90th percentile ribbon

So this relationship is modeled as a **smoothed local distribution of feature values along the rank axis**.

### 4.2 Per-query overlay

This uses the same rolling-window logic, but it separates query-level structure from aggregate structure.

1. For each query independently, compute a rolling median by rank.
2. Plot every per-query rolling median as a faint line.
3. Compute the aggregate rolling median across all selected queries.
4. Plot the aggregate line on top.

This models the relationship as:

- many query-specific local feature-rank trajectories
- plus one pooled trajectory

In other words, it shows whether the mapping is query-stable or only visible after aggregation.

### 4.3 Feature consistency heatmap

This is the repository's rank-bucket stability model.

1. Divide the top `100` ranks into four buckets:
   - `1-25`
   - `26-50`
   - `51-75`
   - `76-100`
2. For each feature, compute its global IQR across all top-100 rows.
3. For each bucket, compute the feature's bucket-specific IQR.
4. Normalize each bucket IQR by the feature's global IQR.
5. Clip the normalized value into `[0, 1]`.
6. Plot one heatmap row per feature and one column per rank bucket.
7. Annotate each cell with the raw IQR value.

This does **not** model direction. It models how tightly or loosely feature values are distributed inside different rank zones.

### 4.4 Violin plots by rank bucket

This is the repository's bucketed distribution-shape model.

1. Use the same four rank buckets.
2. For one feature, collect all numeric values inside each bucket.
3. Plot a violin distribution for each bucket.
4. Display the median and extrema.

This maps the feature-rank relationship as a **set of bucket-conditioned value distributions**, not as a single correlation coefficient.

### 4.5 Median-drop summary

This summary is stored in plot metadata even though it is not a standalone figure.

1. Define a top window:
   - `1-25`
2. Define a comparison window:
   - `76-100`
3. For each feature, compute:
   - median in the top window
   - median in the comparison window
   - difference between those medians
4. Sort features by that difference.

This is a coarse two-window contrast model. It summarizes the mapping as a **difference between early-rank and late-rank medians**.

## 5. Model-Component Relationship Modeling

The repository also asks how features relate to different ranking components, not only to the final hybrid rank.

### 5.1 Feature-rank and feature-score heatmaps by model

Implemented through `analyze_model_correlations`.

Models included in the full comparison are:

- `hybrid`
- `bm25`
- `semantic`
- `pagerank`

For each model and each selected heatmap feature:

1. Build the per-model average article table.
2. Compute Spearman correlation between the feature and:
   - `doc_rank`
   - `doc_score`
3. Store both values.

This produces two parallel mappings:

- feature -> model rank
- feature -> model score

The heatmaps visualize those two matrices side by side.

Important scope note: these model heatmaps use only a restricted feature subset:

- `avg_token_length`
- `char_count`
- `lexical_diversity`
- `flesch_kincaid_grade`
- `sentence_count`
- `word_count`

### 5.2 Feature importance heatmaps by model

Implemented through `compute_model_feature_importance`.

This is a multivariate predictive mapping, not a one-feature-at-a-time association.

For each model:

1. Keep rows with complete numeric values for the selected heatmap features and the target score.
2. Require at least `50` rows and nonzero score variance.
3. Standardize the feature matrix.
4. Fit a `RandomForestRegressor` to predict `doc_score`.
5. Store the forest's built-in impurity-based importance for each feature.
6. Run permutation importance and store:
   - mean importance drop
   - importance standard deviation

This models the relationship as **joint predictive contribution within a multivariate model of component score**.

### 5.3 Per-model rerun of the full association pipeline

The repository also exposes per-model rank-relationship analysis for `bm25` and `sbert`.

For the selected model:

1. Build the per-model average article table.
2. Rename:
   - `doc_rank -> avg_rank`
   - `doc_score -> avg_score`
3. Reuse the full association pipeline from Section 2 unchanged.

So the same six formal association models are rerun, but the target is no longer hybrid average rank. It becomes:

- average BM25 rank and score, or
- average SBERT rank and score

### 5.4 BM25 vs SBERT comparison heatmap

Inside the per-model workflow, the repository also builds a smaller comparison heatmap using only:

- `bm25`
- `semantic` renamed to `sbert`

It computes the same `corr_with_rank` and `corr_with_score` matrices for the restricted feature subset, then plots them as a focused two-model comparison.

### 5.5 Per-model rank-feature plot families

The repository also reruns the plot-based relationship mappings on per-model top-100 rankings.

Supported modes are:

- `bm25`
- `sbert`

For the selected mode:

1. Each saved query ranking is reranked with `build_per_model_ranking`.
2. The top `100` rows by `model_rank` are kept.
3. Those rows are joined to article features by `page_id`.
4. The same plot families from Section 4 are generated again:
   - ribbon plots
   - per-query overlays
   - feature consistency heatmap
   - violin plots
5. The same top-window vs late-window median summary is also computed.

So the structural mapping logic from Section 4 exists in two variants:

- hybrid saved-rank space
- per-model reranked space for BM25 or SBERT

## 6. What The Repository Does Not Model

To define the scope clearly, the code does **not** implement the following as explicit relationship models:

- no causal model of how features create rank changes
- no learned end-to-end ranking model trained directly from the content features in this analysis layer
- no explicit feature-feature interaction terms except the derived ratios and `sentence_complexity`
- no pairwise document-comparison model over features
- no temporal or query-sequence model

So the repository's relationship mapping is mainly:

- article-level association against average rank
- bucketed and smoothed structural summaries over top ranks
- model-specific association against component rank and component score
- multivariate predictive importance for ranking components

## 7. Short Presentation Version

If you need a concise verbal explanation, the cleanest summary is:

1. The project first turns many saved rankings into article-level targets such as average rank, average model rank, or average model score.
2. It then models feature-ranking relationships in four families:
   - single-feature association metrics
   - conditional and threshold-based tests
   - smoothed or bucketed distribution mappings over rank
   - model-specific multivariate importance mappings
3. The code never relies on only one view of the relationship. It deliberately represents the mapping as monotonic, nonlinear, conditional, thresholded, local, bucketed, and model-specific, depending on the analysis view.
