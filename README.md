# Wikipedia Pets Content Optimization

This repository builds a retrieval corpus from Wikipedia's `Category:Pets` graph and now includes the full local retrieval stack:

- Wikipedia crawling and export
- text cleanup for retrieval
- hyperlink graph construction with PageRank
- BM25 document indexing
- MiniLM-based semantic retrieval
- hybrid document ranking
- a Flask/Jinja web UI for querying the corpus, running temporary content-change experiments, analyzing article-level content features, and studying rank-feature relationships

The current system behaves like a document search engine, not a chunk search engine. Chunk embeddings are used only to help score full documents.

## Current Local Snapshot

The local crawl and graph artifacts currently on disk were generated on May 24, 2026.

- `output/wikipedia-pets/manifest.json` reports `21,077` discovered nodes.
- `21,058` pages finished successfully and `19` pages failed.
- The crawl export includes `1,174,075` in-scope hyperlinks.
- `data/graph/metadata.json` contains an article-only graph with `19,567` nodes and `1,139,761` edges.
- `data/corpus/`, `data/corpus_cleaned/`, `output/wikipedia-pets/texts/`, and `output/wikipedia-pets/texts-cleaned/` currently each contain `21,058` text files.
- `data/bm25_index/` exists locally and contains the current BM25 artifacts.
- `data/embeddings/` exists locally and currently includes combined MiniLM outputs plus part files.
- `rankings/queries.json` currently contains `14` saved queries and `rankings/initial_rankings/` contains their stored full ranking parquets.

These numbers describe the current local workspace, not a fixed expectation for future reruns.

## Repository Layout

- `scripts/Export-WikipediaCategory.ps1`
  Windows-first Wikipedia crawler with resumable progress, polite delays, retries, and export generation.
- `scripts/Clean-WikipediaTextExports.ps1`
  Cleans raw text exports under `output/wikipedia-pets/texts/`.
- `scripts/build_embedding_corpus.py`
  Builds the cleaner retrieval corpus in `data/corpus_cleaned/`.
- `retriever/build_hyperlink_graph.py`
  Builds the directed hyperlink graph and computes PageRank.
- `retriever/build_bm25_document_retriever.ipynb`
  Creates the BM25 document index under `data/bm25_index/`.
- `retriever/build_wikipedia_pet_embeddings_minilm.ipynb`
  Current CPU-only MiniLM embedding pipeline.
- `retriever/run_hybrid_rankings.py`
  CLI and reusable backend for hybrid document ranking.
- `association_analysis.py`
  CLI and reusable functions for advanced rank relationship analysis.
- `ui/app.py`
  Flask entry point for the web search UI.
- `ui/templates/`, `ui/static/`, `ui/services/`
  Jinja templates, CSS/JS assets, and UI service layer.
- `rankings/`
  Local query log and saved full ranking results.
- `retriever/bge_model_tests.ipynb`
  Older BGE exploration notebook kept for reference.
- `scripts/bge_model_tests.py`
  Placeholder script file.

## Environment Setup

This repo is currently Windows-oriented because the crawler and one cleanup stage are PowerShell-based.

Create or activate the virtual environment and install dependencies:

```powershell
.venv\Scripts\Activate.ps1
pip install -r .\requrements.txt
```

Notes:

- The dependency file is intentionally named `requrements.txt` in the repo.
- The current retrieval stack expects `Flask`, `pandas`, `numpy`, `pyarrow`, `rank-bm25`, `sentence-transformers`, `spacy`, `matplotlib`, `scipy`, `scikit-learn`, and `pwlf`, all of which are now listed there.
- The current dense retrieval path is CPU-oriented. Do not assume CUDA support.

## End-To-End Workflow

### 1. Crawl Wikipedia

Run the category crawler:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Export-WikipediaCategory.ps1
```

Useful options:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Export-WikipediaCategory.ps1 `
  -BaseCategory 'Category:Pets' `
  -OutputDir '.\output\wikipedia-pets' `
  -RequestDelayMs 3000 `
  -RequestDelayJitterMs 5000 `
  -MaxRetries 3 `
  -IncludeAllNamespaces
```

Resume behavior:

- rerun with the same `-OutputDir` to continue from saved progress
- add `-RetryFailedPages` to retry pages marked as failed

Primary crawler outputs:

- `output/wikipedia-pets/texts/*.txt`
- `output/wikipedia-pets/page_index.json`
- `output/wikipedia-pets/links.csv`
- `output/wikipedia-pets/manifest.json`
- `output/wikipedia-pets/failed_pages.csv`
- `output/wikipedia-pets/progress/`

### 2. Optional Raw Export Cleanup

If you want cleaned copies of the raw crawler exports, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Clean-WikipediaTextExports.ps1
```

By default this reads `output/wikipedia-pets/texts/` and writes `output/wikipedia-pets/texts-cleaned/`.

This stage removes obvious export noise and trims trailing sections such as references and external links.

### 3. Build The Retrieval Corpus

The Python cleaner prepares the corpus used by the current retrieval pipelines:

```powershell
python .\scripts\build_embedding_corpus.py
```

By default it reads `data/corpus/` and writes `data/corpus_cleaned/`.

This cleaner is more aggressive than the PowerShell export cleaner. It removes back matter, drops table-like blocks, compacts list-heavy sections, and keeps the output focused on retrieval-friendly prose.

### 4. Build The Hyperlink Graph

Build the article graph and PageRank:

```powershell
python .\retriever\build_hyperlink_graph.py build
```

By default this writes:

- `data/graph/nodes.csv`
- `data/graph/edges.csv`
- `data/graph/graph.db`
- `data/graph/metadata.json`

Useful variants:

```powershell
python .\retriever\build_hyperlink_graph.py build --completed-only
python .\retriever\build_hyperlink_graph.py build --include-categories
```

Useful graph queries:

```powershell
python .\retriever\build_hyperlink_graph.py query --title "Abyssinian cat"
python .\retriever\build_hyperlink_graph.py query --page-id 7590101
python .\retriever\build_hyperlink_graph.py top --limit 20
```

### 5. Build The BM25 Document Index

The lexical retriever is built through:

- [retriever/build_bm25_document_retriever.ipynb](retriever/build_bm25_document_retriever.ipynb)

This notebook creates a document-level BM25 index under `data/bm25_index/`, including:

- `document_metadata.parquet`
- `tokenized_corpus.pkl`
- `bm25_index.pkl`
- `bm25_config.json`

This index is separate from the embedding artifacts and is used directly by the hybrid ranker.

### 6. Build CPU MiniLM Embeddings

The dense retrieval workflow lives in:

- [retriever/build_wikipedia_pet_embeddings_minilm.ipynb](retriever/build_wikipedia_pet_embeddings_minilm.ipynb)

The notebook is designed for CPU execution and uses:

```python
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
```

Key behavior:

- uses `spacy.blank("en")` with a sentencizer
- chunks text by sentence
- writes metadata and embeddings in parts to reduce memory pressure
- supports resume-friendly processing
- can optionally produce combined outputs

Expected outputs include:

- `data/embeddings/chunk_metadata_parts/chunk_metadata_part_0000.parquet`
- `data/embeddings/embedding_parts/chunk_embeddings_part_0000.npy`
- `data/embeddings/chunk_metadata.parquet`
- `data/embeddings/chunk_embeddings.npy`

### 7. Run Hybrid Document Ranking

The current retrieval backend is:

- [retriever/run_hybrid_rankings.py](retriever/run_hybrid_rankings.py)

Run it from the CLI like this:

```powershell
python .\retriever\run_hybrid_rankings.py --query "are birds good pets?"
```

Multiple queries can be passed in one call:

```powershell
python .\retriever\run_hybrid_rankings.py `
  --query "are birds good pets?" `
  --query "best orange coat dog"
```

Important behavior:

- rankings are computed at the document level
- BM25 contributes `0.4`
- MiniLM semantic retrieval contributes `0.4`
- PageRank contributes `0.2`
- each metric is min-max normalized before the weighted sum is computed
- duplicate queries are not re-executed; existing saved results are reused
- the full ranking is always saved, even if terminal preview is shorter

Saved outputs:

- query log: `rankings/queries.json`
- result files: `rankings/initial_rankings/q000001_query-slug.parquet`

CLI note:

- `--top-k` controls terminal preview length only
- the saved parquet still contains the full ranking

### 8. Launch The Web UI

The web UI lives in the separate root-level `ui/` folder to keep it modular:

```powershell
python .\ui\app.py
```

The app runs on:

- `http://127.0.0.1:5000`

Current UI behavior:

- header navigation between the Query, Rank Relationship, Content Change Experiments, and Content Features areas
- search bar above the results area
- autocomplete suggestions sourced from `rankings/queries.json`
- new queries are executed once and then reused
- configurable display limit, including custom `k`
- client-side sorting by results-table column
- Wikipedia article links open in a new tab
- saved parquet download for the currently selected query
- temporary content-change experiments for a selected query and article
- temporary add/remove link experiments by Wikipedia page ID
- article-level feature search by Wikipedia page ID or article name
- article feature display sourced from `data/content_features/`
- rank relationship analysis against average rank across all saved full-ranking queries
- rank-feature plots for selected saved-query collections
- cached plot reuse under `rankings/content_feature_rank_plots/`

The Rank Relationship page runs the advanced association pipeline from:

- [association_analysis.py](association_analysis.py)

It averages rank and hybrid score across all saved full-ranking queries per article, then uses `avg_rank` as the target variable. The pipeline runs:

- Spearman and Kendall correlation
- normalized mutual information
- partial Spearman correlation controlling for other numeric features
- Mann-Whitney top-10% and top-20% separation tests
- two-segment piecewise breakpoint detection with `pwlf`
- derived ratio feature Spearman correlations

Saved Rank Relationship outputs are cached under `rankings/rank_relationship/` and include:

- `association_summary.csv`
- `dot_plot.png`
- `diverging_bar.png`
- `model_correlations.csv`
- `model_importance.csv`
- `model_correlation_heatmaps.png`
- `model_importance_heatmaps.png`

The model-comparison heatmaps use the ranking components available in each saved ranking file:

- `hybrid`
- `bm25`
- `semantic`
- `pagerank`

For each component, the UI computes average per-article component rank and score across saved queries, then plots feature-rank correlations, feature-score correlations, random-forest feature importance, and permutation feature importance.

The standalone CLI accepts feature and ranking CSV paths:

```powershell
python .\association_analysis.py .\features.csv .\rankings.csv
```

By default the CLI writes `results/association_summary.csv`, `results/dot_plot.png`, and `results/diverging_bar.png`.

The Content Change Experiments page lets you select a saved full-ranking query, choose any article by page ID or title, edit its text in a browser editor, and submit a temporary reranking experiment. The experiment does not modify corpus files, graph files, saved ranking parquet files, or `rankings/queries.json`.

Experiment scoring behavior:

- edited text recomputes the selected article's BM25 score using the stored BM25 IDF/length statistics
- edited text recomputes the selected article's MiniLM score by sentence-packing the temporary text into `512` word chunks
- link additions/removals are accepted as page IDs for outgoing links from the selected article or incoming links toward it
- link edits use a local PageRank edge-contribution approximation for instant feedback instead of recomputing the full graph PageRank
- the UI displays the baseline rank, experimental rank, rank movement statement, component score deltas, local link effects, and an experimental ranking preview

The Content Features area opens on the saved-query feature plot workflow and currently has two subsections:

- `Rank Feature Plots` lets you select saved queries and a plot kind, then joins their top `100` ranked articles to content features.
- `Show Article Level Features` displays one article's metrics from `data/content_features/`.

Available rank-feature plot kinds:

- `Ribbon plots` create one `plot_{feature_name}.png` per numeric feature with a rolling median, IQR band, and 10th-90th percentile band.
- `Per-query overlay` creates one `overlay_{feature_name}.png` per numeric feature with faint query-level rolling medians and a bold aggregate rolling median.
- `Feature consistency heatmap` creates one `heatmap.png` covering all numeric features across rank buckets `1-25`, `26-50`, `51-75`, and `76-100`.
- `Violin plots` create one `violin_{feature_name}.png` per numeric feature showing the feature value distribution across those same four rank buckets.

Rank feature plots use a centered rolling window of `50` rows with `min_periods=10`, a steelblue matplotlib style, and a summary table comparing median feature values in ranks `1-25` against ranks `76-100`. Selecting the same query collection and plot kind again reuses the existing cached plot folder instead of regenerating images.

## Retrieval Design

The current system is document retrieval, not chunk retrieval.

Semantic scoring works like this:

1. score each chunk against the query embedding
2. group chunk scores by document
3. aggregate them into a document score

Current document aggregation weights:

- top chunk: `0.40`
- second chunk: `0.35`
- third chunk: `0.20`
- mean of remaining chunks: `0.05`

If a document has fewer than four chunks, the score is normalized over the weights that actually exist so short documents are not unfairly penalized.

## Generated Local Artifacts

Large generated artifacts are intentionally kept out of git. Important local output areas are:

- `output/wikipedia-pets/`
- `data/corpus_cleaned/`
- `data/graph/`
- `data/bm25_index/`
- `data/embeddings/`
- `rankings/`

## Current And Legacy Retrieval Code

Current paths:

- `retriever/build_bm25_document_retriever.ipynb`
- `retriever/build_wikipedia_pet_embeddings_minilm.ipynb`
- `retriever/run_hybrid_rankings.py`
- `ui/app.py`

Legacy or reference-only paths:

- `retriever/bge_model_tests.ipynb`
- older `FlagEmbedding`/BGE assumptions in the environment and experiments

The current recommended path is BM25 + MiniLM + PageRank through the hybrid ranker and the Flask UI.

## Practical Notes

- `output/wikipedia-pets*`, `data/`, `.venv/`, `__pycache__/`, and `rankings/` are ignored by git.
- The repo contains data-pipeline code, retrieval code, and a local exploration UI rather than a packaged production application.
- Some generated metadata files still contain absolute paths from the machine where they were produced; the code itself resolves paths from the current repo layout.
