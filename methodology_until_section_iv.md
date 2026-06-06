# Methodology Up To `plan.md` Section IV

## Scope

This document explains how the project was built from the beginning of the workflow up to the work covered by Sections I, II, III, and IV of [plan.md](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\plan.md). It intentionally stops before Section V, `Web UI v2`, and does not discuss later optimization-oriented workflows as part of the main methodology here.

The explanation is based on the implemented code and generated local artifacts in this repository, especially:

- `scripts/Export-WikipediaCategory.ps1`
- `scripts/Clean-WikipediaTextExports.ps1`
- `scripts/build_embedding_corpus.py`
- `retriever/build_hyperlink_graph.py`
- `retriever/build_bm25_document_retriever.ipynb`
- `retriever/build_wikipedia_pet_embeddings_minilm.ipynb`
- `retriever/run_hybrid_rankings.py`
- `content_features_analysis/build_content_features.ipynb`
- `association_analysis.py`
- `ui/app.py`
- `ui/services/*`
- `ui/templates/*`

In other words, this is not just a restatement of the plan. It is a code-grounded description of what was actually built.

## Project Formation In One Continuous Narrative

The project began by defining a bounded but still broad content universe: Wikipedia material reachable from `Category:Pets`. That choice created a domain-specific corpus large enough to support ranking experiments, but focused enough that retrieval, graph structure, and article-level content features could still be interpreted within one theme. From there, the work progressed in four major stages.

First, the repository built the corpus itself. A crawler discovered pages and categories underneath the pets category, exported text, collected in-scope links, and preserved enough progress state to resume long-running collection jobs. This gave the project a repeatable local snapshot of the Wikipedia pets space instead of relying on live lookups.

Second, the project turned that raw snapshot into retrieval infrastructure. Three different ranking signals were built: a sparse lexical retriever, a dense neural retriever, and a hyperlink-based PageRank signal. Those signals were normalized and combined into a single hybrid scorer so that each query could produce a full reproducible ranking over the local corpus.

Third, the project moved from retrieval into measurement. A separate content-feature pipeline computed article-level characteristics such as length, tokenization behavior, readability, lexical properties, sentiment, and entity statistics. This step mattered because the goal of the project was not only to rank pages, but to create a measurable bridge between ranking behavior and concrete properties of the ranked content.

Fourth, the project wrapped the pipeline in a local analytical web application. The UI did not begin as an optimization tool. It began as a reproducible workspace for running queries, storing full rankings, inspecting article features, and comparing ranking behavior against the measured feature set. That analytical layer is the final scope of this document.

## I. Building the Retrievers

### I.0 Building the corpus from Wikipedia's pets category

The initial task in the plan was to build a corpus of about 21,000 pages from Wikipedia's pets category. The implemented version of that step lives in [scripts/Export-WikipediaCategory.ps1](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\scripts\Export-WikipediaCategory.ps1).

The crawler does several things at once:

- It resolves the root category, defaulting to `Category:Pets`.
- It traverses descendant categories and pages through the Wikipedia API.
- It keeps discovery state on disk so category traversal can resume after interruption.
- It fetches each page's HTML through the public Wikipedia site rather than only using API extracts.
- It extracts a main content fragment, converts that HTML to plain text, and stores one text file per discovered node.
- It extracts in-scope internal links from the same HTML so the hyperlink graph can later be built locally.

The crawler is deliberately conservative and resumable:

- It applies polite delays with jitter between requests.
- It supports retry logic for transient failures.
- It treats HTTP `404` and `410` as non-retryable.
- It writes per-page progress files under `output/wikipedia-pets/progress/page-state/`.
- It can bootstrap already completed pages from existing text files instead of fetching them again.

The main outputs of the crawl are:

- `output/wikipedia-pets/texts/*.txt`
- `output/wikipedia-pets/page_index.json`
- `output/wikipedia-pets/links.csv`
- `output/wikipedia-pets/manifest.json`
- `output/wikipedia-pets/failed_pages.csv`

In the current local snapshot, `output/wikipedia-pets/manifest.json` shows:

- `21,077` discovered nodes in total
- `1,510` category nodes
- `19,567` page nodes
- `21,058` completed text exports
- `19` failed fetches
- `1,174,075` in-scope hyperlinks

That means the corpus was not built from a manually curated page list. It emerged from a category-graph crawl, and the later retrieval system inherits that structure.

### Raw text export cleanup

The repository includes a first cleanup stage in [scripts/Clean-WikipediaTextExports.ps1](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\scripts\Clean-WikipediaTextExports.ps1). This PowerShell script is not yet the final retrieval cleaner, but it removes obvious export noise from the crawler output.

This stage:

- separates crawler header metadata from article body text
- removes standalone `edit` markers
- truncates trailing sections such as `References`, `Notes`, `Footnotes`, `Citations`, `Sources`, `Bibliography`, `Further reading`, and `External links`
- normalizes excess blank lines

The result is a cleaned copy of the raw export under `output/wikipedia-pets/texts-cleaned/`.

### Retrieval-corpus cleanup

The more retrieval-oriented cleanup step is [scripts/build_embedding_corpus.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\scripts\build_embedding_corpus.py). This is the stronger text-normalization pass used before BM25 and MiniLM retrieval work.

Its purpose is to turn crawler exports into prose-focused retrieval documents. It performs operations that are especially important for both lexical indexing and sentence-based chunking:

- removes back matter starting at headings such as `See also`, `References`, `External links`, and similar sections
- detects table-like blocks and removes them when they appear to be metadata-heavy rather than prose
- compacts runs of bullet items into short declarative sentences such as `The list also includes ...`
- preserves ordinary prose and headings

This design directly reflects the concerns written into Section I of the plan. Long list-heavy or table-like Wikipedia sections are not good retrieval text for either BM25 or MiniLM, so the repository cleans them before indexing.

### I.1 and I.2 Defining the hybrid retriever and the weights

The central ranking formula from the plan was implemented in [retriever/run_hybrid_rankings.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\retriever\run_hybrid_rankings.py):

`hybrid = 0.4 * BM25 + 0.4 * semantic + 0.2 * PageRank`

The exact constants in code are:

- `HYBRID_BM25_WEIGHT = 0.4`
- `HYBRID_SEMANTIC_WEIGHT = 0.4`
- `HYBRID_PAGERANK_WEIGHT = 0.2`

The repository also resolves the normalization concern mentioned in the plan. Before the weighted sum is computed, each score family is min-max normalized per query:

- `bm25_score_raw -> bm25_score_norm`
- `semantic_score_raw -> semantic_score_norm`
- `pagerank -> pagerank_norm`

This is important because the three signals exist on very different numeric scales:

- BM25 scores depend on token overlap and inverse document frequency.
- semantic similarity is a cosine-style dense similarity score after normalized embeddings.
- PageRank is a small graph-centrality value.

Without normalization, the weighted sum would mostly reflect scale artifacts. The implemented design therefore treats the weights as genuine influence weights rather than accidental numeric multipliers.

### I.3 Building the PageRank graph

PageRank is implemented separately in [retriever/build_hyperlink_graph.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\retriever\build_hyperlink_graph.py).

The graph builder reads:

- `output/wikipedia-pets/page_index.json`
- `output/wikipedia-pets/links.csv`

It then:

- builds a node table from the discovered titles
- maps titles to internal graph IDs
- converts link rows into adjacency lists
- computes in-degree and out-degree
- runs iterative PageRank with damping, tolerance, and max-iteration settings
- writes CSV and SQLite outputs for later lookup and UI enrichment

The default graph build excludes categories and keeps article pages only, which is why the current graph metadata shows `19,567` nodes rather than the full `21,077` discovered nodes. In the current local graph snapshot:

- `node_count = 19,567`
- `edge_count = 1,139,761`
- `damping = 0.85`
- `iterations_run = 76`
- `final_delta = 9.80e-09`

An important detail is that failed pages are still present as nodes if they were discovered in the crawl inventory. That matches the spirit of the plan: pages that failed to produce usable text can still participate structurally in the graph if they were part of the discovered hyperlink universe. As a result, PageRank is not limited to only the documents that later become usable for semantic chunking.

### I.4 Choosing and building the neural retriever

The dense retriever was finalized around the MiniLM family and is documented in [retriever/build_wikipedia_pet_embeddings_minilm.ipynb](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\retriever\build_wikipedia_pet_embeddings_minilm.ipynb). The chosen model is:

- `sentence-transformers/all-MiniLM-L6-v2`

This model is used on CPU and creates `384`-dimensional embeddings.

#### Why chunking was necessary

The project does not embed each entire Wikipedia document as one vector. Instead, it treats the semantic retriever as a document retriever built from chunk-level evidence.

Chunking is performed with:

- `spacy.blank("en")`
- a lightweight `sentencizer`
- sentence grouping with a target chunk size of `180` words

This choice reflects the plan directly. The code avoids splitting inside ordinary sentences, and it treats sentence boundaries as the primary unit of chunk assembly. If a single sentence exceeds the `180`-word limit, the notebook records that page as skipped rather than embedding an obviously malformed or list-heavy sentence as if it were a normal paragraph.

#### What the semantic build produces

The MiniLM notebook builds semantic retrieval artifacts in two stages:

1. chunk metadata parts
2. matching `.npy` embedding parts

This part-based design keeps the workflow practical on a Windows CPU machine and avoids requiring all embeddings to be kept in memory at once.

In the current local artifacts:

- `164,651` chunk rows were generated
- chunk metadata was written across `43` part files
- combined metadata and combined embedding files were also produced
- `20,868` distinct documents ended up with chunk embeddings

The semantic pipeline therefore does not cover every completed text file. A `skipped_pages.parquet` report exists in `data/embeddings/chunk_metadata_parts/`, and in the current snapshot it contains `209` skipped rows. Most skips are caused by overlong sentence-like blocks that exceed the `180`-word limit after cleaning, and `19` rows reflect missing text files. This matters because BM25 and content features cover more documents than the dense retriever does.

#### How semantic retrieval works at query time

The runtime semantic scorer in [retriever/run_hybrid_rankings.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\retriever\run_hybrid_rankings.py) follows a document-level aggregation strategy:

1. embed the query once
2. score the query against every stored chunk embedding
3. group chunk scores back to the document level
4. aggregate chunk evidence into one document score

The aggregation weights are:

- best chunk: `0.40`
- second-best chunk: `0.35`
- third-best chunk: `0.20`
- mean of remaining chunks: `0.05`

If a document has fewer than four chunks, the score is normalized over the weights that are actually present so short documents are not penalized for simply having fewer chunks.

This is a very important design choice. The semantic retriever is not a chunk-retrieval UI. It is a document retriever that uses chunk evidence internally.

### Sparse retriever construction

The lexical retriever is built in [retriever/build_bm25_document_retriever.ipynb](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\retriever\build_bm25_document_retriever.ipynb).

Its BM25 settings are stored in `data/bm25_index/bm25_config.json`:

- model: `BM25Okapi`
- `k1 = 1.5`
- `b = 0.75`
- lowercase normalization enabled
- stopword removal enabled
- numbers retained
- minimum token length `2`

The BM25 tokenizer logic was later centralized in `retriever/run_hybrid_rankings.py` and reused in multiple other parts of the project. It:

- lowercases text
- collapses whitespace
- uses a regex that keeps alphabetic tokens and numeric tokens
- removes a manually defined stopword set
- removes one-character tokens

The BM25 build stores:

- `document_metadata.parquet`
- `tokenized_corpus.pkl`
- `bm25_index.pkl`
- `bm25_config.json`

Unlike the semantic retriever, BM25 currently covers all `21,058` completed text files, because it works over the cleaned document text directly and does not require chunk generation to succeed.

### Hybrid retrieval and full-ranking storage

The actual ranking execution layer is [retriever/run_hybrid_rankings.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\retriever\run_hybrid_rankings.py).

Its runtime sequence is:

1. load BM25 document metadata
2. tokenize the query for BM25
3. compute BM25 raw scores
4. compute semantic document scores from chunk embeddings
5. load PageRank values from `data/graph/nodes.csv`
6. merge the three signals by `page_id`
7. fill missing semantic or PageRank values with `0.0`
8. normalize each signal per query
9. compute weighted contributions and hybrid score
10. sort the full corpus into one saved ranking

The saved ranking table contains:

- rank
- hybrid score
- document and page identifiers
- file name
- size metadata
- raw and normalized BM25, semantic, and PageRank values
- weighted component contributions
- PageRank rank
- matched query tokens

This is the point where the project became reproducible. Every query is not just displayed once; it is stored as a full ranking parquet under `rankings/initial_rankings/` and logged in `rankings/queries.json`.

The query log stores:

- a stable `query_id`
- raw query text
- normalized query key
- slug
- execution timestamp
- weights
- normalization method
- semantic aggregation metadata
- path to the stored parquet

The runtime also deduplicates queries. If the same query is run again, the system reuses the saved result instead of re-ranking from scratch.

### I.5 Compiling the query set

The plan called for an initial set of queries to create analyzable rankings. That step is implemented through the ranking log itself rather than through a hard-coded list in source code.

At the time of this document, `rankings/queries.json` contains `15` saved full-ranking queries. They span several pet-related intents, including:

- pet suitability questions
- breed or species identification
- feeding and care queries
- beginner-pet queries
- appearance-based pet queries
- exotic-pet queries

Examples from the saved set include:

- `Are birds good pets?`
- `Best dogs for small apartments`
- `What should you feed a chameleon`
- `Can you keep a fish as a pet?`
- `Dog breed that doesn't shed`
- `Healthy diet for dogs`
- `best exotic pet`

This query collection matters because the later analysis is not built on a single ranking. It is built on repeated full-corpus rankings across a saved query set.

## II. Content Characteristics

### II.0 From candidate ideas to implemented features

The plan began with a broad question: what characteristics of content should be tracked? It considered structural features, readability, lexical diversity, and even possible topic-modeling or semantic features.

The implemented project narrowed this into a feature set that could be:

- computed for every cleaned document
- stored in a stable table
- joined back to retrieval results
- analyzed statistically across many queries

That feature extraction workflow is implemented in [content_features_analysis/build_content_features.ipynb](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\content_features_analysis\build_content_features.ipynb), and its generated outputs live under `data/content_features/`.

The current feature build processed `21,058` documents on `2026-06-04`, with:

- `0` read errors
- VADER sentiment available
- spaCy NER model `en_core_web_sm` available

### II.1 The implemented feature groups

The project ultimately standardized the feature space into four UI-facing groups plus stored entity payloads.

#### 1. Document Size

These features measure how much text exists:

- `char_count`
- `non_whitespace_char_count`
- `word_count`
- `token_count`
- `sentence_count`
- `entity_count`

`token_count` is especially important because it is not a generic tokenizer count. It explicitly reuses the BM25 tokenizer from `retriever.run_hybrid_rankings.tokenize_for_bm25`, so the same tokenization logic drives both retrieval and later feature analysis.

#### 2. Lexical and Surface Metrics

These describe the shape of the language:

- `avg_words_per_sentence`
- `avg_token_length`
- `lexical_density`
- `lexical_diversity`

The feature notebook defines:

- `lexical_density = non_stopword_alphabetic_words / word_count`
- `lexical_diversity = unique_lowercase_words / word_count`

This means lexical density is meant to reflect how content-heavy the language is, while lexical diversity measures vocabulary spread.

#### 3. Readability and Complexity

These reflect how hard the text is to read or how complex its language is:

- `syllable_count`
- `avg_syllables_per_word`
- `flesch_kincaid_grade`
- `gunning_fog_index`

The generated config records the formulas explicitly:

- `flesch_kincaid_grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59`
- `gunning_fog_index = 0.4 * ((words / sentences) + 100 * (complex_words / words))`

These formulas were not left conceptual in the plan. They are fully materialized in the feature table.

#### 4. Sentiment

The current schema includes:

- `sentiment_polarity`
- `sentiment_subjectivity`

The implemented sentiment backend uses VADER when available. Its compound score becomes `sentiment_polarity`. VADER does not provide subjectivity, so `sentiment_subjectivity` currently remains null in practice. This is why the feature schema contains the field, but the hybrid-wide analysis metadata currently reports `15` usable numeric features rather than `16`.

#### 5. Entity statistics and stored entity payloads

The feature system also performs named-entity extraction and stores:

- `entity_count`
- `entities_json`
- `top_entities_json`

The notebook uses spaCy NER and stores:

- all detected entity records in `entities_json`
- a filtered top-entity list in `top_entities_json`

The current config records:

- `top_entity_limit = 20`
- `entity_min_frequency = 2`

So the UI can show both an overall entity count and a compact repeated-entity summary for each article.

### What was considered but not implemented as a standardized feature

It is also important to describe what the current system did not standardize, because that clarifies the actual endpoint of Section II.

The plan mentioned:

- keyword density relative to query
- BERTopic or topic modeling
- broader semantic feature families

Those were explored conceptually, but they are not part of the current stored content-feature table. The implemented feature layer chose a stable document-wide metric set first, rather than introducing query-dependent features or topic-modeling outputs that would be harder to standardize across the whole corpus.

## III. Web UI v1 and Queries

### Overall architecture

The web application entry point is [ui/app.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\app.py). It uses Flask with Jinja templates and a small service layer under `ui/services/`.

The important architectural choice is that the routes stay thin. Most logic is delegated to service modules:

- `ranking_ui_service.py` for saved query execution and ranking views
- `feature_ui_service.py` for article features and rank-feature plots
- `rank_relationship_service.py` for association-analysis caching and output resolution

This means the UI is not only a front-end shell. It is a structured orchestration layer over the stored local ranking and feature artifacts.

### III.1 The first section of the UI: Query

The first major UI section is the `Query` page, implemented by:

- [ui/templates/search.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\search.html)
- [ui/templates/partials/results_table.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\partials\results_table.html)
- [ui/static/js/search.js](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\static\js\search.js)
- [ui/services/ranking_ui_service.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\services\ranking_ui_service.py)

This first section is not just a search box. It is the UI embodiment of Section III.1 from the plan.

#### What the Query page does

The Query page lets the user:

- enter an arbitrary query
- receive autocomplete suggestions from `rankings/queries.json`
- choose how many visible rows to display
- run a new query or reuse a saved one
- load the stored full-ranking parquet for an existing query
- download the current query's saved parquet file
- sort the visible result table client-side by column

#### How the page is structured

The page opens with a `Search Settings` panel that contains:

- a query input field
- a `Results to display` selector
- an optional custom `k` box
- a submit button

The page then displays, when a query is loaded:

- a `Selected Query` summary card
- a `Storage` summary card telling whether the saved result is a full ranking
- an `Execution Mode` summary card telling whether the result was freshly created or reused
- a results preview table

The results table shows:

- hybrid rank
- title
- hybrid score
- normalized BM25 score
- normalized semantic score
- normalized PageRank score
- page ID
- matched query tokens

The title links out to the corresponding Wikipedia article when a URL is available from the graph metadata.

#### Why this page matters methodologically

This page is where the project stopped being only a backend experiment. It turned the retrieval stack into a reproducible ranking workspace:

- the ranking is full-corpus rather than ad hoc top-k only
- the full result is stored to disk
- the visible preview is separate from the stored complete ranking
- repeated queries become shared analytical inputs for later stages

### III.2 Displaying per-content statistics

The plan required article-level content statistics in the UI. That work is implemented by:

- [ui/templates/features_article.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\features_article.html)
- [ui/static/js/features.js](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\static\js\features.js)
- `get_article_features` and related helpers in `ui/services/feature_ui_service.py`

The article-level features page lets the user:

- search by page ID or article title
- use autocomplete suggestions from the article catalog
- open a detail page for one article
- inspect grouped metrics
- inspect top repeated entities
- optionally expand to see all stored detected entities

The UI groups the metrics exactly as the project standardized them:

- Document Size
- Lexical and Surface
- Readability and Complexity
- Sentiment

This is a direct realization of Section III.2 of the plan: per-content statistics are no longer conceptual, they are inspectable for any analyzed document in the feature table.

### III.3 Executing and preserving a query collection

The plan also required that a collection of queries be executed and kept for later modeling. The implemented system does this through persistent query logging and saved ranking parquets rather than through one fixed seed script.

Each query becomes:

- one entry in `rankings/queries.json`
- one parquet in `rankings/initial_rankings/`

This design gives the later analytical pages a stable corpus of ranking outputs to work from. The UI, the analysis service layer, and the statistics code all read from the same saved query pool.

## IV. Analysis of Ranking

Section IV of the plan asked how ranking should be modeled relative to content features, and how to visualize those relationships across many queries without confusing coincidence for stable association. The implemented analysis layer is the part of the repository that answers that question.

### The core analytical target: average rank across saved queries

The hybrid-wide analysis pipeline is built in:

- [association_analysis.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\association_analysis.py)
- [ui/services/rank_relationship_service.py](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\services\rank_relationship_service.py)
- [ui/templates/rank_relationship.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\rank_relationship.html)

Its core methodology is:

1. load all saved full-ranking query parquets
2. compute per-article average hybrid rank and average hybrid score across those queries
3. join those ranking summaries to the content-feature table
4. analyze numeric features against `avg_rank`

This design is very important. The system does not treat one query as the whole truth. It aggregates across a query set so that the feature analysis is based on repeated ranking behavior.

### IV.3 Visualization and analysis techniques that were implemented

The repository did not stop at one correlation coefficient. It implemented multiple complementary analytical views.

#### 1. Advanced rank association analysis

The hybrid-wide association pipeline computes:

- Spearman correlation
- Kendall tau
- normalized mutual information
- partial Spearman correlation while controlling for the other numeric features
- Mann-Whitney separation tests for top `10%` and top `20%` average-rank groups
- two-segment piecewise breakpoint detection using `pwlf`
- derived-ratio feature analysis

The derived-ratio analysis adds:

- `entity_density`
- `token_word_ratio`
- `syllable_density`
- `sentence_complexity`

The pipeline writes:

- `association_summary.csv`
- `dot_plot.png`
- `diverging_bar.png`

The UI exposes those outputs through a cached collection under `rankings/rank_relationship/`.

#### 2. Hybrid rank-feature plots across selected queries

The feature-plot workflow is implemented in:

- [ui/templates/features_rank_plots.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\features_rank_plots.html)
- `get_or_create_rank_feature_plots` and helpers in `ui/services/feature_ui_service.py`

This page lets the user choose one or more saved queries and then join their top `100` ranking rows to the stored feature table.

It supports four plot families:

- `Ribbon plots`
- `Per-query overlay`
- `Feature consistency heatmap`
- `Violin plots`

The plotting design is explicit in code:

- ranks are limited to `1-100`
- rolling plots use a centered window of `50` rows
- `min_periods = 10`
- heatmap and violin views use the buckets `1-25`, `26-50`, `51-75`, and `76-100`

This directly answers the plan's need for visualization techniques that can compare higher and lower ranking regions without reducing everything to a single article or a single query.

#### 3. Article-level feature inspection

The article-level feature page is analytical in a different way. Instead of modeling many rankings at once, it lets the user inspect one document's stored metrics and entity profile in detail.

This is useful for:

- validating what the feature pipeline actually captured
- understanding the measured profile of a specific ranked page
- checking whether a document mapped correctly from retrieval artifacts into the feature table

#### 4. Per-model ranking inspection

The repository also contains an analysis-oriented extension that is not a separate retriever build, but a re-interpretation layer over saved hybrid rankings:

- [ui/templates/per_model_ranking.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\per_model_ranking.html)
- [ui/templates/partials/per_model_results_table.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\partials\per_model_results_table.html)

This page reorders the same saved result file by one component at a time:

- BM25 only
- SBERT only
- PageRank only

It does not create a new corpus or a new index. Instead, it rebuilds ranking order from the stored component score columns and shows:

- model rank
- hybrid rank
- shift versus hybrid
- model score
- original hybrid and component scores

This makes it possible to analyze how the hybrid ranking compares with each underlying signal.

#### 5. Per-model rank association analysis

Another analytical extension is the per-model association pipeline:

- [ui/templates/per_model_rank_relationship.html](D:\University\FCSE\Courses\26S\TUG-26S-BT-Content_Optimization\ui\templates\per_model_rank_relationship.html)
- per-model analysis logic in `ui/services/rank_relationship_service.py`

This page reruns the same association methodology as the hybrid-wide analysis, but it changes the target variable. Instead of hybrid `avg_rank`, it builds:

- BM25-based average rank
- SBERT-based average rank

This means the feature-analysis machinery can be applied separately to the sparse and dense ranking signals.

#### 6. Per-model rank-feature plots

The UI also mirrors the top-100 feature-plot workflow for single-model rankings:

- BM25-only
- SBERT-only

This supports visual comparison of how feature behavior changes depending on whether the underlying rank axis comes from lexical matching or semantic similarity.

### What kinds of ranking analysis the UI can perform now

Staying strictly within the analytical scope of Sections I-IV, the current UI can now be used to do all of the following:

- run arbitrary hybrid queries over the local pets corpus
- preserve those queries as reusable full rankings
- compare the hybrid ranking against BM25-only, SBERT-only, and PageRank-only ordering
- inspect the measured content profile of an individual article
- examine repeated entity statistics for one article
- aggregate many saved query rankings into one average-rank analysis frame
- test rank-feature relationships statistically rather than only visually
- view rolling feature behavior across top-ranked ranges
- compare feature distributions across rank buckets
- inspect whether a feature behaves consistently or variably across the top `100`
- compare feature-rank and feature-score associations across different ranking components
- inspect model-based feature importance heatmaps for ranking components

What this UI is doing at this stage is analysis, not prescription. It gives the project a way to study ranking behavior, observe measured content characteristics, and compare ranking signals on a shared corpus.

## Current end state before Section V

By the end of the work covered here, the repository has reached a stable pre-optimization analytical state:

- the Wikipedia pets corpus has been crawled and cleaned locally
- sparse, dense, and graph-based retrieval signals have been built
- a weighted hybrid ranker produces reproducible full-corpus rankings
- a saved query collection exists and is reused
- a standardized content-feature table has been generated for the cleaned corpus
- the web application can run queries, inspect articles, compare models, and analyze rank-feature relationships

That is the methodological endpoint before Section V begins. The project has already become a retrieval-and-analysis workspace. What it has not yet done, within the scope of this document, is move into the later phase of informed content-optimization intervention.
