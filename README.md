# Wikipedia Pets Content Optimization

This project builds a retrieval dataset from Wikipedia's `Category:Pets` graph and prepares it for content-optimization experiments.

Today the repo covers five concrete stages:

- crawl Wikipedia category descendants and export page text plus link structure
- clean raw exports into a prose-first corpus
- build a directed hyperlink graph and compute PageRank
- chunk article text into semantic-search-ready passages
- create CPU-friendly sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2`

The broader hybrid-retrieval idea from [plan.md](plan.md) is still the long-term direction: combine lexical retrieval, dense retrieval, and PageRank. The dense embedding pipeline has moved away from FlagEmbedding/BGE and is now centered on MiniLM running on CPU.

## Current Local Snapshot

The local crawl outputs currently on disk were generated on May 24, 2026.

- `output/wikipedia-pets/manifest.json` reports `21,077` discovered nodes.
- `21,058` pages finished successfully and `19` pages failed.
- The crawl export includes `1,174,075` in-scope hyperlinks.
- The article-only graph build in `data/graph/metadata.json` contains `19,567` nodes and `1,139,761` edges.
- `data/corpus/`, `data/corpus_cleaned/`, `output/wikipedia-pets/texts/`, and `output/wikipedia-pets/texts-cleaned/` each currently contain `21,058` text files.

Those numbers describe the current local dataset, not a hardcoded expectation for future reruns.

## Repository Layout

- `scripts/Export-WikipediaCategory.ps1`
  Windows-first Wikipedia crawler with resumable progress, polite delays, retry logic, and export generation.
- `scripts/Clean-WikipediaTextExports.ps1`
  Cleans the raw exported text files under `output/wikipedia-pets/texts/` into `output/wikipedia-pets/texts-cleaned/`.
- `scripts/build_embedding_corpus.py`
  Python cleaner that turns `data/corpus/` into the embedding-focused corpus in `data/corpus_cleaned/`.
- `retriever/build_hyperlink_graph.py`
  Builds the hyperlink graph and PageRank outputs from `page_index.json` and `links.csv`.
- `retriever/build_wikipedia_pet_embeddings_minilm.ipynb`
  Current embedding notebook. CPU-only. Writes chunk metadata and embedding parts in batches.
- `retriever/create_file_embeddings.py`
  Legacy FlagEmbedding/BGE script. Kept for reference, not recommended for current runs.
- `retriever/bge_model_tests.ipynb`
  Older BGE exploration notebook.
- `scripts/bge_model_tests.py`
  Currently empty placeholder file.
- `data/`
  Working data area for corpus files, graph outputs, and embedding outputs.
- `output/`
  Raw crawler outputs and crawl progress state.
- `notebooks/`
  Present in the repo, but currently empty. The active notebooks live under `retriever/`.

## Environment Setup

This repository is currently Windows-oriented because the crawler and export cleaner are PowerShell scripts.

Create or activate the virtual environment and install dependencies:

```powershell
.venv\Scripts\Activate.ps1
pip install -r .\requrements.txt
pip install sentence-transformers pyarrow
```

Notes:

- The dependency file in the repo is named `requrements.txt`.
- `sentence-transformers` and `pyarrow` are needed for the current MiniLM notebook but are not listed in `requrements.txt`.
- The embedding notebook is designed to run on CPU. Do not assume CUDA support.

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

Resume and recovery behavior:

- rerun with the same `-OutputDir` to continue from saved progress
- add `-RetryFailedPages` to retry pages previously marked as failed
- older output folders with `texts/*.txt` but no progress metadata can be bootstrapped into the new progress format

Crawler outputs:

- `output/wikipedia-pets/texts/*.txt`
- `output/wikipedia-pets/page_index.json`
- `output/wikipedia-pets/links.csv`
- `output/wikipedia-pets/manifest.json`
- `output/wikipedia-pets/failed_pages.csv`
- `output/wikipedia-pets/progress/`

### 2. Optional Raw Export Cleanup

If you want cleaned copies of the raw crawler text files, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Clean-WikipediaTextExports.ps1
```

By default this reads from `output/wikipedia-pets/texts/` and writes to `output/wikipedia-pets/texts-cleaned/`.

This cleaner:

- removes stray `edit` markers
- trims trailing sections such as `References` and `External links`
- keeps only the title plus cleaned article body

### 3. Build The Embedding Corpus

The Python corpus cleaner is the step that prepares the actual inputs used by the current embedding notebook:

```powershell
python .\scripts\build_embedding_corpus.py
```

By default it reads from `data/corpus/` and writes to `data/corpus_cleaned/`.

This stage is more aggressive than the PowerShell export cleaner. It:

- removes back matter such as references and external links
- drops table-like blocks
- compacts bullet-list runs into short prose sentences
- keeps the cleaned output focused on retrieval-friendly text

If you are starting from a fresh crawl and `data/corpus/` is empty, copy or sync the exported text files there before running this step.

### 4. Build The Hyperlink Graph

The graph stage builds from the crawler's exported `links.csv` and `page_index.json`:

```powershell
python .\retriever\build_hyperlink_graph.py build
```

By default it writes:

- `data/graph/nodes.csv`
- `data/graph/edges.csv`
- `data/graph/graph.db`
- `data/graph/metadata.json`

Useful variants:

```powershell
python .\retriever\build_hyperlink_graph.py build --completed-only
python .\retriever\build_hyperlink_graph.py build --include-categories
```

Useful queries:

```powershell
python .\retriever\build_hyperlink_graph.py query --title "Abyssinian cat"
python .\retriever\build_hyperlink_graph.py query --page-id 7590101
python .\retriever\build_hyperlink_graph.py top --limit 20
```

Graph behavior notes:

- default graph scope is article pages only
- failed article pages can still remain in the graph unless `--completed-only` is used
- this lets unresolved pages act like dangling nodes instead of disappearing from the discovered set

### 5. Build CPU MiniLM Embeddings

The current dense-retrieval workflow lives in:

- [retriever/build_wikipedia_pet_embeddings_minilm.ipynb](retriever/build_wikipedia_pet_embeddings_minilm.ipynb)

Open it with Jupyter and run it top-to-bottom.

The notebook is designed for a Windows machine without CUDA. It explicitly loads:

```python
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
```

Key notebook behavior:

- uses `spacy.blank("en")` with a sentencizer instead of `en_core_web_sm`
- chunks text by sentence with a default chunk size of `180` words
- writes metadata to parquet parts instead of holding the full dataset in memory
- writes embeddings to `.npy` parts that can be resumed independently
- supports optional final combined files and an optional semantic-search demo

Expected embedding outputs:

- `data/embeddings/chunk_metadata_parts/chunk_metadata_part_0000.parquet`
- `data/embeddings/chunk_metadata_parts/skipped_pages.parquet`
- `data/embeddings/embedding_parts/chunk_embeddings_part_0000.npy`
- optional `data/embeddings/chunk_metadata.parquet`
- optional `data/embeddings/chunk_embeddings.npy`

Operational notes:

- start with `batch_size=64`
- if RAM rises above 90%, reduce to `32`
- if RAM stays comfortable, try `128`
- keep the part files if the combined outputs are too large for memory
- because embeddings are normalized, dot product is equivalent to cosine similarity
- `all-MiniLM-L6-v2` produces `384`-dimensional embeddings

## Current And Legacy Embedding Code

Use the MiniLM notebook for current work.

Legacy files still in the repo:

- `retriever/create_file_embeddings.py`
- `retriever/bge_model_tests.ipynb`

Why they are legacy:

- they use `FlagEmbedding` and `BAAI/bge-base-en-v1.5`
- they assume CUDA-oriented configuration
- they reload spaCy inside the chunking path
- they are more memory-heavy than the current batched notebook

## Practical Notes

- `output/wikipedia-pets*`, `data/`, `.venv/`, and `__pycache__/` are ignored by git.
- The repo already contains large generated artifacts locally, but they are not meant to be versioned.
- The current project state is closer to a data pipeline plus retrieval-prep workspace than a finished packaged application.
- The hybrid ranking step described in `plan.md` is still future work. Right now the crawler, graph build, and dense embedding pipeline are the main implemented pieces.
