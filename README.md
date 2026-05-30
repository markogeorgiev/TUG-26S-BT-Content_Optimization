# Wikipedia Pets Scraper

This repo now includes a PowerShell scraper for `https://en.wikipedia.org/wiki/Category:Pets` and every descendant subcategory it can reach through Wikipedia's category graph.

## What It Produces

- One plaintext `.txt` file per collected page or subcategory page.
- A `page_index.json` file with page metadata and file locations.
- A directional `links.csv` file describing in-scope hyperlinks from source page to target page.
- A `manifest.json` summary with crawl counts and the output path.
- A `failed_pages.csv` file listing pages that still failed after retries.
- A `progress/` folder that lets the crawl resume on the next run.

The script saves its outputs under `output/wikipedia-pets/` by default.

## Clean Existing Exports

If you want a second pass over the already-exported text files that removes stray `edit` markers and trims trailing back-matter sections such as `References` and `External links`, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Clean-WikipediaTextExports.ps1
```

By default it reads from `output/wikipedia-pets/texts/` and writes cleaned copies to `output/wikipedia-pets/texts-cleaned/` without modifying the originals. The cleaned files keep only the page title as a header line and drop the extra export metadata fields.

## Build The Hyperlink Graph

The scraper already exports a directional edge list in `output/wikipedia-pets/links.csv`, so the Python graph stage should build from that file instead of re-extracting links from the cleaned text corpus.

Run:

```powershell
python .\scripts\build_hyperlink_graph.py build
```

By default this writes a queryable graph into `data/graph/`:

- `nodes.csv` with node metadata, PageRank, and rank.
- `edges.csv` with normalized directed edges.
- `graph.db` as a SQLite database for fast lookups.
- `metadata.json` with graph and PageRank build settings.

The default scope is article pages only, but it still keeps discovered article nodes whose page fetch failed so every article in the discovered set can still be queried. Those failed pages behave like dangling nodes because their outbound links are unknown. If you want a stricter graph, use:

```powershell
python .\scripts\build_hyperlink_graph.py build --completed-only
```

To include category pages as nodes too, add:

```powershell
python .\scripts\build_hyperlink_graph.py build --include-categories
```

Query a PageRank value by title or `page_id`:

```powershell
python .\scripts\build_hyperlink_graph.py query --title "Abyssinian cat"
python .\scripts\build_hyperlink_graph.py query --page-id 7590101
```

Show the current top-ranked nodes:

```powershell
python .\scripts\build_hyperlink_graph.py top --limit 20
```

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Export-WikipediaCategory.ps1
```

Optional parameters:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Export-WikipediaCategory.ps1 `
  -BaseCategory 'Category:Pets' `
  -OutputDir '.\output\wikipedia-pets' `
  -RequestDelayMs 3000 `
  -RequestDelayJitterMs 5000 `
  -MaxRetries 3 `
  -IncludeAllNamespaces
```

To resume a stopped crawl, run the same command again with the same `-OutputDir`. The script will load `progress/inventory.json`, skip pages that already finished, and continue from the remaining pending pages.

If the crawl directory comes from an older script run and already contains `texts/*.txt` files but no `progress/` metadata yet, the script will bootstrap progress from those saved text files and continue from the remaining pages instead of restarting the full fetch phase.

If you want to re-attempt pages that were previously marked as failed, add:

```powershell
-RetryFailedPages
```

## Implementation Notes

- Category traversal uses MediaWiki's `categorymembers` API so pagination and nested subcategories are handled reliably.
- Page text and hyperlinks come from the rendered HTML, because Wikipedia category pages expose generated member lists in the live HTML blocks such as `mw-subcategories` and `mw-pages`, but those generated lists do not appear in the `action=parse` API output.
- By default the crawl keeps article pages (namespace `0`) plus category pages (namespace `14`), which avoids pulling in maintenance templates from descendant stub categories. Add `-IncludeAllNamespaces` if you want the full namespace spread instead.
- Requests now pause with randomized waits after each API and page fetch. The default pacing is intentionally conservative: `3000ms` base delay plus up to `5000ms` of jitter.
- Retry behavior is also softer for rate limiting: if Wikipedia responds with `429 Too Many Requests`, the script waits much longer before trying again and respects `Retry-After` when it is available.
- Page fetch failures are no longer fatal to the whole crawl. The script records the error, saves progress immediately, and moves on to the next page.
- Progress is persisted per page under `progress/page-state/`, so a rerun can continue without re-fetching everything that already succeeded.
- Category discovery is checkpointed too in `progress/discovery_state.json`, so if the inventory-building phase is interrupted, the next run resumes the category walk rather than starting from the root.
- Long category listings are resumable mid-pagination: the discovery checkpoint now stores the active category plus its `cmcontinue` cursor, so an interrupted `categorymembers` crawl can continue inside that category instead of replaying it from page 1.
- Legacy output folders that only have saved text files can be upgraded in place: the script can infer completed pages from the existing `texts/` files and write fresh progress metadata for future resumes.
- The plaintext extraction is intentionally lightweight and strips common non-prose elements such as styles, scripts, references, navboxes, and infobox-like tables.
