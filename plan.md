# Implemetation Plan 


## I. Building the Retrievers and UI for Querying

0. Build a corpus of ~21,000 pages extracted from WikiPedia's Pets category. 
    - 19 Pages rank into 404 Not Found errors. We ignore these pages when working with the dense and sparse retrievers. 
1. Define: X * BM25 + Y * NEURAL_RETRIEVER + Z * PageRank
2. Figure out ratios (likey X, Y, and Z)
    - Find some way to decide how much influence(importance) each factor should have.
    - We decide for the following importances: 
        - NEURAL_RETRIEVER (SBERT): Y = 0.4 
        - BM25: X = 0.4
        - PageRank: Z = 0.2
    - Ranking requies normalization //TODO eexplain normalization
3. While 1. and 2., build PageRank Graph, we allow 404 Pages to contribute to PageRank.
4. While 3 figure out which NEURAL_RETRIEVER. 
    - For the neural retriever we will use the SBERT-based `sentence-transformers/all-MiniLM-L6-v2` model.
    - Since `all-MiniLM-L6-v2` has a limited input length, longer Wikipedia documents cannot be embedded reliably as a single vector. Instead, each document is split into smaller sentence-based chunks, and each chunk is embedded separately.
    - In our current setup, chunks are built with a target size of approximately 180 words. This is because `all-MiniLM-L6-v2` is optimized for short passages. Larger 512-word chunks risk truncation or weaker semantic representations.
    - Chunking is done using a lightweight `spacy.blank("en")` pipeline with a `sentencizer`. The sentencizer splits each document into sentences, and we then group consecutive sentences into chunks without exceeding the configured chunk size.
    - We do not split inside sentences. If a single sentence exceeds the configured chunk size, that document is skipped or handled separately, because such text usually comes from malformed scraped content, long lists, or references.
    - During retrieval, the query is embedded once and compared against all chunk embeddings. Chunk-level scores are then aggregated back to the document level, so the neural retriever still returns ranked documents rather than isolated chunks.
    - We do not simply use only the top 3 chunks as the whole retrieval result. Instead, the strongest chunks from each document contribute to a document-level score. In the current implementation, the best three chunk scores receive most of the weight, while the remaining chunks can still contribute a smaller amount through their mean score.
    - Before chunking, the scraped Wikipedia text should be cleaned:
        - Remove edit tags under headings.
        - Remove or truncate redundant sections such as "See also", "References", "External links", and similar metadata-heavy sections.
        - Normalize malformed list-heavy text where many bullet items are treated by the sentencizer as one long sentence.
    - Wikipedia texts often include long lists in the format `- [item-1]\n`, which `spacy` may treat as a single sentence. For these cases, we use preprocessing that groups list items into shorter natural-language sentences, for example: `The list also includes [item-1], [item-2], ..., [item-5].`
    - The 404 error pages are ignored for both the dense neural retriever and the sparse BM25 retriever.
5. Compile an initiall selection of queries which will be used to create content rankings. 
6

## II. Content Charateristics

0. Figure out which charateristics of content to track. 
    - From structural features: 
        - Word count, 
        - Sentence count, 
        - Readability (Flesch Reading Ease or Flesch-Kincaid Grade Level), 
        - Lexical diversity
    - From semantic features: 
        - Possibly based on BERTopic
    - What kind of features are there? 


## III. Web UI v1

1. The UI should allow me to execute arbitrary queries and get results based on PageRank, BM25 and SBERT (Based on I.2.)
2. The WebUI should display statistics of content on a per content level (defined in III). 
3. Display averaged content charateristics based on top-k results from queries defined in section I.5.


## IV. Analysis of Ranking 

X. Considering the content charateristics defined in section III, analyze the relationship between these charateristics and ranking. 
    - Figure out how to model relationships like these most effectively. 
    - Look into Spearmans ...//TODO

## V. Web UI v2

Extend the Web UI to allow for informed informed content optimization.

### Change Content Based on [SOMETHING]

### Change Content Based on [SOMETHING]

### Change Content Based on [SOMETHING]

## VI. Informed Content Change Experiments
1. Evaluate whether informaed changes in content charateristics had the expected effect on content ranking. How chaning the features detaile in II, affected ranknig. Was it the way we expected it would, based on VI. 