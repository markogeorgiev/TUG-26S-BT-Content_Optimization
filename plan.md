# Implemetation Plan 

## Building the Retrievers and UI for Querying

0. Build a corpus of ~21,000 pages extracted from WikiPedia's Pets category. 
    - 19 Pages rank into 404 Not Found errors. We ignore these pages when working with the dense and sparse retrievers. 
1. X * BM25 + Y * NEURAL_RETRIEVER + Z * PageRank
2. Figure out ratios (likey X, Y, and Z)
    - Find some way to decide how much influence each factor should have. 
3. While 1. and 2., build PageRank Graph, we allow 404 Pages to contribute to PageRank.
4. While 3 figure out which NEURAL_RETRIEVER. 
    - For the neural retriever we will use SBERT based `all-MiniLM-L6-v2`. 
    - For longer documents we chunk them and consider only the top 3 most relevnat chunks.
    - Preparing documents to be embedded with most retrievers requries that documents are chunked as 512 words per chunk. 
    - Chunking is best done using a `spacy` sentencinizer to split texts into sentences which are below 512 words. However, since our main chunking mechanism is the sentencizer we have to do some preprocessing even before we use it. 
        - First, our scraper createad WikiPedia text with uses edit tags under headings, along with long "See Also" and "References" sections which are redundant, all of which we need to remove.  
        - WikiPedia texts also include long lists in the format: `- [item-1]\n` which `spacy` handles as a single sentence. We use another script for these articles, to simply group every 5 list items into a sentence of the format: `The list also includes [item-1], [item-2], ... , [item-5]`. 
        - 404 error pages will continue to be ignored for the parser.
5. [REQUIREMENT: Finish Step 2.] Implement all the retrievers. 
    - Here I refer to combining all three retrievers into a single callable script agains a query supplied as argument/parameter. 
    - Ideally this script will resemble an API.

## Content Charateristics

0. Figure out which charateristics of content to track. 



X. Compile an initiall selection of queries which will be used to create content rankings. 
X. Start a web application that can be used as UI to interact with ranker and get intial results. 
    