from FlagEmbedding import FlagModel 
from pathlib import Path
import numpy as np
import os
import spacy
import json
import csv


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_DIR = Path(__file__).parent.parent
CORPUS_DIR = ROOT_DIR / 'data' / 'corpus'


def read_file_contents(file_path):
    file_contents = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        file_contents += f.read()
    return file_contents


def smart_text_chunking(text, chunk_size=512):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('sentencizer')

    sentences = [sent.text for sent in nlp(text).sents]

    for sentence in sentences:
        if len(sentence.split()) > chunk_size:
            raise ValueError(f"Sentence is too long to fit in a chunk: {sentence}")

    # 1. Count words in sentences until >= 512, when >= 512, create chunk then start new chunk. 

    chunks = []

    current_chunk = ""

    for sentence in sentences:
        if len(sentence.split()) + len(current_chunk.split()) <= chunk_size:
            current_chunk += " " + sentence 
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    return chunks

def get_file_chunks(corpus_dir=CORPUS_DIR):
    # Matching chunking to embeddings.  
    # - Read the JSON from output/wikipedia_pets/page_index.json. 
    # - Read article file_name, and match it to page_index.json's text_file field without the texts/ prefix. 
    # - Match chunks with files via page_id, so we can track which embeddings belong to which files. 

    # page_id, chunk_id, chunk_text
    chunk_info_file = open("../data/embeddings/chunk_information.csv", mode='w', encoding='utf-8-sig')
    chunk_info_writer = csv.writer(chunk_info_file)

    all_article_chunks = []

    with open('../output/wikipedia-pets/wikipedia-pets/page_index.json', mode='r', encoding='utf-8-sig') as f:
        for entry in json.load(f):
            try: 
                curr_file = entry["text_file"].replace('texts/', '')
                curr_page_id = entry["page_id"]
                open("../data/corpus/" + curr_file, 'r', encoding='utf-8')
                curr_chunks = smart_text_chunking(read_file_contents(CORPUS_DIR / curr_file))
                if len(curr_chunks) == 1:
                    # page_id, chunk_id, chunk_text
                    chunk_info_writer.writerow([curr_page_id, 0, curr_chunks[0]])
                else:
                    for i, chunk in enumerate(curr_chunks):
                        chunk_info_writer.writerow([curr_page_id, i, chunk])
                all_article_chunks.extend(curr_chunks)
                
            except AttributeError as e:
                print(f"Skipping 404 - {curr_file}")

    return all_article_chunks

def get_file_embeddings(corpus_dir=CORPUS_DIR, all_article_chunks=None):
    all_article_chunks = get_file_chunks(corpus_dir) if all_article_chunks is None else all_article_chunks

    model = FlagModel('BAAI/bge-base-en-v1.5')

    embeddings = model.encode(all_article_chunks)

    np.save(ROOT_DIR / 'data' / 'embeddings' / 'file_embeddings.npy', embeddings)

if __name__ == "__main__":
    get_file_embeddings(CORPUS_DIR)

