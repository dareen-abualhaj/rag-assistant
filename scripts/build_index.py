"""Chunk the clean corpus, embed chunks, and populate ChromaDB."""
import argparse
import time

import pandas as pd

from src.chunking import chunk_text
from src.embeddings import embed_texts, load_model
from src.vector_store import add_chunks, get_collection

CORPUS = "data/processed/clean_corpus.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="use only first N docs (for quick tests)")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_parquet(CORPUS)
    if args.limit:
        df = df.head(args.limit)
    print(f"Documents: {len(df)}")

    t0 = time.perf_counter()
    ids, texts, metas = [], [], []
    for row in df.itertuples():
        for i, chunk in enumerate(chunk_text(row.text, args.chunk_size, args.overlap)):
            ids.append(f"{row.id}_{i}")
            texts.append(chunk)
            metas.append({"doc_id": str(row.id), "title": row.title,
                          "url": row.url, "chunk_index": i})
    print(f"Chunks: {len(texts)} (chunking took {time.perf_counter() - t0:.1f}s)")

    model = load_model()
    embeddings = embed_texts(model, texts)

    collection = get_collection(reset=True)
    add_chunks(collection, ids, texts, embeddings, metas)
    print(f"ChromaDB count: {collection.count()}")


if __name__ == "__main__":
    main()
