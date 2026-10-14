"""ChromaDB wrapper: ingestion and semantic search with edge-case handling."""
import logging
from typing import Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)

DB_PATH = "chroma_db"
COLLECTION = "wiki_chunks"
MAX_BATCH = 5000  # stay below Chroma's max batch size


def get_collection(path: str = DB_PATH, name: str = COLLECTION, reset: bool = False):
    client = chromadb.PersistentClient(path=path)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # collection did not exist
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def add_chunks(collection, ids: List[str], texts: List[str],
               embeddings, metadatas: List[Dict]) -> int:
    """Ingest in batches. Uses upsert so re-running does not create duplicates."""
    assert len(ids) == len(texts) == len(embeddings) == len(metadatas), "length mismatch"
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ids in input")
    for i in range(0, len(ids), MAX_BATCH):
        j = i + MAX_BATCH
        collection.upsert(
            ids=ids[i:j],
            documents=texts[i:j],
            embeddings=[e.tolist() for e in embeddings[i:j]],
            metadatas=metadatas[i:j],
        )
    return len(ids)


def search(collection, query_embedding, k: int = 5,
           where: Optional[Dict] = None) -> List[Dict]:
    """Semantic search. Handles empty collection and k > collection size."""
    n = collection.count()
    if n == 0:
        logger.warning("Collection is empty")
        return []
    res = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(k, n),
        where=where,
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
