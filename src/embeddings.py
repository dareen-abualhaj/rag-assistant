"""Generate sentence embeddings and log timing performance."""
import logging
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, 384-dim


def load_model(device: str = None) -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("Loaded %s on device: %s", MODEL_NAME, model.device)
    return model


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts and log total time + throughput."""
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "Embedded %d texts in %.2fs (%.1f texts/sec)",
        len(texts), elapsed, len(texts) / elapsed,
    )
    return embeddings
