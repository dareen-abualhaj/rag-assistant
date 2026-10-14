"""Benchmark semantic search latency over a set of queries."""
import json
import statistics
import time

from src.embeddings import load_model
from src.vector_store import get_collection, search

QUERIES = [
    "Who was the first president of the United States?",
    "How does photosynthesis work in plants?",
    "What causes earthquakes?",
    "History of the Roman Empire",
    "Which animals live in the ocean?",
    "How do airplanes fly?",
    "What is the capital city of France?",
    "Explain how the human heart works",
    "Famous painters of the Renaissance",
    "What is the speed of light?",
]
K = 5
OUT = "docs/retrieval_benchmark.json"


def main():
    model = load_model()
    collection = get_collection()
    print(f"Collection size: {collection.count()}")

    model.encode(["warm up"])  # warm-up so first query is not unfairly slow

    results, embed_ms, search_ms = [], [], []
    for q in QUERIES:
        t0 = time.perf_counter()
        qe = model.encode([q], normalize_embeddings=True)[0]
        t1 = time.perf_counter()
        hits = search(collection, qe, k=K)
        t2 = time.perf_counter()

        embed_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)
        top = hits[0] if hits else None
        print(f"\nQ: {q}")
        print(f"  embed {embed_ms[-1]:.1f} ms | search {search_ms[-1]:.1f} ms")
        if top:
            print(f"  top: {top['metadata']['title']} (distance {top['distance']:.3f})")
        results.append({
            "query": q,
            "embed_ms": round(embed_ms[-1], 2),
            "search_ms": round(search_ms[-1], 2),
            "top_title": top["metadata"]["title"] if top else None,
            "top_distance": round(top["distance"], 4) if top else None,
        })

    total = [e + s for e, s in zip(embed_ms, search_ms)]
    summary = {
        "n_queries": len(QUERIES),
        "k": K,
        "mean_total_ms": round(statistics.mean(total), 2),
        "median_total_ms": round(statistics.median(total), 2),
        "max_total_ms": round(max(total), 2),
        "mean_search_ms": round(statistics.mean(search_ms), 2),
    }
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))

    with open(OUT, "w") as f:
        json.dump({"summary": summary, "queries": results}, f, indent=2)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
