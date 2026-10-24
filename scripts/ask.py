"""Ask the RAG assistant a question.

Usage:
    python -m scripts.ask "How do airplanes fly?"
    python -m scripts.ask            # interactive mode
"""
import argparse
import logging

from src.rag import RAGPipeline


def show(r: dict) -> None:
    print(f"\nStatus: {r['status']}")
    print(f"Answer: {r['answer']}")
    if r["sources"]:
        print("\nSources:")
        for s in r["sources"]:
            print(f"  [{s['n']}] {s['title']} (distance {s['distance']}) {s['url']}")
    for w in r["warnings"]:
        print(f"  WARNING: {w}")
    lat = r["latency"]
    print(f"\nLatency: retrieval {lat['retrieval_ms']} ms | generation {lat['generation_ms']} ms "
          f"| total {lat['total_ms']} ms\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()

    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    rag = RAGPipeline()
    if args.question:
        show(rag.answer(" ".join(args.question)))
        return
    while True:
        q = input("Question (empty to quit): ").strip()
        if not q:
            break
        show(rag.answer(q))


if __name__ == "__main__":
    main()
