"""Evaluate the RAG pipeline on in-domain vs off-topic questions.

Usage:  python -m scripts.evaluate_rag
Writes: docs/rag_evaluation.json and prints a README-ready table.
Uses the real Groq API (about 10 LLM calls), so GROQ_API_KEY must be set in .env.
"""
import json
import logging
import os
import statistics
import time

from src.rag import MAX_DISTANCE, RAGPipeline

IN_DOMAIN = [
    "Who was the first president of the United States?",
    "How does photosynthesis work in plants?",
    "What causes earthquakes?",
    "History of the Roman Empire",
    "How do airplanes fly?",
    "What is the capital city of France?",
    "Explain how the human heart works",
    "What is the speed of light?",
    "How do volcanoes erupt?",
    "What is gravity?",
]
OFF_TOPIC = [
    "Write me a Python function to sort a list",
    "What will the weather be like tomorrow?",
    "Write a poem about my cat",
    "What is my name?",
    "Book me a flight to London for next Friday",
    "What is 2457 times 389?",
]
REFUSED = {"out_of_domain", "no_answer", "unverified"}
OUT_PATH = "docs/rag_evaluation.json"
PAUSE_S = 5.0  # pause after each LLM call so Groq free-tier rate limits do not distort latency


def run(pipeline, in_domain=IN_DOMAIN, off_topic=OFF_TOPIC, pause_s=PAUSE_S):
    rows = []
    for kind, questions in (("in_domain", in_domain), ("off_topic", off_topic)):
        for q in questions:
            r = pipeline.answer(q)
            rows.append({
                "kind": kind,
                "question": q,
                "status": r["status"],
                "top_distance": r["top_distance"],
                "warnings": r.get("warnings", []),
                "retrieval_ms": r["latency"]["retrieval_ms"],
                "generation_ms": r["latency"]["generation_ms"],
                "total_ms": r["latency"]["total_ms"],
                "error": r.get("error"),
                "answer": r["answer"],
            })
            if r["latency"]["generation_ms"] > 0 and pause_s:
                time.sleep(pause_s)
    return rows


def _p95(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[18]


def summarize(rows, threshold=MAX_DISTANCE):
    ind = [r for r in rows if r["kind"] == "in_domain"]
    off = [r for r in rows if r["kind"] == "off_topic"]
    ind_d = [r["top_distance"] for r in ind if r["top_distance"] is not None]
    off_d = [r["top_distance"] for r in off if r["top_distance"] is not None]
    lat = [r["total_ms"] for r in ind if r["status"] == "answered"]
    max_in = max(ind_d) if ind_d else None
    min_off = min(off_d) if off_d else None
    return {
        "n_in_domain": len(ind),
        "n_off_topic": len(off),
        "in_domain_answered": sum(r["status"] == "answered" for r in ind),
        "off_topic_refused": sum(r["status"] in REFUSED for r in off),
        "off_topic_blocked_at_gate": sum(r["status"] == "out_of_domain" for r in off),
        "max_in_domain_distance": max_in,
        "min_off_topic_distance": min_off,
        "threshold": threshold,
        "threshold_separates": (max_in is not None and min_off is not None
                                and max_in <= threshold < min_off),
        "median_total_ms_answered": round(statistics.median(lat), 1) if lat else None,
        "p95_total_ms_answered": round(_p95(lat), 1) if lat else None,
    }


def print_report(rows, s):
    print("\nPer-question results")
    for r in rows:
        d = "n/a" if r["top_distance"] is None else f"{r['top_distance']:.3f}"
        print(f"  [{r['kind']:<9}] {r['status']:<13} dist={d}  {r['total_ms']:>7.0f} ms  {r['question']}")
        if r["warnings"]:
            print(f"      warnings: {r['warnings']}")
        if r.get("error"):
            print(f"      error: {r['error']}")
    f = lambda v, nd=3: "n/a" if v is None else (f"{v:.{nd}f}")
    print("\n| Metric | Result |\n|---|---|")
    print(f"| In-domain questions answered | {s['in_domain_answered']} / {s['n_in_domain']} |")
    print(f"| Off-topic questions refused | {s['off_topic_refused']} / {s['n_off_topic']} "
          f"({s['off_topic_blocked_at_gate']} blocked before the LLM) |")
    print(f"| Max in-domain top distance | {f(s['max_in_domain_distance'])} |")
    print(f"| Min off-topic top distance | {f(s['min_off_topic_distance'])} |")
    print(f"| Threshold {s['threshold']} separates the two groups | "
          f"{'yes' if s['threshold_separates'] else 'no'} |")
    print(f"| Median total latency (answered, with LLM call) | {f(s['median_total_ms_answered'], 0)} ms |")
    print(f"| p95 total latency (answered, with LLM call) | {f(s['p95_total_ms_answered'], 0)} ms |")


def main():
    logging.basicConfig(level=logging.WARNING)
    pipeline = RAGPipeline(log_path="logs/evaluation_latency.jsonl")
    pipeline._embed(["warm up"])  # keep model warm-up out of the latency numbers
    rows = run(pipeline)
    summary = summarize(rows)
    print_report(rows, summary)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": rows}, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()