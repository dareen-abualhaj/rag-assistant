"""Draw a random sample of articles for manual topic validation.

Usage: python -m scripts.make_topic_validation_sample [--n 60] [--force]
"""
import argparse
from pathlib import Path

import pandas as pd

TOPICS = "data/processed/doc_topics.parquet"
CORPUS = "data/processed/clean_corpus.parquet"
OVERVIEW = "docs/topics_overview.csv"
OUT = Path("data/labels/topic_validation.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--force", action="store_true", help="overwrite existing annotations")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} already exists (your annotations are in it). Use --force to overwrite.")

    topics = pd.read_parquet(TOPICS)
    corpus = pd.read_parquet(CORPUS)[["id", "text"]]
    corpus["id"] = corpus["id"].astype(str)
    overview = pd.read_csv(OVERVIEW, keep_default_na=False)[["topic", "label", "top_words"]]

    df = topics.merge(corpus, on="id").merge(overview, on="topic")
    df = df[df["topic"] != -1]
    sample = df.sample(args.n, random_state=42).copy()
    sample["snippet"] = sample["text"].str.replace(r"\s+", " ", regex=True).str[:400]
    sample["fits"] = ""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample[["id", "title", "topic", "label", "top_words", "snippet", "fits"]].to_csv(OUT, index=False)
    print(f"Saved {len(sample)} articles to {OUT}")


if __name__ == "__main__":
    main()
