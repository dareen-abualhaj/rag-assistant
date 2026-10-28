"""Draw 60 articles for manual sentiment labelling.

30 random articles (a representative estimate) + 30 targeted ones, because the random part
alone would contain almost only neutral articles:
  10 predicted positive by the transformer, 10 predicted negative by the transformer,
  10 predicted positive by VADER but neutral by the transformer.

Usage: python -m scripts.make_sentiment_sample [--force]
"""
import argparse
from pathlib import Path

import pandas as pd

PRED = "data/processed/doc_sentiment.parquet"
CORPUS = "data/processed/clean_corpus.parquet"
OUT = Path("data/labels/sentiment_labels.csv")
SNIPPET_CHARS = 600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing annotations")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} already exists (your labels are in it). Use --force to overwrite.")

    pred = pd.read_parquet(PRED)
    assert len(pred) > 5000, "Run the full `python -m scripts.run_sentiment` first (without --limit)."
    corpus = pd.read_parquet(CORPUS)[["id", "text"]]
    corpus["id"] = corpus["id"].astype(str)
    df = pred.merge(corpus, on="id")

    random_part = df.sample(30, random_state=42).assign(group="random")
    rest = df.drop(random_part.index)
    parts = [random_part]
    targeted = [
        ("tf_positive", lambda r: r["tf_label"] == "positive"),
        ("tf_negative", lambda r: r["tf_label"] == "negative"),
        ("vader_pos_tf_neutral", lambda r: (r["vader_label"] == "positive") & (r["tf_label"] == "neutral")),
    ]
    for name, rule in targeted:
        pool = rest[rule(rest)]
        part = pool.sample(min(10, len(pool)), random_state=42).assign(group=name)
        parts.append(part)
        rest = rest.drop(part.index)

    sample = pd.concat(parts).sample(frac=1, random_state=42)  # shuffle: order must not reveal the group
    sample["snippet"] = sample["text"].str.replace(r"\s+", " ", regex=True).str[:SNIPPET_CHARS]
    sample["gold"] = ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample[["id", "title", "group", "snippet", "gold"]].to_csv(OUT, index=False)
    print(f"Saved {len(sample)} articles to {OUT}")
    print(sample["group"].value_counts().to_string())


if __name__ == "__main__":
    main()
