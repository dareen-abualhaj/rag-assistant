"""Sentiment per article with two methods: VADER (lexicon baseline) and a 3-class transformer.
Both read only the lead of each article (first LEAD_CHARS characters).

Usage:
  python -m scripts.run_sentiment [--limit 200]
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CORPUS = "data/processed/clean_corpus.parquet"
OUT = Path("data/processed/doc_sentiment.parquet")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LEAD_CHARS = 1500
MAX_TOKENS = 256
BATCH = 32
VADER_THRESHOLD = 0.05  # standard VADER cut-off for positive / negative


def vader_scores(texts):
    """Mean compound score over the first 10 sentences (long texts saturate otherwise)."""
    sia = SentimentIntensityAnalyzer()
    scores = []
    for t in texts:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()][:10]
        scores.append(float(np.mean([sia.polarity_scores(s)["compound"] for s in sentences])) if sentences else 0.0)
    return np.array(scores)


def vader_label(score):
    return "positive" if score >= VADER_THRESHOLD else "negative" if score <= -VADER_THRESHOLD else "neutral"


def transformer_probs(texts):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    order = [id2label[i] for i in range(len(id2label))]
    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc="transformer"):
        enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            out.append(model(**enc).logits.softmax(-1).numpy())
    return np.vstack(out), order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(CORPUS).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    leads = df["text"].str[:LEAD_CHARS].tolist()
    print(f"Documents: {len(df)}")

    t0 = time.perf_counter()
    v = vader_scores(leads)
    print(f"VADER done in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    probs, order = transformer_probs(leads)
    elapsed = time.perf_counter() - t0
    print(f"Transformer done in {elapsed:.0f}s ({len(df) / elapsed:.1f} docs/s)")

    out = pd.DataFrame({
        "id": df["id"].astype(str),
        "title": df["title"],
        "vader_compound": v.round(4),
        "vader_label": [vader_label(s) for s in v],
        "tf_label": [order[i] for i in probs.argmax(1)],
    })
    for j, name in enumerate(order):
        out[f"tf_{name}"] = probs[:, j].round(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print("\n=== SUMMARY ===")
    for col in ("vader_label", "tf_label"):
        print(f"{col}:", (out[col].value_counts(normalize=True) * 100).round(1).to_dict())
    print(f"Methods agree on {(out['vader_label'] == out['tf_label']).mean():.1%} of articles")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
