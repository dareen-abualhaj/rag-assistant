"""Discover corpus themes with BERTopic (article level).

Outputs:
  data/processed/doc_topics.parquet   id, title, n_chars, topic  (-1 = outlier)
  docs/topics_overview.csv            one row per topic, with an empty `label` column to fill by hand

Usage:
  python -m scripts.run_topic_modeling
  python -m scripts.run_topic_modeling --min-cluster-size 30
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.embeddings import embed_texts, load_model

CORPUS = Path("data/processed/clean_corpus.parquet")
EMB_CACHE = Path("data/interim/doc_embeddings.npy")
OUT_DOCS = Path("data/processed/doc_topics.parquet")
OUT_OVERVIEW = Path("docs/topics_overview.csv")
EMBED_CHARS = 2000  # the embedding model only reads the first ~256 tokens anyway


def get_embeddings(model, docs):
    if EMB_CACHE.exists():
        emb = np.load(EMB_CACHE)
        if emb.shape[0] == len(docs):
            print(f"Loaded cached embeddings {emb.shape} from {EMB_CACHE}")
            return emb
    emb = embed_texts(model, [d[:EMBED_CHARS] for d in docs])
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_CACHE, emb)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster-size", type=int, default=20,
                        help="smaller -> more, finer topics (and fewer outliers)")
    args = parser.parse_args()

    df = pd.read_parquet(CORPUS).reset_index(drop=True)
    docs = df["text"].tolist()
    print(f"Documents: {len(docs)}")

    model = load_model()
    embeddings = get_embeddings(model, docs)

    topic_model = BERTopic(
        embedding_model=model,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=5,
                              metric="euclidean", cluster_selection_method="eom",
                              prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)),
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    out = pd.DataFrame({
        "id": df["id"].astype(str),
        "title": df["title"],
        "n_chars": df["text"].str.len(),
        "topic": topics,
    })
    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DOCS, index=False)

    # overview table for manual validation
    rows = []
    for t, group in out.groupby("topic"):
        words = [w for w, _ in topic_model.get_topic(t)[:8]] if t != -1 else []
        sample = group["title"].sample(min(4, len(group)), random_state=42).tolist()
        rows.append({
            "topic": t,
            "count": len(group),
            "top_words": ", ".join(words),
            "sample_titles": " | ".join(sample),
            "label": "",
        })
    overview = pd.DataFrame(rows).sort_values("count", ascending=False)
    OUT_OVERVIEW.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(OUT_OVERVIEW, index=False)

    # summary
    n_topics = int((out["topic"] != -1).sum() and out.loc[out["topic"] != -1, "topic"].nunique())
    n_out = int((out["topic"] == -1).sum())
    print("\n=== SUMMARY ===")
    print(f"Topics found (excluding outliers): {n_topics}")
    print(f"Outliers (topic -1): {n_out} ({n_out / len(out):.1%})")
    print(f"Median length, outliers vs clustered docs: "
          f"{out.loc[out.topic == -1, 'n_chars'].median():.0f} vs "
          f"{out.loc[out.topic != -1, 'n_chars'].median():.0f} chars")
    print("\nLargest topics:")
    for _, r in overview[overview.topic != -1].head(15).iterrows():
        print(f"  topic {r.topic:>3} | {r['count']:>4} docs | {r.top_words}")
    print(f"\nSaved {OUT_DOCS} and {OUT_OVERVIEW}")


if __name__ == "__main__":
    main()
