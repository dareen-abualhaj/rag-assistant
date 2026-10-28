"""Topic quality report: accuracy on the manual sample + outlier rate by document length.

Usage: python -m scripts.evaluate_topics
"""
import json

import pandas as pd

VALIDATION = "data/labels/topic_validation.csv"
TOPICS = "data/processed/doc_topics.parquet"
OVERVIEW = "docs/topics_overview.csv"
OUT = "docs/topic_evaluation.json"


def main():
    # 1. accuracy on the manually judged sample
    val = pd.read_csv(VALIDATION, keep_default_na=False, dtype=str)
    judged = val[val["fits"].str.strip().isin(["y", "n"])]
    assert len(judged) > 0, "No annotations yet. Run scripts.annotate first."
    acc = (judged["fits"] == "y").mean()
    wrong = judged[judged["fits"] == "n"]
    print(f"Judged {len(judged)} / {len(val)} sampled articles")
    print(f"Topic-label accuracy: {acc:.1%} ({(judged['fits'] == 'y').sum()} fit, {len(wrong)} do not)")
    if len(wrong):
        print("\nArticles that did NOT fit their topic label:")
        for _, r in wrong.iterrows():
            print(f"  - {r['title']}  ->  {r['label']} (topic {r['topic']})")

    # 2. edge case: outliers vs document length
    docs = pd.read_parquet(TOPICS)
    bins = [0, 500, 1000, 2000, 5000, 10**9]
    names = ["<500", "500-1000", "1000-2000", "2000-5000", ">5000"]
    docs["len_bucket"] = pd.cut(docs["n_chars"], bins=bins, labels=names)
    by_len = docs.groupby("len_bucket", observed=True).apply(
        lambda g: pd.Series({"docs": len(g), "outlier_rate": round((g["topic"] == -1).mean(), 3)}),
        include_groups=False,
    )
    print("\nOutlier rate by document length (chars):")
    print(by_len.to_string())

    # 3. share of date/year pages
    overview = pd.read_csv(OVERVIEW, keep_default_na=False)
    date_topics = overview.loc[overview["label"] == "Dates & Years", "topic"].tolist()
    n_dates = int(docs["topic"].isin(date_topics).sum())
    print(f"\nDate/year pages ({len(date_topics)} topics): {n_dates} docs = {n_dates / len(docs):.1%} of corpus")

    result = {
        "sample_size": int(len(judged)),
        "accuracy": round(float(acc), 3),
        "not_fitting": wrong[["title", "label", "topic"]].to_dict("records"),
        "outlier_rate_by_length": {k: v["outlier_rate"] for k, v in by_len.iterrows()},
        "date_pages": {"topics": date_topics, "docs": n_dates, "share": round(n_dates / len(docs), 3)},
        "n_topics": int(docs.loc[docs.topic != -1, "topic"].nunique()),
        "outliers": int((docs.topic == -1).sum()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
