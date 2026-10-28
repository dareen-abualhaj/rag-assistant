"""Evaluate VADER and the transformer against the manual sentiment labels.

Usage: python -m scripts.evaluate_sentiment
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = "data/labels/sentiment_labels.csv"
PRED = "data/processed/doc_sentiment.parquet"
OUT = "docs/sentiment_evaluation.json"
GOLD_MAP = {"p": "positive", "0": "neutral", "n": "negative"}
CLASSES = ["positive", "neutral", "negative"]
METHODS = {"VADER": "vader_label", "Transformer": "tf_label"}


def score(df, col):
    y, p = df["gold"], df[col]
    prec, rec, _, sup = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    return {
        "n": int(len(df)),
        "accuracy": round(float(accuracy_score(y, p)), 3),
        "macro_f1": round(float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)), 3),
        "per_class": {c: {"precision": round(float(a), 3), "recall": round(float(b), 3), "support": int(s)}
                      for c, a, b, s in zip(CLASSES, prec, rec, sup)},
    }


def main():
    lab = pd.read_csv(LABELS, keep_default_na=False, dtype=str)
    lab = lab[lab["gold"].str.strip() != ""].copy()
    assert len(lab) > 0, "No labels yet. Run scripts.annotate first."
    lab["gold"] = lab["gold"].str.strip().str.lower().map(GOLD_MAP)
    assert lab["gold"].notna().all(), "Unexpected label value (use p, 0 or n)"
    pred = pd.read_parquet(PRED)
    pred["id"] = pred["id"].astype(str)
    df = lab.merge(pred[["id", "vader_label", "tf_label"]], on="id")

    subsets = {"all_labelled": df, "random_only": df[df["group"] == "random"]}
    result = {"gold_distribution": df["gold"].value_counts().to_dict(), "subsets": {}}
    print(f"Labelled articles: {len(df)}   gold distribution: {result['gold_distribution']}\n")

    for sname, sub in subsets.items():
        baseline = float((sub["gold"] == "neutral").mean())
        result["subsets"][sname] = {"always_neutral_baseline": round(baseline, 3)}
        print(f"=== {sname} (n={len(sub)}) | always-'neutral' baseline accuracy: {baseline:.1%} ===")
        for mname, col in METHODS.items():
            res = score(sub, col)
            result["subsets"][sname][mname] = res
            print(f"  {mname:<12} accuracy {res['accuracy']:.1%} | macro-F1 {res['macro_f1']:.3f}")
        print()

    for mname, col in METHODS.items():
        cm = pd.crosstab(df["gold"], df[col]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
        print(f"Confusion matrix, {mname} (rows = my label, columns = prediction, all labelled):")
        print(cm.to_string(), "\n")
        result["subsets"]["all_labelled"][mname]["confusion"] = cm.to_dict()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
