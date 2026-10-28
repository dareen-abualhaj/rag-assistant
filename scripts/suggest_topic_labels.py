"""Suggest a short label for each topic with the LLM (to be reviewed by hand).

Reads docs/topics_overview.csv and adds a `label_suggested` column.
Usage: python -m scripts.suggest_topic_labels
"""
import logging
import time

import pandas as pd

from src.llm import LLMError, generate

CSV = "docs/topics_overview.csv"
PAUSE_S = 2.5

PROMPT = """You label clusters of Simple English Wikipedia articles.
Cluster keywords: {words}
Example article titles: {titles}

Give a short topic label (1 to 3 words, Title Case) that describes the whole cluster.
Reply with the label only, nothing else."""


def clean(label: str) -> str:
    line = label.strip().splitlines()[0]
    return line.strip(" \"'*.`")[:40]


def main():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    df = pd.read_csv(CSV, keep_default_na=False)
    df["label_suggested"] = ""
    for i, row in df.iterrows():
        if int(row["topic"]) == -1:
            df.at[i, "label_suggested"] = "Uncategorized"
            continue
        msg = [{"role": "user", "content": PROMPT.format(
            words=row["top_words"], titles=row["sample_titles"])}]
        try:
            label = clean(generate(msg, max_tokens=1500))
        except LLMError as e:
            print(f"topic {row['topic']}: failed ({e})")
            label = ""
        df.at[i, "label_suggested"] = label
        print(f"topic {row['topic']:>3} | {row['top_words'][:50]:<50} -> {label}")
        time.sleep(PAUSE_S)
    df.to_csv(CSV, index=False)
    print(f"\nSaved {CSV}")


if __name__ == "__main__":
    main()
