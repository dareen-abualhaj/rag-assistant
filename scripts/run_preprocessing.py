"""Run the cleaning pipeline on raw data and write a validated clean corpus."""
from pathlib import Path

import pandas as pd

from src.preprocessing import clean_text, is_english

RAW = Path("data/raw/wikipedia_simple.parquet")
OUT = Path("data/processed/clean_corpus.parquet")
MIN_CHARS = 100


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW)
    n_raw = len(df)
    print(f"Raw documents: {n_raw}")

    # 1. Clean
    df["text"] = df["text"].map(clean_text)

    # 2. Drop empty / very short documents
    df = df[df["text"].str.len() >= MIN_CHARS]
    n_after_len = len(df)
    print(f"After removing short docs (<{MIN_CHARS} chars): {n_after_len}")

    # 3. Language filter
    df = df[df["text"].map(is_english)]
    n_after_lang = len(df)
    print(f"After language filter (English only): {n_after_lang}")

    # 4. Drop duplicates
    df = df.drop_duplicates(subset="text")
    n_final = len(df)
    print(f"After removing duplicates: {n_final}")

    # 5. Validation
    assert n_final > 0, "Clean corpus is empty"
    assert df["text"].str.len().min() >= MIN_CHARS, "Found too-short text"
    assert not df["text"].duplicated().any(), "Found duplicate texts"
    assert df["text"].notna().all(), "Found null texts"

    df = df.reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    print("\n--- Stats ---")
    print(f"Raw:   {n_raw} docs | mean length {pd.read_parquet(RAW)['text'].str.len().mean():.0f} chars")
    print(f"Clean: {n_final} docs | mean length {df['text'].str.len().mean():.0f} chars")
    print(f"Kept {n_final / n_raw:.1%} of documents")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

