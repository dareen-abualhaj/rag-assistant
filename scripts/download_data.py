"""Download 6000 Simple English Wikipedia articles into data/raw/."""
from pathlib import Path

from datasets import load_dataset

N_DOCS = 6000
OUT = Path("data/raw/wikipedia_simple.parquet")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split=f"train[:{N_DOCS}]")
    print(f"Downloaded {len(ds)} documents")
    assert len(ds) >= 5000, "Need at least 5000 documents"
    ds.to_parquet(str(OUT))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
