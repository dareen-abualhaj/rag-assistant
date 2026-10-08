# Week 1: Environment, Data Ingestion & Preprocessing

## What I built
- Reproducible Python environment (venv + locked `requirements.txt`)
- `scripts/verify_env.py`: checks CUDA/GPU and core imports (pandas, sentence-transformers, ChromaDB, spaCy)
- `scripts/download_data.py`: downloads 6,000 Simple English Wikipedia articles into `data/raw/` (git-ignored)
- `src/preprocessing.py`: modular cleaning functions (HTML stripping, Unicode normalization, whitespace cleanup, language filtering)
- `tests/test_preprocessing.py`: 10 unit tests (pytest)
- `scripts/run_preprocessing.py`: runs the pipeline and writes `data/processed/clean_corpus.parquet`

## Approach
- **Dataset:** Simple English Wikipedia (`wikimedia/wikipedia`, `20231101.simple`) from Hugging Face: real-world text, easy to download, 6,000 documents.
- **Cleaning:** each step is a separate function so it can be tested and reused. Order: strip HTML -> NFKC Unicode normalization (and removal of zero-width/control chars) -> whitespace collapse.
- **Filtering:** documents under 100 characters are dropped, then non-English documents (`langdetect`, seeded for determinism), then exact duplicates.
- **Validation:** the run script asserts the output is non-empty, has no nulls, no duplicates and no short texts.

## Data statistics
| | Documents | Mean length (chars) |
|---|---|---|
| Raw | 6,000 | 3,898 |
| After cleaning | 5,939 | 3,886 |

- Removed: 52 short docs, 9 non-English docs, 0 duplicates (99.0% kept).

## GPU / CUDA note
My local machine has an Intel Iris Xe GPU (no NVIDIA/CUDA), so `verify_env.py` was run on **Google Colab (Tesla T4)**. Proof: `docs/colab_cuda_check.png`.
Locally, run with `--allow-cpu`.

## How to run
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/verify_env.py --allow-cpu   # drop --allow-cpu on a CUDA machine
python scripts/download_data.py            # -> data/raw/
pytest -v                                  # run unit tests
python -m scripts.run_preprocessing        # -> data/processed/clean_corpus.parquet
```

## Project structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── docs/colab_cuda_check.png
├── scripts/   (verify_env.py, download_data.py, run_preprocessing.py)
├── src/       (preprocessing.py)
├── tests/     (test_preprocessing.py)
└── data/
    ├── raw/        (git-ignored)
    └── processed/  (clean_corpus.parquet)
```