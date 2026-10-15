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
git clone https://github.com/dareen-abualhaj/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/verify_env.py --allow-cpu   # drop --allow-cpu on a CUDA machine
python scripts/download_data.py            # -> data/raw/
pytest -v                                  # run unit tests
python -m scripts.run_preprocessing        # -> data/processed/clean_corpus.parquet

# Week 2
python -m scripts.build_index              # -> chroma_db/
python -m scripts.benchmark_retrieval      # -> docs/retrieval_benchmark.json
```

## Project structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── docs/      (colab_cuda_check.png, retrieval_benchmark.json)
├── scripts/   (verify_env.py, download_data.py, run_preprocessing.py,
│               build_index.py, benchmark_retrieval.py)
├── src/       (preprocessing.py, chunking.py, embeddings.py, vector_store.py)
├── tests/     (test_preprocessing.py, test_chunking.py)
├── chroma_db/ (git-ignored, rebuilt by build_index)
└── data/
    ├── raw/        (git-ignored)
    └── processed/  (clean_corpus.parquet)
```

---

# Week 2: Chunking, Embeddings & Vector DB

## What I built
- `src/chunking.py`: recursive chunking that keeps separators, with unit tests in `tests/`
- `src/embeddings.py`: sentence-transformers embeddings, with embedding time logged
- `src/vector_store.py`: ChromaDB wrapper (ingest + semantic search)
- `scripts/build_index.py`: chunks the clean corpus, embeds it, and populates ChromaDB
- `scripts/benchmark_retrieval.py`: measures retrieval latency per query -> `docs/retrieval_benchmark.json`

## Approach
- **Chunking:** recursive splitting (paragraph -> sentence -> word) with `chunk_size=<N>` and `overlap=<M>`. Separators are kept so chunks stay readable and no text is lost. <one line on why you chose these values>
- **Embeddings:** `<model name>` (<dim> dimensions), run on <CPU/GPU>, batch size <B>.
- **Vector store:** ChromaDB persistent client in `chroma_db/`, distance metric `<cosine/l2>`.
- **Edge cases handled:** <e.g. empty or very short documents skipped, existing collection reused/reset on rebuild, duplicate IDs avoided, k larger than the collection size>.

## Retrieval benchmark
10 queries, top-5 (full results in `docs/retrieval_benchmark.json`):

| Metric | Value |
|---|---|
| Mean total latency (embed + search) | 28.49 ms |
| Median total latency | 23.92 ms |
| Max total latency | 52.18 ms |
| Mean search latency | 12.56 ms |

The top-1 result was clearly relevant for 8 of 10 queries (e.g. "Who was the first president of the United States?" -> *George Washington*). Two were only loosely relevant: "Which animals live in the ocean?" -> *Ray* and "Famous painters of the Renaissance" -> *Florence*. The corpus is only 6,000 articles, so some topics have no dedicated article.

## Rebuilding the vector database
`chroma_db/` is git-ignored because of its size. The deliverable is the populated database, which is rebuilt with:

```bash
python -m scripts.build_index
```
