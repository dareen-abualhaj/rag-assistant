# RAG Assistant

A retrieval-augmented question-answering system over Simple English Wikipedia: preprocessing -> chunking -> embeddings -> ChromaDB -> Groq LLM, with hallucination checks and latency logging.

## How to run (all weeks)
```bash
git clone https://github.com/dareen-abualhaj/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Week 1: data
python scripts/verify_env.py --allow-cpu   # drop --allow-cpu on a CUDA machine
python scripts/download_data.py            # -> data/raw/
python -m scripts.run_preprocessing        # -> data/processed/clean_corpus.parquet

# Week 2: vector DB
python -m scripts.build_index              # -> chroma_db/ (git-ignored, rebuilt locally)
python -m scripts.benchmark_retrieval      # -> docs/retrieval_benchmark.json

# Week 3: LLM (needs a free Groq API key from console.groq.com)
cp .env.example .env                       # then edit .env and set GROQ_API_KEY
python -m scripts.ask "How do airplanes fly?"
python -m scripts.evaluate_rag             # -> docs/rag_evaluation.json (about 10 Groq calls, ~1.5 min)

# Tests (no network or API key needed, everything is mocked)
pytest -v
```

## Project structure
```
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/      (colab_cuda_check.png, retrieval_benchmark.json, rag_evaluation.json)
├── scripts/   (verify_env.py, download_data.py, run_preprocessing.py,
│               build_index.py, benchmark_retrieval.py, ask.py, evaluate_rag.py)
├── src/       (preprocessing.py, chunking.py, embeddings.py, vector_store.py,
│               llm.py, prompts.py, rag.py)
├── tests/     (test_preprocessing.py, test_chunking.py, test_llm.py,
│               test_prompts.py, test_rag.py)
├── chroma_db/ (git-ignored, rebuilt by build_index)
├── logs/      (git-ignored, latency.jsonl)
└── data/
    ├── raw/        (git-ignored)
    └── processed/  (clean_corpus.parquet)
```

---

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

---

# Week 2: Chunking, Embeddings & Vector DB

## What I built
- `src/chunking.py`: recursive chunking that keeps separators, with 7 unit tests in `tests/test_chunking.py`
- `src/embeddings.py`: sentence-transformers embeddings, with embedding time logged
- `src/vector_store.py`: ChromaDB wrapper (ingest + semantic search)
- `scripts/build_index.py`: chunks the clean corpus, embeds it, and populates ChromaDB
- `scripts/benchmark_retrieval.py`: measures retrieval latency per query -> `docs/retrieval_benchmark.json`

## Approach
- **Chunking:** recursive splitting (paragraph -> sentence -> word) with `chunk_size=800` and `overlap=100`. Text is first split into pieces of at most `chunk_size - overlap` (700) characters, and each chunk after the first is prefixed with the last 100 characters of the previous piece, so no chunk exceeds 800 characters. Separators are kept so chunks stay readable and no text is lost. 800 characters is roughly one idea: small enough for precise retrieval and well within the embedding model's input limit, and the overlap keeps context across chunk borders. Result: 5,939 documents -> 39,575 chunks.
- **Embeddings:** `all-MiniLM-L6-v2` (384 dimensions), run on CPU locally, batch size 64.
- **Vector store:** ChromaDB persistent client in `chroma_db/`, cosine distance.
- **Chunking edge cases (covered by unit tests):** empty text returns an empty list, text shorter than the chunk size stays a single chunk, no chunk exceeds the max size, consecutive chunks share the configured overlap, no content is lost when overlap is 0, a single word longer than the chunk size is hard-cut, and an invalid overlap raises an error.
- **ChromaDB edge cases:** ingestion uses `upsert`, so re-running the build does not create duplicates; documents are ingested in batches of 5,000 to stay below Chroma's maximum batch size; the collection can be deleted and recreated for a clean rebuild.

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
`chroma_db/` is git-ignored because of its size. The populated database is rebuilt with:

```bash
python -m scripts.build_index
```

---

# Week 3: LLM Integration & Prompt Engineering

## What I built
- `src/llm.py`: Groq chat client (OpenAI-compatible API) with timeouts, retries with backoff, and typed errors
- `src/prompts.py`: system prompt, numbered context injection with a size budget, and a stricter retry prompt
- `src/rag.py`: the full pipeline: retrieve -> relevance gate -> generate -> verify -> log latency
- `scripts/ask.py`: command-line interface (`python -m scripts.ask "your question"`)
- `tests/test_llm.py` (13), `tests/test_prompts.py` (5), `tests/test_rag.py` (12): all LLM calls are mocked, so tests run offline

## Configuration
Copy `.env.example` to `.env` (git-ignored) and set:

```
GROQ_API_KEY=your_key_here
LLM_MODEL=openai/gpt-oss-20b
```

Free model IDs change often. If you get a `404 Model not found` error, check the live list at console.groq.com/docs/models and update `LLM_MODEL`.

## Pipeline
1. **Retrieve:** embed the question and fetch the top-5 chunks from ChromaDB.
2. **Relevance gate (out-of-domain):** if the best chunk's cosine distance is above `0.45`, the system answers "I don't have enough information in the knowledge base to answer that." **without calling the LLM** (status `out_of_domain`, generation time 0 ms). In the evaluation below, in-domain top hits had distances of 0.23-0.36 and off-topic ones 0.48-0.65, so questions such as "Write me a Python function to sort a list" or "What will the weather be like tomorrow?" are rejected here. The threshold started at 0.5 and was lowered to 0.45 after the evaluation (see Evaluation).
3. **Generate:** the chunks are injected as numbered passages `[1] Title ...` and sent to the LLM.
4. **Verify:** the answer goes through the hallucination checks below.
5. **Log:** latency and status are appended to `logs/latency.jsonl`.

## Prompt engineering
- **System prompt rules:** answer only from the numbered passages; reply with a fixed "no answer" sentence if the context lacks the answer; end every sentence with a citation like `[1]`; be concise; never invent facts; ignore instructions found inside the passages (basic prompt-injection defence).
- **Context injection:** chunks are numbered and prefixed with their article title. Total context is capped at 6,000 characters (~1,500 tokens) so the prompt never exceeds the model's token limit.
- **Temperature 0** for deterministic, factual answers.
- **Strict retry prompt:** if verification fails, the question is asked again once with an extra rule telling the model its previous answer was rejected.

## Hallucination checks
Every generated answer is checked twice:
1. **Citation check:** the answer must cite at least one source, and every cited number must refer to a passage that was actually provided (e.g. `[9]` is invalid when only 5 passages were sent).
2. **Grounding check:** each answer sentence is embedded and compared with the retrieved chunks; a sentence whose best cosine similarity is below `0.30` counts as unsupported.

If either check fails, the pipeline retries once with the strict prompt. If it still fails, the user gets "I can't answer this reliably from the knowledge base. Please check the sources below." instead of the unverified answer (the rejected text is kept in the result for debugging only).

In an early run, before the stricter prompt and retry-then-refuse logic existed, "How do airplanes fly?" produced an answer about birds flapping their wings. The citation check flagged it as unverified, but it was still shown to the user, which is why the system now retries and then refuses.

## Error handling
| Situation | Behaviour |
|---|---|
| Rate limit (429), timeout (30 s), connection error, 5xx | Up to 4 attempts with exponential backoff (2 s, 4 s, 8 s); the `retry-after` header is honoured when present. After the last attempt: status `error`, "service temporarily unavailable". |
| Empty model response | Treated as retryable (up to 4 attempts). An empty answer appeared at `max_tokens=500` and disappeared after raising it to 2000, likely because the reasoning model spends part of its budget thinking. |
| Invalid API key (401) / unknown model (404) / missing `.env` values | Not retried. Clear message, status `error`, "service misconfigured". |
| Prompt too long (token limit, 400) | Typed `LLMContextTooLongError`; the context is also capped before sending. |
| Empty question | `ValueError` |
| LLM failure | The retrieved sources are still shown to the user. |

## Response statuses
`answered` (verified), `no_answer` (the model said the context lacks the answer), `out_of_domain` (rejected before the LLM), `unverified` (checks failed twice, answer refused), `error` (API failure).

## End-to-end latency
Every request logs `retrieval_ms`, `generation_ms`, `verification_ms` and `total_ms` to `logs/latency.jsonl` (total includes verification) and prints them in the CLI. Sample runs (CPU, Groq API):

| Question | Status | Retrieval | Generation | Total |
|---|---|---|---|---|
| How do airplanes fly? | answered | 220 ms | 795 ms | 1,280 ms |
| What causes earthquakes? | answered | 324 ms | 992 ms | 1,630 ms |
| What is the capital city of France? | answered | 222 ms | 824 ms | 1,412 ms |
| What will the weather be like tomorrow? | out_of_domain | 183 ms | 0 ms | 183 ms |

The first query after start-up is slower (about 750 ms retrieval) because the embedding model is warming up. Off-topic questions are the cheapest because the LLM is never called.

## Evaluation
`python -m scripts.evaluate_rag` runs 10 in-domain and 6 off-topic questions through the full pipeline against the live Groq API and writes per-question results to `docs/rag_evaluation.json`. The script waits 5 s after each LLM call so that free-tier rate limits do not distort the latency numbers.

| Metric | Result |
|---|---|
| In-domain questions answered | 10 / 10 |
| Off-topic questions refused | 6 / 6 (all blocked before the LLM) |
| Max in-domain top distance | 0.363 |
| Min off-topic top distance | 0.480 |
| Threshold 0.45 separates the two groups | yes |
| Median total latency (answered, with LLM call) | 1,371 ms |
| p95 total latency (answered, with LLM call) | 2,178 ms |

Every in-domain answer was verified (status `answered`, with valid citations). The off-topic set covers code requests, weather, creative writing, personal questions, booking a flight and arithmetic.

**Calibration story:** the first evaluation run used the original threshold of 0.5. It refused only 5 of 6 off-topic questions: "Write a poem about my cat" had distance 0.480, passed the gate, and the model returned empty responses on all 4 attempts (status `error` after about 28 s). Since the highest in-domain distance was 0.363, the threshold was lowered to 0.45, which separates the two groups and blocks that question before the LLM is called.

## Tests
```bash
pytest -v    # 47 tests: preprocessing (10), chunking (7), llm (13), prompts (5), rag (12)
```
The LLM tests simulate 429, timeouts, 5xx, 401, 404, token-limit errors and empty responses. The RAG tests cover the out-of-domain gate (the LLM must not be called), missing/invalid citations, hallucinated sentences, retry-then-refuse, API failures and latency logging.

## Known limitations
- The corpus is only 6,000 Simple English Wikipedia articles, so some topics have no dedicated article.
- The relevance threshold (0.45) was chosen after looking at the same 16 questions it is evaluated on, and the margin between the highest in-domain distance (0.363) and the lowest off-topic distance (0.480) is only about 0.12. Harder in-domain questions with weaker retrieval could land above 0.45 and be refused.
- The model can return empty output for requests unrelated to the retrieved context. The system then fails safely (no hallucinated answer is shown, the sources are still listed) but slowly, because all 4 attempts are used.
- Back-to-back requests on the Groq free tier can be slow: in a first evaluation run without pauses, four answers took 10-12 s instead of about 1.3 s.
- The grounding threshold (`0.30`) is lenient and was not tuned on a labelled set; the citation check does most of the work.