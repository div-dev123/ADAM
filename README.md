# ADAM

ADAM (Adaptive Dynamic AI Memory) is a lightweight research prototype for memory management in LLM-based conversational systems.

This repository currently implements **Phase 1 and Phase 2**:

```text
User text -> embedding -> importance score -> tier assignment -> SQLite storage
Query -> embedding -> cosine retrieval -> top-k memories
```

Phase 2 adds a transparent importance and tier baseline. It does not implement the complete adaptive ADAM lifecycle.

## Phase 1 implements

- Memory creation from user text.
- Lightweight SentenceTransformer embeddings.
- Local SQLite persistence at `data/adam.db`.
- User-scoped semantic retrieval using cosine similarity.
- Configurable top-k retrieval.
- `last_accessed` and `access_count` updates when memories are returned.
- A small FastAPI API for storing and retrieving memories.
- A reproducible heuristic importance score in the range 0-1.
- Configurable `WORKING`, `SHORT_TERM`, `LONG_TERM`, and `ARCHIVE` assignment.
- Persistence and API responses for `importance_score` and `tier`.

Phase 2 does **not** implement consolidation, duplicate detection, contradiction detection, forgetting, query drift, adaptive scope, multi-signal ranking, context compression, LLM extraction, or LLM responses.

## Architecture

```text
POST /memory
    -> EmbeddingService
  -> HeuristicImportanceScorer
  -> TierAssigner
    -> Memory object
    -> SQLiteStorage.save_memory
    -> data/adam.db

POST /retrieve
    -> EmbeddingService
    -> SQLiteStorage.get_memories
    -> cosine similarity
    -> top-k results
    -> access metadata update
```

The storage layer owns SQLite and SQL statements. The importance and tier modules
are independent of storage. The retrieval layer still ranks using semantic
similarity only; importance, tier, and access metadata are not retrieval signals
yet. This separation makes each component easy to replace or ablate in later
experiments.

## Requirements

- Python 3.10 or newer
- macOS on Apple Silicon, such as an M2 MacBook Air with 8 GB unified memory
- No MongoDB, Redis, Docker, or external database

Ollama is not required for Phase 1 because this phase does not call an LLM. It can be installed later for conversation generation.

## Setup

Run commands from the repository root.

### 1. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

In VS Code, select `.venv/bin/python` as the project interpreter.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The project uses:

- FastAPI and Uvicorn for the API.
- SentenceTransformers for embeddings.
- `all-MiniLM-L6-v2` as the default lightweight embedding model.
- NumPy for the embedding library's numerical operations.
- Pytest and HTTPX for tests and API testing.

The first real embedding call may download and cache `all-MiniLM-L6-v2` from Hugging Face. It produces 384-dimensional vectors and is appropriate for development on the target laptop.

### 3. Optional: install Ollama for later phases

Ollama is not needed to run Phase 1. For later LLM phases:

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:3b
```

`qwen2.5:3b` is the selected lightweight conversation model for the M2/8 GB target. Do not add it to the Phase 1 runtime path yet.

## Run tests

```bash
source .venv/bin/activate
USE_TF=0 python -m pytest -q
```

`USE_TF=0` prevents Transformers from probing an incompatible TensorFlow/Keras installation. ADAM uses PyTorch through SentenceTransformers and does not need TensorFlow.

The tests use a deterministic embedding double, so they do not download a model or require an external service.

## Run the API

```bash
source .venv/bin/activate
USE_TF=0 python -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. FastAPI documentation is at `http://127.0.0.1:8000/docs`.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Store memories

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"The project is called ADAM and focuses on adaptive memory management."}'
```

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"ADAM uses semantic memory retrieval to find relevant information."}'
```

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"The weather today is sunny."}'
```

### Retrieve memories

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","query":"What is ADAM?","top_k":2}'
```

The response contains memories ranked by cosine similarity. The returned memories also report updated `last_accessed` and `access_count` values.

## SQLite storage design

The database is created automatically at:

```text
data/adam.db
```

The `memories` table stores the required metadata, `importance_score`, `tier`,
and the embedding as JSON text. JSON was chosen because it is readable, uses
only the Python standard library, works with SQLite without extensions, and can
be converted later to a vector-database representation. Similarity is
calculated in Python for this small research baseline.

The database file is ignored by Git through `data/*.db`. No credentials or external database service are needed.

## Project structure

```text
ADAM/
├── app/
│   ├── main.py                    # FastAPI endpoints
│   ├── config.py                  # Database and model configuration
│   ├── memory/
│   │   ├── models.py              # Memory dataclass and SQLite row conversion
│   │   ├── storage.py             # SQLite schema and persistence functions
│   │   ├── importance.py          # Configurable heuristic scoring
│   │   └── tiers.py               # Configurable tier assignment
│   └── retrieval/
│       ├── embeddings.py          # Replaceable embedding interface
│       └── retrieval.py            # Semantic retrieval and cosine similarity
├── data/                          # Local runtime data; SQLite DB is ignored
├── tests/
│   └── test_phase1.py             # Phase 1 and Phase 2 unit/API tests
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- SQLite retrieval scans a user's memories in Python, so it is intended for a prototype and modest datasets.
- The default model is loaded lazily and requires local model-cache space.
- There is no authentication, conversation/session management, batching, or production deployment configuration.
- Retrieval ranks only by semantic similarity; importance, tier, recency, and access frequency are intentionally excluded from ranking until a later phase.
- The importance score is a baseline heuristic, not the final ADAM scoring mechanism or an LLM-based judgment.

## Phase 2 importance and tiers

`HeuristicImportanceScorer` combines four bounded signals:

- Persistent language markers such as `my goal is`, `i prefer`, and `remember that`.
- Content length, capped at 20 words.
- Existing access count, capped at three accesses.
- Recency using a one-day exponential half-life.

The default weights are defined in `app/config.py` and passed into
`ImportanceWeights`:

```text
persistent 0.55
length     0.20
recurrence 0.15
recency    0.10
```

The default tier thresholds are also configurable in `app/config.py`:

```text
score <= 0.20              ARCHIVE
0.20 < score <= 0.45      WORKING
0.45 < score <= 0.70      SHORT_TERM
score > 0.70              LONG_TERM
```

These rules are intentionally simple and measurable for ablation studies. A
future scorer can implement the same `score(...)` contract without changing
SQLite storage or the API pipeline.

## Roadmap

The next logical phase is **Phase 3: memory consolidation**, beginning with
duplicate and related-memory handling. It should remain separate from the
importance scorer and preserve the Phase 1 similarity-only retrieval baseline.
