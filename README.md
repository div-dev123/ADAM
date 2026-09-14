# ADAM

ADAM (Adaptive Dynamic AI Memory) is a lightweight research prototype for memory management in LLM-based conversational systems.

The repository currently implements the basic semantic memory system and **Phase 2: importance scoring plus memory lifecycle management**.

```text
Selected user information
    -> embedding
    -> intrinsic importance score
    -> initial lifecycle placement
    -> SQLite storage

Query
    -> embedding
    -> cosine retrieval
    -> top-k memories
```

## Core distinction

ADAM keeps two concepts separate:

```text
importance_score = estimated intrinsic value of the memory (0-1)
tier             = current storage/lifecycle state
```

Importance is a property of the information. Tier is a mutable state that can
change later because of age, usage, relevance, or compression. Moving a memory
between tiers does not change its importance score unless a later phase
explicitly recalculates it.

## Implemented behavior

- Lightweight SentenceTransformer embeddings.
- Local SQLite persistence at `data/adam.db`.
- User-scoped semantic retrieval using cosine similarity only.
- Transparent heuristic importance scoring in the range 0-1.
- Initial placement into `WORKING`, `SHORT_TERM`, or `ARCHIVE`.
- Configurable lifecycle transitions to `LONG_TERM` and `ARCHIVE`.
- Lifecycle metadata: `compression_level`, `last_accessed`, `access_count`,
  `created_at`, and `updated_at`.
- FastAPI endpoints for storing and retrieving memories.

ADAM memories represent selected information extracted from conversations. They
are not intended to be a copy of every raw conversation message. Conversation
history and ADAM memory should remain separate abstractions in future phases.

## Current lifecycle policy

Initial placement may use importance, but it is not a permanent mapping:

- Low-value memories can enter `ARCHIVE` directly.
- Medium-value memories enter `SHORT_TERM`.
- High-value memories enter `WORKING` so they are active when newly created.

Later transitions use lifecycle metadata independently of importance:

- An older, accessed `WORKING` memory can move to `LONG_TERM`.
- An older or compressed `SHORT_TERM` memory can move to `ARCHIVE`.
- `LONG_TERM` and `ARCHIVE` remain stable until a future policy changes them.

The policy is implemented in `app/memory/tiers.py` and is configurable for
experiments. No forgetting, compression operation, or automatic lifecycle pass
is implemented yet; `compression_level` is stored as metadata for the next
phases.

## Requirements

- Python 3.10 or newer
- macOS on Apple Silicon, such as an M2 MacBook Air with 8 GB unified memory
- No MongoDB, Redis, Docker, or external database

Ollama is not required for the current phase. LLM-assisted consolidation is
deferred to a later phase.

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

The project uses FastAPI and Uvicorn for the API, SentenceTransformers with
`all-MiniLM-L6-v2` for embeddings, NumPy for numerical operations, and Pytest
with HTTPX for tests.

The first real embedding call may download and cache `all-MiniLM-L6-v2`. It
produces 384-dimensional vectors and is appropriate for the target laptop.

Lifecycle policy settings can be overridden without changing code:

```bash
export INITIAL_ARCHIVE_THRESHOLD='0.20'
export INITIAL_LONG_TERM_THRESHOLD='0.70'
export WORKING_TO_LONG_TERM_DAYS='7'
export WORKING_TO_LONG_TERM_MIN_ACCESSES='1'
export SHORT_TERM_TO_ARCHIVE_DAYS='30'
export ARCHIVE_COMPRESSION_LEVEL='1'
```

## Run tests

```bash
source .venv/bin/activate
USE_TF=0 python -m pytest -q
```

`USE_TF=0` prevents Transformers from probing an incompatible TensorFlow/Keras
installation. ADAM uses PyTorch through SentenceTransformers and does not need
TensorFlow.

Tests use deterministic embedding doubles, so they do not require a downloaded
model or external services.

## Run the API

```bash
source .venv/bin/activate
USE_TF=0 python -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000` and interactive documentation
is available at `http://127.0.0.1:8000/docs`.

### Store a memory

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"My goal is to pass the AWS certification."}'
```

The response includes `importance_score`, `tier`, `compression_level`,
`created_at`, `updated_at`, `last_accessed`, and `access_count`.

### Retrieve memories

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","query":"What is my goal?","top_k":5}'
```

Retrieval ranks only by cosine similarity. Importance, tier, recency, and
access count are intentionally not ranking signals yet.

## SQLite storage

The database is created automatically at:

```text
data/adam.db
```

The `memories` table stores the content, embedding, user ID, timestamps, access
metadata, importance score, tier, and compression level. Embeddings are stored
as JSON text because the representation is readable, works without SQLite
extensions, and can later be migrated to a vector database.

Existing Phase 1/2 databases are migrated automatically with defaults for new
lifecycle columns. The database is ignored by Git through `data/*.db`.

## Project structure

```text
ADAM/
├── app/
│   ├── main.py                    # FastAPI endpoints
│   ├── config.py                  # Database, scoring, and lifecycle settings
│   ├── memory/
│   │   ├── models.py              # Memory model and SQLite row conversion
│   │   ├── storage.py             # SQLite schema and persistence
│   │   ├── importance.py          # Replaceable heuristic scorer
│   │   └── tiers.py               # Independent lifecycle policy
│   ├── retrieval/
│       ├── embeddings.py          # Replaceable embedding interface
│       ├── retrieval.py            # Write and retrieval orchestration
│       └── similarity.py           # Cosine similarity
├── data/                          # Local runtime data; database is ignored
├── tests/
│   └── test_phase1.py             # Phase 1 and lifecycle tests
├── requirements.txt
├── README.md
└── .gitignore
```

## Importance scoring

`HeuristicImportanceScorer` is a transparent research baseline, not the final
ADAM scoring mechanism. It combines persistent-language markers, content
length, access recurrence, and recency. Its weights are configured in
`app/config.py` and can be replaced with another scorer without changing the
storage interface.

## Limitations and roadmap

- SQLite retrieval scans a user's memories in Python, so this is a prototype for
  modest datasets.
- The lifecycle policy currently exposes transitions but does not run an
  automatic background lifecycle pass.
- Compression metadata exists, but compression itself is not implemented.
- There is no LLM extraction, consolidation, forgetting, query drift, adaptive
  scope, multi-signal retrieval, or final response generation.
- Raw conversation history is not stored or managed by this memory layer.

The next logical phase is **LLM-assisted consolidation**, which will be added
behind a replaceable client interface after the lifecycle separation is stable.
