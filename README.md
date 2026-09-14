# ADAM

ADAM (Adaptive Dynamic AI Memory) is a lightweight research prototype for memory management in LLM-based conversational systems.

The repository implements the basic semantic memory system, Phase 2 importance and lifecycle management, and **Phase 3: LLM-assisted consolidation and compression**.

```text
New memory
  -> embedding
  -> importance score
  -> initial lifecycle tier
  -> bounded semantic candidates
  -> structured LLM decision
  -> SQLite create/update

Query
  -> embedding
  -> cosine retrieval
  -> top-k memories
```

## Importance and tier are separate

```text
importance_score = intrinsic value of the information (0-1)
tier             = current storage/lifecycle state
```

Importance is recalculated only when the content is changed by a related or
contradictory consolidation. Compression and ordinary tier transitions preserve
importance. Tier is allowed to change because of age, usage, relevance, or
compression.

ADAM memories are selected information extracted from conversations. They are
not a copy of every raw conversation message; raw conversation history remains
a separate future abstraction.

## Phase 3 behavior

New memories are classified against only a small set of semantically similar
candidates. The entire database is never sent to the LLM.

- `NEW`: create a new memory.
- `DUPLICATE`: do not create a duplicate; update access metadata.
- `RELATED`: merge useful information, regenerate the embedding, and recalculate importance.
- `CONTRADICTORY`: replace active content with current information and preserve old/new content in history.

Compression is explicit and never runs during normal creation:

```text
WORKING -> LONG_TERM  (compression level 1)
SHORT_TERM -> ARCHIVE (compression level 2)
```

Related, contradictory, duplicate, and compression operations are recorded in
SQLite history for research auditing.

## Requirements

- Python 3.10 or newer
- macOS on Apple Silicon, such as an M2 MacBook Air with 8 GB unified memory
- SQLite, included with Python
- Ollama with `qwen2.5:3b` for the default Phase 3 API path

No MongoDB, Redis, Docker, or external database is used. Query retrieval still
uses cosine similarity only; query drift, adaptive retrieval, and multi-signal
ranking are not implemented.

## Setup

### 1. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the interpreter.

The first embedding call may download and cache `all-MiniLM-L6-v2`, a lightweight
384-dimensional SentenceTransformer model.

### 2. Configure Ollama

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:3b
```

The model is deliberately small for an M2 MacBook Air with 8 GB unified memory.
Tests use a fake LLM and do not require Ollama to run.

Optional provider and consolidation settings:

```bash
export OLLAMA_HOST='http://127.0.0.1:11434'
export OLLAMA_MODEL='qwen2.5:3b'
export CONSOLIDATION_CANDIDATE_LIMIT='3'
export CONSOLIDATION_MIN_SIMILARITY='0.35'
export WORKING_COMPRESSION_LEVEL='1'
export ARCHIVE_COMPRESSION_LEVEL_TARGET='2'
```

Lifecycle settings are also configurable:

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
installation. Tests use deterministic embedding and LLM doubles, so no model
server is needed.

## Run the API

```bash
source .venv/bin/activate
USE_TF=0 python -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is
at `http://127.0.0.1:8000/docs`.

### Store a memory

```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"My goal is to pass the AWS certification."}'
```

The default API path uses Ollama to classify the new memory before persisting it.

### Retrieve memories

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","query":"What is my goal?","top_k":5}'
```

Retrieval ranks only by cosine similarity. Importance, tier, recency, and access
count are intentionally not ranking signals yet.

## SQLite storage

The database is created automatically at `data/adam.db`. The `memories` table
stores content, embedding JSON, user ID, timestamps, access metadata,
`importance_score`, `tier`, and `compression_level`.

The `memory_history` table records consolidation and compression operations with
old content, new content, operation type, reason, and timestamp. Embeddings are
stored as JSON text because it is transparent, works without SQLite extensions,
and can later be migrated to a vector database.

The SQLite database is ignored by Git through `data/*.db`.

## Project structure

```text
ADAM/
├── app/
│   ├── main.py                    # FastAPI endpoints and default wiring
│   ├── config.py                  # Model, threshold, and lifecycle settings
│   ├── llm/
│   │   └── client.py              # LLMClient, OllamaClient, validated results
│   ├── memory/
│   │   ├── models.py              # Memory model and SQLite row conversion
│   │   ├── storage.py             # SQLite persistence and history
│   │   ├── importance.py          # Replaceable heuristic scorer
│   │   ├── tiers.py               # Independent lifecycle policy
│   │   ├── consolidation.py       # Candidate classification and updates
│   │   └── compression.py         # Explicit compressed transitions
│   └── retrieval/
│       ├── embeddings.py          # Replaceable embedding interface
│       ├── retrieval.py           # Write/retrieval orchestration
│       └── similarity.py           # Cosine similarity
├── data/                          # Local runtime data; database is ignored
├── tests/
│   └── test_phase1.py             # Phase 1-3 tests
├── requirements.txt
├── README.md
└── .gitignore
```

## Structured LLM contract

`LLMClient` is the provider abstraction. `OllamaClient` is the current local
experimental provider and sends JSON-mode prompts to `qwen2.5:3b`. Pydantic
validates both `ConsolidationDecision` and `CompressionResult`; arbitrary prose
is rejected instead of being parsed with string heuristics.

A stronger local or Kaggle-backed provider can implement the same interface
without changing ADAM's consolidation, compression, storage, or retrieval
logic.

## Limitations and roadmap

This remains a research prototype. SQLite retrieval scans one user's memories
in Python, there is no authentication or background scheduler, and the current
Ollama provider is the only real LLM provider. Forgetting, query drift, adaptive
scope, multi-signal ranking, context assembly, and final response generation
remain future phases.

The next logical phase is selective forgetting and archiving after the current
consolidation/compression behavior is experimentally evaluated.
