# ADAM

ADAM is a research prototype for managing memory in long-running LLM
conversations. The project is being built incrementally. The repository
currently contains **Phase 1 only**:

```text
Memory model -> Storage -> Embeddings -> Semantic retrieval
```

Later phases will add importance scoring, memory tiers, consolidation,
forgetting, query drift, ranking signals, and evaluation.

## Technology

- Python 3.12
- SentenceTransformers with `all-MiniLM-L6-v2` embeddings
- MongoDB Atlas for persistent storage
- FastAPI and Uvicorn for the future API layer
- Ollama with `qwen2.5:3b` for local conversation generation
- Pytest for tests

The current Phase 1 retrieval test uses the local in-memory store, so it does
not require MongoDB credentials or a running database.

## Setup

### 1. Create a virtual environment

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On macOS with Homebrew Python, `python3` may be replaced with
`/opt/homebrew/bin/python3`.

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, PyMongo, SentenceTransformers, pandas,
scikit-learn, and Pytest. The first embedding run downloads
`all-MiniLM-L6-v2` from Hugging Face and caches it locally. It is a lightweight
384-dimensional model suitable for an M2 MacBook Air with 8 GB unified memory.

### 3. Configure MongoDB Atlas

Create a MongoDB Atlas cluster and database user, allow your development IP in
Atlas Network Access, and copy the Python connection string. Then create a
local `.env` file from [`.env.example`](.env.example), replacing the
placeholders.

The current code reads `MONGODB_URI` from the shell environment. Export it
before running code:

```bash
export MONGODB_URI='mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority'
```

If `MONGODB_URI` is not set, ADAM automatically uses the in-memory store. This
is useful for local tests and does not persist data between processes.

### 4. Install and configure Ollama

If Ollama is not installed:

```bash
brew install ollama
```

Start the local Ollama server in a separate terminal:

```bash
ollama serve
```

In another terminal, download the conversation model used by this project:

```bash
ollama pull qwen2.5:3b
```

Verify it:

```bash
ollama list
ollama run qwen2.5:3b "Explain conversational memory in one sentence."
```

`qwen2.5:3b` is intentionally used instead of a larger model because this
project targets an M2 MacBook Air with 8 GB RAM. SentenceTransformers handles
embeddings separately; Ollama is reserved for conversation generation in later
phases.

## Run Phase 1

Run the semantic retrieval test:

```bash
USE_TF=0 python -m pytest -q tests/test_phase1.py
```

`USE_TF=0` prevents Transformers from probing an incompatible TensorFlow/Keras
installation. ADAM uses PyTorch for SentenceTransformers and does not need
TensorFlow.

The test stores two memories, asks an algorithms-related query, and verifies
that the Java preference is retrieved before an unrelated travel memory.

## Project layout

| path | purpose |
|---|---|
| `memory.py` | `Memory` data model and MongoDB document conversion |
| `memory_store.py` | in-memory store and MongoDB Atlas store |
| `embeddings.py` | lazy SentenceTransformer embedding service |
| `retrieval.py` | Phase 1 semantic retrieval entry point |
| `tests/test_phase1.py` | relevant-memory retrieval test |
| `requirements.txt` | Python dependencies |
| `.env.example` | MongoDB environment variable template |

## Phase 1 design decisions

- Memory persistence is behind a small store interface so local tests do not
  require Atlas.
- MongoDB stores the memory content and embedding together.
- Similarity ranking is calculated in Python for transparency during research.
- MongoDB Atlas Vector Search can be evaluated later after the basic behavior
  and metrics are stable.
- No Redis, Docker, microservices, or large local models are used.
