# ADAM

ADAM is a lightweight research prototype for memory management in LLM-based conversational systems.

The repository currently implements **Phase 1 only**:

```text
Text message -> SentenceTransformer embedding -> Memory storage -> Semantic retrieval
```

Later phases will add importance scoring, memory tiers, consolidation, forgetting, query drift, multi-signal ranking, and context assembly. They are intentionally not implemented yet.

## Phase 1 capabilities

- Store a text memory for a user.
- Generate a local embedding with `all-MiniLM-L6-v2`.
- Persist memories in MongoDB Atlas when `MONGODB_URI` is configured.
- Use an in-memory backend for tests and local development without Atlas.
- Retrieve a user's memories by cosine similarity.
- Expose storage and retrieval through FastAPI.
- Track `memory_id`, `user_id`, `content`, `embedding`, `created_at`, `last_accessed`, and `access_count`.

The LLM and Ollama are not required for Phase 1 retrieval. Ollama is prepared for later conversation-generation phases.

## Requirements

- macOS on Apple Silicon, such as an M2 MacBook Air with 8 GB unified memory
- Python 3.10 or newer
- MongoDB Atlas account for persistent storage
- Ollama installed locally for later phases

No local MongoDB, Redis, Docker, or microservices are used.

## Setup

Run these commands from the repository root.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

VS Code: select `.venv/bin/python` as the project interpreter.

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

The dependencies include FastAPI, Uvicorn, PyMongo, `dnspython` for Atlas SRV URLs, SentenceTransformers, Pytest, pandas, and scikit-learn.

The first real embedding call downloads and caches `all-MiniLM-L6-v2`. This is a lightweight 384-dimensional model suitable for the target MacBook.

### 3. Configure MongoDB Atlas

Create an Atlas cluster and database user, then allow your development IP under Atlas Network Access. Copy the Atlas Python connection string and export it in the terminal:

```bash
export MONGODB_URI='mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority'
```

Optional settings:

```bash
export MONGODB_DATABASE='adam_memory'
export MONGODB_COLLECTION='memories'
export EMBEDDING_MODEL='all-MiniLM-L6-v2'
```

Do not commit credentials. The repository ignores `.env`; for persistent local configuration, create `.env` and load/export its values in your shell.

If `MONGODB_URI` is not set, the API uses the in-memory backend. That is useful for tests, but data disappears when the process stops.

### 4. Set up Ollama for later phases

Ollama is not needed to run Phase 1. If it is not installed:

```bash
brew install ollama
```

Start the server in a separate terminal:

```bash
ollama serve
```

Download the lightweight conversation model selected for this project:

```bash
ollama pull qwen2.5:3b
```

Verify it:

```bash
ollama run qwen2.5:3b "Explain conversational memory in one sentence."
```

`qwen2.5:3b` is selected for the M2/8 GB target. Do not run it simultaneously with unnecessary larger local models.

## Run tests

Activate the environment, then run:

```bash
source .venv/bin/activate
USE_TF=0 python -m pytest -q
```

`USE_TF=0` prevents Transformers from probing an incompatible TensorFlow/Keras installation. ADAM uses PyTorch for SentenceTransformers and does not need TensorFlow.

The tests use fake embeddings and the in-memory backend, so they do not require MongoDB, Hugging Face network access, or Ollama.

## Run the API

```bash
source .venv/bin/activate
USE_TF=0 uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### Store a memory

```bash
curl -X POST http://127.0.0.1:8000/memories \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"The project is called ADAM and focuses on adaptive memory management."}'
```

### Store another memory

```bash
curl -X POST http://127.0.0.1:8000/memories \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","content":"ADAM uses semantic memory retrieval to retrieve relevant information."}'
```

### Search memories

```bash
curl -X POST http://127.0.0.1:8000/memories/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","query":"What is ADAM?","top_k":5}'
```

The response contains ranked memories and their cosine similarity scores. Each returned memory also has updated `last_accessed` and `access_count` values.

## Project structure

```text
ADAM/
├── app/
│   ├── main.py                    # FastAPI application and endpoints
│   ├── config.py                  # Environment-backed settings
│   ├── memory/
│   │   ├── models.py              # Phase 1 Memory dataclass
│   │   └── storage.py             # MongoDB Atlas and test storage
│   └── retrieval/
│       ├── embeddings.py          # SentenceTransformer wrapper
│       └── retrieval.py            # Store and search orchestration
├── tests/
│   └── test_phase1.py             # Storage, retrieval, and API tests
├── requirements.txt
├── .env.example
└── README.md
```

## Design notes

- The storage interface keeps MongoDB-specific code separate from retrieval logic.
- Similarity ranking is calculated in Python for transparent research experiments.
- MongoDB stores embeddings as arrays alongside memory metadata.
- The model is loaded lazily, so importing the API does not immediately load PyTorch.
- Phase 1 has no importance, tier, consolidation, forgetting, drift, ranking-weight, or LLM-response logic.
