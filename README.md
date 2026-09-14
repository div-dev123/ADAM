# ADAM: Adaptive Memory Management Framework

ADAM (Adaptive Dynamic AI Memory) is a lightweight research prototype for memory management in LLM-based conversational systems. It decides what information should be stored, how it should be maintained, when it should be forgotten or archived, and which memories should be retrieved for a new query.

The system includes a **full interactive research web interface** to visually demonstrate and inspect memory extraction, importance scoring, tier transitions, consolidation, and retrieval.

---

## Architecture & Pipelines

ADAM consists of two main pipelines:

### 1. Memory Write Path
```text
New Interaction / Dialogue
        ↓
Heuristic Importance Scoring [0.0 - 1.0]
        ↓
Initial Lifecycle Tier Placement [WORKING, SHORT_TERM, ARCHIVE]
        ↓
Bounded Candidate Search (Cosine Similarity ≥ 0.35)
        ↓
LLM Consolidation Decision
        ├── NEW           → Insert fresh memory
        ├── DUPLICATE     → Increment access count & update timestamp
        ├── RELATED       → Merge facts, regenerate embedding & score
        └── CONTRADICTORY → Overwrite active content, record immutable audit log
        ↓
Storage & Lifecycle Compression (Level 1 / Level 2)
```

### 2. Memory Read Path
```text
User Query
        ↓
SentenceTransformer Embedding (all-MiniLM-L6-v2)
        ↓
Semantic Memory Retrieval (Cosine Similarity)
        ↓
Top-K Memory Ingestion & Context Assembly
        ↓
Local Ollama LLM (qwen2.5:3b)
        ↓
Final Response & Response Memory Evaluation
```

---

## Core Concept: Decoupled Importance vs. Tier

In ADAM, **intrinsic value** and **lifecycle state** are completely decoupled:
- **`importance_score` (0.0 to 1.0)**: Represents the factual significance of the information. Calculated explicitly via Python heuristics (persistence markers, content length, recurrence, and recency decay) so that scoring is deterministic, explainable, and reproducible without stochastic LLM overhead.
- **`tier` (`WORKING`, `SHORT_TERM`, `LONG_TERM`, `ARCHIVE`)**: Represents the operational storage state. Tiers transition according to aging, access counts, and lifecycle compression (`WORKING` ➔ `LONG_TERM` at compression level 1; `SHORT_TERM` ➔ `ARCHIVE` at compression level 2).

---

## Research Web Interface (5 Core Views)

The web interface is served directly from the FastAPI backend at `http://127.0.0.1:8000`.

### 1. Chat & Live Pipeline Inspector
- **Interactive Dialogue**: Connects to the local Ollama LLM with memory injection.
- **Visual Badges**: Attached to **both** user and assistant messages:
  - **Importance Score**: e.g., `⭐ Importance: 0.85 (High)`
  - **Memory Tier**: e.g., `🗄️ WORKING` or `🗄️ SHORT-TERM`
  - **Consolidation Action**: e.g., `⚖️ NEW`, `DUPLICATE`, `RELATED`, `CONTRADICTORY`
  - **Compression Level**: e.g., `📦 L1 Compressed`
- **Expandable ADAM Details**: Click "Details" or "Retrieved Context" on any message to inspect candidate memories, cosine similarity scores, classification reasoning, and text evolution.
- **Live Pipeline Step Indicator**: Visual progress banner showing `[1. Extraction] ➔ [2. Consolidation] ➔ [3. Semantic Retrieval] ➔ [4. Context & LLM] ➔ [5. Response Memory]`.

### 2. Memory Dashboard (4 Tiers)
- **Kanban Columns**: Displays active memories grouped across the four tiers:
  - 🔵 **WORKING**: Active, high-salience facts (Initial Score ≥ 0.70).
  - 🟢 **SHORT-TERM**: Transitory conversational context (Score: 0.20 – 0.70).
  - 🟣 **LONG-TERM**: Persistent memory aged past 7 days with ≥ 1 access (Compressed L1).
  - 🟠 **ARCHIVE**: Low priority / expired memory (Compressed L2).
- **Search & Filters**: Real-time content search, tier filtering, and importance level filtering.
- **Card Actions**: Inspect history, trigger manual tier transitions, or delete memories.

### 3. Memory Lifecycle & State Machine Explorer
- **Interactive State Diagram**: Visualizes transition rules between `NEW`, `WORKING`, `SHORT_TERM`, `LONG_TERM`, and `ARCHIVE`.
- **Memory History Tracer**: Select any stored memory to view its immutable audit timeline, operation reasons, and before/after text diffs.

### 4. Consolidation Audit Feed
- **Global Event Log**: Real-time log of all consolidation events (`NEW`, `DUPLICATE`, `RELATED`, `CONTRADICTORY`, `COMPRESSED`).
- **Diff Inspection**: Shows prior content vs. updated content side-by-side with LLM classification explanations.

### 5. System & Research Metrics Panel
- **Aggregate Statistics**: Total memories, average importance score, total consolidations, compressed memory counts, and tier distribution progress bars.
- **Environment Status**: Live Ollama connection health monitor, active model name, SentenceTransformer configuration, and SQLite path.
- **Database Reset**: Safety modal to clear all research data with explicit confirmation.

---

## Quickstart & Setup

### 1. Environment Setup
```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start Ollama (Local LLM)
In a separate terminal:
```bash
# Start local Ollama server
ollama serve

# Pull the lightweight 3B model (optimized for MacBook M2 8GB)
ollama pull qwen2.5:3b
```
*(Note: If Ollama is offline or not installed, ADAM falls back gracefully, and all memory scoring, storage, and retrieval continue to function.)*

### 3. Run the Application
```bash
source .venv/bin/activate
USE_TF=0 uvicorn app.main:app --reload --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## Running the Automated Test Suite

```bash
source .venv/bin/activate
USE_TF=0 python -m pytest -v
```

All 19 test cases validate:
- SQLite persistence and row schemas
- Heuristic importance scoring boundaries [0, 1]
- Independent lifecycle policy and tier transitions
- Semantic retrieval and cosine ranking
- LLM consolidation actions (`NEW`, `DUPLICATE`, `RELATED`, `CONTRADICTORY`)
- Lifecycle compression transitions and history auditing
- `/chat`, `/metrics`, `/memories`, `/history`, `/system/status`, and `/reset` API endpoints
- Static web interface asset serving

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Research Web Interface SPA |
| `GET` | `/health` | Healthcheck returning phase version |
| `GET` | `/system/status` | Runtime status of Ollama, models, and SQLite |
| `POST` | `/chat` | Executes conversational turn with memory write, retrieval, LLM response, and pipeline trace |
| `POST` | `/memory` | Stores/consolidates a memory directly |
| `POST` | `/retrieve` | Semantic similarity memory retrieval for a query |
| `GET` | `/memories` | List memories with optional `user_id`, `tier`, and `search` query parameters |
| `GET` | `/memory/{id}` | Retrieve a single memory by ID |
| `GET` | `/memory/{id}/history` | Get audit history log for a memory |
| `POST` | `/memory/{id}/transition` | Explicitly transition a memory to a target tier using compression |
| `DELETE` | `/memory/{id}` | Delete a memory and its audit history |
| `GET` | `/history` | List recent global consolidation and compression audit events |
| `GET` | `/metrics` | Calculate live research metrics (counts, tier distribution, averages) |
| `POST` | `/reset` | Clear the research database with `{"confirm": true}` |

---

## Project Structure

```text
ADAM/
├── app/
│   ├── main.py                     # FastAPI entry point, API routes & static file mounting
│   ├── config.py                   # App settings, thresholds, and weights
│   ├── llm/
│   │   └── client.py               # LLMClient base, OllamaClient, structured schemas
│   ├── memory/
│   │   ├── models.py               # Memory data model and SQLite conversions
│   │   ├── storage.py              # SQLite storage, metrics calculation, and audit history
│   │   ├── importance.py           # Heuristic importance scorer
│   │   ├── tiers.py                # Lifecycle policy and tier rules
│   │   ├── consolidation.py        # Candidate search & LLM consolidation write path
│   │   └── compression.py          # Lifecycle compression transitions (L1 & L2)
│   ├── retrieval/
│   │   ├── embeddings.py           # SentenceTransformer (all-MiniLM-L6-v2) embedding service
│   │   ├── similarity.py           # Cosine similarity calculation
│   │   └── retrieval.py            # Retrieval & full conversational turn orchestration
│   └── static/                     # Research Web Interface SPA
│       ├── index.html              # 5-view research UI layout
│       ├── css/
│       │   └── styles.css          # Dark-mode styling, glassmorphism & tier color accents
│       └── js/
│           └── app.js              # State manager, live pipeline animator & API client
├── data/                           # Local SQLite database directory (data/adam.db)
├── tests/
│   ├── test_phase1.py              # Core memory, scoring, consolidation & compression unit tests
│   └── test_web_api.py             # Full web API, chat trace, metrics & static assets tests
├── requirements.txt
└── README.md
```
