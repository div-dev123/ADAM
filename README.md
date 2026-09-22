# ADAM: Adaptive Dynamic AI Memory Framework

ADAM (Adaptive Dynamic AI Memory) is a research-grade memory management framework and interactive runtime designed for LLM-based conversational agents. In long-running conversational and agentic systems, naive context stuffing quickly exhausts context windows, dilutes attention, increases latency and inference costs, and causes catastrophic forgetting or stale information retention.

ADAM addresses this by decoupling **intrinsic factual value** (heuristic importance) from **lifecycle operational state** (memory tiers), while providing an intelligent, auditable write-path consolidation engine, semantic vector retrieval, lossy/lossless lifecycle compression, and an interactive **5-view research web interface**.

---

## Key Highlights

- **Dual-Turn Memory Processing**: Evaluates and consolidates memories from **both** user inputs and assistant responses while maintaining strict **Source-Role Isolation**.
- **Decoupled Architecture**: Importance scoring ($0.0 - 1.0$) is calculated through deterministic, explainable multi-signal heuristics, completely separate from storage tiers (`WORKING`, `SHORT_TERM`, `LONG_TERM`, `ARCHIVE`).
- **Heuristic-First, LLM-Assisted Consolidation**: Resolves incoming information into one of four actions: `NEW`, `DUPLICATE`, `RELATED`, or `CONTRADICTORY`. Fast heuristic bypasses handle high-confidence duplicates and explicit contradictions without LLM latency; ambiguous cases fall back to local LLM classification with merge quality verification.
- **Intelligent Noise & Filler Filtering**: Filters greetings, acknowledgments, small-talk, and LLM boilerplate phrases before they can pollute vector storage.
- **Auditable Lifecycle & Transitions**: Every consolidation, merge, contradiction, and compression event writes an immutable audit record to SQLite with before/after diffs and reasoning.
- **Two-Level Compression Engine**: Compresses aging memories (`WORKING` $\to$ `LONG_TERM` at Level 1; `SHORT_TERM` $\to$ `ARCHIVE` at Level 2) to preserve facts while reducing token overhead.
- **Offline & Graceful Degradation**: Functions standalone with local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and SQLite. If the local Ollama LLM (`qwen2.5:3b`) is offline or busy, the system gracefully falls back to deterministic heuristics.
- **Interactive Research Web Interface**: Built-in 5-view single-page application (SPA) with real-time pipeline visualization, Kanban memory dashboard, state machine explorer, audit feed, and system metrics.

---

## Architecture & Pipelines

ADAM operates across two primary pipelines: a dual-turn **Write Path** and a semantic **Read Path**.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               ADAM WRITE PATH                                    │
│                                                                                  │
│   New Input (User Prompt or Assistant Response)                                  │
│         │                                                                        │
│         ▼                                                                        │
│   [Noise & Filler Filter] ─── (Greeting / Boilerplate) ───► Ignored (No Memory)  │
│         │ (Informative content)                                                  │
│         ▼                                                                        │
│   [Multi-Signal Heuristic Scoring] ──► Importance Score [0.0 - 1.0]              │
│         │                                                                        │
│         ▼                                                                        │
│   [Initial Tier Assignment] ──► WORKING (≥ 0.70) | SHORT_TERM | ARCHIVE (≤ 0.30)│
│         │                                                                        │
│         ▼                                                                        │
│   [Candidate Retrieval] ──► Cosine Similarity (≥ 0.35, same source_role, Top-3) │
│         │                                                                        │
│         ▼                                                                        │
│   [Consolidation Engine]                                                         │
│     ├── Heuristic Duplicate (Sim ≥ 0.95 & Overlap ≥ 85%) ──► Increment Access   │
│     ├── Heuristic Contradiction (Trigger + Topic Overlap) ──► Supersede to ARCH │
│     └── LLM-Assisted Classification (Ollama qwen2.5:3b)                         │
│           ├── RELATED       ──► Merge Facts & Recalculate Embedding              │
│           └── NEW           ──► Insert as Independent Memory                     │
│         │                                                                        │
│         ▼                                                                        │
│   [SQLite Storage & Immutable History Audit]                                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                ADAM READ PATH                                    │
│                                                                                  │
│   User Query                                                                     │
│         │                                                                        │
│         ▼                                                                        │
│   [Embedding Generation] ──► SentenceTransformer (all-MiniLM-L6-v2)              │
│         │                                                                        │
│         ▼                                                                        │
│   [Vector Similarity Search] ──► Cosine Similarity against active memories       │
│         │                                                                        │
│         ▼                                                                        │
│   [Ranking & Access Update] ──► Top-K Selection, Increment access_count, touch  │
│         │                                                                        │
│         ▼                                                                        │
│   [Context Assembly] ──► Formats memories with Tier, Importance, and Sim scores │
│         │                                                                        │
│         ▼                                                                        │
│   [Ollama Generation] ──► Context-injected inference (qwen2.5:3b)                │
│         │                                                                        │
│         ▼                                                                        │
│   Assistant Response ──► Returned to user AND routed to Write Path               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Decoupling Importance vs. Lifecycle Tier

Traditional memory systems conflate how important a piece of information is with how long it should live. In ADAM:

| Dimension | Metric | Determined By | Purpose |
|---|---|---|---|
| **Intrinsic Value** | `importance_score` ($0.0 - 1.0$) | Explainable multi-signal heuristics (intent, specificity, durability, salience, recurrence, recency) | Quantifies factual value, semantic density, and relevance |
| **Operational State** | `tier` (`WORKING`, `SHORT_TERM`, `LONG_TERM`, `ARCHIVE`) | Lifecycle aging rules, access frequencies, and explicit compression events | Governs operational caching, retrieval priority, and storage compaction |

This separation ensures that a core fact (e.g., *"User prefers Rust over Python"*, importance 0.85) remains high-value even when aged into long-term storage or archived if contradicted.

### 2. Four Operational Lifecycle Tiers

```text
                  Initial Score ≥ 0.70
                 ┌────────────────────► WORKING ────────┐
                 │                        │             │ Aged ≥ 7 days +
                 │                        │ Manual      │ ≥ 1 Access (L1 Compress)
                 │                        ▼             ▼
New Memory ──────┼─ Initial Score 0.30 - 0.70 ──► SHORT_TERM ──► LONG_TERM
                 │                        │
                 │                        │ Aged ≥ 30 days or
                 │                        ▼ L2 Compress
                 └────────────────────► ARCHIVE ◄─────── (Superseded by contradiction)
                  Initial Score ≤ 0.30
```

- **`WORKING` (Active Focus)**: High-salience, immediately relevant context (initial score $\ge 0.70$). Stored in uncompressed form.
- **`SHORT_TERM` (Transitory Context)**: Conversational context or intermediate dialogue facts (initial score $0.30 - 0.70$).
- **`LONG_TERM` (Consolidated Knowledge)**: Memories aged past 7 days with at least 1 verified access, subjected to **Level 1 Compression** (concise summary preserving facts).
- **`ARCHIVE` (Historical / Cold Storage)**: Low-priority facts (initial score $\le 0.30$), aged short-term memories ($\ge 30$ days), or superseded memories resulting from contradictions. Subjected to **Level 2 Compression** (compact summary).

---

## Detailed Component Specifications

### 1. Multi-Signal Heuristic Importance Scoring

The importance score is calculated deterministically via a weighted sum of 6 normalized signals:

$$\text{Importance Score} = \min\left(1.0, \max\left(0.0, \sum_{i=1}^{6} w_i \cdot s_i\right)\right)$$

| Signal ($s_i$) | Default Weight ($w_i$) | Description & Extraction Logic |
|---|---|---|
| **`intent`** | `0.35` | Detects user intentions via graduated regex categories: directives ($1.0$), identity/profile ($0.9$), active projects/research ($0.85$), explicit preferences ($0.8$), and team/workflow context ($0.5$). |
| **`specificity`** | `0.25` | Domain-agnostic entity density (acronyms, numbers, quoted terms, capitalized tokens), vocabulary richness (content-to-total word ratio), plus technical terminology bonus. |
| **`durability`** | `0.20` | Architectural/enduring definitions ($0.9$) vs. ephemeral/transient phrases like *"today"*, *"for lunch"*, *"currently raining"* ($0.1$). High intent defaults to $0.85$. |
| **`salience`** | `0.10` | Information density ratio (content words vs. function words), sentence length saturation factor (up to 20 words), and structural punctuation complexity bonus ($+0.15$). |
| **`recurrence`** | `0.05` | Frequency of access: $\min(\text{access\_count} / 3.0, 1.0)$. |
| **`recency`** | `0.05` | Continuous exponential decay based on memory creation timestamp: $0.5^{\Delta t / 86400}$ (24-hour half-life). |

#### Noise & Filler Detection
Before scoring, incoming text is evaluated by `is_filler()`:
- **Greeting Tokens**: `hi`, `hello`, `hey`, `howdy`, `yo`, `good morning`, etc.
- **Acknowledgment Tokens**: `ok`, `cool`, `great`, `thanks`, `understood`, `sounds good`, etc.
- **LLM Boilerplate Phrases**: `as an ai language model`, `how can i help you today?`, `i would be happy to help`, `glad i could help!`, etc.
- If classified as filler, memory creation is bypassed entirely, avoiding vector database bloat.

### 2. Tiered Consolidation Engine

When informative content passes filtering, ADAM compares it against active candidates having cosine similarity $\ge 0.35$ under the same `source_role`:

1. **Source-Role Isolation**: User statements are only consolidated against user memories; assistant responses are stored independently, preventing conversational echo loops.
2. **Heuristic DUPLICATE**: If candidate cosine similarity $\ge 0.95$ and word overlap $\ge 85\%$, ADAM increments the access count and touches the timestamp without creating a duplicate row. If the new statement is richer, it updates the text representation.
3. **Heuristic CONTRADICTORY**: If the new input contains explicit negation/replacement patterns (*"instead of"*, *"no longer"*, *"switched to"*, *"changed to"*, *"stopped using"*) AND shares key topical content words with a candidate:
   - The old memory is marked `superseded_by = new_memory_id`, transitioned to `ARCHIVE`, and tagged with compression level 2.
   - The new memory inherits the operational tier (e.g., `WORKING`).
   - Immutable audit events are recorded for both rows.
4. **LLM Consolidation (`RELATED` vs `NEW`)**: For ambiguous candidates ($0.35 \le \text{similarity} < 0.95$):
   - Invokes local Ollama (`qwen2.5:3b`) with structured JSON schema (`ConsolidationDecision`).
   - If classified as `RELATED`, a **Merge Quality Guard** verifies that the proposed merged text preserves essential keywords from both inputs. If the merge fails verification, it safely falls back to `NEW`.
5. **Fallback Safety**: If Ollama times out or is offline, the consolidation engine safely defaults to `NEW` with zero data loss.

### 3. Lifecycle Compression Engine

ADAM implements two compression levels using the local LLM:
- **Level 1 (Concise Summary)**: Applied when `WORKING` memories age into `LONG_TERM` (e.g., after 7 days with $\ge 1$ access). Preserves all facts while shortening sentence structure.
- **Level 2 (Very Compact Archival)**: Applied when `SHORT_TERM` memories age into `ARCHIVE` (e.g., after 30 days) or when memories are superseded by a contradiction.

Every compression action updates the embedding, increments the `compression_level` attribute, and logs a `COMPRESSED` event in `memory_history`.

---

## Database Architecture & Schema

ADAM uses a local SQLite database (`data/adam.db`) with two tables:

### `memories` Table
| Column | Type | Description |
|---|---|---|
| `memory_id` | `TEXT PRIMARY KEY` | UUID string identifying the memory |
| `user_id` | `TEXT NOT NULL` | User or session identifier |
| `content` | `TEXT NOT NULL` | Textual memory representation |
| `embedding` | `TEXT NOT NULL` | JSON-serialized 384-dimensional vector (`all-MiniLM-L6-v2`) |
| `created_at` | `TEXT NOT NULL` | ISO 8601 UTC creation timestamp |
| `last_accessed` | `TEXT NOT NULL` | ISO 8601 UTC last access timestamp |
| `access_count` | `INTEGER NOT NULL` | Counter incremented on retrieval or duplicate hit |
| `importance_score` | `REAL NOT NULL` | Bounded $[0.0, 1.0]$ heuristic score |
| `tier` | `TEXT NOT NULL` | `WORKING`, `SHORT_TERM`, `LONG_TERM`, or `ARCHIVE` |
| `compression_level` | `INTEGER NOT NULL` | `0` (raw), `1` (L1 concise), `2` (L2 compact) |
| `updated_at` | `TEXT NOT NULL` | ISO 8601 UTC timestamp of last content/tier update |
| `superseded_by` | `TEXT NOT NULL` | Memory ID of superseding memory (for contradictions) |
| `source_role` | `TEXT NOT NULL` | Origin role: `'user'` or `'assistant'` |

### `memory_history` Table
| Column | Type | Description |
|---|---|---|
| `history_id` | `TEXT PRIMARY KEY` | UUID string identifying the audit record |
| `memory_id` | `TEXT NOT NULL` | Associated memory ID |
| `user_id` | `TEXT NOT NULL` | User identifier |
| `operation` | `TEXT NOT NULL` | `NEW`, `DUPLICATE`, `RELATED`, `CONTRADICTORY`, `COMPRESSED`, `PROMOTED` |
| `old_content` | `TEXT` | Prior text content before update or compression |
| `new_content` | `TEXT` | Updated or merged text content |
| `reason` | `TEXT` | Human-readable explanation / classifier rationale |
| `created_at` | `TEXT NOT NULL` | ISO 8601 UTC timestamp of the audit entry |

---

## Research Web Interface (5 Core Views)

The built-in single-page application is served at `http://127.0.0.1:8000`:

1. **Chat & Live Pipeline Inspector**:
   - Interactive conversation with memory injection.
   - **Real-Time Step Banner**: Follows execution across `[1. Extraction] ➔ [2. Consolidation] ➔ [3. Semantic Retrieval] ➔ [4. Context & LLM] ➔ [5. Response Memory]`.
   - **Visual Badges**: Attached to both user and assistant chat bubbles showing Importance Score, Tier, Consolidation Action, and Compression Level.
   - **Expandable Inspector**: Inspect candidate memories, cosine similarities, signal weight breakdowns, and raw LLM reasoning.
2. **Memory Dashboard (Kanban View)**:
   - 4-column layout displaying memories in `WORKING`, `SHORT_TERM`, `LONG_TERM`, and `ARCHIVE`.
   - Real-time search bar, tier filter, and minimum importance threshold slider.
   - Per-card actions: inspect audit history, trigger manual tier transitions, or delete memories.
3. **Memory Lifecycle & State Machine Explorer**:
   - Interactive visual state diagram mapping transitions and policy thresholds.
   - Deep-dive timeline inspector displaying audit history diffs for any selected memory.
4. **Consolidation Audit Feed**:
   - Global event feed of all memory mutations (`NEW`, `DUPLICATE`, `RELATED`, `CONTRADICTORY`, `COMPRESSED`).
   - Side-by-side text comparisons with operation metadata.
5. **System & Research Metrics Panel**:
   - Aggregate counters: total memories, average importance score, consolidation counts, compression statistics.
   - Visual tier distribution progress bars.
   - Live Ollama health check status pill and SQLite database metrics.
   - Safe database reset modal requiring explicit confirmation.

---

## Configuration & Environment Variables

All settings can be customized via environment variables or configured in `app/config.py`:

| Variable | Default | Type | Description |
|---|---|---|---|
| `ADAM_DATABASE_PATH` | `data/adam.db` | Path | SQLite database file location |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | String | SentenceTransformer embedding model identifier |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | URL | Ollama server endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | String | Local LLM model tag for consolidation & chat |
| `OLLAMA_TIMEOUT` | `120.0` | Float | HTTP timeout in seconds for Ollama requests |
| `IMPORTANCE_INTENT_WEIGHT` | `0.35` | Float | Weight for intent detection signal |
| `IMPORTANCE_SPECIFICITY_WEIGHT` | `0.25` | Float | Weight for specificity & entity density signal |
| `IMPORTANCE_DURABILITY_WEIGHT` | `0.20` | Float | Weight for permanence / durability signal |
| `IMPORTANCE_SALIENCE_WEIGHT` | `0.10` | Float | Weight for information density & length signal |
| `IMPORTANCE_RECURRENCE_WEIGHT` | `0.05` | Float | Weight for access recurrence signal |
| `IMPORTANCE_RECENCY_WEIGHT` | `0.05` | Float | Weight for exponential recency decay signal |
| `INITIAL_ARCHIVE_THRESHOLD` | `0.30` | Float | Max importance score for initial placement in `ARCHIVE` |
| `INITIAL_LONG_TERM_THRESHOLD` | `0.70` | Float | Min importance score for initial placement in `WORKING` |
| `CONSOLIDATION_MIN_SIMILARITY` | `0.35` | Float | Minimum cosine similarity to qualify as a consolidation candidate |
| `CONSOLIDATION_CANDIDATE_LIMIT` | `3` | Integer | Max candidates evaluated during consolidation |
| `WORKING_TO_LONG_TERM_DAYS` | `7` | Float | Days before a `WORKING` memory can age into `LONG_TERM` |
| `WORKING_TO_LONG_TERM_MIN_ACCESSES`| `1` | Integer | Minimum accesses required before transition to `LONG_TERM` |
| `SHORT_TERM_TO_ARCHIVE_DAYS` | `30` | Float | Days before a `SHORT_TERM` memory ages into `ARCHIVE` |
| `WORKING_COMPRESSION_LEVEL` | `1` | Integer | Compression level for `WORKING` $\to$ `LONG_TERM` transitions |
| `ARCHIVE_COMPRESSION_LEVEL_TARGET` | `2` | Integer | Compression level for `SHORT_TERM` $\to$ `ARCHIVE` transitions |
| `USE_TF` | `0` | Integer | Set to `0` to disable TensorFlow and suppress PyTorch/TF warnings |

---

## REST API Reference

Interactive OpenAPI (Swagger) documentation is available at `http://127.0.0.1:8000/docs`.

### Summary of Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Research Web Interface SPA |
| `GET` | `/health` | Service health status and phase version |
| `GET` | `/system/status` | Runtime status of Ollama connection, models, and database |
| `POST` | `/chat` | Executes complete conversational turn with full pipeline telemetry |
| `POST` | `/memory` | Stores and consolidates an individual memory directly |
| `POST` | `/retrieve` | Performs semantic cosine similarity retrieval |
| `GET` | `/memories` | Lists memories with optional `user_id`, `tier`, and `search` filters |
| `GET` | `/memory/{id}` | Retrieves a single memory by ID |
| `GET` | `/memory/{id}/history` | Retrieves immutable audit history for a single memory |
| `POST` | `/memory/{id}/transition` | Manually transitions a memory tier using lifecycle compression |
| `DELETE` | `/memory/{id}` | Deletes a memory and its audit records |
| `GET` | `/history` | Global audit feed of recent consolidation/compression events |
| `GET` | `/metrics` | Computes aggregate research metrics and tier distributions |
| `POST` | `/reset` | Clears all data from the database with confirmation |

---

### Request & Response Examples

#### 1. Conversational Chat Turn (`POST /chat`)
**Request:**
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "message": "I prefer using FastAPI and SQLite for backend prototypes.",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "response": "That's a great choice! FastAPI offers rapid asynchronous development with automatic OpenAPI documentation, and SQLite makes prototyping completely self-contained.",
  "user_memory": {
    "memory": {
      "memory_id": "8f3b2d1c-...",
      "user_id": "user-1",
      "content": "I prefer using FastAPI and SQLite for backend prototypes.",
      "importance_score": 0.825,
      "tier": "WORKING",
      "compression_level": 0,
      "access_count": 0
    },
    "action": "NEW",
    "decision_reason": "No similar candidates found in memory",
    "candidates": [],
    "is_stored": true,
    "score_breakdown": {
      "intent": { "value": 0.8, "contribution": 0.28, "reason": "preference: 'i prefer'" },
      "specificity": { "value": 1.0, "contribution": 0.25, "reason": "Tech terms: fastapi, sqlite" },
      "durability": { "value": 0.85, "contribution": 0.17, "reason": "high intent implies enduring value" },
      "salience": { "value": 0.70, "contribution": 0.07, "reason": "Density 0.67, 6/9 content words" }
    }
  },
  "retrieved_memories": [],
  "assistant_memory": {
    "is_stored": true,
    "action": "NEW",
    "importance_score": 0.74,
    "tier": "WORKING"
  },
  "pipeline_stages": [
    { "stage": "User Message Memory Analysis & Scoring", "status": "completed", "detail": "Importance: 0.83 (Tier: WORKING)" },
    { "stage": "User Consolidation Check", "status": "completed", "detail": "Action: NEW | Candidates: 0" },
    { "stage": "Memory Retrieval", "status": "completed", "detail": "Retrieved 0 relevant memories" },
    { "stage": "Context Assembly & LLM Generation", "status": "completed", "detail": "Generated 174 chars" },
    { "stage": "Response Memory Processing", "status": "completed", "detail": "Stored Response Memory: 0.74 (Tier: WORKING, Action: NEW)" }
  ],
  "llm_error": null
}
```

#### 2. Direct Memory Ingestion (`POST /memory`)
**Request:**
```bash
curl -X POST http://127.0.0.1:8000/memory \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "content": "My primary programming language is Python."
  }'
```

#### 3. Semantic Retrieval (`POST /retrieve`)
**Request:**
```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "query": "What language do I code in?",
    "top_k": 2
  }'
```

#### 4. Explicit Lifecycle Transition (`POST /memory/{id}/transition`)
**Request:**
```bash
curl -X POST http://127.0.0.1:8000/memory/8f3b2d1c-.../transition \
  -H "Content-Type: application/json" \
  -d '{ "target_tier": "LONG_TERM" }'
```

---

## Quickstart & Setup

### 1. Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Ollama** (optional, recommended for live LLM consolidation & chat generation): [Download Ollama](https://ollama.com/)

### 2. Environment Setup
```bash
# Clone and enter directory
cd adam_memory

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Ollama (Optional)
In a separate terminal window:
```bash
# Start Ollama daemon
ollama serve

# Pull lightweight 3B model (tested on Apple Silicon M-series and modern CPUs)
ollama pull qwen2.5:3b
```
*(Note: If Ollama is offline or unavailable, ADAM continues operating with local vector embeddings, heuristic scoring, and automatic fallback consolidation).*

### 4. Run the Application
```bash
source .venv/bin/activate
USE_TF=0 uvicorn app.main:app --reload --port 8000
```

Open your browser at:
```
http://127.0.0.1:8000
```

---

## Automated Test Suite

The project includes **33 automated test cases** covering every core subsystem:

```bash
source .venv/bin/activate
USE_TF=0 python -m pytest -v
```

### Test Coverage Breakdown

- **`tests/test_phase1.py` (18 tests)**:
  - Database creation, schema migration, and row serialization.
  - Decoupled importance scoring vs. lifecycle tier placement.
  - Independent lifecycle transitions (`WORKING` $\to$ `LONG_TERM`, `SHORT_TERM` $\to$ `ARCHIVE`).
  - Semantic vector retrieval, ranking, and access count metadata updating.
  - Fast heuristic duplicate detection and timestamp updates.
  - Related memory merging and embedding recalculation.
  - Contradiction handling and immutable audit history logging.
  - Noise, greeting, and filler rejection.
- **`tests/test_web_api.py` (15 tests)**:
  - Full `/chat` conversational pipeline and telemetry trace.
  - Live metrics computation, search, and memory listing endpoints.
  - Manual tier transition and compression endpoints.
  - System status, Ollama health checks, and database reset safety modal.
  - Static HTML/CSS/JS web asset delivery and cache headers.
  - Domain-agnostic entity recognition and multi-signal score breakdowns.
  - Exact duplicate heuristic bypass without LLM overhead.
  - Source-role isolation preventing assistant responses from merging into user memories.
  - Safe contradiction superseding prior memories with active tier preservation.

---

## Example Interaction Scenarios

### Scenario 1: Initial Preference Ingestion
```text
User: "I am building an autonomous memory system using Python and SQLite."
ADAM:
  - Noise Filter: Passed (not filler)
  - Importance Scorer: 0.88 (High - Project intent + technical keywords)
  - Initial Tier: WORKING
  - Candidates: None (Action: NEW)
  - Stored: Stored as fresh WORKING memory
```

### Scenario 2: Incremental Expansion (`RELATED`)
```text
User: "For vector search, I am also using SentenceTransformers with MiniLM."
ADAM:
  - Candidates: Finds candidate ("...building an autonomous memory system...") with similarity ~0.65
  - LLM Consolidation: Identifies non-conflicting topic expansion (RELATED)
  - Merge Quality Guard: Verified
  - Result: Merges into single coherent memory, updates embedding, and records RELATED audit event
```

### Scenario 3: Explicit Update / Contradiction (`CONTRADICTORY`)
```text
User: "Actually, I switched from SQLite to PostgreSQL for better concurrency."
ADAM:
  - Heuristic Contradiction: Trigger "switched from" + shared topic word "sqlite"
  - Prior Memory: Marked superseded_by new ID, moved to ARCHIVE (L2 compressed)
  - New Memory: Inserted into WORKING tier (inherits active priority)
  - Audit Trail: Immutable CONTRADICTORY records logged for both memories
```

### Scenario 4: Small Talk Filtering
```text
User: "Hello, good morning!"
ADAM:
  - Noise Filter: Detected as greeting token
  - Action: No memory created; response generated without polluting vector storage
```

---

## Directory Structure

```text
adam_memory/
├── app/
│   ├── main.py                     # FastAPI application, lifespan, endpoints, static mounts
│   ├── config.py                   # Environment-backed settings, scoring weights, thresholds
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py               # LLMClient interface, OllamaClient (JSON mode), Pydantic schemas
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── models.py               # Memory dataclass, SQLite row mapping, UTC utilities
│   │   ├── storage.py              # SQLite storage engine, schema migrations, audit records, metrics
│   │   ├── importance.py           # Multi-signal importance scorer & filler/boilerplate detector
│   │   ├── tiers.py                # Operational lifecycle tiers (WORKING, SHORT_TERM, LONG_TERM, ARCHIVE)
│   │   ├── consolidation.py        # Candidate search, heuristic-first consolidation, merge guards
│   │   └── compression.py          # Level 1 & Level 2 lifecycle compression transitions
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py           # SentenceTransformer (all-MiniLM-L6-v2) embedding service
│   │   ├── similarity.py           # Cosine similarity calculation
│   │   └── retrieval.py            # RetrievalService orchestrating chat turn, search, and storage
│   └── static/                     # Built-in Research Web Interface (SPA)
│       ├── index.html              # 5-view layout with dark-mode styling
│       ├── css/
│       │   └── styles.css          # Design system, glassmorphism, responsive grid, tier accents
│       └── js/
│           └── app.js              # State store, live pipeline animator, tabs, modals, API client
├── data/
│   └── adam.db                     # Local SQLite database (created on first run)
├── tests/
│   ├── test_phase1.py              # 18 unit tests for core memory, tiers, scoring, consolidation
│   └── test_web_api.py             # 15 tests for web API, chat pipeline, role isolation, heuristics
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive framework documentation
```

---

## License

This project is developed as an open research prototype for adaptive agent memory management. Distributed under the MIT License.
