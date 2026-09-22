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
- **Selective Forgetting & Automatic Memory Lifecycle (Phase 4)**: Mathematical retention scoring based on Ebbinghaus exponential decay, access recurrence, and intrinsic importance. Safely transitions tiers and prunes obsolete, low-value memories with strict anti-amnesia guarantees and permanent audit logging.
- **Adaptive Retrieval & Query Drift Detection (Phase 5)**: Compares current query with conversation context using semantic distance ($1 - \text{cosine similarity}$) and temporal staleness. Dynamically selects memory scope tiers (Low drift $\to$ `WORKING` + `SHORT_TERM`; Medium drift $\to$ includes `LONG_TERM`; High drift $\to$ broadens across all tiers including `ARCHIVE`) with rich telemetry and configurable thresholds.
- **Two-Level Compression Engine**: Compresses aging memories (`WORKING` $\to$ `LONG_TERM` at Level 1; `SHORT_TERM` $\to$ `ARCHIVE` at Level 2) to preserve facts while reducing token overhead.
- **Offline & Graceful Degradation**: Functions standalone with local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and SQLite. If the local Ollama LLM (`qwen2.5:3b`) is offline or busy, the system gracefully falls back to deterministic heuristics.
- **Interactive Research Web Interface**: Built-in 5-view single-page application (SPA) with real-time pipeline visualization, Kanban memory dashboard, state machine explorer, audit feed, and system metrics.

---

## Architecture & Pipelines

ADAM operates across two primary pipelines: a dual-turn **Write Path** and an adaptive **Read Path**.

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
│                       ADAM ADAPTIVE READ PATH (PHASE 5)                          │
│                                                                                  │
│   User Query + Conversation Context (Chat History / Recent Turns)                │
│         │                                                                        │
│         ▼                                                                        │
│   [Query Drift Detector]                                                         │
│     ├── Semantic Distance = 1.0 - CosineSimilarity(Query, Context)               │
│     ├── Temporal Staleness Factor = 1.0 - exp(-0.6931 * Δt / half_life)          │
│     └── Drift Score = (1 - w_time) * SemDist + w_time * TimeFactor               │
│         │                                                                        │
│         ▼                                                                        │
│   [Adaptive Scope Selection]                                                     │
│     ├── LOW Drift (< 0.55)      ──► [WORKING, SHORT_TERM] (Focused)              │
│     ├── MEDIUM Drift (0.55-0.85)──► [WORKING, SHORT_TERM, LONG_TERM] (Expanded)  │
│     └── HIGH Drift (≥ 0.85)     ──► [WORKING, SHORT_TERM, LONG_TERM, ARCHIVE]    │
│         │                                                                        │
│         ▼                                                                        │
│   [Tier-Filtered Candidate Retrieval]                                            │
│         │                                                                        │
│         ├───────────────────────────────────┬────────────────────────────────────┤
│         ▼                                   ▼                                    │
│   [Dense Vector Search]               [Sparse BM25 Search]                       │
│   SentenceTransformer MiniLM          Okapi BM25 Lexical Scoring                 │
│   Cosine Similarity [0.0 - 1.0]       TF-IDF with Length Normalization           │
│         │                                   │                                    │
│         ▼                                   ▼                                    │
│   Dense Ranked Candidates             BM25 Ranked Candidates                     │
│         │                                   │                                    │
│         └───────────────────┬───────────────┘                                    │
│                             ▼                                                    │
│               [Reciprocal Rank Fusion (RRF)]                                     │
│               Score = Σ 1 / (60 + Rank_m)                                        │
│                             │                                                    │
│                             ▼                                                    │
│               [Unified Top-K Memory Ranking & Access Tracking]                   │
│               Enrich metadata: similarity, tier, importance, drift telemetry     │
│               Increment access_count, update last_accessed                       │
│                             │                                                    │
│                             ▼                                                    │
│               [Context Assembly & LLM Generation]                                │
│               Ollama (qwen2.5:3b) Context Injection                              │
│                             │                                                    │
│                             ▼                                                    │
│               Assistant Response ──► Returned to user & routed to Write Path     │
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

### 4. Hybrid Search Engine (Dense + Okapi BM25 with RRF)

Pure vector (dense) search captures semantic concepts but often fails on exact keywords, numbers, acronyms, or specific technical identifiers. Pure keyword (sparse) search misses synonyms and conceptual intent. ADAM unifies both via **Hybrid Search**:

1. **Dense Vector Search**: Encodes user queries via `sentence-transformers/all-MiniLM-L6-v2` and computes cosine similarity against all active candidate memories.
2. **Sparse Lexical Search (Okapi BM25)**: An in-memory, self-contained Okapi BM25 engine with term frequency ($TF$), inverse document frequency ($IDF$), and document length normalization ($k_1=1.5, b=0.75$):
   $$\text{score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
3. **Reciprocal Rank Fusion (RRF)**: Merges the dense and sparse candidate rankings without fragile score normalization:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)} \quad (k=60)$$
   Documents with strong consensus across both semantic understanding and exact keyword grounding rise to the top.

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
| `operation` | `TEXT NOT NULL` | `NEW`, `DUPLICATE`, `RELATED`, `CONTRADICTORY`, `COMPRESSED`, `PROMOTED`, `TIER_TRANSITION`, `FORGOTTEN` |
| `old_content` | `TEXT` | Prior text content before update, compression, or forgotten content snapshot |
| `new_content` | `TEXT` | Updated or merged text content (or `NULL` if forgotten) |
| `reason` | `TEXT` | Human-readable explanation / classifier / lifecycle rationale |
| `created_at` | `TEXT NOT NULL` | ISO 8601 UTC timestamp of the audit entry |

---

### 5. Selective Forgetting & Automatic Memory Lifecycle (Phase 4)

In long-running agentic systems, storing memories indefinitely causes storage bloat, index degradation, and context poisoning with obsolete facts. ADAM introduces a **mathematical, continuous forgetting and lifecycle engine** rather than crude hard-coded deletion rules:

```
                          ┌───────────────────────────┐
                          │   Memory Creation (New)   │
                          └─────────────┬─────────────┘
                                        │
                         Initial Tier by Importance
                                        │
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
       ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
       │    WORKING    │        │  SHORT_TERM   │        │    ARCHIVE    │
       │ (Score ≥ 0.70)│        │(0.30 to 0.70) │        │ (Score ≤ 0.30)│
       └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
               │                        │                        │
        Age ≥ 7d, Inactive       Age ≥ 30d or             Obsolescence Check:
        & Importance ≥ 0.60      Retention < 0.40         S_forget ≥ 0.75,
               │                        │                 Age ≥ 60d, Unprotected
               ▼                        ▼                        │
       ┌───────────────┐                │                        ▼
       │   LONG_TERM   │◄───────────────┘                ┌───────────────┐
       │ (Consolidated)│                                 │   FORGOTTEN   │
       └───────────────┘                                 │ (Audited Log) │
                                                         └───────────────┘
```

#### Mathematical Retention & Forgetting Formulation:
For every memory $m$, ADAM computes a continuous **Retention Score** $R(m) \in [0.0, 1.0]$:
$$R(m) = \max\left(0.0, \min\left(1.0, \, w_i \cdot I(m) + w_r \cdot \text{RecencyScore}(m) + w_f \cdot \text{FrequencyScore}(m) - \text{Penalty}_{\text{comp}}(m)\right)\right)$$

Where:
* **$I(m)$**: Normalized intrinsic importance score $\in [0.0, 1.0]$.
* **$\text{RecencyScore}(m) = e^{-\lambda \cdot \Delta t_{\text{inactive}}}$**: Ebbinghaus exponential decay modeling human forgetting ($\lambda=0.05$, half-life $\approx 14$ days).
* **$\text{FrequencyScore}(m) = \min\left(1.0, \, \frac{\log(1 + \text{access\_count})}{\log(1 + N_{\text{target}})}\right)$**: Logarithmic access boost rewarding frequently retrieved memories.
* **$\text{Penalty}_{\text{comp}}(m) = \text{compression\_level} \times 0.05$**: Minor penalty for highly condensed records.

The **Forgetting Score** $S_{\text{forget}}(m)$ is:
$$S_{\text{forget}}(m) = 1.0 - R(m)$$

#### Policy Rules & Safety Guarantees:
1. **Tier Transitions**:
   - `WORKING` $\to$ `LONG_TERM`: When a working memory reaches age $\ge 7$ days, has been inactive for $\ge 5$ days, but retains high importance ($I \ge 0.60$) with at least 1 access.
   - `SHORT_TERM` $\to$ `ARCHIVE`: When a short-term memory ages past 30 days or its retention score drops below $0.40$.
   - **Active Retention Boost**: Frequently accessed memories ($\text{access\_count} \ge 3$) with recent activity are **immune to demotion**, staying in their active tiers longer.
2. **Selective Forgetting (Deletion)**:
   - A memory is eligible for forgetting if and only if $S_{\text{forget}}(m) \ge 0.75$, the memory is in `ARCHIVE` (or superseded), and it is **not protected**.
3. **Anti-Amnesia Safety Constraints**:
   - Memories with $I(m) \ge 0.70$ (high importance) or $\text{access\_count} \ge 3$ (frequent access) are **strictly protected** against automatic forgetting.
4. **Permanent Audit Trail**:
   - Deletion is completely auditable: when forgotten, ADAM logs `operation="FORGOTTEN"` to `memory_history` with the full content snapshot, mathematical reasoning, and timestamp. Audit history is never deleted.

---

### 6. Adaptive Retrieval & Query Drift Detection (Phase 5)

In dynamic multi-turn interactions, user topics evolve, drift, or abruptly pivot. Traditional retrieval architectures either:
- **Over-restrict scope**: Limiting retrieval only to working memory causes the agent to miss relevant background decisions or historical preferences when discussing older topics.
- **Over-expand scope**: Searching across every tier (including archives and old superseded notes) pollutes context with obsolete or irrelevant candidates, confusing the LLM and diluting attention.

ADAM solves this through **Query Drift Detection**: dynamically analyzing semantic distance and temporal staleness between the incoming query and recent conversational context to select the optimal memory scope.

```
Incoming Query + Conversation Context (Chat History / Recent Turns)
                              │
                              ▼
                 [Query Drift Detector]
       Semantic Distance: D_sem = 1.0 - CosineSimilarity(q, c)
       Temporal Staleness: F_time = 1.0 - exp(-0.6931 * Δt / t_half)
       Drift Score: S_drift = (1 - w_time) * D_sem + w_time * F_time
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
     S_drift < 0.55    0.55 ≤ S_drift < 0.85   S_drift ≥ 0.85
       [LOW DRIFT]       [MEDIUM DRIFT]        [HIGH DRIFT]
             │                │                │
             ▼                ▼                ▼
    Focused Scope:     Expanded Scope:       Broad Scope:
    WORKING,           WORKING,              WORKING,
    SHORT_TERM         SHORT_TERM,           SHORT_TERM,
                       LONG_TERM             LONG_TERM,
                                             ARCHIVE
```

#### Mathematical Drift Formulation:
1. **Semantic Topic Distance**:
   $$D_{\text{sem}}(q, c) = \max\left(0.0, \, \min\left(1.0, \, 1.0 - \text{CosineSimilarity}(\vec{e}_q, \vec{e}_c)\right)\right)$$
   Where $\vec{e}_q$ is the query embedding and $\vec{e}_c$ is the aggregated embedding of recent conversational context.

2. **Temporal Staleness Factor**:
   When conversation turns are separated by significant time gaps $\Delta t$, context relevance decays:
   $$F_{\text{time}}(\Delta t) = 1.0 - \exp\left(-\frac{\ln(2) \cdot \Delta t}{t_{\text{half-life}}}\right)$$
   Where $t_{\text{half-life}}$ defaults to 1.0 hour.

3. **Composite Drift Score**:
   $$S_{\text{drift}} = (1.0 - w_{\text{time}}) \cdot D_{\text{sem}} + w_{\text{time}} \cdot F_{\text{time}}$$
   (Default temporal weight $w_{\text{time}} = 0.20$).

#### Adaptive Scope Selection Policy:
* **LOW Drift ($S_{\text{drift}} < 0.55$)**: The query is strongly aligned with current discussion. Retrieval is focused strictly on active tiers: `[WORKING, SHORT_TERM]`.
* **MEDIUM Drift ($0.55 \le S_{\text{drift}} < 0.85$)**: Topic evolution detected (e.g. pivoting from code endpoints to cloud infrastructure and databases). Retrieval expands to include `[WORKING, SHORT_TERM, LONG_TERM]`.
* **HIGH Drift ($S_{\text{drift}} \ge 0.85$)**: Radical subject change (e.g. pivoting from async Python controllers to restaurants or personal trivia). Retrieval broadens across all tiers: `[WORKING, SHORT_TERM, LONG_TERM, ARCHIVE]`.
* **Unconditioned / Cold-Start Queries**: When no prior context exists (e.g. first turn of a conversation), drift defaults to $0.0$ (`LOW`) with default focused scope `[WORKING, SHORT_TERM]`.

#### Rich Telemetry & Access Tracking:
Every retrieved memory is returned with comprehensive telemetry:
- `similarity`: Hybrid/dense similarity score
- `memory_tier`: Storage tier of origin
- `importance`: Intrinsic heuristic importance score
- `drift_level`: Detected drift category (`LOW`, `MEDIUM`, `HIGH`)
- `scope_selection_reason`: Transparent human-readable explanation of why the scope was chosen
- Every retrieved memory updates its `access_count` and `last_accessed` timestamp, preserving frequently utilized facts across the lifecycle.

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
| `SEARCH_MODE` | `hybrid` | String | Retrieval mode: `hybrid` (Dense + BM25 via RRF), `dense`, or `sparse` |
| `RRF_K` | `60` | Integer | Reciprocal Rank Fusion smoothing parameter $k$ |
| `BM25_K1` | `1.5` | Float | Okapi BM25 term frequency saturation parameter $k_1$ |
| `BM25_B` | `0.75` | Float | Okapi BM25 document length normalization parameter $b$ |
| `LIFECYCLE_FORGETTING_THRESHOLD` | `0.75` | Float | Minimum forgetting score to qualify an obsolete memory for deletion |
| `LIFECYCLE_PROTECTED_IMPORTANCE` | `0.70` | Float | Memories with importance $\ge$ this are strictly protected from deletion |
| `LIFECYCLE_PROTECTED_ACCESS_COUNT` | `3` | Integer | Memories accessed $\ge$ this many times are protected from deletion |
| `LIFECYCLE_WORKING_AGE_DAYS` | `7.0` | Float | Days before a `WORKING` memory transitions to `LONG_TERM` |
| `LIFECYCLE_SHORT_TERM_AGE_DAYS` | `30.0` | Float | Days before a `SHORT_TERM` memory demotes to `ARCHIVE` |
| `LIFECYCLE_ARCHIVE_OBSOLETE_DAYS` | `60.0` | Float | Minimum age in `ARCHIVE` before an obsolete memory can be forgotten |
| `LIFECYCLE_RECENCY_DECAY_RATE` | `0.05` | Float | Exponential decay constant $\lambda$ for Ebbinghaus recency decay |
| `LIFECYCLE_FREQUENT_ACCESS_BOOST_THRESHOLD` | `3` | Integer | Access count required to grant an active retention boost |
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
| `POST` | `/lifecycle/run` | Triggers an automated memory lifecycle evaluation and selective forgetting pass |
| `GET` | `/lifecycle/policy` | Returns active mathematical parameters and thresholds for the memory lifecycle |
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

The project includes **51 automated test cases** covering every core subsystem, including empirical evaluation benchmarks and memory lifecycle management:

```bash
source .venv/bin/activate
USE_TF=0 python -m pytest -v
```

### Empirical Retrieval Benchmark: Hybrid vs. Dense vs. BM25

Run the standalone benchmark suite with live telemetry output:
```bash
USE_TF=0 python -m pytest tests/test_hybrid_search.py -k test_hybrid_search_benchmark_proves_superiority -v -s
```

**Benchmark Results Across Diverse Query Distributions (Lexical, Semantic, Acronym, Multi-Entity):**

| Search Mode | MRR (Mean Reciprocal Rank) | Hit@1 Accuracy | Key Strengths & Failure Modes |
|---|---|---|---|
| **BM25 Only** | `0.9167` | `83.3%` | Exceptional on exact ports/tokens; fails on semantic synonymy (*"hate"* $\to$ *"dislike"*). |
| **Dense Only** | `1.0000` | `100.0%` | Strong on conceptual queries; higher risk of false positives on exact acronyms/version numbers. |
| **Hybrid (RRF)** | `1.0000` | `100.0%` | **Optimal consensus**: fuses semantic meaning with exact keyword grounding. |

### Test Coverage Breakdown

- **`tests/test_lifecycle.py` (9 tests)**:
  - Mathematical retention and forgetting score calculation ($R \in [0, 1]$, $S_{\text{forget}} \in [0, 1]$).
  - Autonomous tier transitions (`WORKING` $\to$ `LONG_TERM`, `SHORT_TERM` $\to$ `ARCHIVE`).
  - Active retention boosts preventing demotion for frequently accessed memories.
  - Old, low-value memory archival.
  - Selective forgetting removing obsolete archived memories.
  - Anti-amnesia protection for high-importance ($I \ge 0.70$) and frequently accessed ($\text{accesses} \ge 3$) memories.
  - Permanent audit logging of `FORGOTTEN` events in `memory_history`.
  - REST API `POST /lifecycle/run` dry-run and live execution modes.
- **`tests/test_hybrid_search.py` (9 tests)**:
  - BM25 tokenization, term frequency, length normalization, empty corpus handling.
  - Reciprocal Rank Fusion (RRF) consensus promotion and disjoint list handling.
  - Hybrid search mode switching (`hybrid`, `dense`, `sparse`).
  - REST API `/retrieve` hybrid telemetry verification (`dense_score`, `bm25_score`, `rrf_score`).
  - Empirical performance benchmark validating Hybrid RRF superiority.
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

### Scenario 5: Selective Forgetting & Lifecycle Pass (Phase 4)
```text
Lifecycle Trigger: POST /lifecycle/run
ADAM Evaluation:
  - Memory A ("Temporary wifi password BlueSky2025", Age 90d, Accesses 0, Importance 0.10, Tier ARCHIVE):
    -> Forgetting Score 0.94 >= 0.75, Unprotected
    -> Action: FORGET. Deleted from active table; logged FORGOTTEN in memory_history.
  - Memory B ("Core production schema definitions", Age 120d, Accesses 6, Importance 0.90, Tier LONG_TERM):
    -> Importance 0.90 >= 0.70 & Accesses 6 >= 3
    -> Action: PROTECT. Anti-amnesia protection guards critical fact from deletion.
```

### Scenario 6: Adaptive Retrieval & Query Drift (Phase 5)
```text
Context: "We are developing an AI agent using FastAPI and Python backend."

Case A (Low Drift - Query: "Which web framework am I using?"):
  - Semantic Distance: 0.48 < 0.55
  - Drift Level: LOW
  - Scope: [WORKING, SHORT_TERM] (Focused)
  - Telemetry: Returns working memory with explanation "Low query drift; topic strongly aligns with recent context."

Case B (Medium Drift - Query: "Where are we deploying Docker containers and PostgreSQL?"):
  - Semantic Distance: 0.76 (0.55 <= Drift < 0.85)
  - Drift Level: MEDIUM
  - Scope: [WORKING, SHORT_TERM, LONG_TERM] (Expanded)
  - Telemetry: Reaches architectural decisions in LONG_TERM; keeps ARCHIVE out of scope.

Case C (High Drift - Query: "What was the name of Luigi's pizza restaurant downtown?"):
  - Semantic Distance: 1.0 >= 0.85
  - Drift Level: HIGH
  - Scope: [WORKING, SHORT_TERM, LONG_TERM, ARCHIVE] (Broadened)
  - Telemetry: Broadens across all tiers, retrieving archived personal note without failing recall.
```

---

## Directory Structure

```text
adam_memory/
├── app/
│   ├── main.py                     # FastAPI application, lifespan, endpoints, static mounts
│   ├── config.py                   # Environment-backed settings, scoring weights, drift thresholds
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py               # LLMClient interface, OllamaClient (JSON mode), Pydantic schemas
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── models.py               # Memory dataclass, SQLite row mapping, UTC utilities
│   │   ├── storage.py              # SQLite storage engine, schema migrations, audit records, metrics
│   │   ├── importance.py           # Multi-signal importance scorer & filler/boilerplate detector
│   │   ├── tiers.py                # Operational lifecycle tiers (WORKING, SHORT_TERM, LONG_TERM, ARCHIVE)
│   │   ├── lifecycle.py            # Phase 4 MemoryLifecycleManager, forgetting policy, anti-amnesia guards
│   │   ├── consolidation.py        # Candidate search, heuristic-first consolidation, merge guards
│   │   └── compression.py          # Level 1 & Level 2 lifecycle compression transitions
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── drift.py                # Phase 5 QueryDriftDetector, semantic distance + temporal decay, scope mappings
│   │   ├── bm25.py                 # Self-contained Okapi BM25 index & tokenizer
│   │   ├── hybrid.py               # HybridSearchService & Reciprocal Rank Fusion (RRF)
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
│   ├── test_adaptive_retrieval.py  # 8 tests for drift detection, adaptive scopes, telemetry & API (Phase 5)
│   ├── test_lifecycle.py           # 9 unit tests for forgetting policy, tier transitions, protection & API (Phase 4)
│   ├── test_hybrid_search.py       # 9 tests for BM25, RRF, hybrid modes & empirical benchmark (Phase 3)
│   ├── test_phase1.py              # 18 unit tests for core memory, tiers, scoring, consolidation
│   └── test_web_api.py             # 15 tests for web API, chat pipeline, role isolation, heuristics
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive framework documentation
```

---

## License

This project is developed as an open research prototype for adaptive agent memory management. Distributed under the MIT License.
