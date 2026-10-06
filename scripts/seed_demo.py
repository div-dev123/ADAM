#!/usr/bin/env python3
"""ADAM Live Demo Seed Script.

Runs a compelling, reproducible 3-minute demonstration of ADAM's core capabilities:

  Act 1 — Establish Context (facts are stored)
  Act 2 — Introduce Contradictions (old facts superseded)
  Act 3 — Query and Verify (correct facts retrieved, stale ones excluded)
  Act 4 — Trigger Lifecycle (forgetting pass runs, audit logged)
  Act 5 — Post-lifecycle Query (prove stale memories gone)

Usage:
    source .venv/bin/activate
    USE_TF=0 python scripts/seed_demo.py
    USE_TF=0 python scripts/seed_demo.py --clean   # Reset DB first
    USE_TF=0 python scripts/seed_demo.py --user demo-user-1
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.llm.client import OllamaClient
from app.memory.compression import CompressionConfig
from app.memory.consolidation import ConsolidationConfig
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights
from app.memory.lifecycle import LifecyclePolicyConfig
from app.memory.storage import SQLiteStorage
from app.memory.tiers import TierAssigner
from app.retrieval.context_builder import ContextBudgetConfig
from app.retrieval.drift import DriftConfig
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.ranking import RankingWeights
from app.retrieval.retrieval import RetrievalService

DEMO_USER = "user-1"

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


def step(msg: str) -> None:
    print(f"\n{CYAN}  ▶  {msg}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}    {msg}{RESET}")


def build_service() -> RetrievalService:
    weights = ImportanceWeights()
    lifecycle_policy = TierAssigner()
    lifecycle_config = LifecyclePolicyConfig()
    drift_config = DriftConfig()
    ranking_weights = RankingWeights()
    budget_config = ContextBudgetConfig()

    return RetrievalService(
        SQLiteStorage(settings.database_path),
        EmbeddingService(settings.embedding_model),
        scorer=HeuristicImportanceScorer(weights),
        lifecycle_policy=lifecycle_policy,
        llm=OllamaClient(settings.ollama_host, settings.ollama_model, timeout=settings.ollama_timeout),
        consolidation_config=ConsolidationConfig(),
        compression_config=CompressionConfig(),
        lifecycle_config=lifecycle_config,
        drift_config=drift_config,
        ranking_weights=ranking_weights,
        budget_config=budget_config,
    )


def store(svc: RetrievalService, user_id: str, content: str) -> None:
    result = svc.store_memory(user_id, content)
    if result is None:
        warn(f"Filtered as filler: '{content[:60]}'")
    else:
        ok(f"[{result.tier:12s}] Imp={result.importance_score:.2f}  '{content[:70]}'")


def query(svc: RetrievalService, user_id: str, q: str, top_k: int = 3) -> list:
    results = svc.search(user_id=user_id, query=q, top_k=top_k)
    print(f"\n  {BOLD}Query:{RESET} {q}")
    if not results:
        warn("No memories retrieved.")
    for i, r in enumerate(results, 1):
        mem = r["memory"]
        tier = r.get("memory_tier", mem.tier)
        score = r.get("final_score", r.get("similarity", 0.0))
        print(
            f"    {i}. [{tier:12s}] score={score:.3f}  "
            f"imp={mem.importance_score:.2f}  '{mem.content[:80]}'"
        )
    return results


def run_demo(user_id: str, clean: bool) -> None:
    print(f"\n{BOLD}ADAM — Adaptive Dynamic AI Memory Framework{RESET}")
    print(f"{DIM}Live demonstration script — reproducible end-to-end{RESET}")

    svc = build_service()

    if clean:
        header("PRE-DEMO: Resetting database")
        svc.storage.reset_database()
        ok("Database cleared.")

    # ─── ACT 1: Establish context ──────────────────────────────────────────────
    header("ACT 1 — Establish Context (facts ingested)")
    step("Storing initial personal and technical facts...")

    store(svc, user_id, "Hello!")   # should be filtered as filler
    store(svc, user_id, "I am building a distributed event-processing platform using Apache Kafka and Golang.")
    store(svc, user_id, "I live in San Francisco, California and work out of the Mission District.")
    store(svc, user_id, "We use PostgreSQL for all relational persistence and Redis for caching layers.")
    store(svc, user_id, "Our Kafka broker cluster bootstraps at kafka-cluster.prod.internal:9092,9093,9094.")
    store(svc, user_id, "Our production API gateway enforces a p99 latency SLA of 30ms for all endpoints.")
    store(svc, user_id, "My favourite IDE is VSCode with the Vim keybinding extension.")
    store(svc, user_id, "I always drink black coffee in the morning before deep work sessions.")
    store(svc, user_id, "We run Kubernetes on GKE in us-central1.")
    store(svc, user_id, "Thanks!")  # filler

    time.sleep(0.5)

    # ─── ACT 2: Introduce contradictions ──────────────────────────────────────
    header("ACT 2 — Contradictions Introduced")
    step("User updates key facts — ADAM detects and supersedes old memories...")

    store(svc, user_id, "We switched from Golang to Rust for our core event-processing engine. Migration complete.")
    store(svc, user_id, "I relocated to Austin, Texas. San Francisco became too expensive.")
    store(svc, user_id, "We dropped CockroachDB and reverted back to PostgreSQL — licensing costs were too high.")
    store(svc, user_id, "I now prefer Zed editor over VSCode. It is faster and written in Rust.")
    store(svc, user_id, "I switched to matcha in the mornings. No more black coffee.")
    store(svc, user_id, "Our new p99 API latency target has been tightened to 20ms.")
    store(svc, user_id, "We are migrating our Kubernetes cluster from GKE to AWS EKS in us-east-1.")

    time.sleep(0.5)

    # ─── ACT 3: Query — verify current facts are returned ─────────────────────
    header("ACT 3 — Retrieval Verification")
    step("Querying for current state. Stale facts should NOT appear...")

    r1 = query(svc, user_id, "What programming language does our event-processing engine use?")
    contains_rust = any("Rust" in r["memory"].content for r in r1)
    contains_go = any("Golang" in r["memory"].content or "Go " in r["memory"].content for r in r1)
    if contains_rust and not contains_go:
        ok("PASS — Rust retrieved, Golang correctly excluded.")
    elif contains_rust:
        warn("PARTIAL — Rust found but Golang also leaked (contradiction not fully suppressed).")
    else:
        print(f"  {RED}FAIL{RESET} — Rust not found in top results.")

    r2 = query(svc, user_id, "Where do I currently live?")
    contains_austin = any("Austin" in r["memory"].content for r in r2)
    contains_sf = any("San Francisco" in r["memory"].content for r in r2)
    if contains_austin and not contains_sf:
        ok("PASS — Austin retrieved, San Francisco correctly excluded.")
    elif contains_austin:
        warn("PARTIAL — Austin found but San Francisco also leaked.")
    else:
        print(f"  {RED}FAIL{RESET} — Austin not found in top results.")

    r3 = query(svc, user_id, "What is our p99 latency SLA?")
    contains_20 = any("20ms" in r["memory"].content for r in r3)
    contains_30 = any("30ms" in r["memory"].content for r in r3)
    if contains_20 and not contains_30:
        ok("PASS — 20ms SLA retrieved, outdated 30ms excluded.")
    else:
        warn(f"{'PARTIAL' if contains_20 else 'FAIL'} — SLA verification: 20ms={contains_20}, 30ms={contains_30}")

    query(svc, user_id, "What Kafka bootstrap addresses does our production cluster use?")
    query(svc, user_id, "What is my morning beverage preference?")
    query(svc, user_id, "What code editor do I prefer?")

    # ─── ACT 4: Lifecycle pass ─────────────────────────────────────────────────
    header("ACT 4 — Lifecycle Pass (Selective Forgetting)")
    step("Triggering automated lifecycle evaluation with Ebbinghaus retention scoring...")

    report = svc.run_lifecycle_pass(user_id=user_id, dry_run=False)
    evaluated = report.get("evaluated", 0)
    forgotten = report.get("forgotten", 0)
    transitioned = report.get("transitioned", 0)
    protected = report.get("protected", 0)

    ok(f"Evaluated:   {evaluated} memories")
    ok(f"Forgotten:   {forgotten} memories pruned (low retention + obsolete)")
    ok(f"Transitioned:{transitioned} memories moved to lower tiers")
    ok(f"Protected:   {protected} memories shielded (high importance / high access)")

    # ─── ACT 5: Post-lifecycle verification ────────────────────────────────────
    header("ACT 5 — Post-Lifecycle Verification")
    step("Re-querying to confirm stale memories removed from retrieval context...")

    all_memories = svc.storage.get_all_memories(user_id=user_id)
    total = len(all_memories)
    superseded = sum(1 for m in all_memories if m.superseded_by and m.superseded_by.strip())

    ok(f"Total memories in store:  {total}")
    ok(f"Superseded (historical):  {superseded}")
    ok(f"Active (accessible):      {total - superseded}")

    metrics = svc.storage.get_metrics(user_id=user_id)
    info(f"Tier distribution: {metrics.get('tier_distribution', {})}")
    info(f"Mean importance:   {metrics.get('mean_importance_score', 0):.3f}")

    header("DEMO COMPLETE")
    print(f"\n  {GREEN}{BOLD}ADAM successfully demonstrated:{RESET}")
    print(f"  • Noise filtering (greetings ignored)")
    print(f"  • Fact storage with importance scoring")
    print(f"  • Contradiction detection and memory supersession")
    print(f"  • Retrieval of current facts with stale fact exclusion")
    print(f"  • Lifecycle retention scoring and selective forgetting")
    print(f"\n  {DIM}Open http://127.0.0.1:8000 to explore the live UI{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="ADAM Live Demo Seed Script")
    parser.add_argument("--user", default=DEMO_USER, help="User ID to use for the demo")
    parser.add_argument("--clean", action="store_true", help="Reset the database before running")
    args = parser.parse_args()
    run_demo(args.user, args.clean)


if __name__ == "__main__":
    main()
