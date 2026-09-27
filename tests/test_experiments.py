"""Unit tests for ADAM research evaluation and ablation framework."""

from pathlib import Path
import tempfile
import pytest

from app.memory.storage import SQLiteStorage
from experiments.configs import ExperimentConfig, get_standard_configurations
from experiments.dataset import (
    BenchmarkDataset,
    ConversationTurn,
    EvalQuery,
    get_default_benchmark_dataset,
)
from experiments.metrics import (
    compute_query_metrics,
    estimate_tokens,
    aggregate_metrics,
)
from experiments.runners import (
    BaseRunner,
    RawHistoryRunner,
    VectorOnlyRunner,
    ImportanceOnlyRunner,
    create_runner,
)


class DummyMockEmbeddings:
    """Mock embeddings for ultra-fast, zero-overhead unit tests."""

    def encode(self, text: str) -> list[float]:
        # Return deterministic 8-dimensional unit vector based on text hash
        val = abs(hash(text)) % 100 / 100.0
        return [val, 1.0 - val, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4]


def test_benchmark_dataset_structure_and_serialization():
    dataset = get_default_benchmark_dataset()
    assert len(dataset.turns) == 14
    assert len(dataset.queries) == 7

    # Verify categories
    categories = {t.category for t in dataset.turns}
    assert "fact" in categories
    assert "filler" in categories
    assert "repeat" in categories
    assert "contradiction" in categories
    assert "preference" in categories

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_benchmark.json"
        dataset.save_json(tmp_file)
        assert tmp_file.exists()

        loaded = BenchmarkDataset.load_json(tmp_file)
        assert loaded.name == dataset.name
        assert len(loaded.turns) == 14
        assert len(loaded.queries) == 7


def test_standard_configurations_registry():
    configs = get_standard_configurations()
    assert "baseline_raw_history" in configs
    assert "baseline_vector_only" in configs
    assert "baseline_importance_only" in configs
    assert "adam_full" in configs
    assert "ablation_no_importance" in configs
    assert "ablation_no_tiers" in configs
    assert "ablation_no_consolidation" in configs
    assert "ablation_no_forgetting" in configs
    assert "ablation_no_query_drift" in configs
    assert "ablation_no_multi_signal" in configs

    # Verify baseline tags
    assert configs["baseline_raw_history"].is_baseline is True
    assert configs["adam_full"].is_baseline is False
    assert configs["ablation_no_consolidation"].enable_consolidation is False
    assert configs["ablation_no_importance"].enable_importance_scoring is False
    assert configs["ablation_no_multi_signal"].enable_multi_signal is False


def test_metrics_computation():
    retrieved = [
        "We quit Rust and switched to Go for all services.",
        "Primary database is PostgreSQL.",
    ]
    target_facts = ["Go", "switched to Go"]
    obsolete_facts = ["Rust", "backend in Rust"]

    # In this test, target "Go" is retrieved, but "Rust" is mentioned in the update notice
    res = compute_query_metrics(
        query_id="q1",
        query_text="What programming language do we use?",
        retrieved_contents=retrieved,
        target_facts=target_facts,
        obsolete_facts=["backend in Rust"],  # specific obsolete claim
        latency_ms=12.5,
    )

    assert res.retrieved_count == 2
    assert res.precision == 0.5  # 1 of 2 retrieved memories contains target fact
    assert res.recall == 1.0  # target fact "Go" retrieved
    assert res.f1 > 0.6
    assert res.obsolete_facts_leaked == 0
    assert res.latency_ms == 12.5


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 2
    assert estimate_tokens("a " * 100) >= 80


def test_runners_database_isolation():
    """Verify that runners operate on dedicated temp databases and never touch production data/adam.db."""
    embeddings = DummyMockEmbeddings()
    configs = get_standard_configurations()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        runner = create_runner(configs["adam_full"], embeddings, work_dir)

        # Database path must be within temp directory, NOT data/adam.db
        assert runner.db_path.parent == work_dir
        assert runner.db_path.name != "adam.db"
        assert runner.db_path.exists()

        turns = [
            ConversationTurn(
                turn_id=1,
                role="user",
                content="Our backend is in Rust.",
                category="fact",
            ),
        ]
        stored = runner.ingest("eval_user_test", turns)
        assert stored == 1
        assert runner.get_stored_memory_count("eval_user_test") == 1

        runner.cleanup()
        assert not runner.db_path.exists()


def test_raw_history_runner():
    embeddings = DummyMockEmbeddings()
    config = get_standard_configurations()["baseline_raw_history"]

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = RawHistoryRunner(config, embeddings, Path(tmpdir))
        turns = [
            ConversationTurn(turn_id=1, role="user", content="Turn 1", category="fact"),
            ConversationTurn(turn_id=2, role="user", content="Turn 2", category="fact"),
            ConversationTurn(turn_id=3, role="user", content="Turn 3", category="fact"),
        ]
        runner.ingest("u1", turns)
        retrieved = runner.retrieve("u1", "query")
        # Returns most recent in reverse chronological order
        assert retrieved == ["Turn 3", "Turn 2", "Turn 1"]
        runner.cleanup()
