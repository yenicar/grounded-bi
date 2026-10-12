from __future__ import annotations

from pathlib import Path
import shutil
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4
import time

from api.dependencies import (
    AppServices,
    PRIORITY_BENCHMARK,
    PRIORITY_INTERACTIVE,
    TaskScheduler,
    build_recommendations,
    build_services,
    ensure_corpus_ready,
    list_documents,
    queue_corpus_job,
)


class NoopScheduler:
    def shutdown(self):
        return None


class FakeCache:
    def __init__(self, payloads):
        self.payloads = payloads

    def load(self, name, default):
        return self.payloads.get(name, default)


def test_build_services_does_not_build_provider_until_needed(monkeypatch):
    provider_calls: list[str] = []

    monkeypatch.setattr("api.dependencies.build_config", lambda: object())
    monkeypatch.setattr("api.dependencies.TaskScheduler", lambda worker_count: NoopScheduler())
    monkeypatch.setattr(
        "api.dependencies.build_provider",
        lambda config: provider_calls.append("provider") or object(),
    )

    services = build_services()

    assert isinstance(services, AppServices)
    assert provider_calls == []

    provider = services.provider

    assert provider is services.provider
    assert provider_calls == ["provider"]


def test_ensure_corpus_ready_rebuilds_when_chunk_files_missing(monkeypatch):
    root = Path.cwd() / ".tmp_test_api_dependencies" / f"case-{uuid4().hex}"
    try:
        child = root / "data" / "chunks" / "story.jsonl"
        parent = root / "data" / "chunks" / "story.parents.jsonl"
        parent.parent.mkdir(parents=True, exist_ok=True)
        # Only create one file so the corpus is out of sync.
        child.write_text("{}", encoding="utf-8")

        services = SimpleNamespace(
            config=SimpleNamespace(
                paths=SimpleNamespace(
                    root=root,
                    cache_dir=root / "data" / "cache",
                )
            ),
            provider=object(),
            pipeline_lock=Lock(),
        )

        payloads = {
            "ingestion_manifest": {
                "documents": [
                    {"doc_name": "story.pdf", "content_hash": "abc123"},
                ]
            },
            "chunk_manifest": {
                "documents": [
                    {
                        "doc_name": "story.pdf",
                        "content_hash": "abc123",
                        "chunk_file_path": "data/chunks/story.jsonl",
                        "parent_chunk_file_path": "data/chunks/story.parents.jsonl",
                    }
                ]
            },
        }

        rebuild_calls: list[str] = []

        monkeypatch.setattr("api.dependencies.CacheStore", lambda _path: FakeCache(payloads))
        monkeypatch.setattr("api.dependencies.ingest_documents", lambda config, verbose=False: rebuild_calls.append("ingest"))
        monkeypatch.setattr("api.dependencies.chunk_documents", lambda config, verbose=False: rebuild_calls.append("chunk"))
        monkeypatch.setattr("api.dependencies.index_documents", lambda config, provider, verbose=False: rebuild_calls.append("index"))

        ensure_corpus_ready(services)

        assert rebuild_calls == ["ingest", "chunk", "index"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_ensure_corpus_ready_skips_when_manifests_and_files_match(monkeypatch):
    root = Path.cwd() / ".tmp_test_api_dependencies" / f"case-{uuid4().hex}"
    try:
        child = root / "data" / "chunks" / "story.jsonl"
        parent = root / "data" / "chunks" / "story.parents.jsonl"
        parent.parent.mkdir(parents=True, exist_ok=True)
        child.write_text("{}", encoding="utf-8")
        parent.write_text("{}", encoding="utf-8")

        services = SimpleNamespace(
            config=SimpleNamespace(
                paths=SimpleNamespace(
                    root=root,
                    cache_dir=root / "data" / "cache",
                )
            ),
            provider=object(),
            pipeline_lock=Lock(),
        )

        payloads = {
            "ingestion_manifest": {
                "documents": [
                    {"doc_name": "story.pdf", "content_hash": "abc123"},
                ]
            },
            "chunk_manifest": {
                "documents": [
                    {
                        "doc_name": "story.pdf",
                        "content_hash": "abc123",
                        "chunk_file_path": "data/chunks/story.jsonl",
                        "parent_chunk_file_path": "data/chunks/story.parents.jsonl",
                    }
                ]
            },
        }

        rebuild_calls: list[str] = []

        monkeypatch.setattr("api.dependencies.CacheStore", lambda _path: FakeCache(payloads))
        monkeypatch.setattr("api.dependencies.ingest_documents", lambda config, verbose=False: rebuild_calls.append("ingest"))
        monkeypatch.setattr("api.dependencies.chunk_documents", lambda config, verbose=False: rebuild_calls.append("chunk"))
        monkeypatch.setattr("api.dependencies.index_documents", lambda config, provider, verbose=False: rebuild_calls.append("index"))

        ensure_corpus_ready(services)

        assert rebuild_calls == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_task_scheduler_prioritizes_interactive_work():
    scheduler = TaskScheduler(worker_count=1)
    try:
        gate = Lock()
        gate.acquire()
        execution_order: list[str] = []

        def slow_eval():
            gate.acquire()
            execution_order.append("eval")

        first_future = scheduler.submit(slow_eval, priority=PRIORITY_BENCHMARK)
        time.sleep(0.05)

        second_future = scheduler.submit(
            lambda: execution_order.append("ask"),
            priority=PRIORITY_INTERACTIVE,
        )

        gate.release()
        first_future.result(timeout=1.0)
        second_future.result(timeout=1.0)

        assert execution_order == ["eval", "ask"]

        execution_order.clear()
        second_gate = Lock()
        second_gate.acquire()

        first_query = scheduler.submit(lambda: execution_order.append("ask-1"), priority=PRIORITY_INTERACTIVE)
        blocking_eval = scheduler.submit(
            lambda: (second_gate.acquire(), execution_order.append("eval-2")),
            priority=PRIORITY_BENCHMARK,
        )
        second_query = scheduler.submit(lambda: execution_order.append("ask-2"), priority=PRIORITY_INTERACTIVE)

        first_query.result(timeout=1.0)
        second_query.result(timeout=1.0)
        second_gate.release()
        blocking_eval.result(timeout=1.0)

        assert execution_order == ["ask-1", "ask-2", "eval-2"]
    finally:
        scheduler.shutdown()


def test_queue_corpus_job_reset_runs_reset_under_pipeline_lock(monkeypatch):
    class InlineScheduler:
        def submit(self, fn, *, priority):
            fn()
            return None

    lock = Lock()
    services = SimpleNamespace(
        config=SimpleNamespace(paths=SimpleNamespace(root=Path.cwd(), corpus_id="hr-resources")),
        scheduler=InlineScheduler(),
        jobs={},
        pipeline_lock=lock,
    )

    monkeypatch.setattr("api.dependencies.activate_corpus", lambda root, corpus_id: None)
    monkeypatch.setattr(
        "api.dependencies.new_job",
        lambda _services, *, kind, filename=None: {
            "job_id": "job-reset",
            "kind": kind,
            "filename": filename,
            "status": "queued",
            "created_at": "2026-05-03T00:00:00Z",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "progress": 0.0,
        },
    )

    observed = {"locked_during_reset": False}

    def fake_reset(_config):
        observed["locked_during_reset"] = lock.locked()

    monkeypatch.setattr("api.dependencies.reset_corpus_artifacts", fake_reset)
    monkeypatch.setattr("api.dependencies._run_pipeline_steps", lambda _services, _job: None)
    monkeypatch.setattr("api.dependencies.update_corpus_stats", lambda *args, **kwargs: None)

    queue_corpus_job(services, kind="reset", corpus_id="hr-resources")

    assert observed["locked_during_reset"] is True


def test_list_documents_computes_health_once(monkeypatch):
    services = SimpleNamespace(
        config=SimpleNamespace(paths=SimpleNamespace(cache_dir=Path.cwd() / "cache", root=Path.cwd(), raw_docs_dir=Path.cwd())),
    )

    payloads = {
        "ingestion_manifest": {
            "documents": [
                {"doc_name": "one.pdf", "last_ingested_at": "2026-05-03T00:00:00Z", "warnings": []},
                {"doc_name": "two.pdf", "last_ingested_at": "2026-05-03T00:00:00Z", "warnings": []},
            ]
        }
    }

    health_calls = {"count": 0}

    monkeypatch.setattr("api.dependencies.CacheStore", lambda _path: FakeCache(payloads))
    monkeypatch.setattr("api.dependencies.load_all_chunks", lambda _config: [])

    def fake_health(_config):
        health_calls["count"] += 1
        return {"drift_state": "current"}

    monkeypatch.setattr("api.dependencies.build_corpus_health", fake_health)

    documents = list_documents(services)

    assert len(documents) == 2
    assert health_calls["count"] == 1


def test_build_recommendations_uses_hr_metadata(monkeypatch):
    services = SimpleNamespace(
        config=SimpleNamespace(paths=SimpleNamespace(corpus_id="hr-resources", cache_dir=Path.cwd() / "cache")),
    )
    payloads = FakeCache(
        {
            "ingestion_manifest": {
                "documents": [
                    {
                        "doc_name": "03_UW_System/UWSystem-1213-FMLA-2026.pdf",
                        "policy_category": "leave",
                        "source_level": "UW System",
                    },
                    {
                        "doc_name": "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf",
                        "policy_category": "leave",
                        "source_level": "Federal",
                    },
                ]
            }
        }
    )

    monkeypatch.setattr("api.dependencies.CacheStore", lambda _path: payloads)

    recommendations = build_recommendations(services)

    assert recommendations.corpus_id == "hr-resources"
    assert recommendations.recommendations
    assert any("FMLA" in item or "leave" in item.lower() or "UW System" in item for item in recommendations.recommendations)
