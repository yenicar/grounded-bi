from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import main as api_main
from api.dependencies import build_ask_response, require_user
from api.models import AskResponse, Citation, CurrentUser, DocumentRecord, EvalReportResponse, EvalReportSummary, JobRecord
from api.routers import corpora as corpora_router
from api.routers import documents as documents_router
from api.routers import eval as eval_router
from api.routers import export as export_router
from api.routers import query as query_router
from api.routers import search as search_router


class ImmediateScheduler:
    def submit(self, fn, *, priority):
        future = Future()
        try:
            result = fn()
        except Exception as exc:  # pragma: no cover - test helper
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self):
        return None


def _make_client(monkeypatch, *, authenticated: bool = True) -> tuple[TestClient, object]:
    services = SimpleNamespace(scheduler=ImmediateScheduler())
    monkeypatch.setattr(api_main, "build_services", lambda: services)

    api_main.app.dependency_overrides[query_router.get_services] = lambda: services
    api_main.app.dependency_overrides[corpora_router.get_services] = lambda: services
    api_main.app.dependency_overrides[documents_router.get_services] = lambda: services
    api_main.app.dependency_overrides[eval_router.get_services] = lambda: services
    api_main.app.dependency_overrides[search_router.get_services] = lambda: services
    if authenticated:
        api_main.app.dependency_overrides[require_user] = lambda: CurrentUser(
            subject="user-123",
            provider="google",
            email="analyst@example.com",
            name="Analyst",
        )

    client = TestClient(api_main.app)
    return client, services


def test_healthcheck_ok(monkeypatch):
    client, _services = _make_client(monkeypatch, authenticated=False)
    try:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_cors_origins_include_local_defaults_and_env(monkeypatch):
    monkeypatch.setenv("TRBI_WEB_ORIGINS", "https://groundedbi.example.com, https://ops.example.com")
    assert api_main._cors_origins() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://groundedbi.example.com",
        "https://ops.example.com",
    ]


def test_auth_me_reports_unauthenticated_when_session_missing(monkeypatch):
    client, _services = _make_client(monkeypatch, authenticated=False)
    try:
        response = client.get("/auth/me")
        assert response.status_code == 200
        assert response.json()["authenticated"] is False
        assert "dev" in response.json()["available_providers"]
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_dev_login_creates_local_session(monkeypatch):
    client, _services = _make_client(monkeypatch, authenticated=False)
    try:
        response = client.get("/auth/login/dev", follow_redirects=False)
        assert response.status_code == 302

        session_response = client.get("/auth/me")
        assert session_response.status_code == 200
        payload = session_response.json()
        assert payload["authenticated"] is True
        assert payload["user"]["provider"] == "dev"
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_protected_route_requires_authentication(monkeypatch):
    client, _services = _make_client(monkeypatch, authenticated=False)
    try:
        response = client.get("/api/documents")
        assert response.status_code == 401
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_ask_endpoint_returns_frontend_contract(monkeypatch):
    client, services = _make_client(monkeypatch)

    def fake_run_question(_services, question: str, *, debug: bool, history=None, grounding_enabled=None, after=None):
        assert _services is services
        assert question == "What changed?"
        assert debug is True
        assert history is None
        assert grounding_enabled is True
        assert after is None
        return AskResponse(
            answer="Grounded answer",
            citations=[Citation(doc_name="brief.pdf", page_range="p.2", excerpt="Evidence excerpt")],
            confidence="High",
            metrics={"total_seconds": 1.2, "total_tokens": 42},
            retrieval_question="What changed?",
            retrieval_debug={"selected": []},
        )

    monkeypatch.setattr(query_router, "run_question", fake_run_question)
    monkeypatch.setattr(query_router, "log_query_event", lambda *args, **kwargs: None)

    try:
        response = client.post(
            "/api/query/ask",
            json={"question": "What changed?", "debug": True, "grounding_enabled": True},
        )
        assert response.status_code == 200
        assert response.json() == {
            "answer": "Grounded answer",
            "citations": [
                {
                    "doc_name": "brief.pdf",
                    "page_range": "p.2",
                    "excerpt": "Evidence excerpt",
                }
            ],
            "confidence": "High",
            "metrics": {"total_seconds": 1.2, "total_tokens": 42},
            "retrieval_question": "What changed?",
            "retrieval_debug": {"selected": []},
        }
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_chat_endpoint_appends_history(monkeypatch):
    client, services = _make_client(monkeypatch)

    def fake_run_question(_services, question: str, *, debug: bool, history=None, grounding_enabled=None, after=None):
        assert _services is services
        assert question == "Follow up"
        assert debug is False
        assert history == [{"user": "First", "assistant": "Answer one"}]
        assert grounding_enabled is True
        assert after is None
        return AskResponse(
            answer="Answer two",
            citations=[],
            confidence="Medium",
            metrics={"total_seconds": 0.8},
            retrieval_question="Follow up",
            retrieval_debug=None,
        )

    monkeypatch.setattr(query_router, "run_question", fake_run_question)
    monkeypatch.setattr(query_router, "log_query_event", lambda *args, **kwargs: None)

    try:
        response = client.post(
            "/api/query/chat",
            json={
                "message": "Follow up",
                "history": [{"user": "First", "assistant": "Answer one"}],
                "debug": False,
                "grounding_enabled": True,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == "Answer two"
        assert payload["history"] == [
            {"user": "First", "assistant": "Answer one"},
            {"user": "Follow up", "assistant": "Answer two"},
        ]
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_cached_response_metrics_report_current_run_cost():
    response = build_ask_response(
        {
            "answer": "Cached answer\n\nConfidence: High",
            "citations": [],
            "retrieval_debug": {
                "selected": [],
                "grounding": {
                    "verdict": "trimmed",
                    "unsupported_claims_removed": ["old cold-run claim"],
                },
            },
            "cache_stats": {"answer_cache": "hit"},
            "metrics": {
                "total_seconds": 29.754,
                "grounding_seconds": 19.147,
                "prompt_tokens": 9000,
                "completion_tokens": 1200,
                "total_tokens": 10577,
                "remote_calls": 0,
                "grounding_verdict": "trimmed",
                "unsupported_claims_removed": 2,
            },
            "retrieval_question": "Where do policies align or differ?",
        },
        include_debug=True,
        api_elapsed=0.041,
    )

    assert response.metrics["total_seconds"] == 0.041
    assert response.metrics["grounding_seconds"] == 0.0
    assert response.metrics["total_tokens"] == 0
    assert response.metrics["remote_calls"] == 0
    assert response.metrics["cache_served"] is True
    assert response.metrics["cold_run_metrics"]["total_seconds"] == 29.754


def test_documents_endpoint_exposes_typed_jobs(monkeypatch):
    client, services = _make_client(monkeypatch)

    monkeypatch.setattr(
        documents_router,
        "list_documents",
        lambda _services: [
            DocumentRecord(
                doc_name="brief.pdf",
                doc_format="pdf",
                page_count=12,
                table_count=2,
                last_ingested_at="2026-04-22T12:00:00Z",
                warnings=[],
                doc_date="2026-04-22",
                doc_author="Analyst",
                doc_type="report",
            )
        ],
    )
    monkeypatch.setattr(
        documents_router,
        "list_active_jobs",
        lambda _services: [
            {
                "job_id": "job123",
                "kind": "upload",
                "filename": "brief.pdf",
                "status": "running",
                "created_at": "2026-04-22T12:01:00Z",
                "started_at": "2026-04-22T12:01:05Z",
            }
        ],
    )
    monkeypatch.setattr(
        documents_router,
        "list_recent_jobs",
        lambda _services: [
            JobRecord(
                job_id="job122",
                kind="upload",
                filename="older.pdf",
                status="completed",
                created_at="2026-04-22T11:55:00Z",
                started_at="2026-04-22T11:55:05Z",
                finished_at="2026-04-22T11:56:00Z",
                error=None,
            ).model_dump()
        ],
    )

    try:
        response = client.get("/api/documents")
        assert response.status_code == 200
        payload = response.json()
        assert payload["documents"][0]["doc_name"] == "brief.pdf"
        assert payload["active_jobs"][0]["job_id"] == "job123"
        assert payload["recent_jobs"][0]["status"] == "completed"
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_corpus_endpoints_expose_registry_and_jobs(monkeypatch):
    client, services = _make_client(monkeypatch)

    monkeypatch.setattr(
        "api.routers.corpora.list_corpora_payload",
        lambda _services: {
            "active_corpus_id": "hr-resources",
            "corpora": [
                {
                    "id": "hr-resources",
                    "name": "HR Resources",
                    "source_path": "data/raw_pdfs/HR Resources",
                    "active": True,
                    "status": "ready",
                    "document_count": 2,
                    "chunk_count": 9,
                    "last_indexed_at": None,
                    "last_error": None,
                    "drift_state": "current",
                    "source_file_count": 2,
                    "active_jobs": [],
                    "recent_jobs": [],
                }
            ],
        },
    )
    monkeypatch.setattr(
        "api.routers.corpora.build_corpus_health_response",
            lambda _services, corpus_id: {
                "corpus_id": corpus_id,
                "drift_state": "current",
                "source_path": "data/raw_pdfs/HR Resources",
                "source_file_count": 2,
                "indexed_document_count": 2,
                "chunk_count": 9,
                "parent_chunk_count": 3,
                "missing_source_count": 0,
                "new_source_count": 0,
                "changed_source_count": 0,
                "missing_chunk_file_count": 0,
                "last_indexed_at": None,
                "missing_sources": [],
                "new_sources": [],
                "changed_sources": [],
            },
        )
    monkeypatch.setattr(
        "api.routers.corpora.queue_corpus_job",
        lambda _services, kind, corpus_id: {
            "job_id": "job-corpus",
            "kind": kind,
            "filename": corpus_id,
            "status": "queued",
            "created_at": "2026-05-02T00:00:00Z",
        },
    )

    try:
        response = client.get("/api/corpora")
        assert response.status_code == 200
        assert response.json()["active_corpus_id"] == "hr-resources"

        health = client.get("/api/corpora/hr-resources/health")
        assert health.status_code == 200
        assert health.json()["drift_state"] == "current"

        rebuild = client.post("/api/corpora/hr-resources/rebuild")
        assert rebuild.status_code == 202
        assert rebuild.json()["job_id"] == "job-corpus"
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_eval_endpoints_match_frontend_expectations(monkeypatch):
    client, services = _make_client(monkeypatch)

    monkeypatch.setattr(
        eval_router,
        "new_job",
        lambda _services, *, kind, filename=None: {
            "job_id": "evaljob1",
            "kind": kind,
            "filename": filename,
            "status": "queued",
            "created_at": "2026-04-22T13:00:00Z",
        },
    )
    monkeypatch.setattr(eval_router, "run_eval_job", lambda _services, job_id: None)
    monkeypatch.setattr(eval_router, "list_eval_reports", lambda _services: [
        EvalReportSummary(
            report_id="eval_report_20260422T130000Z",
            generated_at="20260422T130000Z",
            prompt_count=16,
            averages={"faithfulness_llm": 4.2},
        )
    ])
    monkeypatch.setattr(
        eval_router,
        "get_eval_report",
        lambda _services, report_id: EvalReportResponse(
            report_id=report_id,
            generated_at="20260422T130000Z",
            prompt_count=16,
            judge_failures=1,
            averages={"faithfulness_llm": 4.2},
            results=[
                {
                    "question_id": "q1",
                    "question": "What changed?",
                    "category": "test",
                    "result": {"metrics": {"total_seconds": 1.1}, "citations": []},
                    "scores": {"faithfulness_llm": 4, "relevance_llm": 5},
                    "judge": {"reasoning": "Grounded enough"},
                }
            ],
        ),
    )
    monkeypatch.setattr(
        eval_router,
        "get_job",
        lambda _services, job_id: {
            "job_id": job_id,
            "kind": "eval",
            "status": "completed",
            "filename": "benchmark",
            "created_at": "2026-04-22T13:00:00Z",
            "started_at": "2026-04-22T13:00:05Z",
            "finished_at": "2026-04-22T13:02:00Z",
            "error": None,
            "report_id": "eval_report_20260422T130000Z",
        },
    )

    try:
        run_response = client.post("/api/eval/run")
        assert run_response.status_code == 202
        assert run_response.json()["job_id"] == "evaljob1"

        list_response = client.get("/api/eval/reports")
        assert list_response.status_code == 200
        assert list_response.json()[0]["report_id"] == "eval_report_20260422T130000Z"

        report_response = client.get("/api/eval/reports/eval_report_20260422T130000Z")
        assert report_response.status_code == 200
        assert report_response.json()["judge_failures"] == 1

        job_response = client.get("/api/eval/jobs/evaljob1")
        assert job_response.status_code == 200
        assert job_response.json()["report_id"] == "eval_report_20260422T130000Z"
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_search_endpoint_returns_filtered_results(monkeypatch):
    client, services = _make_client(monkeypatch)

    monkeypatch.setattr(
        search_router,
        "search_corpus",
        lambda _services, *, query, source_level, policy_category, limit: {
            "query": query,
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "doc_name": "01_Federal/FMLA.pdf",
                    "source_relative_path": "01_Federal/FMLA.pdf",
                    "source_level": source_level,
                    "policy_category": policy_category,
                    "page_range": "p.1",
                    "excerpt": "FMLA leave evidence",
                    "score": 2.0,
                    "matched_terms": ["fmla"],
                }
            ],
        },
    )

    try:
        response = client.post(
            "/api/search",
            json={"query": "FMLA leave", "source_level": "Federal", "policy_category": "leave"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["query"] == "FMLA leave"
        assert payload["results"][0]["source_level"] == "Federal"
        assert payload["results"][0]["policy_category"] == "leave"
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()


def test_export_endpoints_return_binary_payloads(monkeypatch):
    client, _services = _make_client(monkeypatch)
    monkeypatch.setattr(export_router, "build_pdf_export", lambda payload: b"%PDF-test")
    monkeypatch.setattr(export_router, "build_pptx_export", lambda payload: b"PPTX-test")

    try:
        pdf_response = client.post(
            "/api/export/pdf",
            json={
                "title": "GroundedBI export",
                "question": "What changed?",
                "answer": "Grounded answer",
                "citations": [],
            },
        )
        assert pdf_response.status_code == 200
        assert pdf_response.content == b"%PDF-test"

        pptx_response = client.post(
            "/api/export/pptx",
            json={
                "title": "GroundedBI export",
                "question": "What changed?",
                "answer": "Grounded answer",
                "citations": [],
            },
        )
        assert pptx_response.status_code == 200
        assert pptx_response.content == b"PPTX-test"
    finally:
        client.close()
        api_main.app.dependency_overrides.clear()
