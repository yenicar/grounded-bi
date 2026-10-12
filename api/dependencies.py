from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count
from pathlib import Path
from queue import PriorityQueue
from threading import Lock, Thread
from typing import Any
from uuid import uuid4
import os

from concurrent.futures import Future

from fastapi import Request

from true_rag_bi.audit import append_query_audit_log
from true_rag_bi.cache import CacheStore
from true_rag_bi.chunking import chunk_documents, load_all_chunks
from true_rag_bi.config import AppConfig, build_config
from true_rag_bi.corpus import (
    activate_corpus,
    build_corpus_health,
    get_corpus,
    load_registry,
    reset_corpus_artifacts,
    save_registry,
    update_corpus_stats,
    upsert_corpus,
)
from true_rag_bi.evaluation import run_evaluation
from true_rag_bi.exporters import export_answer_pdf, export_answer_pptx
from true_rag_bi.indexing import index_documents
from true_rag_bi.ingest import ingest_documents
from true_rag_bi.providers import RAGProvider, build_provider
from true_rag_bi.retrieval import ask_question
from true_rag_bi.utils import read_json, tokenize_for_search, utc_now_iso

from .models import AskResponse, Citation, DocumentRecord, EvalReportResponse, EvalReportSummary, JobStatusResponse, SearchResponse, SearchResult, RecommendationResponse
from .auth import load_user_from_request
from .models import CurrentUser

_CONFIDENCE_RE = re.compile(r"\n*Confidence:\s*(High|Medium|Low)\s*$", re.IGNORECASE)
_SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9._ -]+$")
_SAFE_REPORT_RE = re.compile(r"^eval_report_[A-Za-z0-9T_-]+$")
PRIORITY_INTERACTIVE = 0
PRIORITY_PIPELINE = 5
PRIORITY_BENCHMARK = 9


class TaskScheduler:
    def __init__(self, *, worker_count: int = 2) -> None:
        self._queue: PriorityQueue[tuple[int, int, Any, Future[Any] | None]] = PriorityQueue()
        self._counter = count()
        self._threads = [
            Thread(target=self._worker, name=f"groundedbi-worker-{index}", daemon=True)
            for index in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, fn: Any, *, priority: int) -> Future[Any]:
        future: Future[Any] = Future()
        self._queue.put((priority, next(self._counter), fn, future))
        return future

    def shutdown(self) -> None:
        for _ in self._threads:
            self._queue.put((10**9, next(self._counter), None, None))
        for thread in self._threads:
            thread.join(timeout=1.0)

    def _worker(self) -> None:
        while True:
            _priority, _order, fn, future = self._queue.get()
            try:
                if fn is None:
                    return

                assert future is not None
                if not future.set_running_or_notify_cancel():
                    continue

                try:
                    result = fn()
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._queue.task_done()


@dataclass(slots=True)
class AppServices:
    config: AppConfig
    scheduler: TaskScheduler
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    pipeline_lock: Lock = field(default_factory=Lock)
    _provider: RAGProvider | None = None

    @property
    def provider(self) -> RAGProvider:
        if self._provider is None:
            self._provider = build_provider(self.config)
        return self._provider


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return build_config()


@lru_cache(maxsize=1)
def get_provider() -> RAGProvider:
    return build_provider(get_config())


def build_services() -> AppServices:
    worker_count = max(1, int(os.getenv("TRBI_WORKER_CONCURRENCY", "2")))
    return AppServices(
        config=get_config(),
        scheduler=TaskScheduler(worker_count=worker_count),
    )


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def get_current_user(request: Request) -> CurrentUser | None:
    return load_user_from_request(request)


def require_user(request: Request) -> CurrentUser:
    user = get_current_user(request)
    if user is None:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return user


def new_job(services: AppServices, *, kind: str, filename: str | None = None) -> dict[str, Any]:
    job_id = uuid4().hex[:12]
    payload = {
        "job_id": job_id,
        "kind": kind,
        "filename": filename,
        "status": "queued",
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "progress": 0.0,
    }
    services.jobs[job_id] = payload
    return payload


def list_active_jobs(services: AppServices) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "filename": job.get("filename"),
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "progress": job.get("progress", 0.0),
        }
        for job in services.jobs.values()
        if job["status"] in {"queued", "running"}
    ]


def list_recent_jobs(services: AppServices, *, limit: int = 8) -> list[dict[str, Any]]:
    jobs = sorted(
        services.jobs.values(),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )
    return [
        {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "filename": job.get("filename"),
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job.get("finished_at"),
            "error": job.get("error"),
            "progress": job.get("progress", 0.0),
        }
        for job in jobs[:limit]
    ]


def get_job(services: AppServices, job_id: str) -> JobStatusResponse:
    job = services.jobs.get(job_id)
    if not job:
        raise FileNotFoundError(job_id)
    return JobStatusResponse(
        job_id=job["job_id"],
        kind=job["kind"],
        status=job["status"],
        filename=job.get("filename"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        error=job.get("error"),
        report_id=job.get("report_id"),
        progress=job.get("progress", 0.0),
    )


def _has_pipeline_drift(services: AppServices) -> bool:
    if hasattr(services.config.paths, "raw_docs_dir") and hasattr(services.config.paths, "corpus_id"):
        return build_corpus_health(services.config)["drift_state"] != "current"

    cache = CacheStore(services.config.paths.cache_dir)
    ingestion_manifest = cache.load("ingestion_manifest", {"documents": []})
    chunk_manifest = cache.load("chunk_manifest", {"documents": []})

    ingestion_docs = {
        record["doc_name"]: record
        for record in ingestion_manifest.get("documents", [])
    }
    chunk_docs = {
        record["doc_name"]: record
        for record in chunk_manifest.get("documents", [])
    }

    if set(ingestion_docs) != set(chunk_docs):
        return True

    for doc_name, ingest_record in ingestion_docs.items():
        chunk_record = chunk_docs.get(doc_name)
        if not chunk_record:
            return True

        if chunk_record.get("content_hash") != ingest_record.get("content_hash"):
            return True

        child_chunk_path = services.config.paths.root / chunk_record.get("chunk_file_path", "")
        parent_chunk_path = services.config.paths.root / chunk_record.get("parent_chunk_file_path", "")
        if not child_chunk_path.exists() or not parent_chunk_path.exists():
            return True

    return False


def ensure_corpus_ready(services: AppServices) -> None:
    if not _has_pipeline_drift(services):
        return

    with services.pipeline_lock:
        if not _has_pipeline_drift(services):
            return
        ingest_documents(services.config, verbose=False)
        chunk_documents(services.config, verbose=False)
        index_documents(services.config, services.provider, verbose=False)


def _run_pipeline_steps(services: AppServices, job: dict[str, Any]) -> None:
    job["progress"] = 10.0
    ingest_documents(services.config, verbose=False)
    job["progress"] = 40.0
    chunk_documents(services.config, verbose=False)
    job["progress"] = 70.0
    index_documents(services.config, services.provider, verbose=False)
    job["progress"] = 100.0
    update_corpus_stats(services.config, status="ready")


def run_pipeline_job(services: AppServices, job_id: str) -> None:
    job = services.jobs[job_id]
    job["status"] = "running"
    job["started_at"] = utc_now_iso()
    try:
        with services.pipeline_lock:
            _run_pipeline_steps(services, job)
        job["status"] = "completed"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        update_corpus_stats(services.config, status="failed", last_error=str(exc))
    finally:
        job["finished_at"] = utc_now_iso()


def run_eval_job(services: AppServices, job_id: str) -> None:
    job = services.jobs[job_id]
    job["status"] = "running"
    job["started_at"] = utc_now_iso()
    try:
        job["progress"] = 5.0
        summary = run_evaluation(services.config, services.provider, verbose=False)
        job["progress"] = 100.0
        job["status"] = "completed"
        job["report_id"] = f"eval_report_{summary['generated_at']}"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["finished_at"] = utc_now_iso()


def _page_range(start: int, end: int) -> str:
    if start == end:
        return f"p.{start}"
    return f"pp.{start}-{end}"


def _split_confidence(answer: str) -> tuple[str, str]:
    match = _CONFIDENCE_RE.search(answer)
    if not match:
        cleaned = answer.strip()
        lowered = cleaned.lower()
        if "insufficient" in lowered or "could not find enough grounded evidence" in lowered:
            return cleaned, "Low"
        return cleaned, "Medium"
    cleaned = _CONFIDENCE_RE.sub("", answer).strip()
    confidence = match.group(1).title()
    return cleaned, confidence


def _require_safe_leaf_name(value: str, *, label: str, pattern: re.Pattern[str] | None = None) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{label} is required.")
    if Path(candidate).name != candidate:
        raise ValueError(f"{label} must be a leaf filename.")
    if "\\" in candidate or "/" in candidate:
        raise ValueError(f"{label} must not contain path separators.")
    if ".." in candidate:
        raise ValueError(f"{label} must not contain parent-directory traversal.")
    active_pattern = pattern or _SAFE_LEAF_RE
    if not active_pattern.fullmatch(candidate):
        raise ValueError(f"{label} contains unsupported characters.")
    return candidate


def _build_citations(result: dict[str, Any]) -> list[Citation]:
    selected = result.get("retrieval_debug", {}).get("selected", [])
    citations: list[Citation] = []
    for item in selected:
        excerpt = (item.get("text", "") or "").strip().replace("\n", " ")
        citations.append(
            Citation(
                doc_name=item["doc_name"],
                page_range=_page_range(int(item["page_start"]), int(item["page_end"])),
                excerpt=excerpt[:200],
                source_relative_path=item.get("source_relative_path"),
                source_level=item.get("source_level"),
                policy_category=item.get("policy_category"),
            )
        )
    return citations


def _normalize_current_run_metrics(result: dict[str, Any], *, api_elapsed: float | None = None) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    is_cached = bool(
        metrics.get("cache_served")
        or metrics.get("cached_response")
        or result.get("cache_stats", {}).get("answer_cache") == "hit"
        or metrics.get("remote_calls") == 0
    )

    if api_elapsed is not None:
        metrics["api_response_seconds"] = round(api_elapsed, 3)

    if is_cached:
        cold_metrics = dict(metrics.get("cold_run_metrics") or {})
        for key in (
            "retrieval_seconds",
            "generation_seconds",
            "grounding_seconds",
            "total_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "remote_calls",
            "prompt_context_tokens",
            "grounding_verdict",
            "unsupported_claims_removed",
        ):
            if key in metrics and key not in cold_metrics:
                cold_metrics[key] = metrics[key]

        current_elapsed = round(
            api_elapsed
            if api_elapsed is not None
            else float(metrics.get("cache_lookup_seconds") or metrics.get("total_seconds") or 0.0),
            3,
        )
        metrics.update(
            {
                "retrieval_seconds": current_elapsed,
                "generation_seconds": 0.0,
                "grounding_seconds": 0.0,
                "total_seconds": current_elapsed,
                "remote_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_served": True,
                "cached_response": True,
                "cold_run_metrics": cold_metrics,
            }
        )
        metrics.setdefault("grounding_verdict", "cached")
        metrics.setdefault("unsupported_claims_removed", 0)

    return metrics


def build_ask_response(result: dict[str, Any], *, include_debug: bool, api_elapsed: float | None = None) -> AskResponse:
    answer, confidence = _split_confidence(result["answer"])
    return AskResponse(
        answer=answer,
        citations=_build_citations(result),
        confidence=confidence,  # type: ignore[arg-type]
        metrics=_normalize_current_run_metrics(result, api_elapsed=api_elapsed),
        retrieval_question=result.get("retrieval_question", result.get("question", "")),
        retrieval_debug=result.get("retrieval_debug") if include_debug else None,
    )


def run_question(
    services: AppServices,
    question: str,
    *,
    debug: bool,
    history: list[dict[str, str]] | None = None,
    grounding_enabled: bool | None = None,
    after: str | None = None,
) -> AskResponse:
    started = time.perf_counter()
    future = services.scheduler.submit(
        lambda: _run_question_task(
            services,
            question,
            history=history,
            grounding_enabled=grounding_enabled,
            after=after,
        ),
        priority=PRIORITY_INTERACTIVE,
    )
    result = future.result()
    return build_ask_response(result, include_debug=debug, api_elapsed=time.perf_counter() - started)


def _run_question_task(
    services: AppServices,
    question: str,
    *,
    history: list[dict[str, str]] | None,
    grounding_enabled: bool | None,
    after: str | None,
) -> dict[str, Any]:
    ensure_corpus_ready(services)
    return ask_question(
        services.config,
        services.provider,
        question,
        debug=True,
        verbose=False,
        history=history,
        grounding_enabled=grounding_enabled,
        after=after,
    )


def list_documents(services: AppServices) -> list[DocumentRecord]:
    cache = CacheStore(services.config.paths.cache_dir)
    manifest = cache.load("ingestion_manifest", {"documents": []})
    summary_cache = cache.load("summary_augmented_cache", {"documents": {}})
    document_summaries = summary_cache.get("documents", {})
    # Optimization: Use table_count from manifest if available, fallback to 0
    # instead of loading all chunks (which is very slow for large corpora).
    drift_status = (
        build_corpus_health(services.config).get("drift_state")
        if hasattr(services.config.paths, "raw_docs_dir")
        else None
    )
    documents = []
    for record in manifest.get("documents", []):
        documents.append(
            DocumentRecord(
                doc_name=record["doc_name"],
                corpus_id=getattr(services.config.paths, "corpus_id", None),
                doc_format=record.get("doc_format", Path(record["doc_name"]).suffix.lstrip(".").lower()),
                page_count=int(record.get("page_count", 0)),
                table_count=record.get("table_count", 0),
                last_ingested_at=record.get("last_ingested_at", ""),
                warnings=list(record.get("warnings", [])),
                source_relative_path=record.get("source_relative_path"),
                source_level=record.get("source_level"),
                policy_category=record.get("policy_category"),
                effective_date=record.get("effective_date"),
                drift_status=drift_status,
                doc_date=record.get("doc_date"),
                doc_author=record.get("doc_author"),
                doc_type=record.get("doc_type"),
                doc_summary=record.get("doc_summary") or document_summaries.get(record["doc_name"]),
            )
        )
    documents.sort(key=lambda item: item.doc_name.lower())
    return documents


def list_corpora_payload(services: AppServices) -> dict[str, Any]:
    registry = load_registry(services.config.paths.root)
    active_jobs = list_active_jobs(services)
    recent_jobs = list_recent_jobs(services)
    corpora = []
    for corpus in registry.get("corpora", []):
        is_active = corpus.get("id") == registry.get("active_corpus_id")
        health = build_corpus_health(services.config) if is_active else {}
        corpora.append(
            {
                "id": corpus["id"],
                "name": corpus.get("name", corpus["id"]),
                "source_path": corpus.get("source_path", ""),
                "active": is_active,
                "status": corpus.get("status", "new"),
                "document_count": int(health.get("indexed_document_count", corpus.get("document_count", 0)) or 0),
                "chunk_count": int(health.get("chunk_count", corpus.get("chunk_count", 0)) or 0),
                "last_indexed_at": health.get("last_indexed_at") or corpus.get("last_indexed_at"),
                "last_error": corpus.get("last_error"),
                "drift_state": health.get("drift_state", "unknown" if not is_active else "stale"),
                "source_file_count": int(health.get("source_file_count", 0) or 0),
                "active_jobs": active_jobs if is_active else [],
                "recent_jobs": recent_jobs if is_active else [],
            }
        )
    return {"active_corpus_id": registry.get("active_corpus_id", ""), "corpora": corpora}


def build_corpus_health_response(services: AppServices, corpus_id: str) -> dict[str, Any]:
    if corpus_id != services.config.paths.corpus_id:
        get_corpus(services.config.paths.root, corpus_id)
        payload = list_corpora_payload(services)
        corpus = next(item for item in payload["corpora"] if item["id"] == corpus_id)
        return {
            "corpus_id": corpus_id,
            "drift_state": corpus["drift_state"],
            "source_path": corpus["source_path"],
            "source_file_count": corpus["source_file_count"],
            "generated_document_count": corpus["document_count"],
            "indexed_document_count": corpus["document_count"],
            "chunk_count": corpus["chunk_count"],
            "parent_chunk_count": 0,
            "missing_source_count": 0,
            "new_source_count": 0,
            "changed_source_count": 0,
            "missing_chunk_file_count": 0,
            "last_indexed_at": corpus["last_indexed_at"],
            "missing_sources": [],
            "new_sources": [],
            "changed_sources": [],
        }
    return build_corpus_health(services.config)


def create_corpus(services: AppServices, *, name: str, source_path: str, corpus_id: str | None = None) -> dict[str, Any]:
    return upsert_corpus(services.config.paths.root, name=name, source_path=source_path, corpus_id=corpus_id)


def activate_corpus_for_services(services: AppServices, corpus_id: str) -> dict[str, Any]:
    return activate_corpus(services.config.paths.root, corpus_id)


def replace_corpus_source(services: AppServices, corpus_id: str, source_path: str) -> dict[str, Any]:
    corpus = get_corpus(services.config.paths.root, corpus_id)
    registry = load_registry(services.config.paths.root)
    updated = upsert_corpus(
        services.config.paths.root,
        name=corpus.get("name", corpus_id),
        source_path=source_path,
        corpus_id=corpus_id,
    )
    for item in registry.get("corpora", []):
        if item.get("id") == corpus_id and item.get("active"):
            activate_corpus(services.config.paths.root, corpus_id)
            break
    return updated


def queue_corpus_job(services: AppServices, *, kind: str, corpus_id: str) -> dict[str, Any]:
    if corpus_id != services.config.paths.corpus_id:
        activate_corpus(services.config.paths.root, corpus_id)
    job = new_job(services, kind=kind, filename=corpus_id)

    def task() -> None:
        if kind != "reset":
            run_pipeline_job(services, job["job_id"])
            return

        job["status"] = "running"
        job["started_at"] = utc_now_iso()
        try:
            with services.pipeline_lock:
                job["progress"] = 5.0
                reset_corpus_artifacts(services.config)
                _run_pipeline_steps(services, job)
            job["status"] = "completed"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            update_corpus_stats(services.config, status="failed", last_error=str(exc))
        finally:
            job["finished_at"] = utc_now_iso()

    services.scheduler.submit(task, priority=PRIORITY_PIPELINE)
    return job


def search_corpus(services: AppServices, *, query: str, source_level: str | None, policy_category: str | None, limit: int) -> SearchResponse:
    ensure_corpus_ready(services)
    keyword_index = read_json(services.config.paths.index_dir / "keyword_index.json", {})
    parent_lookup = read_json(services.config.paths.index_dir / "parent_lookup.json", {}).get("parents", {})
    terms = [term for term in tokenize_for_search(query) if len(term) > 2]
    results: list[SearchResult] = []
    for document in keyword_index.get("documents", []):
        if source_level and (document.get("source_level") or "").lower() != source_level.lower():
            continue
        if policy_category and (document.get("policy_category") or "").lower() != policy_category.lower():
            continue
        term_frequencies = document.get("term_frequencies", {})
        matched = [term for term in terms if term_frequencies.get(term)]
        if not matched:
            continue
        parent = parent_lookup.get(document.get("parent_chunk_id"), {})
        text = (parent.get("chunk_text") or "").replace("\n", " ").strip()
        score = float(sum(term_frequencies.get(term, 0) for term in matched))
        results.append(
            SearchResult(
                chunk_id=document["chunk_id"],
                doc_name=document["doc_name"],
                source_relative_path=document.get("source_relative_path"),
                source_level=document.get("source_level"),
                policy_category=document.get("policy_category"),
                page_range=_page_range(int(document["page_start"]), int(document["page_end"])),
                excerpt=text[:300],
                score=score,
                matched_terms=matched[:8],
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return SearchResponse(query=query, results=results[:limit])


def remove_document_artifacts(services: AppServices, doc_name: str) -> None:
    paths = services.config.paths
    manifest = read_json(paths.cache_dir / "ingestion_manifest.json", {"documents": []})
    target = next(
        (
            record
            for record in manifest.get("documents", [])
            if record.get("doc_name") == doc_name or record.get("source_relative_path") == doc_name
        ),
        None,
    )
    if target is None:
        raise FileNotFoundError(doc_name)

    source = paths.raw_docs_dir / target.get("source_relative_path", target["doc_name"])
    if source.exists():
        source.unlink()

    artifact_id = target.get("artifact_id") or Path(target["doc_name"]).stem
    for path in (
        paths.normalized_md_dir / f"{artifact_id}.md",
        paths.chunks_dir / f"{artifact_id}.jsonl",
        paths.chunks_dir / f"{artifact_id}.parents.jsonl",
    ):
        if path.exists():
            path.unlink()


def list_eval_reports(services: AppServices) -> list[EvalReportSummary]:
    reports: list[EvalReportSummary] = []
    for path in sorted(services.config.paths.eval_dir.glob("eval_report_*.json"), reverse=True):
        payload = read_json(path, {})
        if not payload:
            continue
        reports.append(
            EvalReportSummary(
                report_id=path.stem,
                generated_at=payload.get("generated_at", path.stem.removeprefix("eval_report_")),
                prompt_count=int(payload.get("prompt_count", 0)),
                averages={key: float(value) for key, value in payload.get("averages", {}).items()},
            )
        )
    return reports


def get_eval_report(services: AppServices, report_id: str) -> EvalReportResponse:
    safe_report_id = _require_safe_leaf_name(
        report_id,
        label="Report ID",
        pattern=_SAFE_REPORT_RE,
    )
    path = services.config.paths.eval_dir / f"{safe_report_id}.json"
    payload = read_json(path, {})
    if not payload:
        raise FileNotFoundError(safe_report_id)
    return EvalReportResponse(
        report_id=safe_report_id,
        generated_at=payload.get("generated_at", ""),
        prompt_count=int(payload.get("prompt_count", 0)),
        judge_failures=int(payload.get("judge_failures", 0)),
        averages={key: float(value) for key, value in payload.get("averages", {}).items()},
        results=list(payload.get("results", [])),
    )


def run_eval(services: AppServices) -> EvalReportResponse:
    summary = run_evaluation(services.config, services.provider, verbose=False)
    report_id = f"eval_report_{summary['generated_at']}"
    return EvalReportResponse(
        report_id=report_id,
        generated_at=summary["generated_at"],
        prompt_count=int(summary["prompt_count"]),
        judge_failures=int(summary.get("judge_failures", 0)),
        averages={key: float(value) for key, value in summary["averages"].items()},
        results=list(summary["results"]),
    )


def log_query_event(
    services: AppServices,
    *,
    user: CurrentUser,
    route: str,
    question: str,
    response: AskResponse,
) -> None:
    append_query_audit_log(
        services.config.paths,
        user=user.model_dump(),
        route=route,
        question=question,
        retrieval_question=response.retrieval_question,
        citations=[citation.model_dump() for citation in response.citations],
        metrics=dict(response.metrics),
    )


def build_pdf_export(payload: dict[str, Any]) -> bytes:
    return export_answer_pdf(payload)


def build_pptx_export(payload: dict[str, Any]) -> bytes:
    return export_answer_pptx(payload)


def build_recommendations(services: AppServices) -> RecommendationResponse:
    corpus_id = services.config.paths.corpus_id
    cache = CacheStore(services.config.paths.cache_dir)
    manifest = cache.load("ingestion_manifest", {"documents": []})

    docs = manifest.get("documents", [])
    if not docs:
        return RecommendationResponse(
            recommendations=[
                "Show me the HR policies currently loaded in this assistant.",
                "What topics are covered across the current HR documents?",
                "Which policies should I review first for leave, accommodations, or conduct?"
            ],
            corpus_id=corpus_id
        )

    categories = sorted({d.get("policy_category") for d in docs if d.get("policy_category")})
    source_levels = sorted({d.get("source_level") for d in docs if d.get("source_level")})
    first_doc = docs[0]["doc_name"]

    recs: list[str] = []
    if "leave" in categories and {"Federal", "UW System"} & set(source_levels):
        recs.append("Compare Federal and UW System leave guidance for FMLA, sick leave, and related absences.")
    if categories:
        recs.append(f"Summarize the main HR guidance for {categories[0]} across the current corpus.")
    if len(source_levels) >= 2:
        recs.append(f"Where do {source_levels[0]} and {source_levels[1]} policies align or differ?")

    recs.extend([
        "What policy topics are covered in this HR corpus, and which areas have the strongest support?",
        f"Give me a plain-language summary of {first_doc}.",
        "Which documents should I cite first when answering an employee question about policy requirements?",
    ])

    return RecommendationResponse(
        recommendations=recs[:5],
        corpus_id=corpus_id
    )
