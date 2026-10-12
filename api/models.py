from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    debug: bool = False
    grounding_enabled: bool | None = None
    after: date | None = None


class CurrentUser(BaseModel):
    subject: str
    provider: Literal["google", "github", "dev"]
    email: str | None = None
    name: str | None = None
    avatar_url: str | None = None


class AuthSessionResponse(BaseModel):
    authenticated: bool
    user: CurrentUser | None = None
    available_providers: list[Literal["google", "github", "dev"]] = Field(default_factory=list)


class Citation(BaseModel):
    doc_name: str
    page_range: str
    excerpt: str
    source_relative_path: str | None = None
    source_level: str | None = None
    policy_category: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["High", "Medium", "Low"]
    metrics: dict[str, Any]
    retrieval_question: str
    retrieval_debug: dict[str, Any] | None = None


class ChatTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    debug: bool = False
    grounding_enabled: bool | None = None
    after: date | None = None


class ChatResponse(AskResponse):
    history: list[ChatTurn]


class DocumentRecord(BaseModel):
    corpus_id: str | None = None
    doc_name: str
    doc_format: str
    page_count: int
    table_count: int
    last_ingested_at: str
    warnings: list[str] = Field(default_factory=list)
    source_relative_path: str | None = None
    source_level: str | None = None
    policy_category: str | None = None
    effective_date: str | None = None
    drift_status: str | None = None
    doc_date: str | None = None
    doc_author: str | None = None
    doc_type: str | None = None
    doc_summary: str | None = None


class JobRecord(BaseModel):
    job_id: str
    kind: str
    filename: str | None = None
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    progress: float = 0.0


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    active_jobs: list[JobRecord] = Field(default_factory=list)
    recent_jobs: list[JobRecord] = Field(default_factory=list)


class UploadResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    filename: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    report_id: str | None = None
    progress: float = 0.0


class EvalReportSummary(BaseModel):
    report_id: str
    generated_at: str
    prompt_count: int
    averages: dict[str, float]


class EvalReportResponse(BaseModel):
    report_id: str
    generated_at: str
    prompt_count: int
    judge_failures: int = 0
    averages: dict[str, float]
    results: list[dict[str, Any]]


class ExportRequest(BaseModel):
    title: str
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    generated_at: str | None = None


class CorpusRecord(BaseModel):
    id: str
    name: str
    source_path: str
    active: bool = False
    status: str = "new"
    document_count: int = 0
    chunk_count: int = 0
    last_indexed_at: str | None = None
    last_error: str | None = None
    drift_state: str = "stale"
    source_file_count: int = 0
    active_jobs: list[JobRecord] = Field(default_factory=list)
    recent_jobs: list[JobRecord] = Field(default_factory=list)


class CorpusListResponse(BaseModel):
    active_corpus_id: str
    corpora: list[CorpusRecord]


class CorpusCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    source_path: str = Field(..., min_length=1, max_length=500)
    corpus_id: str | None = Field(default=None, max_length=120)


class CorpusReplaceSourceRequest(BaseModel):
    source_path: str = Field(..., min_length=1, max_length=500)


class CorpusHealthResponse(BaseModel):
    corpus_id: str
    drift_state: str
    source_path: str
    source_file_count: int
    generated_document_count: int = 0
    indexed_document_count: int
    chunk_count: int
    parent_chunk_count: int = 0
    missing_source_count: int = 0
    new_source_count: int = 0
    changed_source_count: int = 0
    missing_chunk_file_count: int = 0
    last_indexed_at: str | None = None
    missing_sources: list[str] = Field(default_factory=list)
    new_sources: list[str] = Field(default_factory=list)
    changed_sources: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    source_level: str | None = None
    policy_category: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    chunk_id: str
    doc_name: str
    source_relative_path: str | None = None
    source_level: str | None = None
    policy_category: str | None = None
    page_range: str
    excerpt: str
    score: float
    matched_terms: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class RecommendationResponse(BaseModel):
    recommendations: list[str]
    corpus_id: str
