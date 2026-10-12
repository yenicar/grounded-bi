from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import (
    AppServices,
    activate_corpus_for_services,
    build_corpus_health_response,
    create_corpus,
    get_services,
    list_corpora_payload,
    queue_corpus_job,
    replace_corpus_source,
    require_user,
)
from ..models import (
    CorpusCreateRequest,
    CorpusHealthResponse,
    CorpusListResponse,
    CorpusRecord,
    CorpusReplaceSourceRequest,
    JobStatusResponse,
)

router = APIRouter(prefix="/api/corpora", tags=["corpora"], dependencies=[Depends(require_user)])


@router.get("", response_model=CorpusListResponse)
def list_corpora(services: AppServices = Depends(get_services)) -> CorpusListResponse:
    return CorpusListResponse(**list_corpora_payload(services))


@router.post("", response_model=CorpusRecord, status_code=status.HTTP_201_CREATED)
def create_corpus_endpoint(
    request: CorpusCreateRequest,
    services: AppServices = Depends(get_services),
) -> CorpusRecord:
    try:
        create_corpus(
            services,
            name=request.name,
            source_path=request.source_path,
            corpus_id=request.corpus_id,
        )
        payload = list_corpora_payload(services)
        created = next(item for item in payload["corpora"] if item["name"] == request.name)
        return CorpusRecord(**created)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{corpus_id}/health", response_model=CorpusHealthResponse)
def get_corpus_health(
    corpus_id: str,
    services: AppServices = Depends(get_services),
) -> CorpusHealthResponse:
    try:
        return CorpusHealthResponse(**build_corpus_health_response(services, corpus_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {corpus_id}") from exc


@router.post("/{corpus_id}/activate", response_model=CorpusRecord)
def activate_corpus_endpoint(
    corpus_id: str,
    services: AppServices = Depends(get_services),
) -> CorpusRecord:
    try:
        activate_corpus_for_services(services, corpus_id)
        payload = list_corpora_payload(services)
        active = next(item for item in payload["corpora"] if item["id"] == corpus_id)
        return CorpusRecord(**active)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {corpus_id}") from exc


@router.post("/{corpus_id}/rebuild", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def rebuild_corpus_endpoint(
    corpus_id: str,
    services: AppServices = Depends(get_services),
) -> JobStatusResponse:
    try:
        return JobStatusResponse(**queue_corpus_job(services, kind="rebuild", corpus_id=corpus_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {corpus_id}") from exc


@router.post("/{corpus_id}/reset", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def reset_corpus_endpoint(
    corpus_id: str,
    services: AppServices = Depends(get_services),
) -> JobStatusResponse:
    try:
        return JobStatusResponse(**queue_corpus_job(services, kind="reset", corpus_id=corpus_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {corpus_id}") from exc


@router.post("/{corpus_id}/replace-source", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def replace_corpus_source_endpoint(
    corpus_id: str,
    request: CorpusReplaceSourceRequest,
    services: AppServices = Depends(get_services),
) -> JobStatusResponse:
    try:
        replace_corpus_source(services, corpus_id, request.source_path)
        return JobStatusResponse(**queue_corpus_job(services, kind="replace-source", corpus_id=corpus_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {corpus_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
