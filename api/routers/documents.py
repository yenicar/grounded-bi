from __future__ import annotations

from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from true_rag_bi.utils import ensure_directory, standardize_filename

from ..dependencies import (
    PRIORITY_PIPELINE,
    AppServices,
    get_services,
    list_active_jobs,
    list_documents,
    list_recent_jobs,
    new_job,
    remove_document_artifacts,
    require_user,
    run_pipeline_job,
)
from ..models import DocumentListResponse, UploadResponse

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(require_user)])

_SUPPORTED_UPLOADS = {".pdf", ".docx", ".xlsx", ".csv"}


def _dedupe_destination_name(destination_dir: Path, requested_name: str) -> str:
    candidate = Path(requested_name)
    stem = candidate.stem
    suffix = candidate.suffix
    final_name = candidate.name
    counter = 2
    while (destination_dir / final_name).exists():
        final_name = f"{stem}-{counter}{suffix}"
        counter += 1
    return final_name


@router.get("", response_model=DocumentListResponse)
def get_documents(services: AppServices = Depends(get_services)) -> DocumentListResponse:
    return DocumentListResponse(
        documents=list_documents(services),
        active_jobs=list_active_jobs(services),
        recent_jobs=list_recent_jobs(services),
    )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile = File(...),
    services: AppServices = Depends(get_services),
) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_UPLOADS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    standardized_name = standardize_filename(filename)
    standardized_name = _dedupe_destination_name(services.config.paths.raw_docs_dir, standardized_name)
    destination = services.config.paths.raw_docs_dir / standardized_name
    ensure_directory(destination.parent)
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    job = new_job(services, kind="upload", filename=standardized_name)
    services.scheduler.submit(
        lambda: run_pipeline_job(services, job["job_id"]),
        priority=PRIORITY_PIPELINE,
    )
    return UploadResponse(
        job_id=job["job_id"],
        status=job["status"],
        filename=standardized_name,
        message=(
            f"Upload accepted as {standardized_name}. Ingestion and indexing started in the background."
        ),
    )


@router.delete("/{doc_name:path}", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def delete_document(
    doc_name: str,
    services: AppServices = Depends(get_services),
) -> UploadResponse:
    try:
        remove_document_artifacts(services, doc_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_name}") from exc

    job = new_job(services, kind="delete", filename=doc_name)
    services.scheduler.submit(
        lambda: run_pipeline_job(services, job["job_id"]),
        priority=PRIORITY_PIPELINE,
    )
    return UploadResponse(
        job_id=job["job_id"],
        status=job["status"],
        filename=doc_name,
        message="Document removed. Corpus refresh started in the background.",
    )
