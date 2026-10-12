from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import (
    PRIORITY_BENCHMARK,
    AppServices,
    get_eval_report,
    get_job,
    get_services,
    list_eval_reports,
    new_job,
    require_user,
    run_eval_job,
)
from ..models import EvalReportResponse, EvalReportSummary, JobStatusResponse, UploadResponse

router = APIRouter(prefix="/api/eval", tags=["evaluation"], dependencies=[Depends(require_user)])


@router.post("/run", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def run_evaluation_endpoint(
    services: AppServices = Depends(get_services),
) -> UploadResponse:
    job = new_job(services, kind="eval", filename="benchmark")
    services.scheduler.submit(
        lambda: run_eval_job(services, job["job_id"]),
        priority=PRIORITY_BENCHMARK,
    )
    return UploadResponse(
        job_id=job["job_id"],
        status=job["status"],
        filename="benchmark",
        message="Evaluation accepted and started in the background.",
    )


@router.get("/reports", response_model=list[EvalReportSummary])
def get_eval_reports(services: AppServices = Depends(get_services)) -> list[EvalReportSummary]:
    return list_eval_reports(services)


@router.get("/reports/{report_id}", response_model=EvalReportResponse)
def get_eval_report_endpoint(report_id: str, services: AppServices = Depends(get_services)) -> EvalReportResponse:
    try:
        return get_eval_report(services, report_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Evaluation report not found: {report_id}") from exc


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_eval_job_endpoint(job_id: str, services: AppServices = Depends(get_services)) -> JobStatusResponse:
    try:
        return get_job(services, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Evaluation job not found: {job_id}") from exc
