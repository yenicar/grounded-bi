from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ..dependencies import build_pdf_export, build_pptx_export, require_user
from ..models import ExportRequest

router = APIRouter(prefix="/api/export", tags=["export"], dependencies=[Depends(require_user)])


@router.post("/pdf")
def export_pdf(request: ExportRequest) -> Response:
    payload = request.model_dump()
    data = build_pdf_export(payload)
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{request.title or "groundedbi-export"}.pdf"'},
    )


@router.post("/pptx")
def export_pptx(request: ExportRequest) -> Response:
    payload = request.model_dump()
    data = build_pptx_export(payload)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{request.title or "groundedbi-export"}.pptx"'},
    )
