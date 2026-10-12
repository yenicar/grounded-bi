from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import AppServices, get_services, require_user, search_corpus
from ..models import SearchRequest, SearchResponse

router = APIRouter(prefix="/api", tags=["search"], dependencies=[Depends(require_user)])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    services: AppServices = Depends(get_services),
) -> SearchResponse:
    return search_corpus(
        services,
        query=request.query,
        source_level=request.source_level,
        policy_category=request.policy_category,
        limit=request.limit,
    )
