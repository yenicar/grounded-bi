from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import AppServices, get_services, log_query_event, require_user, run_question
from ..models import CurrentUser
from ..models import AskRequest, AskResponse, ChatRequest, ChatResponse, ChatTurn, RecommendationResponse

router = APIRouter(prefix="/api/query", tags=["query"], dependencies=[Depends(require_user)])


@router.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(
    request: AskRequest,
    services: AppServices = Depends(get_services),
    user: CurrentUser = Depends(require_user),
) -> AskResponse:
    response = run_question(
        services,
        request.question,
        debug=request.debug,
        grounding_enabled=request.grounding_enabled,
        after=request.after.isoformat() if request.after else None,
    )
    log_query_event(services, user=user, route="ask", question=request.question, response=response)
    return response


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(
    request: ChatRequest,
    services: AppServices = Depends(get_services),
    user: CurrentUser = Depends(require_user),
) -> ChatResponse:
    history = [turn.model_dump() for turn in request.history]
    response = run_question(
        services,
        request.message,
        debug=request.debug,
        history=history,
        grounding_enabled=request.grounding_enabled,
        after=request.after.isoformat() if request.after else None,
    )
    log_query_event(services, user=user, route="chat", question=request.message, response=response)
    updated_history = request.history + [ChatTurn(user=request.message, assistant=response.answer)]
    return ChatResponse(**response.model_dump(), history=updated_history)


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    services: AppServices = Depends(get_services),
) -> RecommendationResponse:
    from ..dependencies import build_recommendations

    return build_recommendations(services)
