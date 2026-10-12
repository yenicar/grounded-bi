from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from ..auth import (
    available_auth_providers,
    build_dev_user,
    build_user_from_callback,
    clear_user_session,
    dev_auth_enabled,
    get_auth_settings,
    get_oauth_client,
    pop_post_auth_redirect,
    store_user_session,
    store_post_auth_redirect,
)
from ..dependencies import get_current_user
from ..models import AuthSessionResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=AuthSessionResponse)
def auth_me(request: Request) -> AuthSessionResponse:
    user = get_current_user(request)
    return AuthSessionResponse(
        authenticated=user is not None,
        user=user,
        available_providers=available_auth_providers(),
    )


@router.get("/login/{provider}")
async def auth_login(provider: str, request: Request):
    next_url = request.query_params.get("next")
    store_post_auth_redirect(request, next_url)
    if provider == "dev":
        if not dev_auth_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dev auth is disabled.")
        store_user_session(request, build_dev_user())
        return RedirectResponse(pop_post_auth_redirect(request) or get_auth_settings().success_redirect, status_code=302)
    client = get_oauth_client(provider)  # type: ignore[arg-type]
    redirect_uri = request.url_for("auth_callback", provider=provider)
    return await client.authorize_redirect(request, str(redirect_uri))


@router.get("/callback/{provider}", name="auth_callback")
async def auth_callback(provider: str, request: Request) -> RedirectResponse:
    user = await build_user_from_callback(request, provider)  # type: ignore[arg-type]
    store_user_session(request, user)
    return RedirectResponse(pop_post_auth_redirect(request) or get_auth_settings().success_redirect, status_code=302)


@router.post("/logout", response_model=AuthSessionResponse)
def auth_logout(request: Request) -> AuthSessionResponse:
    clear_user_session(request)
    return AuthSessionResponse(
        authenticated=False,
        user=None,
        available_providers=available_auth_providers(),
    )
