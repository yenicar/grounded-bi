from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from fastapi import HTTPException, Request, status

from .models import CurrentUser

ProviderName = Literal["google", "github", "dev"]
_SESSION_USER_KEY = "auth_user"
_POST_AUTH_REDIRECT_KEY = "auth_next_url"


@dataclass(frozen=True, slots=True)
class AuthSettings:
    secret_key: str
    session_cookie_secure: bool
    success_redirect: str


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    return AuthSettings(
        secret_key=os.getenv("TRBI_AUTH_SECRET", "trbi-dev-insecure-secret"),
        session_cookie_secure=_env_flag("TRBI_SESSION_COOKIE_SECURE", default=False),
        success_redirect=os.getenv("TRBI_AUTH_SUCCESS_REDIRECT", "http://localhost:5173/library"),
    )


def dev_auth_enabled() -> bool:
    configured = os.getenv("TRBI_DEV_AUTH_ENABLED")
    if configured is None:
        return True
    return _env_flag("TRBI_DEV_AUTH_ENABLED", default=True)


def available_auth_providers() -> list[ProviderName]:
    providers: list[ProviderName] = []
    if dev_auth_enabled():
        providers.append("dev")
    if os.getenv("TRBI_GOOGLE_CLIENT_ID") and os.getenv("TRBI_GOOGLE_CLIENT_SECRET"):
        providers.append("google")
    if os.getenv("TRBI_GITHUB_CLIENT_ID") and os.getenv("TRBI_GITHUB_CLIENT_SECRET"):
        providers.append("github")
    return providers


def _build_oauth():
    try:
        from authlib.integrations.starlette_client import OAuth
    except ImportError as exc:
        raise RuntimeError(
            "Authlib is required for OAuth login. Install project dependencies with the auth extra."
        ) from exc

    oauth = OAuth()
    if "google" in available_auth_providers():
        oauth.register(
            name="google",
            client_id=os.getenv("TRBI_GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("TRBI_GOOGLE_CLIENT_SECRET"),
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    if "github" in available_auth_providers():
        oauth.register(
            name="github",
            client_id=os.getenv("TRBI_GITHUB_CLIENT_ID"),
            client_secret=os.getenv("TRBI_GITHUB_CLIENT_SECRET"),
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email"},
        )
    return oauth


@lru_cache(maxsize=1)
def get_oauth_registry():
    return _build_oauth()


def get_oauth_client(provider: ProviderName):
    if provider == "dev":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The dev auth provider does not use the OAuth client flow.",
        )
    if provider not in available_auth_providers():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OAuth provider is not configured: {provider}",
        )
    oauth = get_oauth_registry()
    client = getattr(oauth, provider, None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OAuth provider is not available: {provider}",
        )
    return client


def store_user_session(request: Request, user: CurrentUser) -> None:
    request.session[_SESSION_USER_KEY] = user.model_dump()


def clear_user_session(request: Request) -> None:
    request.session.pop(_SESSION_USER_KEY, None)


def store_post_auth_redirect(request: Request, next_url: str | None) -> None:
    if next_url:
        request.session[_POST_AUTH_REDIRECT_KEY] = next_url
    else:
        request.session.pop(_POST_AUTH_REDIRECT_KEY, None)


def pop_post_auth_redirect(request: Request) -> str | None:
    value = request.session.pop(_POST_AUTH_REDIRECT_KEY, None)
    if isinstance(value, str) and value.strip():
        return value
    return None


def build_dev_user() -> CurrentUser:
    name = os.getenv("TRBI_DEV_AUTH_NAME", "Local Analyst")
    email = os.getenv("TRBI_DEV_AUTH_EMAIL", "analyst@localhost")
    return CurrentUser(
        subject=os.getenv("TRBI_DEV_AUTH_SUBJECT", "dev-analyst"),
        provider="dev",
        email=email,
        name=name,
    )


def load_user_from_request(request: Request) -> CurrentUser | None:
    cached = getattr(request.state, "user", None)
    if isinstance(cached, CurrentUser):
        return cached

    if "session" not in request.scope:
        request.state.user = None
        return None

    raw_user = request.session.get(_SESSION_USER_KEY)
    if not raw_user:
        request.state.user = None
        return None

    try:
        user = CurrentUser.model_validate(raw_user)
    except Exception:
        clear_user_session(request)
        request.state.user = None
        return None

    request.state.user = user
    return user


async def build_user_from_callback(request: Request, provider: ProviderName) -> CurrentUser:
    client = get_oauth_client(provider)
    token = await client.authorize_access_token(request)

    if provider == "google":
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = await client.userinfo(token=token)
        return CurrentUser(
            subject=str(userinfo.get("sub") or userinfo.get("email") or "google-user"),
            provider="google",
            email=userinfo.get("email"),
            name=userinfo.get("name"),
            avatar_url=userinfo.get("picture"),
        )

    response = await client.get("user", token=token)
    userinfo = response.json()
    email = userinfo.get("email")
    if not email:
        email_response = await client.get("user/emails", token=token)
        emails = email_response.json()
        primary = next((item for item in emails if item.get("primary")), None)
        if primary:
            email = primary.get("email")
    return CurrentUser(
        subject=str(userinfo.get("id") or userinfo.get("login") or "github-user"),
        provider="github",
        email=email,
        name=userinfo.get("name") or userinfo.get("login"),
        avatar_url=userinfo.get("avatar_url"),
    )
