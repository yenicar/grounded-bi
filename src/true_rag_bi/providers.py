from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import AppConfig


@dataclass(slots=True)
class GenerationResult:
    text: str
    usage: dict[str, Any]


@runtime_checkable
class RAGProvider(Protocol):
    embedding_model: str
    answer_model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def generate_answer(self, prompt: str) -> GenerationResult: ...


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(keyword in msg for keyword in ("quota", "rate", "unavailable", "503", "429", "resource exhausted"))


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
_resolved_model_cache: dict[str, str] = {}


def _load_gemini_api_key(config: AppConfig) -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    secrets_path = config.paths.root / ".secrets" / "gemini_key.txt"
    if secrets_path.exists():
        file_key = secrets_path.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key

    raise RuntimeError(
        "Gemini API key not found. Set GEMINI_API_KEY or place the key in .secrets/gemini_key.txt."
    )


class GeminiProvider:
    def __init__(self, config: AppConfig) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai is required. Install project dependencies first.") from exc

        api_key = _load_gemini_api_key(config)
        genai.configure(api_key=api_key)
        self._genai = genai
        self._configured_embedding_model = config.embedding_model
        self._embedding_model_resolved = False
        self.embedding_model = self._normalize_embedding_model(config.embedding_model)
        self.answer_model = config.answer_model
        self._answer_client = genai.GenerativeModel(self.answer_model)

    @staticmethod
    def _normalize_embedding_model(model_name: str) -> str:
        if model_name.startswith("models/"):
            return model_name
        return f"models/{model_name}"

    def _resolve_embedding_model(self, configured_model: str) -> str:
        if configured_model in _resolved_model_cache:
            return _resolved_model_cache[configured_model]
        resolved = self._do_resolve_embedding_model(configured_model)
        _resolved_model_cache[configured_model] = resolved
        return resolved

    def _ensure_embedding_model(self) -> str:
        if not self._embedding_model_resolved:
            self.embedding_model = self._resolve_embedding_model(self._configured_embedding_model)
            self._embedding_model_resolved = True
        return self.embedding_model

    def _do_resolve_embedding_model(self, configured_model: str) -> str:
        requested = self._normalize_embedding_model(configured_model)
        available = [
            model.name
            for model in self._genai.list_models()
            if "embedContent" in getattr(model, "supported_generation_methods", [])
        ]
        if requested in available:
            return requested

        for fallback in ("models/gemini-embedding-001", "models/gemini-embedding-2-preview"):
            if fallback in available:
                return fallback

        if available:
            return available[0]

        raise RuntimeError("No Gemini embedding model with embedContent support is available for this API key.")

    def _extract_embedding(self, response: Any) -> list[float]:
        if isinstance(response, dict):
            if "embedding" in response:
                embedding = response["embedding"]
                if embedding and isinstance(embedding[0], list):
                    return embedding[0]
                return embedding
            if "embeddings" in response and response["embeddings"]:
                item = response["embeddings"][0]
                if isinstance(item, dict) and "values" in item:
                    return item["values"]
        raise RuntimeError("Unexpected Gemini embedding response format.")

    @_retry
    def _embed_single(self, text: str, task_type: str) -> list[float]:
        response = self._genai.embed_content(
            model=self._ensure_embedding_model(),
            content=text,
            task_type=task_type,
        )
        return self._extract_embedding(response)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self._embed_single, text, "retrieval_document"): index
                for index, text in enumerate(texts)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return [embedding for embedding in results if embedding is not None]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text, "retrieval_query")

    @_retry
    def generate_answer(self, prompt: str) -> GenerationResult:
        response = self._answer_client.generate_content(prompt)
        usage = {}
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            usage = {
                "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
                "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
                "total_token_count": getattr(usage_metadata, "total_token_count", None),
            }
        return GenerationResult(text=response.text.strip(), usage=usage)


def build_provider(config: AppConfig) -> RAGProvider:
    if config.provider_name != "gemini":
        raise RuntimeError(f"Unsupported provider: {config.provider_name}")
    return GeminiProvider(config)
