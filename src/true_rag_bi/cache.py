from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from .config import AppConfig
from .utils import read_json, write_json


@dataclass(frozen=True, slots=True)
class PipelineFingerprint:
    version: str
    provider_name: str
    embedding_model: str
    answer_model: str
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    child_chunk_target_tokens: int
    child_chunk_overlap_tokens: int
    retrieval_top_k: int
    retrieval_fetch_k: int
    retrieval_rrf_k: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pipeline_fingerprint(config: AppConfig) -> PipelineFingerprint:
    return PipelineFingerprint(
        version=config.version,
        provider_name=config.provider_name,
        embedding_model=config.embedding_model,
        answer_model=config.answer_model,
        chunk_target_tokens=config.chunk_target_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
        child_chunk_target_tokens=config.child_chunk_target_tokens,
        child_chunk_overlap_tokens=config.child_chunk_overlap_tokens,
        retrieval_top_k=config.retrieval_top_k,
        retrieval_fetch_k=config.retrieval_fetch_k,
        retrieval_rrf_k=config.retrieval_rrf_k,
    )


def build_query_cache_key(normalized_query: str, pipeline_key: str, corpus_fingerprint: str) -> str:
    raw = f"{normalized_query}|{pipeline_key}|{corpus_fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_corpus_fingerprint(chunks: list[dict[str, Any]]) -> str:
    records = [f"{chunk['chunk_id']}|{chunk['content_hash']}" for chunk in sorted(chunks, key=lambda item: item["chunk_id"])]
    raw = "\n".join(records)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def load(self, name: str, default: Any) -> Any:
        path = self.path_for(name)
        lock_path = path.with_suffix(".lock")
        with FileLock(str(lock_path), timeout=10):
            return read_json(path, default)

    def save(self, name: str, payload: Any) -> None:
        path = self.path_for(name)
        lock_path = path.with_suffix(".lock")
        with FileLock(str(lock_path), timeout=10):
            write_json(path, payload)
