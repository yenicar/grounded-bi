from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def corpus_id(self) -> str:
        from .corpus import active_corpus_id

        return active_corpus_id(self.root)

    @property
    def corpora_dir(self) -> Path:
        return self.data_dir / "corpora"

    @property
    def corpus_dir(self) -> Path:
        from .corpus import active_corpus_dir

        return active_corpus_dir(self.root)

    @property
    def raw_docs_dir(self) -> Path:
        from .corpus import active_source_dir

        return active_source_dir(self.root)

    @property
    def raw_pdfs_dir(self) -> Path:
        return self.raw_docs_dir

    @property
    def normalized_md_dir(self) -> Path:
        return self.corpus_dir / "normalized_md"

    @property
    def chunks_dir(self) -> Path:
        return self.corpus_dir / "chunks"

    @property
    def index_dir(self) -> Path:
        return self.corpus_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.corpus_dir / "cache"

    @property
    def eval_dir(self) -> Path:
        return self.corpus_dir / "eval"

    @property
    def audit_dir(self) -> Path:
        return self.corpus_dir / "audit"

    @property
    def chroma_dir(self) -> Path:
        explicit = os.getenv("TRBI_CHROMA_DIR")
        if explicit:
            return Path(explicit)
        return self.corpus_dir / "chroma"


@dataclass(slots=True)
class AppConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    provider_name: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    answer_model: str = "gemini-2.5-flash"
    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 120
    child_chunk_target_tokens: int = 220
    child_chunk_overlap_tokens: int = 40
    retrieval_top_k: int = 8
    retrieval_fetch_k: int = 16
    retrieval_rrf_k: int = 60
    prompt_max_context_chunks: int = 4
    prompt_max_context_tokens: int = 1600
    prompt_max_chunk_tokens: int = 420
    query_cache_limit: int = 200
    chat_history_turns: int = 6
    answer_grounding_enabled: bool = False
    version: str = "v4_fast_compare_grounded"


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_config() -> AppConfig:
    return AppConfig(
        provider_name=os.getenv("TRBI_PROVIDER", "gemini"),
        embedding_model=os.getenv("TRBI_EMBEDDING_MODEL", "gemini-embedding-001"),
        answer_model=os.getenv("TRBI_ANSWER_MODEL", "gemini-2.5-flash"),
        chunk_target_tokens=_int_env("TRBI_CHUNK_TARGET_TOKENS", 800),
        chunk_overlap_tokens=_int_env("TRBI_CHUNK_OVERLAP_TOKENS", 120),
        child_chunk_target_tokens=_int_env("TRBI_CHILD_CHUNK_TARGET_TOKENS", 220),
        child_chunk_overlap_tokens=_int_env("TRBI_CHILD_CHUNK_OVERLAP_TOKENS", 40),
        retrieval_top_k=_int_env("TRBI_RETRIEVAL_TOP_K", 8),
        retrieval_fetch_k=_int_env("TRBI_RETRIEVAL_FETCH_K", 16),
        retrieval_rrf_k=_int_env("TRBI_RETRIEVAL_RRF_K", 60),
        prompt_max_context_chunks=_int_env("TRBI_PROMPT_MAX_CONTEXT_CHUNKS", 4),
        prompt_max_context_tokens=_int_env("TRBI_PROMPT_MAX_CONTEXT_TOKENS", 1600),
        prompt_max_chunk_tokens=_int_env("TRBI_PROMPT_MAX_CHUNK_TOKENS", 420),
        query_cache_limit=_int_env("TRBI_QUERY_CACHE_LIMIT", 200),
        chat_history_turns=_int_env("TRBI_CHAT_HISTORY_TURNS", 6),
        answer_grounding_enabled=_bool_env("TRBI_ANSWER_GROUNDING_ENABLED", False),
        version=os.getenv("TRBI_VERSION", "v4_fast_compare_grounded"),
    )
