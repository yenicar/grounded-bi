from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken

logger = logging.getLogger("grounded_bi")

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]+")
_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = False
    return _TOKENIZER


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u200b", "")
    lines = [_MULTI_SPACES.sub(" ", line).strip() for line in normalized.splitlines()]
    normalized = "\n".join(lines)
    normalized = _MULTI_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def normalize_query(text: str) -> str:
    return " ".join(normalize_text(text).lower().split())


def tokenize_for_search(text: str) -> list[str]:
    normalized = normalize_query(text)
    return _SEARCH_TOKEN_RE.findall(normalized)


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def standardize_filename(filename: str) -> str:
    path = Path(filename)
    stem = normalize_text(path.stem)
    extension = path.suffix.lower()
    normalized_stem = slugify(stem)
    if not normalized_stem:
        normalized_stem = "document"
    if not extension:
        return normalized_stem
    return f"{normalized_stem}{extension}"


def estimate_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer:
        return max(1, len(tokenizer.encode(text)))
    words = max(1, len(text.split()))
    return max(1, int(words * 1.3))


def configure_logging(verbose: bool = False) -> None:
    """Configure structured logging for the grounded_bi package."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    logging.getLogger("grounded_bi").setLevel(level)
