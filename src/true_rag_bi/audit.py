from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ProjectPaths
from .utils import ensure_directory, utc_now_iso


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def hash_question(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def append_query_audit_log(
    paths: ProjectPaths,
    *,
    user: dict[str, Any],
    route: str,
    question: str,
    retrieval_question: str,
    citations: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "timestamp": utc_now_iso(),
        "user": user,
        "route": route,
        "question_hash": hash_question(question),
        "question_length": len(question),
        "question_preview": question[:200],
        "retrieval_question": retrieval_question,
        "citations": citations,
        "metrics": metrics,
        "latency_seconds": metrics.get("total_seconds"),
    }
    _append_jsonl(paths.audit_dir / "query_log.jsonl", record)
    return record
