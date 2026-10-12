from __future__ import annotations

import re
from datetime import datetime

from .utils import normalize_text

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        r")\s+\d{1,2},\s+\d{4}\b",
        re.IGNORECASE,
    ),
]
_AUTHOR_PATTERNS = [
    re.compile(r"^(?:author|prepared by|from|by)\s*:\s*(.+)$", re.IGNORECASE),
    re.compile(r"^(?:author|prepared by|from|by)\s+(.+)$", re.IGNORECASE),
]


def _normalize_date(value: str) -> str | None:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _extract_doc_date(candidate_text: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(candidate_text)
        if not match:
            continue
        value = match.group(0)
        normalized = _normalize_date(value)
        if normalized:
            return normalized
    return None


def _extract_doc_author(lines: list[str]) -> str | None:
    for line in lines[:20]:
        cleaned = normalize_text(line)
        if not cleaned:
            continue
        for pattern in _AUTHOR_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                author = normalize_text(match.group(1)).strip(" -")
                if author:
                    return author[:120]
    return None


def _infer_doc_type(doc_name: str, sample_text: str) -> str:
    lowered_name = doc_name.lower()
    lowered_text = sample_text.lower()
    if "transcript" in lowered_name or "transcript" in lowered_text:
        return "transcript"
    if "memo" in lowered_name or "memo" in lowered_text:
        return "memo"
    if "email" in lowered_name or "subject:" in lowered_text or "from:" in lowered_text:
        return "email"
    if any(keyword in lowered_name for keyword in ("report", "brief", "analysis")) or "executive summary" in lowered_text:
        return "report"
    if doc_name.lower().endswith((".xlsx", ".csv")):
        return "spreadsheet"
    return "unknown"


def extract_document_metadata(doc_name: str, pages: list[str]) -> dict[str, str | None]:
    sample_pages = [normalize_text(page) for page in pages[:3] if normalize_text(page)]
    sample_text = "\n\n".join(sample_pages).strip()
    lines = [line for line in sample_text.splitlines() if line.strip()]
    return {
        "doc_date": _extract_doc_date(sample_text) if sample_text else None,
        "doc_author": _extract_doc_author(lines),
        "doc_type": _infer_doc_type(doc_name, sample_text),
    }
