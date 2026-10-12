from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from true_rag_bi.audit import append_query_audit_log
from true_rag_bi.config import ProjectPaths
from true_rag_bi.metadata import extract_document_metadata


def test_extract_document_metadata_parses_date_author_and_type():
    metadata = extract_document_metadata(
        "Board Memo.pdf",
        [
            "Board Memo\nDate: April 22, 2026\nAuthor: Jane Analyst\n\nThis memo summarizes the findings.",
            "Follow-up page",
        ],
    )

    assert metadata["doc_date"] == "2026-04-22"
    assert metadata["doc_author"] == "Jane Analyst"
    assert metadata["doc_type"] == "memo"


def test_append_query_audit_log_writes_jsonl_record():
    root = Path.cwd() / ".tmp_test_phase4" / f"case-{uuid4().hex}"
    try:
        paths = ProjectPaths(root=root)
        append_query_audit_log(
            paths,
            user={"subject": "dev-analyst", "provider": "dev"},
            route="ask",
            question="What changed?",
            retrieval_question="What changed?",
            citations=[{"doc_name": "brief.pdf", "page_range": "p.2", "excerpt": "Evidence"}],
            metrics={"total_seconds": 1.2},
        )

        audit_path = paths.audit_dir / "query_log.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["route"] == "ask"
        assert payload["user"]["subject"] == "dev-analyst"
        assert payload["latency_seconds"] == 1.2
    finally:
        shutil.rmtree(root, ignore_errors=True)
