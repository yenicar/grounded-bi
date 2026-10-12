from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from true_rag_bi.config import AppConfig, ProjectPaths
from true_rag_bi.corpus import (
    build_corpus_health,
    get_active_corpus,
    list_source_files,
    reset_corpus_artifacts,
)
from true_rag_bi.ingest import ingest_documents


def _tmp_root() -> Path:
    root = Path.cwd() / ".tmp_test_corpus" / uuid4().hex
    root.mkdir(parents=True)
    return root


def test_default_active_corpus_prefers_hr_resources():
    tmp_path = _tmp_root()
    try:
        source = tmp_path / "data" / "raw_pdfs" / "HR Resources"
        source.mkdir(parents=True)

        corpus = get_active_corpus(tmp_path)

        assert corpus["id"] == "hr-resources"
        assert corpus["name"] == "HR Resources"
        assert corpus["source_path"] == "data/raw_pdfs/HR Resources"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_list_source_files_is_recursive_and_preserves_relative_paths():
    tmp_path = _tmp_root()
    try:
        source = tmp_path / "data" / "raw_pdfs" / "HR Resources"
        federal = source / "01_Federal"
        state = source / "02_Wisconsin"
        federal.mkdir(parents=True)
        state.mkdir(parents=True)
        (federal / "FMLA.pdf").write_bytes(b"one")
        (state / "FMLA.pdf").write_bytes(b"two")

        files = list_source_files(source)

        assert [item["source_relative_path"] for item in files] == [
            "01_Federal/FMLA.pdf",
            "02_Wisconsin/FMLA.pdf",
        ]
        assert files[0]["artifact_id"] != files[1]["artifact_id"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_ingest_documents_recurses_and_records_hr_metadata(monkeypatch):
    tmp_path = _tmp_root()
    try:
        source = tmp_path / "data" / "raw_pdfs" / "HR Resources" / "01_Federal"
        source.mkdir(parents=True)
        (source / "DOL-FMLA-EmployeeGuide-2026.pdf").write_bytes(b"fake pdf")
        config = AppConfig(paths=ProjectPaths(root=tmp_path))

        monkeypatch.setattr(
            "true_rag_bi.ingest._extract_document",
            lambda _path: (["FMLA employee guide effective 2026"], []),
        )

        payload = ingest_documents(config)

        assert payload["documents"][0]["doc_name"] == "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf"
        assert payload["documents"][0]["source_relative_path"] == "01_Federal/DOL-FMLA-EmployeeGuide-2026.pdf"
        assert payload["documents"][0]["source_level"] == "Federal"
        assert payload["documents"][0]["policy_category"] == "leave"
        assert (tmp_path / "data" / "corpora" / "hr-resources" / "normalized_md").exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_corpus_health_detects_manifest_source_mismatch():
    tmp_path = _tmp_root()
    try:
        source = tmp_path / "data" / "raw_pdfs" / "HR Resources"
        source.mkdir(parents=True)
        (source / "Current.pdf").write_bytes(b"new")
        cache = tmp_path / "data" / "corpora" / "hr-resources" / "cache"
        cache.mkdir(parents=True)
        (cache / "ingestion_manifest.json").write_text(
            '{"documents":[{"doc_name":"Missing.pdf","source_relative_path":"Missing.pdf","source_pdf_hash":"old"}]}',
            encoding="utf-8",
        )
        config = AppConfig(paths=ProjectPaths(root=tmp_path))

        health = build_corpus_health(config)

        assert health["drift_state"] == "stale"
        assert health["missing_source_count"] == 1
        assert health["new_source_count"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_corpus_health_reports_generated_markdown_before_manifest_is_final():
    tmp_path = _tmp_root()
    try:
        source = tmp_path / "data" / "raw_pdfs" / "HR Resources"
        source.mkdir(parents=True)
        (source / "Policy.pdf").write_bytes(b"source")
        generated = tmp_path / "data" / "corpora" / "hr-resources" / "normalized_md"
        generated.mkdir(parents=True)
        (generated / "policy-abc123.md").write_text("# Policy\n", encoding="utf-8")
        config = AppConfig(paths=ProjectPaths(root=tmp_path))

        health = build_corpus_health(config)

        assert health["generated_document_count"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_reset_corpus_artifacts_preserves_source_files():
    tmp_path = _tmp_root()
    try:
        source = tmp_path / "data" / "raw_pdfs" / "HR Resources"
        source.mkdir(parents=True)
        source_file = source / "Policy.pdf"
        source_file.write_bytes(b"source")
        generated = tmp_path / "data" / "corpora" / "hr-resources" / "chunks"
        generated.mkdir(parents=True)
        (generated / "stale.jsonl").write_text("{}", encoding="utf-8")
        config = AppConfig(paths=ProjectPaths(root=tmp_path))

        reset_corpus_artifacts(config)

        assert source_file.exists()
        assert not generated.exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
