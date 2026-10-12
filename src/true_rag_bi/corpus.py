from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .utils import ensure_directory, read_json, sha256_file, sha256_text, slugify, utc_now_iso, write_json

SUPPORTED_CORPUS_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv"}
REGISTRY_VERSION = 1


def corpora_dir(root: Path) -> Path:
    return root / "data" / "corpora"


def registry_path(root: Path) -> Path:
    return corpora_dir(root) / "registry.json"


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _default_source(root: Path) -> tuple[str, str, str]:
    hr_source = root / "data" / "raw_pdfs" / "HR Resources"
    if hr_source.exists():
        return "hr-resources", "HR Resources", "data/raw_pdfs/HR Resources"
    return "default", "Default Corpus", "data/raw_pdfs"


def _default_registry(root: Path) -> dict[str, Any]:
    corpus_id, name, source_path = _default_source(root)
    now = utc_now_iso()
    return {
        "version": REGISTRY_VERSION,
        "active_corpus_id": corpus_id,
        "corpora": [
            {
                "id": corpus_id,
                "name": name,
                "source_path": source_path,
                "active": True,
                "status": "new",
                "document_count": 0,
                "chunk_count": 0,
                "last_indexed_at": None,
                "last_error": None,
                "created_at": now,
                "updated_at": now,
            }
        ],
    }


def load_registry(root: Path, *, create: bool = True) -> dict[str, Any]:
    path = registry_path(root)
    payload = read_json(path, None)
    if payload:
        return payload
    payload = _default_registry(root)
    if create:
        write_json(path, payload)
    return payload


def save_registry(root: Path, payload: dict[str, Any]) -> None:
    write_json(registry_path(root), payload)


def get_active_corpus(root: Path) -> dict[str, Any]:
    registry = load_registry(root)
    active_id = registry.get("active_corpus_id")
    for corpus in registry.get("corpora", []):
        if corpus.get("id") == active_id:
            return corpus
    if registry.get("corpora"):
        return registry["corpora"][0]
    payload = _default_registry(root)
    save_registry(root, payload)
    return payload["corpora"][0]


def get_corpus(root: Path, corpus_id: str) -> dict[str, Any]:
    registry = load_registry(root)
    for corpus in registry.get("corpora", []):
        if corpus.get("id") == corpus_id:
            return corpus
    raise KeyError(corpus_id)


def resolve_source_path(root: Path, source_path: str | Path) -> Path:
    raw = Path(source_path)
    if not raw.is_absolute():
        raw = root / raw
    resolved = raw.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("Source path must stay inside the project workspace.") from exc
    return resolved


def active_source_dir(root: Path) -> Path:
    return resolve_source_path(root, get_active_corpus(root)["source_path"])


def active_corpus_id(root: Path) -> str:
    return str(get_active_corpus(root)["id"])


def active_corpus_dir(root: Path) -> Path:
    return corpora_dir(root) / active_corpus_id(root)


def artifact_id_for_relative_path(relative_path: str) -> str:
    path = Path(relative_path)
    base = slugify("/".join([*path.parts[:-1], path.stem]))
    if not base:
        base = "document"
    return f"{base}-{sha256_text(relative_path.lower())[:10]}"


def list_source_files(source_dir: Path) -> list[dict[str, Any]]:
    if not source_dir.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_CORPUS_EXTENSIONS:
            continue
        relative_path = path.relative_to(source_dir).as_posix()
        files.append(
            {
                "path": path,
                "source_relative_path": relative_path,
                "artifact_id": artifact_id_for_relative_path(relative_path),
                "source_hash": sha256_file(path),
            }
        )
    return files


def infer_source_level(relative_path: str) -> str | None:
    value = relative_path.lower()
    if "01_federal" in value or "federal" in value or value.startswith("ada") or value.startswith("dol") or value.startswith("eeoc") or value.startswith("osha"):
        return "Federal"
    if "02_wisconsin" in value or "wisconsin" in value or value.startswith("uws "):
        return "Wisconsin"
    if "03_uw_system" in value or "uwsystem" in value or "uw system" in value:
        return "UW System"
    if "04_uwo_hr" in value or "uwo" in value:
        return "UWO HR"
    return None


def infer_policy_category(relative_path: str, text: str = "") -> str | None:
    value = f"{relative_path} {text}".lower()
    categories = {
        "leave": ("fmla", "leave", "vacation", "holiday", "sick", "absence"),
        "accommodation": ("ada", "accommodation", "disability", "religious"),
        "conduct": ("conduct", "ethic", "bullying", "disciplinary"),
        "safety": ("osha", "safety", "inclement", "emergency"),
        "hiring": ("recruitment", "appointment", "employment", "veteran"),
        "grievance": ("grievance", "complaint"),
        "remote work": ("remote", "telework", "telecommut"),
        "discipline": ("dismissal", "termination", "layoff", "discipline"),
    }
    for category, terms in categories.items():
        if any(term in value for term in terms):
            return category
    return None


def build_corpus_health(config: Any) -> dict[str, Any]:
    paths = config.paths
    source_files = list_source_files(paths.raw_docs_dir)
    generated_document_count = 0
    if paths.normalized_md_dir.exists():
        generated_document_count = sum(1 for _path in paths.normalized_md_dir.glob("*.md"))
    manifest = read_json(paths.cache_dir / "ingestion_manifest.json", {"documents": []})
    chunk_manifest = read_json(paths.cache_dir / "chunk_manifest.json", {"documents": []})
    index_manifest = read_json(paths.index_dir / "index_manifest.json", {})

    source_by_rel = {item["source_relative_path"]: item for item in source_files}
    docs = manifest.get("documents", [])
    manifest_by_rel = {
        record.get("source_relative_path") or record.get("doc_name"): record
        for record in docs
    }
    missing = sorted(set(manifest_by_rel) - set(source_by_rel))
    new = sorted(set(source_by_rel) - set(manifest_by_rel))
    changed = sorted(
        rel
        for rel in set(source_by_rel) & set(manifest_by_rel)
        if manifest_by_rel[rel].get("source_pdf_hash") != source_by_rel[rel]["source_hash"]
    )

    chunk_docs = {record.get("doc_name") for record in chunk_manifest.get("documents", [])}
    manifest_docs = {record.get("doc_name") for record in docs}
    chunk_mismatch = chunk_docs != manifest_docs
    missing_chunk_files = 0
    for record in chunk_manifest.get("documents", []):
        child = paths.root / record.get("chunk_file_path", "")
        parent = paths.root / record.get("parent_chunk_file_path", "")
        if not child.exists() or not parent.exists():
            missing_chunk_files += 1

    drift_state = "current"
    if missing or new or changed or chunk_mismatch or missing_chunk_files or not index_manifest:
        drift_state = "stale"

    return {
        "corpus_id": paths.corpus_id,
        "drift_state": drift_state,
        "source_path": _relative_to_root(paths.root, paths.raw_docs_dir),
        "source_file_count": len(source_files),
        "generated_document_count": generated_document_count,
        "indexed_document_count": len(docs),
        "chunk_count": int(chunk_manifest.get("chunk_count", 0) or 0),
        "parent_chunk_count": int(chunk_manifest.get("parent_chunk_count", 0) or 0),
        "missing_source_count": len(missing),
        "new_source_count": len(new),
        "changed_source_count": len(changed),
        "missing_chunk_file_count": missing_chunk_files,
        "last_indexed_at": index_manifest.get("updated_at"),
        "missing_sources": missing[:20],
        "new_sources": new[:20],
        "changed_sources": changed[:20],
    }


def reset_corpus_artifacts(config: Any) -> None:
    artifact_root = config.paths.corpus_dir
    for name in ("cache", "normalized_md", "chunks", "index", "eval", "audit", "chroma"):
        target = artifact_root / name
        if target.exists():
            shutil.rmtree(target)
    ensure_directory(artifact_root)


def activate_corpus(root: Path, corpus_id: str) -> dict[str, Any]:
    registry = load_registry(root)
    found: dict[str, Any] | None = None
    for corpus in registry.get("corpora", []):
        active = corpus.get("id") == corpus_id
        corpus["active"] = active
        if active:
            found = corpus
    if found is None:
        raise KeyError(corpus_id)
    registry["active_corpus_id"] = corpus_id
    found["updated_at"] = utc_now_iso()
    save_registry(root, registry)
    return found


def upsert_corpus(root: Path, *, name: str, source_path: str, corpus_id: str | None = None) -> dict[str, Any]:
    source = resolve_source_path(root, source_path)
    if not source.exists() or not source.is_dir():
        raise ValueError("Source path must be an existing directory.")
    registry = load_registry(root)
    cid = corpus_id or slugify(name) or artifact_id_for_relative_path(source.name)
    rel_source = _relative_to_root(root, source)
    now = utc_now_iso()
    record = None
    for existing in registry.get("corpora", []):
        if existing.get("id") == cid:
            record = existing
            break
    if record is None:
        record = {
            "id": cid,
            "created_at": now,
            "status": "new",
            "document_count": 0,
            "chunk_count": 0,
            "last_indexed_at": None,
            "last_error": None,
        }
        registry.setdefault("corpora", []).append(record)
    record.update(
        {
            "name": name,
            "source_path": rel_source,
            "active": False,
            "updated_at": now,
        }
    )
    save_registry(root, registry)
    return record


def update_corpus_stats(config: Any, *, status: str, last_error: str | None = None) -> None:
    root = config.paths.root
    registry = load_registry(root)
    corpus_id = config.paths.corpus_id
    manifest = read_json(config.paths.cache_dir / "ingestion_manifest.json", {"documents": []})
    chunk_manifest = read_json(config.paths.cache_dir / "chunk_manifest.json", {})
    for corpus in registry.get("corpora", []):
        if corpus.get("id") == corpus_id:
            corpus["status"] = status
            corpus["document_count"] = len(manifest.get("documents", []))
            corpus["chunk_count"] = int(chunk_manifest.get("chunk_count", 0) or 0)
            corpus["last_error"] = last_error
            corpus["last_indexed_at"] = utc_now_iso() if status == "ready" else corpus.get("last_indexed_at")
            corpus["updated_at"] = utc_now_iso()
            break
    save_registry(root, registry)
