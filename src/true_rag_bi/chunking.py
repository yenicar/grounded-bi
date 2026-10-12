from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import CacheStore
from .config import AppConfig
from .utils import estimate_tokens, read_jsonl, sha256_text, slugify, utc_now_iso, write_jsonl

_TABLE_ROW_MARKER = "\n| "


def _parse_blocks(markdown_text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_page = 1
    current_section = "Document"
    paragraph_lines: list[str] = []

    def flush() -> None:
        if not paragraph_lines:
            return
        text = "\n".join(paragraph_lines).strip()
        if text:
            blocks.append(
                {
                    "page": current_page,
                    "section": current_section,
                    "text": text,
                    "token_estimate": estimate_tokens(text),
                }
            )
        paragraph_lines.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## Page "):
            flush()
            try:
                current_page = int(line.split("Page ", 1)[1].strip())
            except ValueError:
                current_page = current_page
            current_section = line
            continue
        if line.startswith("#"):
            flush()
            current_section = line.lstrip("#").strip() or current_section
            continue
        if not line:
            flush()
            continue
        paragraph_lines.append(line)

    flush()
    return blocks


def _build_chunk_records(
    blocks: list[dict[str, Any]],
    *,
    doc_name: str,
    doc_id: int,
    target_tokens: int,
    overlap_tokens: int,
    id_prefix: str,
    base_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not blocks:
        return []

    chunks: list[dict[str, Any]] = []
    index = 0
    sequence = 1

    while index < len(blocks):
        start_index = index
        chunk_blocks: list[dict[str, Any]] = []
        token_count = 0

        while index < len(blocks):
            candidate = blocks[index]
            can_fit = token_count + candidate["token_estimate"] <= target_tokens
            if chunk_blocks and not can_fit and token_count >= int(target_tokens * 0.6):
                break
            chunk_blocks.append(candidate)
            token_count += candidate["token_estimate"]
            index += 1
            if token_count >= target_tokens:
                break

        if not chunk_blocks:
            chunk_blocks.append(blocks[index])
            token_count = chunk_blocks[0]["token_estimate"]
            index += 1

        chunk_text = "\n\n".join(block["text"] for block in chunk_blocks)
        content_hash = sha256_text(chunk_text)
        chunk_id = f"{slugify(doc_name)}-{id_prefix}-{sequence:03d}-{content_hash[:12]}"
        has_table = _TABLE_ROW_MARKER in f"\n{chunk_text}\n" and "\n| ---" in f"\n{chunk_text}\n"
        record = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page_start": chunk_blocks[0]["page"],
            "page_end": chunk_blocks[-1]["page"],
            "section": chunk_blocks[0]["section"],
            "chunk_text": chunk_text,
            "char_count": len(chunk_text),
            "token_estimate": token_count,
            "content_hash": content_hash,
            "has_table": has_table,
            "block_start": start_index,
            "block_end": start_index + len(chunk_blocks) - 1,
        }
        if base_payload:
            record.update(base_payload)
        chunks.append(record)
        sequence += 1

        if index >= len(blocks):
            break

        overlap_block_count = 0
        overlap_count = 0
        for block in reversed(chunk_blocks):
            overlap_count += block["token_estimate"]
            overlap_block_count += 1
            if overlap_count >= overlap_tokens:
                break
        index = max(start_index + 1, index - overlap_block_count)

    return chunks


def _build_parent_and_child_chunks(
    doc_name: str,
    doc_id: int,
    markdown_text: str,
    parent_target_tokens: int,
    parent_overlap_tokens: int,
    child_target_tokens: int,
    child_overlap_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = _parse_blocks(markdown_text)
    parent_chunks = _build_chunk_records(
        blocks,
        doc_name=doc_name,
        doc_id=doc_id,
        target_tokens=parent_target_tokens,
        overlap_tokens=parent_overlap_tokens,
        id_prefix="parent",
    )

    child_chunks: list[dict[str, Any]] = []
    child_sequence = 1
    for parent in parent_chunks:
        parent_blocks = blocks[parent["block_start"] : parent["block_end"] + 1]
        parent_metadata = {
            "parent_chunk_id": parent["chunk_id"],
            "parent_page_start": parent["page_start"],
            "parent_page_end": parent["page_end"],
            "parent_section": parent["section"],
            "parent_content_hash": parent["content_hash"],
        }
        parent_children = _build_chunk_records(
            parent_blocks,
            doc_name=doc_name,
            doc_id=doc_id,
            target_tokens=child_target_tokens,
            overlap_tokens=child_overlap_tokens,
            id_prefix="child",
            base_payload=parent_metadata,
        )
        for child in parent_children:
            child["child_sequence"] = child_sequence
            child_sequence += 1
        child_chunks.extend(parent_children)

    return parent_chunks, child_chunks


def split_markdown_into_chunks(
    doc_name: str,
    doc_id: int,
    markdown_text: str,
    target_tokens: int,
    overlap_tokens: int,
    child_target_tokens: int | None = None,
    child_overlap_tokens: int | None = None,
) -> list[dict[str, Any]]:
    if child_target_tokens is None or child_overlap_tokens is None:
        return _build_chunk_records(
            _parse_blocks(markdown_text),
            doc_name=doc_name,
            doc_id=doc_id,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            id_prefix="chunk",
        )

    _, child_chunks = _build_parent_and_child_chunks(
        doc_name=doc_name,
        doc_id=doc_id,
        markdown_text=markdown_text,
        parent_target_tokens=target_tokens,
        parent_overlap_tokens=overlap_tokens,
        child_target_tokens=child_target_tokens,
        child_overlap_tokens=child_overlap_tokens,
    )
    return child_chunks


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_all_chunks(config: AppConfig) -> list[dict[str, Any]]:
    return read_jsonl(config.paths.chunks_dir / "all_chunks.jsonl")


def load_parent_chunks(config: AppConfig) -> list[dict[str, Any]]:
    return read_jsonl(config.paths.chunks_dir / "all_parent_chunks.jsonl")


def chunk_documents(config: AppConfig, verbose: bool = False) -> dict[str, Any]:
    paths = config.paths
    cache = CacheStore(paths.cache_dir)
    ingestion_manifest = cache.load("ingestion_manifest", {"documents": []})
    chunk_manifest = cache.load("chunk_manifest", {"documents": [], "updated_at": None})
    previous_records = {record["doc_name"]: record for record in chunk_manifest.get("documents", [])}
    chunk_config_key = (
        f"parent:{config.chunk_target_tokens}:{config.chunk_overlap_tokens}|"
        f"child:{config.child_chunk_target_tokens}:{config.child_chunk_overlap_tokens}"
    )

    active_child_records: list[dict[str, Any]] = []
    active_parent_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    for doc in ingestion_manifest.get("documents", []):
        markdown_path = paths.root / doc["normalized_md_path"]
        artifact_id = doc.get("artifact_id") or slugify(doc["doc_name"])
        child_chunk_path = paths.chunks_dir / f"{artifact_id}.jsonl"
        parent_chunk_path = paths.chunks_dir / f"{artifact_id}.parents.jsonl"
        existing = previous_records.get(doc["doc_name"])

        if (
            existing
            and existing.get("content_hash") == doc["content_hash"]
            and existing.get("chunk_config_key") == chunk_config_key
            and child_chunk_path.exists()
            and parent_chunk_path.exists()
        ):
            cache_hits += 1
            manifest_record = dict(existing)
            manifest_record["last_chunked_at"] = utc_now_iso()
            manifest_records.append(manifest_record)
            active_child_records.extend(read_jsonl(child_chunk_path))
            active_parent_records.extend(read_jsonl(parent_chunk_path))
            if verbose:
                print(f"[chunk] cache hit for {doc['doc_name']}")
            continue

        cache_misses += 1
        markdown_text = _load_markdown(markdown_path)
        metadata_payload = {
            "doc_date": doc.get("doc_date"),
            "doc_author": doc.get("doc_author"),
            "doc_type": doc.get("doc_type"),
            "source_relative_path": doc.get("source_relative_path", doc["doc_name"]),
            "source_level": doc.get("source_level"),
            "policy_category": doc.get("policy_category"),
            "effective_date": doc.get("effective_date"),
        }
        parent_chunks, child_chunks = _build_parent_and_child_chunks(
            doc_name=doc["doc_name"],
            doc_id=doc["doc_id"],
            markdown_text=markdown_text,
            parent_target_tokens=config.chunk_target_tokens,
            parent_overlap_tokens=config.chunk_overlap_tokens,
            child_target_tokens=config.child_chunk_target_tokens,
            child_overlap_tokens=config.child_chunk_overlap_tokens,
        )
        for record in parent_chunks:
            record.update(metadata_payload)
        for record in child_chunks:
            record.update(metadata_payload)
        write_jsonl(child_chunk_path, child_chunks)
        write_jsonl(parent_chunk_path, parent_chunks)
        active_child_records.extend(child_chunks)
        active_parent_records.extend(parent_chunks)
        manifest_records.append(
            {
                "doc_name": doc["doc_name"],
                "content_hash": doc["content_hash"],
                "chunk_config_key": chunk_config_key,
                "chunk_file_path": str(child_chunk_path.relative_to(paths.root)),
                "parent_chunk_file_path": str(parent_chunk_path.relative_to(paths.root)),
                "chunk_count": len(child_chunks),
                "parent_chunk_count": len(parent_chunks),
                "last_chunked_at": utc_now_iso(),
            }
        )
        if verbose:
            print(
                f"[chunk] generated {len(child_chunks)} child chunks and "
                f"{len(parent_chunks)} parent chunks for {doc['doc_name']}"
            )

    if cache_misses > 0:
        write_jsonl(paths.chunks_dir / "all_chunks.jsonl", active_child_records)
        write_jsonl(paths.chunks_dir / "all_parent_chunks.jsonl", active_parent_records)
    payload = {
        "documents": manifest_records,
        "updated_at": utc_now_iso(),
        "cache_stats": {"hits": cache_hits, "misses": cache_misses},
        "chunk_count": len(active_child_records),
        "parent_chunk_count": len(active_parent_records),
        "chunk_config_key": chunk_config_key,
    }
    cache.save("chunk_manifest", payload)
    return payload
