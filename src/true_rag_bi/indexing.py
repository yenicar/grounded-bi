from __future__ import annotations

from typing import Any

from .cache import CacheStore, build_corpus_fingerprint, build_pipeline_fingerprint
from .chunking import load_all_chunks, load_parent_chunks
from .config import AppConfig
from .providers import RAGProvider
from .utils import estimate_tokens, read_json, tokenize_for_search, utc_now_iso, write_json


def _get_collection(config: AppConfig):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for indexing. Install project dependencies first.") from exc

    fingerprint = build_pipeline_fingerprint(config)
    collection_name = f"bi_{config.paths.corpus_id}_{fingerprint.key()[:12]}"
    client = chromadb.PersistentClient(path=str(config.paths.chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)
    _cleanup_stale_collections(client, config.paths.corpus_id, collection_name)
    return collection, collection_name


def _cleanup_stale_collections(client: Any, corpus_id: str, active_name: str) -> None:
    # Remove collections left behind by prior pipeline fingerprints for this corpus.
    # Without this they accumulate on disk and slow ChromaDB startup.
    prefix = f"bi_{corpus_id}_"
    try:
        existing = client.list_collections()
    except Exception:
        return
    for collection in existing:
        name = getattr(collection, "name", collection)
        if isinstance(name, str) and name.startswith(prefix) and name != active_name:
            try:
                client.delete_collection(name)
            except Exception:
                continue


def _build_keyword_index(chunks: list[dict[str, Any]], corpus_fingerprint: str) -> dict[str, Any]:
    doc_frequencies: dict[str, int] = {}
    documents: list[dict[str, Any]] = []
    total_length = 0

    for chunk in chunks:
        tokens = tokenize_for_search(chunk["chunk_text"])
        token_count = len(tokens)
        total_length += token_count
        term_frequencies: dict[str, int] = {}
        for token in tokens:
            term_frequencies[token] = term_frequencies.get(token, 0) + 1
        for token in set(tokens):
            doc_frequencies[token] = doc_frequencies.get(token, 0) + 1
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "parent_chunk_id": chunk["parent_chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_name": chunk["doc_name"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section": chunk["section"],
                "content_hash": chunk["content_hash"],
                "parent_page_start": chunk["parent_page_start"],
                "parent_page_end": chunk["parent_page_end"],
                "parent_section": chunk["parent_section"],
                "parent_content_hash": chunk["parent_content_hash"],
                "has_table": bool(chunk.get("has_table")),
                "doc_date": chunk.get("doc_date"),
                "doc_author": chunk.get("doc_author"),
                "doc_type": chunk.get("doc_type"),
                "source_relative_path": chunk.get("source_relative_path", chunk["doc_name"]),
                "source_level": chunk.get("source_level"),
                "policy_category": chunk.get("policy_category"),
                "effective_date": chunk.get("effective_date"),
                "token_count": token_count,
                "term_frequencies": term_frequencies,
                # chunk_text intentionally omitted — already in JSONL files and ChromaDB.
                # Removing it keeps keyword_index.json ~60% smaller at scale.
            }
        )

    document_count = len(documents)
    avg_doc_length = round(total_length / document_count, 3) if document_count else 0.0
    return {
        "corpus_fingerprint": corpus_fingerprint,
        "document_count": document_count,
        "avg_doc_length": avg_doc_length,
        "doc_frequencies": doc_frequencies,
        "documents": documents,
        "updated_at": utc_now_iso(),
    }


def _build_graph_index(chunks: list[dict[str, Any]], corpus_fingerprint: str) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    cooccurrences: dict[str, dict[str, int]] = {}
    metadata_terms = {
        "policy",
        "category",
        "source",
        "level",
        "document",
        "summary",
        "page",
        "section",
    }

    for chunk in chunks:
        graph_text = " ".join(
            str(value or "")
            for value in [
                chunk.get("doc_name"),
                chunk.get("section"),
                chunk.get("policy_category"),
                chunk.get("source_level"),
                chunk.get("doc_type"),
                chunk.get("doc_summary"),
                chunk.get("chunk_text"),
            ]
        )
        terms = [
            term
            for term in tokenize_for_search(graph_text)
            if len(term) > 2 and term not in metadata_terms
        ]
        graph_terms = sorted(set(terms))[:80]
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "parent_chunk_id": chunk["parent_chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_name": chunk["doc_name"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section": chunk["section"],
                "content_hash": chunk["content_hash"],
                "parent_page_start": chunk["parent_page_start"],
                "parent_page_end": chunk["parent_page_end"],
                "parent_section": chunk["parent_section"],
                "parent_content_hash": chunk["parent_content_hash"],
                "has_table": bool(chunk.get("has_table")),
                "doc_date": chunk.get("doc_date"),
                "doc_author": chunk.get("doc_author"),
                "doc_type": chunk.get("doc_type"),
                "source_relative_path": chunk.get("source_relative_path", chunk["doc_name"]),
                "source_level": chunk.get("source_level"),
                "policy_category": chunk.get("policy_category"),
                "effective_date": chunk.get("effective_date"),
                "doc_summary": chunk.get("doc_summary", ""),
                "graph_terms": graph_terms,
            }
        )
        for index, term in enumerate(graph_terms[:24]):
            neighbors = cooccurrences.setdefault(term, {})
            for neighbor in graph_terms[index + 1 : index + 13]:
                neighbors[neighbor] = neighbors.get(neighbor, 0) + 1
                reverse = cooccurrences.setdefault(neighbor, {})
                reverse[term] = reverse.get(term, 0) + 1

    term_neighbors = {
        term: [
            neighbor
            for neighbor, _count in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:12]
        ]
        for term, neighbors in cooccurrences.items()
    }
    return {
        "corpus_fingerprint": corpus_fingerprint,
        "documents": documents,
        "term_neighbors": term_neighbors,
        "updated_at": utc_now_iso(),
    }


def index_documents(config: AppConfig, provider: RAGProvider, verbose: bool = False) -> dict[str, Any]:
    paths = config.paths
    cache = CacheStore(paths.cache_dir)
    chunks = load_all_chunks(config)
    parent_chunks = load_parent_chunks(config)
    parent_lookup = {chunk["chunk_id"]: chunk for chunk in parent_chunks}
    original_count = len(chunks)
    chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_text"].strip()
        and "[[NO_EXTRACTED_TEXT]]" not in chunk["chunk_text"]
        and chunk.get("token_estimate", 0) > 10
    ]
    if len(chunks) < original_count:
        print(f"[index] filtered {original_count - len(chunks)} empty/stub chunks before embedding")
    if not chunks:
        raise RuntimeError("No chunks available. Run ingestion and chunking first.")

    pipeline = build_pipeline_fingerprint(config)
    collection, collection_name = _get_collection(config)
    
    # --- Intelligent Ingestion: Summary-Augmented Pass ---
    # We generate a global summary for each document to provide 'global awareness' to local chunks.
    _augment_chunks_with_summaries(config, provider, chunks)
    
    embedding_manifest = cache.load("embedding_manifest", {"records": {}, "updated_at": None})
    records = embedding_manifest.get("records", {})

    current_ids = {chunk["chunk_id"] for chunk in chunks}
    stale_ids = [
        chunk_id
        for chunk_id, record in records.items()
        if record.get("collection_name") == collection_name and chunk_id not in current_ids
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)
        for chunk_id in stale_ids:
            records.pop(chunk_id, None)

    to_index: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    for chunk in chunks:
        record = records.get(chunk["chunk_id"])
        if (
            record
            and record.get("content_hash") == chunk["content_hash"]
            and record.get("collection_name") == collection_name
            and record.get("pipeline_key") == pipeline.key()
        ):
            cache_hits += 1
            continue
        cache_misses += 1
        to_index.append(chunk)

    batch_size = 12
    if to_index:
        import math

        n_batches = math.ceil(len(to_index) / batch_size)
        print(f"[index] embedding {len(to_index)} new chunks across {n_batches} batches...")
    else:
        print(f"[index] all {len(chunks)} chunks already indexed (full cache hit)")

    for offset in range(0, len(to_index), batch_size):
        batch = to_index[offset : offset + batch_size]
        embeddings = provider.embed_documents([chunk["chunk_text"] for chunk in batch])
        metadatas = [
            {
                "doc_id": chunk["doc_id"],
                "doc_name": chunk["doc_name"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section": chunk["section"],
                "token_estimate": chunk["token_estimate"],
                "content_hash": chunk["content_hash"],
                "has_table": bool(chunk.get("has_table")),
                "parent_chunk_id": chunk["parent_chunk_id"],
                "parent_page_start": chunk["parent_page_start"],
                "parent_page_end": chunk["parent_page_end"],
                "parent_section": chunk["parent_section"],
                "parent_content_hash": chunk["parent_content_hash"],
                "doc_date": chunk.get("doc_date"),
                "doc_author": chunk.get("doc_author"),
                "doc_type": chunk.get("doc_type"),
                "source_relative_path": chunk.get("source_relative_path", chunk["doc_name"]),
                "source_level": chunk.get("source_level"),
                "policy_category": chunk.get("policy_category"),
                "effective_date": chunk.get("effective_date"),
                "doc_summary": chunk.get("doc_summary", ""),
            }
            for chunk in batch
        ]
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["chunk_text"] for chunk in batch],
            metadatas=metadatas,
            embeddings=embeddings,
        )
        for chunk in batch:
            records[chunk["chunk_id"]] = {
                "collection_name": collection_name,
                "content_hash": chunk["content_hash"],
                "pipeline_key": pipeline.key(),
                "indexed_at": utc_now_iso(),
            }
        cache.save(
            "embedding_manifest",
            {
                "records": records,
                "updated_at": utc_now_iso(),
                "cache_stats": {"hits": cache_hits, "misses": cache_misses},
            },
        )
        if verbose and to_index:
            pct = round(100 * (offset + len(batch)) / len(to_index))
            print(f"[index] batch {offset // batch_size + 1}/{n_batches} ({pct}%) complete")

    corpus_fingerprint = build_corpus_fingerprint(chunks)
    keyword_index = _build_keyword_index(chunks, corpus_fingerprint)
    graph_index = _build_graph_index(chunks, corpus_fingerprint)
    write_json(paths.index_dir / "parent_lookup.json", {"parents": parent_lookup, "updated_at": utc_now_iso()})

    summary = {
        "collection_name": collection_name,
        "pipeline": pipeline.as_dict(),
        "pipeline_key": pipeline.key(),
        "corpus_fingerprint": corpus_fingerprint,
        "chunk_count": len(chunks),
        "parent_chunk_count": len(parent_chunks),
        "updated_at": utc_now_iso(),
        "cache_stats": {"hits": cache_hits, "misses": cache_misses},
    }
    write_json(paths.index_dir / "index_manifest.json", summary)
    write_json(paths.index_dir / "keyword_index.json", keyword_index)
    write_json(paths.index_dir / "graph_index.json", graph_index)
    return summary


def _augment_chunks_with_summaries(config: AppConfig, provider: RAGProvider, chunks: list[dict[str, Any]]) -> None:
    """Generates document-level summaries to enrich chunk metadata."""
    cache = CacheStore(config.paths.cache_dir)
    summary_cache = cache.load("summary_augmented_cache", {"documents": {}})
    doc_summaries = summary_cache.get("documents", {})
    
    # Identify unique documents and their full text
    doc_groups: dict[str, list[str]] = {}
    for chunk in chunks:
        doc_name = chunk["doc_name"]
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(chunk["chunk_text"])
    
    updated = False
    for doc_name, text_parts in doc_groups.items():
        if doc_name in doc_summaries:
            continue
            
        print(f"[index] generating intelligent summary for {doc_name}...")
        full_text = "\n\n".join(text_parts)
        # Truncate if too long for summary pass
        if estimate_tokens(full_text) > 8000:
            full_text = full_text[:32000] # Rough estimate
            
        summary_prompt = (
            "Summarize this document in 3-5 sentences for an executive RAG system. "
            "Identify the primary topic, key entities, and main conclusion. "
            "Return only the summary text.\n\n"
            f"Document: {doc_name}\nContent:\n{full_text}"
        )
        
        try:
            res = provider.generate_answer(summary_prompt)
            doc_summaries[doc_name] = res.text.strip()
            updated = True
        except Exception as exc:
            print(f"[index] summary failed for {doc_name}: {exc}")
            doc_summaries[doc_name] = "Summary unavailable."

    if updated:
        cache.save("summary_augmented_cache", {"documents": doc_summaries, "updated_at": utc_now_iso()})
    
    # Inject summaries into chunks
    for chunk in chunks:
        chunk["doc_summary"] = doc_summaries.get(chunk["doc_name"], "")
