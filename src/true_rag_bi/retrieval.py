from __future__ import annotations

import json as json_lib
import math
import re
import textwrap
import time
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from .cache import CacheStore, build_pipeline_fingerprint, build_query_cache_key
from .config import AppConfig
from .providers import RAGProvider
from .utils import estimate_tokens, normalize_query, read_json, tokenize_for_search, utc_now_iso

_QUERY_STOPWORDS = {
    "and",
    "about",
    "against",
    "between",
    "could",
    "does",
    "evidence",
    "for",
    "from",
    "have",
    "into",
    "its",
    "of",
    "our",
    "over",
    "more",
    "puts",
    "should",
    "that",
    "than",
    "their",
    "them",
    "there",
    "these",
    "this",
    "through",
    "under",
    "what",
    "which",
    "with",
    "would",
}
_QUANT_TERMS = {
    # Budgets and costs
    "budget",
    "cost",
    "costs",
    "capital",
    "capex",
    "opex",
    "expense",
    "expenses",
    "spend",
    "spending",
    # Revenue and profitability
    "revenue",
    "revenues",
    "profit",
    "profitability",
    "margin",
    "margins",
    "gross",
    "net",
    "loss",
    "losses",
    "earnings",
    "ebitda",
    "ebit",
    # Valuation and return
    "roi",
    "return",
    "cagr",
    "yield",
    "dividend",
    "valuation",
    "equity",
    "debt",
    "leverage",
    # Cash and liquidity
    "cashflow",
    "cash",
    "runway",
    "burn",
    "liquidity",
    # Pricing
    "pricing",
    "price",
    "prices",
    "premium",
    # Currencies and scale
    "usd",
    "eur",
    "gbp",
    "million",
    "billion",
    "percent",
    # Breakeven and targets
    "break-even",
    "breakeven",
    "target",
    "targets",
    "projection",
    "projections",
    "forecast",
    # Time periods
    "q1",
    "q2",
    "q3",
    "q4",
    "year",
    "years",
    "month",
    "months",
    "annual",
    "quarterly",
    # Comparative
    "table",
    "tables",
    "versus",
    "vs",
    "compare",
    "comparison",
    "increase",
    "decrease",
    "growth",
    "decline",
}

_COMPARE_TERMS = {
    "align",
    "aligned",
    "alignment",
    "alignments",
    "compare",
    "compared",
    "comparison",
    "contrast",
    "differ",
    "differs",
    "difference",
    "differences",
    "different",
    "versus",
    "vs",
}


@dataclass(slots=True)
class RetrievalCandidate:
    chunk_id: str
    parent_chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    section: str
    content_hash: str
    parent_page_start: int
    parent_page_end: int
    parent_section: str
    parent_content_hash: str
    has_table: bool
    similarity_score: float
    effective_score: float
    text: str
    child_text: str
    matched_terms: list[str]
    selected: bool
    selection_reason: str
    keyword_score: float = 0.0
    graph_score: float = 0.0
    fusion_score: float = 0.0
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    graph_rank: int | None = None
    retrieval_channels: list[str] = field(default_factory=list)
    supporting_child_chunk_ids: list[str] = field(default_factory=list)
    doc_date: str | None = None
    doc_author: str | None = None
    doc_type: str | None = None
    source_relative_path: str | None = None
    source_level: str | None = None
    policy_category: str | None = None
    effective_date: str | None = None
    doc_summary: str | None = None


def _load_index_manifest(config: AppConfig) -> dict[str, Any]:
    manifest_path = config.paths.index_dir / "index_manifest.json"
    manifest = read_json(manifest_path, {})
    if not manifest.get("collection_name"):
        raise RuntimeError("Index manifest not found. Run `bi-rag index` first.")
    return manifest


def _load_collection(config: AppConfig, manifest: dict[str, Any] | None = None):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for retrieval. Install project dependencies first.") from exc

    manifest = manifest or _load_index_manifest(config)
    collection_name = manifest["collection_name"]

    client = chromadb.PersistentClient(path=str(config.paths.chroma_dir))
    return client.get_collection(collection_name), manifest


def _load_parent_lookup(config: AppConfig, expected_corpus_fingerprint: str) -> dict[str, dict[str, Any]]:
    payload = read_json(config.paths.index_dir / "parent_lookup.json", {})
    parents = payload.get("parents", {})
    if not parents:
        raise RuntimeError("Parent lookup not found. Run `bi-rag index` first.")
    return parents


def _flatten_query_results(
    results: dict[str, Any],
    query_terms: set[str],
    parent_lookup: dict[str, dict[str, Any]],
) -> list[RetrievalCandidate]:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

    candidates: list[RetrievalCandidate] = []
    for rank, (chunk_id, metadata, text, distance) in enumerate(zip(ids, metadatas, documents, distances), start=1):
        lower_text = text.lower()
        matched_terms = sorted(term for term in query_terms if term in lower_text)
        similarity = 0.0 if distance is None else max(0.0, 1.0 - float(distance))
        parent_chunk_id = metadata["parent_chunk_id"]
        parent = parent_lookup.get(parent_chunk_id, {})
        candidates.append(
            RetrievalCandidate(
                chunk_id=chunk_id,
                parent_chunk_id=parent_chunk_id,
                doc_name=metadata["doc_name"],
                page_start=int(metadata["page_start"]),
                page_end=int(metadata["page_end"]),
                section=metadata.get("section", "Document"),
                content_hash=metadata["content_hash"],
                parent_page_start=int(metadata.get("parent_page_start", metadata["page_start"])),
                parent_page_end=int(metadata.get("parent_page_end", metadata["page_end"])),
                parent_section=metadata.get("parent_section", metadata.get("section", "Document")),
                parent_content_hash=metadata.get("parent_content_hash", metadata["content_hash"]),
                has_table=bool(metadata.get("has_table")),
                similarity_score=round(similarity, 4),
                effective_score=round(similarity, 4),
                text=parent.get("chunk_text", text),
                child_text=text,
                matched_terms=matched_terms[:6],
                selected=False,
                selection_reason="retrieved",
                semantic_rank=rank,
                retrieval_channels=["semantic"],
                supporting_child_chunk_ids=[chunk_id],
                doc_date=metadata.get("doc_date"),
                doc_author=metadata.get("doc_author"),
                doc_type=metadata.get("doc_type"),
                source_relative_path=metadata.get("source_relative_path"),
                source_level=metadata.get("source_level"),
                policy_category=metadata.get("policy_category"),
                effective_date=metadata.get("effective_date"),
                doc_summary=metadata.get("doc_summary"),
            )
        )
    return candidates


_keyword_index_cache: dict[str, dict[str, Any]] = {}
_graph_index_cache: dict[str, dict[str, Any]] = {}


def _load_keyword_index(config: AppConfig, expected_corpus_fingerprint: str) -> dict[str, Any]:
    global _keyword_index_cache
    cache_key = f"{config.paths.corpus_id}:{expected_corpus_fingerprint}"
    if cache_key in _keyword_index_cache:
        return _keyword_index_cache[cache_key]
    if False:
        # Validate cached fingerprint still matches
        if _keyword_index_cache.get("corpus_fingerprint") == expected_corpus_fingerprint:
            return _keyword_index_cache
        # Fingerprint changed — invalidate and reload
        _keyword_index_cache = None

    keyword_index = read_json(config.paths.index_dir / "keyword_index.json", {})
    if not keyword_index:
        raise RuntimeError("Keyword index not found. Run `bi-rag index` first.")
    if keyword_index.get("corpus_fingerprint") != expected_corpus_fingerprint:
        raise RuntimeError("Keyword index is out of date for the current corpus. Run `bi-rag index` again.")
    _keyword_index_cache[cache_key] = keyword_index
    return _keyword_index_cache[cache_key]


def _load_graph_index(config: AppConfig, expected_corpus_fingerprint: str) -> dict[str, Any]:
    global _graph_index_cache
    cache_key = f"{config.paths.corpus_id}:{expected_corpus_fingerprint}"
    if cache_key in _graph_index_cache:
        return _graph_index_cache[cache_key]
    graph_index = read_json(config.paths.index_dir / "graph_index.json", {})
    if not graph_index:
        return {}
    if graph_index.get("corpus_fingerprint") != expected_corpus_fingerprint:
        return {}
    _graph_index_cache[cache_key] = graph_index
    return graph_index


def _score_keyword_candidates(
    keyword_index: dict[str, Any],
    parent_lookup: dict[str, dict[str, Any]],
    query_terms: list[str],
    fetch_k: int,
    after: str | None = None,
) -> list[RetrievalCandidate]:
    if not query_terms:
        return []

    doc_frequencies = keyword_index.get("doc_frequencies", {})
    documents = keyword_index.get("documents", [])
    document_count = max(1, int(keyword_index.get("document_count", len(documents) or 1)))
    avg_doc_length = float(keyword_index.get("avg_doc_length", 0.0) or 1.0)
    k1 = 1.5
    b = 0.75
    scored: list[RetrievalCandidate] = []

    for document in documents:
        doc_date = document.get("doc_date")
        if after and (not doc_date or doc_date < after):
            continue
        score = 0.0
        term_frequencies = document.get("term_frequencies", {})
        doc_length = max(1, int(document.get("token_count", 0) or 1))
        matched_terms = [term for term in query_terms if term_frequencies.get(term)]
        if not matched_terms:
            continue
        for term in matched_terms:
            tf = term_frequencies.get(term, 0)
            df = doc_frequencies.get(term, 0)
            idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
        parent_chunk_id = document["parent_chunk_id"]
        parent = parent_lookup.get(parent_chunk_id, {})
        scored.append(
            RetrievalCandidate(
                chunk_id=document["chunk_id"],
                parent_chunk_id=parent_chunk_id,
                doc_name=document["doc_name"],
                page_start=int(document["page_start"]),
                page_end=int(document["page_end"]),
                section=document.get("section", "Document"),
                content_hash=document["content_hash"],
                parent_page_start=int(document.get("parent_page_start", document["page_start"])),
                parent_page_end=int(document.get("parent_page_end", document["page_end"])),
                parent_section=document.get("parent_section", document.get("section", "Document")),
                parent_content_hash=document.get("parent_content_hash", document["content_hash"]),
                has_table=bool(document.get("has_table")),
                similarity_score=0.0,
                effective_score=0.0,
                # chunk_text was removed from keyword_index.json (Phase 2 optimization).
                # Use parent_lookup for the full context text; child_text is unavailable from
                # this path but parent text is all we need for generation.
                text=parent.get("chunk_text", ""),
                child_text=parent.get("chunk_text", ""),
                matched_terms=sorted(set(matched_terms))[:6],
                selected=False,
                selection_reason="retrieved",
                keyword_score=round(score, 4),
                retrieval_channels=["keyword"],
                supporting_child_chunk_ids=[document["chunk_id"]],
                doc_date=document.get("doc_date"),
                doc_author=document.get("doc_author"),
                doc_type=document.get("doc_type"),
                source_relative_path=document.get("source_relative_path") or parent.get("source_relative_path"),
                source_level=document.get("source_level") or parent.get("source_level"),
                policy_category=document.get("policy_category") or parent.get("policy_category"),
                effective_date=document.get("effective_date") or parent.get("effective_date"),
                doc_summary=document.get("doc_summary") or parent.get("doc_summary"),
            )
        )

    top_keyword = sorted(scored, key=lambda item: item.keyword_score, reverse=True)[:fetch_k]
    for rank, candidate in enumerate(top_keyword, start=1):
        candidate.keyword_rank = rank
    return top_keyword


def _score_graph_candidates(
    graph_index: dict[str, Any],
    parent_lookup: dict[str, dict[str, Any]],
    query_terms: list[str],
    fetch_k: int,
    after: str | None = None,
) -> list[RetrievalCandidate]:
    if not query_terms or not graph_index:
        return []

    term_neighbors = graph_index.get("term_neighbors", {})
    expanded_terms = set(query_terms)
    for term in query_terms:
        expanded_terms.update(str(item) for item in term_neighbors.get(term, [])[:12])

    scored: list[RetrievalCandidate] = []
    for document in graph_index.get("documents", []):
        doc_date = document.get("doc_date")
        if after and (not doc_date or doc_date < after):
            continue

        graph_terms = set(str(term) for term in document.get("graph_terms", []))
        category = str(document.get("policy_category") or "").lower()
        source_level = str(document.get("source_level") or "").lower()
        summary_terms = set(tokenize_for_search(document.get("doc_summary") or ""))

        direct_matches = graph_terms & set(query_terms)
        neighbor_matches = graph_terms & expanded_terms
        metadata_matches = {
            term
            for term in query_terms
            if term and (term == category or term == source_level or term in summary_terms)
        }
        if not direct_matches and not neighbor_matches and not metadata_matches:
            continue

        score = len(direct_matches) * 2.0 + len(neighbor_matches) * 0.8 + len(metadata_matches) * 1.2
        parent_chunk_id = document["parent_chunk_id"]
        parent = parent_lookup.get(parent_chunk_id, {})
        matched_terms = sorted(direct_matches | neighbor_matches | metadata_matches)[:8]
        scored.append(
            RetrievalCandidate(
                chunk_id=document["chunk_id"],
                parent_chunk_id=parent_chunk_id,
                doc_name=document["doc_name"],
                page_start=int(document["page_start"]),
                page_end=int(document["page_end"]),
                section=document.get("section", "Document"),
                content_hash=document["content_hash"],
                parent_page_start=int(document.get("parent_page_start", document["page_start"])),
                parent_page_end=int(document.get("parent_page_end", document["page_end"])),
                parent_section=document.get("parent_section", document.get("section", "Document")),
                parent_content_hash=document.get("parent_content_hash", document["content_hash"]),
                has_table=bool(document.get("has_table")),
                similarity_score=0.0,
                effective_score=0.0,
                text=parent.get("chunk_text", ""),
                child_text=parent.get("chunk_text", ""),
                matched_terms=matched_terms,
                selected=False,
                selection_reason="retrieved",
                graph_score=round(score, 4),
                retrieval_channels=["graph"],
                supporting_child_chunk_ids=[document["chunk_id"]],
                doc_date=document.get("doc_date"),
                doc_author=document.get("doc_author"),
                doc_type=document.get("doc_type"),
                source_relative_path=document.get("source_relative_path") or parent.get("source_relative_path"),
                source_level=document.get("source_level") or parent.get("source_level"),
                policy_category=document.get("policy_category") or parent.get("policy_category"),
                effective_date=document.get("effective_date") or parent.get("effective_date"),
                doc_summary=document.get("doc_summary") or parent.get("doc_summary"),
            )
        )

    top_graph = sorted(scored, key=lambda item: item.graph_score, reverse=True)[:fetch_k]
    for rank, candidate in enumerate(top_graph, start=1):
        candidate.graph_rank = rank
    return top_graph


def _fuse_candidates(
    semantic_candidates: list[RetrievalCandidate],
    keyword_candidates: list[RetrievalCandidate],
    rrf_k: int,
    graph_candidates: list[RetrievalCandidate] | None = None,
) -> list[RetrievalCandidate]:
    by_chunk_id: dict[str, RetrievalCandidate] = {}

    for candidate in semantic_candidates:
        candidate.fusion_score = 1 / (rrf_k + (candidate.semantic_rank or 0))
        by_chunk_id[candidate.chunk_id] = candidate

    for keyword_candidate in keyword_candidates:
        existing = by_chunk_id.get(keyword_candidate.chunk_id)
        keyword_contribution = 1 / (rrf_k + (keyword_candidate.keyword_rank or 0))
        if existing:
            existing.keyword_score = keyword_candidate.keyword_score
            existing.keyword_rank = keyword_candidate.keyword_rank
            existing.fusion_score = round(existing.fusion_score + keyword_contribution, 6)
            existing.matched_terms = sorted(set(existing.matched_terms + keyword_candidate.matched_terms))[:6]
            if "keyword" not in existing.retrieval_channels:
                existing.retrieval_channels.append("keyword")
            continue

        keyword_candidate.fusion_score = round(keyword_contribution, 6)
        by_chunk_id[keyword_candidate.chunk_id] = keyword_candidate

    for graph_candidate in graph_candidates or []:
        existing = by_chunk_id.get(graph_candidate.chunk_id)
        graph_contribution = 1 / (rrf_k + (graph_candidate.graph_rank or 0))
        if existing:
            existing.graph_score = graph_candidate.graph_score
            existing.graph_rank = graph_candidate.graph_rank
            existing.fusion_score = round(existing.fusion_score + graph_contribution, 6)
            existing.matched_terms = sorted(set(existing.matched_terms + graph_candidate.matched_terms))[:8]
            if "graph" not in existing.retrieval_channels:
                existing.retrieval_channels.append("graph")
            continue

        graph_candidate.fusion_score = round(graph_contribution, 6)
        by_chunk_id[graph_candidate.chunk_id] = graph_candidate

    fused = list(by_chunk_id.values())
    fused.sort(key=lambda item: (item.fusion_score, item.similarity_score, item.keyword_score, item.graph_score), reverse=True)
    return fused


def _collapse_to_parent_contexts(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    by_parent_id: dict[str, RetrievalCandidate] = {}

    for candidate in candidates:
        existing = by_parent_id.get(candidate.parent_chunk_id)
        if not existing:
            by_parent_id[candidate.parent_chunk_id] = candidate
            candidate.chunk_id = candidate.parent_chunk_id
            candidate.page_start = candidate.parent_page_start
            candidate.page_end = candidate.parent_page_end
            candidate.section = candidate.parent_section
            candidate.content_hash = candidate.parent_content_hash
            continue

        existing.fusion_score = round(existing.fusion_score + candidate.fusion_score, 6)
        existing.similarity_score = max(existing.similarity_score, candidate.similarity_score)
        existing.keyword_score = max(existing.keyword_score, candidate.keyword_score)
        existing.graph_score = max(existing.graph_score, candidate.graph_score)
        existing.semantic_rank = min(
            value for value in [existing.semantic_rank, candidate.semantic_rank] if value is not None
        ) if (existing.semantic_rank is not None or candidate.semantic_rank is not None) else None
        existing.keyword_rank = min(
            value for value in [existing.keyword_rank, candidate.keyword_rank] if value is not None
        ) if (existing.keyword_rank is not None or candidate.keyword_rank is not None) else None
        existing.graph_rank = min(
            value for value in [existing.graph_rank, candidate.graph_rank] if value is not None
        ) if (existing.graph_rank is not None or candidate.graph_rank is not None) else None
        existing.matched_terms = sorted(set(existing.matched_terms + candidate.matched_terms))[:6]
        existing.supporting_child_chunk_ids = sorted(
            set(existing.supporting_child_chunk_ids + candidate.supporting_child_chunk_ids)
        )
        existing.has_table = existing.has_table or candidate.has_table
        for channel in candidate.retrieval_channels:
            if channel not in existing.retrieval_channels:
                existing.retrieval_channels.append(channel)
        if candidate.doc_summary and not existing.doc_summary:
            existing.doc_summary = candidate.doc_summary

    collapsed = list(by_parent_id.values())
    collapsed.sort(key=lambda item: (item.fusion_score, item.similarity_score, item.keyword_score, item.graph_score), reverse=True)
    return collapsed


def _is_quantitative_query(query_terms: list[str], question: str) -> bool:
    normalized_question = normalize_query(question)
    return any(term in _QUANT_TERMS for term in query_terms) or any(token in normalized_question for token in _QUANT_TERMS)


def _is_comparison_query(question: str) -> bool:
    terms = set(tokenize_for_search(question))
    return bool(terms & _COMPARE_TERMS)


def _answer_shape_instruction(question: str) -> str:
    if not _is_comparison_query(question):
        return ""
    return textwrap.dedent(
        """
        The question asks for a comparison. Structure the answer with exactly these sections:
        1. Alignments
        2. Differences / Divergences
        3. Evidence gaps
        In "Differences / Divergences", only include differences directly supported by retrieved context.
        If no retrieved context supports a difference, say "The retrieved evidence does not show a specific difference for this point."
        Do not relabel UW-specific implementation details as differences unless the context contrasts them against a Federal requirement.
        """
    ).strip()


def _apply_table_boost(candidates: list[RetrievalCandidate], *, is_quantitative: bool) -> None:
    if not is_quantitative:
        return
    for candidate in candidates:
        if candidate.has_table:
            candidate.fusion_score = round(candidate.fusion_score + 0.02, 6)
            if "table_boost" not in candidate.retrieval_channels:
                candidate.retrieval_channels.append("table_boost")


def _max_chunks_per_doc(*, is_quantitative: bool, top_k: int) -> int:
    if top_k <= 4:
        return 1
    return 2 if is_quantitative else 3


def apply_diversity_filter(
    candidates: list[RetrievalCandidate],
    top_k: int,
    *,
    is_quantitative: bool = False,
) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    doc_counts: dict[str, int] = {}
    seen_hashes: set[str] = set()
    max_per_doc = _max_chunks_per_doc(is_quantitative=is_quantitative, top_k=top_k)

    for candidate in sorted(candidates, key=lambda item: (item.fusion_score, item.similarity_score, item.keyword_score, item.graph_score), reverse=True):
        if candidate.parent_content_hash in seen_hashes:
            candidate.effective_score = 0.0
            candidate.selection_reason = "dropped_duplicate_content"
            continue

        if doc_counts.get(candidate.doc_name, 0) >= max_per_doc:
            candidate.effective_score = 0.0
            candidate.selection_reason = "trimmed_doc_limit"
            continue

        doc_penalty = 0.08 * doc_counts.get(candidate.doc_name, 0)
        base_score = candidate.fusion_score or candidate.similarity_score or candidate.keyword_score
        candidate.effective_score = round(base_score - doc_penalty, 4)
        if len(selected) < top_k:
            candidate.selected = True
            if candidate.graph_rank and (candidate.semantic_rank or candidate.keyword_rank):
                candidate.selection_reason = "selected_graph_hybrid_match"
            elif candidate.semantic_rank and candidate.keyword_rank:
                candidate.selection_reason = "selected_hybrid_match"
            elif candidate.graph_rank:
                candidate.selection_reason = "selected_graph_match"
            elif doc_counts.get(candidate.doc_name):
                candidate.selection_reason = "selected_with_doc_diversity_penalty"
            elif candidate.keyword_rank:
                candidate.selection_reason = "selected_keyword_match"
            else:
                candidate.selection_reason = "selected_top_similarity"
            selected.append(candidate)
            seen_hashes.add(candidate.parent_content_hash)
            doc_counts[candidate.doc_name] = doc_counts.get(candidate.doc_name, 0) + 1
        else:
            candidate.selection_reason = "trimmed_after_diversity_filter"

    return selected


def _format_citation(candidate: RetrievalCandidate) -> str:
    if candidate.page_start == candidate.page_end:
        page_text = f"p.{candidate.page_start}"
    else:
        page_text = f"pp.{candidate.page_start}-{candidate.page_end}"
    return f"{candidate.doc_name} {page_text} chunk:{candidate.chunk_id}"


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    if not words:
        return text
    trimmed = words[:]
    while trimmed and estimate_tokens(" ".join(trimmed)) > max_tokens:
        trimmed = trimmed[:- max(1, len(trimmed) // 8)]
    return " ".join(trimmed).strip()


def _split_context_passages(text: str) -> list[str]:
    passages = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]
    return passages or [text.strip()]


def _build_context_snippet(candidate: RetrievalCandidate, query_terms: list[str], max_tokens: int) -> str:
    passages = _split_context_passages(candidate.text)
    scored: list[tuple[int, str]] = []
    for passage in passages:
        lowered = passage.lower()
        score = sum(2 for term in query_terms if term and term in lowered)
        if candidate.child_text and candidate.child_text.strip() and candidate.child_text.strip() in passage:
            score += 3
        if candidate.has_table and "|" in passage:
            score += 2
        scored.append((score, passage))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected_passages: list[str] = []
    budget_remaining = max_tokens
    for _, passage in scored:
        passage_tokens = estimate_tokens(passage)
        if selected_passages and passage_tokens > budget_remaining:
            continue
        clipped = _truncate_to_token_budget(passage, min(passage_tokens, budget_remaining))
        if clipped:
            selected_passages.append(clipped)
            budget_remaining -= estimate_tokens(clipped)
        if budget_remaining <= 40:
            break

    if not selected_passages:
        fallback = candidate.child_text or candidate.text
        return _truncate_to_token_budget(fallback, max_tokens)

    return "\n\n".join(selected_passages).strip()


def _select_prompt_contexts(
    candidates: list[RetrievalCandidate],
    *,
    query_terms: list[str],
    max_context_chunks: int,
    max_context_tokens: int,
    max_chunk_tokens: int,
) -> list[RetrievalCandidate]:
    prompt_contexts: list[RetrievalCandidate] = []
    remaining = max_context_tokens

    for candidate in candidates:
        if len(prompt_contexts) >= max_context_chunks:
            break
        if remaining <= 80:
            break
        snippet_budget = min(max_chunk_tokens, remaining)
        snippet_text = _build_context_snippet(candidate, query_terms, snippet_budget)
        snippet_tokens = estimate_tokens(snippet_text)
        if not snippet_text or snippet_tokens <= 0:
            continue
        prompt_contexts.append(replace(candidate, text=snippet_text))
        remaining -= snippet_tokens

    return prompt_contexts

def _build_prompt(
    question: str,
    selected: list[RetrievalCandidate],
    history: list[dict[str, str]] | None,
    graph_context: str | None = None,
) -> str:
    history_text = ""
    if history:
        turns = []
        for entry in history[-6:]:
            turns.append(f"User: {entry['user']}")
            turns.append(f"Assistant: {entry['assistant']}")
        history_text = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

    context_lines = []
    unique_summaries = {}
    for index, candidate in enumerate(selected, start=1):
        citation = _format_citation(candidate)
        context_lines.append(f"[Context {index}] {citation}\n{candidate.text}")
        if candidate.doc_summary and candidate.doc_name not in unique_summaries:
            unique_summaries[candidate.doc_name] = candidate.doc_summary
    
    context_text = "\n\n".join(context_lines)
    
    summary_text = ""
    if unique_summaries:
        summary_text = "Relevant Document Summaries:\n" + "\n".join(
            f"- {doc}: {sum_text}" for doc, sum_text in unique_summaries.items()
        ) + "\n\n"

    graph_text = ""
    if graph_context:
        graph_text = "Global Holistic Context (from Knowledge Graph):\n" + graph_context + "\n\n"
    task_instruction = _answer_shape_instruction(question)
    if task_instruction:
        task_instruction = f"Task-specific answer requirements:\n{task_instruction}\n\n"

    return textwrap.dedent(
        f"""
        You are a careful business intelligence analyst.
        Use only the retrieved context below.
        If the evidence is insufficient, say so plainly.
        Do not invent facts, numbers, or citations.
        Do not mention any source, context, document, or page that is not included below.
        Do not reference "Context 5" style labels in the answer.
        Write a concise answer for an internal analyst.
        Prefer short bullets or short paragraphs.
        Omit any detail that is not explicitly supported by the retrieved context.
        Do not guess missing stakeholders, dates, motives, or numbers.
        {task_instruction}

        {history_text}{graph_text}{summary_text}Question:
        {question}

        Retrieved context:
        {context_text}

        End your answer with one short confidence line in this format:
        Confidence: High|Medium|Low
        """
    ).strip()


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    try:
        payload = json_lib.loads(raw_text)
        return payload if isinstance(payload, dict) else None
    except json_lib.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json_lib.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except json_lib.JSONDecodeError:
            return None


def _ensure_confidence_line(answer: str, verdict: str) -> str:
    if re.search(r"Confidence:\s*(High|Medium|Low)\s*$", answer, re.IGNORECASE):
        return answer.strip()
    if verdict == "grounded":
        confidence = "High"
    elif verdict == "trimmed":
        confidence = "Medium"
    else:
        confidence = "Low"
    return f"{answer.strip()}\n\nConfidence: {confidence}".strip()


def _build_grounding_prompt(question: str, draft_answer: str, contexts: list[RetrievalCandidate]) -> str:
    context_lines = []
    for index, candidate in enumerate(contexts, start=1):
        citation = _format_citation(candidate)
        context_lines.append(f"[Context {index}] {citation}\n{candidate.text}")
    context_text = "\n\n".join(context_lines)
    return textwrap.dedent(
        f"""
        You are validating a draft RAG answer for strict groundedness.
        Remove or rewrite any statement that is not directly supported by the retrieved context.
        Do not add new facts.
        If the context is insufficient, say so plainly and briefly.
        Keep the answer concise.
        End the final answer with Confidence: High|Medium|Low.

        Question:
        {question}

        Retrieved context:
        {context_text}

        Draft answer:
        {draft_answer}

        Return ONLY valid JSON with this exact schema:
        {{
          "grounded_answer": "<revised answer>",
          "unsupported_claims_removed": ["<claim 1>", "<claim 2>"],
          "verdict": "grounded|trimmed|insufficient"
        }}
        """
    ).strip()


def _ground_answer(
    provider: RAGProvider,
    *,
    question: str,
    draft_answer: str,
    contexts: list[RetrievalCandidate],
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    grounding_prompt = _build_grounding_prompt(question, draft_answer, contexts)
    review = provider.generate_answer(grounding_prompt)
    payload = _extract_json_object(review.text.strip()) or {}

    grounded_answer = payload.get("grounded_answer")
    if not isinstance(grounded_answer, str) or not grounded_answer.strip():
        grounded_answer = draft_answer
    unsupported = payload.get("unsupported_claims_removed")
    if not isinstance(unsupported, list):
        unsupported = []
    unsupported = [str(item) for item in unsupported if str(item).strip()]
    verdict = str(payload.get("verdict", "parse_failed")).strip().lower()
    if verdict not in {"grounded", "trimmed", "insufficient"}:
        verdict = "trimmed" if unsupported else "grounded"

    grounded_answer = _ensure_confidence_line(grounded_answer, verdict)
    debug = {
        "draft_answer": draft_answer,
        "grounded_answer": grounded_answer,
        "unsupported_claims_removed": unsupported,
        "verdict": verdict,
        "review_raw": review.text.strip(),
    }
    return grounded_answer, debug, grounding_prompt, review.usage


def _estimate_costs(records: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for prompt, answer, usage in records:
        prompt_tokens += int(usage.get("prompt_token_count") or estimate_tokens(prompt))
        completion_tokens += int(usage.get("candidates_token_count") or estimate_tokens(answer))
        total_tokens += int(
            usage.get("total_token_count")
            or (
                int(usage.get("prompt_token_count") or estimate_tokens(prompt))
                + int(usage.get("candidates_token_count") or estimate_tokens(answer))
            )
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _rewrite_query_for_retrieval(
    provider: RAGProvider,
    question: str,
    history: list[dict[str, str]],
) -> str:
    if not history:
        return question
    turns = "\n".join(
        f"User: {entry['user']}\nAssistant: {entry['assistant'][:300]}"
        for entry in history[-3:]
    )
    prompt = (
        "Rewrite the follow-up question as a fully self-contained standalone question "
        "that can be answered without the conversation history. "
        "Output only the rewritten question, nothing else.\n\n"
        f"Conversation so far:\n{turns}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )
    try:
        rewritten = provider.generate_answer(prompt).text.strip()
        return rewritten or question
    except Exception:
        return question


def ask_question(
    config: AppConfig,
    provider: RAGProvider,
    question: str,
    debug: bool = False,
    verbose: bool = False,
    history: list[dict[str, str]] | None = None,
    after: str | None = None,
    grounding_enabled: bool | None = None,
) -> dict[str, Any]:
    request_started = time.perf_counter()
    effective_grounding_enabled = (
        getattr(config, "answer_grounding_enabled", False) if grounding_enabled is None else grounding_enabled
    )
    normalized_question = normalize_query(question)
    retrieval_question = _rewrite_query_for_retrieval(provider, question, history) if history else question
    normalized_retrieval_question = normalize_query(retrieval_question)
    index_manifest = _load_index_manifest(config)
    pipeline = build_pipeline_fingerprint(config)
    corpus_fingerprint = index_manifest["corpus_fingerprint"]
    cache_key = build_query_cache_key(
        f"{normalized_question}||{normalized_retrieval_question}||grounding:{int(effective_grounding_enabled)}",
        pipeline.key(),
        corpus_fingerprint,
    )
    legacy_cache_key = build_query_cache_key(
        f"{normalized_question}||{normalized_retrieval_question}",
        pipeline.key(),
        corpus_fingerprint,
    )

    cache = CacheStore(config.paths.cache_dir)
    cache_started = time.perf_counter()
    query_cache = cache.load("query_cache", {"entries": {}, "updated_at": None})
    entries = query_cache.get("entries", {})
    cached = entries.get(cache_key)
    if not cached and not effective_grounding_enabled:
        cached = entries.get(legacy_cache_key)
    cache_elapsed = max(0.0, time.perf_counter() - cache_started)

    cache_stats = {
        "answer_cache": "miss",
        "query_embedding": "miss",
        "retrieval_prep": "miss",
    }

    if cached and cached.get("response"):
        cache_stats["answer_cache"] = "hit"
        response = dict(cached["response"])
        metrics = dict(response.get("metrics", {}))
        cold_metrics = {
            key: metrics.get(key)
            for key in (
                "retrieval_seconds",
                "generation_seconds",
                "grounding_seconds",
                "total_seconds",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "remote_calls",
                "prompt_context_tokens",
            )
            if key in metrics
        }
        request_elapsed = max(0.0, time.perf_counter() - request_started)
        current_elapsed = round(cache_elapsed, 3)
        metrics.update(
            {
                "retrieval_seconds": round(request_elapsed, 3),
                "generation_seconds": 0.0,
                "grounding_seconds": 0.0,
                "total_seconds": round(request_elapsed, 3),
                "remote_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_answer_tokens": estimate_tokens(str(response.get("answer", ""))),
                "cache_served": True,
                "cached_response": True,
                "cache_lookup_seconds": current_elapsed,
                "grounding_enabled": effective_grounding_enabled,
                "cold_run_metrics": cold_metrics,
            }
        )
        response["metrics"] = metrics
        response["cache_stats"] = cache_stats
        response.setdefault("retrieval_question", retrieval_question)
        if not debug:
            response["retrieval_debug"] = {}
        return response

    query_embedding: list[float]
    if cached and cached.get("query_embedding"):
        cache_stats["query_embedding"] = "hit"
        query_embedding = cached["query_embedding"]
    else:
        query_embedding = provider.embed_query(retrieval_question)

    retrieval_started = time.perf_counter()
    collection, _ = _load_collection(config, index_manifest)
    keyword_index = _load_keyword_index(config, corpus_fingerprint)
    graph_index = _load_graph_index(config, corpus_fingerprint)
    parent_lookup = _load_parent_lookup(config, corpus_fingerprint)
    query_terms = [
        term for term in tokenize_for_search(retrieval_question)
        if len(term) > 2 and term not in _QUERY_STOPWORDS
    ]
    is_quantitative_query = _is_quantitative_query(query_terms, retrieval_question)
    query_term_set = set(query_terms)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.retrieval_fetch_k,
        include=["documents", "metadatas", "distances"],
        where={"doc_date": {"$gte": after}} if after else None,
    )
    semantic_candidates = _flatten_query_results(results, query_term_set, parent_lookup)
    keyword_candidates = _score_keyword_candidates(keyword_index, parent_lookup, query_terms, config.retrieval_fetch_k, after=after)
    graph_candidates = _score_graph_candidates(graph_index, parent_lookup, query_terms, config.retrieval_fetch_k, after=after)
    raw_candidates = _collapse_to_parent_contexts(
        _fuse_candidates(semantic_candidates, keyword_candidates, config.retrieval_rrf_k, graph_candidates=graph_candidates)
    )
    _apply_table_boost(raw_candidates, is_quantitative=is_quantitative_query)
    selected = apply_diversity_filter(raw_candidates, config.retrieval_top_k, is_quantitative=is_quantitative_query)
    prompt_contexts = _select_prompt_contexts(
        selected,
        query_terms=query_terms,
        max_context_chunks=config.prompt_max_context_chunks,
        max_context_tokens=config.prompt_max_context_tokens,
        max_chunk_tokens=config.prompt_max_chunk_tokens,
    )
    retrieval_elapsed = time.perf_counter() - retrieval_started
    cache_stats["retrieval_prep"] = "hit" if raw_candidates else "miss"

    if not selected:
        response = {
            "question": question,
            "retrieval_question": retrieval_question,
            "answer": "I could not find enough grounded evidence in the indexed corpus to answer that confidently.",
            "citations": [],
            "retrieval_debug": {"candidates": [], "selected": []},
            "cache_stats": cache_stats,
            "metrics": {
                "retrieval_seconds": round(retrieval_elapsed, 3),
                "generation_seconds": 0.0,
                "total_seconds": round(time.perf_counter() - request_started, 3),
                "remote_calls": 1,
                "grounding_enabled": effective_grounding_enabled,
            },
            "filters": {"after": after} if after else {},
            "generated_at": utc_now_iso(),
        }
        entries[cache_key] = {"query_embedding": query_embedding, "response": response}
        _trim_query_cache(entries, config.query_cache_limit)
        cache.save("query_cache", {"entries": entries, "updated_at": utc_now_iso()})
        return response

    if not prompt_contexts:
        prompt_contexts = selected[: max(1, min(len(selected), config.prompt_max_context_chunks))]

    prompt = _build_prompt(question, prompt_contexts, history)
    generation_started = time.perf_counter()
    generation = provider.generate_answer(prompt)
    generation_elapsed = time.perf_counter() - generation_started
    grounding_prompt = ""
    grounding_usage: dict[str, Any] = {}
    grounding_elapsed = 0.0
    if effective_grounding_enabled:
        grounding_started = time.perf_counter()
        final_answer, grounding_debug, grounding_prompt, grounding_usage = _ground_answer(
            provider,
            question=question,
            draft_answer=generation.text,
            contexts=prompt_contexts,
        )
        grounding_elapsed = time.perf_counter() - grounding_started
    else:
        final_answer = _ensure_confidence_line(generation.text, "grounded")
        grounding_debug = {
            "draft_answer": generation.text,
            "grounded_answer": final_answer,
            "unsupported_claims_removed": [],
            "verdict": "not_run",
            "review_raw": "",
        }

    citations = [_format_citation(candidate) for candidate in prompt_contexts]
    debug_payload = {
        "candidates": [
            {
                "chunk_id": candidate.chunk_id,
                "doc_name": candidate.doc_name,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "section": candidate.section,
                "parent_chunk_id": candidate.parent_chunk_id,
                "has_table": candidate.has_table,
                "similarity_score": candidate.similarity_score,
                "keyword_score": candidate.keyword_score,
                "graph_score": candidate.graph_score,
                "fusion_score": candidate.fusion_score,
                "semantic_rank": candidate.semantic_rank,
                "keyword_rank": candidate.keyword_rank,
                "graph_rank": candidate.graph_rank,
                "retrieval_channels": candidate.retrieval_channels,
                "supporting_child_chunk_ids": candidate.supporting_child_chunk_ids,
                "source_relative_path": candidate.source_relative_path,
                "source_level": candidate.source_level,
                "policy_category": candidate.policy_category,
                "effective_date": candidate.effective_date,
                "effective_score": candidate.effective_score,
                "matched_terms": candidate.matched_terms,
                "selected": candidate.selected,
                "selection_reason": candidate.selection_reason,
            }
            for candidate in raw_candidates
        ],
        "retrieved": [
            {
                "context_order": order,
                "chunk_id": candidate.chunk_id,
                "parent_chunk_id": candidate.parent_chunk_id,
                "citation": _format_citation(candidate),
                "doc_name": candidate.doc_name,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "has_table": candidate.has_table,
                "matched_terms": candidate.matched_terms,
                "similarity_score": candidate.similarity_score,
                "keyword_score": candidate.keyword_score,
                "graph_score": candidate.graph_score,
                "fusion_score": candidate.fusion_score,
                "semantic_rank": candidate.semantic_rank,
                "keyword_rank": candidate.keyword_rank,
                "graph_rank": candidate.graph_rank,
                "retrieval_channels": candidate.retrieval_channels,
                "supporting_child_chunk_ids": candidate.supporting_child_chunk_ids,
                "source_relative_path": candidate.source_relative_path,
                "source_level": candidate.source_level,
                "policy_category": candidate.policy_category,
                "effective_date": candidate.effective_date,
                "effective_score": candidate.effective_score,
                "selection_reason": candidate.selection_reason,
                "text": candidate.text,
                "child_text": candidate.child_text,
            }
            for order, candidate in enumerate(selected, start=1)
        ],
        "selected": [
            {
                "context_order": order,
                "chunk_id": candidate.chunk_id,
                "parent_chunk_id": candidate.parent_chunk_id,
                "citation": _format_citation(candidate),
                "doc_name": candidate.doc_name,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "has_table": candidate.has_table,
                "matched_terms": candidate.matched_terms,
                "similarity_score": candidate.similarity_score,
                "keyword_score": candidate.keyword_score,
                "graph_score": candidate.graph_score,
                "fusion_score": candidate.fusion_score,
                "semantic_rank": candidate.semantic_rank,
                "keyword_rank": candidate.keyword_rank,
                "graph_rank": candidate.graph_rank,
                "retrieval_channels": candidate.retrieval_channels,
                "supporting_child_chunk_ids": candidate.supporting_child_chunk_ids,
                "source_relative_path": candidate.source_relative_path,
                "source_level": candidate.source_level,
                "policy_category": candidate.policy_category,
                "effective_date": candidate.effective_date,
                "effective_score": candidate.effective_score,
                "selection_reason": candidate.selection_reason,
                "text": candidate.text,
                "child_text": candidate.child_text,
            }
            for order, candidate in enumerate(prompt_contexts, start=1)
        ],
        "grounding": grounding_debug,
    }

    api_calls = [(prompt, generation.text, generation.usage)]
    if effective_grounding_enabled:
        api_calls.append((grounding_prompt, final_answer, grounding_usage))

    metrics = _estimate_costs(api_calls)
    metrics.update(
        {
            "retrieval_seconds": round(retrieval_elapsed, 3),
            "generation_seconds": round(generation_elapsed, 3),
            "grounding_seconds": round(grounding_elapsed, 3),
            "total_seconds": round(time.perf_counter() - request_started, 3),
            "remote_calls": (2 if history else 1) + (1 if effective_grounding_enabled else 0),
            "retrieved_contexts": len(selected),
            "prompt_contexts": len(prompt_contexts),
            "prompt_context_tokens": sum(estimate_tokens(candidate.text) for candidate in prompt_contexts),
            "grounding_verdict": grounding_debug["verdict"],
            "grounding_enabled": effective_grounding_enabled,
            "unsupported_claims_removed": len(grounding_debug["unsupported_claims_removed"]),
            "graph_candidates": len(graph_candidates),
        }
    )

    response = {
        "question": question,
        "retrieval_question": retrieval_question,
        "answer": final_answer,
        "citations": citations,
        "retrieval_debug": debug_payload,
        "cache_stats": cache_stats,
        "metrics": metrics,
        "filters": {"after": after} if after else {},
        "generated_at": utc_now_iso(),
    }

    entries[cache_key] = {"query_embedding": query_embedding, "response": response}
    _trim_query_cache(entries, config.query_cache_limit)
    cache.save("query_cache", {"entries": entries, "updated_at": utc_now_iso()})

    if verbose:
        print(f"[ask] retrieved {len(selected)} grounded chunks")

    if debug:
        return response

    redacted_response = dict(response)
    redacted_response["retrieval_debug"] = {}
    return redacted_response


def _trim_query_cache(entries: dict[str, Any], limit: int) -> None:
    if len(entries) <= limit:
        return
    ordered = sorted(
        entries.items(),
        key=lambda item: item[1].get("response", {}).get("generated_at", item[1].get("created_at", "")),
    )
    while len(ordered) > limit:
        key, _ = ordered.pop(0)
        entries.pop(key, None)


def chat_session(config: AppConfig, provider: RAGProvider, debug: bool = False, verbose: bool = False) -> None:
    history: list[dict[str, str]] = []
    print("Business Intelligence RAG chat is ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = ask_question(config, provider, question, debug=debug, verbose=verbose, history=history)
        print("\nAssistant>")
        print(result["answer"])
        if result["citations"]:
            print("\nSources:")
            for citation in result["citations"]:
                print(f"- {citation}")
        if debug and result["retrieval_debug"]:
            print("\nRetrieval debug:")
            for item in result["retrieval_debug"]["selected"]:
                print(
                    f"- order={item['context_order']} chunk={item['chunk_id']} "
                    f"score={item['effective_score']} fusion={item['fusion_score']} "
                    f"channels={','.join(item['retrieval_channels']) or 'none'} "
                    f"terms={','.join(item['matched_terms']) or 'none'} reason={item['selection_reason']}"
                )

        history.append({"user": question, "assistant": result["answer"]})
        history = history[-config.chat_history_turns :]
