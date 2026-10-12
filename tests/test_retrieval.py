from pathlib import Path
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.retrieval import (
    RetrievalCandidate,
    _ensure_confidence_line,
    _extract_json_object,
    _build_context_snippet,
    _build_prompt,
    _apply_table_boost,
    _fuse_candidates,
    _is_quantitative_query,
    _score_graph_candidates,
    _select_prompt_contexts,
    _score_keyword_candidates,
    _is_comparison_query,
    apply_diversity_filter,
    ask_question,
)


class RetrievalTests(unittest.TestCase):
    def test_diversity_filter_prefers_unique_content(self) -> None:
        candidates = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="Doc1.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=False,
                similarity_score=0.95,
                effective_score=0.95,
                text="Budget pressure is rising in Asia.",
                child_text="Budget pressure is rising in Asia.",
                matched_terms=["budget", "asia"],
                selected=False,
                selection_reason="retrieved",
                supporting_child_chunk_ids=["a"],
            ),
            RetrievalCandidate(
                chunk_id="b",
                parent_chunk_id="parent-a",
                doc_name="Doc1.pdf",
                page_start=2,
                page_end=2,
                section="Page 2",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=False,
                similarity_score=0.94,
                effective_score=0.94,
                text="Budget pressure is rising in Asia.",
                child_text="Budget pressure is rising in Asia.",
                matched_terms=["budget", "asia"],
                selected=False,
                selection_reason="retrieved",
                supporting_child_chunk_ids=["b"],
            ),
            RetrievalCandidate(
                chunk_id="c",
                parent_chunk_id="parent-c",
                doc_name="Doc2.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-c",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-c",
                has_table=False,
                similarity_score=0.80,
                effective_score=0.80,
                text="Europe has stricter but more predictable compliance requirements.",
                child_text="Europe has stricter but more predictable compliance requirements.",
                matched_terms=["europe"],
                selected=False,
                selection_reason="retrieved",
                supporting_child_chunk_ids=["c"],
            ),
        ]

        selected = apply_diversity_filter(candidates, top_k=2)
        self.assertEqual([item.chunk_id for item in selected], ["a", "c"])
        self.assertEqual(candidates[1].effective_score, 0.0)
        self.assertEqual(candidates[1].selection_reason, "dropped_duplicate_content")

    def test_keyword_scoring_prefers_exact_business_term(self) -> None:
        keyword_index = {
            "document_count": 2,
            "avg_doc_length": 5.0,
            "doc_frequencies": {"acme": 1, "budget": 2, "asia": 1},
            "documents": [
                {
                    "chunk_id": "a",
                    "parent_chunk_id": "parent-a",
                    "doc_name": "Doc1.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "section": "Page 1",
                    "content_hash": "hash-a",
                    "parent_page_start": 1,
                    "parent_page_end": 1,
                    "parent_section": "Page 1",
                    "parent_content_hash": "parent-hash-a",
                    "token_count": 5,
                    "term_frequencies": {"acme": 1, "budget": 1},
                    "chunk_text": "Acme budget pressure",
                },
                {
                    "chunk_id": "b",
                    "parent_chunk_id": "parent-b",
                    "doc_name": "Doc2.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "section": "Page 1",
                    "content_hash": "hash-b",
                    "parent_page_start": 1,
                    "parent_page_end": 1,
                    "parent_section": "Page 1",
                    "parent_content_hash": "parent-hash-b",
                    "token_count": 5,
                    "term_frequencies": {"budget": 1, "asia": 1},
                    "chunk_text": "Asia budget outlook",
                },
            ],
        }
        parent_lookup = {
            "parent-a": {"chunk_text": "Parent A context"},
            "parent-b": {"chunk_text": "Parent B context"},
        }

        candidates = _score_keyword_candidates(keyword_index, parent_lookup, ["acme", "budget"], fetch_k=4)
        self.assertEqual(candidates[0].chunk_id, "a")
        self.assertGreater(candidates[0].keyword_score, candidates[1].keyword_score)
        self.assertEqual(candidates[0].keyword_rank, 1)
        self.assertEqual(candidates[0].text, "Parent A context")

    def test_fusion_combines_semantic_and_keyword_signals(self) -> None:
        semantic = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="Doc1.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=False,
                similarity_score=0.9,
                effective_score=0.9,
                text="Acme budget pressure",
                child_text="Acme budget pressure",
                matched_terms=["budget"],
                selected=False,
                selection_reason="retrieved",
                semantic_rank=2,
                retrieval_channels=["semantic"],
                supporting_child_chunk_ids=["a"],
            ),
            RetrievalCandidate(
                chunk_id="b",
                parent_chunk_id="parent-b",
                doc_name="Doc2.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-b",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-b",
                has_table=False,
                similarity_score=0.95,
                effective_score=0.95,
                text="Asia growth outlook",
                child_text="Asia growth outlook",
                matched_terms=["asia"],
                selected=False,
                selection_reason="retrieved",
                semantic_rank=1,
                retrieval_channels=["semantic"],
                supporting_child_chunk_ids=["b"],
            ),
        ]
        keyword = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="Doc1.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=False,
                similarity_score=0.0,
                effective_score=0.0,
                text="Acme budget pressure",
                child_text="Acme budget pressure",
                matched_terms=["acme", "budget"],
                selected=False,
                selection_reason="retrieved",
                keyword_score=4.2,
                keyword_rank=1,
                retrieval_channels=["keyword"],
                supporting_child_chunk_ids=["a"],
            ),
        ]

        fused = _fuse_candidates(semantic, keyword, rrf_k=60)
        self.assertEqual(fused[0].chunk_id, "a")
        self.assertIn("semantic", fused[0].retrieval_channels)
        self.assertIn("keyword", fused[0].retrieval_channels)
        self.assertGreater(fused[0].fusion_score, fused[1].fusion_score)

    def test_graph_scoring_surfaces_related_chunk_when_exact_text_misses(self) -> None:
        graph_index = {
            "documents": [
                {
                    "chunk_id": "a",
                    "parent_chunk_id": "parent-a",
                    "doc_name": "LeavePolicy.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "section": "Page 1",
                    "content_hash": "hash-a",
                    "parent_page_start": 1,
                    "parent_page_end": 1,
                    "parent_section": "Page 1",
                    "parent_content_hash": "parent-hash-a",
                    "has_table": False,
                    "policy_category": "leave",
                    "source_level": "Federal",
                    "doc_summary": "Covers family and medical leave eligibility.",
                    "graph_terms": ["fmla", "absence", "eligibility"],
                },
                {
                    "chunk_id": "b",
                    "parent_chunk_id": "parent-b",
                    "doc_name": "ConductPolicy.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "section": "Page 1",
                    "content_hash": "hash-b",
                    "parent_page_start": 1,
                    "parent_page_end": 1,
                    "parent_section": "Page 1",
                    "parent_content_hash": "parent-hash-b",
                    "has_table": False,
                    "policy_category": "conduct",
                    "source_level": "UW System",
                    "doc_summary": "Covers workplace conduct.",
                    "graph_terms": ["discipline", "ethics"],
                },
            ],
            "term_neighbors": {
                "fmla": ["leave", "absence", "eligibility"],
                "leave": ["fmla", "absence"],
            },
        }
        parent_lookup = {
            "parent-a": {"chunk_text": "Employees must satisfy eligibility criteria."},
            "parent-b": {"chunk_text": "Employees must follow conduct expectations."},
        }

        candidates = _score_graph_candidates(
            graph_index,
            parent_lookup,
            ["fmla", "leave"],
            fetch_k=4,
        )

        self.assertEqual(candidates[0].chunk_id, "a")
        self.assertEqual(candidates[0].graph_rank, 1)
        self.assertGreater(candidates[0].graph_score, 0)
        self.assertIn("graph", candidates[0].retrieval_channels)

    def test_fusion_combines_graph_signal_with_semantic_and_keyword(self) -> None:
        semantic = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="LeavePolicy.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=False,
                similarity_score=0.72,
                effective_score=0.72,
                text="Eligibility criteria.",
                child_text="Eligibility criteria.",
                matched_terms=["eligibility"],
                selected=False,
                selection_reason="retrieved",
                semantic_rank=2,
                retrieval_channels=["semantic"],
                supporting_child_chunk_ids=["a"],
            )
        ]
        keyword: list[RetrievalCandidate] = []
        graph = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="LeavePolicy.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=False,
                similarity_score=0.0,
                effective_score=0.0,
                text="Eligibility criteria.",
                child_text="Eligibility criteria.",
                matched_terms=["fmla", "leave"],
                selected=False,
                selection_reason="retrieved",
                graph_score=3.0,
                graph_rank=1,
                retrieval_channels=["graph"],
                supporting_child_chunk_ids=["a"],
            )
        ]

        fused = _fuse_candidates(semantic, keyword, rrf_k=60, graph_candidates=graph)

        self.assertEqual(fused[0].chunk_id, "a")
        self.assertIn("semantic", fused[0].retrieval_channels)
        self.assertIn("graph", fused[0].retrieval_channels)
        self.assertGreater(fused[0].fusion_score, 1 / 62)

    def test_table_boost_applies_for_quantitative_query(self) -> None:
        candidates = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="Doc1.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=True,
                similarity_score=0.5,
                effective_score=0.5,
                text="Parent table context",
                child_text="Child table context",
                matched_terms=["budget", "usd"],
                selected=False,
                selection_reason="retrieved",
                fusion_score=0.05,
            ),
            RetrievalCandidate(
                chunk_id="b",
                parent_chunk_id="parent-b",
                doc_name="Doc2.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-b",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-b",
                has_table=False,
                similarity_score=0.5,
                effective_score=0.5,
                text="Narrative context",
                child_text="Narrative child context",
                matched_terms=["budget"],
                selected=False,
                selection_reason="retrieved",
                fusion_score=0.05,
            ),
        ]
        self.assertTrue(_is_quantitative_query(["budget", "usd"], "Compare budget in USD"))
        _apply_table_boost(candidates, is_quantitative=True)
        self.assertGreater(candidates[0].fusion_score, candidates[1].fusion_score)
        self.assertIn("table_boost", candidates[0].retrieval_channels)

    def test_diversity_filter_limits_same_document_for_quantitative_queries(self) -> None:
        candidates = [
            RetrievalCandidate(
                chunk_id="a",
                parent_chunk_id="parent-a",
                doc_name="Doc1.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-a",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-a",
                has_table=True,
                similarity_score=0.9,
                effective_score=0.9,
                text="Doc1 table A",
                child_text="Doc1 child A",
                matched_terms=["budget", "usd"],
                selected=False,
                selection_reason="retrieved",
                fusion_score=0.09,
            ),
            RetrievalCandidate(
                chunk_id="b",
                parent_chunk_id="parent-b",
                doc_name="Doc1.pdf",
                page_start=2,
                page_end=2,
                section="Page 2",
                content_hash="hash-b",
                parent_page_start=2,
                parent_page_end=2,
                parent_section="Page 2",
                parent_content_hash="parent-hash-b",
                has_table=True,
                similarity_score=0.85,
                effective_score=0.85,
                text="Doc1 table B",
                child_text="Doc1 child B",
                matched_terms=["budget", "usd"],
                selected=False,
                selection_reason="retrieved",
                fusion_score=0.085,
            ),
            RetrievalCandidate(
                chunk_id="c",
                parent_chunk_id="parent-c",
                doc_name="Doc1.pdf",
                page_start=3,
                page_end=3,
                section="Page 3",
                content_hash="hash-c",
                parent_page_start=3,
                parent_page_end=3,
                parent_section="Page 3",
                parent_content_hash="parent-hash-c",
                has_table=False,
                similarity_score=0.8,
                effective_score=0.8,
                text="Doc1 narrative",
                child_text="Doc1 child C",
                matched_terms=["budget"],
                selected=False,
                selection_reason="retrieved",
                fusion_score=0.08,
            ),
            RetrievalCandidate(
                chunk_id="d",
                parent_chunk_id="parent-d",
                doc_name="Doc2.pdf",
                page_start=1,
                page_end=1,
                section="Page 1",
                content_hash="hash-d",
                parent_page_start=1,
                parent_page_end=1,
                parent_section="Page 1",
                parent_content_hash="parent-hash-d",
                has_table=False,
                similarity_score=0.79,
                effective_score=0.79,
                text="Doc2 narrative",
                child_text="Doc2 child D",
                matched_terms=["budget"],
                selected=False,
                selection_reason="retrieved",
                fusion_score=0.079,
            ),
        ]
        selected = apply_diversity_filter(candidates, top_k=4, is_quantitative=True)
        self.assertEqual([item.chunk_id for item in selected], ["a", "d"])
        self.assertEqual(candidates[1].selection_reason, "trimmed_doc_limit")
        self.assertEqual(candidates[2].selection_reason, "trimmed_doc_limit")

    def test_build_context_snippet_prefers_matching_passages(self) -> None:
        candidate = RetrievalCandidate(
            chunk_id="a",
            parent_chunk_id="parent-a",
            doc_name="Doc1.pdf",
            page_start=1,
            page_end=2,
            section="Page 1",
            content_hash="hash-a",
            parent_page_start=1,
            parent_page_end=2,
            parent_section="Page 1",
            parent_content_hash="parent-hash-a",
            has_table=False,
            similarity_score=0.91,
            effective_score=0.91,
            text=(
                "General introduction.\n\n"
                "Asia budget pressure rose sharply due to capital needs and marketing spend.\n\n"
                "Unrelated closing note."
            ),
            child_text="Asia budget pressure rose sharply due to capital needs and marketing spend.",
            matched_terms=["asia", "budget"],
            selected=False,
            selection_reason="retrieved",
            supporting_child_chunk_ids=["a"],
        )
        snippet = _build_context_snippet(candidate, ["asia", "budget"], 40)
        self.assertIn("Asia budget pressure", snippet)

    def test_select_prompt_contexts_respects_budget(self) -> None:
        candidates = []
        for index in range(4):
            candidates.append(
                RetrievalCandidate(
                    chunk_id=f"c{index}",
                    parent_chunk_id=f"p{index}",
                    doc_name=f"Doc{index}.pdf",
                    page_start=1,
                    page_end=1,
                    section="Page 1",
                    content_hash=f"hash-{index}",
                    parent_page_start=1,
                    parent_page_end=1,
                    parent_section="Page 1",
                    parent_content_hash=f"parent-hash-{index}",
                    has_table=False,
                    similarity_score=0.8 - index * 0.01,
                    effective_score=0.8 - index * 0.01,
                    text=("budget asia " * 180).strip(),
                    child_text=("budget asia " * 40).strip(),
                    matched_terms=["budget", "asia"],
                    selected=True,
                    selection_reason="selected_top_similarity",
                    supporting_child_chunk_ids=[f"c{index}"],
                )
            )
        prompt_candidates = _select_prompt_contexts(
            candidates,
            query_terms=["budget", "asia"],
            max_context_chunks=2,
            max_context_tokens=120,
            max_chunk_tokens=70,
        )
        self.assertLessEqual(len(prompt_candidates), 2)
        self.assertTrue(all(item.text for item in prompt_candidates))

    def test_extract_json_object_recovers_embedded_payload(self) -> None:
        raw = 'prefix {"grounded_answer":"Only supported facts.\\n\\nConfidence: Low","unsupported_claims_removed":["x"],"verdict":"trimmed"} suffix'
        payload = _extract_json_object(raw)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["verdict"], "trimmed")

    def test_ensure_confidence_line_adds_default_for_trimmed_answers(self) -> None:
        answer = _ensure_confidence_line("Only supported facts remain.", "trimmed")
        self.assertTrue(answer.endswith("Confidence: Medium"))

    def test_comparison_prompt_requires_alignments_and_differences(self) -> None:
        prompt = _build_prompt(
            "Where do Federal and UW System policies align or differ?",
            [],
            None,
        )
        self.assertTrue(_is_comparison_query("Where do Federal and UW System policies align or differ?"))
        self.assertIn("Alignments", prompt)
        self.assertIn("Differences / Divergences", prompt)
        self.assertIn("Evidence gaps", prompt)
        self.assertIn("The retrieved evidence does not show a specific difference", prompt)

    def test_cache_hit_reports_current_run_metrics(self) -> None:
        class FakeCache:
            def __init__(self, _cache_dir) -> None:
                pass

            def load(self, name, default):
                if name == "query_cache":
                    return {
                        "entries": {
                            "cache-key": {
                                "response": {
                                    "answer": "Cached answer\n\nConfidence: High",
                                    "citations": [],
                                    "confidence": "High",
                                    "metrics": {
                                        "retrieval_seconds": 12.5,
                                        "generation_seconds": 8.0,
                                        "grounding_seconds": 10.0,
                                        "total_seconds": 30.5,
                                        "prompt_tokens": 9000,
                                        "completion_tokens": 500,
                                        "total_tokens": 9500,
                                        "remote_calls": 2,
                                    },
                                    "retrieval_question": "What changed?",
                                    "retrieval_debug": {"selected": []},
                                }
                            }
                        }
                    }
                return default

            def save(self, name, payload):
                raise AssertionError("cache hit should not rewrite cache")

        with mock.patch("true_rag_bi.retrieval.CacheStore", FakeCache), \
            mock.patch("true_rag_bi.retrieval._load_index_manifest", return_value={"collection_name": "chunks", "corpus_fingerprint": "corpus"}), \
            mock.patch("true_rag_bi.retrieval._load_keyword_index", return_value={}), \
            mock.patch("true_rag_bi.retrieval._load_graph_index", return_value={}), \
            mock.patch("true_rag_bi.retrieval._load_parent_lookup", return_value={}), \
            mock.patch("true_rag_bi.retrieval.build_query_cache_key", return_value="cache-key"):
            config = SimpleNamespace(
                paths=SimpleNamespace(cache_dir=Path(".")),
                version="test",
                provider_name="fake",
                embedding_model="fake-embed",
                answer_model="fake-answer",
                chunk_target_tokens=800,
                chunk_overlap_tokens=120,
                child_chunk_target_tokens=220,
                child_chunk_overlap_tokens=40,
                retrieval_top_k=8,
                retrieval_fetch_k=16,
                retrieval_rrf_k=60,
                answer_grounding_enabled=False,
            )
            result = ask_question(config, object(), "What changed?")

        metrics = result["metrics"]
        self.assertEqual(metrics["remote_calls"], 0)
        self.assertEqual(metrics["prompt_tokens"], 0)
        self.assertEqual(metrics["completion_tokens"], 0)
        self.assertEqual(metrics["total_tokens"], 0)
        self.assertGreaterEqual(metrics["retrieval_seconds"], 0.0)
        self.assertGreaterEqual(metrics["retrieval_seconds"], metrics["cache_lookup_seconds"])
        self.assertEqual(metrics["generation_seconds"], 0.0)
        self.assertEqual(metrics["grounding_seconds"], 0.0)
        self.assertGreater(metrics["cached_answer_tokens"], 0)
        self.assertFalse(metrics["grounding_enabled"])
        self.assertTrue(metrics["cache_served"])
        self.assertEqual(metrics["cold_run_metrics"]["total_tokens"], 9500)


if __name__ == "__main__":
    unittest.main()
