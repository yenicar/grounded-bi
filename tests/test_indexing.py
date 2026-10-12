from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.indexing import _build_graph_index, _build_keyword_index, _cleanup_stale_collections


class IndexingTests(unittest.TestCase):
    def _sample_chunks(self) -> list[dict]:
        return [
            {
                "chunk_id": "doc1-child-001-abc",
                "parent_chunk_id": "doc1-parent-001-xyz",
                "doc_id": 1,
                "doc_name": "Report.pdf",
                "page_start": 1,
                "page_end": 1,
                "section": "Page 1",
                "content_hash": "hash-a",
                "parent_page_start": 1,
                "parent_page_end": 2,
                "parent_section": "Page 1",
                "parent_content_hash": "parent-hash-a",
                "has_table": False,
                "chunk_text": "Asia expansion budget overview for Q2.",
                "token_estimate": 8,
            },
            {
                "chunk_id": "doc1-child-002-def",
                "parent_chunk_id": "doc1-parent-001-xyz",
                "doc_id": 1,
                "doc_name": "Report.pdf",
                "page_start": 2,
                "page_end": 2,
                "section": "Page 2",
                "content_hash": "hash-b",
                "parent_page_start": 1,
                "parent_page_end": 2,
                "parent_section": "Page 1",
                "parent_content_hash": "parent-hash-a",
                "has_table": True,
                "chunk_text": "| Region | Budget |\n| --- | --- |\n| Asia | 70M |",
                "token_estimate": 14,
            },
        ]

    def test_keyword_index_builds_correct_document_count(self) -> None:
        chunks = self._sample_chunks()
        index = _build_keyword_index(chunks, "fingerprint-abc")
        self.assertEqual(index["document_count"], 2)

    def test_keyword_index_contains_expected_terms(self) -> None:
        chunks = self._sample_chunks()
        index = _build_keyword_index(chunks, "fingerprint-abc")
        doc_freq = index["doc_frequencies"]
        self.assertIn("asia", doc_freq)
        self.assertIn("budget", doc_freq)
        self.assertIn("region", doc_freq)

    def test_keyword_index_has_table_flag_preserved(self) -> None:
        chunks = self._sample_chunks()
        index = _build_keyword_index(chunks, "fingerprint-abc")
        has_table_flags = [doc["has_table"] for doc in index["documents"]]
        self.assertIn(True, has_table_flags)
        self.assertIn(False, has_table_flags)

    def test_graph_index_contains_terms_and_neighbors(self) -> None:
        chunks = self._sample_chunks()
        index = _build_graph_index(chunks, "fingerprint-abc")
        self.assertEqual(index["corpus_fingerprint"], "fingerprint-abc")
        self.assertEqual(len(index["documents"]), 2)
        self.assertIn("budget", index["documents"][0]["graph_terms"])
        self.assertIn("asia", index["term_neighbors"])

    def test_empty_chunks_are_excluded_before_indexing(self) -> None:
        """Simulates the filter in index_documents() that strips empty/stub chunks."""
        all_chunks = [
            {"chunk_id": "a", "chunk_text": "", "token_estimate": 0},
            {"chunk_id": "b", "chunk_text": "[[NO_EXTRACTED_TEXT]]", "token_estimate": 5},
            {"chunk_id": "c", "chunk_text": "short", "token_estimate": 3},
            {"chunk_id": "d", "chunk_text": "valid content here for analysis", "token_estimate": 15},
        ]
        filtered = [
            c for c in all_chunks
            if c["chunk_text"].strip()
            and "[[NO_EXTRACTED_TEXT]]" not in c["chunk_text"]
            and c.get("token_estimate", 0) > 10
        ]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["chunk_id"], "d")


class CleanupStaleCollectionsTests(unittest.TestCase):
    def test_deletes_collections_with_matching_prefix_except_active(self) -> None:
        client = MagicMock()
        names = [
            "bi_hr_aaaaaaaaaaaa",
            "bi_hr_bbbbbbbbbbbb",
            "bi_hr_cccccccccccc",
            "bi_other_dddddddddddd",
        ]
        collections = []
        for name in names:
            col = MagicMock()
            col.name = name
            collections.append(col)
        client.list_collections.return_value = collections

        _cleanup_stale_collections(client, corpus_id="hr", active_name="bi_hr_aaaaaaaaaaaa")

        deleted = {call.args[0] for call in client.delete_collection.call_args_list}
        self.assertEqual(deleted, {"bi_hr_bbbbbbbbbbbb", "bi_hr_cccccccccccc"})

    def test_swallows_list_collections_failure(self) -> None:
        client = MagicMock()
        client.list_collections.side_effect = RuntimeError("chroma offline")
        # Should not raise.
        _cleanup_stale_collections(client, corpus_id="hr", active_name="bi_hr_active")
        client.delete_collection.assert_not_called()

    def test_swallows_individual_delete_failure(self) -> None:
        client = MagicMock()
        stale_a = MagicMock()
        stale_a.name = "bi_hr_aaaa"
        stale_b = MagicMock()
        stale_b.name = "bi_hr_bbbb"
        client.list_collections.return_value = [stale_a, stale_b]
        client.delete_collection.side_effect = [RuntimeError("locked"), None]

        _cleanup_stale_collections(client, corpus_id="hr", active_name="bi_hr_active")

        self.assertEqual(client.delete_collection.call_count, 2)


if __name__ == "__main__":
    unittest.main()
