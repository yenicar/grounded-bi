from pathlib import Path
import sys
import unittest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.chunking import split_markdown_into_chunks
from true_rag_bi.utils import estimate_tokens


SAMPLE_MARKDOWN = """# Demo Document

## Page 1

Europe remains attractive because its regulations are strict but predictable.

The budget outlook is more stable in Europe than in Asia.

## Page 2

Asia offers faster growth, but tariffs and shipping create added volatility.

Retail partners worry that Acme's brand is still unclear.
"""


class ChunkingTests(unittest.TestCase):
    def test_chunking_preserves_page_ranges(self) -> None:
        chunks = split_markdown_into_chunks(
            doc_name="Demo.pdf",
            doc_id=1,
            markdown_text=SAMPLE_MARKDOWN,
            target_tokens=35,
            overlap_tokens=8,
            child_target_tokens=16,
            child_overlap_tokens=4,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertGreaterEqual(chunks[-1]["page_end"], 2)
        self.assertTrue(all(chunk["chunk_id"].startswith("demo-pdf-child-") for chunk in chunks))
        self.assertTrue(all(chunk["parent_chunk_id"].startswith("demo-pdf-parent-") for chunk in chunks))

    def test_token_estimate_uses_tokenizer(self) -> None:
        estimated = estimate_tokens("This is a test sentence with ten words in it.")
        self.assertGreaterEqual(estimated, 8)
        self.assertLessEqual(estimated, 14)

    def test_chunking_tags_markdown_tables(self) -> None:
        markdown = """# Table Demo

## Page 1

### Extracted Table 1
| Region | Budget |
| --- | --- |
| Asia | 70M |
| Europe | 50M |
"""
        chunks = split_markdown_into_chunks(
            doc_name="Table.pdf",
            doc_id=2,
            markdown_text=markdown,
            target_tokens=60,
            overlap_tokens=8,
            child_target_tokens=30,
            child_overlap_tokens=4,
        )
        self.assertTrue(any(chunk["has_table"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
