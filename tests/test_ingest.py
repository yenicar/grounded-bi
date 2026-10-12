from pathlib import Path
import sys
import unittest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.ingest import _build_markdown, _format_markdown_table, _looks_like_structured_table, _merge_page_text_and_tables


class IngestTests(unittest.TestCase):
    def test_build_markdown_wraps_pages_with_headers(self) -> None:
        markdown = _build_markdown("demo.pdf", ["alpha", "beta"])
        self.assertIn("# demo.pdf", markdown)
        self.assertIn("## Page 1", markdown)
        self.assertIn("alpha", markdown)
        self.assertIn("## Page 2", markdown)
        self.assertIn("beta", markdown)

    def test_format_markdown_table_builds_valid_table(self) -> None:
        table = _format_markdown_table(
            [
                ["Region", "Budget"],
                ["Asia", "70M"],
                ["Europe", "50M"],
            ]
        )
        self.assertIn("| Region | Budget |", table)
        self.assertIn("| Asia | 70M |", table)
        self.assertIn("| Europe | 50M |", table)

    def test_merge_page_text_and_tables_appends_extracted_tables(self) -> None:
        merged = _merge_page_text_and_tables(
            "Asia expansion budget overview",
            ["| Region | Budget |\n| --- | --- |\n| Asia | 70M |"],
        )
        self.assertIn("Asia expansion budget overview", merged)
        self.assertIn("### Extracted Table 1", merged)
        self.assertIn("| Asia | 70M |", merged)

    def test_false_positive_fragmented_text_table_is_rejected(self) -> None:
        looks_structured = _looks_like_structured_table(
            [
                ["Acme Electro", "nics – Fina", "ncial Models", "for Internatio", "nal Expans", "ion"],
                ["Internal Analysis (Spr", "eadsheet Ex", "cerpt and Co", "mmentary)", "", ""],
                ["Prepared by: Finance", "Department", "(Office of th", "e CFO)", "", ""],
            ]
        )
        self.assertFalse(looks_structured)


if __name__ == "__main__":
    unittest.main()
