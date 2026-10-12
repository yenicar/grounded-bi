from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.cli import _MAX_QUERY_LENGTH, _build_parser


class CLIParserTests(unittest.TestCase):
    def test_ask_command_parses_question(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["ask", "What are the risks in Asia?"])
        self.assertEqual(args.command, "ask")
        self.assertEqual(args.question, "What are the risks in Asia?")
        self.assertFalse(args.debug)

    def test_ask_command_debug_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["ask", "Budget question", "--debug"])
        self.assertTrue(args.debug)

    def test_index_command_verbose_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["index", "--verbose"])
        self.assertEqual(args.command, "index")
        self.assertTrue(args.verbose)

    def test_eval_command_custom_prompts_path(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["eval", "--prompts", "custom_prompts.json"])
        self.assertEqual(args.command, "eval")
        self.assertEqual(args.prompts, Path("custom_prompts.json"))


class CLIGuardTests(unittest.TestCase):
    def test_query_length_constant_is_reasonable(self) -> None:
        self.assertGreaterEqual(_MAX_QUERY_LENGTH, 500)
        self.assertLessEqual(_MAX_QUERY_LENGTH, 10000)

    def test_long_query_raises_runtime_error(self) -> None:
        from true_rag_bi.cli import _run

        long_question = "x" * (_MAX_QUERY_LENGTH + 1)
        mock_args = type("Args", (), {
            "command": "ask",
            "question": long_question,
            "debug": False,
            "verbose": False,
        })()

        with self.assertRaises(RuntimeError) as ctx:
            _run(mock_args, None)  # config not needed — guard fires before it's used

        self.assertIn("exceeds", str(ctx.exception).lower())

    def test_normal_query_does_not_raise_length_error(self) -> None:
        """A short query should NOT raise a length error (may raise others from missing config/index)."""
        from true_rag_bi.cli import _run

        short_question = "What is the budget?"
        mock_args = type("Args", (), {
            "command": "ask",
            "question": short_question,
            "debug": False,
            "verbose": False,
        })()

        # Should not raise a RuntimeError about query length.
        # It will raise a different RuntimeError about missing provider/index — that's expected.
        try:
            _run(mock_args, None)
        except RuntimeError as exc:
            self.assertNotIn("exceeds", str(exc).lower())
        except Exception:
            pass  # Other exceptions (AttributeError etc) are fine — we only care about the length guard


if __name__ == "__main__":
    unittest.main()
