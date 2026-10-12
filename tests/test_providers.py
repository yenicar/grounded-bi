from __future__ import annotations

import sys
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.config import AppConfig


class ProviderRetryTests(unittest.TestCase):
    """Tests for retry logic and parallel embedding in GeminiProvider."""

    def _make_provider(self):
        """Build a GeminiProvider with a fully mocked genai module."""
        from true_rag_bi.providers import GeminiProvider

        mock_genai = MagicMock()
        # Simulate a model that supports embedContent
        mock_model = MagicMock()
        mock_model.name = "models/gemini-embedding-001"
        mock_model.supported_generation_methods = ["embedContent"]
        mock_genai.list_models.return_value = [mock_model]

        config = AppConfig()
        with patch("true_rag_bi.providers._load_gemini_api_key", return_value="fake-key"):
            with patch("true_rag_bi.providers.GeminiProvider.__init__", lambda self, cfg: None):
                provider = GeminiProvider.__new__(GeminiProvider)
                provider._genai = mock_genai
                provider.embedding_model = "models/gemini-embedding-001"
                provider.answer_model = "gemini-2.5-flash"
                provider._answer_client = mock_genai.GenerativeModel()
        return provider

    def test_embed_documents_returns_correct_count(self) -> None:
        provider = self._make_provider()
        fake_embedding = [0.1, 0.2, 0.3]

        with patch.object(provider, "_embed_single", return_value=fake_embedding):
            results = provider.embed_documents(["text one", "text two", "text three"])

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], fake_embedding)

    def test_embed_documents_preserves_order(self) -> None:
        """Parallel execution must return results in input order, not completion order."""
        provider = self._make_provider()
        embeddings = [[float(i)] for i in range(5)]

        call_count = [0]

        def ordered_embed(text: str, task_type: str) -> list[float]:
            idx = call_count[0]
            call_count[0] += 1
            return embeddings[idx]

        with patch.object(provider, "_embed_single", side_effect=ordered_embed):
            results = provider.embed_documents([f"text {i}" for i in range(5)])

        self.assertEqual(len(results), 5)
        # Order must be preserved regardless of parallel execution
        for i, result in enumerate(results):
            self.assertEqual(result, embeddings[i], f"Result at index {i} was out of order")

    def test_is_retryable_matches_quota_errors(self) -> None:
        from true_rag_bi.providers import _is_retryable

        self.assertTrue(_is_retryable(Exception("429 resource exhausted")))
        self.assertTrue(_is_retryable(Exception("503 service unavailable")))
        self.assertTrue(_is_retryable(Exception("quota exceeded for this project")))
        self.assertTrue(_is_retryable(Exception("rate limit exceeded")))

    def test_is_retryable_does_not_match_auth_errors(self) -> None:
        from true_rag_bi.providers import _is_retryable

        self.assertFalse(_is_retryable(Exception("invalid api key")))
        self.assertFalse(_is_retryable(Exception("permission denied")))
        self.assertFalse(_is_retryable(Exception("404 not found")))

    def test_provider_initialization_does_not_call_list_models(self) -> None:
        """FastAPI startup should not block on Gemini model discovery."""
        from true_rag_bi.providers import GeminiProvider

        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.generativeai")

        def list_models():
            raise AssertionError("list_models should be lazy")

        fake_genai.configure = MagicMock()
        fake_genai.GenerativeModel = MagicMock()
        fake_genai.list_models = list_models

        config = AppConfig()
        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.generativeai": fake_genai},
        ):
            with patch("true_rag_bi.providers._load_gemini_api_key", return_value="fake-key"):
                provider = GeminiProvider(config)

        self.assertEqual(provider.embedding_model, "models/gemini-embedding-001")


if __name__ == "__main__":
    unittest.main()
