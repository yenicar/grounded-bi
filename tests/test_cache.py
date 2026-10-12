from pathlib import Path
import sys
import unittest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from true_rag_bi.cache import build_pipeline_fingerprint, build_query_cache_key
from true_rag_bi.config import AppConfig


class CacheTests(unittest.TestCase):
    def test_pipeline_fingerprint_changes_when_retrieval_changes(self) -> None:
        base = AppConfig()
        changed = AppConfig(retrieval_top_k=12)
        self.assertNotEqual(build_pipeline_fingerprint(base).key(), build_pipeline_fingerprint(changed).key())

    def test_query_cache_key_is_stable(self) -> None:
        key_one = build_query_cache_key("budget pressure", "pipe-1", "corpus-1")
        key_two = build_query_cache_key("budget pressure", "pipe-1", "corpus-1")
        self.assertEqual(key_one, key_two)


if __name__ == "__main__":
    unittest.main()
