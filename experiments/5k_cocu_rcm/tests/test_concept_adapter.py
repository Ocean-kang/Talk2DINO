import sys
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from data.build_concept_archive import adapt_source, normalize_rows


class TestConceptAdapter(unittest.TestCase):
    def test_matched_detailed_format(self):
        source = {"123": [{"concept": "Bicycles", "score": 0.6, "best_head": 4}]}
        rows = normalize_rows(adapt_source(source)[123])
        self.assertEqual(rows[0]["normalized_concept"], "bicycle")
        self.assertEqual(rows[0]["best_head"], 4)

    def test_audit_record_format(self):
        source = {"123": {"image_id": 123, "concepts": [{"concept": "clock"}]}}
        rows = normalize_rows(adapt_source(source)[123])
        self.assertEqual(rows[0]["concept"], "clock")


if __name__ == "__main__":
    unittest.main()
