import json
import tempfile
import unittest
from pathlib import Path

from caption_policy import CAPTIONS_PER_IMAGE, select_five_captions
from build_raw_captions import load_split


class CaptionPolicyRegressionTests(unittest.TestCase):
    def make_annotations(self, order=None):
        rows = [
            {"id": 100 + i, "image_id": 2193, "caption": "caption {}".format(i)}
            for i in range(6)
        ]
        if order is not None:
            rows = [rows[i] for i in order]
        return rows

    def test_six_annotations_selects_first_five_by_annotation_id(self):
        expected = ["caption {}".format(i) for i in range(5)]
        selected_a, dropped_a = select_five_captions(self.make_annotations(), 2193)
        selected_b, dropped_b = select_five_captions(
            self.make_annotations([5, 3, 1, 4, 2, 0]), 2193
        )
        self.assertEqual(selected_a, expected)
        self.assertEqual(selected_b, expected)
        self.assertEqual(dropped_a, 1)
        self.assertEqual(dropped_b, 1)
        self.assertEqual(len(selected_a), CAPTIONS_PER_IMAGE)

    def test_full_loader_normalizes_six_annotations_and_is_order_invariant(self):
        images = [{"id": 2193, "file_name": "COCO_train2014_000002193.jpg"}]
        ann_a = self.make_annotations()
        ann_b = self.make_annotations([5, 2, 4, 1, 3, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.json"
            path_b = Path(tmp) / "b.json"
            path_a.write_text(json.dumps({"images": images, "annotations": ann_a}))
            path_b.write_text(json.dumps({"images": images, "annotations": ann_b}))
            rows_a, stats_a = load_split(path_a, "train", return_stats=True)
            rows_b, stats_b = load_split(path_b, "train", return_stats=True)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(len(rows_a[0]["raw_captions"]), 5)
        self.assertEqual(stats_a["discarded_extra_annotations"], 1)
        self.assertEqual(stats_b["discarded_extra_annotations"], 1)

    def test_load_split_keeps_legacy_rows_only_return_by_default(self):
        images = [{"id": 2193, "file_name": "COCO_train2014_000002193.jpg"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.json"
            path.write_text(json.dumps({"images": images, "annotations": self.make_annotations()}))
            rows = load_split(path, "train")
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows[0]["raw_captions"]), 5)

    def test_fewer_than_five_annotations_fails(self):
        with self.assertRaises(ValueError):
            select_five_captions(self.make_annotations()[:4], 2193)


if __name__ == "__main__":
    unittest.main()
