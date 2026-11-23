import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ECSD_DIR = ROOT / "experiments" / "5k_ecsd"
if str(ECSD_DIR) not in sys.path:
    sys.path.insert(0, str(ECSD_DIR))
FULL_DIR = Path(__file__).resolve().parents[1]
if str(FULL_DIR) not in sys.path:
    sys.path.insert(0, str(FULL_DIR))

from ecsd_core import build_candidates, content_tokens  # noqa: E402
from generate_ecsd import select_five  # noqa: E402
from build_raw_captions import load_split  # noqa: E402
import torch  # noqa: E402


class FullEcsdSmokeTests(unittest.TestCase):
    def test_candidate_selection_returns_five_source_grounded_captions(self):
        raw = [
            "A man is riding a bicycle on a street.",
            "A cyclist rides past several parked cars.",
            "The man wears a red shirt and a helmet.",
            "Buildings line the road beside the cyclist.",
            "A person is riding a bike outdoors.",
        ]
        candidates = build_candidates(raw)
        texts = list(dict.fromkeys(x["text"] for x in candidates))
        generator = torch.Generator().manual_seed(7)
        vectors = torch.randn(len(texts), 32, generator=generator)
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        emb = {text: vectors[i] for i, text in enumerate(texts)}
        selected, trace = select_five(candidates, raw, emb)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(trace), 5)
        source_facts = set().union(*(content_tokens(x) for x in raw))
        for caption in selected:
            self.assertTrue(content_tokens(caption).issubset(source_facts))

    def test_official_caption_json_parser(self):
        payload = {
            "images": [{"id": 1, "file_name": "COCO_train2014_000000000001.jpg"}],
            "annotations": [
                {"id": i, "image_id": 1, "caption": "caption {}".format(i)}
                for i in range(5)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions_train2014.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_split(path, "train")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], 1)
        self.assertEqual(len(rows[0]["raw_captions"]), 5)


if __name__ == "__main__":
    unittest.main()
