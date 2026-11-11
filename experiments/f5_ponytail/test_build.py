import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run import ROOT

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires the upstream PyTorch environment")
class DataBuild(unittest.TestCase):
    def test_original_ids_edited_features_and_counterfactual_filter(self):
        sys.path.insert(0, str(ROOT))
        import build_data

        basis = torch.eye(768)[:12]
        raw_z = torch.eye(512)[7]
        teacher = SimpleNamespace(
            device="cpu",
            encode=lambda words: torch.stack([basis[0] for _ in words]),
            sim=lambda h, z: torch.nn.functional.normalize(h, dim=-1) @ z.T,
            projector=SimpleNamespace(project_clip_txt=lambda z: torch.nn.functional.pad(z, (0, 256))),
            clip=SimpleNamespace(tokenize=lambda texts, **kwargs: torch.zeros(len(texts), 1)),
            text=SimpleNamespace(encode_text=lambda tokens: torch.ones(len(tokens), 512)),
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            work = Path(tmp)
            anns = [dict(id=i * 10 + j, image_id=i, caption=text, ann_feats=raw_z)
                    for i, text in [(1, "A street"), (2, "A bicycle"), (3, "A sidewalk")]
                    for j in range(5)]
            images = [dict(id=i, disentangled_self_attn=basis) for i in (1, 2, 3)]
            torch.save(dict(annotations=anns, images=images), work / "base.pth")
            torch.save({1: basis, 2: -basis, 3: -basis}, work / "heads.pth")
            (work / "local.json").write_text(json.dumps({"1": [2]}))
            (work / "cls.json").write_text(json.dumps({"1": [2, 3]}))
            (work / "gallery.json").write_text(json.dumps(["bicycle"]))
            args = SimpleNamespace(base_train=work / "base.pth", heads=work / "heads.pth",
                                   local_neighbors=work / "local.json", cls_neighbors=work / "cls.json",
                                   gallery=work / "gallery.json", limit=1, device="cpu", batch=8,
                                   official=None, config=None, negative_margin=.05, floor=.1, margin=.02)
            with patch.object(build_data, "Teacher", return_value=teacher):
                stats = build_data.build(args, work, ["cf_l03"])
            result = torch.load(work / "cf_l03/train.pth", weights_only=False)
            self.assertEqual([a["id"] for a in result["annotations"]], list(range(10, 15)))
            self.assertIn("bicycle", result["annotations"][0]["caption"])
            self.assertTrue(torch.equal(result["annotations"][0]["ann_feats"], torch.ones(512)))
            self.assertTrue(torch.equal(result["annotations"][1]["ann_feats"], raw_z))
            self.assertEqual(stats["cf_l03"]["changed_captions"], 1)
            audit = json.loads((work / "audit.jsonl").read_text())
            self.assertEqual(audit["variants"]["cf_l03"]["selected"][0]["negative_ids"], [3])
