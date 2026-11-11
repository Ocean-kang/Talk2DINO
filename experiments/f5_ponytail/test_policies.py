import json
from pathlib import Path
import tempfile
import unittest

from policies import VARIANTS, assign, select
from run import ROOT, TASKS, moments, summarize


class Policies(unittest.TestCase):
    def test_assignment_global_and_deterministic(self):
        costs = [[0, 1, 20, 20, 20], [0, 10, 20, 20, 20]]
        self.assertEqual(assign(costs, False), (0, 0))
        self.assertEqual(assign(costs, True), (1, 0))
        self.assertEqual(assign([[0] * 5] * 3, True), (0, 1, 2))
        self.assertEqual(assign([], True), ())

    def test_gate_negatives_and_lambda(self):
        rows = [
            dict(concept="a", best_head=0, target_score=.4, lower_half=[], filtered=[(99, .3)]),
            dict(concept="b", best_head=1, target_score=.39, lower_half=[], filtered=[]),
            dict(concept="c", best_head=2, target_score=.07, lower_half=[], filtered=[]),
        ]
        coverage = [.2, .1, .01]
        def chosen(v):
            return select(rows, coverage, VARIANTS[v], .1, .02, 3)
        self.assertEqual([r["concept"] for r in chosen("cf_l00")], ["a", "b"])
        self.assertEqual([r["concept"] for r in chosen("cf_l03")], ["b", "a"])
        self.assertEqual(chosen("cf_l03")[1]["negative_ids"], [99])
        self.assertIsNone(chosen("cf_l03")[0]["negative_mean"])
        self.assertEqual(len(chosen("f5_control")), 3)

    def test_eval_configs_and_fixed_vs_paired_summary(self):
        for directory, stem in TASKS.values():
            cfg = ROOT / "src/open_vocabulary_segmentation/configs" / directory
            self.assertTrue((cfg / f"dinotext_{stem}_vitb_mlp_infonce.yml").is_file())
            self.assertTrue((cfg / f"eval_{stem}_pamr.yml").is_file())
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            work = Path(tmp)
            for task in TASKS:
                folders = [(work / "fixed" / "official", 60), (work / "fixed" / "old_f5", 61)]
                for seed in range(123, 128):
                    folders += [(work / "f5_control" / str(seed) / "eval", seed - 60),
                                (work / "threshold" / str(seed) / "eval", seed - 59)]
                for path, value in folders:
                    (path / task).mkdir(parents=True)
                    (path / task / "metrics.json").write_text(json.dumps({"task": task, "miou": value}))
            summarize(work, ["f5_control", "threshold"])
            row = json.loads((work / "summary.json").read_text())["voc20"]["variants"]["threshold"]
            self.assertEqual(row["delta_vs_fixed"], {"official": 6, "old_f5": 5})
            self.assertEqual(row["paired"]["ci95"], [1, 1])
            self.assertAlmostEqual(moments([1, 2, 3, 4, 5])["std"], 2.5 ** .5)


if __name__ == "__main__":
    unittest.main()

