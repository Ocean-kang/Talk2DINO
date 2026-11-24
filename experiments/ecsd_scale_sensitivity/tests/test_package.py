import importlib.util
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "data" / "build_nested_subsets.py"
spec = importlib.util.spec_from_file_location("build_nested_subsets", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NestedSubsetTests(unittest.TestCase):
    def test_scale_names(self):
        self.assertEqual(module.scale_name(5000, 82783), "5k")
        self.assertEqual(module.scale_name(40000, 82783), "40k")
        self.assertEqual(module.scale_name(82783, 82783), "full")

    def test_subset_data_preserves_annotations(self):
        data = {
            "images": [{"id": 1}, {"id": 2}, {"id": 3}],
            "annotations": [
                {"image_id": 1, "id": 10},
                {"image_id": 2, "id": 20},
                {"image_id": 3, "id": 30},
            ],
        }
        out = module.subset_data(data, [1, 3])
        self.assertEqual([x["id"] for x in out["images"]], [1, 3])
        self.assertEqual([x["image_id"] for x in out["annotations"]], [1, 3])


if __name__ == "__main__":
    unittest.main()
