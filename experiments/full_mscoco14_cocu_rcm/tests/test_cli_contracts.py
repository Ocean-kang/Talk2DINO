"""Static CLI contract checks for scripts that are coupled to upstream modules."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

EXP = Path(__file__).resolve().parents[1]


class CliContractTests(unittest.TestCase):
    def test_full_concept_discovery_namespace_matches_parser(self):
        path = EXP / "data" / "generate_full_cocu_matches.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        accessed = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "a"
        }
        declared = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            flag = node.args[0].value
            if isinstance(flag, str) and flag.startswith("--"):
                declared.add(flag[2:].replace("-", "_"))
        self.assertEqual(
            accessed - declared,
            set(),
            msg=f"Undeclared argparse attributes: {sorted(accessed - declared)}",
        )


if __name__ == "__main__":
    unittest.main()
