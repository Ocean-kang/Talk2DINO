#!/usr/bin/env python3
"""Install this caption normalization fix into an existing Talk2DINO checkout."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "experiments" / "full_mscoco14_ecsd"
SOURCE = Path(__file__).resolve().parent

required = ["build_raw_captions.py", "generate_ecsd.py", "prepare.py", "evaluate_cisd.py"]
missing = [name for name in required if not (TARGET / name).is_file()]
if missing:
    raise SystemExit(
        "Expected existing experiment at {}. Missing: {}".format(TARGET, ", ".join(missing))
    )

for name in ["caption_policy.py", "build_raw_captions.py", "validate_caption_contract.py"]:
    shutil.copy2(SOURCE / name, TARGET / name)

tests_dir = TARGET / "tests"
tests_dir.mkdir(exist_ok=True)
shutil.copy2(SOURCE / "tests" / "test_caption_policy.py", tests_dir / "test_caption_policy.py")
shutil.copy2(SOURCE / "README_FIX.md", TARGET / "README_FIX.md")

print("[OK] Caption normalization fix installed at {}".format(TARGET))
print("Run: python -m unittest discover -s experiments/full_mscoco14_ecsd/tests -p 'test_caption_policy.py' -v")
