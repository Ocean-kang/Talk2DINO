"""Redirect upstream's hard-coded training RNG seed before loading train.py."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import train_util

seed = int(sys.argv.pop(1))
original_set_seed = train_util.set_seed
train_util.set_seed = lambda _: original_set_seed(seed)
print(f"[TRAIN] seed={seed}", flush=True)
runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
