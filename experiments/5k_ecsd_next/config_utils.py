"""Small JSON configuration loader for the ECSD next-stage experiment."""
import json
from pathlib import Path

def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "default.json"
    path = Path(path)
    if not path.is_absolute():
        # Paths passed from repository root are preferred; otherwise use package-relative.
        if not path.is_file():
            path = Path(__file__).resolve().parent / path
    return json.loads(path.read_text(encoding="utf-8"))
