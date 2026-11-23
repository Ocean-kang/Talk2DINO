import json
from pathlib import Path


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "default.json"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Config not found: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))
