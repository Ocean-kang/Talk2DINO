"""Atomic and conservative artifact writers for ECSD v3."""
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def _backup_existing(path: Path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(path.name + ".bak." + stamp)
    shutil.copy2(path, backup)
    return backup


def atomic_write_text(path, text, overwrite=False, validate_existing=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            if validate_existing is not None:
                try:
                    old = validate_existing(path)
                except Exception as exc:
                    raise RuntimeError(
                        "Existing artifact is invalid: {}. It was not modified. "
                        "Use --overwrite to preserve a backup and replace it. ({})".format(path, exc)
                    ) from exc
                if old == text:
                    return "reused"
            raise FileExistsError(
                "Refusing to replace existing artifact: {}. Use --overwrite intentionally.".format(path)
            )
        _backup_existing(path)

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return "written"


def atomic_write_json(path, value, overwrite=False, indent=2):
    text = json.dumps(value, ensure_ascii=False, indent=indent) + "\n"

    def validate_json(existing_path):
        with Path(existing_path).open("r", encoding="utf-8") as handle:
            old = json.load(handle)
        return json.dumps(old, ensure_ascii=False, indent=indent) + "\n"

    return atomic_write_text(path, text, overwrite=overwrite, validate_existing=validate_json)


def atomic_write_jsonl(path, rows, overwrite=False):
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in rows)

    def validate_jsonl(existing_path):
        parsed = []
        with Path(existing_path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSONL at line {}: {}".format(line_no, exc))
        return "".join(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in parsed)

    return atomic_write_text(path, text, overwrite=overwrite, validate_existing=validate_jsonl)


def atomic_write_csv(path, fieldnames, rows, overwrite=False):
    import csv
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    expected = buffer.getvalue()
    return atomic_write_text(
        path, expected, overwrite=overwrite,
        validate_existing=lambda existing_path: Path(existing_path).read_text(encoding="utf-8"),
    )
