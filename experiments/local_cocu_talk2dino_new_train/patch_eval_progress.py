#!/usr/bin/env python3
"""Enable live carriage-return progress rendering for segmentation evaluation."""
from pathlib import Path

pipeline = Path("experiments/local_cocu_talk2dino_new_train/pipeline.py")
if not pipeline.is_file():
    raise SystemExit(
        f"Not found: {pipeline}. Run this script from the Talk2DINO repository root."
    )

text = pipeline.read_text(encoding="utf-8")
if "def run(cmd: list[str], gpu: str, log: Path | None = None," not in text:
    raise SystemExit(
        "Expected patched run() implementation was not found. "
        "Apply the previous port/progress patch first; no files changed."
    )
if "live_progress: bool = False" not in text:
    raise SystemExit("run() does not expose live_progress; no files changed.")

old = "                args.gpu, dest / \"run.log\")"
new = "                args.gpu, dest / \"run.log\", live_progress=True)"
count = text.count(old)
if count != 1:
    already = "args.gpu, dest / \"run.log\", live_progress=True)" in text
    if count == 0 and already:
        print("[SKIP] Evaluation already has live_progress=True.")
        raise SystemExit(0)
    raise SystemExit(
        f"Expected exactly one evaluation subprocess call, found {count}; no files changed."
    )

backup = pipeline.with_suffix(".py.bak")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

text = text.replace(old, new, 1)
if "args.gpu, dest / \"run.log\", live_progress=True)" not in text:
    raise SystemExit("Internal validation failed; no files changed.")

pipeline.write_text(text, encoding="utf-8")
print("[OK] Enabled live progress forwarding for segmentation evaluation.")
print("[BACKUP]", backup)
print("[NEXT] Re-run the eval stage; no retraining is required.")
