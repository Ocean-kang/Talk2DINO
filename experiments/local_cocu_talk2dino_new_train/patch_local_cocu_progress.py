#!/usr/bin/env python3
from pathlib import Path

ROOT = Path.cwd()
EXP = ROOT / "experiments/local_cocu_talk2dino_new_train"
PIPELINE = EXP / "pipeline.py"
README = EXP / "README.md"

if not PIPELINE.is_file():
    raise SystemExit(f"Not found: {PIPELINE}. Run from Talk2DINO repository root.")

source = PIPELINE.read_text(encoding="utf-8")
start = source.find("def run(")
end = source.find("\ndef relative(", start)
if start < 0 or end < 0:
    raise SystemExit("Expected run()/relative() block not found; no files changed.")

new_run = r"""def make_env(gpu: str) -> dict[str, str]:
    # WORLD_SIZE=1: delegate ephemeral port allocation to OS/TCPStore.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONHASHSEED", "123")
    env["PYTHONUNBUFFERED"] = "1"
    env.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
    })
    return env


def run(cmd: list[str], gpu: str, log: Path | None = None,
        live_progress: bool = False) -> None:
    env = make_env(gpu)
    args = [str(x) for x in cmd]
    print("[CMD]", " ".join(args), flush=True)
    if log is None:
        subprocess.run(args, cwd=ROOT, env=env, check=True)
        return

    log.parent.mkdir(parents=True, exist_ok=True)
    if not live_progress:
        with log.open("w", encoding="utf-8") as output:
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                output.write(line)
                print(line, end="", flush=True)
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, args)
        return

    # tqdm redraws with carriage returns, so stream chunks instead of lines.
    # This keeps the progress bar live and stores the same output in train.log.
    with log.open("wb") as output:
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
        assert proc.stdout is not None
        try:
            fd = proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                output.write(chunk)
                output.flush()
                binary_stdout = getattr(sys.stdout, "buffer", None)
                if binary_stdout is not None:
                    binary_stdout.write(chunk)
                    binary_stdout.flush()
                else:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
        finally:
            proc.stdout.close()
        code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, args)
"""
source = source[:start] + new_run + source[end:]

old_call = '         "--name_pedix", proj[len("vitb_mlp_infonce_"):]], args.gpu, work / "train.log")'
new_call = '         "--name_pedix", proj[len("vitb_mlp_infonce_"):]], args.gpu, work / "train.log",\n        live_progress=True)'
if old_call not in source:
    raise SystemExit("Expected train subprocess call not found; no files changed.")
source = source.replace(old_call, new_call, 1)

if '"MASTER_PORT": "0"' not in source or "live_progress=True" not in source:
    raise SystemExit("Validation failed; no files changed.")

readme_text = README.read_text(encoding="utf-8") if README.is_file() else ""
note = ("**端口与训练进度条**：子进程使用 `MASTER_PORT=0`，由操作系统/TCPStore "
        "选择临时端口，避免固定的 `29501` 冲突。训练阶段按字节块实时转发 stdout/stderr，"
        "保留 tqdm 的回车刷新行为；完整输出仍写入 `train.log`。\\n\\n")
anchor = "**运行前验证**：所有数据相对路径均以仓库根目录为起点。"
if readme_text and note.strip() not in readme_text:
    if anchor in readme_text:
        readme_text = readme_text.replace(anchor, note + anchor, 1)
    else:
        readme_text += "\\n\\n" + note

PIPELINE.write_text(source, encoding="utf-8")
if README.is_file():
    README.write_text(readme_text, encoding="utf-8")

print("[OK] Updated:", PIPELINE)
if README.is_file():
    print("[OK] Updated:", README)
