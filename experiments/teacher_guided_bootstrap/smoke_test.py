"""Small structural checks for the side-car bootstrap model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.model import ProjectionLayer
from bootstrap_model import TeacherGuidedProjectionLayer


def main():
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs/vitb_mlp_infonce.yaml").read_text()
    )["model"]

    torch.manual_seed(123)
    baseline = ProjectionLayer.from_config(cfg)
    bootstrap = TeacherGuidedProjectionLayer.from_config(cfg)
    bootstrap.load_state_dict(baseline.state_dict())
    teacher = ProjectionLayer.from_config(cfg)
    bootstrap.attach_teacher(teacher, 0.0)

    visual = torch.randn(4, 12, 768)
    text = torch.randn(4, 512)

    baseline.train()
    bootstrap.train()
    expected = baseline(visual, text)
    actual = bootstrap(visual, text)
    if not torch.equal(expected, actual):
        raise AssertionError("alpha=0 is not a literal original-model pass-through")

    if set(bootstrap.state_dict()) != set(baseline.state_dict()):
        extra = set(bootstrap.state_dict()) - set(baseline.state_dict())
        raise AssertionError(f"Teacher leaked into Student state_dict: {sorted(extra)}")

    bootstrap.attach_teacher(teacher, 0.3)
    bootstrap.eval()
    actual_eval = bootstrap(visual, text)
    baseline.eval()
    expected_eval = baseline(visual, text)
    if not torch.equal(expected_eval, actual_eval):
        raise AssertionError("eval/inference must remain Student-only")

    print("OK: alpha=0 == original forward")
    print("OK: Teacher is absent from Student checkpoint keys")
    print("OK: eval/inference == original Student-only forward")


if __name__ == "__main__":
    main()
