#!/usr/bin/env python3
"""Scheme D: Head Attention Region 验证; invoke common.runner, no src changes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.local_cocu_talk2dino.common.runner import main

if __name__ == '__main__':
    main('D')
