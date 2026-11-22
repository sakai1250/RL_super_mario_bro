#!/usr/bin/env python3
"""追加学習: クリアできなかったステージを、保存済みの続きから学習し直す。

使い方 (リポジトリ直下で):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_extra_stages.py 2-2 0
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_extra_stages.py 7-2 1

- 既存のチェックポイント (重み・ステップ・ε) を引き継いで再開する。
- 120万ステップごとに区切り、クリアするまで区切りを繰り返す。
- 結果は results/full_run/<stage>/done.json に上書きされる。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_all_stages as tas

ROUND_STEPS = 1_200_000  # 1区切りあたりの追加ステップ数


def main() -> int:
    stage = sys.argv[1]
    group = int(sys.argv[2])
    import json

    done_path = tas.RESULTS / stage / "done.json"
    steps = 0
    if done_path.exists():
        steps = int(json.loads(done_path.read_text(encoding="utf-8")).get("steps", 0))
    round_no = 0
    while True:
        round_no += 1
        tas.BUDGET = steps + ROUND_STEPS
        tas.log(f"extra {stage}: round {round_no} cumulative budget {tas.BUDGET}")
        result = tas.train_stage(stage, group)
        steps = int(result["steps"])
        if result["cleared"]:
            tas.log(f"extra {stage}: CLEARED in round {round_no}")
            return 0
        tas.log(f"extra {stage}: round {round_no} finished without clear; continuing")


if __name__ == "__main__":
    raise SystemExit(main())
