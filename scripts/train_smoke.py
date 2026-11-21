#!/usr/bin/env python3
"""学習スモーク: 数千ステップ回して速度・loss・保存の読み直しを確認する。

    .venv/bin/python scripts/train_smoke.py [STEPS] [DEVICE] [ENVS]
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QCoreApplication  # noqa: E402

from mario_app.agent import MarioNet, load_weights  # noqa: E402
from mario_app.game_worker import GameWorker, RunConfig  # noqa: E402
from mario_app.hyperparams import Hyperparams  # noqa: E402
from mario_app.pacing import SPEED_MAX  # noqa: E402


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    device = sys.argv[2] if len(sys.argv) > 2 else "auto"
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    _app = QCoreApplication([])
    out = Path(tempfile.mkdtemp(prefix="train_smoke_"))
    worker = GameWorker(RunConfig(
        model_path=None, stage="1-1", hp=Hyperparams(), learning=True, speed=SPEED_MAX,
        device=device, autosave_every=0, models_dir=str(out), n_envs=n_envs))
    stats, saves, failures = [], [], []
    worker.stats_ready.connect(stats.append)
    worker.saved.connect(saves.append)
    worker.failed.connect(failures.append)
    first = {}

    def on_stats(s):
        first.setdefault("t", time.time())
        first.setdefault("step", s["agent_step"])
        if s["agent_step"] >= steps:
            worker.request_save()
            worker.request_stop()

    worker.stats_ready.connect(on_stats)
    t0 = time.time()
    worker.run()
    elapsed = time.time() - t0
    if failures:
        print(failures[0])
        return 1
    last = stats[-1]
    rate = (last["agent_step"] - first["step"]) / (time.time() - first["t"])
    print(f"device: {last['device']}  envs: {last.get('envs', 1)}  agent_step: {last['agent_step']}  updates: {last['updates']}")
    print(f"loss: {last['loss']}  finite: {last['loss'] is not None and math.isfinite(last['loss'])}")
    print(f"elapsed: {elapsed:.1f}s  speed: {rate:.1f} agent steps/s (= {rate * 12 / 60:.1f}x real time)")
    net = MarioNet()
    load_weights(net, saves[0]["path"])
    print("saved & reloaded:", Path(saves[0]["path"]).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
