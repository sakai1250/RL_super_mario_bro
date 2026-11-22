#!/usr/bin/env python3
"""クリア済みモデルの再生動画を録る (高速版)。

使い方:
    .venv/bin/python scripts/record_clear.py STAGE CKPT OUT.mp4
        [--attempts N] [--eps E] [--seed0 S] [--save-best]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPEAT = 12
MAX_FRAMES = 9_000
STUCK_FRAMES = 2_500


def play_once(net, stage: str, seed: int, eps: float):
    import numpy as np
    import torch

    from mario_app.env import ACTION_LABELS, grab_screen, make_env

    rng = np.random.default_rng(seed)
    env = make_env(stage)
    obs, _info = env.reset(seed=seed)
    frames = []
    action = 0
    max_x = 0.0
    last_improve = 0
    cleared = False
    for i in range(MAX_FRAMES):
        if i - last_improve > STUCK_FRAMES:
            break
        if i % REPEAT == 0:
            if eps > 0 and rng.random() < eps:
                action = int(rng.integers(len(ACTION_LABELS)))
            else:
                x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    action = int(net(x, "online")[0].argmax().item())
        obs, _r, term, trunc, info = env.step(action)
        frames.append(grab_screen(env))
        x_now = float(info.get("x_pos", 0.0))
        if x_now > max_x:
            max_x = x_now
            last_improve = i
        if info.get("flag_get"):
            cleared = True
            if not (term or trunc):
                for _ in range(30):
                    obs, _r, term, trunc, info = env.step(0)
                    frames.append(grab_screen(env))
                    if term or trunc:
                        break
            break
        if term or trunc:
            break
    env.close()
    return cleared, frames, max_x


def encode(frames, out_path: Path) -> None:
    import numpy as np

    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "60",
           "-i", "-", "-vf", "scale=512:480:flags=neighbor",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.asarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.returncode}")


def main() -> int:
    import torch

    torch.set_num_threads(1)
    from mario_app.agent import MarioNet, load_weights

    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--attempts", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.075)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--save-best", action="store_true")
    args = ap.parse_args()

    net = MarioNet()
    load_weights(net, args.ckpt, device="cpu")
    net.eval()

    best = None
    for attempt in range(args.attempts):
        cleared, frames, max_x = play_once(net, args.stage,
                                           args.seed0 + attempt, args.eps)
        if best is None or max_x > best[2]:
            best = (cleared, frames, max_x)
        if cleared:
            best = (cleared, frames, max_x)
            break
    cleared, frames, max_x = best
    if cleared or args.save_best:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        encode(frames, out)
    print(json.dumps({"stage": args.stage, "cleared": cleared,
                      "attempts_used": attempt + 1, "frames": len(frames),
                      "max_x": max_x, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
