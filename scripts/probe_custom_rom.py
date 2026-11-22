#!/usr/bin/env python3
"""カスタムROMの検証: 1-1 の地形データを書き換えたROMで環境が動き、
地形が変わって見えるかを、元ROMと画面比較で確かめる。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ORIG = ROOT / ".venv/lib/python3.13/site-packages/gym_super_mario_bros/_roms/super-mario-bros.nes"
CUSTOM = Path("/tmp/smb_custom.nes")
STREAM_OFF = 0x269E  # 1-1 の地形データの開始位置（実測で特定）


def build_custom() -> None:
    data = bytearray(ORIG.read_bytes())
    assert data[STREAM_OFF] == 0x50 and data[STREAM_OFF + 2] == 0x07 and data[STREAM_OFF + 3] == 0x81
    # 最初のオブジェクト（?ブロック）の X を 0 -> 2 にして 2 列右へ動かす
    data[STREAM_OFF + 2] = 0x27
    CUSTOM.write_bytes(bytes(data))
    print("custom ROM written:", CUSTOM)


def play(rom_path: str, out_png: str):
    import gym_super_mario_bros.smb_env as smb_env

    smb_env.smb1_rom_path = lambda: rom_path
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_all_stages import save_png

    from mario_app.env import grab_screen, make_env

    env = make_env("1-1")
    env.reset()
    max_x = 0
    shot = None
    for i in range(900):
        _obs, _r, term, trunc, info = env.step(3)  # right + B
        x = int(info.get("x_pos", 0))
        max_x = max(max_x, x)
        if x >= 150 and shot is None:
            shot = grab_screen(env)
        if term or trunc:
            break
    env.close()
    save_png(Path(out_png), shot)
    print(out_png, "max_x:", max_x)
    return shot


if __name__ == "__main__":
    build_custom()
    a = play(str(ORIG), "/tmp/probe_orig.png")
    b = play(str(CUSTOM), "/tmp/probe_custom.png")
    print("pixel diff:", int(np.abs(a.astype(int) - b.astype(int)).sum()))
