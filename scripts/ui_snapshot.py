#!/usr/bin/env python3
"""UI をオフスクリーンで PNG に保存する（目視確認用）。

    .venv/bin/python scripts/ui_snapshot.py OUT_DIR --gallery
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget  # noqa: E402

from mario_app.ui.style import PALETTE, build_qss, load_pixel_font  # noqa: E402
from mario_app.ui.widgets import BoxPanel, Key, Lcd, Led  # noqa: E402


def gallery(out: Path) -> Path:
    root = QWidget()
    root.setStyleSheet(f"QWidget#Root {{ background: {PALETTE['body']}; }}")
    root.setObjectName("Root")
    col = QVBoxLayout(root)
    col.setContentsMargins(24, 24, 24, 24)
    box = BoxPanel("SPEED／速度")
    row = QHBoxLayout()
    for text, on in (("0.25×", False), ("1×", False), ("2×", True), ("MAX", False)):
        k = Key(text, checkable=True)
        k.setChecked(on)
        row.addWidget(k)
    box.body.addLayout(row)
    box.body.addWidget(Lcd("0.412", chars=8))
    col.addWidget(box)
    row2 = QHBoxLayout()
    dis = Key("▶ 開始")
    dis.setEnabled(False)
    row2.addWidget(dis)
    row2.addWidget(Key("停止", tone="accent"))
    start = Key("開始")
    start.set_glyph("play")
    row2.addWidget(start)
    col.addLayout(row2)
    leds = QHBoxLayout()
    for label, on in (("学習", True), ("保存", False), ("MPS", True)):
        led = Led(label)
        led.set_on(on)
        leds.addWidget(led)
    wrap = QWidget()
    wrap.setStyleSheet(f"background: {PALETTE['maroon']};")
    wrap.setLayout(leds)
    col.addWidget(wrap)
    root.resize(560, 330)
    path = out / "gallery.png"
    root.grab().save(str(path))
    return path


def windows(out: Path) -> list[Path]:
    """メインウィンドウを 3 サイズで保存する（サンプルのフレームと統計入り）。"""
    from mario_app.hyperparams import hp_to_dict, Hyperparams
    from mario_app.ui.main_window import MainWindow

    win = MainWindow(models_dir=out, screenshot_dir=out)
    win.last_stats = {
        "episode": 37, "agent_step": 48210, "x_pos": 1284, "fps": 58, "epsilon": 0.412,
        "ep_reward": 128.4, "best_reward": 301.2, "loss": 0.0213, "avg_q": 4.18, "buffer_size": 6214,
        "ep_steps": 212, "action": "right + A", "hp": hp_to_dict(Hyperparams()), "learning": True,
        "device": "mps", "last_saved": "00:58:02 mario_x.pth", "unsaved_steps": 1204,
    }
    win.log_box.set_rewards([10 + i * 6 + (i % 5) * 9 for i in range(40)])
    win.log_box.append_log("γ 0.90 → 0.95 (手動)")
    win.show()
    paths = []
    for w, h in ((860, 600), (1180, 840), (1600, 1000)):
        win.resize(w, h)
        win._refresh_stats()
        app = QApplication.instance()
        app.processEvents()
        path = out / f"{w}x{h}.png"
        win.grab().save(str(path))
        paths.append(path)
    return paths


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(build_qss(load_pixel_font()))
    if "--gallery" in sys.argv:
        print(gallery(out))
    else:
        for path in windows(out):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
