#!/usr/bin/env python3
"""全ステージ学習のライブ表示。

results/full_run/live/group<g>.npz に書き出される全環境の最新フレームと
group<g>.json の進捗を読み、2 系統分を 1 ウィンドウに並べて表示する。

使い方 (リポジトリ直下で):
    DISPLAY=:0 .venv/bin/python scripts/live_view.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIVE = ROOT / "results" / "full_run" / "live"
GROUPS = ((0, "ワールド 1-4"), (1, "ワールド 5-8"))


def main() -> int:
    import numpy as np
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    class GroupGrid(QWidget):
        """1 系統分の全環境フレームを格子状に描く。"""

        def __init__(self) -> None:
            super().__init__()
            self.frames: dict[int, QPixmap] = {}

        def set_frames(self, frames: dict) -> None:
            grid: dict[int, QPixmap] = {}
            for aid, frame in frames.items():
                h, w = frame.shape[:2]
                image = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                grid[int(aid)] = QPixmap.fromImage(image)
            self.frames = grid
            self.update()

        def paintEvent(self, event) -> None:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor("#16110F"))
            aids = sorted(self.frames)
            if not aids:
                p.setPen(QColor("#CBBFA5"))
                p.setFont(QFont("Sans", 16))
                p.drawText(self.rect(), Qt.AlignCenter, "学習の開始を待っています")
                p.end()
                return
            n = len(aids)
            cols = 6 if n > 6 else n
            rows = -(-n // cols)
            gap = 6
            cw = max(1, (self.width() - gap * (cols + 1)) // cols)
            ch = max(1, (self.height() - gap * (rows + 1)) // rows)
            p.setFont(QFont("Sans", 11, QFont.Bold))
            for idx, aid in enumerate(aids):
                pix = self.frames[aid]
                cell_x = gap + (idx % cols) * (cw + gap)
                cell_y = gap + (idx // cols) * (ch + gap)
                target = pix.size().scaled(cw, ch, Qt.KeepAspectRatio)
                x = cell_x + (cw - target.width()) // 2
                y = cell_y + (ch - target.height()) // 2
                p.drawPixmap(x, y, pix.scaled(target, Qt.IgnoreAspectRatio,
                                              Qt.FastTransformation))
                label = f"ENV {aid + 1}"
                tw = p.fontMetrics().horizontalAdvance(label) + 10
                p.fillRect(x + 3, y + 3, tw, 20, QColor(31, 18, 14, 200))
                p.setPen(QColor("#F2E7CE"))
                p.drawText(x + 8, y + 18, label)
            p.end()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = QWidget()
    win.setWindowTitle("全ステージ学習 ライブ")
    win.resize(1600, 1000)
    lay = QVBoxLayout(win)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(6)

    headers: dict[int, QLabel] = {}
    grids: dict[int, GroupGrid] = {}
    for group, title in GROUPS:
        head = QLabel(f"{title} — 接続待ち")
        head.setFont(QFont("Sans", 15, QFont.Bold))
        head.setStyleSheet("color:#F2E7CE;")
        lay.addWidget(head)
        grid = GroupGrid()
        lay.addWidget(grid, 1)
        headers[group] = head
        grids[group] = grid
    win.setStyleSheet("background:#241B16;")
    win.show()

    mtimes: dict[int, float] = {}

    def refresh() -> None:
        for group, title in GROUPS:
            npz = LIVE / f"group{group}.npz"
            meta_f = LIVE / f"group{group}.json"
            if meta_f.exists():
                try:
                    meta = json.loads(meta_f.read_text(encoding="utf-8"))
                    age = time.time() - float(meta.get("updated", 0))
                    state = "学習中" if age < 5 else "停止中"
                    headers[group].setText(
                        f"{title} — ステージ {meta.get('stage')} / "
                        f"step {int(meta.get('steps', 0)):,} / "
                        f"episode {int(meta.get('episodes', 0)):,} / {state}")
                except Exception:
                    pass
            if npz.exists():
                mt = npz.stat().st_mtime
                if mt != mtimes.get(group):
                    mtimes[group] = mt
                    try:
                        with np.load(npz) as data:
                            frames = {int(k): data[k] for k in data.files}
                        grids[group].set_frames(frames)
                    except Exception:
                        pass

    timer = QTimer(win)
    timer.setInterval(500)
    timer.timeout.connect(refresh)
    timer.start()
    refresh()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
