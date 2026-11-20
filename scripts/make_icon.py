#!/usr/bin/env python3
"""アプリアイコン (1024x1024 PNG) を生成する。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("icon_1024.png")
    app = QGuiApplication(sys.argv[:1])
    app.setApplicationName("Mario RL Player")

    size = 1024
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)

    margin = 64
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#E52521"))
    painter.drawRoundedRect(
        QRectF(margin, margin, size - 2 * margin, size - 2 * margin),
        size * 0.22,
        size * 0.22,
    )

    painter.setPen(QColor("#FFFFFF"))
    font = QFont("Menlo", int(size * 0.52), QFont.Black)
    painter.setFont(font)
    painter.drawText(image.rect().adjusted(0, -int(size * 0.02), 0, 0), Qt.AlignCenter, "M")

    painter.end()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(out), "PNG"):
        print(f"failed to save {out}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
