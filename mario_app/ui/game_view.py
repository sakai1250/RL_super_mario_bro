"""ゲーム画面。NES 画面をアスペクト比を保って描き、黒いベゼルで囲む。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .style import PALETTE, pixel_font
from .widgets import SHADOW


class GameView(QWidget):
    NES_W, NES_H = 256, 240

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(256, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._source: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._stage = "3-4"

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._source = QPixmap.fromImage(image)
        self._rescale()
        self.update()

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        self.update()

    def set_idle(self) -> None:
        self._source = None
        self._scaled = None
        self.update()

    def has_frame(self) -> bool:
        return self._source is not None

    def _target_size(self) -> QSize:
        if self._source is None:
            return QSize(0, 0)
        return self._source.size().scaled(max(1, self.width()), max(1, self.height()), Qt.KeepAspectRatio)

    def _rescale(self) -> None:
        if self._source is None:
            self._scaled = None
            return
        target = self._target_size()
        sw, sh = self._source.width(), self._source.height()
        if target.width() % sw == 0 and target.height() % sh == 0:
            self._scaled = self._source.scaled(target, Qt.IgnoreAspectRatio, Qt.FastTransformation)
            return
        # 整数倍でないサイズ: 一度整数倍に拡大してから滑らかに縮めると、ドットの輪郭を保てる
        up = -(-target.width() // sw)
        pixmap = self._source
        if up > 1:
            pixmap = pixmap.scaled(sw * up, sh * up, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self._scaled = pixmap.scaled(target, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16110F"))
        if self._scaled is None:
            self._paint_placeholder(p)
        else:
            x = (self.width() - self._scaled.width()) // 2
            y = (self.height() - self._scaled.height()) // 2
            p.drawPixmap(x, y, self._scaled)
            self._paint_chip(p, f"WORLD {self._stage}", x + 8, y + 8)
        p.end()

    def _paint_placeholder(self, p: QPainter) -> None:
        p.setFont(pixel_font(24))
        p.setPen(QColor(PALETTE["cream"]))
        p.drawText(self.rect().adjusted(0, -20, 0, -20), Qt.AlignCenter, "SUPER MARIO BROS.")
        p.setFont(pixel_font(16))
        p.setPen(QColor(PALETTE["key_lo"]))
        p.drawText(self.rect().adjusted(0, 24, 0, 24), Qt.AlignCenter, "「開始」を押すとプレイが始まります")

    def _paint_chip(self, p: QPainter, text: str, left: int, top: int) -> None:
        p.setFont(pixel_font(16))
        w = p.fontMetrics().horizontalAdvance(text) + 12
        p.fillRect(left, top, w, 22, QColor(31, 18, 14, 200))
        p.setPen(QColor(PALETTE["cream"]))
        p.drawText(left + 6, top + 17, text)


class Bezel(QWidget):
    """黒いベゼルと硬い影の中にゲーム画面を置く。"""

    PAD = 10

    def __init__(self) -> None:
        super().__init__()
        self.game_view = GameView()
        lay = QVBoxLayout(self)
        m = self.PAD + 3
        lay.setContentsMargins(m, m, m + SHADOW, m + SHADOW)
        lay.addWidget(self.game_view)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        box = self.rect().adjusted(0, 0, -SHADOW, -SHADOW)
        ink = QColor(PALETTE["ink"])
        p.fillRect(box.translated(SHADOW, SHADOW), ink)
        p.fillRect(box, ink)
        p.fillRect(box.adjusted(3, 3, -3, -3), QColor("#16110F"))
        p.end()
