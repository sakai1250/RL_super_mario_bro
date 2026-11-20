"""えんじ色の帯 + 金の線 + 墨の線のヘッダー（タイトル・LED）。"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .style import PALETTE, pixel_font
from .widgets import Led


class Header(QWidget):
    """えんじ色の帯 + 金の線 + 墨の線。"""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(62)
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 0, 24, 4)
        title = QLabel("MARIO RL")
        title.setFont(pixel_font(32))
        title.setStyleSheet(f"color: {PALETTE['cream']}; font-size: 32px;")
        plate = QLabel("TRAINER")
        plate.setFont(pixel_font(16))
        plate.setStyleSheet(f"color: {PALETTE['cream']}; border: 1px solid {PALETTE['cream']}; padding: 0 8px;")
        row.addWidget(title)
        row.addSpacing(14)
        row.addWidget(plate)
        row.addStretch(1)
        self.learn_led = Led("学習")
        self.save_led = Led("保存")
        self.device_led = Led("CPU")
        for led in (self.learn_led, self.save_led, self.device_led):
            row.addWidget(led)
            row.addSpacing(10)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, 54, QColor(PALETTE["maroon"]))
        p.fillRect(0, 54, w, 4, QColor(PALETTE["gold"]))
        p.fillRect(0, 58, w, h - 58, QColor(PALETTE["ink"]))
        p.end()
