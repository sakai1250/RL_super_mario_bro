"""LOG ボックス: 報酬グラフ、読み出し、ハイパラ変更のログ。"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

from .style import PALETTE, pixel_font
from .widgets import BORDER, BoxPanel, Lcd

MAX_LOG_LINES = 5


def moving_average(values: Sequence[float], window: int = 20) -> list[float]:
    """先頭は窓が満ちるまで、その時点までの平均。"""
    out: list[float] = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


class RewardChart(QWidget):
    """液晶色の地に、エピソード報酬（点）と移動平均（線）を描く。"""

    def __init__(self) -> None:
        super().__init__()
        self._values: list[float] = []
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(72)

    def set_values(self, values: Sequence[float]) -> None:
        self._values = list(values)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        r = self.rect()
        p.fillRect(r, QColor(PALETTE["ink"]))
        inner = r.adjusted(BORDER, BORDER, -BORDER, -BORDER)
        p.fillRect(inner, QColor(PALETTE["lcd"]))
        if len(self._values) >= 2:
            lo, hi = min(self._values), max(self._values)
            span = (hi - lo) or 1.0
            n = len(self._values)

            def pt(i: int, v: float) -> QPointF:
                x = inner.left() + 6 + (inner.width() - 12) * i / (n - 1)
                y = inner.bottom() - 6 - (inner.height() - 12) * (v - lo) / span
                return QPointF(x, y)

            ink = QColor(PALETTE["lcd_ink"])
            p.setPen(ink)
            for i, v in enumerate(self._values):
                q = pt(i, v)
                p.drawRect(QRect(int(q.x()), int(q.y()), 1, 1))
            p.setPen(QPen(ink, 2))
            ma = moving_average(self._values)
            for i in range(1, n):
                p.drawLine(pt(i - 1, ma[i - 1]), pt(i, ma[i]))
        p.end()


class LogBox(BoxPanel):
    READOUTS = (
        ("reward", "報酬"), ("best", "最高"), ("loss", "loss"), ("q", "Q値"),
        ("buffer", "バッファ"), ("ep_steps", "EP歩数"), ("action", "行動"),
        ("sps", "step/s"),
    )

    def __init__(self) -> None:
        super().__init__("LOG／記録")
        self.chart = RewardChart()
        self.body.addWidget(self.chart)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self._lcds: dict[str, Lcd] = {}
        for i, (key, label) in enumerate(self.READOUTS):
            name = QLabel(label)
            name.setFont(pixel_font(16))
            lcd = Lcd("—", chars=9)
            self._lcds[key] = lcd
            grid.addWidget(name, i // 2, (i % 2) * 2)
            grid.addWidget(lcd, i // 2, (i % 2) * 2 + 1)
        self.body.addLayout(grid)
        self._log_label = QLabel("")
        self._log_label.setFont(pixel_font(16))
        self._log_label.setWordWrap(True)
        self.body.addWidget(self._log_label)
        self._lines: list[str] = []

    def set_rewards(self, values: Sequence[float]) -> None:
        self.chart.set_values(values)

    def set_readouts(self, stats: dict) -> None:
        def fmt(value, spec: str = ",.1f") -> str:
            return "—" if value is None else format(value, spec)

        capacity = stats.get("hp", {}).get("buffer_capacity")
        buffer = stats.get("buffer_size")
        values = {
            "reward": fmt(stats.get("ep_reward")),
            "best": fmt(stats.get("best_reward")),
            "loss": fmt(stats.get("loss"), ".4f"),
            "q": fmt(stats.get("avg_q"), ".2f"),
            "buffer": "—" if buffer is None else (f"{buffer}/{capacity}" if capacity else str(buffer)),
            "ep_steps": fmt(stats.get("ep_steps"), ","),
            "action": str(stats.get("action", "—")),
            "sps": (f"{int(stats.get('envs', 1) or 1)}x{stats.get('steps_per_sec', 0):.0f}/s"
                    if int(stats.get("envs", 1) or 1) > 1
                    else fmt(stats.get("fps"), ".0f") + "FPS"),
        }
        for key, text in values.items():
            self._lcds[key].set_text(text)

    def append_log(self, line: str) -> None:
        self._lines = (self._lines + [line])[-MAX_LOG_LINES:]
        self._log_label.setText("\n".join(self._lines))

    def log_lines(self) -> list[str]:
        return list(self._lines)
