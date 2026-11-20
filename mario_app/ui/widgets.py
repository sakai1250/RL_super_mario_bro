"""ファミコン調の自前描画ウィジェット（箱・キー・液晶・LED）。

絵文字・グラデーション・ぼかし影・光彩は使わない。角は直角。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPolygonF
from PySide6.QtWidgets import QAbstractButton, QFrame, QSizePolicy, QVBoxLayout, QWidget

from .style import PALETTE, pixel_font

SHADOW = 5
BORDER = 3


def _c(name: str) -> QColor:
    return QColor(PALETTE[name])


def draw_glyph(painter: QPainter, kind: str, rect: QRect) -> None:
    """▶ Ⅱ ■ ◀ を図形として描く（ドットフォントに字形が無いため）。"""
    painter.save()
    painter.setPen(Qt.NoPen)
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    if kind in ("play", "right"):
        painter.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x, y + h), QPointF(x + w, y + h / 2)]))
    elif kind == "left":
        painter.drawPolygon(QPolygonF([QPointF(x + w, y), QPointF(x + w, y + h), QPointF(x, y + h / 2)]))
    elif kind == "stop":
        painter.drawRect(rect)
    elif kind == "pause":
        bar = max(2, w // 3)
        painter.drawRect(QRect(x, y, bar, h))
        painter.drawRect(QRect(x + w - bar, y, bar, h))
    painter.restore()


class BoxPanel(QFrame):
    """墨枠・パネル色・右下の硬い影・左上のえんじの札を持つ箱。"""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(BORDER + 11, 24, BORDER + 11 + SHADOW, BORDER + 9 + SHADOW)
        self.body.setSpacing(8)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        top = 12
        box = QRect(0, top, self.width() - SHADOW, self.height() - top - SHADOW)
        p.fillRect(box.translated(SHADOW, SHADOW), _c("ink"))
        p.fillRect(box, _c("ink"))
        p.fillRect(box.adjusted(BORDER, BORDER, -BORDER, -BORDER), _c("panel"))
        p.setFont(pixel_font(16))
        tab_w = p.fontMetrics().horizontalAdvance(self._title) + 20
        tab = QRect(12, 0, tab_w, 24)
        p.fillRect(tab, _c("maroon"))
        p.setPen(_c("cream"))
        p.drawText(tab, Qt.AlignCenter, self._title)
        p.end()


class Key(QAbstractButton):
    """押し込み式のキー。チェック・無効・accent（えんじ）の見た目を持つ。"""

    def __init__(self, text: str, *, checkable: bool = False, tone: str = "normal") -> None:
        super().__init__()
        self.setText(text)
        self.setCheckable(checkable)
        self._tone = tone
        self._glyph: str | None = None
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(34)

    def set_glyph(self, kind: str | None) -> None:
        self._glyph = kind
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self.text()) + 32, 36)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        down = self.isDown() or self.isChecked() or self._tone == "accent"
        r = self.rect()
        p.fillRect(r, _c("ink"))
        inner = r.adjusted(BORDER, BORDER, -BORDER, -BORDER)
        p.fillRect(inner, _c("maroon") if down else _c("key"))
        if down:
            p.fillRect(QRect(inner.x(), inner.y(), inner.width(), 3), QColor("#3F100B"))
        else:
            p.fillRect(QRect(inner.x(), inner.y(), inner.width(), 3), _c("key_hi"))
            p.fillRect(QRect(inner.x(), inner.bottom() - 2, inner.width(), 3), _c("key_lo"))
        color = _c("cream") if down else _c("ink")
        if not self.isEnabled():
            color.setAlpha(110)
        p.setPen(color)
        p.setBrush(color)
        p.setFont(pixel_font(16))
        text_rect = inner
        if self._glyph and not self.text():
            gs = min(12, inner.height() - 6)
            draw_glyph(p, self._glyph, QRect(inner.center().x() - gs // 2, inner.center().y() - gs // 2, gs, gs))
        elif self._glyph:
            gs = 12
            gx = inner.x() + (inner.width() - gs - 8 - p.fontMetrics().horizontalAdvance(self.text())) // 2
            draw_glyph(p, self._glyph, QRect(gx, inner.center().y() - gs // 2, gs, gs))
            text_rect = inner.adjusted(gx - inner.x() + gs + 8, 0, 0, 0)
            p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())
        else:
            p.drawText(text_rect, Qt.AlignCenter, self.text())
        if not self.isEnabled():
            p.fillRect(r, QColor(230, 222, 198, 90))
        p.end()


class Lcd(QWidget):
    """緑がかった液晶風の表示。chars 桁分の固定幅で右寄せ。"""

    clicked = Signal()

    def __init__(self, text: str = "", chars: int = 0) -> None:
        super().__init__()
        self._text = text
        self._chars = chars
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        font = pixel_font(16)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(font)
        digits = self._chars if self._chars else len(self._text)
        width = fm.horizontalAdvance("0") * digits if self._chars else fm.horizontalAdvance(self._text)
        return QSize(width + 2 * (BORDER + 6), 26)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        r = self.rect()
        p.fillRect(r, _c("ink"))
        inner = r.adjusted(BORDER, BORDER, -BORDER, -BORDER)
        p.fillRect(inner, _c("lcd"))
        p.setPen(_c("lcd_ink"))
        p.setFont(pixel_font(16))
        p.drawText(inner.adjusted(0, 0, -6, 0), Qt.AlignVCenter | Qt.AlignRight, self._text)
        p.end()


class Led(QWidget):
    """ラベル付きの丸い LED。flash() で一定時間だけ点灯する。"""

    def __init__(self, label: str) -> None:
        super().__init__()
        self._label = label
        self._on = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self.set_on(False))
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def is_on(self) -> bool:
        return self._on

    def set_label(self, label: str) -> None:
        self._label = label
        self.updateGeometry()
        self.update()

    def set_on(self, on: bool) -> None:
        if on != self._on:
            self._on = on
            self.update()

    def flash(self, ms: int = 2000) -> None:
        self.set_on(True)
        self._timer.start(ms)

    def sizeHint(self) -> QSize:
        from PySide6.QtGui import QFontMetrics
        return QSize(18 + QFontMetrics(pixel_font(16)).horizontalAdvance(self._label) + 4, 20)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        cy = self.height() // 2
        p.setPen(_c("ink"))
        p.setBrush(_c("led_on") if self._on else _c("led_off"))
        p.drawEllipse(QRectF(1, cy - 5, 10, 10))
        p.setPen(_c("cream"))
        p.setFont(pixel_font(16))
        p.drawText(QRect(18, 0, self.width() - 18, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self._label)
        p.end()
