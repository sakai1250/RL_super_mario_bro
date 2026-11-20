"""ハイパラの 10 セル（◀ 液晶 ▶）と、それを並べる HYPER ボックス。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QLineEdit, QWidget

from ..hyperparams import HP_SPECS, Hyperparams, clamp_hp, format_hp, parse_hp, step_value, with_value
from .style import pixel_font
from .widgets import BoxPanel, Key, Lcd


class StepperCell(QWidget):
    """ラベル + ◀ + 液晶 + ▶。液晶クリックで直接入力できる。"""

    valueCommitted = Signal(str, float)

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        spec = HP_SPECS[name]
        self._value: float = spec.minimum
        self._editing = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        label = QLabel(spec.label)
        label.setFont(pixel_font(16))
        label.setMinimumWidth(64)
        row.addWidget(label)
        self._left = self._arrow("left", -1)
        self._right = self._arrow("right", +1)
        self._lcd = Lcd("", chars=9)
        self._lcd.clicked.connect(self._begin_edit)
        self._edit = QLineEdit(self._lcd)
        self._edit.hide()
        self._edit.returnPressed.connect(self._finish_edit)
        self._edit.installEventFilter(self)
        row.addWidget(self._left)
        row.addWidget(self._lcd)
        row.addWidget(self._right)

    def _arrow(self, glyph: str, direction: int) -> Key:
        key = Key("")
        key.set_glyph(glyph)
        key.setFixedSize(24, 26)
        key.setMinimumHeight(26)
        key.setAutoRepeat(True)          # 長押しで連続して刻む
        key.setAutoRepeatDelay(400)
        key.setAutoRepeatInterval(90)
        key.clicked.connect(lambda: self.step(direction))
        return key

    # ------------------------------------------------------------- public api
    def value(self) -> float:
        return self._value

    def display_text(self) -> str:
        return self._lcd.text()

    def set_value(self, value: float) -> None:
        """表示だけを更新する（編集中は無視）。"""
        if self._editing:
            return
        self._value = value
        self._lcd.set_text(format_hp(self.name, value))

    def set_cell_enabled(self, enabled: bool) -> None:
        self.setEnabled(enabled)

    def step(self, direction: int) -> None:
        new = step_value(self.name, self._value, direction)
        if new != self._value:
            self._commit(new)

    def commit_text(self, text: str) -> None:
        parsed = parse_hp(self.name, text)
        if parsed is None:
            self._lcd.set_text(format_hp(self.name, self._value))
            return
        self._commit(clamp_hp(self.name, parsed))

    # --------------------------------------------------------------- internals
    def _commit(self, value: float) -> None:
        self._value = value
        self._lcd.set_text(format_hp(self.name, value))
        self.valueCommitted.emit(self.name, float(value))

    def _begin_edit(self) -> None:
        if not self.isEnabled():
            return
        self._editing = True
        self._edit.setGeometry(self._lcd.rect())
        self._edit.setText(format_hp(self.name, self._value))
        self._edit.show()
        self._edit.setFocus()
        self._edit.selectAll()

    def _finish_edit(self) -> None:
        text = self._edit.text()
        self._end_edit()
        self.commit_text(text)

    def _end_edit(self) -> None:
        self._editing = False
        self._edit.hide()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent, Qt

        if obj is self._edit and event.type() in (QEvent.ShortcutOverride, QEvent.KeyPress):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
                if event.type() == QEvent.ShortcutOverride:
                    event.accept()  # ウィンドウの Return / Esc ショートカットより入力欄を優先する
                    return False
                if event.key() == Qt.Key_Escape:  # Esc で取り消し
                    self._end_edit()
                    return True
        return super().eventFilter(obj, event)


class HyperBox(BoxPanel):
    hp_changed = Signal(str, float)

    def __init__(self) -> None:
        super().__init__("HYPER／ハイパラ")
        self._hp = Hyperparams()
        self._cells: dict[str, StepperCell] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        for i, name in enumerate(HP_SPECS):
            cell = StepperCell(name)
            cell.valueCommitted.connect(self._on_cell)
            self._cells[name] = cell
            grid.addWidget(cell, i // 2, i % 2)
        self.body.addLayout(grid)
        self.set_hyperparams(self._hp)

    def cells(self) -> list[StepperCell]:
        return list(self._cells.values())

    def cell(self, name: str) -> StepperCell:
        return self._cells[name]

    def hyperparams(self) -> Hyperparams:
        return self._hp

    def set_hyperparams(self, hp: Hyperparams) -> None:
        """プログラムからの設定。シグナルは出さない。"""
        self._hp = hp
        for name, cell in self._cells.items():
            cell.set_value(getattr(hp, name))

    def set_live_epsilon(self, value: float) -> None:
        """実際の ε（減衰後）を表示する。``hyperparams().epsilon`` は変えない。"""
        self._cells["epsilon"].set_value(value)

    def _on_cell(self, name: str, value: float) -> None:
        self._hp = with_value(self._hp, name, value)
        self.hp_changed.emit(name, value)
