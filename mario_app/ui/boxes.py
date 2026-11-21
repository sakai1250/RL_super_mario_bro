"""SPEED / MODEL / SAVE ボックス。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QWidget

from ..checkpoint import find_models
from ..env import STAGES
from ..pacing import SPEEDS
from ..paths import MODELS_DIR, PROJECT_DIR
from .style import pixel_font
from .widgets import BoxPanel, Key, Lcd

NEW_MODEL_LABEL = "新規（ランダム初期化）"
AUTOSAVE_MAX = 100_000
AUTOSAVE_STEP = 1_000


class SpeedBox(BoxPanel):
    speed_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__("SPEED／速度")
        self._speed = 1.0
        self._keys: dict[float, Key] = {}
        row = QHBoxLayout()
        row.setSpacing(6)
        for label, value in SPEEDS:
            key = Key(label, checkable=True)
            key.clicked.connect(lambda _=False, v=value: self._on_key(v))
            self._keys[value] = key
            row.addWidget(key)
        self.body.addLayout(row)
        self._sync_keys()

    def key_for(self, speed: float) -> Key:
        return self._keys[speed]

    def speed(self) -> float:
        return self._speed

    def set_speed(self, value: float) -> None:
        self._speed = value
        self._sync_keys()

    def _on_key(self, value: float) -> None:
        self._speed = value
        self._sync_keys()
        self.speed_changed.emit(value)

    def _sync_keys(self) -> None:
        for value, key in self._keys.items():
            key.setChecked(value == self._speed)


class ModelBox(BoxPanel):
    """モデル・ステージ・デバイス・自動再開（実行中は無効）。"""

    model_changed = Signal(bool)  # 引数: 新規モデルか

    def __init__(self, roots: Sequence[Path] | None = None) -> None:
        super().__init__("MODEL／モデル")
        self._roots = list(roots) if roots is not None else [PROJECT_DIR, MODELS_DIR]

        row = QHBoxLayout()
        row.setSpacing(6)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_index)
        self.refresh_key = Key("更新")
        self.refresh_key.clicked.connect(self.refresh_models)
        self.open_key = Key("開く")
        self.open_key.clicked.connect(self.choose_model)
        row.addWidget(self.model_combo, 1)
        row.addWidget(self.refresh_key)
        row.addWidget(self.open_key)
        self.body.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.stage_combo = QComboBox()
        for stage in STAGES:
            self.stage_combo.addItem(f"WORLD {stage}", stage)
        self.stage_combo.setCurrentIndex(STAGES.index("3-4"))
        self.device_combo = QComboBox()
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA", "cuda")
        if torch.backends.mps.is_available():
            self.device_combo.addItem("MPS", "mps")
        self.device_combo.addItem("CPU", "cpu")
        self.auto_key = Key("自動再開", checkable=True)
        self.auto_key.setChecked(True)
        row2.addWidget(self.stage_combo, 1)
        row2.addWidget(self.device_combo)
        row2.addWidget(self.auto_key)
        self.body.addLayout(row2)

        self.refresh_models()

    def inputs(self) -> list[QWidget]:
        return [self.model_combo, self.refresh_key, self.open_key,
                self.stage_combo, self.device_combo, self.auto_key]

    def set_locked(self, locked: bool) -> None:
        for widget in self.inputs():
            widget.setEnabled(not locked)

    # ------------------------------------------------------------------ values
    def model_path(self) -> str | None:
        return self.model_combo.currentData()

    def is_new_model(self) -> bool:
        return self.model_path() is None

    def stage(self) -> str:
        return self.stage_combo.currentData() or "3-4"

    def set_stage(self, stage: str) -> None:
        self.stage_combo.setCurrentIndex(STAGES.index(stage))

    def device(self) -> str:
        return self.device_combo.currentData()

    def set_device(self, device: str) -> None:
        index = self.device_combo.findData(device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)

    def auto_restart(self) -> bool:
        return self.auto_key.isChecked()

    # ------------------------------------------------------------------ models
    def refresh_models(self) -> None:
        current = self.model_path()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem(NEW_MODEL_LABEL, None)
        for path in find_models(self._roots):
            self.model_combo.addItem(path.name, str(path))
        index = self.model_combo.findData(current) if current else 0
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)

    def select_path(self, path: str) -> None:
        index = self.model_combo.findData(path)
        if index < 0:
            self.model_combo.addItem(Path(path).name, path)
            index = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(index)

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "モデルを選択", str(PROJECT_DIR), "PyTorch モデル (*.pth *.pt);;すべてのファイル (*)"
        )
        if path:
            self.select_path(path)

    def _on_model_index(self, _index: int) -> None:
        self.model_changed.emit(self.is_new_model())


class SaveBox(BoxPanel):
    save_requested = Signal()
    autosave_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__("SAVE／保存")
        self._autosave = 5000
        row = QHBoxLayout()
        row.setSpacing(8)
        self.save_key = Key("SAVE NOW")
        self.save_key.clicked.connect(self.save_requested)
        row.addWidget(self.save_key, 1)
        auto = QLabel("AUTO")
        auto.setFont(pixel_font(16))
        left = Key("")
        left.set_glyph("left")
        left.setFixedSize(24, 26)
        left.setMinimumHeight(26)
        left.setAutoRepeat(True)
        left.clicked.connect(lambda: self.step_autosave(-1))
        right = Key("")
        right.set_glyph("right")
        right.setFixedSize(24, 26)
        right.setMinimumHeight(26)
        right.setAutoRepeat(True)
        right.clicked.connect(lambda: self.step_autosave(+1))
        self._auto_lcd = Lcd("", chars=9)
        row.addWidget(auto)
        row.addWidget(left)
        row.addWidget(self._auto_lcd)
        row.addWidget(right)
        self.body.addLayout(row)
        self._last = Lcd("LAST —", chars=0)
        self._unsaved = QLabel("")
        self._unsaved.setFont(pixel_font(16))
        self.body.addWidget(self._last)
        self.body.addWidget(self._unsaved)
        self._refresh_auto()

    def autosave_every(self) -> int:
        return self._autosave

    def step_autosave(self, direction: int) -> None:
        new = min(AUTOSAVE_MAX, max(0, self._autosave + direction * AUTOSAVE_STEP))
        if new != self._autosave:
            self._autosave = new
            self._refresh_auto()
            self.autosave_changed.emit(new)

    def _refresh_auto(self) -> None:
        self._auto_lcd.set_text("OFF" if self._autosave == 0 else f"{self._autosave}")

    def set_last_saved(self, text: str) -> None:
        self._last.set_text(f"LAST {text}" if text else "LAST —")

    def set_unsaved(self, steps: int) -> None:
        self._unsaved.setText(f"未保存 +{steps:,} STEP" if steps > 0 else "")
