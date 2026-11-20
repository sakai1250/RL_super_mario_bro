"""ファミコン調のメインウィンドウ。"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Slot
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..game_worker import GameWorker, RunConfig
from ..hyperparams import HP_SPECS, defaults_for, format_hp, with_value
from ..paths import MODELS_DIR, PROJECT_DIR, SCREENSHOT_DIR
from .boxes import ModelBox, SaveBox, SpeedBox
from .game_view import Bezel
from .header import Header
from .hyper_box import HyperBox
from .log_box import LogBox
from .menu import build_menu
from .style import build_qss, load_pixel_font
from .widgets import Key, Lcd

DEFAULT_WINDOW_SIZE = (1180, 840)
PANEL_MIN_WIDTH = 510
PANEL_MAX_WIDTH = 560


def needs_save_prompt(stats: dict | None) -> bool:
    """停止・終了時に保存確認を出すか（未保存の更新があるとき）。"""
    return bool(stats) and stats.get("unsaved_updates", 0) > 0


class MainWindow(QMainWindow):
    def __init__(self, models_dir: Path = MODELS_DIR, screenshot_dir: Path = SCREENSHOT_DIR) -> None:
        super().__init__()
        self._models_dir = Path(models_dir)
        self._screenshot_dir = Path(screenshot_dir)
        app = QGuiApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(load_pixel_font()))

        self.setWindowTitle("Mario RL Player")

        self._worker: GameWorker | None = None
        self._thread: QThread | None = None
        self.last_stats: dict | None = None
        self._last_frame_seq = -1

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(120)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(16)
        self._frame_timer.timeout.connect(self._pull_frame)

        build_menu(self)
        self._build_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("モデルを選んで「開始」を押してください", 6000)
        self._shortcuts = [
            self._shortcut("Space", self.toggle_pause),
            self._shortcut("Return", self.start_run),
            self._shortcut("Esc", self.stop_run),
        ]
        self.apply_model_defaults(is_new=self.model_box.is_new_model())
        self._set_running_ui(False)
        self.resize(self._initial_size().expandedTo(self.minimumSizeHint()))

    # ---------------------------------------------------------------- building
    @staticmethod
    def _initial_size() -> QSize:
        size = QSize(*DEFAULT_WINDOW_SIZE)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            size = size.boundedTo(QSize(int(avail.width() * 0.92), int(avail.height() * 0.92)))
        return size

    def _shortcut(self, seq: str, slot) -> QShortcut:
        sc = QShortcut(QKeySequence(seq), self)
        sc.activated.connect(slot)
        return sc

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = Header()
        outer.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(24, 16, 20, 12)
        body.setSpacing(22)
        outer.addLayout(body, 1)

        # ---- 左: ゲーム画面 / 液晶 / 操作キー
        left = QVBoxLayout()
        left.setSpacing(10)
        self.bezel = Bezel()
        self.game_view = self.bezel.game_view
        left.addWidget(self.bezel, 1)
        readout = QHBoxLayout()
        self.lcd_ep, self.lcd_step = Lcd("EP ---", chars=6), Lcd("STEP ------", chars=11)
        self.lcd_x, self.lcd_fps = Lcd("X ----", chars=6), Lcd("--FPS", chars=5)
        for lcd in (self.lcd_ep, self.lcd_step, self.lcd_x, self.lcd_fps):
            readout.addWidget(lcd)
        left.addLayout(readout)
        tx = QHBoxLayout()
        tx.setSpacing(8)
        self.start_key = Key("開始")
        self.start_key.set_glyph("play")
        self.pause_key = Key("一時停止")
        self.pause_key.set_glyph("pause")
        self.stop_key = Key("停止")
        self.stop_key.set_glyph("stop")
        self.learn_key = Key("学習", checkable=True)
        for key in (self.start_key, self.pause_key, self.stop_key, self.learn_key):
            key.setMinimumHeight(44)
            tx.addWidget(key)
        self.start_key.clicked.connect(self.start_run)
        self.pause_key.clicked.connect(self.toggle_pause)
        self.stop_key.clicked.connect(lambda: self.stop_run())
        self.learn_key.clicked.connect(self._on_learn_clicked)
        left.addLayout(tx)
        body.addLayout(left, 1)

        # ---- 右: 縦スクロール
        self.model_box = ModelBox(roots=[PROJECT_DIR, self._models_dir])
        self.speed_box = SpeedBox()
        self.hyper_box = HyperBox()
        self.save_box = SaveBox()
        self.log_box = LogBox()
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(14)
        for box in (self.model_box, self.speed_box, self.hyper_box, self.save_box, self.log_box):
            col.addWidget(box)
        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(container)
        scroll.setMinimumWidth(PANEL_MIN_WIDTH)
        scroll.setMaximumWidth(PANEL_MAX_WIDTH)
        body.addWidget(scroll, 0)

        self.model_box.model_changed.connect(self.apply_model_defaults)
        self.model_box.stage_combo.currentIndexChanged.connect(
            lambda _=0: self.game_view.set_stage(self.model_box.stage())
        )
        self.speed_box.speed_changed.connect(self._on_speed)
        self.hyper_box.hp_changed.connect(self.route_hp_change)
        self.save_box.save_requested.connect(self.save_weights)
        self.save_box.autosave_changed.connect(self._on_autosave)

    # ---------------------------------------------------------------- live ops
    def apply_model_defaults(self, is_new: bool) -> None:
        self.learn_key.setChecked(is_new)
        self.hyper_box.set_hyperparams(defaults_for(is_new))

    def route_hp_change(self, name: str, value: float) -> None:
        hp = with_value(self.hyper_box.hyperparams(), name, value)
        shown = format_hp(name, getattr(hp, name))
        self.log_box.append_log(f"{HP_SPECS[name].label} → {shown} (手動)")
        if self._worker is None:
            return
        if name == "epsilon":
            self._worker.set_epsilon(getattr(hp, name))
        else:
            self._worker.update_hyperparams(hp)

    def _on_speed(self, speed: float) -> None:
        if self._worker is not None:
            self._worker.set_speed(speed)

    def _on_learn_clicked(self) -> None:
        if self._worker is not None:
            self._worker.set_learning(self.learn_key.isChecked())

    def _on_autosave(self, n: int) -> None:
        if self._worker is not None:
            self._worker.set_autosave_every(n)

    def save_weights(self) -> None:
        if self._worker is None:
            self.statusBar().showMessage("実行中ではないので保存できる重みがありません", 3000)
            return
        self._worker.request_save()

    # ---------------------------------------------------------------- run flow
    def start_run(self) -> None:
        if self._thread is not None:
            return
        model_path = self.model_box.model_path()
        if model_path is not None and not Path(model_path).is_file():
            QMessageBox.warning(self, "モデルがありません", "選択したモデルのファイルが見つかりません。")
            return
        config = RunConfig(
            model_path=model_path,
            stage=self.model_box.stage(),
            hp=self.hyper_box.hyperparams(),
            learning=self.learn_key.isChecked(),
            speed=self.speed_box.speed(),
            auto_restart=self.model_box.auto_restart(),
            device=self.model_box.device(),
            autosave_every=self.save_box.autosave_every(),
            models_dir=str(self._models_dir),
        )
        self._worker = GameWorker(config)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        w, t = self._worker, self._thread
        t.started.connect(w.run)
        w.stats_ready.connect(self._on_stats)
        w.episode_finished.connect(self._on_episode_finished)
        w.state_changed.connect(self._on_state_changed)
        w.saved.connect(self._on_saved)
        w.failed.connect(self._on_failed)
        w.finished.connect(t.quit)
        w.finished.connect(w.deleteLater)
        t.finished.connect(t.deleteLater)
        t.finished.connect(self._on_thread_finished)

        self.last_stats = None
        self._last_frame_seq = -1
        self._stats_timer.start()
        self._frame_timer.start()
        self.log_box.set_rewards([])
        self._set_running_ui(True)
        self.header.device_led.set_label(config.device.upper() if config.device != "auto" else "AUTO")
        self.statusBar().showMessage("ゲームを開始しています…")
        t.start()

    def toggle_pause(self) -> None:
        if self._worker is None:
            return
        if self._worker.is_paused():
            self._worker.request_resume()
            self.pause_key.setText("一時停止")
        else:
            self._worker.request_pause()
            self.pause_key.setText("再開")

    def stop_run(self, prompt: bool = True) -> None:
        if self._worker is None:
            return
        if prompt and needs_save_prompt(self.last_stats):
            choice = self._ask_save()
            if choice == "cancel":
                return
            if choice == "save":
                self._worker.request_save()
        self._worker.request_stop()
        self.stop_key.setEnabled(False)
        self.pause_key.setEnabled(False)
        self.statusBar().showMessage("停止しています…")

    def _ask_save(self) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("未保存の学習があります")
        box.setText("学習した重みがまだ保存されていません。")
        save = box.addButton("保存して停止", QMessageBox.AcceptRole)
        discard = box.addButton("保存しない", QMessageBox.DestructiveRole)
        box.addButton("キャンセル", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        return "save" if clicked is save else "discard" if clicked is discard else "cancel"

    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        self._stats_timer.stop()
        self._frame_timer.stop()
        self._set_running_ui(False)
        self.pause_key.setText("一時停止")
        for led in (self.header.learn_led, self.header.device_led):
            led.set_on(False)
        self.statusBar().showMessage("停止しました", 4000)

    def _set_running_ui(self, running: bool) -> None:
        self.model_box.set_locked(running)
        self.open_action.setEnabled(not running)
        self.start_key.setEnabled(not running)
        self.pause_key.setEnabled(running)
        self.stop_key.setEnabled(running)

    # ------------------------------------------------------------------- slots
    def _pull_frame(self) -> None:
        if self._worker is None:
            return
        seq, frame = self._worker.latest_frame()
        if frame is not None and seq != self._last_frame_seq:
            self._last_frame_seq = seq
            self.game_view.set_frame(frame)

    @Slot(object)
    def _on_stats(self, stats: dict) -> None:
        self.last_stats = stats

    def _refresh_stats(self) -> None:
        s = self.last_stats
        if not s:
            return
        self.lcd_ep.set_text(f"EP {s.get('episode', 0):03d}")
        self.lcd_step.set_text(f"STEP {s.get('agent_step', 0):06d}")
        self.lcd_x.set_text(f"X {s.get('x_pos', 0):.0f}")
        self.lcd_fps.set_text(f"{s.get('fps', 0):.0f}FPS")
        self.hyper_box.set_live_epsilon(s.get("epsilon", 0.0))
        self.save_box.set_last_saved(s.get("last_saved") or "")
        self.save_box.set_unsaved(s.get("unsaved_steps", 0))
        self.log_box.set_readouts(s)
        running = self._worker is not None and not self._worker.is_paused()
        self.header.learn_led.set_on(bool(s.get("learning")) and running)
        self.header.device_led.set_label(str(s.get("device", "")).upper())
        self.header.device_led.set_on(self._worker is not None)

    @Slot(object)
    def _on_episode_finished(self, summary: dict) -> None:
        self.log_box.set_rewards([h.get("reward", 0.0) for h in summary.get("history", [])])
        message = (f"Episode {summary.get('episode')} 終了 — 報酬 {summary.get('reward', 0.0):,.1f}"
                   f" / {summary.get('steps', 0):,} 歩")
        if summary.get("flag"):
            message += " ゴール!"
        self.statusBar().showMessage(message, 6000)

    @Slot(object)
    def _on_saved(self, info: dict) -> None:
        self.header.save_led.flash(2000)
        name = Path(info["path"]).name
        self.statusBar().showMessage(f"重みを保存しました: {name}", 5000)
        self.log_box.append_log(f"保存 {name}")

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        if state == "loading":
            self.statusBar().showMessage("モデルを読み込んでいます…")
        elif state == "running":
            self.pause_key.setText("一時停止")
            self.statusBar().showMessage("実行中 — Space で一時停止 / Esc で停止", 4000)
        elif state == "paused":
            self.pause_key.setText("再開")

    @Slot(str)
    def _on_failed(self, traceback_text: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("エラーが発生しました")
        box.setText("ゲームを続行できませんでした。")
        box.setDetailedText(traceback_text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    # ----------------------------------------------------------------- actions
    def save_screenshot(self) -> None:
        if not self.game_view.has_frame():
            self.statusBar().showMessage("スクリーンショットできる画面がまだありません", 3000)
            return
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self._screenshot_dir / f"mario_{stamp}.png"
        if self.game_view.grab().save(str(path), "PNG"):
            self.statusBar().showMessage(f"スクリーンショットを保存しました: {path.name}", 5000)
        else:
            self.statusBar().showMessage("保存に失敗しました", 4000)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_action.setChecked(False)
        else:
            self.showFullScreen()
            self.fullscreen_action.setChecked(True)

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            if needs_save_prompt(self.last_stats):
                choice = self._ask_save()
                if choice == "cancel":
                    event.ignore()
                    return
                if choice == "save":
                    self._worker.request_save()
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(8000)
        event.accept()
