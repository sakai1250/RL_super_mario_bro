import numpy as np

from mario_app.hyperparams import Hyperparams
from mario_app.pacing import SPEED_MAX
from mario_app.ui.main_window import MainWindow, needs_save_prompt


class FakeWorker:
    def __init__(self):
        self.calls = []

    def _rec(self, name, arg=None):
        self.calls.append((name, arg))

    def set_speed(self, v): self._rec("set_speed", v)
    def set_learning(self, on): self._rec("set_learning", on)
    def set_epsilon(self, v): self._rec("set_epsilon", v)
    def update_hyperparams(self, hp): self._rec("update_hyperparams", hp)
    def set_autosave_every(self, n): self._rec("set_autosave_every", n)
    def request_save(self): self._rec("request_save")
    def request_pause(self): self._rec("request_pause")
    def request_resume(self): self._rec("request_resume")
    def request_stop(self): self._rec("request_stop")
    def is_paused(self): return False
    def latest_frame(self): return 0, None


def make(tmp_path):
    return MainWindow(models_dir=tmp_path, screenshot_dir=tmp_path)


def test_needs_save_prompt():
    assert needs_save_prompt(None) is False
    assert needs_save_prompt({"unsaved_updates": 0}) is False
    assert needs_save_prompt({"unsaved_updates": 3}) is True


def test_default_and_minimum_size_fit_the_spec(qapp, tmp_path):
    win = make(tmp_path)
    hint = win.minimumSizeHint()
    assert hint.width() <= 910 and hint.height() <= 600
    assert win.width() >= hint.width() and win.height() >= hint.height()


def test_idle_state_enables_setup_and_disables_live_stop(qapp, tmp_path):
    win = make(tmp_path)
    assert win.start_key.isEnabled() and not win.stop_key.isEnabled()
    assert not win.pause_key.isEnabled() and win.model_box.model_combo.isEnabled()


def test_apply_model_defaults(qapp, tmp_path):
    win = make(tmp_path)
    win.apply_model_defaults(is_new=True)
    assert win.learn_key.isChecked() and win.hyper_box.cell("epsilon").display_text() == "1.000"
    win.apply_model_defaults(is_new=False)
    assert not win.learn_key.isChecked() and win.hyper_box.cell("epsilon").display_text() == "0.000"


def test_hp_changes_route_to_the_worker(qapp, tmp_path):
    win = make(tmp_path)
    win._worker = FakeWorker()
    win.route_hp_change("epsilon", 0.3)
    win.route_hp_change("gamma", 0.95)
    assert win._worker.calls[0] == ("set_epsilon", 0.3)
    assert win._worker.calls[1][0] == "update_hyperparams" and win._worker.calls[1][1].gamma == 0.95


def test_route_hp_change_without_worker_does_not_raise(qapp, tmp_path):
    make(tmp_path).route_hp_change("gamma", 0.95)


def test_live_keys_call_the_worker(qapp, tmp_path):
    win = make(tmp_path)
    win._worker = FakeWorker()
    win.speed_box.key_for(2.0).click()
    win.learn_key.click()
    win.save_box.save_key.click()
    calls = win._worker.calls
    assert ("set_speed", 2.0) in calls and any(c[0] == "set_learning" for c in calls)
    assert ("request_save", None) in calls


def test_screenshot_is_saved_as_png(qapp, tmp_path):
    win = make(tmp_path)
    win.game_view.set_frame(np.zeros((240, 256, 3), np.uint8))
    win.save_screenshot()
    assert list(tmp_path.glob("mario_*.png"))


def test_enter_and_escape_in_the_value_editor_do_not_trigger_window_shortcuts(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    win = make(tmp_path)
    win.show()
    qapp.setActiveWindow(win)
    hits = []
    for shortcut in win._shortcuts:
        shortcut.activated.connect(lambda: hits.append(1))
    cell = win.hyper_box.cell("gamma")
    cell._begin_edit()
    cell._edit.setText("0.5")
    QTest.keyClick(cell._edit, Qt.Key_Return)
    assert hits == [] and win.hyper_box.hyperparams().gamma == 0.5
    cell._begin_edit()
    QTest.keyClick(cell._edit, Qt.Key_Escape)
    assert hits == [] and not cell._editing
