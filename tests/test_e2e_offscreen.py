import time

import pytest
from PySide6.QtWidgets import QApplication

from mario_app.hyperparams import Hyperparams
from mario_app.pacing import SPEED_MAX
from mario_app.ui.main_window import MainWindow


def wait_until(predicate, timeout):
    """GUI スレッドのイベントを回しつつ待つ。QTest.qWait は GIL を離さずワーカーを止めるので使わない。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("タイムアウトしました")


@pytest.mark.slow
def test_gui_run_with_live_changes_and_save(qapp, tmp_path):
    win = MainWindow(models_dir=tmp_path, screenshot_dir=tmp_path)
    win.model_box.set_device("cpu")
    win.model_box.set_stage("1-1")
    win.speed_box.set_speed(SPEED_MAX)
    win.apply_model_defaults(True)
    win.hyper_box.set_hyperparams(
        Hyperparams(batch_size=8, burnin=40, learn_every=1, sync_every=50, buffer_capacity=1000))
    win.start_run()
    wait_until(lambda: win.last_stats and win.last_stats["agent_step"] >= 150, timeout=60)
    win.speed_box.key_for(0.25).click()
    wait_until(lambda: win.last_stats["speed"] == 0.25, timeout=15)
    win.hyper_box.cell("gamma").commit_text("0.95")
    wait_until(lambda: win.last_stats["hp"]["gamma"] == 0.95, timeout=15)
    win.save_weights()
    wait_until(lambda: list(tmp_path.glob("mario_*_s*.pth")), timeout=15)
    win.stop_run(prompt=False)
    wait_until(lambda: win._worker is None, timeout=20)
    assert win.last_stats["updates"] > 0
