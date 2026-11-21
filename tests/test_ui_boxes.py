from mario_app.pacing import SPEED_MAX
from mario_app.ui.boxes import ModelBox, SaveBox, SpeedBox
from mario_app.ui.log_box import LogBox, moving_average


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
    assert moving_average([], 20) == []


def test_speedbox_default_and_signal(qapp):
    box = SpeedBox()
    seen = []
    box.speed_changed.connect(seen.append)
    assert box.speed() == 1.0
    box.key_for(SPEED_MAX).click()
    assert seen == [SPEED_MAX] and box.speed() == SPEED_MAX
    box.set_speed(0.5)
    assert seen == [SPEED_MAX] and box.speed() == 0.5


def test_modelbox_first_entry_is_new_and_signals(qapp, tmp_path):
    (tmp_path / "a.pth").touch()
    box = ModelBox(roots=[tmp_path])
    seen = []
    box.model_changed.connect(seen.append)
    assert box.is_new_model() and box.model_path() is None
    box.select_path(str(tmp_path / "a.pth"))
    assert seen == [False] and box.model_path() == str(tmp_path / "a.pth")


def test_modelbox_locked_disables_inputs(qapp, tmp_path):
    box = ModelBox(roots=[tmp_path])
    box.set_locked(True)
    assert all(not w.isEnabled() for w in box.inputs())
    box.set_locked(False)
    assert all(w.isEnabled() for w in box.inputs())


def test_modelbox_defaults_and_setters(qapp, tmp_path):
    box = ModelBox(roots=[tmp_path])
    assert box.stage() == "3-4" and box.auto_restart() is True
    assert box.device() in {"auto", "cuda", "mps", "cpu"}
    box.set_stage("1-1")
    box.set_device("cpu")
    assert box.stage() == "1-1" and box.device() == "cpu"


def test_savebox_autosave_stepper_bounds(qapp):
    box = SaveBox()
    seen = []
    box.autosave_changed.connect(seen.append)
    assert box.autosave_every() == 5000
    for _ in range(5):
        box.step_autosave(-1)
    assert box.autosave_every() == 0
    for _ in range(200):
        box.step_autosave(+1)
    assert box.autosave_every() == 100_000


def test_savebox_emits_save_requested(qapp):
    box = SaveBox()
    hits = []
    box.save_requested.connect(lambda: hits.append(1))
    box.save_key.click()
    assert hits == [1]


def test_logbox_keeps_last_five_lines(qapp):
    box = LogBox()
    for i in range(8):
        box.append_log(f"line {i}")
    assert box.log_lines() == [f"line {i}" for i in range(3, 8)]
