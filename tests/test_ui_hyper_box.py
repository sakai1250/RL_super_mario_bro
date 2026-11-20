import pytest

from mario_app.hyperparams import HP_SPECS, Hyperparams
from mario_app.ui.hyper_box import HyperBox, StepperCell


def test_step_emits_clamped_value(qapp):
    cell = StepperCell("gamma")
    seen = []
    cell.valueCommitted.connect(lambda n, v: seen.append((n, v)))
    cell.set_value(0.995)
    cell.step(+1)
    assert seen == [("gamma", 0.999)]


def test_commit_text_valid_clamps_and_invalid_is_ignored(qapp):
    cell = StepperCell("epsilon")
    seen = []
    cell.valueCommitted.connect(lambda n, v: seen.append((n, v)))
    cell.set_value(0.5)
    cell.commit_text("1.5")
    assert seen == [("epsilon", 1.0)]
    cell.set_value(0.5)
    cell.commit_text("abc")
    cell.commit_text("")
    assert seen == [("epsilon", 1.0)] and cell.value() == 0.5
    assert cell.display_text() == "0.500"


def test_hyperbox_has_ten_cells_in_spec_order(qapp):
    assert [c.name for c in HyperBox().cells()] == list(HP_SPECS)


def test_set_hyperparams_does_not_emit(qapp):
    box = HyperBox()
    seen = []
    box.hp_changed.connect(lambda n, v: seen.append(n))
    box.set_hyperparams(Hyperparams(gamma=0.5))
    assert seen == [] and box.hyperparams().gamma == 0.5
    assert box.cell("gamma").display_text() == "0.50"


def test_live_epsilon_changes_display_only(qapp):
    box = HyperBox()
    box.set_hyperparams(Hyperparams(epsilon=1.0))
    box.set_live_epsilon(0.3)
    assert box.cell("epsilon").display_text() == "0.300" and box.hyperparams().epsilon == 1.0


def test_user_edit_updates_hyperparams_and_emits(qapp):
    box = HyperBox()
    seen = []
    box.hp_changed.connect(lambda n, v: seen.append((n, v)))
    box.cell("learning_rate").step(+1)
    assert seen and seen[0][0] == "learning_rate" and seen[0][1] == pytest.approx(2e-4)
    assert box.hyperparams().learning_rate == pytest.approx(2e-4)
