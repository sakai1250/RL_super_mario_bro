import dataclasses
import threading

from mario_app.controls import RunControls
from mario_app.hyperparams import Hyperparams


def make():
    return RunControls(speed=1.0, learning=True, hp=Hyperparams(), autosave_every=5000)


def test_take_epsilon_and_save_are_one_shot():
    c = make()
    c.request_epsilon(0.3)
    assert c.take_epsilon() == 0.3 and c.take_epsilon() is None
    c.request_save()
    assert c.take_save_request() is True and c.take_save_request() is False


def test_setters_are_visible_across_threads():
    c = make()
    t = threading.Thread(target=lambda: (
        c.set_speed(0.5), c.set_learning(False),
        c.set_hyperparams(dataclasses.replace(c.hyperparams, gamma=0.5)), c.set_autosave_every(100)))
    t.start()
    t.join()
    assert (c.speed, c.learning, c.hyperparams.gamma, c.autosave_every) == (0.5, False, 0.5, 100)


def test_pause_and_stop_flags():
    c = make()
    assert not c.is_paused()
    c.request_pause()
    assert c.is_paused()
    c.request_resume()
    assert not c.is_paused()
    assert not c.is_stopped()
    c.request_stop()
    assert c.is_stopped() and c.stop_event.is_set()
