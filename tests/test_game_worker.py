import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mario_app.agent import MarioNet, load_weights
from mario_app.game_worker import GameWorker, RunConfig
from mario_app.hyperparams import Hyperparams
from mario_app.pacing import SPEED_MAX

HP = Hyperparams(batch_size=8, burnin=40, learn_every=1, sync_every=50,
                 buffer_capacity=1000, epsilon=1.0, epsilon_decay=0.99)


def cfg(tmp_path, model_path=None, **kw):
    base = dict(model_path=model_path, stage="1-1", hp=HP, learning=True, speed=SPEED_MAX,
                device="cpu", autosave_every=0, models_dir=str(tmp_path))
    base.update(kw)
    return RunConfig(**base)


@dataclass
class Out:
    stats: list = field(default_factory=list)
    saves: list = field(default_factory=list)
    failed: list = field(default_factory=list)


def run_until(worker, predicate, on_stats=None, save_on_stop=False):
    out = Out()
    worker.saved.connect(out.saves.append)
    worker.failed.connect(out.failed.append)

    def handle(s):
        out.stats.append(s)
        if on_stats:
            on_stats(s)
        if predicate(s):
            if save_on_stop:
                worker.request_save()
            worker.request_stop()

    worker.stats_ready.connect(handle)
    worker.run()
    return out


def test_new_model_learns_and_manual_save_is_loadable(tmp_path, qapp):
    out = run_until(GameWorker(cfg(tmp_path)), lambda s: s["agent_step"] >= 120, save_on_stop=True)
    assert not out.failed and out.saves
    last = out.stats[-1]
    assert last["learning"] is True and last["buffer_size"] > 0
    assert math.isfinite(last["loss"]) and last["device"] == "cpu"
    load_weights(MarioNet(), out.saves[0]["path"])
    meta = json.loads(Path(out.saves[0]["path"]).with_suffix(".json").read_text())
    assert meta["source"] == "new" and meta["auto"] is False


def test_play_mode_does_not_learn_and_can_still_save(tmp_path, qapp):
    out = run_until(GameWorker(cfg(tmp_path, learning=False)), lambda s: s["agent_step"] >= 40, save_on_stop=True)
    assert out.stats[-1]["buffer_size"] == 0 and out.stats[-1]["updates"] == 0
    meta = json.loads(Path(out.saves[0]["path"]).with_suffix(".json").read_text())
    assert meta["updates"] == 0


def test_live_controls_take_effect(tmp_path, qapp):
    w = GameWorker(cfg(tmp_path))
    applied = []

    def on(s):
        if s["agent_step"] >= 30 and not applied:
            applied.append(1)
            w.set_speed(0.5)
            w.set_learning(False)
            w.set_epsilon(0.2)
            w.update_hyperparams(dataclasses.replace(w.controls.hyperparams, gamma=0.5))

    out = run_until(w, lambda s: bool(applied) and s["speed"] == 0.5 and s["learning"] is False
                    and s["hp"]["gamma"] == 0.5, on_stats=on)
    assert out.stats[-1]["epsilon"] == pytest.approx(0.2)


def test_autosave_overwrites_a_single_file(tmp_path, qapp):
    out = run_until(GameWorker(cfg(tmp_path, autosave_every=50)), lambda s: s["agent_step"] >= 160)
    autos = [s for s in out.saves if s["auto"]]
    assert len(autos) >= 2 and len({s["path"] for s in autos}) == 1
    assert len(list(tmp_path.glob("*_auto.pth"))) == 1


def test_corrupt_model_emits_failed_and_finishes(tmp_path, qapp):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"junk")
    w = GameWorker(cfg(tmp_path, model_path=str(bad)))
    failed, done = [], []
    w.failed.connect(failed.append)
    w.finished.connect(lambda: done.append(1))
    w.run()
    assert failed and done


def test_natural_end_without_auto_restart_saves_unsaved_learning(tmp_path, qapp):
    w = GameWorker(cfg(tmp_path, auto_restart=False, hp=dataclasses.replace(HP, burnin=3)))
    saves = []
    w.saved.connect(saves.append)
    w.stats_ready.connect(lambda s: w.request_stop() if s["agent_step"] >= 3000 else None)
    w.run()
    assert saves and saves[-1]["auto"] is True and saves[-1]["updates"] > 0


def test_user_stop_does_not_autosave(tmp_path, qapp):
    out = run_until(GameWorker(cfg(tmp_path)), lambda s: s["agent_step"] >= 80)
    assert out.saves == []
