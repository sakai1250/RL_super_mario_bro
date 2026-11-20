import json
from datetime import datetime

import torch

from mario_app.agent import MarioNet, load_weights
from mario_app.checkpoint import (
    auto_path,
    build_meta,
    find_models,
    manual_path,
    save_checkpoint,
)
from mario_app.hyperparams import Hyperparams

NOW = datetime(2026, 11, 20, 0, 58, 12)


def test_manual_path_name_and_no_collision(tmp_path):
    p1 = manual_path(tmp_path, 4800, NOW)
    assert p1.name == "mario_20261120-005812_s4800.pth"
    p1.touch()
    p2 = manual_path(tmp_path, 4800, NOW)
    assert p2 != p1 and p2.name.startswith("mario_20261120-005812_s4800") and p2.suffix == ".pth"


def test_auto_path_is_stable_per_run(tmp_path):
    assert auto_path(tmp_path, NOW).name == "mario_20261120-005812_auto.pth" == auto_path(tmp_path, NOW).name


def test_save_checkpoint_roundtrip_and_json(tmp_path):
    net = MarioNet()
    meta = build_meta(hp=Hyperparams(), epsilon=0.4, agent_step=10, episode=2, updates=0, seed=15,
                      stage="1-1", device="cpu", source="new", auto=False, saved_at=NOW)
    path = save_checkpoint(net, tmp_path / "models" / "a.pth", meta)
    state = torch.load(path, weights_only=True)
    assert {k.split(".")[0] for k in state} == {"online", "target"}
    assert all(t.device.type == "cpu" for t in state.values())
    dst = MarioNet()
    load_weights(dst, path)
    assert all(torch.equal(dst.state_dict()[k], v) for k, v in net.state_dict().items())
    loaded = json.loads(path.with_suffix(".json").read_text())
    assert {"hyperparams", "epsilon", "agent_step", "episode", "updates", "seed", "stage",
            "device", "source", "auto", "saved_at", "torch_version"} <= set(loaded)
    assert loaded["updates"] == 0 and loaded["hyperparams"]["gamma"] == 0.9


def test_save_checkpoint_leaves_no_temp_files(tmp_path):
    meta = build_meta(hp=Hyperparams(), epsilon=0.4, agent_step=1, episode=1, updates=0, seed=15,
                      stage="1-1", device="cpu", source="new", auto=True, saved_at=NOW)
    save_checkpoint(MarioNet(), tmp_path / "a.pth", meta)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "a.pth"]


def test_find_models_orders_preferred_first(tmp_path):
    for n in ["zeta.pth", "mario_param_new.pth", "Alpha.pth", "mario_param_new_best.pth", "note.txt"]:
        (tmp_path / n).touch()
    assert [p.name for p in find_models([tmp_path])] == [
        "mario_param_new_best.pth", "mario_param_new.pth", "Alpha.pth", "zeta.pth"]


def test_find_models_ignores_missing_root(tmp_path):
    assert find_models([tmp_path / "nope"]) == []
