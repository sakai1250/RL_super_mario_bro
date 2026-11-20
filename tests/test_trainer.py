import dataclasses
import math

import numpy as np
import pytest
import torch

from mario_app.agent import MarioNet
from mario_app.hyperparams import Hyperparams
from mario_app.trainer import DQNTrainer, resolve_device


def make_trainer(**hp_overrides):
    base = dict(batch_size=8, burnin=0, learn_every=1, sync_every=1000, buffer_capacity=200)
    base.update(hp_overrides)
    return DQNTrainer(MarioNet(), Hyperparams(**base), torch.device("cpu"), seed=1)


def fill(trainer, n):
    rng = np.random.default_rng(0)
    for i in range(n):
        s = rng.random((4, 84, 84), dtype=np.float32)
        trainer.cache(s, s, i % 7, 1.0, False)


def test_act_decays_epsilon_only_when_learning():
    t = make_trainer(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    s = np.zeros((4, 84, 84), np.float32)
    t.act(s, learning=True)
    assert t.epsilon == pytest.approx(0.5)
    t.act(s, learning=False)
    assert t.epsilon == pytest.approx(0.5)
    for _ in range(10):
        t.act(s, learning=True)
    assert t.epsilon == pytest.approx(0.1)


def test_act_explores_at_epsilon_one_and_exploits_at_zero():
    t = make_trainer(epsilon=1.0)
    s = np.random.rand(4, 84, 84).astype(np.float32)
    action, q = t.act(s, learning=False)
    assert q is None and 0 <= action < 7
    t.set_epsilon(0.0)
    action, q = t.act(s, learning=False)
    assert q.shape == (7,) and action == int(np.argmax(q))


def test_curr_step_counts_acts():
    t = make_trainer()
    s = np.zeros((4, 84, 84), np.float32)
    for _ in range(3):
        t.act(s, learning=False)
    assert t.curr_step == 3


def test_learn_skips_before_burnin_and_off_cycle():
    t = make_trainer(burnin=5, learn_every=2)
    fill(t, 40)
    t.curr_step = 4
    assert t.learn() is None
    t.curr_step = 7
    assert t.learn() is None
    t.curr_step = 8
    assert t.learn() is not None


def test_learn_skips_when_buffer_smaller_than_batch():
    t = make_trainer(batch_size=32)
    fill(t, 10)
    t.curr_step = 1000
    assert t.learn() is None
    t.set_hyperparams(dataclasses.replace(t.hp, buffer_capacity=1000, batch_size=64))
    fill(t, 20)
    t.curr_step = 1001
    assert t.learn() is None


def test_sync_target_runs_before_the_burnin_check():
    t = make_trainer(burnin=100_000, sync_every=10)
    with torch.no_grad():
        t.net.online[0].weight.add_(1.0)
    t.curr_step = 10
    assert t.learn() is None
    assert torch.equal(t.net.target[0].weight, t.net.online[0].weight)


def test_td_target_matches_manual_double_dqn():
    t = make_trainer(gamma=0.9)
    with torch.no_grad():
        t.net.target[0].weight.mul_(1.5)
    ns = torch.rand(8, 4, 84, 84)
    r = torch.rand(8)
    d = torch.tensor([0., 1., 0., 0., 1., 0., 0., 0.])
    best = t.net(ns, "online").argmax(1)
    q = t.net(ns, "target")[torch.arange(8), best]
    assert torch.allclose(t.td_target(r, ns, d), r + (1 - d) * 0.9 * q)


def test_update_changes_online_but_not_target():
    t = make_trainer()
    fill(t, 40)
    t.curr_step = 1
    before_o = t.net.online[0].weight.clone()
    before_t = t.net.target[0].weight.clone()
    result = t.learn()
    assert result is not None and math.isfinite(result.loss) and t.updates == 1
    assert not torch.equal(t.net.online[0].weight, before_o)
    assert torch.equal(t.net.target[0].weight, before_t)


def test_gamma_zero_makes_target_equal_reward():
    t = make_trainer(gamma=0.0)
    r = torch.rand(4)
    assert torch.allclose(t.td_target(r, torch.rand(4, 4, 84, 84), torch.zeros(4)), r)


def test_set_hyperparams_applies_lr_and_resizes_buffer():
    t = make_trainer()
    fill(t, 50)
    t.set_hyperparams(dataclasses.replace(t.hp, learning_rate=5e-4, buffer_capacity=1000))
    assert all(g["lr"] == 5e-4 for g in t.optimizer.param_groups)
    assert t.buffer.capacity == 1000 and len(t.buffer) == 50


def test_set_epsilon_clamps():
    t = make_trainer()
    t.set_epsilon(1.7)
    assert t.epsilon == 1.0


def test_resolve_device_falls_back_to_cpu():
    assert resolve_device("mps", mps_available=False).type == "cpu"
    assert resolve_device("auto", mps_available=False).type == "cpu"
    assert resolve_device("auto", mps_available=True).type == "mps"
    assert resolve_device("cpu", mps_available=True).type == "cpu"
