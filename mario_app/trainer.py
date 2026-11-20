"""ノートブックの ``Mario`` クラスを移植した DQN 学習器（Double DQN 形式の TD ターゲット）。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .agent import MarioNet
from .hyperparams import Hyperparams, clamp_hp
from .replay import ReplayBuffer

ACTION_DIM = 7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str, mps_available: bool | None = None) -> torch.device:
    """``auto`` / ``mps`` は MPS が使えれば MPS、無ければ CPU。"""
    if requested == "cpu":
        return torch.device("cpu")
    if mps_available is None:
        mps_available = torch.backends.mps.is_available()
    return torch.device("mps" if mps_available else "cpu")


@dataclass(frozen=True)
class LearnResult:
    loss: float
    mean_q: float


class DQNTrainer:
    def __init__(
        self,
        net: MarioNet,
        hp: Hyperparams,
        device: torch.device,
        seed: int = 15,
    ) -> None:
        self.device = device
        self.net = net.float().to(device)
        self._hp = hp
        self.epsilon = float(clamp_hp("epsilon", hp.epsilon))
        self.curr_step = 0
        self.updates = 0
        self._rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(hp.buffer_capacity)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=hp.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.net.eval()

    # --------------------------------------------------------------- settings
    @property
    def hp(self) -> Hyperparams:
        return self._hp

    def set_hyperparams(self, hp: Hyperparams) -> None:
        """``epsilon`` 以外を反映する（ε は ``set_epsilon`` と ``act`` の減衰だけが動かす）。"""
        if hp == self._hp:
            return
        if hp.learning_rate != self._hp.learning_rate:
            for group in self.optimizer.param_groups:
                group["lr"] = hp.learning_rate
        if hp.buffer_capacity != self._hp.buffer_capacity:
            self.buffer.resize(hp.buffer_capacity)
        self._hp = hp

    def set_epsilon(self, value: float) -> None:
        self.epsilon = float(clamp_hp("epsilon", value))

    # ------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, state: np.ndarray, learning: bool) -> tuple[int, np.ndarray | None]:
        if self._rng.random() < self.epsilon:
            action, q_values = int(self._rng.integers(ACTION_DIM)), None
        else:
            x = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
            q_values = self.net(x, "online")[0].cpu().numpy().astype(np.float32)
            action = int(np.argmax(q_values))

        if learning:
            self.epsilon = max(self._hp.epsilon_min, self.epsilon * self._hp.epsilon_decay)
        self.curr_step += 1
        return action, q_values

    # ----------------------------------------------------------------- learn
    def cache(self, state, next_state, action: int, reward: float, done: bool) -> None:
        self.buffer.add(state, next_state, action, reward, done)

    @torch.no_grad()
    def td_target(
        self, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        best = self.net(next_state, "online").argmax(dim=1)
        next_q = self.net(next_state, "target")[torch.arange(len(best), device=best.device), best]
        return (reward + (1 - done.float()) * self._hp.gamma * next_q).float()

    def sync_target(self) -> None:
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self) -> LearnResult | None:
        hp = self._hp
        if self.curr_step % hp.sync_every == 0:
            self.sync_target()
        if self.curr_step < hp.burnin:
            return None
        if self.curr_step % hp.learn_every != 0:
            return None
        if len(self.buffer) < hp.batch_size:
            return None

        s, ns, a, r, d = self.buffer.sample(hp.batch_size, self._rng)
        dev = self.device
        s_t = torch.as_tensor(s, device=dev)
        ns_t = torch.as_tensor(ns, device=dev)
        a_t = torch.as_tensor(a, device=dev)
        r_t = torch.as_tensor(r, device=dev)
        d_t = torch.as_tensor(d, device=dev)

        target = self.td_target(r_t, ns_t, d_t)
        with torch.enable_grad():
            q_all = self.net(s_t, "online")
            estimate = q_all[torch.arange(hp.batch_size, device=dev), a_t]
            loss = self.loss_fn(estimate, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.updates += 1
        return LearnResult(loss=float(loss.item()), mean_q=float(estimate.detach().mean().item()))

