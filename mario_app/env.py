"""Super Mario Bros. 環境のラッパー群。

ノートブックの前処理（グレースケール → 84x84 → 4 フレームスタック）と
カスタム報酬を、Gymnasium API 向けに移植したもの。
アクション繰り返し（skip=12）は描画の都合でワーカー側が担当する。
"""

from __future__ import annotations

from collections import deque

import gym_super_mario_bros
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium import spaces
from nes_py.wrappers import JoypadSpace

STAGES: list[str] = [f"{world}-{stage}" for world in range(1, 9) for stage in range(1, 5)]

ACTION_LABELS = [
    "NOOP",
    "right",
    "right + A",
    "right + B",
    "right + A + B",
    "A",
    "left",
]

# エージェント 1 ステップあたりに進める NES フレーム数（ノートブックと同じ 12）
ACTION_REPEAT = 12


class GrayScaleObservation(gym.ObservationWrapper):
    """RGB (H, W, 3) uint8 → グレースケール float (1, H, W)。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(1, h, w), dtype=np.float32)

    def observation(self, observation: np.ndarray) -> torch.Tensor:
        # HWC uint8 → CHW float（ノートブックと同じ順序・重み）
        x = torch.as_tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float32)
        gray = 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3]
        return gray


class ResizeObservation(gym.ObservationWrapper):
    """(1, 240, 256) → (84, 84) に縮小し 0–1 に正規化。"""

    def __init__(self, env: gym.Env, shape: int = 84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, observation: torch.Tensor) -> np.ndarray:
        x = observation.unsqueeze(0)
        # 学習時（torchvision 0.13 の T.Resize）と同じく antialias なしで縮小する。
        # antialias=True にすると入力分布が変わり、学習済みモデルの性能が落ちる。
        x = F.interpolate(
            x,
            size=self.shape,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        x = x.squeeze(0) / 255.0
        return x.numpy().astype(np.float32)


class FrameStack(gym.Wrapper):
    """直近 k フレームを (k, H, W) に積む軽量実装。"""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames: deque[np.ndarray] = deque(maxlen=k)
        base_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(k, *base_shape),
            dtype=np.float32,
        )

    def _stack(self) -> np.ndarray:
        return np.stack(self.frames).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stack(), reward, terminated, truncated, info


class RewardShaper:
    """ノートブックの ``CustomRewardAndDoneEnv`` の報酬計算を移植。

    - 前進 +10 / 停止 -2 / 後退 -4
    - 最長距離更新 +10、ゴール +100
    - 残機減少ペナルティ -30 * (life - 1)
    - 最後に 0.2 倍

    エピソード開始時に ``reset()`` を呼ぶ（最長距離もリセット）。
    """

    def __init__(self) -> None:
        self.cur_x = 0.0
        self.max_x = 0.0

    def reset(self) -> None:
        self.cur_x = 0.0
        self.max_x = 0.0

    def shape(self, total_reward: float, info: dict) -> float:
        x_pos = float(info.get("x_pos", 0.0))

        if info.get("flag_get"):
            total_reward += 100.0

        if x_pos >= self.max_x:
            self.max_x = x_pos
            total_reward += 10.0

        if (x_pos < self.max_x) & (x_pos - self.max_x > -100):
            total_reward += 5.0

        if x_pos > self.cur_x:
            total_reward += 10.0
        elif x_pos == self.cur_x:
            total_reward -= 2.0
        else:
            total_reward -= 4.0

        self.cur_x = x_pos
        total_reward -= 30.0 * (float(info.get("life", 1)) - 1.0)
        return total_reward * 0.2


def make_env(stage: str = "3-4", render_mode: str = "rgb_array") -> gym.Env:
    """学習時と同じ前処理を施した環境を作る（skip ラッパーは含まない）。"""
    if stage not in STAGES:
        raise ValueError(f"不明なステージです: {stage}")

    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{stage}-v0",
        render_mode=render_mode,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, k=4)
    return env


def grab_screen(env: gym.Env) -> np.ndarray:
    """エミュレータの現在画面を RGB uint8 (240, 256, 3) でコピーして返す。"""
    frame = env.render()
    if frame is None:
        frame = env.unwrapped.screen
    return np.array(frame, dtype=np.uint8, copy=True)
