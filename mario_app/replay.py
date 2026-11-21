"""uint8 で持つ環状リプレイバッファ（1 遷移 56,448 B）。"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    BYTES_PER_TRANSITION = 2 * 4 * 84 * 84

    def __init__(self, capacity: int, state_shape: tuple[int, int, int] = (4, 84, 84)) -> None:
        if capacity < 1:
            raise ValueError("capacity は 1 以上が必要です")
        self._shape = state_shape
        self._allocate(capacity)
        self._size = 0
        self._pos = 0

    # ------------------------------------------------------------- storage
    def _allocate(self, capacity: int) -> None:
        self._capacity = capacity
        self._states = np.zeros((capacity, *self._shape), dtype=np.uint8)
        self._next_states = np.zeros((capacity, *self._shape), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    @staticmethod
    def _quantize(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        return np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    # ----------------------------------------------------------------- api
    def add(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        i = self._pos
        self._states[i] = self._quantize(state)
        self._next_states[i] = self._quantize(next_state)
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = 1.0 if done else 0.0
        self._pos = (i + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        if batch_size > self._size:
            raise ValueError(f"バッファ件数 {self._size} がバッチ {batch_size} 未満です")
        idx = rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[idx].astype(np.float32) / 255.0,
            self._next_states[idx].astype(np.float32) / 255.0,
            self._actions[idx].copy(),
            self._rewards[idx].copy(),
            self._dones[idx].copy(),
        )

    def resize(self, new_capacity: int) -> None:
        """容量を変える。縮小時は新しい遷移を残す。"""
        if new_capacity < 1:
            raise ValueError("capacity は 1 以上が必要です")
        if new_capacity == self._capacity:
            return
        # 古い順に並べた添字
        if self._size < self._capacity:
            order = np.arange(self._size)
        else:
            order = (np.arange(self._size) + self._pos) % self._capacity
        keep = order[-min(self._size, new_capacity):]
        old = (self._states, self._next_states, self._actions, self._rewards, self._dones)
        self._allocate(new_capacity)
        n = len(keep)
        self._states[:n] = old[0][keep]
        self._next_states[:n] = old[1][keep]
        self._actions[:n] = old[2][keep]
        self._rewards[:n] = old[3][keep]
        self._dones[:n] = old[4][keep]
        self._size = n
        self._pos = n % new_capacity
