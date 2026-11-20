"""UI スレッドからワーカースレッドへ操作を渡す、ロック付きの共有状態。"""

from __future__ import annotations

from threading import Event, Lock

from .hyperparams import Hyperparams


class RunControls:
    def __init__(self, *, speed: float, learning: bool, hp: Hyperparams, autosave_every: int) -> None:
        self._lock = Lock()
        self._speed = speed
        self._learning = learning
        self._hp = hp
        self._autosave_every = autosave_every
        self._epsilon: float | None = None
        self._save_requested = False
        self._pause = Event()
        self.stop_event = Event()

    # ---- live values (frozen / immutable, so a plain read under the lock is atomic)
    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    def set_speed(self, value: float) -> None:
        with self._lock:
            self._speed = value

    @property
    def learning(self) -> bool:
        with self._lock:
            return self._learning

    def set_learning(self, on: bool) -> None:
        with self._lock:
            self._learning = on

    @property
    def hyperparams(self) -> Hyperparams:
        with self._lock:
            return self._hp

    def set_hyperparams(self, hp: Hyperparams) -> None:
        with self._lock:
            self._hp = hp

    @property
    def autosave_every(self) -> int:
        with self._lock:
            return self._autosave_every

    def set_autosave_every(self, n: int) -> None:
        with self._lock:
            self._autosave_every = n

    # ---- one-shot requests
    def request_epsilon(self, value: float) -> None:
        with self._lock:
            self._epsilon = value

    def take_epsilon(self) -> float | None:
        with self._lock:
            value, self._epsilon = self._epsilon, None
            return value

    def request_save(self) -> None:
        with self._lock:
            self._save_requested = True

    def take_save_request(self) -> bool:
        with self._lock:
            requested, self._save_requested = self._save_requested, False
            return requested

    # ---- pause / stop
    def request_pause(self) -> None:
        self._pause.set()

    def request_resume(self) -> None:
        self._pause.clear()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def request_stop(self) -> None:
        self.stop_event.set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()
