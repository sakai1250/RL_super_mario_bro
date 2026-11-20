"""再生速度に応じたペーシング（毎フレーム、現在の速度から再計算する）。"""

from __future__ import annotations

import math
import time
from typing import Callable

BASE_FPS = 60.0
SPEED_MAX = math.inf

SPEEDS: tuple[tuple[str, float], ...] = (
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1×", 1.0),
    ("2×", 2.0),
    ("4×", 4.0),
    ("MAX", SPEED_MAX),
)


def frame_period(speed: float) -> float:
    if speed == SPEED_MAX:
        return 0.0
    return 1.0 / (BASE_FPS * max(0.05, speed))


def episode_pause(speed: float) -> float:
    """エピソード終了後に死亡・ゴール画面を見せる秒数。"""
    if speed == SPEED_MAX:
        return 0.0
    return 1.1 / speed


class DisplayThrottle:
    """画面更新の間引き。有限の速度は N フレームごと、MAX は壁時計で上限 fps。"""

    def __init__(self, max_fps: float = 30.0, clock: Callable[[], float] = time.perf_counter) -> None:
        self._interval = 1.0 / max_fps
        self._clock = clock
        self._last = -math.inf

    def should_emit(self, speed: float, frame_count: int) -> bool:
        if speed == SPEED_MAX:
            now = self._clock()
            if now - self._last >= self._interval:
                self._last = now
                return True
            return False
        return frame_count % max(1, round(speed)) == 0
