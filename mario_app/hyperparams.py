"""DQN ハイパーパラメータの定義・範囲・刻み・表示形式。

``Hyperparams`` は frozen dataclass。実行中の変更は ``with_value`` で新しい値を作って差し替える。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Hyperparams:
    """ノートブック ``sakai_rl_mario.ipynb`` の設定値と同じ既定値。"""

    epsilon: float = 1.0
    epsilon_decay: float = 0.9999555
    epsilon_min: float = 0.075
    gamma: float = 0.9
    learning_rate: float = 1e-4
    batch_size: int = 32
    learn_every: int = 3
    sync_every: int = 1000
    burnin: int = 1000
    buffer_capacity: int = 10_000


@dataclass(frozen=True)
class HpSpec:
    name: str
    label: str
    minimum: float
    maximum: float
    step: float
    mode: Literal["add", "mul"]
    integer: bool


# name: (label, minimum, maximum, step, mode, integer)
_RAW_SPECS: dict[str, tuple] = {
    "epsilon": ("ε", 0.0, 1.0, 0.01, "add", False),
    "epsilon_decay": ("ε減衰", 1e-6, 1.0, 1e-6, "add", False),
    "epsilon_min": ("ε下限", 0.0, 1.0, 0.005, "add", False),
    "gamma": ("γ", 0.0, 0.999, 0.01, "add", False),
    "learning_rate": ("学習率", 1e-6, 1e-2, 2.0, "mul", False),
    "batch_size": ("バッチ", 8, 256, 2.0, "mul", True),
    "learn_every": ("学習間隔", 1, 50, 1, "add", True),
    "sync_every": ("同期間隔", 10, 100_000, 100, "add", True),
    "burnin": ("burnin", 0, 100_000, 100, "add", True),
    "buffer_capacity": ("バッファ", 1_000, 100_000, 1_000, "add", True),
}

HP_SPECS: dict[str, HpSpec] = {
    name: HpSpec(name, *raw) for name, raw in _RAW_SPECS.items()
}

_DECIMALS = {"epsilon": 3, "epsilon_decay": 7, "epsilon_min": 3, "gamma": 2}


def defaults_for(new_model: bool) -> Hyperparams:
    """新規モデルは ε=1.0、既存モデルは ε=0（今までの再生と同じ）。"""
    return Hyperparams(epsilon=1.0 if new_model else 0.0)


def clamp_hp(name: str, value: float) -> float | int:
    spec = HP_SPECS[name]
    clamped = min(spec.maximum, max(spec.minimum, float(value)))
    if spec.integer:
        return int(min(spec.maximum, max(spec.minimum, round(clamped))))
    return clamped


def step_value(name: str, value: float, direction: int) -> float | int:
    spec = HP_SPECS[name]
    sign = 1 if direction >= 0 else -1
    if spec.mode == "mul":
        new = value * spec.step if sign > 0 else value / spec.step
    else:
        new = round(value + sign * spec.step, 10)
    return clamp_hp(name, new)


def with_value(hp: Hyperparams, name: str, value: float) -> Hyperparams:
    return dataclasses.replace(hp, **{name: clamp_hp(name, value)})


def format_hp(name: str, value: float) -> str:
    spec = HP_SPECS[name]
    if spec.integer:
        return str(int(value))
    if name == "learning_rate":
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"{float(mantissa):g}e{int(exponent)}"
    return f"{value:.{_DECIMALS[name]}f}"


def parse_hp(name: str, text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def hp_to_dict(hp: Hyperparams) -> dict[str, float | int]:
    return dataclasses.asdict(hp)
