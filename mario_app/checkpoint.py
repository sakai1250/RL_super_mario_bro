"""重みの保存（ノートブックと同じ形式）と、横置きの JSON。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

from .agent import MarioNet
from .hyperparams import Hyperparams, hp_to_dict

PREFERRED_MODELS = ("mario_param_new_best.pth", "mario_param_new.pth")


def manual_path(directory: Path, agent_step: int, now: datetime) -> Path:
    stem = f"mario_{now:%Y%m%d-%H%M%S}_s{agent_step}"
    path = directory / f"{stem}.pth"
    n = 2
    while path.exists():
        path = directory / f"{stem}_{n}.pth"
        n += 1
    return path


def auto_path(directory: Path, run_started: datetime) -> Path:
    return directory / f"mario_{run_started:%Y%m%d-%H%M%S}_auto.pth"


def build_meta(
    *,
    hp: Hyperparams,
    epsilon: float,
    agent_step: int,
    episode: int,
    updates: int,
    seed: int,
    stage: str,
    device: str,
    source: str,
    auto: bool,
    saved_at: datetime,
) -> dict:
    return {
        "hyperparams": hp_to_dict(hp),
        "epsilon": epsilon,
        "agent_step": agent_step,
        "episode": episode,
        "updates": updates,
        "seed": seed,
        "stage": stage,
        "device": device,
        "source": source,
        "auto": auto,
        "saved_at": saved_at.isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }


def _replace_atomically(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(net: MarioNet, path: Path, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    _replace_atomically(path, lambda tmp: torch.save(state, tmp))
    _replace_atomically(
        path.with_suffix(".json"),
        lambda tmp: tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"),
    )
    return path


def find_models(roots: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if Path(root).is_dir():
            found.extend(Path(root).glob("*.pth"))

    def key(p: Path):
        rank = PREFERRED_MODELS.index(p.name) if p.name in PREFERRED_MODELS else len(PREFERRED_MODELS)
        return (rank, p.name.lower())

    return sorted(found, key=key)
