"""Q ネットワーク（ノートブックと同一）とチェックポイントの読み込み。

ノートブック ``sakai_rl_mario.ipynb`` の ``MarioNet`` と同一アーキテクチャ。
チェックポイントは ``MarioNet.state_dict()``（online / target 両方を含む）を想定するが、
online のみ・``module.`` プレフィックス付き・``state_dict`` ラップ付きにも対応する。
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


class MarioNet(nn.Module):
    """DQN の online / target ネットワーク（ノートブックと同一）。"""

    def __init__(self, input_dim: tuple[int, int, int] = (4, 84, 84), output_dim: int = 7):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"入力は 84x84 である必要があります: {h}x{w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, model: str = "online") -> torch.Tensor:
        if model == "online":
            return self.online(x)
        if model == "target":
            return self.target(x)
        raise ValueError(f"未知のモデル指定: {model!r}")


def _clean_state_dict(raw: dict) -> dict[str, torch.Tensor]:
    """チェックポイントのキーゆれを吸収する。"""
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        key = key.removeprefix("module.")
        cleaned[key] = value

    if not any(k.startswith("online.") for k in cleaned):
        # MarioNet ではなく online の Sequential だけ保存されていた場合
        cleaned = {f"online.{k}": v for k, v in cleaned.items()}
    return cleaned


def load_weights(net: MarioNet, path: str | Path, device: str | torch.device = "cpu") -> Path:
    """チェックポイントを ``net`` に読み込む。壊れた・非対応のファイルは例外にする。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {path}")

    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # 破損・非対応形式
        raise ValueError(f"モデルを読み込めませんでした: {path.name}\n{exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"未対応のチェックポイント形式です: {type(raw).__name__}")

    state = _clean_state_dict(raw)
    expected = net.state_dict()
    missing_online = sorted(k for k in expected if k.startswith("online.") and k not in state)
    if missing_online:
        raise ValueError(
            "チェックポイントの構造がこのアプリのネットワークと一致しません。\n"
            f"不足キー: {', '.join(missing_online[:3])} ..."
        )

    net.load_state_dict({k: v for k, v in state.items() if k in expected}, strict=False)
    if not any(k.startswith("target.") for k in state):
        net.target.load_state_dict(net.online.state_dict())
    net.eval()
    return path
