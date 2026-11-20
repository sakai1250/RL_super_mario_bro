import pytest
import torch

from mario_app.agent import MarioNet, load_weights


def test_load_weights_roundtrip(tmp_path):
    src = MarioNet()
    torch.save(src.state_dict(), tmp_path / "m.pth")
    dst = MarioNet()
    load_weights(dst, tmp_path / "m.pth")
    assert all(torch.equal(dst.state_dict()[k], v) for k, v in src.state_dict().items())


def test_load_weights_accepts_online_only_and_module_prefix(tmp_path):
    src = MarioNet()
    torch.save({f"module.online.{k}": v for k, v in src.online.state_dict().items()}, tmp_path / "m.pth")
    dst = MarioNet()
    load_weights(dst, tmp_path / "m.pth")
    assert torch.equal(dst.target.state_dict()["0.weight"], src.online.state_dict()["0.weight"])


def test_load_weights_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_weights(MarioNet(), tmp_path / "none.pth")


def test_load_weights_rejects_garbage_file(tmp_path):
    (tmp_path / "bad.pth").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        load_weights(MarioNet(), tmp_path / "bad.pth")
