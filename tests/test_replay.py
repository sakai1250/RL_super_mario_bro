import numpy as np
import pytest

from mario_app.replay import ReplayBuffer


def _t(i):
    return np.full((4, 84, 84), (i % 10) / 10, np.float32)


def test_roundtrip_within_quantization_error():
    rng = np.random.default_rng(0)
    s = rng.random((4, 84, 84), dtype=np.float32)
    ns = rng.random((4, 84, 84), dtype=np.float32)
    buf = ReplayBuffer(10)
    buf.add(s, ns, 3, 1.5, False)
    bs, bns, a, r, d = buf.sample(1, rng)
    assert bs.shape == (1, 4, 84, 84) and bs.dtype == np.float32
    assert np.abs(bs[0] - s).max() <= 0.5 / 255 + 1e-6
    assert np.abs(bns[0] - ns).max() <= 0.5 / 255 + 1e-6
    assert a.dtype == np.int64 and a[0] == 3 and r[0] == 1.5 and d[0] == 0.0


def test_overwrites_oldest_when_full():
    buf = ReplayBuffer(3)
    for i in range(5):
        buf.add(_t(i), _t(i), i, 0.0, False)
    assert len(buf) == 3
    assert sorted(buf.sample(3, np.random.default_rng(0))[2].tolist()) == [2, 3, 4]


def test_resize_shrink_keeps_newest():
    buf = ReplayBuffer(5)
    for i in range(5):
        buf.add(_t(i), _t(i), i, 0.0, False)
    buf.resize(2)
    assert buf.capacity == 2 and len(buf) == 2
    assert sorted(buf.sample(2, np.random.default_rng(0))[2].tolist()) == [3, 4]


def test_resize_grow_keeps_all_then_accepts_more():
    buf = ReplayBuffer(3)
    for i in range(5):
        buf.add(_t(i), _t(i), i, 0.0, False)
    buf.resize(6)
    assert len(buf) == 3
    buf.add(_t(9), _t(9), 9, 0.0, True)
    assert len(buf) == 4
    assert sorted(buf.sample(4, np.random.default_rng(0))[2].tolist()) == [2, 3, 4, 9]


def test_sample_larger_than_len_raises():
    buf = ReplayBuffer(5)
    buf.add(_t(0), _t(0), 0, 0.0, False)
    with pytest.raises(ValueError):
        buf.sample(2, np.random.default_rng(0))


def test_bytes_per_transition_constant():
    assert ReplayBuffer.BYTES_PER_TRANSITION == 2 * 4 * 84 * 84
