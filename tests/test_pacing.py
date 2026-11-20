import pytest

from mario_app.pacing import SPEED_MAX, SPEEDS, DisplayThrottle, episode_pause, frame_period


def test_frame_period():
    assert frame_period(1.0) == pytest.approx(1 / 60)
    assert frame_period(4.0) == pytest.approx(1 / 240)
    assert frame_period(SPEED_MAX) == 0.0


def test_episode_pause():
    assert episode_pause(1.0) == pytest.approx(1.1)
    assert episode_pause(2.0) == pytest.approx(0.55)
    assert episode_pause(SPEED_MAX) == 0.0


def test_throttle_finite_speed_uses_frame_count():
    t = DisplayThrottle()
    assert [t.should_emit(4.0, n) for n in range(1, 9)] == [False, False, False, True, False, False, False, True]
    assert all(t.should_emit(0.25, n) for n in range(1, 4))


def test_throttle_max_uses_clock():
    now = [0.0]
    t = DisplayThrottle(30.0, clock=lambda: now[0])
    assert t.should_emit(SPEED_MAX, 1) is True
    assert t.should_emit(SPEED_MAX, 2) is False
    now[0] += 0.04
    assert t.should_emit(SPEED_MAX, 3) is True


def test_speeds_end_with_max():
    assert SPEEDS[-1] == ("MAX", SPEED_MAX)
    assert [v for _, v in SPEEDS[:5]] == [0.25, 0.5, 1.0, 2.0, 4.0]
