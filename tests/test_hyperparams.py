import dataclasses

import pytest

from mario_app.hyperparams import (
    HP_SPECS,
    Hyperparams,
    clamp_hp,
    defaults_for,
    format_hp,
    hp_to_dict,
    parse_hp,
    step_value,
    with_value,
)


def test_defaults_match_notebook():
    hp = Hyperparams()
    assert (hp.epsilon, hp.epsilon_decay, hp.epsilon_min, hp.gamma, hp.learning_rate,
            hp.batch_size, hp.learn_every, hp.sync_every, hp.burnin, hp.buffer_capacity) == \
           (1.0, 0.9999555, 0.075, 0.9, 1e-4, 32, 3, 1000, 1000, 10_000)


def test_defaults_for_existing_model_has_zero_epsilon():
    assert defaults_for(new_model=False).epsilon == 0.0
    assert defaults_for(new_model=True).epsilon == 1.0


@pytest.mark.parametrize("name,raw,expected", [
    ("gamma", 1.5, 0.999), ("epsilon", -0.2, 0.0), ("learning_rate", 1.0, 1e-2),
    ("batch_size", 3, 8), ("buffer_capacity", 10, 1000), ("learn_every", 2.6, 3)])
def test_clamp_hp(name, raw, expected):
    assert clamp_hp(name, raw) == expected


def test_step_value_add_and_mul():
    assert step_value("learning_rate", 1e-4, +1) == pytest.approx(2e-4)
    assert step_value("batch_size", 32, +1) == 64
    assert step_value("batch_size", 8, -1) == 8
    assert step_value("sync_every", 1000, +1) == 1100
    assert step_value("epsilon", 0.995, +1) == 1.0
    assert step_value("epsilon_decay", 0.9999555, +1) == pytest.approx(0.9999565)


def test_with_value_returns_new_instance():
    hp = Hyperparams()
    new = with_value(hp, "gamma", 0.95)
    assert new.gamma == 0.95 and hp.gamma == 0.9 and new is not hp


def test_hyperparams_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Hyperparams().gamma = 0.5


def test_format_hp():
    assert [format_hp("learning_rate", 1e-4), format_hp("learning_rate", 2e-4),
            format_hp("epsilon_decay", 0.9999555), format_hp("epsilon", 0.412),
            format_hp("gamma", 0.9), format_hp("batch_size", 32),
            format_hp("buffer_capacity", 10000)] == \
           ["1e-4", "2e-4", "0.9999555", "0.412", "0.90", "32", "10000"]


def test_parse_hp():
    assert parse_hp("learning_rate", "1e-4") == pytest.approx(1e-4)
    assert parse_hp("buffer_capacity", "10,000") == 10000
    assert parse_hp("gamma", "abc") is None and parse_hp("gamma", "") is None


def test_specs_follow_field_order_and_dict():
    assert list(HP_SPECS) == [f.name for f in dataclasses.fields(Hyperparams)]
    assert hp_to_dict(Hyperparams())["gamma"] == 0.9
