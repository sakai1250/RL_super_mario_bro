import math

from mario_app.agent import MarioNet, load_weights
from mario_app.game_worker import GameWorker, RunConfig
from mario_app.hyperparams import Hyperparams
from mario_app.pacing import SPEED_MAX


def test_parallel_workers_share_one_learner_and_save_loadable(tmp_path, qapp):
    hp = Hyperparams(batch_size=8, burnin=0, learn_every=1, sync_every=50,
                     buffer_capacity=1000, epsilon=1.0, epsilon_decay=0.99)
    worker = GameWorker(RunConfig(
        model_path=None, stage="1-1", hp=hp, learning=True, speed=SPEED_MAX,
        device="cpu", autosave_every=0, models_dir=str(tmp_path), n_envs=2))
    stats, saves, failures = [], [], []
    worker.stats_ready.connect(stats.append)
    worker.saved.connect(saves.append)
    worker.failed.connect(failures.append)

    def on_stats(s):
        if s["agent_step"] >= 80:
            worker.request_save()
            worker.request_stop()

    worker.stats_ready.connect(on_stats)
    worker.run()

    assert not failures
    last = stats[-1]
    assert last["envs"] == 2 and last["agent_step"] >= 80
    assert last["updates"] > 0 and math.isfinite(last["loss"])
    assert last["steps_per_sec"] > 0
    assert saves
    load_weights(MarioNet(), saves[0]["path"])
