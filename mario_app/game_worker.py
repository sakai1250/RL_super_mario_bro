"""バックグラウンドでゲームを進め、必要なら DQN 学習も行うワーカー（QThread 上で実行）。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, Slot

from .agent import MarioNet, load_weights
from .checkpoint import auto_path, build_meta, manual_path, save_checkpoint
from .controls import RunControls
from .env import ACTION_LABELS, ACTION_REPEAT, RewardShaper, grab_screen, make_env
from .hyperparams import Hyperparams, hp_to_dict
from .pacing import DisplayThrottle, episode_pause, frame_period
from .paths import MODELS_DIR
from .trainer import DQNTrainer, resolve_device, seed_everything

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    model_path: str | None  # None = 新規（ランダム初期化）
    stage: str = "3-4"
    hp: Hyperparams = field(default_factory=Hyperparams)
    learning: bool = True
    speed: float = 1.0
    auto_restart: bool = True
    seed: int = 15
    device: str = "auto"
    autosave_every: int = 5000
    models_dir: str = str(MODELS_DIR)
    n_envs: int = 8


class GameWorker(QObject):
    """ゲームループを実行し、フレームと統計をシグナルで送る。"""

    stats_ready = Signal(object)          # dict
    episode_finished = Signal(object)     # dict
    state_changed = Signal(str)           # loading / running / paused / finished / error
    saved = Signal(object)                # dict: path, auto, agent_step, updates
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.config = config
        self.controls = RunControls(
            speed=config.speed,
            learning=config.learning,
            hp=config.hp,
            autosave_every=config.autosave_every,
        )
        self._frame_count = 0
        self._frame_lock = Lock()
        self._latest_frame = None
        self._frame_seq = 0
        self._throttle = DisplayThrottle()

    # ------------------------------------------------------------- controls
    def set_speed(self, value: float) -> None:
        self.controls.set_speed(value)

    def set_learning(self, on: bool) -> None:
        self.controls.set_learning(on)

    def set_epsilon(self, value: float) -> None:
        self.controls.request_epsilon(value)

    def update_hyperparams(self, hp: Hyperparams) -> None:
        self.controls.set_hyperparams(hp)

    def set_autosave_every(self, n: int) -> None:
        self.controls.set_autosave_every(n)

    def request_save(self) -> None:
        self.controls.request_save()

    def request_stop(self) -> None:
        self.controls.request_stop()

    def request_pause(self) -> None:
        self.controls.request_pause()

    def request_resume(self) -> None:
        self.controls.request_resume()

    def is_paused(self) -> bool:
        return self.controls.is_paused()

    # ------------------------------------------------------------- internals
    def _sleep(self, seconds: float) -> bool:
        """停止要求に反応できる sleep。停止されたら False。"""
        if seconds <= 0:
            return not self.controls.is_stopped()
        return not self.controls.stop_event.wait(seconds)

    def _emit_frame(self, env) -> None:
        self._frame_count += 1
        if self._throttle.should_emit(self.controls.speed, self._frame_count):
            frame = grab_screen(env)
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_seq += 1

    def latest_frame(self):
        """UI から最新フレームを取得する（コピー済みの ndarray）。"""
        with self._frame_lock:
            return self._frame_seq, self._latest_frame

    def _store_frame(self, frame) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_seq += 1

    # ------------------------------------------------------------- parallel run
    def _run_parallel(self) -> None:
        """複数プロセスの環境から経験を集め、1 つのネットワークでまとめて更新する。"""
        procs: list[mp.Process] = []
        control_qs: list = []
        stop_event = None
        pause_event = None
        trainer: DQNTrainer | None = None
        save_now = None
        saved_step = 0
        saved_updates = 0
        ended_unattended = False
        try:
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
            self.state_changed.emit("loading")
            cfg = self.config
            controls = self.controls
            n_envs = max(1, int(cfg.n_envs))

            seed_everything(cfg.seed)
            device = resolve_device(cfg.device)
            net = MarioNet((4, 84, 84), len(ACTION_LABELS))
            if cfg.model_path is not None:
                load_weights(net, cfg.model_path, "cpu")
                source = Path(cfg.model_path).name
            else:
                source = "new"
            model_name = source
            trainer = DQNTrainer(net, controls.hyperparams, device, cfg.seed)

            run_started = datetime.now()
            models_dir = Path(cfg.models_dir)
            last_saved: str | None = None
            losses: deque[float] = deque(maxlen=100)
            qs: deque[float] = deque(maxlen=100)
            history: deque[dict] = deque(maxlen=300)
            actor_stats = [
                {"ep_reward": 0.0, "ep_steps": 0, "x_pos": 0.0, "life": 2,
                 "action": 0, "q": None}
                for _ in range(n_envs)
            ]

            completed = 0
            best_reward: float | None = None
            best_x = 0.0
            total_frames = 0
            learning = controls.learning

            def save(auto: bool) -> None:
                nonlocal saved_step, saved_updates, last_saved
                now = datetime.now()
                path = (
                    auto_path(models_dir, run_started)
                    if auto
                    else manual_path(models_dir, trainer.curr_step, now)
                )
                meta = build_meta(
                    hp=trainer.hp, epsilon=trainer.epsilon, agent_step=trainer.curr_step,
                    episode=completed + 1, updates=trainer.updates, seed=cfg.seed,
                    stage=cfg.stage, device=device.type, source=source, auto=auto,
                    saved_at=now,
                )
                save_checkpoint(trainer.net, path, meta)
                saved_step, saved_updates = trainer.curr_step, trainer.updates
                last_saved = f"{now:%H:%M:%S} {path.name}"
                self.saved.emit({
                    "path": str(path), "auto": auto,
                    "agent_step": trainer.curr_step, "updates": trainer.updates,
                })

            save_now = save

            last_stats = 0.0
            sps = 0.0
            fps = 0.0
            sps_window_start = time.perf_counter()
            sps_window_steps = 0
            fps_window_frames = 0
            last_broadcast = 0
            last_autosave_bucket = 0
            paused_sent = False

            def publish_stats(force: bool = False) -> None:
                nonlocal last_stats, sps, fps
                nonlocal sps_window_start, sps_window_steps, fps_window_frames
                now = time.perf_counter()
                if not force and now - last_stats < 0.1:
                    return
                last_stats = now
                elapsed = now - sps_window_start
                if elapsed >= 0.5:
                    sps = sps_window_steps / elapsed
                    fps = fps_window_frames / elapsed
                    sps_window_start = now
                    sps_window_steps = 0
                    fps_window_frames = 0
                a0 = actor_stats[0]
                self.stats_ready.emit(
                    {
                        "episode": completed + 1,
                        "ep_reward": a0["ep_reward"],
                        "ep_steps": a0["ep_steps"],
                        "total_steps": total_frames,
                        "x_pos": a0["x_pos"],
                        "max_x": best_x,
                        "life": a0["life"],
                        "fps": fps,
                        "steps_per_sec": sps,
                        "envs": n_envs,
                        "action": ACTION_LABELS[a0["action"]],
                        "q": a0["q"],
                        "model": model_name,
                        "speed": controls.speed,
                        "best_reward": best_reward,
                        "agent_step": trainer.curr_step,
                        "epsilon": trainer.epsilon,
                        "loss": float(np.mean(losses)) if losses else None,
                        "avg_q": float(np.mean(qs)) if qs else None,
                        "buffer_size": len(trainer.buffer),
                        "updates": trainer.updates,
                        "learning": learning,
                        "hp": hp_to_dict(trainer.hp),
                        "device": device.type,
                        "last_saved": last_saved,
                        "unsaved_steps": trainer.curr_step - saved_step,
                        "unsaved_updates": trainer.updates - saved_updates,
                    }
                )

            ctx = mp.get_context("spawn")
            result_q = ctx.Queue(maxsize=max(64, n_envs * 64))
            control_qs = [ctx.Queue() for _ in range(n_envs)]
            epsilon = ctx.Value("d", trainer.epsilon)
            learning_v = ctx.Value("i", 1 if learning else 0)
            pause_event = ctx.Event()
            stop_event = ctx.Event()
            speed_v = ctx.Value("d", controls.speed)
            initial = {
                k: v.detach().cpu()
                for k, v in trainer.net.online.state_dict().items()
            }

            from .parallel import actor_main

            for i in range(n_envs):
                proc = ctx.Process(
                    target=actor_main,
                    args=(
                        i, cfg.stage, cfg.seed + i, cfg.auto_restart, result_q,
                        control_qs[i], initial, epsilon, learning_v, pause_event,
                        stop_event, speed_v,
                    ),
                    daemon=True,
                )
                proc.start()
                procs.append(proc)

            self.state_changed.emit("running")

            def handle(message) -> None:
                nonlocal completed, best_reward, best_x, total_frames, ended_unattended
                nonlocal sps_window_steps, fps_window_frames
                kind = message[0]
                if kind == "t":
                    (_, aid, state_u8, next_u8, action, reward, done, q, frames,
                     x_pos, life, ep_reward, ep_steps, label) = message
                    total_frames += frames
                    sps_window_steps += 1
                    fps_window_frames += frames
                    trainer.note_transition(learning)
                    if learning:
                        trainer.cache(state_u8, next_u8, action, reward, done)
                        result = trainer.learn()
                        if result is not None:
                            losses.append(result.loss)
                            qs.append(result.mean_q)
                    actor_stats[aid].update(
                        ep_reward=ep_reward, ep_steps=ep_steps, x_pos=x_pos,
                        life=life, action=label, q=q,
                    )
                    best_x = max(best_x, x_pos)
                elif kind == "episode":
                    _, _aid, reward, steps, x_pos, flag = message
                    completed += 1
                    best_reward = reward if best_reward is None else max(best_reward, reward)
                    history.append(
                        {"episode": completed, "reward": reward, "steps": steps,
                         "max_x": x_pos, "flag": flag}
                    )
                    self.episode_finished.emit(
                        {"episode": completed, "reward": reward, "steps": steps,
                         "max_x": x_pos, "best_reward": best_reward, "flag": flag,
                         "history": list(history)}
                    )
                    publish_stats(force=True)
                    if not cfg.auto_restart:
                        ended_unattended = True
                        controls.request_stop()
                elif kind == "frame":
                    self._store_frame(message[1])
                elif kind == "error":
                    raise RuntimeError(message[2])

            while not controls.is_stopped():
                if controls.is_paused():
                    pause_event.set()
                    if not paused_sent:
                        self.state_changed.emit("paused")
                        sps = 0.0
                        sps_window_steps = 0
                        sps_window_start = time.perf_counter()
                        publish_stats(force=True)
                        paused_sent = True
                    if controls.take_save_request():
                        save(auto=False)
                    time.sleep(0.05)
                    continue
                if paused_sent:
                    pause_event.clear()
                    self.state_changed.emit("running")
                    paused_sent = False

                trainer.set_hyperparams(controls.hyperparams)
                learning = controls.learning
                learning_v.value = 1 if learning else 0
                epsilon_request = controls.take_epsilon()
                if epsilon_request is not None:
                    trainer.set_epsilon(epsilon_request)
                epsilon.value = trainer.epsilon
                speed_v.value = controls.speed
                if controls.take_save_request():
                    save(auto=False)

                got = False
                processed = 0
                deadline = time.perf_counter() + 0.05
                while processed < 128 and time.perf_counter() < deadline:
                    try:
                        message = result_q.get(timeout=0.01)
                    except queue.Empty:
                        break
                    got = True
                    handle(message)
                    processed += 1
                    if controls.is_stopped():
                        break

                broadcast_every = max(100, trainer.hp.sync_every // 4)
                if trainer.curr_step - last_broadcast >= broadcast_every:
                    last_broadcast = trainer.curr_step
                    weights = {
                        k: v.detach().cpu()
                        for k, v in trainer.net.online.state_dict().items()
                    }
                    for control_q in control_qs:
                        try:
                            control_q.put_nowait(("weights", weights))
                        except queue.Full:
                            pass

                every = controls.autosave_every
                if every > 0:
                    bucket = trainer.curr_step // every
                    if bucket > last_autosave_bucket and trainer.updates > saved_updates:
                        last_autosave_bucket = bucket
                        save(auto=True)

                if not got and all(not proc.is_alive() for proc in procs):
                    break
                if not got:
                    time.sleep(0.001)
                publish_stats()

        except Exception:  # noqa: BLE001 - 例外は UI に通知してアプリを落とさない
            ended_unattended = True
            self.failed.emit(traceback.format_exc())
        finally:
            if stop_event is not None:
                stop_event.set()
            if pause_event is not None:
                pause_event.clear()
            deadline = time.perf_counter() + 3.0
            for proc in procs:
                proc.join(timeout=max(0.05, deadline - time.perf_counter()))
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            deadline = time.perf_counter() + 1.0
            for proc in procs:
                proc.join(timeout=max(0.05, deadline - time.perf_counter()))
            try:
                if trainer is not None and save_now is not None:
                    if self.controls.take_save_request():
                        save_now(auto=False)
                    elif ended_unattended and trainer.updates > saved_updates:
                        save_now(auto=True)
            except Exception:  # noqa: BLE001
                logger.exception("終了時の保存に失敗しました")
            self.state_changed.emit("finished")
            self.finished.emit()

    # ------------------------------------------------------------------ run
    @Slot()
    def run(self) -> None:
        if max(1, int(self.config.n_envs)) > 1:
            self._run_parallel()
            return
        env = None
        trainer: DQNTrainer | None = None
        save_now = None
        saved_step = 0
        saved_updates = 0
        ended_unattended = False  # 自動再開 OFF の終了や例外（ユーザーの停止ではない）
        try:
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
            self.state_changed.emit("loading")
            cfg = self.config
            controls = self.controls

            seed_everything(cfg.seed)
            device = resolve_device(cfg.device)
            net = MarioNet((4, 84, 84), len(ACTION_LABELS))
            if cfg.model_path is not None:
                load_weights(net, cfg.model_path, "cpu")
                source = Path(cfg.model_path).name
            else:
                source = "new"
            model_name = source
            trainer = DQNTrainer(net, controls.hyperparams, device, cfg.seed)

            env = make_env(cfg.stage, render_mode="rgb_array")
            state, info = env.reset(seed=cfg.seed)
            shaper = RewardShaper()

            run_started = datetime.now()
            models_dir = Path(cfg.models_dir)
            last_saved: str | None = None
            losses: deque[float] = deque(maxlen=100)
            qs: deque[float] = deque(maxlen=100)

            episode = 1
            ep_reward = 0.0
            ep_steps = 0
            total_steps = 0
            best_reward: float | None = None
            best_x = 0.0
            history: deque[dict] = deque(maxlen=300)

            pending = 0
            action = 0
            action_label = ACTION_LABELS[0]
            repeat_reward = 0.0
            max_q: np.ndarray | None = None
            learning = controls.learning

            next_frame_time = time.perf_counter()
            fps_window_start = time.perf_counter()
            fps_window_frames = 0
            fps = 0.0
            last_stats = 0.0
            paused_sent = False

            def save(auto: bool) -> None:
                nonlocal saved_step, saved_updates, last_saved
                now = datetime.now()
                path = (
                    auto_path(models_dir, run_started)
                    if auto
                    else manual_path(models_dir, trainer.curr_step, now)
                )
                meta = build_meta(
                    hp=trainer.hp, epsilon=trainer.epsilon, agent_step=trainer.curr_step,
                    episode=episode, updates=trainer.updates, seed=cfg.seed, stage=cfg.stage,
                    device=device.type, source=source, auto=auto, saved_at=now,
                )
                save_checkpoint(trainer.net, path, meta)
                saved_step, saved_updates = trainer.curr_step, trainer.updates
                last_saved = f"{now:%H:%M:%S} {path.name}"
                self.saved.emit({
                    "path": str(path), "auto": auto,
                    "agent_step": trainer.curr_step, "updates": trainer.updates,
                })

            save_now = save

            def publish_stats(force: bool = False) -> None:
                nonlocal last_stats, fps, fps_window_frames, fps_window_start
                now = time.perf_counter()
                if not force and now - last_stats < 0.1:
                    return
                last_stats = now
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    fps = fps_window_frames / elapsed
                    fps_window_start = now
                    fps_window_frames = 0
                self.stats_ready.emit(
                    {
                        "episode": episode,
                        "ep_reward": ep_reward,
                        "ep_steps": ep_steps,
                        "total_steps": total_steps,
                        "x_pos": float(info.get("x_pos", 0.0)),
                        "max_x": best_x,
                        "life": int(info.get("life", 0)),
                        "fps": fps,
                        "action": action_label,
                        "q": None if max_q is None else float(np.max(max_q)),
                        "model": model_name,
                        "speed": controls.speed,
                        "best_reward": best_reward,
                        "agent_step": trainer.curr_step,
                        "epsilon": trainer.epsilon,
                        "loss": float(np.mean(losses)) if losses else None,
                        "avg_q": float(np.mean(qs)) if qs else None,
                        "buffer_size": len(trainer.buffer),
                        "updates": trainer.updates,
                        "learning": learning,
                        "hp": hp_to_dict(trainer.hp),
                        "device": device.type,
                        "last_saved": last_saved,
                        "unsaved_steps": trainer.curr_step - saved_step,
                        "unsaved_updates": trainer.updates - saved_updates,
                    }
                )

            self._emit_frame(env)
            self.state_changed.emit("running")

            while not controls.is_stopped():
                # ---- pause
                if controls.is_paused():
                    if not paused_sent:
                        self.state_changed.emit("paused")
                        fps = 0.0
                        fps_window_frames = 0
                        fps_window_start = time.perf_counter()
                        publish_stats(force=True)
                        paused_sent = True
                    if controls.take_save_request():
                        save(auto=False)
                    if not self._sleep(0.05):
                        break
                    next_frame_time = time.perf_counter()
                    continue
                if paused_sent:
                    self.state_changed.emit("running")
                    paused_sent = False

                # ---- 12 フレームごとに行動を決定（ノートブックの skip=12）
                if pending == 0:
                    epsilon_request = controls.take_epsilon()
                    if epsilon_request is not None:
                        trainer.set_epsilon(epsilon_request)
                    trainer.set_hyperparams(controls.hyperparams)
                    learning = controls.learning
                    if controls.take_save_request():
                        save(auto=False)
                    action, max_q = trainer.act(state, learning)
                    action_label = ACTION_LABELS[action]
                    pending = ACTION_REPEAT
                    repeat_reward = 0.0

                next_state, reward, terminated, truncated, info = env.step(action)
                repeat_reward += float(reward)
                pending -= 1
                if terminated or truncated:
                    # エピソード終了後は同じ環境を step できない（nes-py の仕様）
                    pending = 0

                self._emit_frame(env)
                fps_window_frames += 1
                total_steps += 1

                # ---- 行動の繰り返しが終わったタイミングで報酬・状態を確定
                if pending == 0:
                    shaped = shaper.shape(repeat_reward, info)
                    ep_reward += shaped
                    ep_steps += 1
                    x_now = float(info.get("x_pos", 0.0))
                    best_x = max(best_x, x_now)
                    done = terminated or truncated or int(info.get("life", 2)) == 1

                    if learning:
                        trainer.cache(state, next_state, action, shaped, done)
                        result = trainer.learn()
                        if result is not None:
                            losses.append(result.loss)
                            qs.append(result.mean_q)
                        every = controls.autosave_every
                        if (every > 0 and trainer.curr_step % every == 0
                                and trainer.updates > saved_updates):
                            save(auto=True)
                    state = next_state

                    if done:
                        best_reward = ep_reward if best_reward is None else max(best_reward, ep_reward)
                        flag = bool(info.get("flag_get", False))
                        history.append(
                            {"episode": episode, "reward": ep_reward, "steps": ep_steps,
                             "max_x": x_now, "flag": flag}
                        )
                        self.episode_finished.emit(
                            {"episode": episode, "reward": ep_reward, "steps": ep_steps,
                             "max_x": x_now, "best_reward": best_reward, "flag": flag,
                             "history": list(history)}
                        )
                        publish_stats(force=True)

                        if not cfg.auto_restart:
                            ended_unattended = True
                            break
                        if not self._sleep(episode_pause(controls.speed)):
                            break

                        state, info = env.reset()
                        shaper.reset()
                        episode += 1
                        ep_reward = 0.0
                        ep_steps = 0
                        pending = 0
                        repeat_reward = 0.0
                        max_q = None
                        next_frame_time = time.perf_counter()
                        self._emit_frame(env)
                        publish_stats(force=True)

                publish_stats()

                # ---- 実時間ペーシング（1 エミュレータフレーム = 1/60 秒 ÷ 速度）
                period = frame_period(controls.speed)
                if period <= 0:
                    next_frame_time = time.perf_counter()
                    continue
                next_frame_time += period
                delay = next_frame_time - time.perf_counter()
                if delay > 0:
                    if not self._sleep(delay):
                        break
                else:
                    next_frame_time = time.perf_counter()

        except Exception:  # noqa: BLE001 - 例外は UI に通知してアプリを落とさない
            ended_unattended = True
            self.failed.emit(traceback.format_exc())
        finally:
            try:
                # 停止直前の保存要求を取りこぼさない
                if trainer is not None and save_now is not None:
                    if self.controls.take_save_request():
                        save_now(auto=False)
                    elif ended_unattended and trainer.updates > saved_updates:
                        save_now(auto=True)  # ユーザーの確認なしに終わった学習を失わない
            except Exception:  # noqa: BLE001
                logger.exception("終了時の保存に失敗しました")
            if env is not None:
                try:
                    env.close()
                except Exception:
                    logger.exception("環境のクローズに失敗しました")
            self.state_changed.emit("finished")
            self.finished.emit()
