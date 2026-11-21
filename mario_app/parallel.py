"""並列学習用の環境プロセス。

各プロセスが 1 つの環境を回し、観測・行動・報酬をメインプロセスへ送る。
メインプロセスが 1 つのリプレイバッファとネットワークを所有してまとめて更新する
（DQN の学習則はそのまま。行動だけを各環境プロセスで選ぶ）。
"""

from __future__ import annotations

import queue
import time
import traceback

import numpy as np
import torch

from .agent import MarioNet
from .env import ACTION_LABELS, ACTION_REPEAT, RewardShaper, grab_screen, make_env
from .pacing import SPEED_MAX, DisplayThrottle, episode_pause, frame_period
from .trainer import seed_everything


def quantize(frame: np.ndarray) -> np.ndarray:
    """0–1 の float 観測を、リプレイバッファと同じ uint8 に落とす。"""
    return np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)


def actor_main(
    actor_id: int,
    stage: str,
    seed: int,
    auto_restart: bool,
    result_q,
    control_q,
    initial_online: dict,
    epsilon,
    learning,
    pause_event,
    stop_event,
    speed,
) -> None:
    """1 環境を回すプロセスの本体。"""
    try:
        torch.set_num_threads(1)
        seed_everything(seed)
        net = MarioNet((4, 84, 84), len(ACTION_LABELS))
        net.online.load_state_dict(initial_online)
        net.eval()
        rng = np.random.default_rng(seed)
        env = make_env(stage, render_mode="rgb_array")
        throttle = DisplayThrottle()

        def send(message) -> bool:
            while not stop_event.is_set():
                try:
                    result_q.put(message, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def wait(seconds: float) -> bool:
            """停止に反応する sleep。"""
            deadline = time.perf_counter() + max(0.0, seconds)
            while not stop_event.is_set():
                left = deadline - time.perf_counter()
                if left <= 0:
                    return True
                time.sleep(min(0.05, left))
            return False

        state, info = env.reset(seed=seed)
        shaper = RewardShaper()
        frame_throttle = DisplayThrottle(max_fps=10.0)

        def send_frame() -> None:
            """画面フレームは落としてよいので、キューが一杯なら捨てる。"""
            f = grab_screen(env)
            small = f.reshape(120, 2, 128, 2, 3).mean(axis=(1, 3)).astype("uint8")
            try:
                if actor_id == 0:
                    result_q.put_nowait(("frame", small))
                else:
                    result_q.put_nowait(("frame2", actor_id, small))
            except queue.Full:
                pass
        frame_count = 0
        ep_reward = 0.0
        ep_steps = 0
        pending = 0
        action = 0
        action_label = 0
        action_frames = 0
        repeat_reward = 0.0
        max_q: float | None = None
        next_frame_time = time.perf_counter()

        send_frame()

        while not stop_event.is_set():
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.05)
            if stop_event.is_set():
                break

            # メインプロセスからの重み更新を取り込む（最新だけ使う）
            latest_weights = None
            while True:
                try:
                    kind, payload = control_q.get_nowait()
                except queue.Empty:
                    break
                if kind == "weights":
                    latest_weights = payload
            if latest_weights is not None:
                net.online.load_state_dict(latest_weights)

            if pending == 0:
                eps = float(epsilon.value)
                if rng.random() < eps:
                    action, max_q = int(rng.integers(len(ACTION_LABELS))), None
                else:
                    with torch.no_grad():
                        x = torch.as_tensor(
                            np.asarray(state, dtype=np.float32)
                        ).unsqueeze(0)
                        q_values = net(x, "online")[0].numpy()
                    action = int(np.argmax(q_values))
                    max_q = float(np.max(q_values))
                action_label = action
                pending = ACTION_REPEAT
                action_frames = 0
                repeat_reward = 0.0

            next_state, reward, terminated, truncated, info = env.step(action)
            repeat_reward += float(reward)
            pending -= 1
            action_frames += 1
            frame_count += 1
            if terminated or truncated:
                pending = 0

            current_speed = float(speed.value)
            if frame_throttle.should_emit(SPEED_MAX, frame_count):
                send_frame()

            if pending == 0:
                shaped = shaper.shape(repeat_reward, info)
                ep_reward += shaped
                ep_steps += 1
                x_now = float(info.get("x_pos", 0.0))
                life = int(info.get("life", 2))
                done = terminated or truncated or life == 1
                if not send(
                    (
                        "t",
                        actor_id,
                        quantize(state),
                        quantize(next_state),
                        action,
                        shaped,
                        done,
                        max_q,
                        action_frames,
                        x_now,
                        life,
                        ep_reward,
                        ep_steps,
                        action_label,
                    )
                ):
                    break
                state = next_state

                if done:
                    flag = bool(info.get("flag_get", False))
                    send(("episode", actor_id, ep_reward, ep_steps, x_now, flag))
                    if not auto_restart:
                        break
                    if not wait(episode_pause(current_speed)):
                        break
                    state, info = env.reset()
                    shaper.reset()
                    ep_reward = 0.0
                    ep_steps = 0
                    pending = 0
                    action_frames = 0
                    repeat_reward = 0.0
                    max_q = None
                    next_frame_time = time.perf_counter()
                    if actor_id == 0:
                        send(("frame", grab_screen(env)))

            period = frame_period(current_speed)
            if period <= 0:
                next_frame_time = time.perf_counter()
                continue
            next_frame_time += period
            delay = next_frame_time - time.perf_counter()
            if delay > 0:
                if not wait(delay):
                    break
            else:
                next_frame_time = time.perf_counter()

        env.close()
    except Exception:  # noqa: BLE001 - 失敗はメインプロセスに伝える
        try:
            result_q.put(("error", actor_id, traceback.format_exc()), timeout=1.0)
        except Exception:
            pass
