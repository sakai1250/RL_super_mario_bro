#!/usr/bin/env python3
"""フル学習: 全ステージを順に学習し、最後にモデルをマージして評価する。

使い方 (リポジトリ直下で):
    .venv/bin/python scripts/train_all_stages.py            # 全体 (2グループ並行→マージ→評価)
    .venv/bin/python scripts/train_all_stages.py --worker 0 # ワールド1-4 (GPUは環境変数で指定)
    .venv/bin/python scripts/train_all_stages.py --worker 1 # ワールド5-8
    .venv/bin/python scripts/train_all_stages.py --merge    # マージと評価だけ

- 各ステージは 12 並列環境で学習し、クリア (flag) を確認したら次へ進む。
- 1 ステージの上限は BUDGET ステップ。上限到達時は最良チェックポイントを残して次へ。
- results/full_run/<stage>/done.json があるステージは飛ばす (再開可能)。
- 学習後、クリアしたステージの重みを平均化して merged を作り、全ステージで評価する。
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "full_run"
BUDGET = 1_200_000          # 1ステージあたりの agent step 上限
N_ENVS = 12                 # 1ステージあたりの並列環境数
AUTOSAVE_EVERY = 50_000


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "progress.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------- PNG 保存
def save_png(path: Path, arr) -> None:
    import numpy as np

    arr = np.asarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def render_montage(stage: str, ckpt: Path, out_png: Path) -> None:
    """学習済みモデルで少しプレイし、画面を 2x2 モンタージュで保存する。"""
    import numpy as np
    import torch

    from mario_app.agent import MarioNet, load_weights
    from mario_app.env import grab_screen, make_env

    net = MarioNet()
    load_weights(net, ckpt, device="cpu")
    net.eval()
    env = make_env(stage)
    obs, _info = env.reset()
    frames = []
    grab_at = {300, 900, 1500, 2100}
    for i in range(2400):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            action = int(net(x, "online")[0].argmax().item())
        obs, _r, term, trunc, _info = env.step(action)
        if i in grab_at:
            frames.append(grab_screen(env))
        if term or trunc:
            obs, _info = env.reset()
    env.close()
    if not frames:
        return
    h, w, _ = frames[0].shape
    while len(frames) < 4:
        frames.append(frames[-1])
    sheet = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    for idx, f in enumerate(frames[:4]):
        sheet[(idx // 2) * h:(idx // 2 + 1) * h, (idx % 2) * w:(idx % 2 + 1) * w] = f
    save_png(out_png, sheet)


# ---------------------------------------------------------------- 学習
def train_stage(stage: str, group: int) -> dict:
    from PySide6.QtCore import QCoreApplication

    from mario_app.game_worker import GameWorker, RunConfig
    from mario_app.hyperparams import Hyperparams
    from mario_app.pacing import SPEED_MAX

    _app = QCoreApplication.instance() or QCoreApplication([])
    stage_dir = RESULTS / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # 既存の自動保存があれば、重み・ステップ・ε を引き継いで再開する
    resume_step = 0
    model_path = None
    hp = Hyperparams()
    cands = sorted(stage_dir.glob("*.pth"), key=lambda pth: pth.stat().st_mtime)
    if cands:
        ckpt_path = cands[-1]
        try:
            meta = json.loads(ckpt_path.with_suffix(".json").read_text(encoding="utf-8"))
            resume_step = int(meta.get("agent_step", 0))
            hp = Hyperparams(epsilon=float(meta.get("epsilon", hp.epsilon)))
            model_path = str(ckpt_path)
            log(f"{stage}: resume from {ckpt_path.name} step {resume_step} eps {hp.epsilon:.4f}")
        except Exception as exc:  # メタが読めなければ新規開始
            log(f"{stage}: resume failed ({exc}); starting fresh")
            resume_step = 0
            model_path = None
            hp = Hyperparams()

    worker = GameWorker(RunConfig(
        model_path=model_path, stage=stage, hp=hp, learning=True,
        speed=SPEED_MAX, device="cuda", autosave_every=AUTOSAVE_EVERY,
        models_dir=str(stage_dir), n_envs=N_ENVS, seed=15 + group))

    state = {"cleared": False, "steps": resume_step, "episodes": 0,
             "best_reward": None, "saves": [], "failed": None}
    last_bucket = {"v": -1}

    # ライブ表示用に、全環境の最新フレームを定期的にファイルへ書き出す
    import threading

    live_stop = threading.Event()

    def live_tap() -> None:
        import numpy as np

        live_dir = RESULTS / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        while not live_stop.is_set():
            try:
                _seq, frames = worker.latest_frames()
                if frames:
                    tmp = live_dir / f"group{group}.tmp.npz"
                    np.savez(tmp, **{str(k): v for k, v in frames.items()})
                    os.replace(tmp, live_dir / f"group{group}.npz")
                    meta_out = {"stage": stage, "group": group,
                                "steps": state.get("steps", 0),
                                "episodes": state.get("episodes", 0),
                                "updated": time.time()}
                    (live_dir / f"group{group}.json").write_text(
                        json.dumps(meta_out), encoding="utf-8")
            except Exception:
                pass
            live_stop.wait(0.5)

    threading.Thread(target=live_tap, daemon=True).start()

    def on_episode(e):
        if e.get("flag"):
            state["cleared"] = True
            log(f"{stage}: CLEARED at episode {e['episode']} reward {e['reward']:.1f}")
            worker.request_save()
            worker.request_stop()

    def on_stats(s):
        total = resume_step + s["agent_step"]
        state["steps"] = total
        state["episodes"] = s["episode"]
        state["best_reward"] = s.get("best_reward")
        bucket = total // 100_000
        if bucket != last_bucket["v"]:
            last_bucket["v"] = bucket
            log(f"{stage}: step {total} episodes {s['episode']} "
                f"best_reward {s.get('best_reward')} loss {s.get('loss')}")
        if total >= BUDGET and not state["cleared"]:
            log(f"{stage}: budget reached ({BUDGET})")
            worker.request_save()
            worker.request_stop()

    worker.episode_finished.connect(on_episode)
    worker.stats_ready.connect(on_stats)
    worker.saved.connect(lambda sv: state["saves"].append(sv["path"]))
    worker.failed.connect(lambda msg: state.update(failed=msg))
    worker.run()
    live_stop.set()
    if state["failed"]:
        raise RuntimeError(state["failed"])

    ckpt = state["saves"][-1] if state["saves"] else None
    if ckpt is None:
        cands = sorted(stage_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        ckpt = str(cands[-1]) if cands else None
    result = {"stage": stage, "cleared": state["cleared"], "steps": state["steps"],
              "best_reward": state["best_reward"], "checkpoint": ckpt}
    (stage_dir / "done.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    log(f"{stage}: done cleared={result['cleared']} steps={result['steps']} ckpt={ckpt}")
    if ckpt:
        try:
            render_montage(stage, Path(ckpt), RESULTS / "frames" / f"{stage}.png")
            log(f"{stage}: frame saved")
        except Exception as exc:  # 描画失敗で学習全体を止めない
            log(f"{stage}: frame failed: {exc}")
    return result


def run_worker(group: int) -> None:
    from mario_app.env import STAGES

    worlds = (1, 2, 3, 4) if group == 0 else (5, 6, 7, 8)
    stages = [s for s in STAGES if int(s.split("-")[0]) in worlds]
    log(f"worker {group}: stages {stages[0]}..{stages[-1]} device env={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for stage in stages:
        done = RESULTS / stage / "done.json"
        if done.exists():
            log(f"{stage}: skip (already done)")
            continue
        train_stage(stage, group)
    log(f"worker {group}: all stages finished")


# ---------------------------------------------------------------- マージと評価
def merge_and_eval() -> None:
    import torch

    from mario_app.agent import MarioNet, _clean_state_dict
    from mario_app.checkpoint import save_checkpoint

    RESULTS.mkdir(parents=True, exist_ok=True)
    dones = []
    for d in sorted(RESULTS.glob("*-*")):
        f = d / "done.json"
        if f.exists():
            dones.append(json.loads(f.read_text(encoding="utf-8")))
    cleared = [d for d in dones if d.get("cleared") and d.get("checkpoint")]
    log(f"merge: {len(cleared)} cleared stages of {len(dones)} done")
    summary = {"stages": dones, "merged_from": [d["stage"] for d in cleared]}
    if cleared:
        acc = None
        for d in cleared:
            raw = torch.load(d["checkpoint"], map_location="cpu")
            sd = _clean_state_dict(raw)
            if acc is None:
                acc = {k: v.float().clone() for k, v in sd.items()}
            else:
                for k in acc:
                    if k in sd:
                        acc[k] += sd[k].float()
        n = float(len(cleared))
        merged_sd = {k: (v / n) for k, v in acc.items()}
        net = MarioNet()
        net.load_state_dict({k: v for k, v in merged_sd.items() if k in net.state_dict()},
                            strict=False)
        merged_dir = RESULTS / "merged"
        merged_path = merged_dir / "merged.pth"
        save_checkpoint(net, merged_path, {"merged_from": summary["merged_from"]})
        log(f"merge: saved {merged_path}")
        summary["merged_model"] = str(merged_path)
        summary["merged_eval"] = eval_model(merged_path)
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    lines = ["# Full run summary", "",
             f"- done stages: {len(dones)} / 32",
             f"- cleared stages: {[d['stage'] for d in dones if d.get('cleared')]}",
             f"- merged from: {summary['merged_from']}"]
    if "merged_eval" in summary:
        lines.append(f"- merged model cleared: {[s for s, ok in summary['merged_eval'].items() if ok]}")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("merge: summary written")


def eval_model(model_path: Path) -> dict:
    from PySide6.QtCore import QCoreApplication

    from mario_app.env import STAGES
    from mario_app.game_worker import GameWorker, RunConfig
    from mario_app.hyperparams import Hyperparams
    from mario_app.pacing import SPEED_MAX

    _app = QCoreApplication.instance() or QCoreApplication([])
    results = {}
    for stage in STAGES:
        worker = GameWorker(RunConfig(
            model_path=str(model_path), stage=stage, hp=Hyperparams(), learning=False,
            speed=SPEED_MAX, device="cuda", autosave_every=0,
            models_dir=str(RESULTS / "eval_tmp"), n_envs=4, seed=99))
        seen = {"eps": 0, "cleared": False}

        def on_episode(e, seen=seen, worker=worker):
            seen["eps"] += 1
            if e.get("flag"):
                seen["cleared"] = True
            if seen["eps"] >= 5 or seen["cleared"]:
                worker.request_stop()

        def on_stats(s, seen=seen, worker=worker):
            if s["agent_step"] >= 60_000:
                worker.request_stop()

        worker.episode_finished.connect(on_episode)
        worker.stats_ready.connect(on_stats)
        worker.failed.connect(lambda msg: None)
        try:
            worker.run()
        except Exception as exc:
            log(f"eval {stage}: error {exc}")
        results[stage] = seen["cleared"]
        log(f"eval {stage}: cleared={seen['cleared']} episodes={seen['eps']}")
    return results


# ---------------------------------------------------------------- 全体
def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        run_worker(int(sys.argv[2]))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_and_eval()
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    log("orchestrator: starting two workers")
    procs = []
    for group in (0, 1):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(group)
        fh = open(RESULTS / f"group{group}.log", "a", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(group)],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append(p)
        log(f"orchestrator: worker {group} pid {p.pid} (GPU {group})")
    for p in procs:
        p.wait()
    log("orchestrator: workers done, merging")
    merge_and_eval()
    log("orchestrator: ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
