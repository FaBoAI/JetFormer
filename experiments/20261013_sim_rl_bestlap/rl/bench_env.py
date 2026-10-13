"""RL 環境の速さとメモリを測る(2026-10-13、RL 環境の段)。

    jf-heavy --mem 2G --lock rl -- python -u rl/bench_env.py [--out 結果.json] [--seconds 8]

ワーカーの数 W = 1〜6(1 ワーカー 16 環境)で、Python の側から見た steps/s(全環境の合計)と、
各ワーカー(Node)と Python の常駐メモリ(VmRSS)を測る。行動は 2 通り:
  follower: 観測の先読み点(1 m 先)を追う簡単な追従(長いエピソード = 学習が進んだときに近い)
  random:   一様乱数(すぐ衝突してリセットが多い = 学習の始めに近い)
同じ機械でほかの処理が動いていると遅く出るので、測ったときの負荷(loadavg)も残す。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from vec_env import NodeVecEnv


def rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def follower(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fwd, left = obs[:, 8 + 2 * 3] * 4, obs[:, 9 + 2 * 3] * 4   # 1.0 m 先の参照点(車の座標系)
    alpha = np.arctan2(left, fwd)
    steer = np.arctan2(2 * 0.26 * np.sin(alpha), np.maximum(np.hypot(fwd, left), 0.3))
    a0 = np.clip(steer / 0.52 + rng.normal(0, 0.05, len(obs)), -1, 1)
    a1 = np.full(len(obs), 2 * 2.0 / 4.5 - 1) + rng.normal(0, 0.05, len(obs))   # 約 2 m/s
    return np.stack([a0, a1], 1).astype(np.float32)


def bench(n_workers: int, kind: str, seconds: float, envs_per_worker: int = 16) -> dict:
    rng = np.random.default_rng(0)
    t_start = time.time()
    with NodeVecEnv(n_workers=n_workers, envs_per_worker=envs_per_worker, seed=1) as env:
        startup = time.time() - t_start
        obs = env.reset()
        for _ in range(30):
            a = follower(obs, rng) if kind == "follower" else rng.uniform(-1, 1, (env.num_envs, 2)).astype(np.float32)
            obs, *_ = env.step(a)
        steps, episodes, laps, t0 = 0, 0, 0, time.time()
        reasons: dict[str, int] = {}
        while time.time() - t0 < seconds:
            a = follower(obs, rng) if kind == "follower" else rng.uniform(-1, 1, (env.num_envs, 2)).astype(np.float32)
            obs, r, te, tr, info = env.step(a)
            steps += env.num_envs
            for e in info["episodes"]:
                episodes += 1; laps += e["completed_laps"]; reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
        dt = time.time() - t0
        worker_rss = [rss_mb(p) for p in env.worker_pids()]
        return {"workers": n_workers, "envs": env.num_envs, "actions": kind, "steps_per_s": steps / dt, "steps_per_s_per_worker": steps / dt / n_workers,
                "measured_s": dt, "steps": steps, "episodes": episodes, "mean_episode_frames": steps / max(episodes, 1),
                "completed_laps": laps, "reasons": reasons, "startup_s": startup, "worker_rss_mb": worker_rss,
                "worker_rss_total_mb": float(np.nansum(worker_rss)), "python_rss_mb": rss_mb(os.getpid()),
                "loadavg_after": os.getloadavg()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/env/bench.json")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args()
    out = {"schema": "rl-bestlap-env-bench/v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cpu_count": os.cpu_count(),
           "loadavg_before": os.getloadavg(), "envs_per_worker": 16, "rows": []}
    for kind in ("follower", "random"):
        for w in range(1, args.max_workers + 1):
            row = bench(w, kind, args.seconds)
            out["rows"].append(row)
            print(f"{kind:8s} W={w} 環境 {row['envs']:3d}: {row['steps_per_s']:8.0f} steps/s(1 ワーカーあたり {row['steps_per_s_per_worker']:6.0f})"
                  f" 平均エピソード {row['mean_episode_frames']:5.0f} フレーム  RSS ワーカー計 {row['worker_rss_total_mb']:5.0f} MB・Python {row['python_rss_mb']:4.0f} MB"
                  f"  負荷 {row['loadavg_after'][0]:.1f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("結果:", args.out)


if __name__ == "__main__":
    main()
