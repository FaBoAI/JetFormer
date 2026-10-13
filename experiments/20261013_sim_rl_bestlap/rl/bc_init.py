"""予備策 1: RC 教師の行動で方策を初期化する(BC + DAgger 風)。2026-10-13、RL の学習の段(design.md 2.9 節 / protocol.json fallbacks_rl)

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    CUDA_VISIBLE_DEVICES= jf-heavy --mem 2G --lock rl -- python -u rl/bc_init.py --out ~/jetformer_reports/sim_rl_bestlap_20261013/rl/bc-init

すること:
  1. 64 環境を RC 教師の約束の範囲(ゲート 0・横 ±3 cm・向き ±3°)の開始にし、教師の操作で走らせて (観測, 教師の行動) を集める。
  2. 方策の平均の MLP(PPO と同じ形 79→256→256→2、tanh)を 2 乗誤差で学習する(観測の正規化は集めた観測の平均・分散)。
  3. DAgger 風に 4 回: いまの方策(決定的 + 小さなゆらぎ)で走らせ、訪れた状態に教師の行動を付けて足し、学び直す。
     教師はどの状態でも参照に戻る操作を返す(フィードバックの制御器)ので、方策が外れた状態にも正しい行動が付く。
  4. 書き出し: init.pt(train_rl.py --init が読む {model, rms})、policy.json(JS の方策の形。BC の前の値の評価用)、bc.json(集計)。
PPO はこのあと train_rl.py --init <init.pt> --log-std-init -1.6 で 1,500 万ステップ(手順どおり)。
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vec_env import NodeVecEnv  # noqa: E402
from ppo import ActorCritic, PPOConfig, RunningMeanStd, export_policy_json, OBS_DIM  # noqa: E402


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def contract_starts(n: int, seed: int) -> list[dict]:
    """教師の約束の範囲の開始を n 個(横ずれ・向きを格子 + 乱数で散らす)。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append({"gate": 0, "left_offset_m": float(rng.uniform(-0.03, 0.03)), "left_yaw_deg": float(rng.uniform(-3.0, 3.0)),
                    "seed": int(2026101300 + i), "scenario": "baseline", "noiseStd": 0})
    out[0].update(left_offset_m=0.0, left_yaw_deg=0.0)
    return out


def rollout(env: NodeVecEnv, frames: int, policy=None, noise: float = 0.0, rng=None):
    """frames フレーム走らせて (観測, 教師の行動) を集める。policy が None なら教師の操作で走る。"""
    obs = env.reset()
    teach = env.last_teacher_action
    X, Y, eps = [], [], []
    for t in range(frames):
        ok = np.isfinite(teach).all(1)
        X.append(obs[ok]); Y.append(teach[ok])
        if policy is None:
            a = np.where(np.isfinite(teach), teach, 0.0)
        else:
            a = policy(obs)
            if noise > 0:
                a = a + noise * rng.standard_normal(a.shape)
            a = np.clip(a, -1, 1)
        obs, r, term, trunc, info = env.step(a.astype(np.float32))
        teach = info["teacher_action"]
        eps += info["episodes"]
    return np.concatenate(X), np.concatenate(Y), eps


def fit(model: ActorCritic, rms: RunningMeanStd, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    xn = torch.from_numpy(rms.normalize(X))
    y = torch.from_numpy(Y.astype(np.float32))
    n = xn.shape[0]
    loss_v = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = ((model.actor(xn[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        loss_v = tot / n
    return loss_v


def summarize(eps: list[dict]) -> dict:
    if not eps:
        return {"n": 0}
    reasons: dict[str, int] = {}
    for e in eps:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    lap1 = [e["laps"][0]["elapsed"] for e in eps if e.get("laps") and e["laps"][0]["gates"] == 33 and e["laps"][0]["collisions"] == 0]
    return {"n": len(eps), "reasons": reasons, "valid_lap1": len(lap1), "lap1_mean": float(np.mean(lap1)) if lap1 else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="予備策 1: 教師の行動での初期化(BC + DAgger 風)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="main")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--frames", type=int, default=600, help="1 回の収集のフレーム数(64 環境 × 600 = 38,400 状態)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--torch-threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig()
    model = ActorCritic(cfg)
    rms = RunningMeanStd()
    log = {"schema": "rl-bestlap-bc/v1", "started_at": now(), "args": {k: str(v) for k, v in vars(args).items()}, "rounds": []}
    starts = contract_starts(64, args.seed)
    t0 = time.time()
    with NodeVecEnv(n_workers=4, envs_per_worker=16, seed=args.seed, variant=args.variant, mode="train", teacher_actions=True, starts=starts) as env:
        X, Y, eps = rollout(env, args.frames)
        rms.update(X)
        loss = fit(model, rms, X, Y, args.epochs, args.lr, args.seed)
        log["rounds"].append({"round": 0, "driver": "teacher", "states": int(X.shape[0]), "episodes": summarize(eps), "fit_mse": loss})
        print(f"[{now()}] 教師で {X.shape[0]:,} 状態 / 学習の 2 乗誤差 {loss:.5f} / {summarize(eps)}", flush=True)

        def pol(o):
            with torch.no_grad():
                return model.actor(torch.from_numpy(rms.normalize(o))).clamp(-1, 1).numpy()

        for k in range(1, args.rounds + 1):
            Xk, Yk, epk = rollout(env, args.frames, policy=pol, noise=0.1, rng=rng)
            X, Y = np.concatenate([X, Xk]), np.concatenate([Y, Yk])
            loss = fit(model, rms, X, Y, max(10, args.epochs // 2), args.lr * 0.5, args.seed + k)
            s = summarize(epk)
            log["rounds"].append({"round": k, "driver": "policy+noise0.1", "states_added": int(Xk.shape[0]), "states_total": int(X.shape[0]),
                                  "episodes_during_collection": s, "fit_mse": loss})
            print(f"[{now()}] DAgger {k}: +{Xk.shape[0]:,} 状態(計 {X.shape[0]:,})/ 走らせた回 {s} / 2 乗誤差 {loss:.5f}", flush=True)
    meta = {"variant": args.variant, "seed": args.seed, "steps": 0, "bc": True, "created_at": now(), "rounds": args.rounds}
    torch.save({"model": model.state_dict(), "rms": rms.state_dict(), "meta": meta}, args.out / "init.pt")
    (args.out / "policy.json").write_text(json.dumps(export_policy_json(model, rms, cfg, meta)))
    log["finished_at"] = now()
    log["wall_s"] = time.time() - t0
    (args.out / "bc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    print(f"[{now()}] 書き出しました: {args.out}/init.pt・policy.json({time.time() - t0:.0f} 秒)", flush=True)


if __name__ == "__main__":
    main()
