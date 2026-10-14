"""PPO の学習(CPU)。2026-10-13、RL の学習の段(design.md 2.7〜2.10 節 / protocol.json algorithm・budget・selection)

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    CUDA_VISIBLE_DEVICES= jf-heavy --mem 2G --lock rl -- python -u rl/train_rl.py --variant main --seed 0 --steps 15000000 \
        --out ~/jetformer_reports/sim_rl_bestlap_20261013/rl/main-s0

出力(--out の下):
  run.json                   設定・protocol の sha256・開始と再開の記録
  progress.jsonl             更新ごと(16,384 ステップ)の学習の値と、その間に終わったエピソードの集計(周回時間・完走・報酬)
  last.pt                    再開用(50 万ステップごと・終わるとき。原子的に置き換える)
  checkpoints/step_N.pt      200 万ステップごとと最後の再開用の一式(方策・価値・最適化・正規化・ステップ数・乱数)
  policies/step_N.json       そのときの決定的な方策(JS の rl/policy_forward.mjs が読む重み。正規化は固定値)
  evals/step_N.json          Node の高速版のコア・JS の方策で REC60・STD・ROB24 を走らせた結果(rl/evaluate_policy.mjs)
  done.json                  予定のステップを終えた印
電源断のあとは同じコマンドで last.pt から続ける(環境はリセットし直すので完全には同じにならない。run.json の resumes に残す)。
"""
from __future__ import annotations

import os

# numpy(OpenBLAS)のスレッドが空回りして Node のワーカーの CPU を奪うので 1 つにする(収集の行列は小さい)。torch の更新には効かない
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vec_env import NodeVecEnv  # noqa: E402
from ppo import PPO, PPOConfig, export_policy_json  # noqa: E402

REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_bestlap_20261013")
PROTOCOL = REPORTS / "protocol.json"
NODE = os.environ.get("RL_NODE", "/usr/bin/node")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode())


def atomic_torch_save(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def valid_lap(l: dict) -> bool:
    return l.get("gates") == 33 and l.get("crossings") == 1 and l.get("collisions") == 0


def is_slow_weak(e: dict) -> bool:
    """r3: 回の物理が「遅い舵 × 弱いブレーキ」の角か(env_core.mjs isSlowSteerWeakBrake と同じ)。"""
    ph = ((e.get("start") or {}).get("perturb") or {}).get("physics") or {}
    return ph.get("steerRate", 9) <= 1.0 + 1e-12 and ph.get("brakeDeceleration", 9) <= 0.9 + 1e-12


def summarize_episodes(eps: list[dict]) -> dict:
    """この更新の間に終わったエピソードの集計(学習曲線用。確率的な方策の値)。"""
    if not eps:
        return {"n": 0}
    reasons: dict[str, int] = {}
    for e in eps:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    lap1, lapf, lap1_g0 = [], [], []
    two_valid = 0
    for e in eps:
        laps = e.get("laps") or []
        if laps and valid_lap(laps[0]):
            lap1.append(laps[0]["elapsed"])
            if e["start"]["gate"] == 0:
                lap1_g0.append(laps[0]["elapsed"])
        for l in laps[1:]:
            if valid_lap(l):
                lapf.append(l["elapsed"])
        if e["reason"] == "collection_duration_complete" and e.get("collisions", 0) == 0 and len(laps) >= 2 and all(valid_lap(l) for l in laps):
            two_valid += 1

    def st(a):
        return {"n": len(a), "mean": float(np.mean(a)), "min": float(np.min(a)), "median": float(np.median(a))} if a else {"n": 0}

    n = len(eps)
    return {"n": n, "reasons": reasons, "return_mean": float(np.mean([e["return"] for e in eps])),
            "frames_mean": float(np.mean([e["frames"] for e in eps])), "progress_m_mean": float(np.mean([e["progress_m"] for e in eps])),
            "survived_20s_frac": reasons.get("collection_duration_complete", 0) / n,
            "two_valid_laps_frac": two_valid / n,
            "laps_per_episode": float(np.mean([e.get("completed_laps") or 0 for e in eps])),
            "lap1": st(lap1), "lap1_gate0": st(lap1_g0), "flying": st(lapf),
            "min_clearance_mean": float(np.mean([e["min_clearance"] for e in eps])),
            "max_speed_mean": float(np.mean([e["max_speed"] for e in eps])),
            "reward_parts_mean": {k: float(np.mean([e["reward_parts"][k] for e in eps])) for k in eps[0]["reward_parts"]}}


class AsyncEvaluator:
    """200 万ステップごとの評価(Node の JS の方策)を裏で 1 つずつ走らせる。学習は止めない。"""

    def __init__(self, out: Path, variant: str):
        self.out, self.variant = out, variant
        self.queue: list[Path] = []
        self.proc: subprocess.Popen | None = None
        self.current: Path | None = None
        self.log = None

    def eval_path(self, policy_json: Path) -> Path:
        return self.out / "evals" / policy_json.name

    def submit(self, policy_json: Path) -> None:
        if not self.eval_path(policy_json).exists() and policy_json not in self.queue and policy_json != self.current:
            self.queue.append(policy_json)
        self.poll()

    def poll(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            if self.log:
                self.log.close()
            if code != 0:
                print(f"[{now()}] 評価が失敗しました(終了コード {code}): {self.current}", flush=True)
            self.proc, self.current, self.log = None, None, None
        if self.proc is None and self.queue:
            p = self.queue.pop(0)
            ev = self.eval_path(p)
            ev.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(ev.with_suffix(".log"), "w")
            cmd = [NODE, str(HERE / "evaluate_policy.mjs"), "--policy", str(HERE / "policy_forward.mjs"), "--policy-arg", str(p),
                   "--variant", self.variant, "--core", "fast", "--out", str(ev)]
            self.proc = subprocess.Popen(cmd, cwd=str(HERE.parent), stdout=self.log, stderr=subprocess.STDOUT)
            self.current = p

    def wait_all(self) -> None:
        while self.proc is not None or self.queue:
            self.poll()
            time.sleep(0.5)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO の学習(RL のベストラップ)")
    ap.add_argument("--variant", choices=["main", "open"], default="main")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True, help="学習の総ステップ数 B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=4, help="更新(torch)のスレッド数。収集の間は numpy で計算し、更新の間は Node のワーカーが止まっている")
    ap.add_argument("--ckpt-every", type=int, default=2_000_000)
    ap.add_argument("--save-every", type=int, default=500_000, help="再開用の last.pt を書く間隔")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--log-std-init", type=float, default=None)
    ap.add_argument("--lr-start", type=float, default=None)
    ap.add_argument("--init", type=Path, default=None, help="方策を初期化する重み(予備策 1 の BC。.pt の model)")
    ap.add_argument("--tag", default="")
    # r2(手順の外の参考の追加実験): 物理のずれのランダム化で、選んだ方策から学習を続ける
    ap.add_argument("--dr", default=None, help="回ごとの物理のずれ(env_core.mjs DR_SPECS の名前。例 r2)。省略すると今までと同じ")
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="方策・価値・観測の正規化・log σ をこのチェックポイント(.pt)から始める(最適化の状態とステップ数は新しくする)")
    # r3(確認 r2 のあと。手順の外の参考の追加実験): すき間の約束つきの報酬
    ap.add_argument("--reward", default=None, choices=["r3"], help="報酬の名前(env_core.mjs REWARDS)。省略すると本番と同じ")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    out: Path = args.out
    for sub in ("checkpoints", "policies", "evals"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"[{now()}] {out} はもう終わっています(done.json)。評価の抜けだけを埋めます", flush=True)

    cfg = PPOConfig()
    if args.log_std_init is not None:
        cfg.log_std_init = args.log_std_init
    if args.lr_start is not None:
        cfg.lr_start = args.lr_start

    run_path = out / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {
        "schema": "rl-bestlap-run/v1", "variant": args.variant, "seed": args.seed, "total_steps": args.steps, "tag": args.tag,
        "n_envs": args.workers * args.envs_per_worker, "workers": args.workers, "envs_per_worker": args.envs_per_worker,
        "ppo": cfg.to_dict(), "torch_threads": args.torch_threads, "torch": torch.__version__, "python": sys.version.split()[0],
        "protocol_sha256": sha256(PROTOCOL), "started_at": now(), "resumes": [], "init": str(args.init) if args.init else None,
        "dr": args.dr, "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None, "reward": args.reward, "argv": sys.argv}
    if run.get("total_steps") != args.steps:
        raise SystemExit(f"run.json の total_steps({run.get('total_steps')})と --steps({args.steps})が違います")

    env = NodeVecEnv(n_workers=args.workers, envs_per_worker=args.envs_per_worker, seed=args.seed, variant=args.variant, mode="train",
                     dr=args.dr, reward=args.reward)
    run["env"] = env.info
    agent = PPO(env, cfg, total_steps=args.steps, seed=args.seed)
    evaluator = AsyncEvaluator(out, args.variant)

    last = out / "last.pt"
    if last.exists():
        d = torch.load(last, map_location="cpu", weights_only=False)
        agent.load_state_dict(d)
        reset_seed = (args.seed * 1_000_003 + agent.steps) & 0x7FFFFFFF
        run["resumes"].append({"at": now(), "steps": agent.steps, "env_reset_seed": reset_seed})
        print(f"[{now()}] last.pt から再開します: {agent.steps:,} ステップ", flush=True)
        agent.start(reset_seed)
    else:
        if args.init_ckpt is not None:
            d = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            agent.model.load_state_dict(d["model"])   # actor・critic・log σ(学習の終わりの値のまま)
            agent.rms.load_state_dict(d["rms"])
            if args.log_std_init is not None:
                with torch.no_grad():
                    agent.model.log_std.fill_(cfg.log_std_init)
            run["init_ckpt_sha256"] = sha256(args.init_ckpt)
            run["init_ckpt_steps"] = int(d.get("steps", 0))
        if args.init is not None:
            d = torch.load(args.init, map_location="cpu", weights_only=False)
            agent.model.load_state_dict(d["model"])
            if "rms" in d:
                agent.rms.load_state_dict(d["rms"])
            with torch.no_grad():
                agent.model.log_std.fill_(cfg.log_std_init)
            run["init_sha256"] = sha256(args.init)
        agent.start(args.seed)
    atomic_json(run_path, run)

    # 抜けている評価を埋める(電源断で評価の途中だったもの)
    if not args.no_eval:
        for p in sorted((out / "policies").glob("step_*.json")):
            evaluator.submit(p)

    stop_flag = {"stop": False}

    def on_term(signum, frame):
        stop_flag["stop"] = True
        print(f"[{now()}] 止める合図を受けました。区切りで保存して終わります", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def save_resume(path: Path) -> None:
        atomic_torch_save(path, agent.state_dict())

    def checkpoint(final: bool = False) -> None:
        tag = f"step_{agent.steps:09d}"
        save_resume(out / "checkpoints" / f"{tag}.pt")
        meta = {"variant": args.variant, "seed": args.seed, "steps": agent.steps, "final": final, "created_at": now(),
                "protocol_sha256": run["protocol_sha256"], "features_version": env.info.get("features_version"),
                "reference_loop_sha256": env.info.get("reference_loop_sha256"), "speed_cap_mps": env.info.get("speed_cap_mps")}
        pj = out / "policies" / f"{tag}.json"
        atomic_write_bytes(pj, json.dumps(export_policy_json(agent.model, agent.rms, cfg, meta)).encode())
        if not args.no_eval:
            evaluator.submit(pj)

    next_ckpt = (agent.steps // args.ckpt_every + 1) * args.ckpt_every
    next_save = (agent.steps // args.save_every + 1) * args.save_every
    t_start, s_start = time.time(), agent.steps
    try:
        with open(out / "progress.jsonl", "a") as plog:
            while agent.steps < args.steps and not stop_flag["stop"]:
                t0 = time.time()
                batch, episodes, rstats = agent.collect()
                t1 = time.time()
                ustats = agent.update(batch)
                t2 = time.time()
                rec = {"t": now(), "steps": agent.steps, "update": agent.updates, "collect_s": t1 - t0, "update_s": t2 - t1,
                       "steps_per_s": (args.workers * args.envs_per_worker * cfg.rollout_steps) / (t2 - t0),
                       "steps_per_s_run": (agent.steps - s_start) / (t2 - t_start), **ustats, **rstats, "episodes": summarize_episodes(episodes)}
                if args.dr:
                    # r2: 標準の物理の回と、ずらした回を分けても集計する(学習曲線用)
                    rec["episodes_nominal"] = summarize_episodes([e for e in episodes if not e["start"].get("perturb")])
                    rec["episodes_perturbed"] = summarize_episodes([e for e in episodes if e["start"].get("perturb")])
                    # r3: 遅い舵(≤ 1.0 rad/s)× 弱いブレーキ(≤ 0.9 m/s²)の角の回(確認 r2 の must-fix の場所)
                    rec["episodes_slow_steer_weak_brake"] = summarize_episodes([e for e in episodes if is_slow_weak(e)])
                plog.write(json.dumps(rec) + "\n")
                plog.flush()
                if agent.updates % 10 == 0:
                    e = rec["episodes"]
                    print(f"[{now()}] {agent.steps:,} ステップ {rec['steps_per_s']:.0f}/s 回 {e.get('n')} 報酬 {e.get('return_mean', float('nan')):.2f} "
                          f"20 秒 {e.get('survived_20s_frac', 0):.2f} 2 周有効 {e.get('two_valid_laps_frac', 0):.2f} "
                          f"1 周目 {e.get('lap1', {}).get('mean')} フライング {e.get('flying', {}).get('mean')} "
                          f"KL {rec['approx_kl']} σ {np.exp(rec['log_std']).round(3).tolist()}", flush=True)
                if agent.steps >= next_save:
                    save_resume(last)
                    next_save = (agent.steps // args.save_every + 1) * args.save_every
                if agent.steps >= next_ckpt and agent.steps < args.steps:
                    checkpoint()
                    next_ckpt = (agent.steps // args.ckpt_every + 1) * args.ckpt_every
                evaluator.poll()
        save_resume(last)
        if agent.steps >= args.steps and not (out / "done.json").exists():
            if not list((out / "checkpoints").glob(f"step_{agent.steps:09d}.pt")):
                checkpoint(final=True)
            if not args.no_eval:
                evaluator.wait_all()
            atomic_json(out / "done.json", {"finished_at": now(), "steps": agent.steps, "wall_s_this_session": time.time() - t_start})
            print(f"[{now()}] 終わりました: {agent.steps:,} ステップ", flush=True)
        elif not args.no_eval:
            evaluator.wait_all()
    finally:
        env.close()
        if stop_flag["stop"]:
            evaluator.stop()
        else:
            evaluator.wait_all()


if __name__ == "__main__":
    main()
