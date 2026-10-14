"""no_wall r3 の学習(2026-10-14、r3 の学習の段)。rl/train_modes.py(sha256 は r3/no_wall_r3/sources_sha256.txt)の写しに run 'no_wall_r3' を足したもの。
    protocol_no_wall_r3.json training のとおり: 初期化 = no_wall r2 の選んだ点 step_004014080(方策 153c709c…)、種 23、最大 600 万ステップ、lr 5e-5 → 3e-5、
    環境 = r3/nw/train/env_nw3.mjs の run 'no_wall_r3'(物理の内訳・速さの見張り・開始・止まりの見張り)。tuning は使わない(決まりの値だけ)。
    検証の組の評価はここではしない(シールド v4 つきで r3/nw/run_sets_nw3.mjs が別に行う)。
    起動: r3/nw/run_train_nw3.sh(jf-heavy --mem 2G --lock rl -- nice -n 15。止めたあとは同じコマンドで last.pt から続く)
以下は元の説明。
強化学習モードの学習(PPO、CPU)。2026-10-13、sim_rl_modes の実装の段(design.md 5 章・protocol.json budget・selection・allowed_tuning)。

    cd ~/jetformer/experiments/20261013_sim_rl_modes
    rl/run_mode.sh no_wall-main no_wall 10000000            # jf-heavy --mem 2G --lock rl -- nice -n 15 で動かす包み
    # 直に: CUDA_VISIBLE_DEVICES= python -u rl/train_modes.py --run no_wall --steps 10000000 --out <出力>

--run: no_wall | recover | cond(line と speed の 1 本)| obstacle | speed(r2)| line(r2)| obstacle_r2・obstacle_r2b(r2)。種・初期化元は modes.json / protocol.json の値が既定。
出力(--out の下):
  run.json            設定・使った値(tuning を含む)・modes.json / protocol.json の sha256・初期化元・開始と再開の記録
  progress.jsonl      更新ごとの学習の値と、その間に終わったエピソードの集計
  last.pt             再開用(50 万ステップごと・終わるとき・止める合図のとき。原子的に置き換える)
  checkpoints/step_N.pt   100 万ステップごとの一式(再開・選び直しに使える)
  policies/step_N.json    そのときの決定的な方策(schema jf-sim-rl-mode-policy/v1。ブラウザの mode_runtime.mjs が読む)
  evals/<mode>/step_N.json  検証の組(protocol selection.validation_sets、高速版)の結果(--eval のとき。rl/eval_modes.mjs)
  done.json           予定のステップを終えた印
電源断・止める合図のあとは同じコマンドで last.pt から続く(環境はリセットし直すので完全には同じにならない。run.json の resumes に残す)。
"""
from __future__ import annotations

import os

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
from vec_env_nw3 import ModeVecEnv  # noqa: E402
from ppo_modes import PPO, PPOConfig, export_policy_json  # noqa: E402

REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
PROTOCOL = REPORTS / "protocol.json"
MODES_JSON = REPORTS / "modes.json"
NODE = os.environ.get("RL_NODE", "/usr/bin/node")
BESTLAP_RL = Path("/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl")
INIT = {
    "main_s1_10M": BESTLAP_RL / "main-s1/checkpoints/step_010010624.pt",
    "r2_dr_5M": BESTLAP_RL / "r2-dr-s11/checkpoints/step_005013504.pt",
    # 速さいろいろ r2(2026-10-14、protocol_speed_r2.json): cond の本番(種 41)の 401 万ステップ
    "cond_main_4M": REPORTS / "train/speed/cond-main/checkpoints/step_004014080.pt",
    # 経路いろいろ r2(2026-10-14、protocol_line_r2.json): 第 1 回の作り直しの選んだ方策(cond-retry の 501 万)
    "line_r1_retry_5M": REPORTS / "train/line/cond-retry/checkpoints/step_005013504.pt",
    # no_wall r3(2026-10-14、protocol_no_wall_r3.json training.init): r2 の選んだ点(方策 153c709c…)
    "no_wall_r2_4M": REPORTS / "train/no_wall/r2/checkpoints/step_004014080.pt",
}
# no_wall r3 の追補の決まり(protocol.json は変えない。読むときに r3/protocol_r3.sha256 と照合する)
PROTOCOL_NW_R3 = REPORTS / "r3/protocol_no_wall_r3.json"
PROTOCOL_R3_SHA = REPORTS / "r3/protocol_r3.sha256"
# 速さいろいろ r2 の追補の決まり(protocol.json は変えない。この追補の sha256 も run.json に残す)
PROTOCOL_SPEED_R2 = REPORTS / "protocol_speed_r2.json"
# 経路いろいろ r2 の追補の決まり(同じく protocol.json は変えない)
PROTOCOL_LINE_R2 = REPORTS / "protocol_line_r2.json"
# 障害物よけ r2 の追補の決まり(同じく protocol.json は変えない。2026-10-14 学習の段 obstacle r2)
PROTOCOL_OBSTACLE_R2 = REPORTS / "protocol_obstacle_r2.json"
# 障害物よけ r2 の後半(obstacle_r2b)の追記の決まり(r2 の追補は変えずに、後半の変更だけを足す)
PROTOCOL_OBSTACLE_R2B = REPORTS / "protocol_obstacle_r2b.json"
RUNS = {  # run → (評価するモード, 観測の包み, 既定の初期化元, 既定の種)
    "no_wall": (["no_wall"], "base79", "r2_dr_5M", 21),
    "recover": (["recover"], "base79v", "main_s1_10M", 31),
    "cond": (["line", "speed"], "cond97", "main_s1_10M", 41),
    "obstacle": (["obstacle"], "obst95", "main_s1_10M", 61),
    # 速さいろいろ r2(2026-10-14): cond97 の方策を速さいろいろだけで学び直す(検証は --protocol speed_r2 の組)
    "speed": (["speed"], "cond97", "cond_main_4M", 43),
    # 経路いろいろ r2(2026-10-14): cond97 の方策を経路いろいろだけ・目標の線 v2 で学び直す(検証は --protocol line_r2 の組)
    "line": (["line"], "cond97", "line_r1_retry_5M", 44),
    # 障害物よけ r2(2026-10-14、protocol_obstacle_r2.json): obst95 のまま、学習の環境(env_modes.mjs trainConfig('obstacle_r2'))と正規化・探索を直して学び直す。種 62 = 61 + 1
    "obstacle_r2": (["obstacle"], "obst95", "main_s1_10M", 62),
    # 障害物よけ r2 の後半(2026-10-14、protocol_obstacle_r2b.json): r2 の前半のチェックポイント(--init に .pt のパス)から、進みの見張り・通り道の目標を足した環境で続ける
    "obstacle_r2b": (["obstacle"], "obst95", "main_s1_10M", 62),
    # no_wall r3(2026-10-14): r2 の点から種 23(protocol の外の作り直し。deviations.md)
    "no_wall_r3": (["no_wall"], "base79", "no_wall_r2_4M", 23),
}
# 書き出す方策の mode(無ければ run の名前。obstacle_r2 の方策は走行中は「障害物よけ」obstacle のもの)
EXPORT_MODE = {"obstacle_r2": "obstacle", "obstacle_r2b": "obstacle", "no_wall_r3": "no_wall"}
# 検証の評価に足す引数(run ごと)
EVAL_EXTRA = {"speed": ["--protocol", "speed_r2"], "line": ["--protocol", "line_r2"]}
# modes.json の tuning_ranges(1 回だけの調整はこの範囲の中。範囲の外は止める)
TUNING_KEYS = {
    # hard_dr・governor: no_wall の作り直し(r2、2026-10-14)で足した学習の条件(modes.json の範囲の外の鍵。deviations.md に記録)。範囲の検査は env_modes.mjs の値の形だけ
    "no_wall": ["smooth_steer", "smooth_speed", "traction_req_w", "friction_circle_w", "clearance_barrier_w", "clearance_barrier_start_m", "terminal_clearance_m",
                "hard_dr", "governor"],
    "recover": ["gate_miss_penalty", "magnitude_curriculum_max", "zone_width_m", "types_p"],
    "cond": ["tracking_w", "tracking_scale_m", "room_clearance_m"],
    "obstacle": ["obstacle_barrier_w", "count_per_layout_max"],
    "obstacle_r2": ["obstacle_barrier_w", "count_per_layout_max"],
    "obstacle_r2b": ["obstacle_barrier_w", "count_per_layout_max"],
    # speed(r2): protocol_speed_r2.json training.tuning_ranges の鍵(範囲もそこから読む)
    "speed": ["tracking_w", "tracking_scale_m", "clearance_barrier_w", "clearance_barrier_start_m", "terminal_clearance_m", "c_v_levels_p"],
    # line(r2): protocol_line_r2.json training.tuning_ranges の鍵(範囲もそこから読む)
    "line": ["progress_track_sigma_m", "tracking_w", "tracking_scale_m", "clearance_barrier_w", "clearance_barrier_start_m", "terminal_clearance_m"],
}


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


def check_tuning(run: str, tuning: dict) -> None:
    if run == "no_wall_r3":
        if tuning:
            raise SystemExit("no_wall_r3 は決まり(protocol_no_wall_r3.json)の値だけで学習します。tuning は使えません")
        return
    if run == "speed":
        ranges = json.loads(PROTOCOL_SPEED_R2.read_text())["training"]["tuning_ranges"]
    elif run == "line":
        ranges = json.loads(PROTOCOL_LINE_R2.read_text())["training"]["tuning_ranges"]
    else:
        modes = json.loads(MODES_JSON.read_text())
        key = "line" if run == "cond" else "obstacle" if run in ("obstacle_r2", "obstacle_r2b") else run
        ranges = next(m for m in modes["modes"] if m["key"] == key)["training"]["tuning_ranges"]
    for k, v in tuning.items():
        if k not in TUNING_KEYS[run]:
            raise SystemExit(f"tuning の {k} は {run} で変えられる値ではありません({TUNING_KEYS[run]})")
        r = ranges.get(k)
        if isinstance(r, list) and not (min(r) - 1e-12 <= v <= max(r) + 1e-12):
            raise SystemExit(f"tuning の {k}={v} が modes.json の範囲 {r} の外です")


def warmup_obs_stats(agent, env, steps_per_env: int, seed: int, lo: int, var_floor: float = 1e-2) -> dict:
    """新しい入力(lo 番目以降の次元)の観測の正規化の平均・分散を、初期化の方策(今の log σ の雑音つき)で環境を走らせて数え直す。
    (2026-10-14 学習の段 obstacle r2、protocol_obstacle_r2.json training.obs_norm。確認 obstacle r1 の指摘 3: widen_state_dict の平均 0・分散 1 が
    初期化元の大きな数え count のせいで学習中ほとんど動かなかった)。元の lo 次元と count はそのまま。新しい列の 1 層目の重みは 0 なので、初めの走りは変わらない。"""
    raw = env.reset(seed)
    A, C, log_std = agent._np_params()
    std = np.exp(log_std)
    rng = np.random.default_rng(seed)
    xs = [raw[:, lo:].astype(np.float64)]
    episodes = 0
    for _ in range(steps_per_env):
        mu = agent._np_mlp(A, agent.norm(raw))
        a = np.clip(mu + std * rng.standard_normal(mu.shape), -1.0, 1.0).astype(np.float32)
        raw, _, term, trunc, info = env.step(a)
        episodes += len(info["episodes"])
        xs.append(raw[:, lo:].astype(np.float64))
    X = np.concatenate(xs)
    before = {"mean": agent.rms.mean[lo:].tolist(), "var": agent.rms.var[lo:].tolist()}
    agent.rms.mean[lo:] = X.mean(0)
    agent.rms.var[lo:] = np.maximum(X.var(0), var_floor)
    return {"dims": [lo, int(agent.rms.mean.shape[0])], "steps_per_env": steps_per_env, "frames": int(X.shape[0]), "episodes_ended": episodes, "seed": seed,
            "var_floor": var_floor, "before": before, "mean": agent.rms.mean[lo:].tolist(), "var": agent.rms.var[lo:].tolist(),
            "raw_std": X.std(0).tolist(), "count_kept": float(agent.rms.count)}


def summarize(eps: list[dict]) -> dict:
    if not eps:
        return {"n": 0}
    reasons: dict[str, int] = {}
    for e in eps:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    lap1 = [e["laps"][0]["elapsed"] for e in eps if e["laps"] and e["laps"][0]["gates"] == 33 and e["laps"][0]["crossings"] == 1 and e["laps"][0]["collisions"] == 0]
    m = lambda k: float(np.mean([e[k] for e in eps if e.get(k) is not None])) if any(e.get(k) is not None for e in eps) else None  # noqa: E731
    n = len(eps)
    return {"n": n, "reasons": reasons, "return_mean": m("return"), "frames_mean": m("frames"), "progress_m_mean": m("progress_m"),
            "survived_20s_frac": reasons.get("collection_duration_complete", 0) / n, "lap1_mean": float(np.mean(lap1)) if lap1 else None,
            "lap1_n": len(lap1), "min_clearance_mean": m("min_clearance"), "injected_frames_mean": m("injected_frames"), "events_mean": m("events"),
            "gate_misses_mean": m("gate_misses"), "tracking_abs_mean": m("tracking_abs_mean"), "rho_comb_gt1_frac": m("rho_comb_gt1_frac"),
            "min_obstacle_clearance_mean": m("min_obstacle_clearance"), "perturb_scale": m("perturb_scale"),
            "reward_parts_mean": {k: float(np.mean([e["reward_parts"][k] for e in eps])) for k in eps[0]["reward_parts"]}}


class AsyncEvaluator:
    """チェックポイントごとの検証の組の評価(Node、rl/eval_modes.mjs)を裏で 1 つずつ走らせる。学習は止めない。"""

    def __init__(self, out: Path, modes: list[str], extra: list[str] | None = None):
        self.out, self.modes, self.extra = out, modes, list(extra or [])
        self.queue: list[tuple[str, Path]] = []
        self.proc = None
        self.current = None
        self.log = None

    def eval_path(self, mode: str, p: Path) -> Path:
        return self.out / "evals" / mode / p.name

    def submit(self, p: Path) -> None:
        for mode in self.modes:
            if not self.eval_path(mode, p).exists() and (mode, p) not in self.queue and (mode, p) != self.current:
                self.queue.append((mode, p))
        self.poll()

    def poll(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            if self.log:
                self.log.close()
            if self.proc.returncode != 0:
                print(f"[{now()}] 検証の評価が失敗しました(終了コード {self.proc.returncode}): {self.current}", flush=True)
            self.proc, self.current, self.log = None, None, None
        if self.proc is None and self.queue:
            mode, p = self.queue.pop(0)
            ev = self.eval_path(mode, p)
            ev.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(ev.with_suffix(".log"), "w")
            cmd = [NODE, str(HERE / "eval_modes.mjs"), "--mode", mode, "--set", "validation", "--core", "fast", "--policy", str(p), "--out", str(ev),
                   *self.extra]
            self.proc = subprocess.Popen(cmd, cwd=str(HERE.parent), stdout=self.log, stderr=subprocess.STDOUT)
            self.current = (mode, p)

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
    ap = argparse.ArgumentParser(description="強化学習モードの PPO の学習")
    ap.add_argument("--run", choices=list(RUNS), required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init", default=None, help="初期化元: main_s1_10M | r2_dr_5M | .pt のパス(既定は modes.json の値)")
    ap.add_argument("--tuning", default=None, help="1 回だけの調整(JSON の文字列かファイル)。modes.json の tuning_ranges の中だけ")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=1_000_000)
    ap.add_argument("--save-every", type=int, default=500_000)
    ap.add_argument("--eval", action="store_true", help="チェックポイントごとに検証の組を裏で評価する")
    ap.add_argument("--lr-start", type=float, default=None)
    ap.add_argument("--lr-end", type=float, default=None, help="下調べだけ(既定 3e-5。modes.json ppo_finetune)")
    ap.add_argument("--log-std-init", type=float, default=None, help="下調べだけ: 初期化元の log σ を使わず、この値で始める(modes.json は「初期化元から引き継ぐ」)")
    ap.add_argument("--rollout-steps", type=int, default=None, help="試しの短い学習だけ(既定 256)")
    ap.add_argument("--obs-warmup-steps", type=int, default=0,
                    help="obstacle r2: 新しい入力の正規化を、学習の前に初期化の方策で環境を 環境ごとにこのステップ数 走らせて数え直す(0 = しない。今までと同じ)")
    args = ap.parse_args()

    modes_eval, wrapper, init_default, seed_default = RUNS[args.run]
    seed = seed_default if args.seed is None else args.seed
    tuning = {}
    if args.tuning:
        tuning = json.loads(Path(args.tuning).read_text()) if Path(args.tuning).exists() else json.loads(args.tuning)
        check_tuning(args.run, tuning)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    out: Path = args.out
    for sub in ("checkpoints", "policies", "evals"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig()
    if args.lr_start is not None:
        cfg.lr_start = args.lr_start
    if args.lr_end is not None:
        cfg.lr_end = args.lr_end
    if args.rollout_steps is not None:
        cfg.rollout_steps = args.rollout_steps
    init_name = args.init or init_default
    init_path = INIT.get(init_name, Path(init_name))
    if not init_path.exists():
        print(f"[{now()}] 初期化元 {init_path} が読めないので main_s1_10M にします(modes.json no_wall.policy.init の決まり)", flush=True)
        init_name, init_path = "main_s1_10M", INIT["main_s1_10M"]

    run_path = out / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {
        "schema": "jf-sim-rl-mode-run/v1", "run": args.run, "modes": modes_eval, "obs_wrapper": wrapper, "seed": seed, "total_steps": args.steps,
        "tag": args.tag, "tuning": tuning, "init": init_name, "init_path": str(init_path), "init_sha256": sha256(init_path),
        "n_envs": args.workers * args.envs_per_worker, "workers": args.workers, "envs_per_worker": args.envs_per_worker, "ppo": cfg.to_dict(),
        "torch_threads": args.torch_threads, "torch": torch.__version__, "python": sys.version.split()[0],
        "protocol_sha256": sha256(PROTOCOL), "modes_json_sha256": sha256(MODES_JSON), "started_at": now(), "resumes": [], "argv": sys.argv,
        **({"protocol_speed_r2_sha256": sha256(PROTOCOL_SPEED_R2)} if args.run == "speed" else {}),
        **({"protocol_line_r2_sha256": sha256(PROTOCOL_LINE_R2)} if args.run == "line" else {}),
        **({"protocol_obstacle_r2_sha256": sha256(PROTOCOL_OBSTACLE_R2)} if args.run in ("obstacle_r2", "obstacle_r2b") else {}),
        **({"protocol_obstacle_r2b_sha256": sha256(PROTOCOL_OBSTACLE_R2B)} if args.run == "obstacle_r2b" else {})}
    if args.run == "no_wall_r3":
        raw = PROTOCOL_NW_R3.read_bytes()
        want = next(l.split()[0] for l in PROTOCOL_R3_SHA.read_text().splitlines() if l.strip().endswith("protocol_no_wall_r3.json"))
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            raise SystemExit(f"protocol_no_wall_r3.json の sha256 {got} が固めた値 {want} と違います")
        run.setdefault("protocol_no_wall_r3_sha256", got)
        prot = json.loads(raw)
        if seed != prot["training"]["seed"] or args.steps > prot["training"]["steps_max"]:
            raise SystemExit("種・ステップ数が protocol_no_wall_r3.json training と違います")
        run.setdefault("code_sha256", {p.name: sha256(p) for p in sorted(HERE.glob("*")) if p.suffix in (".py", ".mjs")})
    if run.get("total_steps") != args.steps or run.get("run") != args.run:
        raise SystemExit(f"run.json({run.get('run')}, {run.get('total_steps')})と引数({args.run}, {args.steps})が違います")
    if run.get("tuning", {}) != tuning:
        raise SystemExit(f"run.json の tuning {run.get('tuning')} と引数 {tuning} が違います(続きは同じ値で)")

    env = ModeVecEnv(args.run, n_workers=args.workers, envs_per_worker=args.envs_per_worker, seed=seed, tuning=tuning)
    run["env"] = env.info
    agent = PPO(env, cfg, total_steps=args.steps, seed=seed, obs_dim=env.obs_dim)
    evaluator = AsyncEvaluator(out, modes_eval, EVAL_EXTRA.get(args.run))
    last = out / "last.pt"
    if last.exists():
        d = torch.load(last, map_location="cpu", weights_only=False)
        agent.load_state_dict(d)
        reset_seed = (seed * 1_000_003 + agent.steps) & 0x7FFFFFFF
        run["resumes"].append({"at": now(), "steps": agent.steps, "env_reset_seed": reset_seed})
        print(f"[{now()}] last.pt から再開します: {agent.steps:,} ステップ", flush=True)
        agent.start(reset_seed)
    else:
        d = torch.load(init_path, map_location="cpu", weights_only=False)
        agent.init_from_checkpoint(d, log_std_init=args.log_std_init)
        if args.log_std_init is not None:
            run["log_std_override"] = args.log_std_init
        run["init_steps"] = int(d.get("steps", 0))
        run["init_log_std"] = agent.model.log_std.detach().tolist()
        init_dim = len(d["rms"]["mean"])
        if args.obs_warmup_steps > 0 and init_dim < env.obs_dim:
            ws = warmup_obs_stats(agent, env, args.obs_warmup_steps, (seed * 7_919 + 104_729) & 0x7FFFFFFF, init_dim)
            run["obs_warmup"] = ws
            print(f"[{now()}] 新しい入力 {ws['dims']} の正規化を数え直しました({ws['frames']:,} フレーム): 平均 {np.round(ws['mean'], 3).tolist()} "
                  f"σ {np.round(np.sqrt(ws['var']), 3).tolist()}", flush=True)
        agent.start(seed)
    atomic_json(run_path, run)
    if args.eval:
        for p in sorted((out / "policies").glob("step_*.json")):
            evaluator.submit(p)

    stop = {"stop": False}

    def on_term(signum, frame):
        stop["stop"] = True
        print(f"[{now()}] 止める合図を受けました。区切りで保存して終わります", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def checkpoint(final: bool = False) -> None:
        tag = f"step_{agent.steps:09d}"
        atomic_torch_save(out / "checkpoints" / f"{tag}.pt", agent.state_dict())
        meta = {"run": args.run, "modes": modes_eval, "seed": seed, "steps": agent.steps, "final": final, "created_at": now(), "init": init_name,
                "tuning": tuning, "protocol_sha256": run["protocol_sha256"], "modes_json_sha256": run["modes_json_sha256"],
                "features_version": env.info.get("features_version"), "reference_loop_sha256": env.info.get("reference_loop_sha256")}
        # 目標の線の版(line r2 = 'v2')。走行中の mode_runtime.mjs は meta.line_profile を見て同じ線を使う。無い run は今までと同じ meta
        lp = (env.info.get("train_config") or {}).get("line_profile")
        if lp:
            meta["line_profile"] = lp
        pj = out / "policies" / f"{tag}.json"
        atomic_write_bytes(pj, json.dumps(export_policy_json(agent.model, agent.rms, cfg, meta, wrapper, EXPORT_MODE.get(args.run, args.run))).encode())
        if args.eval:
            evaluator.submit(pj)

    next_ckpt = (agent.steps // args.ckpt_every + 1) * args.ckpt_every
    next_save = (agent.steps // args.save_every + 1) * args.save_every
    t_start, s_start = time.time(), agent.steps
    try:
        with open(out / "progress.jsonl", "a") as plog:
            while agent.steps < args.steps and not stop["stop"]:
                t0 = time.time()
                batch, episodes, rstats = agent.collect()
                t1 = time.time()
                ustats = agent.update(batch)
                t2 = time.time()
                rec = {"t": now(), "steps": agent.steps, "update": agent.updates, "collect_s": t1 - t0, "update_s": t2 - t1,
                       "steps_per_s": (agent.n * cfg.rollout_steps) / (t2 - t0), "steps_per_s_run": (agent.steps - s_start) / (t2 - t_start),
                       **ustats, **rstats, "episodes": summarize(episodes)}
                plog.write(json.dumps(rec) + "\n")
                plog.flush()
                if agent.updates % 10 == 0 or agent.steps >= args.steps:
                    e = rec["episodes"]
                    print(f"[{now()}] {agent.steps:,} ステップ {rec['steps_per_s']:.0f}/s 回 {e.get('n')} 報酬 {e.get('return_mean')} "
                          f"20 秒 {e.get('survived_20s_frac')} 1 周目 {e.get('lap1_mean')} 最小すき間 {e.get('min_clearance_mean')} "
                          f"理由 {e.get('reasons')} KL {rec['approx_kl']} σ {np.exp(rec['log_std']).round(3).tolist()}", flush=True)
                if agent.steps >= next_save:
                    atomic_torch_save(last, agent.state_dict())
                    next_save = (agent.steps // args.save_every + 1) * args.save_every
                if agent.steps >= next_ckpt and agent.steps < args.steps:
                    checkpoint()
                    next_ckpt = (agent.steps // args.ckpt_every + 1) * args.ckpt_every
                evaluator.poll()
        atomic_torch_save(last, agent.state_dict())
        if agent.steps >= args.steps and not (out / "done.json").exists():
            if not (out / "checkpoints" / f"step_{agent.steps:09d}.pt").exists():
                checkpoint(final=True)
            if args.eval:
                evaluator.wait_all()
            atomic_json(out / "done.json", {"finished_at": now(), "steps": agent.steps, "wall_s_this_session": time.time() - t_start})
            print(f"[{now()}] 終わりました: {agent.steps:,} ステップ", flush=True)
    finally:
        env.close()
        if stop["stop"]:
            evaluator.stop()
        else:
            evaluator.wait_all()


if __name__ == "__main__":
    main()
