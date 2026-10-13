"""選んだ方策の JS(rl/policy_forward.mjs)と PyTorch(policy.pt・TorchScript)の行動の一致を、実際に訪れる 1 万状態で確かめる。

2026-10-13、RL の学習の段(protocol.json algorithm.deploy_policy: JS と Python の差 ≤ 1e-4、1 万状態)。

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python rl/check_js_torch.py <best のフォルダー(policy.json・policy.pt・policy_ts.pt がある)> [--out check.json]

状態の集め方: 64 環境(手順の開始の分布、学習用の終了)を、書き出した方策(決定的 + ゆらぎ σ 0.2)で走らせた観測 10,016 個。
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vec_env import NodeVecEnv  # noqa: E402
from ppo import policy_numpy_forward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("best", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=10_000)
    args = ap.parse_args()
    w = json.loads((args.best / "policy.json").read_text())
    variant = w["meta"].get("variant", "main")
    ts = torch.jit.load(str(args.best / "policy_ts.pt"))
    rng = np.random.default_rng(0)
    obs_all = []
    with NodeVecEnv(n_workers=2, envs_per_worker=32, seed=12345, variant=variant, mode="train") as env:
        o = env.reset()
        while sum(x.shape[0] for x in obs_all) < args.n:
            obs_all.append(o.copy())
            a = policy_numpy_forward(w, o) + 0.2 * rng.standard_normal((o.shape[0], 2))
            o, *_ = env.step(np.clip(a, -1, 1).astype(np.float32))
    X = np.concatenate(obs_all).astype(np.float32)
    with torch.no_grad():
        a_ts = ts(torch.from_numpy(X)).numpy().astype(np.float64)
    a_np32 = policy_numpy_forward(w, X, np.float32).astype(np.float64)
    with tempfile.TemporaryDirectory() as td:
        fo, fa = Path(td) / "obs.f32", Path(td) / "act.f64"
        X.tofile(fo)
        subprocess.run(["/usr/bin/node", str(HERE / "policy_check.mjs"), str(args.best / "policy.json"), str(fo), str(fa)], check=True,
                       capture_output=True)
        a_js = np.fromfile(fa, dtype=np.float64).reshape(-1, 2)
    res = {"states": int(X.shape[0]), "variant": variant, "policy_json": str(args.best / "policy.json"),
           "max_abs_diff_js_vs_torchscript": float(np.abs(a_js - a_ts).max()),
           "max_abs_diff_js_vs_numpy_float32": float(np.abs(a_js - a_np32).max()),
           "mean_abs_diff_js_vs_torchscript": float(np.abs(a_js - a_ts).mean()),
           "saturated_frac": float((np.abs(a_js) >= 1).mean()), "threshold": 1e-4}
    res["pass"] = res["max_abs_diff_js_vs_torchscript"] <= 1e-4 and res["max_abs_diff_js_vs_numpy_float32"] <= 1e-4
    out = args.out or (args.best / "js_torch_check.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
