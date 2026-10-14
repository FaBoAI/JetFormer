"""r3(確認 r2 のあと。手順の外の参考の追加実験)の方策を書き出す: rl/r3/<run>/ に JSON の重み・ESM・PyTorch・TorchScript・再開用・出どころ。2026-10-13

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python -I rl/r3/export_r3.py --run r3-s21

報告する方策は、学習を始める前に「各種の最後のチェックポイント」と決めた(deviations.md 2026-10-13 22:40)。--step は使わない(途中は表に並べるだけ)。
選定(rl/selection.json・rl/best/)と記録の予定は変えない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_rl as A  # noqa: E402


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["r3-s21", "r3-s22"])
    args = ap.parse_args()
    run = A.RL / args.run
    if not (run / "done.json").exists():
        raise SystemExit(f"{run} はまだ終わっていません(done.json が無い)")
    pol = sorted((run / "policies").glob("step_*.json"))[-1]   # 最後のチェックポイント(前もって決めた)
    ck = run / "checkpoints" / (pol.stem + ".pt")
    dst = A.RL / "r3" / args.run
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pol, dst / "policy.json")
    w = json.loads(pol.read_text())
    (dst / "policy_weights.mjs").write_text(
        "// RL の方策の重み(r3: 角・辺を多めに引くランダム化 + すき間の約束で学習を続けた参考の方策)。ブラウザの差し込みモジュール用の ESM\nexport default "
        + json.dumps(w) + ";\n")
    shutil.copy2(ck, dst / "checkpoint.pt")
    A.export_torch(ck, w, dst)
    ev = run / "evals" / pol.name
    if ev.exists():
        shutil.copy2(ev, dst / "eval_fast.json")
    rj = json.loads((run / "run.json").read_text())
    env = rj.get("env", {})
    prov = {"run": args.run, "variant": rj["variant"], "seed": rj["seed"], "step_finetune": w["meta"]["steps"],
            "init_ckpt": rj.get("init_ckpt"), "init_ckpt_sha256": rj.get("init_ckpt_sha256"), "init_ckpt_steps": rj.get("init_ckpt_steps"),
            "dr": rj.get("dr"), "dr_spec": env.get("dr_spec"), "reward": rj.get("reward"), "reward_version": env.get("reward_version"),
            "reward_values": env.get("reward"), "policy_json_sha256": sha256(dst / "policy.json"), "source_policy": str(pol), "source_checkpoint": str(ck),
            "features_version": env.get("features_version"), "reference_loop_sha256": env.get("reference_loop_sha256"),
            "speed_cap_mps": env.get("speed_cap_mps"), "protocol_sha256": rj.get("protocol_sha256"), "ppo": rj.get("ppo"),
            "started_at": rj.get("started_at"), "resumes": rj.get("resumes"),
            "control_delay_frames_trained": "0 か 1(ずらす回の半分 = 全体の 4 割が遅延 1。観測の前回の行動は適用した行動)",
            "note": "手順の外の参考の方策(記録・選定には使っていない)。JS は rl/policy_forward.mjs + policy.json、観測は rl/features.mjs"}
    (dst / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: prov[k] for k in ("run", "step_finetune", "policy_json_sha256", "reward_version")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
