"""r2(手順の外の参考の追加実験)の方策を書き出す: rl/r2/dr/ に JSON の重み・ESM・PyTorch・TorchScript・再開用・出どころ。2026-10-13

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python -I rl/r2/export_dr.py [--step 最後]

報告する方策は前もって「最後のチェックポイント」と決めた(deviations.md 2026-10-13 21:25)。選定(rl/selection.json・rl/best/)は変えない。
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

RUN = A.RL / "r2-dr-s11"
DST = A.RL / "r2" / "dr"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None, help="省略すると最後のチェックポイント")
    args = ap.parse_args()
    pols = sorted((RUN / "policies").glob("step_*.json"))
    pol = pols[-1] if args.step is None else RUN / "policies" / f"step_{args.step:09d}.json"
    ck = RUN / "checkpoints" / (pol.stem + ".pt")
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pol, DST / "policy.json")
    w = json.loads(pol.read_text())
    (DST / "policy_weights.mjs").write_text("// RL の方策の重み(r2: 物理のずれのランダム化で学習を続けた参考の方策)。ブラウザの差し込みモジュール用の ESM\nexport default "
                                            + json.dumps(w) + ";\n")
    shutil.copy2(ck, DST / "checkpoint.pt")
    A.export_torch(ck, w, DST)
    ev = RUN / "evals" / pol.name
    if ev.exists():
        shutil.copy2(ev, DST / "eval_fast.json")
    rj = json.loads((RUN / "run.json").read_text())
    prov = {"run": "r2-dr-s11", "variant": rj["variant"], "seed": rj["seed"], "step_finetune": w["meta"]["steps"],
            "init_ckpt": rj.get("init_ckpt"), "init_ckpt_sha256": rj.get("init_ckpt_sha256"), "init_ckpt_steps": rj.get("init_ckpt_steps"),
            "dr": rj.get("dr"), "dr_spec": rj.get("env", {}).get("dr_spec"), "policy_json_sha256": sha256(DST / "policy.json"),
            "source_policy": str(pol), "source_checkpoint": str(ck), "reward_version": rj.get("env", {}).get("reward_version"),
            "features_version": rj.get("env", {}).get("features_version"), "reference_loop_sha256": rj.get("env", {}).get("reference_loop_sha256"),
            "speed_cap_mps": rj.get("env", {}).get("speed_cap_mps"), "protocol_sha256": rj.get("protocol_sha256"), "ppo": rj.get("ppo"),
            "control_delay_frames_trained": "0 か 1(回の 4 割が遅延 1。観測の前回の行動は適用した行動)",
            "note": "手順の外の参考の方策(記録・選定には使っていない)。JS は rl/policy_forward.mjs + policy.json、観測は rl/features.mjs"}
    (DST / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(prov, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
