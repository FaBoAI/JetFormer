"""no_wall r2 のチェックポイントの選び(protocol selection.rule、2026-10-14)。

    python select_r2.py <run の出力(train/no_wall/r2)> <selection の出力 .json>

検証の組(evals/no_wall/step_*.json、高速版、シールド v3)で judged を全部満たすチェックポイントのうち一番後のもの。無ければ満たす項目が一番多いもの
(同じなら主な数字 = 最小すき間 min(N2, N3) が大きいもの)。
"""
import hashlib
import json
import sys
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
cands = []
for p in sorted((run / "evals" / "no_wall").glob("step_*.json")):
    ev = json.loads(p.read_text())
    J = ev["judged"]
    pol = run / "policies" / p.name
    n_pass = sum(1 for v in J.values() if v.get("pass") is True)
    cands.append({"checkpoint": p.stem, "steps": int(p.stem.split("_")[1]), "all_judged_pass": bool(ev.get("all_judged_pass")), "n_pass": n_pass,
                  "failed": [k for k, v in J.items() if v.get("pass") is False],
                  "N1": J["N1_no_collision"]["value"], "N2": J["N2_clearance_nominal"]["value"], "N3": J["N3_clearance_randomised"]["value"],
                  "N4": J["N4_lap_from_stop"]["value"], "N5": J["N5_completion"]["value"], "N6": J["N6_friction"]["value"], "N7": J["N7_shield"]["value"],
                  "shield": (ev.get("shield") or {}).get("version"), "R200_shield_fraction": ev.get("reported", {}).get("R200_shield_fraction"),
                  "policy": str(pol), "policy_sha256": hashlib.sha256(pol.read_bytes()).hexdigest()})
ok = [c for c in cands if c["all_judged_pass"]]
if ok:
    sel, why = max(ok, key=lambda c: c["steps"]), "検証の組で judged を全部満たすチェックポイントのうち一番後"
else:
    sel = max(cands, key=lambda c: (c["n_pass"], min(c["N2"] or 0, c["N3"] or 0), c["steps"]))
    why = "全部を満たすものが無いので、満たす項目が一番多いもの(同じなら最小すき間)"
res = {"schema": "jf-sim-rl-mode-selection/v1", "mode": "no_wall", "run": "r2", "rule": "protocol selection.rule", "why": why,
       "validation": "検証の組(種 validation_base、Halton 10,001〜、88 回)、高速版、シールド v3(rl/nw/eval_checkpoints.sh)",
       "selected": sel["checkpoint"], "policy": sel["policy"], "policy_sha256": sel["policy_sha256"], "candidates": cands}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
print(json.dumps({"selected": sel["checkpoint"], "why": why, "n": len(cands), "pass": [c["checkpoint"] for c in ok]}, ensure_ascii=False))
