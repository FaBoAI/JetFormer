"""障害物よけ(obstacle)の学習の段(r1): チェックポイントの選択(protocol.json selection.rule)。2026-10-14

検証の組(evals/obstacle/step_*.json、高速版、27 回 = 13 配置 × 2 開始 + OB-EMPTY)で judged を全部満たすチェックポイントのうち一番後のもの。
無ければ満たす項目が一番多いもの(同じなら主な数字 = 有効な回の数 O1、それも同じなら後のもの)。結果を --out に書く。

    python -I rl/ob/select_checkpoint.py --run ~/jetformer_reports/sim_rl_modes_20261013/train/obstacle/main \\
        --out ~/jetformer_reports/sim_rl_modes_20261013/train/obstacle/selection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted((a.run / "evals" / "obstacle").glob("step_*.json")):
        d = json.loads(f.read_text())
        J = d["judged"]
        passed = [k for k, v in J.items() if v.get("pass") is True]
        o1 = J["O1_valid"]["value"].split("/")
        pol = a.run / "policies" / (f.stem + ".json")
        rows.append({"checkpoint": f.stem, "step": int(f.stem.split("_")[1]), "eval": str(f), "policy": str(pol),
                     "policy_sha256": hashlib.sha256(pol.read_bytes()).hexdigest(), "passed": passed, "n_passed": len(passed), "n_judged": len(J),
                     "all_judged_pass": bool(d.get("all_judged_pass")), "valid": int(o1[0]), "runs": int(o1[1]),
                     "O2": J["O2_clearance"]["value"], "O3": J["O3_lap1_median"]["value"], "O4": J["O4_empty"]["value"]})
    if not rows:
        raise SystemExit("検証の結果がありません")
    allp = [r for r in rows if r["all_judged_pass"]]
    if allp:
        sel, how = max(allp, key=lambda r: r["step"]), "judged を全部満たす一番後のもの"
    else:
        sel = max(rows, key=lambda r: (r["n_passed"], r["valid"], r["step"]))
        how = "全部を満たすものが無いので、満たす項目が一番多いもの(同じなら有効な回の数 O1、それも同じなら後のもの)"
    out = {"schema": "jf-sim-rl-mode-selection/v1", "mode": "obstacle", "selected_at": datetime.now().isoformat(timespec="seconds"),
           "rule": "protocol selection.rule: 検証の組で judged を全部満たすチェックポイントのうち一番後のもの。無ければ満たす項目が一番多いもの"
                   "(同じなら obstacle の主な数字 = 有効な回の数)",
           "validation": "検証の組(種 validation_base = 806000 + i、配置 validation-000〜012、Halton 10,001〜)、27 回、高速版(rl/eval_modes.mjs --set validation)",
           "how": how, "selected": sel["checkpoint"], "policy": sel["policy"], "policy_sha256": sel["policy_sha256"], "candidates": rows}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for r in rows:
        mark = " ←選択" if r is sel else ""
        o2 = f"{r['O2'] * 100:.1f} cm" if r["O2"] is not None else "—"
        o3 = f"{r['O3']:.2f} 秒" if r["O3"] is not None else "—"
        print(f"{r['step']:>11,} 満たす {r['n_passed']}/{r['n_judged']} 有効 {r['valid']}/{r['runs']} O2 {o2} O3 {o3} O4 {r['O4']}{mark}")
    print("選んだ:", sel["policy"], how)


if __name__ == "__main__":
    main()
