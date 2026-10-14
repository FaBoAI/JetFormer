"""離脱復活の学習の段(r1): チェックポイントの選択(protocol.json selection.rule)。2026-10-13

検証の組(evals/recover/step_*.json、高速版)で judged を全部満たすチェックポイントのうち一番後のもの。無ければ満たす項目が一番多いもの
(同じなら主な数字 = 出来事の成功率 C2 が高いもの、それも同じなら後のもの)。結果を <run>/selection.json に書く。

    python -I rl/recover_r1/select_checkpoint.py --run ~/jetformer_reports/sim_rl_modes_20261013/train/recover/main
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted((a.run / "evals" / "recover").glob("step_*.json")):
        d = json.loads(f.read_text())
        J = d["judged"]
        passed = [k for k, v in J.items() if v.get("pass") is True]
        rows.append({"step": int(f.stem.split("_")[1]), "eval": str(f), "policy": str(a.run / "policies" / (f.stem + ".json")),
                     "passed": passed, "n_passed": len(passed), "all": bool(d.get("all_judged_pass")),
                     "C2_success": J["C2_success"]["value"], "C1_valid": J["C1_valid"]["value"],
                     "summary": {k: v.get("value") for k, v in J.items()}})
    if not rows:
        raise SystemExit("検証の結果がありません")
    allp = [r for r in rows if r["all"]]
    if allp:
        sel, how = max(allp, key=lambda r: r["step"]), "judged を全部満たす一番後のもの"
    else:
        sel = max(rows, key=lambda r: (r["n_passed"], r["C2_success"], r["step"]))
        how = "全部を満たすものが無いので、満たす項目が一番多いもの(同じなら C2 の成功率、それも同じなら後のもの)"
    out = {"selected_at": datetime.now().isoformat(timespec="seconds"), "rule": "protocol.json selection.rule", "how": how,
           "selected": sel, "candidates": rows}
    (a.run / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for r in rows:
        mark = "←選択" if r is sel else ""
        print(f"{r['step']:>10,} 満たす {r['n_passed']}/6 {','.join(r['passed'])} C1 {r['C1_valid']} C2 {r['C2_success']:.3f} {mark}")
    print("選んだ:", sel["policy"], how)


if __name__ == "__main__":
    main()
