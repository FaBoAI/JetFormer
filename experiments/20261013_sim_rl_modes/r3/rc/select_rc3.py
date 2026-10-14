"""離脱復活 r3 のチェックポイントの選択(protocol_recover_r3.json selection.rule)。2026-10-14、学習の段 recover r3。

    python3 -I r3/rc/select_rc3.py --run <学習の出力>      → <run>/selection.json と <run>/../best/(policy.json・checkpoint.pt の写し)

決まり: 検証の組(シールドあり)で C1_r3・C2・C3・C4・C5_r3・C6_r3・C7 を全部満たすチェックポイントのうち一番後のもの。
無ければ満たす数が一番多く、同じなら C2 の成功率が高いもの(それも同じなら後のもの)。試験の結果を見て選び直さない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted((a.run / "evals" / "recover").glob("step_*.json")):
        d = json.loads(p.read_text())
        J = d["judged"]
        rows.append({"step": int(p.stem.split("_")[1]), "eval": str(p), "all": bool(d["all_judged_pass"]),
                     "n_passed": sum(1 for v in J.values() if v.get("pass")), "passed": [k for k, v in J.items() if v.get("pass")],
                     "C2": J["C2"]["value"], "summary": {k: v.get("value") for k, v in J.items()}})
    if not rows:
        raise SystemExit("検証の結果がありません")
    allp = [r for r in rows if r["all"]]
    if allp:
        sel = max(allp, key=lambda r: r["step"]); how = "全部を満たすもののうち一番後"
    else:
        sel = max(rows, key=lambda r: (r["n_passed"], r["C2"], r["step"])); how = "全部を満たすものが無いので、満たす数が一番多いもの(同じなら C2、それも同じなら後のもの)"
    pol = a.run / "policies" / f"step_{sel['step']:09d}.json"
    ck = a.run / "checkpoints" / f"step_{sel['step']:09d}.pt"
    best = a.run.parent / "best"
    best.mkdir(exist_ok=True)
    shutil.copy2(pol, best / "policy.json"); shutil.copy2(ck, best / "checkpoint.pt")
    out = {"selected_at": datetime.now().isoformat(timespec="seconds"), "rule": "protocol_recover_r3.json selection.rule", "how": how,
           "selected": {**sel, "policy": str(pol), "policy_sha256": hashlib.sha256(pol.read_bytes()).hexdigest()}, "candidates": rows}
    (a.run / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"step": sel["step"], "how": how, "passed": sel["passed"], "policy_sha256": out["selected"]["policy_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
