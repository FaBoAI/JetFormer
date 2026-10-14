"""no_wall r3 のチェックポイントの選び(2026-10-14)。protocol_no_wall_r3.json selection.rule:
検証の組(シールド v4 あり)で N1_r3〜N7_r3 の形を全部満たし止まり 0 のチェックポイントのうち一番後のもの。
無ければ満たす数が一番多く、同じなら標準の物理の最小すき間が大きいもの。試験の組は見ない。

    python3 select_nw3.py <no_wall_r3 の出力> → <出力>/selection.json
"""
import json
import sys
from pathlib import Path

O = Path(sys.argv[1])
keys = ["N1_r3", "N2_r3", "N3_r3", "N4", "N5_r3", "N6", "N7_r3", "stall_zero"]
rows = []
for ev in sorted(O.glob("val/step_*/eval_validation.json")):
    e = json.loads(ev.read_text())
    J = e["judged"]
    n = sum(1 for k in keys if J[k]["pass"])
    rows.append({"checkpoint": ev.parent.name, "steps": int(ev.parent.name.split("_")[1]), "policy": e["policy"]["path"], "policy_sha256": e["policy"]["sha256"],
                 "passed": n, "all": n == len(keys), "failed": [k for k in keys if not J[k]["pass"]],
                 "N2_min_nominal": J["N2_r3"]["value"], "N3_min_dr": J["N3_r3"]["value"], "collisions": J["N1_r3"]["collisions"], "stalls": J["stall_zero"]["value"],
                 "N5": J["N5_r3"]["value"], "N4": J["N4"]["value"], "N7": J["N7_r3"]["value"], "ui_text": e["reported"]["ui_text"]})
allpts = sorted(int(p.stem.split("_")[1]) for p in (O / "train/policies").glob("step_*.json"))
seen = {r["steps"] for r in rows}
cands = [r for r in rows if r["all"]]
if cands:
    sel = max(cands, key=lambda r: r["steps"])
    later_missing = [s for s in allpts if s > sel["steps"] and s not in seen]
    if later_missing:
        raise SystemExit(f"選んだ点より後に調べていない点があります: {later_missing}")
    why = "検証の組で全部(N1_r3〜N7_r3・止まり 0)を満たす点のうち一番後"
else:
    missing = [s for s in allpts if s not in seen]
    if missing:
        raise SystemExit(f"全部を満たす点が無いのに、調べていない点があります: {missing}")
    sel = max(rows, key=lambda r: (r["passed"], r["N2_min_nominal"]))
    why = "全部を満たす点が無いので、満たす数が一番多く、同じなら標準の物理の最小すき間が大きい点"
out = {"rule": "protocol_no_wall_r3.json selection.rule", "selected": sel, "why": why, "checkpoints": rows,
       "not_evaluated": [s for s in allpts if s not in seen],
       "not_evaluated_note": "選んだ点より前で調べていない点は、決まり(全部を満たす点のうち一番後)に関係しないので調べていない(時間の節約)"}
(O / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
for r in rows:
    print(r["checkpoint"], r["passed"], "/", len(keys), r["failed"], "N2", round(r["N2_min_nominal"] * 100, 2), "N3", round(r["N3_min_dr"] * 100, 2), "衝突", r["collisions"], "止まり", r["stalls"])
print("選んだ:", sel["checkpoint"], why)
