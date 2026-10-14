"""no_wall r4b のチェックポイントの選び(select_nw4.py の写しを protocol_no_wall_r4b.json selection に合わせたもの。2026-10-14 20 時、電源断のあとの再開で足した)。
候補: r4b の学習の点(train/policies、50 万ごと)・r4 の学習(種 24、100 万で止めた)の点(train_a/policies、val/a_step_*)・step_000000000(r3 の選んだ方策)。
順(「一番後」): r4c の点 > r4b の点(ステップの順)> r4 の点 > step_000000000(r4c は protocol_no_wall_r4c.json、21 時に足した)。
決まり(protocol_no_wall_r4.json selection.rule のまま): 検証の組(シールド v5 あり)で N1_r4〜N7_r4 の形を全部満たし止まり 0 の点のうち一番後のもの。
無ければ満たす数が一番多く、同じなら標準の物理の最小すき間が大きいもの。試験の組は見ない。

    python3 select_nw4b.py <no_wall_r4 の出力> → <出力>/selection.json
"""
import json
import sys
from pathlib import Path

O = Path(sys.argv[1])
keys = ["N1_r4", "N2_r4", "N3_r4", "N4", "N5_r4", "N6", "N7_r4", "stall_zero"]


def order(name):
    if name.startswith("c_step_"):
        return (3, int(name.split("_")[2]))
    if name.startswith("a_step_"):
        return (1, int(name.split("_")[2]))
    s = int(name.split("_")[1])
    return (2, s) if s > 0 else (0, 0)


rows = []
for ev in sorted(O.glob("val/*/eval_validation.json")):
    e = json.loads(ev.read_text())
    J = e["judged"]
    n = sum(1 for k in keys if J[k]["pass"])
    name = ev.parent.name
    rows.append({"checkpoint": name, "run": {3: "r4c(種 26、r4b の終わりから)", 2: "r4b(種 25)", 1: "r4(種 24、100 万で止めた)", 0: "学習の始まり = r3 の選んだ方策"}[order(name)[0]],
                 "order": list(order(name)), "steps": order(name)[1], "policy": e["policy"]["path"], "policy_sha256": e["policy"]["sha256"],
                 "passed": n, "all": n == len(keys), "failed": [k for k in keys if not J[k]["pass"]],
                 "N2_min_nominal": J["N2_r4"]["value"], "N3_min_dr": J["N3_r4"]["value"], "collisions": J["N1_r4"]["collisions"], "stalls": J["stall_zero"]["value"],
                 "stall_ids": J["stall_zero"].get("ids"), "N5": J["N5_r4"]["value"], "N4": J["N4"]["value"], "N7": J["N7_r4"]["value"], "ui_text": e["reported"]["ui_text"]})
allpts = [("step_%09d" % int(p.stem.split("_")[1])) for p in (O / "train/policies").glob("step_*.json")]
allpts += ["a_" + p.stem for p in (O / "train_a/policies").glob("step_*.json") if int(p.stem.split("_")[1]) > 0]
allpts += ["c_" + p.stem for p in (O / "train_c/policies").glob("step_*.json") if int(p.stem.split("_")[1]) > 0]
allpts = sorted(set(allpts), key=order)
seen = {r["checkpoint"] for r in rows}
cands = [r for r in rows if r["all"]]
if cands:
    sel = max(cands, key=lambda r: tuple(r["order"]))
    later_missing = [n for n in allpts if order(n) > tuple(sel["order"]) and n not in seen]
    if later_missing:
        raise SystemExit(f"選んだ点より後に調べていない点があります: {later_missing}")
    why = "検証の組で全部(N1_r4〜N7_r4・止まり 0)を満たす点のうち一番後(順: r4c の点 > r4b の点 > r4 の点 > 学習の始まり)"
else:
    missing = [n for n in allpts if n not in seen]
    if missing:
        raise SystemExit(f"全部を満たす点が無いのに、調べていない点があります: {missing}")
    # 満たす数・標準の物理の最小すき間がちょうど同じ点が並ぶ(シールドが c_nominal 0.081 m で止めるので、最小すき間は同じ回 VN117 の 8.1005 cm になりやすい)。
    # 決まりはその先を書いていないので、本ルールの「一番後」と同じ向きで、順が後の点を選ぶ(21:37、r4c の終わりの点と r4b の 2〜3 個の点の検証を見て、残りの点の検証の前に足した)
    sel = max(rows, key=lambda r: (r["passed"], r["N2_min_nominal"], tuple(r["order"])))
    ties = [r["checkpoint"] for r in rows if (r["passed"], r["N2_min_nominal"]) == (sel["passed"], sel["N2_min_nominal"])]
    why = ("全部を満たす点が無いので、満たす数が一番多く、同じなら標準の物理の最小すき間が大きい点"
           + (f"(同じ点 {len(ties)} 個: {', '.join(ties)}。決まりに書かれていない同点は順が一番後の点)" if len(ties) > 1 else ""))
rows.sort(key=lambda r: tuple(r["order"]))
out = {"rule": "protocol_no_wall_r4.json selection.rule(候補と順は protocol_no_wall_r4b.json・protocol_no_wall_r4c.json selection)", "selected": sel, "why": why, "checkpoints": rows,
       "not_evaluated": [n for n in allpts if n not in seen],
       "not_evaluated_note": "選んだ点より前(順)で調べていない点は、決まり(全部を満たす点のうち一番後)に関係しないので調べていない(時間の節約)"}
(O / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
for r in rows:
    print(r["checkpoint"], r["run"], r["passed"], "/", len(keys), r["failed"], "N2", round(r["N2_min_nominal"] * 100, 2), "N3", round(r["N3_min_dr"] * 100, 2),
          "衝突", r["collisions"], "止まり", r["stalls"], r["stall_ids"])
print("選んだ:", sel["checkpoint"], sel["policy"], why)
