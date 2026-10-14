"""経路いろいろ r4b のチェックポイントの選び方(protocol_line_r4b.json selection)。2026-10-14 夜、line r4 の段のやり直し。r3/line4/select_line_r4.py の写しを元に。

検証の組(LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-VG30R40-R4B + LN-FAM6、シールド v5b、高速版)で L1_r4b・L2〜L11 を
全部満たすチェックポイントのうち、むずかしい組(VHEAD40・VCORNER・VBOX40・VG30R40)の最小すき間(衝突・止まりの回は 0)が一番大きいもの。
0.1 mm に丸めて同じなら escape_run が少ないもの、それも同じなら L3 の 95% 点が小さいもの(それも同じなら後のもの)。
全部を満たすものが無ければ、満たす数が一番多く、同じならむずかしい組の失敗(衝突・止まり・有効な回の 3 cm 未満)が少ないもの。
候補: r4b の 50 万ごとのチェックポイント(r3/line_r4b/train/main/evals/line/step_*.json)と、r4 の 3,014,656・4,014,080
(r3/line_r4b/train/r4_ckpt/evals/r4_step_*.json)。

    python3 r3/line4b/select_line_r4b.py [--out r3/line_r4b/selection.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
KEYS = ["L1_r4", "L2_r4", "L3_tracking", "L4_distinct", "L5_vm3", "L6_lap", "L7_shield", "L8_no_collision", "L9_no_stall", "L10_no_reverse_P_FAM6", "L11_escape_r4b"]
P4B = REP / "r3/protocol_line_r4b.json"
P4B_SHA = REP / "r3/protocol_line_r4b.sha256"
RUN = REP / "r3/line_r4b/train"
R4_CK = REP / "r3/line_r4/train/main/checkpoints"


def row_of(p: Path, origin: str) -> dict:
    d = json.loads(p.read_text())
    if d.get("set") != "validation" or not d.get("protocol_line_r4b_sha256"):
        raise SystemExit(f"{p} は r4b の検証の組の結果ではありません")
    J, R = d["judged"], d["reported"]
    n_pass = sum(1 for k in KEYS if J.get(k, {}).get("pass") is True)
    l3 = J["L3_tracking"]["value"] or {}
    l4 = J["L4_distinct"].get("value") or {}
    step = int(p.stem.split("_")[-1])
    esc = sum(v.get("escape_runs", 0) for v in (R.get("by_set") or {}).values())
    return {"origin": origin, "step": step, "eval": str(p), "policy_sha256": d["policy"]["sha256"], "policy": d["policy"]["path"],
            "all_pass": bool(d["all_judged_pass"]), "n_pass": n_pass, "failed": [k for k in KEYS if J.get(k, {}).get("pass") is not True],
            "hard_min_clearance": R.get("hard_min_clearance"), "hard_fail": R.get("hard_fail"), "escape_runs": esc,
            "L1": J["L1_r4"]["value"], "L2": J["L2_r4"]["value"], "L3_median": l3.get("median"), "L3_p95": l3.get("p95"),
            "L4_lr": l4.get("left_right_mean"), "L4_io": l4.get("inside_outside_mean"), "L4_pair": l4.get("min_pair_max_1m_avg"),
            "L5": J["L5_vm3"]["value"], "L6": J["L6_lap"]["value"], "L7": J["L7_shield"]["value"], "L8": J["L8_no_collision"]["value"],
            "L9": J["L9_no_stall"]["value"], "L10": J["L10_no_reverse_P_FAM6"]["value"], "L11": J["L11_escape_r4b"]["value"],
            "by_set": R.get("by_set"), "shield_fraction": R.get("shield_fraction"), "shield_fallbacks": R.get("shield_fallbacks"),
            "shield_mjs_sha256": (d.get("shield") or {}).get("shield_mjs_sha256")}


def r1e4(x):
    return round(x, 4) if x is not None else -1.0


def select(rows: list[dict]) -> dict:
    p95 = lambda r: r1e4(r["L3_p95"]) if r["L3_p95"] is not None else 9.9  # noqa: E731
    order = lambda r: (0 if r["origin"] == "r4" else 1, r["step"])  # noqa: E731  (後のもの = r4b の後のチェックポイント)
    ok = [r for r in rows if r["all_pass"]]
    if ok:
        best = sorted(ok, key=lambda r: (-r1e4(r["hard_min_clearance"]), r["escape_runs"], p95(r), tuple(-x for x in order(r))))[0]
        return {"rule": "全部満たすもののうち、むずかしい組(VHEAD40・VCORNER・VBOX40・VG30R40)の最小すき間が一番大きいもの(0.1 mm で丸めて同じなら escape_run が少ないもの、"
                        "それも同じなら L3 の 95% 点が小さいもの、それも同じなら後のもの)", "selected": best}
    best = sorted(rows, key=lambda r: (-r["n_pass"], r["hard_fail"] if r["hard_fail"] is not None else 1e9, tuple(-x for x in order(r))))[0]
    return {"rule": "全部満たすものが無いので、満たす数が一番多く、同じならむずかしい組の失敗が少ないもの(同じなら後のもの)", "selected": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    want = P4B_SHA.read_text().split()[0]
    got = hashlib.sha256(P4B.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"protocol_line_r4b.json の sha256 {got} が固めた値 {want} と違います")
    rows = [row_of(p, "r4b") for p in sorted((RUN / "main/evals/line").glob("step_*.json"))]
    rows += [row_of(p, "r4") for p in sorted((RUN / "r4_ckpt/evals").glob("r4_step_*.json"))]
    want_n = 8 + 2
    if len(rows) != want_n:
        print(f"注意: 候補が {len(rows)} 個です(決まりでは {want_n} 個)")
    if not rows:
        raise SystemExit("検証の結果がありません")
    res = select(rows)
    sel = res["selected"]
    pol = Path(sel["policy"])
    ck = (pol.parent.parent / "checkpoints" / (pol.stem + ".pt")) if sel["origin"] == "r4b" else (R4_CK / (pol.stem + ".pt"))
    res.update({"schema": "jf-sim-rl-line-r4b-selection/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                "protocol_line_r4b_sha256": got, "candidates": rows, "selected_checkpoint": str(ck)})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"rule": res["rule"], "origin": sel["origin"], "step": sel["step"], "all_pass": sel["all_pass"], "failed": sel["failed"],
                      "hard_min_clearance": sel["hard_min_clearance"], "escape_runs": sel["escape_runs"], "L3_p95": sel["L3_p95"], "L2": sel["L2"],
                      "candidates": [[r["origin"], r["step"], r["all_pass"], r["n_pass"], r["hard_min_clearance"], r["hard_fail"], r["escape_runs"]] for r in rows]},
                     ensure_ascii=False))
