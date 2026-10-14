"""経路いろいろ r4 のチェックポイントの選び方(protocol_line_r4.json selection)。2026-10-14、line r4 の段。r3/line/select_line_r3.py の写しを元に。

検証の組(LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-FAM6、シールド v5 あり、高速版)で L1_r4〜L10 を全部満たすチェックポイントのうち、
むずかしい組(VHEAD40・VCORNER・VBOX40)の最小すき間(衝突・止まりの回は 0)が一番大きいもの。0.1 mm に丸めて同じなら L3 の 95% 点が小さいもの
(それも同じなら後のチェックポイント)。全部を満たすものが無ければ、満たす数が一番多く、同じならむずかしい組の失敗(衝突・止まり・有効な回の 3 cm 未満)が少ないもの。
候補: 学習の 100 万ごとのチェックポイントと初期化元(step_000000000 = r3 の方策 + シールド v5)。

    python3 r3/line4/select_line_r4.py <学習の出力の場所> [--out selection.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

KEYS = ["L1_r4", "L2_r4", "L3_tracking", "L4_distinct", "L5_vm3", "L6_lap", "L7_shield", "L8_no_collision", "L9_no_stall", "L10_no_reverse_P_FAM6"]
P4 = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_line_r4.json")
P4_SHA = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_line_r4.sha256")


def row_of(p: Path) -> dict:
    d = json.loads(p.read_text())
    J, R = d["judged"], d["reported"]
    n_pass = sum(1 for k in KEYS if J.get(k, {}).get("pass") is True)
    l3 = J["L3_tracking"]["value"] or {}
    l4 = J["L4_distinct"].get("value") or {}
    return {"step": int(p.stem.split("_")[1]), "eval": str(p), "policy_sha256": d["policy"]["sha256"], "policy": d["policy"]["path"],
            "all_pass": bool(d["all_judged_pass"]), "n_pass": n_pass, "failed": [k for k in KEYS if J.get(k, {}).get("pass") is not True],
            "hard_min_clearance": R.get("hard_min_clearance"), "hard_fail": R.get("hard_fail"),
            "L1": J["L1_r4"]["value"], "L2": J["L2_r4"]["value"], "L3_median": l3.get("median"), "L3_p95": l3.get("p95"),
            "L4_lr": l4.get("left_right_mean"), "L4_io": l4.get("inside_outside_mean"), "L4_pair": l4.get("min_pair_max_1m_avg"),
            "L5": J["L5_vm3"]["value"], "L6": J["L6_lap"]["value"], "L7": J["L7_shield"]["value"], "L8": J["L8_no_collision"]["value"],
            "L9": J["L9_no_stall"]["value"], "L10": J["L10_no_reverse_P_FAM6"]["value"],
            "by_set": R.get("by_set"), "shield_fraction": R.get("shield_fraction"), "shield_ver": (d.get("shield") or {}).get("shield_mjs_sha256")}


def r1e4(x):
    return round(x, 4) if x is not None else -1.0


def select(rows: list[dict]) -> dict:
    p95 = lambda r: r1e4(r["L3_p95"]) if r["L3_p95"] is not None else 9.9  # noqa: E731
    ok = [r for r in rows if r["all_pass"]]
    if ok:
        best = sorted(ok, key=lambda r: (-r1e4(r["hard_min_clearance"]), p95(r), -r["step"]))[0]
        return {"rule": "全部満たすもののうち、むずかしい組(VHEAD40・VCORNER・VBOX40)の最小すき間が一番大きいもの(0.1 mm で丸めて同じなら L3 の 95% 点が小さいもの、それも同じなら後のもの)",
                "selected": best}
    best = sorted(rows, key=lambda r: (-r["n_pass"], r["hard_fail"] if r["hard_fail"] is not None else 1e9, -r["step"]))[0]
    return {"rule": "全部満たすものが無いので、満たす数が一番多く、同じならむずかしい組の失敗が少ないもの(同じなら後のもの)", "selected": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    want = P4_SHA.read_text().split()[0]
    got = hashlib.sha256(P4.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"protocol_line_r4.json の sha256 {got} が固めた値 {want} と違います")
    rows = [row_of(p) for p in sorted((a.run_dir / "evals" / "line").glob("step_*.json"))]
    if not rows:
        raise SystemExit("検証の結果がありません")
    res = select(rows)
    sel = res["selected"]
    pol = Path(sel["policy"])
    res.update({"schema": "jf-sim-rl-line-r4-selection/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                "protocol_line_r4_sha256": got, "candidates": rows,
                "selected_checkpoint": str(pol.parent.parent / "checkpoints" / (pol.stem + ".pt")) if sel["step"] > 0 else "初期化元(r3/line_r3/train/main/checkpoints/step_005013504.pt)"})
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"rule": res["rule"], "step": sel["step"], "all_pass": sel["all_pass"], "failed": sel["failed"],
                      "hard_min_clearance": sel["hard_min_clearance"], "L3_p95": sel["L3_p95"], "L2": sel["L2"],
                      "candidates": [[r["step"], r["all_pass"], r["n_pass"], r["hard_min_clearance"], r["hard_fail"]] for r in rows]}, ensure_ascii=False))
