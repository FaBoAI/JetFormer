"""経路いろいろ r3 のチェックポイントの選び方(protocol_line_r3.json selection)。2026-10-14、r3 の学習の段 line。

検証の組(LN-V60-R3 + LN-VHEAD40-R3 + LN-FAM6、シールドあり、高速版)で L1_r3〜L8 を全部満たすチェックポイントのうち、
LN-VHEAD40-R3 の最小すき間(全部の回。衝突した回は 0)が一番大きいもの。同じなら L3 の 95% 点が小さいもの。
全部を満たすものが無ければ、満たす数が一番多く、同じなら LN-VHEAD40-R3 の止まり + 衝突の数が少ないもの。初期化元は候補に入れない(学習の 100 万ごと)。
「同じ」は、学習の前に決めた丸め: すき間は 0.1 mm(1e-4 m)、L3 の 95% 点は 0.1 mm で丸めて比べる。それでも同じなら後のチェックポイント。

    python3 r3/line/select_line_r3.py <学習の出力の場所> [--out selection.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

KEYS = ["L1_r3", "L2_r3", "L3_tracking", "L4_distinct", "L5_vm3", "L6_lap", "L7_shield", "L8_no_collision"]
P3 = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_line_r3.json")


def row_of(p: Path) -> dict:
    d = json.loads(p.read_text())
    J, R = d["judged"], d["reported"]
    n_pass = sum(1 for k in KEYS if J.get(k, {}).get("pass") is True)
    l3 = J["L3_tracking"]["value"] or {}
    l4 = J["L4_distinct"].get("value") or {}
    return {"step": int(p.stem.split("_")[1]), "eval": str(p), "policy_sha256": d["policy"]["sha256"], "policy": d["policy"]["path"],
            "all_pass": bool(d["all_judged_pass"]), "n_pass": n_pass, "failed": [k for k in KEYS if J.get(k, {}).get("pass") is not True],
            "head_min_clearance": R.get("head_min_clearance_all"), "head_stall_plus_collision": R.get("head_stall_plus_collision"),
            "L1": J["L1_r3"]["value"], "L2": J["L2_r3"]["value"], "L3_median": l3.get("median"), "L3_p95": l3.get("p95"),
            "L4_lr": l4.get("left_right_mean"), "L4_io": l4.get("inside_outside_mean"), "L4_pair": l4.get("min_pair_max_1m_avg"),
            "L5": J["L5_vm3"]["value"], "L6": J["L6_lap"]["value"], "L7": J["L7_shield"]["value"], "L8": J["L8_no_collision"]["value"],
            "stalls": len(R.get("stalls") or []), "shield_fraction": R.get("shield_fraction")}


def r1e4(x):
    return round(x, 4) if x is not None else -1.0


def select(rows: list[dict]) -> dict:
    p95 = lambda r: r1e4(r["L3_p95"]) if r["L3_p95"] is not None else 9.9  # noqa: E731
    ok = [r for r in rows if r["all_pass"]]
    if ok:
        best = sorted(ok, key=lambda r: (-r1e4(r["head_min_clearance"]), p95(r), -r["step"]))[0]
        return {"rule": "全部満たすもののうち LN-VHEAD40-R3 の最小すき間が一番大きいもの(0.1 mm で丸めて同じなら L3 の 95% 点が小さいもの、それも同じなら後のもの)",
                "selected": best}
    best = sorted(rows, key=lambda r: (-r["n_pass"], r["head_stall_plus_collision"], -r["step"]))[0]
    return {"rule": "全部満たすものが無いので、満たす数が一番多く、同じなら LN-VHEAD40-R3 の止まり + 衝突が少ないもの(同じなら後のもの)", "selected": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = [row_of(p) for p in sorted((a.run_dir / "evals" / "line").glob("step_*.json"))]
    rows = [r for r in rows if r["step"] > 0]   # 初期化元(0 ステップ)は候補に入れない
    if not rows:
        raise SystemExit("検証の結果がありません")
    res = select(rows)
    sel = res["selected"]
    pol = Path(sel["policy"])
    res.update({"schema": "jf-sim-rl-line-r3-selection/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                "protocol_line_r3_sha256": hashlib.sha256(P3.read_bytes()).hexdigest(), "candidates": rows,
                "selected_checkpoint": str(pol.parent.parent / "checkpoints" / (pol.stem + ".pt"))})
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"rule": res["rule"], "step": sel["step"], "all_pass": sel["all_pass"], "failed": sel["failed"],
                      "head_min_clearance": sel["head_min_clearance"], "L3_p95": sel["L3_p95"], "L2": sel["L2"]}, ensure_ascii=False))
