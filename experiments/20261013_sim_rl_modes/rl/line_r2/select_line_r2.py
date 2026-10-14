"""経路いろいろ r2 のチェックポイントの選び方(protocol_line_r2.json selection)。

検証の組 LN-V60-R2 + LN-FAM6(高速版)で L1〜L6 を全部満たすチェックポイントのうち、L3 の 95% 点が一番小さいもの(同じなら後のもの)。
全部を満たすものが無ければ、満たす数が一番多く、同じなら L3 の 95% 点が小さいもの(同じなら後のもの)。初期化元は候補に入れない。

    python3 rl/line_r2/select_line_r2.py <学習の出力の場所> [--out selection.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

KEYS = ["L1_valid", "L2_clearance", "L3_tracking", "L4_distinct", "L5_vm3", "L6_lap"]


def row_of(p: Path) -> dict:
    d = json.loads(p.read_text())
    J = d["judged"]
    n_pass = sum(1 for k in KEYS if J.get(k, {}).get("pass") is True)
    l3 = J["L3_tracking"]["value"] or {}
    l4 = J["L4_distinct"].get("value") or {}
    return {"step": int(p.stem.split("_")[1]), "eval": str(p), "policy_sha256": d["policy"]["sha256"], "policy": d["policy"]["path"],
            "all_pass": bool(d["all_judged_pass"]), "n_pass": n_pass, "failed": [k for k in KEYS if J.get(k, {}).get("pass") is not True],
            "L1": J["L1_valid"]["value"], "L2": J["L2_clearance"]["value"], "L3_median": l3.get("median"), "L3_p95": l3.get("p95"),
            "L4_lr": l4.get("left_right_mean"), "L4_io": l4.get("inside_outside_mean"), "L4_pair": l4.get("min_pair_max_1m_avg"),
            "L5": J["L5_vm3"]["value"], "L6": J["L6_lap"]["value"]}


def select(rows: list[dict]) -> dict:
    p95 = lambda r: r["L3_p95"] if r["L3_p95"] is not None else 9.9  # noqa: E731
    ok = [r for r in rows if r["all_pass"]]
    if ok:
        best = sorted(ok, key=lambda r: (p95(r), -r["step"]))[0]
        return {"rule": "全部満たすもののうち L3 の 95% 点が一番小さいもの(同じなら後のもの)", "selected": best}
    best = sorted(rows, key=lambda r: (-r["n_pass"], p95(r), -r["step"]))[0]
    return {"rule": "全部満たすものが無いので、満たす数が一番多く、同じなら L3 の 95% 点が小さいもの", "selected": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = [row_of(p) for p in sorted((a.run_dir / "evals" / "line").glob("step_*.json"))]
    if not rows:
        raise SystemExit("検証の結果がありません")
    res = select(rows)
    sel = res["selected"]
    pol = Path(sel["policy"])
    res.update({"schema": "jf-sim-rl-line-r2-selection/v1", "created_at": datetime.now().isoformat(timespec="seconds"),
                "protocol_line_r2_sha256": hashlib.sha256(Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol_line_r2.json").read_bytes()).hexdigest(),
                "candidates": rows, "selected_checkpoint": str(pol.parent.parent / "checkpoints" / (pol.stem + ".pt"))})
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"rule": res["rule"], "step": sel["step"], "all_pass": sel["all_pass"], "failed": sel["failed"], "L3_p95": sel["L3_p95"], "L2": sel["L2"]}, ensure_ascii=False))
