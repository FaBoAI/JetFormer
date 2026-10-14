"""速さいろいろ第 2 回のチェックポイントの選び方(protocol_speed_r2.json selection。2026-10-14、学習の段 speed r2)。

    python3 -I rl/speed_r2/select_speed_r2.py <run の出力>/evals/speed <出力 selection.json>

決まり: 検証の組 SP-V120-R2 + SP-STD3 で S1〜S5(S2 は第 2 回の数え方 = 全部の回、衝突はすき間 0)を全部満たすもののうち S2 の値が一番大きいもの
(同じなら後のもの)。全部を満たすものが無ければ、満たす数が一番多く、同じなら S2 の値が大きいもの(同じなら後のもの)。
評価の結果が protocol_speed_r2 の組で作られたこと(set validation・protocol_speed_r2_sha256・set_spec.count 120)を確かめる。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
R2 = REPORTS / "protocol_speed_r2.json"
KEYS = ["S1_valid", "S2_clearance", "S3_lap_order", "S4_speed_order", "S5_vm4"]


def main() -> None:
    ev_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    r2_sha = hashlib.sha256(R2.read_bytes()).hexdigest()
    want = (REPORTS / "protocol_speed_r2.sha256").read_text().split()[0]
    if r2_sha != want:
        raise SystemExit(f"protocol_speed_r2.json の sha256 {r2_sha} が固めた値 {want} と違います")
    rows = []
    for p in sorted(ev_dir.glob("step_*.json")):
        ev = json.loads(p.read_text())
        if ev.get("set") != "validation" or ev.get("protocol_speed_r2_sha256") != r2_sha or (ev.get("set_spec") or {}).get("count") != 120:
            raise SystemExit(f"{p} は第 2 回の検証の組の結果ではありません")
        J = ev["judged"]
        passed = [k for k in KEYS if J[k]["pass"] is True]
        s3 = J["S3_lap_order"]["value"]
        rows.append({"step": int(p.stem.split("_")[1]), "policy": ev["policy"]["path"], "policy_sha256": ev["policy"]["sha256"],
                     "n_pass": len(passed), "all_pass": len(passed) == len(KEYS), "passed": passed,
                     "S1": J["S1_valid"]["value"], "S2": J["S2_clearance"]["value"], "S2_below_3cm": J["S2_clearance"].get("below_3cm"),
                     "S2_below_5cm": len(J["S2_clearance"].get("below_5cm") or []), "collided": J["S2_clearance"].get("collided"),
                     "S3": s3, "S3_diff": (s3["t5"] - s3["t1"]) if s3.get("t5") and s3.get("t1") else None,
                     "S4_mean_speed": [round(x["mean_speed"], 4) if x["mean_speed"] is not None else None for x in J["S4_speed_order"]["value"]],
                     "S5": J["S5_vm4"]["value"], "std3_min_clearance": ev["reported"].get("std3_min_clearance")})
    if not rows:
        raise SystemExit("評価の結果がありません")
    full = [r for r in rows if r["all_pass"]]
    key = lambda r: (r["S2"] if r["S2"] is not None else -1.0, r["step"])  # noqa: E731
    if full:
        best, how = max(full, key=key), "S1〜S5 を全部満たすもののうち S2 が一番大きいもの(同じなら後のもの)"
    else:
        top = max(r["n_pass"] for r in rows)
        best, how = max([r for r in rows if r["n_pass"] == top], key=key), f"全部を満たすものが無いので、満たす数 {top} のうち S2 が一番大きいもの"
    res = {"schema": "jf-sim-rl-speed-r2-selection/v1", "protocol_speed_r2_sha256": r2_sha, "rule": json.loads(R2.read_text())["selection"]["rule"],
           "how": how, "selected": best, "candidates": rows}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"selected_step": best["step"], "how": how, "S2": best["S2"], "all_pass": best["all_pass"]}, ensure_ascii=False))
    for r in rows:
        print(r["step"], r["n_pass"], r["passed"], r["S1"], r["S2"], r["S3_diff"], r["S5"], r["S2_below_5cm"])


if __name__ == "__main__":
    main()
