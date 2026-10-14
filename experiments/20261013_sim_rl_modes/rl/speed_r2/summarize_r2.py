"""速さいろいろ r2 の試験・報告の組の結果をまとめる(報告の表の数字。2026-10-14、学習の段 speed r2)。

    python3 -I rl/speed_r2/summarize_r2.py <eval の結果.json> [...] --out <summary.json>

c_v ごとの 1 周目・2 周目・平均の速さ・すき間(最小・中央値・3 cm / 5 cm 未満)・止まった回・衝突、
開始から 3 m 以降のすき間は eval の結果に無いので数えない(図のデータ figdata_r2 で数える)。
SP-F300-R2 のときは「60 回の組が S2 を通る見込み」(1 − 3 cm 未満の割合)^60 も出す。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def q(v, p):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(path: Path) -> dict:
    ev = json.loads(path.read_text())
    rows = ev["tests"]["SP-P60"]
    cl = lambda r: 0.0 if (r.get("invalid_spawn") or r.get("collisions") or r.get("min_clearance") is None) else r["min_clearance"]  # noqa: E731
    out = {"file": str(path), "set": ev["set"], "set_name": (ev.get("set_spec") or {}).get("name"), "core": ev["core"], "policy_sha256": ev["policy"]["sha256"],
           "runs": len(rows), "all_judged_pass": ev["all_judged_pass"],
           "judged": {k: {"pass": v["pass"], "value": v["value"]} for k, v in ev["judged"].items()}, "by_c_v": []}
    for cv in sorted({r["c_v"] for r in rows}):
        rs = [r for r in rows if r["c_v"] == cv]
        c = [cl(r) for r in rs]
        lap1 = [r["laps"][0] for r in rs if r.get("laps")]
        lap2 = [r["laps"][1] for r in rs if r.get("laps") and len(r["laps"]) > 1]
        sp = [r["speed"]["mean_running"] for r in rs if r.get("speed", {}).get("mean_running") is not None]
        out["by_c_v"].append({"c_v": cv, "n": len(rs), "valid": sum(1 for r in rs if r.get("valid_episode")), "collisions": sum(1 for r in rs if r.get("collisions")),
                              "stalled_no_lap": sum(1 for r in rs if r.get("valid_episode") and not r.get("laps")),
                              "lap1_mean": statistics.mean(lap1) if lap1 else None, "lap1_min": min(lap1) if lap1 else None, "lap1_max": max(lap1) if lap1 else None,
                              "lap2_mean": statistics.mean(lap2) if lap2 else None, "mean_speed": statistics.mean(sp) if sp else None,
                              "clear_min": min(c), "clear_median": q(c, .5), "below_3cm": sum(1 for x in c if x < .03), "below_5cm": sum(1 for x in c if x < .05),
                              "gate_miss": sum(1 for r in rs if r.get("gate_miss"))})
    c_all = [cl(r) for r in rows]
    n3 = sum(1 for x in c_all if x < .03)
    out["all"] = {"clear_min": min(c_all), "clear_median": q(c_all, .5), "below_3cm": n3, "below_5cm": sum(1 for x in c_all if x < .05),
                  "collisions": sum(1 for r in rows if r.get("collisions")), "stalled_no_lap": sum(1 for r in rows if r.get("valid_episode") and not r.get("laps")),
                  "gate_miss": sum(1 for r in rows if r.get("gate_miss")),
                  "p_pass_60": (1 - n3 / len(rows)) ** 60, "lap1_range": None}
    l1 = [r["laps"][0] for r in rows if r.get("valid_episode") and r.get("laps")]
    if l1:
        out["all"]["lap1_range"] = max(l1) - min(l1)
    out["lowest"] = [{"id": r["id"], "c_v": r["c_v"], "start": r["start"], "reason": r.get("reason"), "collisions": r.get("collisions"), "min_clearance": cl(r)}
                     for r in sorted(rows, key=cl)[:6]]
    out["std3"] = [{"id": r["id"], "lap_from_stop": r.get("lap_from_stop"), "min_clearance": r.get("min_clearance")} for r in ev["tests"].get("SP-STD3", [])]
    out["reported"] = {k: v for k, v in ev.get("reported", {}).items() if k in ("variation", "clearance_by_c_v", "std3_min_clearance")}
    return out


def main() -> None:
    args = sys.argv[1:]
    outp = Path(args[args.index("--out") + 1]) if "--out" in args else None
    files = [Path(a) for i, a in enumerate(args) if a != "--out" and (i == 0 or args[i - 1] != "--out")]
    res = [summarize(p) for p in files]
    for r in res:
        print(json.dumps({k: r[k] for k in ("set", "set_name", "runs", "all_judged_pass", "all")}, ensure_ascii=False))
        for b in r["by_c_v"]:
            print("  ", json.dumps(b, ensure_ascii=False))
    if outp:
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
