"""no_wall r4b: 確認 r3 の組の走らせ直し(reg_jobs_nw4b.py + r3/ck_nw3/run_ck3.mjs)をまとめる(2026-10-14 20 時、電源断のあとの再開で足した)。
同じ組の確認 r3 の結果(r3 の候補 = 方策 3b099f9b… + シールド v4.1)と並べ、シールドあり / なしの対で救った・起こしたを数える。報告だけ(判定には使わない)。

    python3 summ_reg_nw4b.py <走らせ直しのフォルダー(rows_*.jsonl)> <出力の summary.json>
"""
import glob
import json
import sys
from pathlib import Path

CK = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_no_wall_r3")
d, outp = Path(sys.argv[1]), Path(sys.argv[2])


def load(paths):
    rows = {}
    for p in paths:
        for l in open(p):
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                rows[r["id"]] = r
    return list(rows.values())


def nominal(r):
    return not r.get("phys") and not r.get("delay") and (r.get("sg", 1) == 1) and (r.get("vg", 1) == 1) and not r.get("fc")


def key(r):
    s = r["start"]
    return json.dumps([s["gate"], round(s["off"], 6), round(s["yaw"], 6), r.get("phys"), r.get("delay", 0), r.get("sg", 1), r.get("vg", 1), bool(r.get("fc")), r.get("T", 20)],
                      sort_keys=True)


def table(rows):
    out = {}
    sets = sorted({r["set"] for r in rows})
    for s in sets:
        R = [r for r in rows if r["set"] == s and not r.get("error")]
        E = [r for r in rows if r["set"] == s and r.get("error")]
        if not R:
            out[s] = {"n": 0, "errors": len(E)}
            continue
        fbs = {}
        for r in R:
            for k, v in (r.get("fb") or {}).items():
                fbs[k] = fbs.get(k, 0) + v
        mins = sorted(R, key=lambda r: r["min_clearance"])
        out[s] = {"n": len(R), "errors": len(E), "nominal": all(nominal(r) for r in R),
                  "collisions": sum(1 for r in R if r["collisions"]),
                  "collision_ids": [r["id"] for r in R if r["collisions"]],
                  "min_clearance_m": mins[0]["min_clearance"], "min_where": mins[0]["id"],
                  "below_8cm": sum(1 for r in R if r["min_clearance"] < .08 and not r["collisions"]),
                  "below_8cm_ids": [r["id"] for r in R if r["min_clearance"] < .08 and not r["collisions"]][:20],
                  "below_5cm": sum(1 for r in R if r["min_clearance"] < .05 and not r["collisions"]),
                  "stalls": sum(1 for r in R if r["stall"]), "stall_ids": [r["id"] for r in R if r["stall"]][:20],
                  "valid_episode_r3": sum(1 for r in R if r["valid_episode_r3"]),
                  "shield_fraction_mean": sum(r["shield_fraction"] for r in R) / len(R),
                  "shield_fraction_max": max(r["shield_fraction"] for r in R),
                  "escape_frames": sum(v for k, v in fbs.items() if k.startswith("escape")),
                  "runs_with_escape": sum(1 for r in R if any(k.startswith("escape") for k in (r.get("fb") or {}))),
                  "fb_totals": fbs,
                  "lap1_median_s": sorted(r["first_lap"] for r in R if r.get("first_lap"))[len([r for r in R if r.get("first_lap")]) // 2] if any(r.get("first_lap") for r in R) else None,
                  "pos_err_max": max(r["pos_err"] for r in R), "overlap_frames": sum(r["overlap_frames"] for r in R),
                  "l0_frames": sum(r["l0"] for r in R), "bad_keys": sum(r["bad_keys"] for r in R),
                  "ms_per_frame_median": sorted(r["ms_per_frame"] for r in R)[len(R) // 2]}
    return out


def pairs(rows, on, off):
    A = {key(r): r for r in rows if r["set"] == on and not r.get("error")}
    B = {key(r): r for r in rows if r["set"] == off and not r.get("error")}
    ks = sorted(set(A) & set(B))
    res = {"pairs": len(ks), "off_collisions": 0, "on_collisions": 0, "saved": 0, "caused": 0, "caused_ids": [], "off_below_8cm": 0, "on_below_8cm": 0,
           "on_stalls_off_not": 0}
    for k in ks:
        a, b = A[k], B[k]
        res["off_collisions"] += bool(b["collisions"])
        res["on_collisions"] += bool(a["collisions"])
        res["saved"] += bool(b["collisions"]) and not a["collisions"]
        if a["collisions"] and not b["collisions"]:
            res["caused"] += 1
            res["caused_ids"].append(a["id"])
        res["off_below_8cm"] += b["min_clearance"] < .08 or bool(b["collisions"])
        res["on_below_8cm"] += a["min_clearance"] < .08 or bool(a["collisions"])
        res["on_stalls_off_not"] += bool(a["stall"]) and not b["stall"]
    return res


new = load(glob.glob(str(d / "rows_*.jsonl")))
old = load(glob.glob(str(CK / "runs/rows_*.jsonl")) + glob.glob(str(CK / "runs_nbr/rows_*.jsonl")) + glob.glob(str(CK / "runs_x/rows_*.jsonl")))
want = {"LOWC", "HCALL", "HCOFF", "NBR", "EDGE2", "REG-K0", "REG-K7", "REG-KS", "REG-IN2", "REG-VX"}
old = [r for r in old if r["set"] in want]
S = {"what": "確認 r3 の組を run_ck3.mjs(変えない)で選んだ方策 + シールド v5 に走らせた結果と、確認 r3 の結果(r3 の候補 + v4.1)。報告だけ",
     "policy_sha": sorted({r["policy_sha"] for r in new if r.get("policy_sha")}),
     "new": table(new), "check_r3": table(old),
     "pairs_new": {"NBR": pairs(new, "NBR", "NBR-OFF"), "LOWC": pairs(new, "LOWC", "LOWC-OFF"), "HCALL": pairs(new, "HCALL", "HCOFF")},
     "pairs_check_r3": {"HCALL": pairs(old, "HCALL", "HCOFF")},
     "check_r3_failure_starts": {}}
# 確認 r3 の崩れの開始(衝突・5.63 cm・止まり)を名前で引く
for name, pred in [("NBR 衝突(ゲート 20・−0.13・−13°)", lambda r: r["set"] == "NBR" and r["start"]["gate"] == 20 and abs(r["start"]["off"] + .13) < 1e-6 and abs(r["start"]["yaw"] + 13) < 1e-6),
                   ("NBR 5.63 cm(ゲート 20・−0.124・−13.0°)", lambda r: r["set"] == "LOWC" and r["start"]["gate"] == 20 and abs(r["start"]["yaw"] + 13) < .05 and abs(r["start"]["off"] + .124) < .001)]:
    S["check_r3_failure_starts"][name] = {"new": [{k: r[k] for k in ("id", "collisions", "min_clearance", "stall", "valid_laps", "shield_fraction")} for r in new if pred(r)],
                                          "check_r3": [{k: r[k] for k in ("id", "collisions", "min_clearance", "stall", "valid_laps", "shield_fraction")} for r in old if pred(r)]}
old_stall_ids = [r["id"] for r in old if r["stall"] and r["set"] in ("LOWC", "EDGE2")]
S["check_r3_stall_starts"] = {"ids": old_stall_ids,
                              "new": [{k: r[k] for k in ("id", "collisions", "min_clearance", "stall", "valid_laps", "shield_fraction")} | {"escape": sum(v for kk, v in (r.get("fb") or {}).items() if kk.startswith("escape"))}
                                      for r in new if r["id"] in old_stall_ids]}
outp.write_text(json.dumps(S, ensure_ascii=False, indent=1) + "\n")
for s, t in S["new"].items():
    if t.get("n"):
        print(f"{s:9s} n {t['n']:4d} err {t['errors']} coll {t['collisions']} min {100 * t['min_clearance_m']:.2f} <8 {t['below_8cm']} stall {t['stalls']} v3 {t['valid_episode_r3']} "
              f"sh {t['shield_fraction_mean']:.3f} esc {t['runs_with_escape']}")
print("対:", json.dumps(S["pairs_new"], ensure_ascii=False))
print("確認 r3 の止まりの開始:", json.dumps(S["check_r3_stall_starts"]["new"], ensure_ascii=False)[:800])
