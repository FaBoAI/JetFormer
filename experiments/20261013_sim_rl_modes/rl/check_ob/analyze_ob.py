"""障害物よけ(obstacle)の確認 r1 の集計(2026-10-14、確認の段)。

protocol.json acceptance.obstacle の文言(O1〜O4)・definitions(valid_episode・clearance・lap_from_stop)・variation_metrics(VM3 の式)から
自前で判定する。学習・評価の段のコード(eval_modes.mjs judge・variation_metrics.mjs・rl/ob/*)は使わない。

    python3 -I analyze_ob.py → check_obstacle_r1/summary.json
"""
import json
import math
import statistics as st
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
OUT = REP / "check_obstacle_r1"
RUNS = OUT / "runs"


def load(name):
    rows = []
    for p in sorted(RUNS.glob(f"{name}*.jsonl")):
        if name == "FRESH" and "FAST" in p.name:
            continue
        rows += [json.loads(line) for line in p.open()]
    return rows


def q(a, p):
    b = sorted(a)
    if not b:
        return None
    i = p * (len(b) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)


def judge(rows, empty_row):
    ok = [r for r in rows if r.get("valid_episode")]
    lap1 = [r["laps"][0]["elapsed"] for r in ok if r["laps"]]
    mc = min((r["min_clearance"] for r in ok), default=None)
    return {
        "O1_valid": {"value": f"{len(ok)}/{len(rows)}", "need": f">= {math.ceil(0.95 * len(rows))}", "pass": len(ok) >= math.ceil(0.95 * len(rows))},
        "O2_clearance_m": {"value": mc, "need": ">= 0.03", "pass": mc is not None and mc >= 0.03},
        "O3_lap1_median_s": {"value": q(lap1, 0.5) if lap1 else None, "need": "<= 10.5", "pass": bool(lap1) and q(lap1, 0.5) <= 10.5, "n": len(lap1)},
        "O4_empty_s": {"value": empty_row.get("lap_from_stop"), "need": "<= 9.0", "pass": bool(empty_row.get("lap_complete")) and empty_row["lap_from_stop"] <= 9.0},
    }


def collisions(rows):
    col = [r for r in rows if r.get("collisions")]
    ts = [r["collision"]["t"] for r in col]
    return {
        "n": len(col), "added": sum(1 for r in col if r["collision"]["added"]), "course": [f'{r["id"]}:{r["collision"]["collider"]}' for r in col if not r["collision"]["added"]],
        "t_le_2_1s": sum(1 for t in ts if t <= 2.1), "t_median": q(ts, 0.5), "t_quartiles": [q(ts, 0.25), q(ts, 0.75)],
        "first_obstacle_ahead_m_of_early": q([r["obstacle_ahead_m"] for r in col if r["collision"]["t"] <= 2.1 and r.get("obstacle_ahead_m") is not None], 0.5),
        "frames_survived_mean": st.mean([r["frames"] for r in rows]),
    }


def passages(rows):
    ps = [p for r in rows for p in r.get("passages", []) if p["completed"]]
    cl = [p["clearance_frame_min"] for p in ps]
    return {"completed": len(ps), "left": sum(1 for p in ps if p["side"] > 0), "right": sum(1 for p in ps if p["side"] < 0),
            "clearance_cm": {"min": None if not cl else 100 * min(cl), "p05": None if not cl else 100 * q(cl, 0.05), "median": None if not cl else 100 * q(cl, 0.5)},
            "speed_median": q([p["v"] for p in ps], 0.5), "abs_d_lat_median_cm": None if not ps else 100 * q([abs(p["d_lat"]) for p in ps], 0.5)}


def vm_spread(rows, key):
    """VM2〜VM4 の式: 絶対の s の 0.05 m のマス(開始から 3 m 以降の最初の通過)で、全シーンがそろうマスの、平均からの |差| の平均"""
    profs = [r["prof"][key] for r in rows]
    if len(profs) < 2:
        return None
    nb = len(profs[0])
    vals = []
    for j in range(nb):
        col = [p[j] for p in profs]
        if any(v is None for v in col):
            continue
        m = sum(col) / len(col)
        vals.append(sum(abs(v - m) for v in col) / len(col))
    return {"value": sum(vals) / len(vals) if vals else None, "bins": len(vals)}


def pair_diff(a, b, key="ey"):
    """同じ開始の 2 回: 両方がある s のマスでの |差| の平均"""
    pa, pb = a["prof"][key], b["prof"][key]
    d = [abs(x - y) for x, y in zip(pa, pb) if x is not None and y is not None]
    return (sum(d) / len(d), len(d)) if d else (None, 0)


def exploits(all_rows):
    cont = [r for r in all_rows if not r.get("single") and not r.get("invalid_spawn")]
    valid = [r for r in cont if r.get("valid_episode")]
    lap_bad = [f'{r["set"]}:{r["id"]}' for r in all_rows for l in r.get("laps", []) if not (l["gates"] == 33 and l["crossings"] == 1 and l["collisions"] == 0)]
    return {
        "runs": len(all_rows),
        "overlap_without_collision": [f'{r["set"]}:{r["id"]}' for r in all_rows if r.get("overlap_frames", 0) > 0 and not r.get("collisions")],
        "overlap_frames_total_in_collision_runs": sum(r.get("overlap_frames", 0) for r in all_rows if r.get("collisions")),
        "audit_pos_err_max_m": max(r.get("audit_pos_err", 0) for r in all_rows),
        "gate_stuck_max_valid": max((r["gate_stuck_max"] for r in valid), default=0),
        "gates_mismatch_valid": [f'{r["set"]}:{r["id"]}' for r in valid if r["gates_by_s"] != r["passed_gates"]],
        "crossings_total": sum(len(r["own_crossings"]) for r in all_rows),
        "crossings_outside": [f'{r["set"]}:{r["id"]}' for r in all_rows for c in r["own_crossings"] if not c["inside"]],
        "laps_not_33_gates_1_crossing_0_collision": lap_bad,
        "valid_runs": len(valid), "valid_laps_total": sum(r["valid_laps"] for r in valid),
    }


def labels(all_rows):
    L = [r["labels"] for r in all_rows if "labels" in r]
    ob = [r for r in all_rows if r["policy"] == "obstacle" and r.get("layout_id")]
    return {
        "frames": sum(x["frames"] for x in L), "l0": sum(x["l0"] for x in L),
        "ph": {k: sum(x["ph"].get(k, 0) for x in L) for k in {k for x in L for k in x["ph"]}},
        "ev_nonzero": sum(x["ev_nonzero"] for x in L), "sh_key": sum(x["sh_key"] for x in L), "es_key": sum(x["es_key"] for x in L),
        "bad_keys": sum(x["bad_keys"] for x in L), "bytes_max": max(x["bytes_max"] for x in L), "cl_diff_mm_max": max(x["cl_diff_mm_max"] for x in L),
        "p0_layout_id_ok": sum(1 for r in ob if (r["labels"]["p0"] or {}).get("layout_id") == r["layout_id"]), "p0_layout_runs": len(ob),
        "p0_keys": sorted({k for r in all_rows if r["labels"]["p0"] for k in r["labels"]["p0"]}),
        "seen_obstacle_err_max": max(r.get("seen_obstacle_err", 0) for r in all_rows),
    }


def main():
    V = load("VREPLAY")
    F = load("FRESH")
    FF = load("FRESH_FAST")
    E = load("EMPTY")
    N = load("NOOB")
    R = load("REC")
    S = {}
    # 1) 学習の段の検証の組の結果と、同じ条件の自前の走り
    ev = json.load((REP / "train/obstacle/main/evals/obstacle/step_002015232.json").open())
    rows_t = {r["id"]: r for r in ev["tests"]["OB-T100"]}
    vr = [r for r in V if r["test"] == "VREPLAY"]
    ve = [r for r in V if r["test"] == "VREPLAY_EMPTY"][0]
    mism = []
    dmc = 0.0
    for r in vr:
        t = rows_t[r["id"]]
        same = t["reason"] == r["reason"] and t["frames"] == r["frames"] and t["valid_episode"] == r["valid_episode"] and \
            len(t["laps"]) == len(r["laps"]) and all(abs(a - b["elapsed"]) < 1e-6 for a, b in zip(t["laps"], r["laps"]))
        if not same:
            mism.append(r["id"])
        if t.get("min_clearance") is not None:
            dmc = max(dmc, abs(t["min_clearance"] - r["min_clearance"]))
    S["vreplay"] = {"runs": len(vr), "mismatch": mism, "min_clearance_max_abs_diff_m": dmc, "judged": judge(vr, ve),
                    "trainer_judged": {k: v["value"] for k, v in ev["judged"].items()}, "collisions": collisions(vr), "passages": passages(vr)}
    # 2) 新しい配置・新しい開始(試験の組と同じ作り方)
    e_std = [r for r in E if r["id"] == "obstacle-STD"][0]
    S["fresh"] = {"judged": judge(F, e_std), "by_start": {k: judge([r for r in F if r["id"].endswith(k)], e_std)["O1_valid"]["value"] for k in ("-STD", "-R")},
                  "by_count": {n: f'{sum(1 for r in F if r["n_obstacles"] == n and r["valid_episode"])}/{sum(1 for r in F if r["n_obstacles"] == n)}' for n in (1, 2, 3, 4)},
                  "valid_ids": [r["id"] for r in F if r["valid_episode"]], "valid_min_clearance_cm": sorted(round(100 * r["min_clearance"], 2) for r in F if r["valid_episode"]),
                  "collisions": collisions(F), "passages": passages(F),
                  "valid_layouts_distinct": len({r["layout_id"] for r in F if r["valid_episode"]})}
    S["fresh_fast"] = {"valid": f'{sum(1 for r in FF if r["valid_episode"])}/{len(FF)}', "collisions": collisions(FF), "passages": passages(FF)}
    # 同じ開始で障害物よけ・最速の生き残ったフレーム
    ff = {r["id"]: r for r in FF}
    longer = sum(1 for r in F if r["frames"] > ff[r["id"]]["frames"])
    shorter = sum(1 for r in F if r["frames"] < ff[r["id"]]["frames"])
    S["fresh_pair"] = {"obstacle_longer": longer, "obstacle_shorter": shorter, "same": len(F) - longer - shorter,
                       "frames_mean_obstacle": st.mean(r["frames"] for r in F), "frames_mean_fastest": st.mean(r["frames"] for r in FF)}
    # 3) 障害物なし
    S["empty"] = {r["id"]: {"lap_from_stop": r.get("lap_from_stop"), "complete": r.get("lap_complete"), "min_clearance_cm": round(100 * r["min_clearance"], 2),
                            "start": r["start"]} for r in E}
    ov = [r["lap_from_stop"] for r in E if r["id"].startswith("obstacle-V")]
    S["empty_variants_obstacle"] = {"n": len(ov), "over_9s": sum(1 for x in ov if x is None or x > 9.0), "max": max(ov), "min": min(ov)}
    nob = {r["id"]: r for r in N}
    pairs = []
    for i in range(20):
        a, b = nob.get(f"N{i}-obstacle"), nob.get(f"N{i}-fastest")
        if not a or not b:
            continue
        d, nb = pair_diff(a, b)
        pairs.append({"i": i, "ey_diff_cm": None if d is None else 100 * d, "bins": nb, "valid_o": a["valid_episode"], "valid_f": b["valid_episode"],
                      "v_o": a["speed"]["mean_running"], "v_f": b["speed"]["mean_running"], "lap1_o": a["laps"][0]["elapsed"] if a["laps"] else None,
                      "lap1_f": b["laps"][0]["elapsed"] if b["laps"] else None, "minc_o": a["min_clearance"], "minc_f": b["min_clearance"]})
    dd = [p["ey_diff_cm"] for p in pairs if p["ey_diff_cm"] is not None]
    S["noob"] = {"pairs": len(pairs), "valid_obstacle": sum(p["valid_o"] for p in pairs), "valid_fastest": sum(p["valid_f"] for p in pairs),
                 "ey_diff_cm_mean": st.mean(dd), "ey_diff_cm_min": min(dd), "ey_diff_cm_max": max(dd),
                 "speed_mean_obstacle": st.mean(p["v_o"] for p in pairs), "speed_mean_fastest": st.mean(p["v_f"] for p in pairs),
                 "lap1_median_obstacle": q([p["lap1_o"] for p in pairs if p["lap1_o"]], 0.5), "lap1_median_fastest": q([p["lap1_f"] for p in pairs if p["lap1_f"]], 0.5),
                 "min_clearance_median_cm_obstacle": 100 * q([p["minc_o"] for p in pairs], 0.5), "min_clearance_median_cm_fastest": 100 * q([p["minc_f"] for p in pairs], 0.5),
                 "VM3_obstacle_m": vm_spread([nob[f"N{p['i']}-obstacle"] for p in pairs if p["valid_o"]], "ey"),
                 "VM3_fastest_m": vm_spread([nob[f"N{p['i']}-fastest"] for p in pairs if p["valid_f"]], "ey"), "rows": pairs}
    # 4) 記録の組(記録の決まり)
    S["rec"] = {"valid": f'{sum(1 for r in R if r["valid_episode"])}/{len(R)}', "collisions": collisions(R), "passages": passages(R)}
    # 5) ばらつき(記録できる見本が無いので、新しい配置の有効な回で参考の値)
    fv = [r for r in F if r["valid_episode"]]
    S["variation_ref"] = {"fresh_valid_runs": len(fv), "VM3_m_fresh_valid": vm_spread(fv, "ey"), "VM4_mps_fresh_valid": vm_spread(fv, "v"),
                          "distinct_layouts_among_valid": len({r["layout_id"] for r in fv}),
                          "expected_valid_of_15": 15 * len(fv) / len(F), "rec_valid": S["rec"]["valid"],
                          "note": "記録の決まり(RA1: 15 シーンの 14 以上が有効)では、有効の割合 5% 前後・記録の組 0/30 で見本が集まらない"}
    # 6) ずる・ラベル
    allr = V + F + FF + E + N + R
    S["exploits"] = exploits(allr)
    S["labels"] = labels(allr)
    for name, src in (("feat_stats", OUT / "feat_stats.json"), ("layout_audit", OUT / "layout_audit.json")):
        if src.exists():
            d = json.load(src.open())
            if name == "layout_audit":
                d = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in d.items()}
            S[name] = d
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
    show = {k: S[k] for k in ("vreplay", "fresh", "fresh_fast", "fresh_pair", "empty_variants_obstacle", "rec", "variation_ref", "exploits", "labels")}
    show["noob"] = {k: v for k, v in S["noob"].items() if k != "rows"}
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
