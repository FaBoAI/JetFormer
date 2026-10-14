"""障害物よけ(obstacle)の確認 r2 の集計(2026-10-14、確認の段 r2)。

protocol.json acceptance.obstacle の文言(O1〜O4)・definitions(valid_episode・clearance・lap_from_stop)・variation_metrics(VM3 の式)・
modes.json recording_common.prefilter(候補はプランの 3 倍まで)から自前で判定する。学習・評価の段のコード(eval_modes.mjs judge・
variation_metrics.mjs・rl/ob/*・rl/ob2/*)は使わない。

    python3 -I analyze_ob2.py → check_obstacle_r2/summary.json
"""
import json
import math
import statistics as st
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
OUT = REP / "check_obstacle_r2"
RUNS = OUT / "runs"


def load(name):
    rows = []
    for p in sorted(RUNS.glob(f"{name}*.jsonl")):
        if name in ("FRESH", "FRESH_") and ("FAST" in p.name or "R1" in p.name):
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


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def binom_tail_ge(n, p, k):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def judge(rows, empty_row):
    ok = [r for r in rows if r.get("valid_episode")]
    lap1 = [r["laps"][0]["elapsed"] for r in ok if r["laps"]]
    mc = min((r["min_clearance"] for r in ok), default=None)
    need = math.ceil(0.95 * len(rows))
    return {
        "O1_valid": {"value": f"{len(ok)}/{len(rows)}", "need": f">= {need}", "pass": len(ok) >= need,
                     "valid_with_zero_laps": sum(1 for r in ok if not r["laps"])},
        "O2_clearance_m": {"value": mc, "need": ">= 0.03", "pass": mc is not None and mc >= 0.03,
                           "valid_below_3cm": sum(1 for r in ok if r["min_clearance"] < 0.03)},
        "O3_lap1_median_s": {"value": q(lap1, 0.5) if lap1 else None, "need": "<= 10.5", "pass": bool(lap1) and q(lap1, 0.5) <= 10.5, "n": len(lap1)},
        "O4_empty_s": None if empty_row is None else {"value": empty_row.get("lap_from_stop"), "need": "<= 9.0",
                                                      "pass": bool(empty_row.get("lap_complete")) and empty_row["lap_from_stop"] <= 9.0},
    }


def collisions(rows):
    col = [r for r in rows if r.get("collisions")]
    ts = [r["collision"]["t"] for r in col]
    return {
        "n": len(col), "added": sum(1 for r in col if r["collision"]["added"]),
        "course": sorted(f'{r["id"]}:{r["collision"]["collider"]}:{r["collision"]["kind"]}' for r in col if not r["collision"]["added"]),
        "t_le_2_1s": sum(1 for t in ts if t <= 2.1), "t_median": q(ts, 0.5), "t_quartiles": [q(ts, 0.25), q(ts, 0.75)],
        # 衝突の直前の速さ = 最後に記録したフレーム(ステップの前の状態)の v。sim は衝突したフレームで速さを 0 にするので trace は使わない
        "speed_at_collision_median": q([r["fr"][-1][6] for r in col], 0.5),
        "early_s_from_start_median": q([r["collision"].get("s_from_start") for r in col if r["collision"]["t"] <= 2.1], 0.5),
        "early_speed_median": q([r["fr"][-1][6] for r in col if r["collision"]["t"] <= 2.1], 0.5),
        "frames_survived_mean": st.mean([r["frames"] for r in rows]) if rows else None,
    }


def passages(rows):
    ps = [p for r in rows for p in r.get("passages", []) if p["completed"]]
    cl = [p["clearance_frame_min"] for p in ps]
    return {"completed": len(ps), "left": sum(1 for p in ps if p["side"] > 0), "right": sum(1 for p in ps if p["side"] < 0),
            "clearance_cm": {"min": None if not cl else 100 * min(cl), "p05": None if not cl else 100 * q(cl, 0.05), "median": None if not cl else 100 * q(cl, 0.5)},
            "speed_median": q([p["v"] for p in ps], 0.5), "abs_d_lat_median_cm": None if not ps else 100 * q([abs(p["d_lat"]) for p in ps], 0.5),
            "obstacle_left_of_nominal_passed_right": None}


def vm_spread(rows, key):
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
    pa, pb = a["prof"][key], b["prof"][key]
    d = [abs(x - y) for x, y in zip(pa, pb) if x is not None and y is not None]
    return (sum(d) / len(d), len(d)) if d else (None, 0)


def bucket(d):
    return "1.5-2.5" if d < 2.5 else "2.5-3.5" if d < 3.5 else "3.5-5" if d < 5 else "5+"


def compare_replay(mine, trainer_rows, id_key="id"):
    t = {r[id_key]: r for r in trainer_rows}
    mism, dmc, lapd = [], 0.0, 0.0
    for r in mine:
        tr = t.get(r["id"])
        if tr is None:
            mism.append(r["id"] + ":なし")
            continue
        tl = [x if isinstance(x, (int, float)) else x.get("elapsed") for x in tr.get("laps") or []]
        ml = [x["elapsed"] for x in r.get("laps") or []]
        same = tr["reason"] == r["reason"] and tr["frames"] == r["frames"] and bool(tr.get("valid_episode")) == bool(r.get("valid_episode")) and len(tl) == len(ml)
        if same:
            lapd = max([lapd] + [abs(a - b) for a, b in zip(tl, ml)])
        if r.get("single") and tr.get("lap_from_stop") is not None:
            lapd = max(lapd, abs(tr["lap_from_stop"] - (r.get("lap_from_stop") or 0)))
        if not same:
            mism.append(r["id"])
        if tr.get("min_clearance") is not None and (r.get("valid_episode") or r.get("single")):
            dmc = max(dmc, abs(tr["min_clearance"] - r["min_clearance"]))   # 衝突した回は重なりの深さの測り方が違う(自前は負の値)ので比べない
    return {"runs": len(mine), "mismatch": mism, "lap_time_max_abs_diff_s": lapd, "min_clearance_max_abs_diff_m": dmc}


def exploits(all_rows):
    cont = [r for r in all_rows if not r.get("single") and not r.get("invalid_spawn")]
    valid = [r for r in cont if r.get("valid_episode")]
    lap_bad = [f'{r["set"]}:{r["id"]}' for r in all_rows for l in r.get("laps", []) if not (l["gates"] == 33 and l["crossings"] == 1 and l["collisions"] == 0)]
    return {
        "runs": len(all_rows), "frames": sum(r["frames"] for r in all_rows),
        "overlap_without_collision": [f'{r["set"]}:{r["id"]}' for r in all_rows if r.get("overlap_frames", 0) > 0 and not r.get("collisions")],
        "overlap_frames_total_in_collision_runs": sum(r.get("overlap_frames", 0) for r in all_rows if r.get("collisions")),
        "audit_pos_err_max_m": max(r.get("audit_pos_err", 0) for r in all_rows),
        "gate_stuck_max_valid": max((r["gate_stuck_max"] for r in valid), default=0),
        "gates_mismatch_valid": [f'{r["set"]}:{r["id"]}:{r["gates_by_s"]}-{r["passed_gates"]}' for r in valid if r["gates_by_s"] != r["passed_gates"]],
        "crossings_total": sum(len(r["own_crossings"]) for r in all_rows),
        "crossings_outside": [f'{r["set"]}:{r["id"]}' for r in all_rows for c in r["own_crossings"] if not c["inside"]],
        "laps_not_33_gates_1_crossing_0_collision": lap_bad,
        "valid_runs": len(valid), "valid_laps_total": sum(r["valid_laps"] for r in valid),
        "seen_obstacle_err_max": max(r.get("seen_obstacle_err", 0) for r in all_rows),
    }


def labels(all_rows):
    L = [r["labels"] for r in all_rows if "labels" in r]
    ob = [r for r in all_rows if r["policy"] in ("obstacle", "obstacle_r1") and r.get("layout_id")]
    return {
        "frames": sum(x["frames"] for x in L), "l0": sum(x["l0"] for x in L),
        "ph": {k: sum(x["ph"].get(k, 0) for x in L) for k in {k for x in L for k in x["ph"]}},
        "ev_nonzero": sum(x["ev_nonzero"] for x in L), "sh_key": sum(x["sh_key"] for x in L), "es_key": sum(x["es_key"] for x in L),
        "bad_keys": sum(x["bad_keys"] for x in L), "bytes_max": max(x["bytes_max"] for x in L), "cl_diff_mm_max": max(x["cl_diff_mm_max"] for x in L),
        "p0_layout_id_ok": sum(1 for r in ob if (r["labels"]["p0"] or {}).get("layout_id") == r["layout_id"]), "p0_layout_runs": len(ob),
        "p0_policy_prefix": sorted({(r["labels"]["p0"] or {}).get("policy") for r in all_rows if r["labels"]["p0"]}),
        "p0_keys": sorted({k for r in all_rows if r["labels"]["p0"] for k in r["labels"]["p0"]}),
    }


def below3(rows):
    out = []
    for r in rows:
        if r.get("valid_episode") and r["min_clearance"] < 0.03:
            out.append({"id": r["id"], "min_cm": round(100 * r["min_clearance"], 3), "t": round(r["min_t"], 2), "kind": r["min_kind"],
                        "added_min_cm": None if r.get("min_clear_added") is None else round(100 * r["min_clear_added"], 3),
                        "start_clearance_cm": round(100 * r["start_clearance"], 2)})
    return sorted(out, key=lambda x: x["min_cm"])


def main():
    S = {}
    SEL = load("SELREPLAY")
    V = load("VREPLAY")
    F = load("FRESH_")
    F = [r for r in F if r["set"] == "FRESH"]
    FF = load("FRESH_FAST")
    FR1 = load("FRESH_R1")
    E = load("EMPTY")
    N = load("NOOB")
    R = load("REC")
    # 1) 学習の段の選びの組を自前で(元のコア)
    selT = json.load((REP / "train/obstacle/r2/final/selected_selection_original.json").open())
    sel100 = [r for r in SEL if r["test"] == "OB-V100-R2"]
    sel_empty = [r for r in SEL if r["test"] == "OB-EMPTY"][0]
    S["selreplay"] = {"compare": compare_replay(SEL, selT["rows"]), "judged": judge(sel100, sel_empty),
                      "trainer_judged": {k: v["value"] for k, v in selT["judged"].items()},
                      "var6": {r["id"]: r.get("lap_from_stop") for r in SEL if r["test"] == "EMPTY-VAR6"},
                      "below_3cm": below3(sel100), "collisions": collisions(sel100), "passages": passages(sel100)}
    # 2) protocol の検証の組
    vT = json.load((REP / "train/obstacle/r2/r2b/main/evals/obstacle/step_008011776.json").open())
    v26 = [r for r in V if r["test"] == "VREPLAY"]
    ve = [r for r in V if r["test"] == "VREPLAY_EMPTY"][0]
    S["vreplay"] = {"compare": compare_replay(v26, vT["tests"]["OB-T100"]), "judged": judge(v26, ve),
                    "trainer_judged": {k: v["value"] for k, v in vT["judged"].items()}, "below_3cm": below3(v26)}
    # 3) 新しい配置
    e_std = [r for r in E if r["id"] == "obstacle-STD"][0]
    lay_ids = sorted({r["layout_id"] for r in F})
    batches = []
    for b in range(3):
        L = set(lay_ids[50 * b:50 * b + 50])
        rows = [r for r in F if r["layout_id"] in L]
        j = judge(rows, e_std)
        batches.append({"layouts": f"{50*b:03d}-{50*b+49:03d}", "O1": j["O1_valid"]["value"], "O2_cm": round(100 * j["O2_clearance_m"]["value"], 3),
                        "valid_below_3cm": j["O2_clearance_m"]["valid_below_3cm"], "O3_s": j["O3_lap1_median_s"]["value"], "zero_lap_valid": j["O1_valid"]["valid_with_zero_laps"]})
    nv = sum(1 for r in F if r["valid_episode"])
    p_hat = nv / len(F)
    ci = wilson(nv, len(F))
    by_bucket = {}
    for r in F:
        b = bucket(r["obstacle_ahead_m"])
        by_bucket.setdefault(b, [0, 0])
        by_bucket[b][0] += r["valid_episode"]
        by_bucket[b][1] += 1
    ta = json.load((OUT / "test_starts_audit.json").open())
    test_b = ta["test"]["all"]["by_bucket"]
    strat = sum(test_b[k] * by_bucket[k][0] / by_bucket[k][1] for k in test_b)
    # 試験に似た開始(開始のすき間 ≥ 4.94 cm = 試験の組の最小)だけで O2
    F_testlike = [r for r in F if r["start_clearance"] >= 0.0494]
    S["fresh"] = {
        "runs": len(F), "judged_all300": judge(F, e_std), "batches_of_100": batches,
        "valid_rate": p_hat, "valid_rate_wilson95": ci,
        "prob_ge95_of_100_at_p_hat": binom_tail_ge(100, p_hat, 95), "prob_ge95_of_100_at_ci_upper": binom_tail_ge(100, ci[1], 95),
        "by_start": {k: f'{sum(1 for r in F if r["id"].endswith(k) and r["valid_episode"])}/{sum(1 for r in F if r["id"].endswith(k))}' for k in ("-STD", "-R")},
        "by_count": {n: f'{sum(1 for r in F if r["n_obstacles"] == n and r["valid_episode"])}/{sum(1 for r in F if r["n_obstacles"] == n)}' for n in (1, 2, 3, 4)},
        "by_first_obstacle_ahead": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_bucket.items())},
        "test_bucket_counts": test_b, "test_stratified_expected_valid": strat,
        "valid_layouts_both_starts": sum(1 for L in lay_ids if all(r["valid_episode"] for r in F if r["layout_id"] == L)),
        "layouts_both_fail": sum(1 for L in lay_ids if not any(r["valid_episode"] for r in F if r["layout_id"] == L)),
        "below_3cm": below3(F), "O2_testlike_start": {"runs": len(F_testlike), "valid": sum(1 for r in F_testlike if r["valid_episode"]),
                                                       "min_cm": 100 * min(r["min_clearance"] for r in F_testlike if r["valid_episode"]),
                                                       "below_3cm": sum(1 for r in F_testlike if r["valid_episode"] and r["min_clearance"] < 0.03)},
        "crawl_valid": {"zero_lap": [r["id"] for r in F if r["valid_episode"] and not r["laps"]],
                        "one_lap": sum(1 for r in F if r["valid_episode"] and len(r["laps"]) == 1),
                        "progress_2s_min_p05_m": q([r["crawl"]["progress_2s_min_m"] for r in F if r["valid_episode"]], 0.05),
                        "progress_lt_0_6m": [r["id"] for r in F if r["valid_episode"] and r["crawl"]["progress_2s_min_m"] < 0.6]},
        "collisions": collisions(F), "passages": passages(F),
        "invalid_spawn": sum(1 for r in F if r.get("invalid_spawn")),
    }
    S["fresh_fast"] = {"valid": f'{sum(1 for r in FF if r["valid_episode"])}/{len(FF)}', "collisions": collisions(FF), "passages": passages(FF)}
    ffm = {r["id"]: r for r in FF}
    S["fresh_pair"] = {"obstacle_longer": sum(1 for r in F if r["frames"] > ffm[r["id"]]["frames"]),
                       "obstacle_shorter": sum(1 for r in F if r["frames"] < ffm[r["id"]]["frames"]),
                       "both_valid": sum(1 for r in F if r["valid_episode"] and ffm[r["id"]]["valid_episode"]),
                       "only_fastest_valid": [r["id"] for r in F if not r["valid_episode"] and ffm[r["id"]]["valid_episode"]],
                       "speed_running_median_obstacle_valid": q([r["speed"]["mean_running"] for r in F if r["valid_episode"]], 0.5),
                       "lap1_median_obstacle_valid": q([r["laps"][0]["elapsed"] for r in F if r["valid_episode"] and r["laps"]], 0.5)}
    S["fresh_r1"] = {"valid": f'{sum(1 for r in FR1 if r["valid_episode"])}/{len(FR1)}', "collisions": collisions(FR1), "passages": passages(FR1)}
    f50 = [r for r in F if r["layout_id"] in set(lay_ids[:50])]
    S["fresh_r1"]["r2_same_100"] = f'{sum(1 for r in f50 if r["valid_episode"])}/{len(f50)}'
    # 4) 障害物なし
    def em(pol):
        rows = [r for r in E if r["policy"] == pol]
        std = [r for r in rows if r["id"] == f"{pol}-STD"][0]
        v6 = [r for r in rows if r["test"] == "EMPTY_V"]
        rv = [r for r in rows if r["test"] == "EMPTY_RV"]
        allv = [r.get("lap_from_stop") for r in v6 + rv]
        return {"STD": std.get("lap_from_stop"), "STD_complete": std.get("lap_complete"), "STD_min_clearance_cm": round(100 * std["min_clearance"], 2),
                "var6": {f'{r["start"]["left_offset_m"]:+.2f}m,{r["start"]["left_yaw_deg"]:+.0f}deg': r.get("lap_from_stop") for r in v6},
                "var30_over_9s": sum(1 for x in allv if x is None or x > 9.0), "var30_n": len(allv),
                "var30_max": max(x for x in allv if x is not None), "var30_median": q([x for x in allv if x is not None], 0.5),
                "incomplete": [r["id"] for r in v6 + rv if not r.get("lap_complete")]}
    S["empty"] = {"obstacle": em("obstacle"), "fastest": em("fastest")}
    # 5) 障害物なしの連続周回
    nob = {r["id"]: r for r in N}
    pairs = []
    for i in range(20):
        a, b = nob.get(f"N{i}-obstacle"), nob.get(f"N{i}-fastest")
        d, nb = pair_diff(a, b)
        pairs.append({"i": i, "ey_diff_cm": None if d is None else 100 * d, "valid_o": a["valid_episode"], "valid_f": b["valid_episode"],
                      "v_o": a["speed"]["mean_running"], "v_f": b["speed"]["mean_running"], "lap1_o": a["laps"][0]["elapsed"] if a["laps"] else None,
                      "lap1_f": b["laps"][0]["elapsed"] if b["laps"] else None, "minc_o": a["min_clearance"], "minc_f": b["min_clearance"]})
    dd = [p["ey_diff_cm"] for p in pairs if p["ey_diff_cm"] is not None]
    S["noob"] = {"pairs": len(pairs), "valid_obstacle": sum(p["valid_o"] for p in pairs), "valid_fastest": sum(p["valid_f"] for p in pairs),
                 "ey_diff_cm_mean": st.mean(dd), "ey_diff_cm_min": min(dd), "ey_diff_cm_max": max(dd),
                 "speed_mean_obstacle": st.mean(p["v_o"] for p in pairs), "speed_mean_fastest": st.mean(p["v_f"] for p in pairs),
                 "lap1_median_obstacle": q([p["lap1_o"] for p in pairs if p["lap1_o"]], 0.5), "lap1_median_fastest": q([p["lap1_f"] for p in pairs if p["lap1_f"]], 0.5),
                 "min_clearance_median_cm_obstacle": 100 * q([p["minc_o"] for p in pairs], 0.5), "min_clearance_median_cm_fastest": 100 * q([p["minc_f"] for p in pairs], 0.5),
                 "min_clearance_min_cm_obstacle": 100 * min(p["minc_o"] for p in pairs), "min_clearance_min_cm_fastest": 100 * min(p["minc_f"] for p in pairs),
                 "VM3_obstacle_m": vm_spread([nob[f"N{p['i']}-obstacle"] for p in pairs if p["valid_o"]], "ey"),
                 "VM3_fastest_m": vm_spread([nob[f"N{p['i']}-fastest"] for p in pairs if p["valid_f"]], "ey"), "rows": pairs}
    # 6) 記録のプランをまねる(配置を順に、同じ配置で 3 回まで、15 シーンで止める。modes.json prefilter: 候補はプランの 3 倍 = 45 まで)
    recby = {}
    for r in R:
        recby.setdefault(r["layout_id"], []).append(r)
    order = sorted(recby)
    scenes, tries, scenes_at45 = [], 0, None
    for lid in order:
        if len(scenes) >= 15:
            break
        for r in sorted(recby[lid], key=lambda x: x["id"]):
            tries += 1
            if r["valid_episode"]:
                scenes.append(r)
            if tries == 45:
                scenes_at45 = len(scenes)
            if r["valid_episode"]:
                break
    if scenes_at45 is None:
        scenes_at45 = len(scenes)
    # 同じまねで「1 周以上」も条件にしたとき(止まったままの回を替える)
    scenes_l, tries_l = [], 0
    for lid in order:
        if len(scenes_l) >= 15:
            break
        for r in sorted(recby[lid], key=lambda x: x["id"]):
            tries_l += 1
            if r["valid_episode"] and r["laps"]:
                scenes_l.append(r)
                break
    rec_all_valid = [r for r in R if r["valid_episode"]]
    S["rec"] = {"tries_total": len(R), "valid_total": len(rec_all_valid), "valid_rate": len(rec_all_valid) / len(R),
                "layouts_with_any_valid": sum(1 for lid in order if any(r["valid_episode"] for r in recby[lid])),
                "plan_emulation": {"scenes": len(scenes), "tries_used": tries, "scenes_within_45_tries": scenes_at45,
                                   "distinct_layouts": len({r["layout_id"] for r in scenes}),
                                   "VM3_m": vm_spread(scenes, "ey"), "VM4_mps": vm_spread(scenes, "v"),
                                   "scene_min_clearance_cm": sorted(round(100 * r["min_clearance"], 2) for r in scenes),
                                   "scenes_below_3cm": sum(1 for r in scenes if r["min_clearance"] < 0.03),
                                   "lap1_s": sorted(round(r["laps"][0]["elapsed"], 3) for r in scenes if r["laps"]),
                                   "zero_lap_scenes": [r["id"] for r in scenes if not r["laps"]],
                                   "VM3_m_excluding_zero_lap": vm_spread([r for r in scenes if r["laps"]], "ey")},
                "plan_emulation_require_lap": {"scenes": len(scenes_l), "tries_used": tries_l, "distinct_layouts": len({r["layout_id"] for r in scenes_l}),
                                               "VM3_m": vm_spread(scenes_l, "ey"), "VM4_mps": vm_spread(scenes_l, "v"),
                                               "scenes_below_3cm": sum(1 for r in scenes_l if r["min_clearance"] < 0.03),
                                               "scene_min_clearance_cm": sorted(round(100 * r["min_clearance"], 2) for r in scenes_l)},
                "VM3_all_valid_with_laps_m": vm_spread([r for r in rec_all_valid if r["laps"]], "ey"),
                "VM3_all_valid_distinct_layouts": len({r["layout_id"] for r in rec_all_valid}),
                "RA1_without_prefilter_expected_valid_of_15": 15 * len(rec_all_valid) / len(R),
                "RA1_without_prefilter_prob_ge14": binom_tail_ge(15, len(rec_all_valid) / len(R), 14),
                "collisions": collisions(R), "passages": passages(R)}
    # 7) ずる・ラベル
    allr = SEL + V + F + FF + FR1 + E + N + R
    S["exploits"] = exploits(allr)
    S["labels"] = labels(allr)
    for name in ("feat_stats", "probe_obs", "test_starts_audit", "probe_freeze"):
        src = OUT / f"{name}.json"
        if src.exists():
            d = json.load(src.open())
            if name == "test_starts_audit":
                d = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in d.items()}
            if name == "probe_freeze":
                d = {"grid": d["grid"], **{k: {kk: vv for kk, vv in v.items() if kk not in ("early_terminal_rows",)} for k, v in d["policies"].items()}}
            if name == "probe_obs":
                d = {**d, "slot3": {k: v for k, v in d["slot3"].items() if k != "rows"}}
            S[name] = d
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
    show = {k: S[k] for k in ("selreplay", "vreplay", "fresh", "fresh_fast", "fresh_pair", "fresh_r1", "empty", "rec", "exploits", "labels")}
    show["noob"] = {k: v for k, v in S["noob"].items() if k != "rows"}
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
