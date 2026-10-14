"""no_wall r3 の判定(2026-10-14、r3 の学習の段)。run_sets_nw3.mjs の 1 行 1 回の JSONL から、protocol_no_wall_r3.json acceptance の判定を作る。

    python3 judge_nw3.py --rows <rows_test.jsonl|rows_validation.jsonl> --which test|validation --out <eval.json>

試験(test): N1_r3〜N7_r3(N4・N6 は protocol.json のまま)と reported(NW-OFF-R3・作動の内訳・NW-G の範囲の外・LOWMU・最速との比べ・ρ・8〜9 cm の回・線のばらつき)。
検証(validation、NW-V-R3): 同じ形の判定(N5 の 95% は組ごとに切り上げ)と、止まりの数(選びの条件「止まり 0」)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
# jf-shield/v4.1 で決まりの v4 に足した速さの見張りの鍵(r3/nw/runtime_v4/shield.mjs の SHIELD_DEFAULTS と同じ。test_shield_v4.mjs で照合)
V41_GOVERNOR_EXTRA = {"brake_without_known": True, "brake_probe": {"enabled": True, "v_min": 2.0, "factor": 0.85, "every": 15}}
PROT_R3 = REP / "r3/protocol_no_wall_r3.json"
PROT = REP / "protocol.json"


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    errs = [r["id"] for r in rows if r.get("error")]
    if errs:
        raise SystemExit(f"エラーの行があります(走らせ直してから判定する): {errs[:10]}")
    by = defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
    return rows, by


def cmin(rs):
    return min((r["min_clearance"] for r in rs), default=None)


def shield_frac(rs):
    fr = sum(r["frames"] for r in rs)
    return sum(r["shield_frames"] for r in rs) / fr if fr else None


def rho(rs):
    n = sum(r["rho"]["n"] for r in rs)
    cn = sum(r["rho"]["comb_n"] for r in rs)
    return {"rho_req_gt1": sum(r["rho"]["req_gt1"] for r in rs) / n if n else None,
            "rho_comb_gt1": sum(r["rho"]["comb_gt1"] for r in rs) / cn if cn else None,
            "rho_req_max": max((r["rho"]["req_max"] for r in rs), default=None),
            "rho_comb_max": max((r["rho"]["comb_max"] for r in rs), default=None)}


def set_stats(rs):
    """組ごとのまとめ(報告用)"""
    if not rs:
        return {"n": 0}
    fr = sum(r["frames"] for r in rs)
    lab = lambda k: sum(r["labels"].get(k) or 0 for r in rs)  # noqa: E731
    lap1 = [r["laps"][0]["elapsed"] for r in rs if r.get("laps") and r["laps"][0]["gates"] == 33 and r["laps"][0]["crossings"] == 1 and r["laps"][0]["collisions"] == 0]
    cl = [r["min_clearance"] for r in rs]
    return {"n": len(rs), "collisions": sum(1 for r in rs if r["collisions"]), "errors": sum(1 for r in rs if r.get("error")),
            "min_clearance_m": min(cl), "min_clearance_id": min(rs, key=lambda r: r["min_clearance"])["id"],
            "median_clearance_m": q(cl, .5), "below_5cm": sum(1 for c in cl if c < .05), "below_8cm": sum(1 for c in cl if c < .08),
            "valid_episode": sum(1 for r in rs if r.get("valid_episode")), "valid_episode_r3": sum(1 for r in rs if r.get("valid_episode_r3")),
            "stall": sum(1 for r in rs if r.get("stall")), "stall_ids": [r["id"] for r in rs if r.get("stall")][:20],
            "zero_valid_laps": sum(1 for r in rs if r.get("valid_episode") and r.get("valid_laps", 0) == 0),
            "shield_fraction": sum(r["shield_frames"] for r in rs) / fr if fr else None,
            "shield_fraction_governor": lab("sh_governor") / fr if fr else None, "shield_fraction_q_below": lab("sh_q") / fr if fr else None,
            "shield_fraction_e_below": lab("sh_e") / fr if fr else None,
            "shield_run_max": max(r["shield_frames"] / max(1, r["frames"]) for r in rs),
            "lap1_median_s": q(lap1, .5), "lap1_max_s": max(lap1) if lap1 else None, "lap1_n": len(lap1),
            "mean_running_speed_mps": sum(r["mean_running_speed"] for r in rs) / len(rs),
            "ctrl_ms_per_frame_median": q([r["ctrl_ms_per_frame"] for r in rs if r.get("ctrl_ms_per_frame")], .5),
            "ctrl_ms_per_frame_p99": q([r["ctrl_ms_per_frame"] for r in rs if r.get("ctrl_ms_per_frame")], .99),
            "collision_ids": [{"id": r["id"], "t": (r.get("collision") or {}).get("time"), "with": (r.get("collision") or {}).get("collider"),
                               "gate": r["start"].get("gate"), "dr": r.get("dr")} for r in rs if r["collisions"]][:30]}


def ceil95(n):
    return math.ceil(.95 * n - 1e-9)


def judge_common(groups, std_single, n5_all, n5_frac, rho_set):
    """groups: {'N1': rows, 'N2': rows, 'N3': rows}。戻り値: judged の dict"""
    J = {}
    n1 = groups["N1"]
    col = [r for r in n1 if r["collisions"] or r.get("error")]
    J["N1_r3"] = {"value": f"{len(n1) - len(col)}/{len(n1)}", "runs": len(n1), "collisions": len(col), "pass": len(col) == 0 and len(n1) > 0}
    v2 = cmin(groups["N2"])
    J["N2_r3"] = {"value": v2, "runs": len(groups["N2"]), "where": min(groups["N2"], key=lambda r: r["min_clearance"])["id"], "pass": v2 is not None and v2 >= .08}
    v3 = cmin(groups["N3"])
    J["N3_r3"] = {"value": v3, "runs": len(groups["N3"]), "where": min(groups["N3"], key=lambda r: r["min_clearance"])["id"], "pass": v3 is not None and v3 >= .05}
    s = std_single
    lap = s.get("lap_from_stop") if s else None
    J["N4"] = {"value": lap, "pass": lap is not None and lap <= 12.0}
    bad = [r["id"] for r in n5_all if not r.get("valid_episode_r3")]
    fr = {k: (sum(1 for r in rs if r.get("valid_episode_r3")), len(rs), need) for k, (rs, need) in n5_frac.items()}
    J["N5_r3"] = {"value": {"all_required": f"{len(n5_all) - len(bad)}/{len(n5_all)}", "not_valid": bad[:30],
                            **{k: f"{a}/{b}(要 {c})" for k, (a, b, c) in fr.items()}},
                  "pass": not bad and all(a >= c for a, b, c in fr.values())}
    rh = rho(rho_set)
    J["N6"] = {"value": {"rho_req_gt1": rh["rho_req_gt1"], "rho_comb_gt1": rh["rho_comb_gt1"]}, "pass": rh["rho_req_gt1"] <= .02 and rh["rho_comb_gt1"] <= .02}
    sf = shield_frac(rho_set)
    runmax = max(r["shield_frames"] / max(1, r["frames"]) for r in rho_set)
    J["N7_r3"] = {"value": {"fraction": sf, "run_max": runmax}, "pass": sf <= .05 and runmax <= .25}
    return J


def judge_test(by):
    std = by["NW-STD"]
    std_single = next(r for r in std if r["id"] == "single")
    std_cont = next(r for r in std if r["id"] == "continuous")
    longs = by["NW-LONG-R3"]
    long_std = [r for r in longs if not r.get("dr")]
    long_dr = [r for r in longs if r.get("dr")]
    N1 = by["NW-N300-R3"] + by["NW-GRID-R3"] + std + by["NW-G"] + by["NW-R200-R3"] + by["NW-VX-R3"] + by["NW-KX-R3"] + by["NW-FC30-R3"] + longs
    N2 = by["NW-N300-R3"] + by["NW-GRID-R3"] + std + long_std
    N3 = by["NW-G"] + by["NW-R200-R3"] + by["NW-VX-R3"] + by["NW-KX-R3"] + by["NW-FC30-R3"] + long_dr
    n5_all = by["NW-N300-R3"] + by["NW-GRID-R3"] + [std_cont] + long_std + by["NW-FC30-R3"]
    n5_frac = {"NW-R200-R3": (by["NW-R200-R3"], 190), "NW-VX-R3": (by["NW-VX-R3"], 285), "NW-KX-R3": (by["NW-KX-R3"], 376)}
    J = judge_common({"N1": N1, "N2": N2, "N3": N3}, std_single, n5_all, n5_frac, by["NW-N300-R3"])
    return J, N1, N2, N3


def judge_validation(by):
    std = by["V-STD"]
    std_single = next(r for r in std if r["id"] == "single")
    std_cont = next(r for r in std if r["id"] == "continuous")
    N1 = by["V-N"] + std + by["V-R"] + by["V-VX"] + by["V-KX"]
    N2 = by["V-N"] + std
    N3 = by["V-R"] + by["V-VX"] + by["V-KX"]
    n5_all = by["V-N"] + [std_cont]
    n5_frac = {k: (by[k], ceil95(len(by[k]))) for k in ("V-R", "V-VX", "V-KX")}
    J = judge_common({"N1": N1, "N2": N2, "N3": N3}, std_single, n5_all, n5_frac, by["V-N"])
    stalls = [r["id"] for r in N1 if r.get("stall")]
    J["stall_zero"] = {"value": len(stalls), "ids": stalls[:30], "pass": not stalls}
    return J, N1, N2, N3


def line_compare(by):
    """最速との比べ(確認 r2 の analyze_nw_r2.py と同じ作り方。N300 の最初の 150 と FAST は同じ開始)"""
    fn = {r["id"]: r for r in by.get("NW-N300-R3", []) if "path" in r}
    ff = {r["id"]: r for r in by.get("FAST", [])}
    if not fn or not ff:
        return None
    L = 26.12

    def binned(rows, key):
        B = defaultdict(list)
        for r in rows:
            seen, dist, prev = set(), 0, None
            for p in r["path"]:
                x, z, v, s, ey = p[:5]
                if prev is not None:
                    dist += math.hypot(x - prev[0], z - prev[1])
                prev = (x, z)
                if dist < 3:
                    continue
                b = int((s % L) / .05)
                if b in seen:
                    continue
                seen.add(b)
                B[b].append(ey if key == "ey" else v)
        return B

    def spread(B, nmin):
        out = []
        for b, xs in B.items():
            if len(xs) >= nmin:
                m = sum(xs) / len(xs)
                out.append(sum(abs(x - m) for x in xs) / len(xs))
        return sum(out) / max(1, len(out)), len(out)

    common = sorted(set(fn) & set(ff), key=lambda k: int(k[1:]))
    rn = [fn[k] for k in common if fn[k]["collisions"] == 0 and fn[k]["valid_episode"]]
    rf = [ff[k] for k in common if ff[k]["collisions"] == 0 and ff[k]["valid_episode"]]
    Bn, Bf = binned(rn, "ey"), binned(rf, "ey")
    Vn, Vf = binned(rn, "v"), binned(rf, "v")
    nmin_n, nmin_f = int(.8 * len(rn)), int(.8 * len(rf))
    diffs, vd = [], []
    for b in Bn:
        if b in Bf and len(Bn[b]) >= nmin_n and len(Bf[b]) >= nmin_f:
            diffs.append(abs(sum(Bn[b]) / len(Bn[b]) - sum(Bf[b]) / len(Bf[b])))
            vd.append(sum(Vn[b]) / len(Vn[b]) - sum(Vf[b]) / len(Vf[b]))
    sn, nb = spread(Bn, nmin_n)
    sf, nbf = spread(Bf, nmin_f)
    return {"starts": len(common), "no_wall_valid": len(rn), "fastest_valid": len(rf),
            "fastest_collisions": sum(1 for k in common if ff[k]["collisions"]),
            "fastest_min_clearance_median_cm": 100 * q([ff[k]["min_clearance"] for k in common if not ff[k]["collisions"]], .5),
            "no_wall_min_clearance_median_cm": 100 * q([fn[k]["min_clearance"] for k in common], .5),
            "fastest_below_8cm": sum(1 for k in common if ff[k]["min_clearance"] < .08),
            "no_wall_below_8cm": sum(1 for k in common if fn[k]["min_clearance"] < .08),
            "line_diff_mean_cm": 100 * sum(diffs) / max(1, len(diffs)), "line_diff_p90_cm": 100 * (q(diffs, .9) or 0),
            "line_diff_max_cm": 100 * max(diffs) if diffs else None, "bins": len(diffs),
            "speed_diff_mean_mps": sum(vd) / max(1, len(vd)), "speed_diff_p10_mps": q(vd, .1), "speed_diff_p90_mps": q(vd, .9),
            "within_mode_spread_ey_cm": {"no_wall": 100 * sn, "fastest": 100 * sf, "bins": [nb, nbf]},
            "mean_running_speed": {"no_wall": sum(r["mean_running_speed"] for r in rn) / max(1, len(rn)),
                                   "fastest": sum(r["mean_running_speed"] for r in rf) / max(1, len(rf))},
            "lap1_median_s": {"no_wall": q([r["laps"][0]["elapsed"] for r in rn if r["laps"]], .5), "fastest": q([r["laps"][0]["elapsed"] for r in rf if r["laps"]], .5)},
            "rho": {"no_wall": rho(rn), "fastest": rho(rf)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--which", choices=["test", "validation"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None)
    a = ap.parse_args()
    rows, by = load(a.rows)
    meta = json.loads(Path(a.meta or a.rows.replace(".jsonl", ".meta.json")).read_text())
    if a.which == "test":
        J, N1, N2, N3 = judge_test(by)
    else:
        J, N1, N2, N3 = judge_validation(by)
    judged_keys = ["N1_r3", "N2_r3", "N3_r3", "N4", "N5_r3", "N6", "N7_r3"]
    all_pass = all(J[k]["pass"] for k in judged_keys)
    n_col = sum(1 for r in N1 if r["collisions"])
    X, Y = cmin(N2), cmin(N3)
    ui_text = f"{len(N1)} 回の試験で衝突 {n_col}・最小すき間 {100 * X:.1f} cm(標準の物理)/ {100 * max(Y, 0):.1f} cm(物理のずれ)"
    rep = {"ui_text": ui_text, "ui_numbers": {"runs": len(N1), "collisions": n_col, "min_clearance_nominal_m": X, "min_clearance_randomised_m": Y},
           "sets": {k: set_stats(v) for k, v in by.items() if k not in ("FAST",)}}
    if a.which == "test":
        off = by.get("NW-OFF-R3", [])
        rep["NW_OFF_R3"] = {of: set_stats([r for r in off if r.get("of") == of]) for of in ("NW-N300-R3", "NW-R200-R3", "NW-VX-R3")}
        # シールドあり / なしの対(救った・起こした)
        on = {(r["set"], r["id"]): r for k in ("NW-N300-R3", "NW-R200-R3", "NW-VX-R3") for r in by.get(k, [])}
        pairs = [(on.get((r["of"], r["id"].split(":", 1)[1])), r) for r in off]
        pairs = [(x, y) for x, y in pairs if x]
        rep["shield_pairs"] = {"pairs": len(pairs), "saved": sum(1 for x, y in pairs if y["collisions"] and not x["collisions"]),
                               "caused": sum(1 for x, y in pairs if x["collisions"] and not y["collisions"])}
        rep["NW_G_OUT"] = [{"id": r["id"], "name": r.get("name"), "reason": r["reason"], "collisions": r["collisions"], "min_clearance": r["min_clearance"]}
                           for r in by.get("NW-G-OUT", [])]
        rep["LOWMU"] = set_stats(by.get("LOWMU", []))
        rep["fastest_compare"] = line_compare(by)
        nom = by["NW-N300-R3"] + by["NW-GRID-R3"] + by["NW-STD"] + [r for r in by["NW-LONG-R3"] if not r.get("dr")]
        rep["nominal_8_to_9cm"] = [{"set": r["set"], "id": r["id"], "gate": r["start"].get("gate"), "left_offset_m": r["start"].get("left_offset_m"),
                                    "left_yaw_deg": r["start"].get("left_yaw_deg"), "min_clearance": r["min_clearance"], "min_frame": r.get("min_frame")}
                                   for r in nom if r["min_clearance"] < .09]
        rep["rho_N300"] = rho(by["NW-N300-R3"])
        rep["KX_by_corner"] = {}
        for c in sorted({r["corner"] for r in by.get("NW-KX-R3", [])}):
            rs = [r for r in by["NW-KX-R3"] if r["corner"] == c]
            d = rs[0]["dr"]
            rep["KX_by_corner"][str(c)] = {"physics": d["physics"], "delay": d["delay"], "steerGain": d["steerGain"], "speedGain": d["speedGain"],
                                           **{k: v for k, v in set_stats(rs).items() if k in ("n", "collisions", "min_clearance_m", "valid_episode_r3", "stall", "shield_fraction", "lap1_median_s")}}
    ev = {"schema": "jf-sim-rl-mode-eval/v1", "created_at": datetime.now(timezone.utc).isoformat(), "mode": "no_wall", "set": a.which,
          "core": "original", "core_note": "元の sim-core.js(リリースの dist/sim-core.js、sha256 aaeb2379…)。摩擦円の回だけ protocol definitions.core_judged のとおり高速版の写し(r3/nw/fast_core_nw3.mjs)",
          "policy": {"path": meta["policy"], "sha256": meta["policy_sha256"], "schema": "jf-sim-rl-mode-policy/v1", "obs_dim": 79},
          "judged": J, "all_judged_pass": all_pass, "judged_keys": judged_keys, "reported": rep, "runs": len(rows),
          "protocol_sha256": sha256(PROT), "protocol_no_wall_r3_sha256": sha256(PROT_R3), "rows_meta": meta}
    shv = next((r.get("shield_version") for r in rows if r.get("shield_version")), None)
    pp = json.loads(PROT_R3.read_text())["runtime"]["shield"]["params"]
    used = json.loads(json.dumps(pp))
    if shv == "jf-shield/v4.1":
        used["governor"].update(V41_GOVERNOR_EXTRA)
    ev["shield"] = {"version": shv, "shield_mjs_sha256": meta.get("rt_shield_sha256") or meta["code_sha256"].get("runtime_v4/shield.mjs"),
                    "params_protocol": pp, "params_used": used,
                    "deviation": "jf-shield/v4.1 = 決まりの v4 + governor の 2 つの鍵(brake_without_known・brake_probe)。deviations.md・train_no_wall_r3.md" if shv == "jf-shield/v4.1" else None}
    Path(a.out).write_text(json.dumps(ev, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"all_judged_pass": all_pass, **{k: (J[k]["pass"], J[k]["value"]) for k in J}, "ui_text": ui_text}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
