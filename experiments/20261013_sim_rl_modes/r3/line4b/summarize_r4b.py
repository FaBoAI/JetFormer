"""経路いろいろ r4b の結果の集計(2026-10-14 夜、報告 train_line_r4.md の表の元)。判定はしない(判定は eval_line_r4b.mjs の結果のまま)。

    python3 r3/line4b/summarize_r4b.py  → r3/line_r4b/summary_r4b.json と、標準出力に表(Markdown)
"""
from __future__ import annotations

import json
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
O = REP / "r3/line_r4b"


def load(p):
    p = Path(p)
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def clr(r):
    if r.get("invalid_spawn") or r.get("collisions"):
        return 0.0
    return r.get("min_clearance") or 0.0


def set_rows(d, sets=None):
    out = {}
    for t, rows in (d or {}).get("tests", {}).items():
        if sets and t not in sets:
            continue
        n = len(rows)
        out[t] = {"n": n, "valid_r3": sum(1 for r in rows if r.get("valid_r3")), "collisions": sum(1 for r in rows if r.get("collisions") or r.get("invalid_spawn")),
                  "stalls": sum(1 for r in rows if r.get("stall")), "below3_valid": sum(1 for r in rows if r.get("valid_r3") and (r.get("min_clearance") or 0) < .03),
                  "below3_any": sum(1 for r in rows if clr(r) < .03), "min_all_cm": round(100 * min([0.0 if r.get("stall") else clr(r) for r in rows] or [float("nan")]), 2),
                  "shield_runs": sum(1 for r in rows if r.get("shield_frames")), "shield_frames": sum(r.get("shield_frames") or 0 for r in rows),
                  "escape_runs": sum(1 for r in rows if r.get("escape_frames")), "reverse_runs": sum(1 for r in rows if r.get("reverse_frames")),
                  "brake_first_frames": sum((r.get("shield_fallbacks") or {}).get("brake_first", 0) for r in rows),
                  "brake_first_runs": sum(1 for r in rows if (r.get("shield_fallbacks") or {}).get("brake_first")),
                  "meta_shield_runs": sum(1 for r in rows if r.get("teacher_shield_meta")),
                  "neg_speed_cmd_runs": sum(1 for r in rows if r.get("neg_speed_cmd_frames"))}
    return out


def main():
    S = {}
    sel = load(O / "selection.json")
    if sel:
        S["selection"] = {"selected": {k: sel["selected"].get(k) for k in ("origin", "step", "policy_sha256", "all_pass", "failed", "hard_min_clearance", "escape_runs", "L3_p95", "L2")},
                          "rule": sel["rule"],
                          "candidates": [{k: c.get(k) for k in ("origin", "step", "all_pass", "n_pass", "failed", "hard_min_clearance", "hard_fail", "escape_runs", "L1", "L2", "L3_p95",
                                                                 "L5", "L11", "shield_fraction")} for c in sel["candidates"]]}
    T = load(O / "eval_test_r4b.json")
    if T:
        S["test"] = {"all_judged_pass": T["all_judged_pass"], "judged": {k: {"pass": v.get("pass"), "value": v.get("value")} for k, v in T["judged"].items()},
                     "by_set": set_rows(T), "reported": {k: T["reported"].get(k) for k in ("hard_min_clearance", "hard_fail", "shield_fallbacks", "shield_kept", "large_heading",
                                                                                            "below_5cm", "not_valid_r3", "by_family", "tracking_gt12", "lap1", "speed_mean_running_P",
                                                                                            "variation_P", "shield_gt5pct", "reverse", "stalls")},
                     "wall_s": T.get("wall_s"), "created_at": T.get("created_at"), "policy": T["policy"]["sha256"], "shield": T["shield"]}
    F = load(O / "eval_fresh_r4b.json")
    if F:
        S["fresh"] = {"blocks": F["reported"].get("blocks"), "all": F["reported"].get("all"), "g30w": F["reported"].get("g30w"), "by_set": set_rows(F)}
    R = load(O / "eval_regression_r4b.json")
    if R:
        S["regression"] = {"by_set": set_rows(R), "fail5": R["reported"].get("fail5"), "r3b_fail16": R["reported"].get("r3b_fail16")}
    C = {}
    for name in ("eval_test_noshield", "eval_fresh_noshield", "eval_r3pol_noshield_g30_box", "eval_r4init_noshield_g30_box", "eval_test_r3pol_v5b", "eval_test_r3pol_r3shield"):
        d = load(O / f"compare/{name}.json")
        if d:
            C[name] = {"by_set": set_rows(d), "policy": d["policy"]["sha256"] if d.get("policy") else None,
                       "shield": (d.get("shield") or {}).get("mode_runtime", {}).get("shield_params", {}).get("select") if d.get("shield") else None,
                       "all_judged_pass": d.get("all_judged_pass"), "failed": [k for k, v in d.get("judged", {}).items() if v.get("pass") is False]}
    fa = load(O / "compare/eval_fastest_p60.json")
    if fa:
        C["fastest"] = fa["reported"].get("fastest")
        if T:
            # 同じ開始どうしの線の違い(5 cm のマスの |Δe_y| の平均)
            P = {r["id"]: r for r in T["tests"]["LN-P60-R4"] if r.get("valid_r3") and r.get("prof")}
            Q = {r["id"]: r for r in fa["tests"]["LN-P60-R4-FAST"] if r.get("valid_r3") and r.get("prof")}
            ds = []
            for k in sorted(set(P) & set(Q)):
                a, b = P[k]["prof"]["ey"], Q[k]["prof"]["ey"]
                v = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
                if v:
                    ds.append(sum(v) / len(v))
            C["vs_fastest_dey"] = {"pairs": len(ds), "mean_m": sum(ds) / len(ds) if ds else None, "min_m": min(ds) if ds else None}
    S["compare"] = C
    B = load(O / "browser/browser_fid.json")
    if B:
        S["browser_fid"] = {"pass": B.get("pass"), "browser": B.get("browser"), "page_env": B.get("page_env"),
                            "fids": {k: {x: v.get(x) for x in ("frames", "exact_same", "max_abs_steering", "max_abs_speed", "info_mismatch", "sh_same_frames", "metadata_same",
                                                                 "shield_frames", "shield_frames_node", "ms_mean", "ms_p95", "pass", "error")} for k, v in B.get("fids", {}).items()}}
    sh = load("/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/line_r4/shield.json")
    if sh:
        S["fid_check_node"] = sh.get("fid_check_node")
        S["fids"] = sh.get("fids")
    (O / "summary_r4b.json").write_text(json.dumps(S, ensure_ascii=False, indent=1) + "\n")
    # 表
    if T:
        print("| 組 | 回 | 有効 r3 | 衝突 | 止まり | 有効で 3 cm 未満 | 最小すき間 | シールドが働いた回 | 抜け出し | brake_first の回 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for t, v in S["test"]["by_set"].items():
            print(f"| {t} | {v['n']} | {v['valid_r3']} | {v['collisions']} | {v['stalls']} | {v['below3_valid']} | {v['min_all_cm']:.2f} cm | {v['shield_runs']} | {v['escape_runs']} | {v['brake_first_runs']} |")
    for name, c in C.items():
        if isinstance(c, dict) and "by_set" in c:
            print(f"\n{name}: " + "; ".join(f"{t}: 衝突 {v['collisions']}・止まり {v['stalls']}・3cm未満(有効) {v['below3_valid']}・最小 {v['min_all_cm']} cm" for t, v in c["by_set"].items()))
    print("\n書いた:", O / "summary_r4b.json")


if __name__ == "__main__":
    main()
