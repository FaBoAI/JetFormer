"""no_wall r2: 試験でない組(VAL・G16・ABL・CORNER・FNOM ほか)の結果をまとめ、シールドが「救った衝突」と「起こした衝突」を数える(2026-10-14)。

    python compare_sets.py <rows のフォルダー> [--off <シールドなしの rows のフォルダー>] [--ref <比べる rows のフォルダー>] [--json out.json]

rows_*.jsonl(verify_nw2.mjs / 確認 r1 の verify_nw.mjs の形)を全部読む。set ごとに: 回の数・衝突・最小すき間・5 cm 未満・8 cm 未満・完走 + 1 周・
シールドのフレームの割合・1 周目の中央値。--off があれば、同じ id のシールドなしの回と比べて 救った(なしで衝突・ありで衝突なし)/
起こした(なしで衝突なし・ありで衝突)を数える。
"""
import argparse
import glob
import json
import os
import statistics


def load(d):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "rows_*.jsonl"))):
        if any(p.endswith(f"_{k}.jsonl") for k in range(8)):
            continue
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["set"], {})[(r["test"] + ":" + r["id"]) if r.get("test") else r["id"]] = r
    pg = os.path.join(d, "probe_g16.json")
    if os.path.exists(pg) and "G16" not in rows:   # 確認 r1 の probe_g16.json(形が違う)
        n = 0
        for x in json.load(open(pg)):
            rid = f"sr{x['sr']}-g{x['g']}o{x['o']}y{x['y']}".replace("sr1-", "sr1-").replace("o0.0", "o0")
            rows.setdefault("G16", {})[rid] = {"collisions": x["col"], "min_clearance": x["min"], "shield_frames": x["sh"], "valid_episode": x["valid"], "valid_laps": 1 if x["valid"] else 0, "frames": 600}
            n += 1
    return rows


def summ(rs):
    rs = [r for r in rs if not r.get("error") and not r.get("invalid_spawn")]
    if not rs:
        return {"n": 0}
    mins = [max(0.0, r.get("min_clearance") or 0.0) for r in rs]
    fr = sum(r.get("frames", 600) for r in rs)
    lap1 = [r["laps"][0]["elapsed"] for r in rs if r.get("laps")]
    return {"n": len(rs), "collisions": sum(1 for r in rs if r.get("collisions")), "min_cm": round(100 * min(mins), 2),
            "lt5": sum(1 for m in mins if m < .05), "lt8": sum(1 for m in mins if m < .08),
            "valid_lap": sum(1 for r in rs if r.get("valid_episode") and r.get("valid_laps", 0) >= 1),
            "shield_frac": round(sum(r.get("shield_frames", 0) for r in rs) / max(1, fr), 4),
            "lap1_median": round(statistics.median(lap1), 3) if lap1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--off", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R = load(a.rows)
    OFF = load(a.off) if a.off else {}
    REF = load(a.ref) if a.ref else {}
    out = {}
    for s in sorted(R):
        if s.endswith("OFF"):
            continue
        res = {"on": summ(list(R[s].values()))}
        off = R.get(s + "OFF") or OFF.get(s + "OFF") or {}
        if off:
            res["off"] = summ(list(off.values()))
            saved = [i for i, r in R[s].items() if i in off and off[i].get("collisions") and not r.get("collisions")]
            caused = [i for i, r in R[s].items() if i in off and not off[i].get("collisions") and r.get("collisions")]
            both = [i for i, r in R[s].items() if i in off and off[i].get("collisions") and r.get("collisions")]
            zero_sh = [i for i, r in R[s].items() if r.get("collisions") and not r.get("shield_frames")]
            res["saved"], res["caused"], res["both"], res["collided_zero_shield"] = len(saved), caused, len(both), zero_sh
        ref = REF.get(s) or {}
        if ref:
            res["ref"] = summ(list(ref.values()))
            common = [i for i in R[s] if i in ref]
            res["ref_newly_collide"] = [i for i in common if R[s][i].get("collisions") and not ref[i].get("collisions")]
            res["ref_fixed"] = len([i for i in common if not R[s][i].get("collisions") and ref[i].get("collisions")])
        if any(r.get("corner") for r in R[s].values()):
            by = {}
            for r in R[s].values():
                by.setdefault(r.get("corner"), []).append(r)
            res["by_corner"] = {k: summ(v) for k, v in sorted(by.items())}
        out[s] = res
        print(s, json.dumps(res, ensure_ascii=False))
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
