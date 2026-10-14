"""no_wall r2 の線のちがい(確認 r1 の analyze_nw.py と同じ数え方。2026-10-14)。標準の物理・新しい 150 の開始(FNOM)で、
r2 と最速(確認 r1 の FAST)・r1(確認 r1 の FNOM)の 5 cm のマスの平均 e_y の差、走っているときの速さ、1 周目、モードの中の線のばらつき。

    python line_compare.py <r2 の rows_FNOM.jsonl> <確認 r1 の rows_FNOM_FAST_LOWMU.jsonl> <出力.json>
"""
import json
import math
import sys
from collections import defaultdict

L = 26.12


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = min(len(xs) - 1, f + 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def binned(rows, idx):
    B = defaultdict(list)
    for r in rows:
        seen, dist, prev = set(), 0, None
        for p in r["path"]:
            x, z, s = p[0], p[1], p[3]
            if prev is not None:
                dist += math.hypot(x - prev[0], z - prev[1])
            prev = (x, z)
            if dist < 3:
                continue
            b = int((s % L) / .05)
            if b in seen:
                continue
            seen.add(b)
            B[b].append(p[idx])
    return B


def spread(B, nmin):
    out = []
    for xs in B.values():
        if len(xs) >= nmin:
            m = sum(xs) / len(xs)
            out.append(sum(abs(x - m) for x in xs) / len(xs))
    return sum(out) / max(1, len(out))


def compare(a, b):
    Ba, Bb = binned(a, 4), binned(b, 4)
    Va, Vb = binned(a, 2), binned(b, 2)
    na, nb = int(.8 * len(a)), int(.8 * len(b))
    d, vd = [], []
    for k in Ba:
        if k in Bb and len(Ba[k]) >= na and len(Bb[k]) >= nb:
            d.append(abs(sum(Ba[k]) / len(Ba[k]) - sum(Bb[k]) / len(Bb[k])))
            vd.append(sum(Va[k]) / len(Va[k]) - sum(Vb[k]) / len(Vb[k]))
    return {"line_diff_mean_cm": 100 * sum(d) / max(1, len(d)), "line_diff_p90_cm": 100 * q(d, .9), "line_diff_max_cm": 100 * max(d), "bins": len(d),
            "speed_diff_mean_mps": sum(vd) / max(1, len(vd))}


r2 = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ck = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
r1 = {r["id"]: r for r in ck if r["set"] == "FNOM"}
fa = {"FN" + r["id"][2:]: r for r in ck if r["set"] == "FAST"}
ok = lambda r: r["collisions"] == 0 and r["valid_episode"]  # noqa: E731
ids = [r["id"] for r in r2 if r["id"] in r1 and r["id"] in fa]
R2 = {r["id"]: r for r in r2}
a = [R2[i] for i in ids if ok(R2[i])]
b1 = [r1[i] for i in ids if ok(r1[i])]
bf = [fa[i] for i in ids if ok(fa[i])]
res = {"starts": len(ids), "valid": {"r2": len(a), "r1": len(b1), "fastest": len(bf)},
       "r2_vs_fastest": compare(a, bf), "r2_vs_r1": compare(a, b1), "r1_vs_fastest": compare(b1, bf),
       "within_mode_spread_cm": {"r2": 100 * spread(binned(a, 4), int(.8 * len(a))), "r1": 100 * spread(binned(b1, 4), int(.8 * len(b1))),
                                 "fastest": 100 * spread(binned(bf, 4), int(.8 * len(bf)))},
       "mean_running_speed": {k: sum(r["mean_running_speed"] for r in v) / len(v) for k, v in (("r2", a), ("r1", b1), ("fastest", bf))},
       "lap1_median_s": {k: q([r["laps"][0]["elapsed"] for r in v if r["laps"]], .5) for k, v in (("r2", a), ("r1", b1), ("fastest", bf))},
       "min_clearance_median_cm": {k: 100 * q([r["min_clearance"] for r in v], .5) for k, v in (("r2", a), ("r1", b1), ("fastest", bf))},
       "below_8cm": {"r2": sum(1 for r in a if r["min_clearance"] < .08), "r1": sum(1 for r in b1 if r["min_clearance"] < .08)}}
json.dump(res, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
