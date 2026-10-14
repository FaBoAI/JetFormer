"""no_wall r3 の下調べの比べ(2026-10-14): 同じ組を 2 つの走らせ方(シールド v4 / v4.1)で走らせた行を並べる。判定には使わない。
    python3 compare_dev.py <rows_a.jsonl> <rows_b.jsonl> [名前a 名前b] [--json 出力]"""
import json
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from judge_nw3 import load, set_stats  # noqa: E402

a, b = sys.argv[1], sys.argv[2]
na, nb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[3].startswith("--") else ("A", "B")
out = {}
for nm, p in ((na, a), (nb, b)):
    rows, by = load(p)
    out[nm] = {}
    for k, rs in by.items():
        st = set_stats(rs)
        probe = sum((r["labels"]["fb"].get("brake_probe", 0) + sum(v for kk, v in r["labels"]["fb"].items() if "brake_probe" in kk and kk != "brake_probe")) for r in rs)
        st["probe_frames_per_run"] = probe / len(rs)
        st["lap0"] = sum(1 for r in rs if r.get("valid_laps", 0) == 0)
        out[nm][k] = st
keys = ["n", "collisions", "stall", "valid_episode_r3", "lap0", "min_clearance_m", "below_8cm", "shield_fraction", "shield_fraction_governor", "shield_run_max", "probe_frames_per_run", "lap1_median_s", "mean_running_speed_mps"]
for k in sorted(out[na]):
    print(f"== {k}")
    for nm in (na, nb):
        s = out[nm].get(k, {})
        print(f"  {nm:6s} " + " ".join(f"{x}={round(s[x], 4) if isinstance(s.get(x), float) else s.get(x)}" for x in keys))
        if s.get("collision_ids"):
            print("        衝突:", [(c["id"], round(c["t"] or 0, 2)) for c in s["collision_ids"][:8]])
        if s.get("stall_ids"):
            print("        止まり:", s["stall_ids"][:8])
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), ensure_ascii=False, indent=1)
