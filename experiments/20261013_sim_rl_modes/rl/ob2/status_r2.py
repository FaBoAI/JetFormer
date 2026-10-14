"""障害物よけ r2 の学習の様子(学習中の集計と、チェックポイントごとの評価)を短く出す。2026-10-14
    python3 rl/ob2/status_r2.py [train/obstacle/r2/main]
"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/obstacle/r2/main")
rows = [json.loads(x) for x in (run / "progress.jsonl").read_text().splitlines() if x.strip()]
last = rows[-20:]
n = sum(r["episodes"]["n"] for r in last)
rs = {}
for r in last:
    for k, v in r["episodes"].get("reasons", {}).items():
        rs[k] = rs.get(k, 0) + v
import math
oc = [r["episodes"].get("min_obstacle_clearance_mean") for r in last if r["episodes"].get("min_obstacle_clearance_mean") is not None]
l1 = [r["episodes"].get("lap1_mean") for r in last if r["episodes"].get("lap1_mean")]
print(f"{rows[-1]['t']} {rows[-1]['steps']:,} steps {rows[-1]['steps_per_s_run']:.0f}/s 直近20更新: 回 {n} 走り切り {rs.get('collection_duration_complete', 0) / max(1, n):.3f} "
      f"理由 {rs} 障害物すき間 {sum(oc) / max(1, len(oc)) * 100:.1f}cm 1周目 {sum(l1) / max(1, len(l1)):.2f} σ {[round(math.exp(x), 3) for x in rows[-1]['log_std']]} KL {rows[-1]['approx_kl']:.4f}")
for p in sorted((run / "policies").glob("step_*.json")):
    e2, ep = run / "evals_r2" / p.name, run / "evals" / "obstacle" / p.name
    s = p.stem
    if e2.exists():
        d = json.loads(e2.read_text()); j = d["judged"]; r = d["reported"]
        s += (f" | 選び {j['O1_valid']['value']} O2 {j['O2_clearance']['value'] and round(j['O2_clearance']['value'] * 100, 2)}cm O3 {j['O3_lap1_median']['value'] and round(j['O3_lap1_median']['value'], 2)} "
              f"O4 {j['O4_empty']['value'] and round(j['O4_empty']['value'], 3)} VAR6max {r['var6']['max'] and round(r['var6']['max'], 3)} 早い衝突 {r['failures']['early_le_2_1s']} "
              f"3cm未満 {r['valid_below_3cm']} VM3 {r['VM3_valid_runs'] and (round(r['VM3_valid_runs']['mean_abs_dev'], 4) if r['VM3_valid_runs']['mean_abs_dev'] is not None else None)} 全部 {d['all_judged_pass']}")
    if ep.exists():
        d = json.loads(ep.read_text()); j = d["judged"]
        s += f" | 検証27 {j['O1_valid']['value']} O2 {j['O2_clearance']['value'] and round(j['O2_clearance']['value'] * 100, 2)}"
    print(s)
# 有効だが周を回っていない回(止まったまま 20 秒 = valid_episode の定義には入るが見本にならない)
for p in sorted((run / "evals_r2").glob("step_*.json")):
    d = json.loads(p.read_text())
    ok = [r for r in d["rows"] if r["test"] == "OB-V100-R2" and r["valid_episode"]]
    nolap = [r["id"] for r in ok if len(r["laps"]) < 1]
    lt2 = [r["id"] for r in ok if len(r["laps"]) < 2]
    print(p.stem, "有効", len(ok), "周 0 の有効", len(nolap), "周 < 2 の有効", len(lt2), nolap[:4])
