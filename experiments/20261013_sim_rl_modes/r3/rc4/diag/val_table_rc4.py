"""離脱復活 r4: 検証の組の結果をチェックポイントごとに 1 行で(報告用)。  python3 -I r3/rc4/diag/val_table_rc4.py <学習の出力>"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
for p in sorted((run / "evals" / "recover").glob("step_*.json")):
    d = json.loads(p.read_text())
    J, rep = d["judged"], d["reported"]
    c5 = J["C5_r4"]["value"]
    print(p.stem, "全部" if d["all_judged_pass"] else f"{d['n_passed']}/8", "C1", J["C1_r3"]["value"], "C2 %.3f" % J["C2"]["value"],
          "C3 %.2f/%.2f" % (J["C3"]["value"]["median"], J["C3"]["value"]["p90"]), "C4 %.2f/%.3f" % (J["C4"]["value"]["events_per_episode"], J["C4"]["value"]["masked_fraction"]),
          "C5", c5["grid_yaw"], c5["grid_lateral"], c5["random_yaw"], c5["random_lateral"],
          "C6 %.2fs %s %s %.2fcm" % (J["C6_r4"]["value"]["lap_from_stop"] or -1, J["C6_r4"]["value"]["rec60"], J["C6_r4"]["value"]["unperturbed"], (J["C6_r4"]["value"]["min_clearance"] or 0) * 100),
          "C7", "ok" if J["C7"]["pass"] else "NG", "C8", J["C8"]["value"]["perturbed_valid_with_miss"], J["C8"]["value"]["unperturbed_with_miss"],
          "band", rep.get("kick_band", {}).get("band_right"), rep.get("kick_band", {}).get("band_left"), rep.get("kick_band", {}).get("outside"),
          "invalid", [(r["id"], r["reason"], r["stall"]) for r in rep.get("invalid_p", [])], "unpert_fail", rep.get("unperturbed_fail_ids"),
          "replaced %.2f" % (rep.get("replaced_fraction") or 0), "shield %.4f" % rep["shield"]["all_rows"], "wall %ds" % d["wall_s"])
