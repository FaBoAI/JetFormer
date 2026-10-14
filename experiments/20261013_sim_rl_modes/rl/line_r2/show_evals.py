"""検証の組の結果を 1 行ずつ表示する(経路いろいろ r2 の学習の見張り)。  python3 rl/line_r2/show_evals.py <学習の出力の場所>"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
for p in sorted((run / "evals" / "line").glob("step_*.json")):
    d = json.loads(p.read_text())
    J = d["judged"]
    v4 = J["L4_distinct"].get("value") or {}
    mark = lambda k: "o" if J[k]["pass"] else "x"  # noqa: E731
    print(f"{int(p.stem.split('_')[1]):>11,} L1 {J['L1_valid']['value']}{mark('L1_valid')} L2 {J['L2_clearance']['value'] * 100:5.2f}cm{mark('L2_clearance')} "
          f"L3 {J['L3_tracking']['value']['median'] * 100:4.1f}/{J['L3_tracking']['value']['p95'] * 100:5.1f}cm{mark('L3_tracking')} "
          f"L4 {v4.get('left_right_mean', 0):.3f}/{v4.get('inside_outside_mean', 0):.3f}/{v4.get('min_pair_max_1m_avg', 0):.3f}{mark('L4_distinct')} "
          f"L5 {J['L5_vm3']['value']:.4f}{mark('L5_vm3')} L6 {J['L6_lap']['value'] or 0:.2f}{mark('L6_lap')} {'ALL' if d['all_judged_pass'] else ''}")
