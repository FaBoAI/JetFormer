#!/usr/bin/env python3
"""select_teacher.mjs を走行の組に分けて並べて走らせた結果をまとめ、先生(合計の多いほう。同じなら T_margin)と k を決める
(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 2・3)

    python3 -I merge_select.py --out <まとめ.json> <部分 1.json> <部分 2.json> ...
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('parts', nargs='+')
a = ap.parse_args()
rows, every = [], None
for p in a.parts:
    d = json.loads(Path(p).read_text())
    rows += d['rows']
    every = d['every']
totals = {k: sum(r[k]['recoverable'] for r in rows) for k in ('fast', 'margin')}
tried = {k: sum(r[k]['tried'] for r in rows) for k in ('fast', 'margin')}
teacher = 'fast' if totals['fast'] > totals['margin'] else 'margin'
for r in rows:
    r['k'] = r[teacher]['takeover']['k']
rows.sort(key=lambda r: int(r['name'].split('-D')[1].split('-')[0]))
out = {'schema': 'jf-sim-diverse-dagger-teacher-select/v1', 'rule': 'protocol.json dagger.procedure 2・3(合計の多いほうの先生。同じなら T_margin)',
       'validity_frames': 60, 'every': every, 'totals': totals, 'tried': tried, 'teacher': teacher, 'parts': a.parts, 'rows': rows}
Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print(json.dumps({'totals': totals, 'tried': tried, 'teacher': teacher, 'k': [(r['name'], r['k'], r['frames'], r['reason']) for r in rows]}, ensure_ascii=False))
