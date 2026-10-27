"""反証 r2-rec2-4: 本物のラベル(読むだけ)で、AI の区間の終わりから 0.5 秒以内の人側の衝突がどれだけあるか"""
import json
import sys
from pathlib import Path
import numpy as np

LD = Path('/home/jetson/jetformer/data/labels')
rows = []
n_ai_coll = 0
n_logs = 0
for pj in sorted(LD.glob('*.json')):
    pn = pj.with_suffix('.npz')
    if not pn.exists():
        continue
    try:
        info = json.loads(pj.read_text(encoding='utf-8'))
        with np.load(pn) as z:
            t = z['t_s'].astype(float)
            ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
    except Exception as e:
        continue
    ev = info.get('events') or {}
    if not ai.any():
        continue
    n_logs += 1
    # AI の区間の終わり(ai が True → False)
    ends = t[1:][ai[:-1] & ~ai[1:]]
    for c in ev.get('collisions', []):
        if c.get('ai'):
            n_ai_coll += 1
            continue
        d = c['t_s'] - ends
        d = d[(d >= 0) & (d <= 0.5)]
        if len(d):
            rows.append((pj.stem, c['t_s'], c['g'], c.get('loc'), round(float(d.min()), 3)))
print('AI のある記録', n_logs, '本・AI 側の衝突', n_ai_coll, '件')
print('AI の終わりから 0.5 秒以内の人側の衝突', len(rows), '件・8G 以上', sum(r[2] >= 8 for r in rows),
      '・4G 以上', sum(r[2] >= 4 for r in rows))
for r in rows:
    print(r)
