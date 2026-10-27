"""反証 r2-rec2-4: 0.5 秒以内の人側の衝突の前後(交代の種類・速さ)と、AI 側の 8G 以上の数(読むだけ)"""
import json
from pathlib import Path
import numpy as np

LD = Path('/home/jetson/jetformer/data/labels')
ai8 = 0
for pj in sorted(LD.glob('*.json')):
    pn = pj.with_suffix('.npz')
    if not pn.exists():
        continue
    try:
        info = json.loads(pj.read_text(encoding='utf-8'))
        with np.load(pn) as z:
            t = z['t_s'].astype(float)
            ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
    except Exception:
        continue
    ev = info.get('events') or {}
    if not ai.any():
        continue
    ends = t[1:][ai[:-1] & ~ai[1:]]
    for c in ev.get('collisions', []):
        if c.get('ai'):
            ai8 += c['g'] >= 8
            continue
        d = c['t_s'] - ends
        d = d[(d >= 0) & (d <= 0.5)]
        if not len(d):
            continue
        tk = [x for x in ev.get('takeovers', []) if abs(x['t_s'] - (c['t_s'] - d.min())) < 0.2]
        print(pj.stem, 't', c['t_s'], 'g', c['g'], c.get('loc'), 'dt', round(float(d.min()), 3),
              'v_before', c.get('speed_before_mps'), 'erpm_b', c.get('erpm_before'), 'outcome', c.get('outcome'),
              'side', c.get('side'), 'takeover', [(x.get('class'), x.get('failure')) for x in tk],
              'desc', (info.get('description') or '')[:50])
print('AI 側の 8G 以上', ai8)
