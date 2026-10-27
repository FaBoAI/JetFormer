"""反証(w2-safety-rv7-safety-r2-1): 本物の AI 側のカメラの空き(処理したフレームの間 > 0.1 秒)の長さの分け方(data/raw を読むだけ)"""
import glob, os, json
import numpy as np
from collections import Counter
ROOT = '/home/jetson/jetformer/data/raw'
neu = []
per_day = Counter()
ai_s = Counter()
for fp in sorted(glob.glob(ROOT + '/2026*/*/AI_CMD.npz')):
    if os.path.getsize(fp) == 0:
        continue
    try:
        d = np.load(fp)
        t = d['timestamps_us'].astype(np.int64) / 1e6
        sel = d['select'].astype(bool); hold = d['hold'].astype(bool); dry = d['dry_run'].astype(bool)
    except Exception:
        continue
    day = fp.split('/')[-3][:8]
    ai = sel & ~dry & ~hold
    dt = np.diff(t)
    ok = ai[:-1] & ai[1:]
    ai_s[day] += float(np.clip(dt[ok], 0, 0.2).sum())
    for x in dt[ok & (dt > 0.1)]:
        neu.append((day, round(float(x - 0.1), 3)))
        per_day[day] += 1
vals = np.array([v for _, v in neu if _ in ('20261023', '20261024')])
print('ai_s', {k: round(v) for k, v in ai_s.items()})
print('n gaps', dict(per_day))
for lim in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2):
    print('neu < %.2f: %d' % (lim, int((vals < lim).sum())))
print('quantiles', np.round(np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9]), 3))
