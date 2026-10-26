#!/usr/bin/env python3
"""GM の出口(DE に入る 1.5 秒前)から DE の中までのステア(読むだけ)。人 = 10/10 の人の記録(ぶつからない・1 m/s 以上)、
AI = 10/23 公式 8 本(ずっと AI)。中央は ref_us.steer_center。ただし 10/10 の人の記録は ref_us が既定の 1500 のままなので、
同じ日の AI の記録の 1443 µs を使う。+ = 左。
    python3 de_line_wide.py → de_line_wide.json(fig_de_line.py が読む)"""
import glob, json, os
import numpy as np
LAB = '/home/jetson/jetformer/data/labels'
HERE = os.path.dirname(os.path.abspath(__file__))
runs = {r['log']: r for r in json.load(open(os.path.join(HERE, 'runs.json'), encoding='utf-8'))}
T = np.arange(-1.5, 1.7, 1 / 30)
G = {'human': [], 'ai_clean': [], 'ai_hit': []}
for f in sorted(glob.glob(LAB + '/20261010*.json')) + sorted(glob.glob(LAB + '/20261023-*.json')):
    d = json.load(open(f, encoding='utf-8'))
    z = np.load(f[:-5] + '.npz')
    c0 = 1443.0 if not d.get('ai_log') and d['log'].startswith('20261010') else d['ref_us']['steer_center']
    de = z['loc_code'] == 3
    idx = np.flatnonzero(np.diff(np.r_[0, de.astype(int), 0]))
    r = runs.get(d['log'])
    for a, b in zip(idx[::2], idx[1::2]):
        if b - a < 40 or a < 45:
            continue
        w = slice(a - 45, b)
        b2 = min(len(de), b + 9)
        if not d.get('ai_log'):
            if np.nanmin(z['speed_mps'][w]) < 1.0 or z['collision'][a - 45:b2].any():
                continue
            col, k = z['human_pwm'][:, 0], 'human'
        elif r and r['group'] == 'OFFICIAL' and r['ai_no'] >= 8 and z['ai'][w].all():
            k = 'ai_hit' if (z['collision'][a:b2] & z['ai'][a:b2]).any() else 'ai_clean'
            col = z['out'][:, 0]
        else:
            continue
        t = z['t_s'][w] - z['t_s'][a]
        s = col[w] - c0
        ok = np.isfinite(s)
        G[k].append(np.interp(T, t[ok], s[ok]))
out = {}
for k, v in G.items():
    S = np.array(v)
    out[k] = dict(n=len(S), t=T.tolist(), med=np.median(S, 0).tolist(), q25=np.percentile(S, 25, 0).tolist(),
                  q75=np.percentile(S, 75, 0).tolist())
    print(k, len(S), ' '.join('%+.1f:%+4.0f' % (x, np.median(S[:, int(round((x + 1.5) * 30))])) for x in (-0.9, -0.3, 0, 0.3, 0.6, 0.9)))
json.dump(out, open(os.path.join(HERE, 'de_line_wide.json'), 'w'))
