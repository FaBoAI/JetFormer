#!/usr/bin/env python3
"""DE の線: DE に入ってからのステアを、人の通過・AI のぶつからない通過・ぶつかった通過で比べる(読むだけ)。
- 人 = 10/10〜10/11 の人の記録(ai_log でない)で、DE の区間ずっと 1 m/s 以上・衝突なしの通過。ステアは human_pwm
- AI = 10/23 の公式 8 本(ai_no 8〜15)の、ずっと AI が操作していた通過。ステアは車に出した値 out
- 値は中央からの µs。+ = 左(jetformer/tof_guard.py の約束)。中央は記録の ref_us.steer_center。ただし人の記録(ai_log でない)は
  ref_us が既定の 1500 のままなので、同じ日の AI の記録の中央(10/10 は 1443 µs)を使う(10/24 に直した。1500 で引くと人が −60 µs に見える)
- peak = 0.3 秒から衝突(ぶつからない通過は 0.95 秒)までの左への切りの最大
    python3 de_line.py   → de_line.json(fig_de_line.py が読む)"""
import glob, json, os
import numpy as np
LAB = '/home/jetson/jetformer/data/labels'
HERE = os.path.dirname(os.path.abspath(__file__))
runs = {r['log']: r for r in json.load(open(os.path.join(HERE, 'runs.json'), encoding='utf-8'))}
T = np.arange(0, 1.7, 1 / 30)
HUMAN_CENTER = {'20261010': 1443.0, '20261011': 1443.0}     # 人の記録の ref_us は 1500 のまま → 同じ日の AI の記録の中央


def prof(z, a, b, col, c0):
    t = z['t_s'][a:b] - z['t_s'][a]
    s = col[a:b] - c0
    ok = np.isfinite(s)
    return np.interp(T, t[ok], s[ok]) if ok.sum() > 10 else None


G = {'human': [], 'ai_clean': [], 'ai_hit': []}
peak = {'ai_clean': [], 'ai_hit': []}
hit_t = []
for f in sorted(glob.glob(LAB + '/2026101*.json')) + sorted(glob.glob(LAB + '/20261023-*.json')):
    d = json.load(open(f, encoding='utf-8'))
    z = np.load(f[:-5] + '.npz')
    c0 = (d.get('ref_us') or {}).get('steer_center', 1441.0)
    if not d.get('ai_log'):
        c0 = HUMAN_CENTER.get(d['log'][:8], c0)
    de = z['loc_code'] == 3
    idx = np.flatnonzero(np.diff(np.r_[0, de.astype(int), 0]))
    r = runs.get(d['log'])
    for a, b in zip(idx[::2], idx[1::2]):
        if b - a < 40:
            continue
        b2 = min(len(de), b + 9)
        if not d.get('ai_log'):
            if np.nanmin(z['speed_mps'][a:b]) < 1.0 or z['collision'][a:b2].any():
                continue
            p = prof(z, a, b, z['human_pwm'][:, 0], c0)
            if p is not None:
                G['human'].append(p)
        elif r and r['group'] == 'OFFICIAL' and r['ai_no'] >= 8 and z['ai'][a:b].all():
            h = np.flatnonzero(z['collision'][a:b2] & z['ai'][a:b2])
            k = 'ai_hit' if len(h) else 'ai_clean'
            G[k].append(prof(z, a, b, z['out'][:, 0], c0))
            end = a + (h[0] if len(h) else int(0.95 * 30))
            peak[k].append(float((z['out'][a + 9:max(end, a + 10), 0] - c0).max()))
            if len(h):
                hit_t.append(float(z['t_s'][a + h[0]] - z['t_s'][a]))
out = {}
for k, v in G.items():
    S = np.array(v)
    out[k] = {'n': len(S), 't': T.tolist(), 'med': np.median(S, 0).tolist(), 'q25': np.percentile(S, 25, 0).tolist(),
              'q75': np.percentile(S, 75, 0).tolist()}
    print(k, len(S), ' '.join('%.1f:%+.0f' % (x, np.median(S[:, int(x * 30)])) for x in (0, .2, .4, .6, .8, .9, 1.0, 1.2, 1.5)))
out['peak'], out['hit_t'] = peak, hit_t
json.dump(out, open(os.path.join(HERE, 'de_line.json'), 'w'))
