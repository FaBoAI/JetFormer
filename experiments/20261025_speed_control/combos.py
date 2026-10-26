#!/usr/bin/env python3
"""速さの天井の 9 組(全体 4.3/4.5/4.8 × 左旋回中 3.8/4.0/4.2)の下見。左旋回 = 舵 ≥ +100 µs または gz ≥ G_LEFT deg/s(環境変数、既定 60)
(切れてから 0.25 秒保つ)。理想の天井(惰性 + こすれで落ちて天井で止まる・モデルの反応なし)の縦の仮想の再生で、
(1) 出っ張り 9 件・LH 3 件の当たる時刻の仮想の速さ (2) No.9・2 回目の周の失う時間 (3) 天井を超えたフレームの割合。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I combos.py
"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
sys.argv = [sys.argv[0], os.path.join(H, 'ceiling_shadow.json')]
import ceiling_shadow as cs  # noqa: E402  (読み込みで 1 回まわる)

def vtrace(d, cap, a0, a1):
    """[a0, a1) の仮想の速さの列(フレームごと)"""
    at, v = d['at'], d['v']
    idx = np.nonzero((at >= a0) & (at < a1))[0]
    out = np.full(len(idx), np.nan)
    if len(idx) < 2:
        return idx, out
    vp = v[idx[0]]
    out[0] = vp
    for q in range(len(idx) - 1):
        i, i2 = idx[q], idx[q + 1]
        dt = at[i2] - at[i]
        vr, vr2, c = v[i], v[i2], cap[i]
        tgt = min(vr2, c)
        if dt <= 0 or dt > 0.2 or vr < 0.3:
            vn = vr2
        elif vp > c + 1e-6:
            vn = min(vr2, max(c, vp - (cs.C0 + cs.KS * vp * vp * abs(d['kap'][i])) * dt))
        elif vp >= vr - 1e-6:
            vn = tgt
        else:
            vn = min(tgt, vp + max(min(cs.A_CAP, (tgt - vp) / cs.TAU_UP), (vr2 - vr) / dt) * dt)
        out[q + 1] = vn
        vp = vn
    return idx, out

G_LEFT = float(os.environ.get("G_LEFT", "60"))
rows = []
for CA in (4.3, 4.5, 4.8):
    for CL in (3.8, 4.0, 4.2):
        row = {'all': CA, 'left': CL, 'events': []}
        caps = {}
        for log, d in cs.D.items():
            L = cs.left_mask(d["at"], d["steer"], d["gz5"], 100, G_LEFT)
            caps[log] = np.where(L, CL, CA)
        for ev in cs.res['events']:
            d = cs.D[ev['log']]
            idx, vv = vtrace(d, caps[ev['log']], ev['t_hit'] - 3.0, ev['t_hit'] - 0.03)
            row['events'].append({'kind': ev['kind'], 'no': ev['no'], 't': ev['t_hit'], 'v_rec': round(float(d['v'][idx[-1]]), 2),
                                  'v_virt': round(float(vv[-1]), 2), 'vmax_rec': ev['v_max_2s'],
                                  'vmax_virt': round(float(np.nanmax(vv[-60:])), 2)})
        for key, f in (('No.9', lambda l: l['no'] == '9'), ('2回目', lambda l: l['session'] == '10/24 2回目'),
                       ('3回目', lambda l: l['session'] == '10/24 3回目'), ('10/23', lambda l: l['session'] == '10/23 公式')):
            Ls = [l for l in cs.laps if f(l) and l['log'] in cs.D and 'caps' in cs.D[l['log']]]
            lo = [cs.virtual_loss(cs.D[l['log']], caps[l['log']], l['t_start'], l['t_end']) for l in Ls]
            lo = [x for x in lo if x]
            row[key] = {'laps': len(lo), 'loss_mean': round(float(np.mean([x[0] for x in lo])), 3) if lo else None,
                        'loss_max': round(float(np.max([x[0] for x in lo])), 3) if lo else None,
                        'touched_mean_s': round(float(np.mean([x[1] for x in lo])), 2) if lo else None}
        fr = {}
        for log, d in cs.D.items():
            mv = d['ai'] & (d['v'] > 0.5)
            if mv.sum() >= 60:
                fr[log] = round(float((mv & (d['v'] > caps[log])).sum() / mv.sum()), 4)
        row['over_frac'] = fr
        sw = [e for e in row['events'] if e['kind'] == 'SWX']
        row['swx_virt_le_cl+0.05'] = sum(1 for e in sw if e['v_virt'] <= CL + 0.05)
        rows.append(row)
        print('全体 %.1f 左 %.1f | SWX 仮想の当たる速さ ≤ 左+0.05: %d/%d | No.9 失う %.3f s/周(最大 %.3f) 2回目 %.3f | No.9 超え %.1f%%'
              % (CA, CL, row['swx_virt_le_cl+0.05'], len(sw), row['No.9']['loss_mean'], row['No.9']['loss_max'],
                 row['2回目']['loss_mean'], 100 * fr.get('20261024-162033', 0)))
        if CA == 4.5 and CL == 4.0:
            for e in row['events']:
                print('   ', e)
json.dump(rows, open(os.path.join(H, 'combos_g%d.json' % int(G_LEFT)), 'w'), ensure_ascii=False, indent=1)
