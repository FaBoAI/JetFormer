#!/usr/bin/env python3
"""速さの天井の下見(2026-10-25、spec.md のため)。読むだけ。生の記録・ラベル・設定は変えない。GPU なし。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I ceiling_shadow.py [出力.json]

1. 「左に曲がっている最中」の決め方の候補(車に出した舵 out − 中央 ≥ S、ジャイロ gz(5 サンプル平均)≥ G、その OR)ごとに、
   AI のフレーム(AI 側・dry run でない・前へ 0.5 m/s 超)のうち天井(全体 4.5・左旋回中 4.0 m/s)を超えたフレームの割合。
2. 駐車場の出っ張り 9 件・LH の黒いカーテン 3 件(cam_check.json の SWX・LHC)で、当たる前 2.0 秒の中で天井が最初に
   掛かる時刻(当たる何秒前か)と、そのときの速さ。
3. 理想の天井(速さが天井を超えたら、惰性 c0 + こすれ k_s·v²·|κ| で落ち、天井で止まる。モデルの反応は入れない)の
   縦の仮想の再生で、周ごとの失う時間(10/16 の g1_replay.virtual_laps の簡単な形)。
速さ v = −eRPM × 0.5e-3(撮像の時刻までの最新 3 サンプルの中央値)。舵 + = 左。
"""
import json
import math
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261024_report/verify')
import rawload  # noqa: E402

REP = '/home/jetson/jetformer_reports/bench_20261024_report'
COL = '/home/jetson/jetformer/experiments/20261024_report/col'
PCAL = '/home/jetson/jetformer_reports/sim_calibration_20261016/profiles/P_cal.json'
KV = 0.5e-3
C_ALL, C_LEFT = 4.5, 4.0
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, 'ceiling_shadow.json')

tt = json.load(open(os.path.join(REP, 'tt_data.json')))
runs = {r['log']: r for r in tt['runs']}
laps = [l for l in tt['laps'] if not l.get('false_end')]
col = json.load(open(os.path.join(COL, 'col_data.json')))
cam = json.load(open(os.path.join(COL, 'cam_check.json')))['events']
events = []
for e in col['events']:
    k = '%s@%s' % (e['log'], e['t_s'])
    if cam.get(k) in ('SWX', 'LHC'):
        events.append(dict(kind=cam[k], log=e['log'], no=e['ai_no'], t_us=e['t_us'], g=e['g'], setting=e['setting']))
lo = json.load(open(PCAL))['longitudinal']
C0, KS, TAU_UP, A_CAP = lo['c0'], lo['k_s'], lo['tau_up'], lo['a_cap']

CANDS = {
    'steer>=100': dict(S=100, G=None),
    'steer>=60': dict(S=60, G=None),
    'steer>=150': dict(S=150, G=None),
    'gz>=30': dict(S=None, G=30),
    'gz>=45': dict(S=None, G=45),
    'gz>=60': dict(S=None, G=60),
    'gz>=90': dict(S=None, G=90),
    'steer>=100|gz>=45': dict(S=100, G=45),
    'steer>=100|gz>=60': dict(S=100, G=60),
    'steer>=150|gz>=45': dict(S=150, G=45),
    'none(4.5 only)': dict(S=None, G=None),
}
HOLD_S = 0.25          # 左旋回の判定を、条件が切れてから保つ時間


def left_mask(t, steer, gz5, S, G, hold=HOLD_S):
    raw = np.zeros(len(t), bool)
    if S is not None:
        raw |= steer >= S
    if G is not None:
        raw |= gz5 >= G
    out = raw.copy()
    last = -1e9
    for i in range(len(t)):
        if raw[i]:
            last = t[i]
        elif t[i] - last <= hold:
            out[i] = True
    return out


def per_run(log):
    r = rawload.load(log)
    meta = r['meta']
    drv = meta.get('drive') or {}
    center = float(drv.get('steer_center_us') or 1441.0)
    neutral = float(drv.get('throttle_neutral_us') or 1496.0)
    at = r['a_t']
    ai = r['a_sel'] & ~r['a_dry']
    vt, ve = r['v_t'], r['v_erpm']
    j = np.searchsorted(vt, at, side='right') - 1
    jj = np.clip(np.stack([j - 2, j - 1, j]), 0, len(vt) - 1)
    v = np.median(-ve[jj] * KV, axis=0)
    v[j < 0] = 0.0
    it, gz = r['i_t'], r['i_gz']
    k = np.searchsorted(it, at, side='right') - 1
    kk = np.clip(np.stack([k - m for m in range(5)]), 0, len(it) - 1)
    gz5 = gz[kk].mean(axis=0)
    steer = r['a_out'][:, 0] - center
    thr = neutral - r['a_out'][:, 1]
    # 曲率(縦の再生のこすれ用): ヨー[rad/s] ÷ 速さ
    kap = np.where(v > 0.3, np.deg2rad(gz5) / np.maximum(v, 0.3), 0.0)
    return dict(log=log, t0_us=int(r['t0']), at=at, ai=ai, v=v, gz5=gz5, steer=steer, thr=thr, kap=kap)


def virtual_loss(d, cap_t, a0, a1):
    """[a0, a1) の AI の周の縦の仮想の再生。cap_t = フレームごとの天井。返り値 (失う時間 s, 天井で速さを変えた時間 s)"""
    at, v = d['at'], d['v']
    m = (at >= a0) & (at < a1)
    idx = np.nonzero(m)[0]
    if len(idx) < 30:
        return None
    vp = v[idx[0]]
    loss = 0.0
    touched = 0.0
    for q in range(len(idx) - 1):
        i, i2 = idx[q], idx[q + 1]
        dt = at[i2] - at[i]
        if dt <= 0 or dt > 0.2:
            vp = v[i2]
            continue
        vr, vr2 = v[i], v[i2]
        c = cap_t[i]
        if vr < 0.3:
            vp = vr2
            continue
        target = min(vr2, c)
        if vp > c + 1e-6:          # 天井より速い: 惰性で落ちる(天井までで止まる)
            a_ = -(C0 + KS * vp * vp * abs(d['kap'][i]))
            vn = min(vr2, max(c, vp + a_ * dt))
        elif vp >= vr - 1e-6:      # 記録どおり(または天井で止めている)
            vn = target
        else:                      # 天井で記録より遅くなった: 駆動の速さか記録の加速の大きい方で戻る
            a_rec = (vr2 - vr) / dt
            a_ = max(min(A_CAP, (target - vp) / TAU_UP), a_rec)
            vn = min(target, vp + a_ * dt)
        if vn < vr2 - 1e-6:
            touched += dt
        ds = 0.5 * (vr + vr2) * dt
        if ds > 0:
            loss += ds / max(0.5 * (vp + vn), 0.3) - dt
        vp = vn
    return loss, touched


res = {'note': __doc__, 'caps': [C_ALL, C_LEFT], 'hold_s': HOLD_S, 'cands': list(CANDS), 'runs': [], 'events': [],
       'laps': []}
D = {}
for log in sorted(runs):
    try:
        D[log] = per_run(log)
    except Exception as e:      # noqa: BLE001
        print('skip', log, e)
for log, d in D.items():
    mv = d['ai'] & (d['v'] > 0.5)
    row = {'log': log, 'no': runs[log].get('no'), 'session': runs[log].get('session'),
           'setting': runs[log].get('setting'), 'ai_frames': int(mv.sum())}
    if mv.sum() < 60:
        continue
    row['v_p90'] = round(float(np.percentile(d['v'][mv], 90)), 2)
    row['v_max'] = round(float(np.max(d['v'][mv])), 2)
    for name, c in CANDS.items():
        L = left_mask(d['at'], d['steer'], d['gz5'], c['S'], c['G'])
        cap = np.where(L, C_LEFT, C_ALL)
        over = mv & (d['v'] > cap)
        row[name] = {'left_frac': round(float((mv & L).mean() / max(mv.mean(), 1e-9)), 3),
                     'over_frac': round(float(over.sum() / mv.sum()), 4),
                     'over_all_frac': round(float((mv & (d['v'] > C_ALL)).sum() / mv.sum()), 4),
                     'over_left_frac': round(float((mv & L & (d['v'] > C_LEFT) & (d['v'] <= C_ALL)).sum() / mv.sum()), 4)}
        d.setdefault('caps', {})[name] = cap
    res['runs'].append(row)

for e in events:
    d = D.get(e['log'])
    if d is None:
        continue
    th = (e['t_us'] - d['t0_us']) / 1e6
    w = (d['at'] >= th - 2.0) & (d['at'] < th - 0.03)
    idx = np.nonzero(w)[0]
    ev = dict(e, t_hit=round(th, 3), v_max_2s=round(float(d['v'][idx].max()), 2),
              v_at_hit=round(float(d['v'][idx[-1]]), 2), steer_last=round(float(d['steer'][idx[-1]]), 1),
              gz5_last=round(float(d['gz5'][idx[-1]]), 1))
    # 当たる前の 2 秒の舵・ジャイロ(0.1 秒ごと、図と判定の下見)
    ev['trace'] = [[round(float(th - d['at'][i]), 2), round(float(d['v'][i]), 2), round(float(d['steer'][i])),
                    round(float(d['gz5'][i]))] for i in idx[::3]]
    for name in CANDS:
        cap = d['caps'][name]
        over = idx[d['v'][idx] > cap[idx]]
        if len(over):
            i0 = over[0]
            ev[name] = {'lead_s': round(float(th - d['at'][i0]), 2), 'v_then': round(float(d['v'][i0]), 2),
                        'cap_then': float(cap[i0]), 'over_s': round(float(len(over) / 30.0), 2)}
        else:
            ev[name] = None
    res['events'].append(ev)

for lp in laps:
    d = D.get(lp['log'])
    if d is None or 'caps' not in d:
        continue
    row = {'log': lp['log'], 'no': lp['no'], 'session': lp['session'], 'setting': lp['setting'],
           't_start': lp['t_start'], 'lap_s': lp['lap_s'], 'kind': lp['kind'], 'p90': lp['speed_mps']['p90']}
    for name in ('steer>=100', 'gz>=45', 'steer>=100|gz>=45', 'steer>=100|gz>=60', 'none(4.5 only)'):
        o = virtual_loss(d, d['caps'][name], lp['t_start'], lp['t_end'])
        row[name] = None if o is None else {'loss_s': round(o[0], 3), 'touched_s': round(o[1], 2)}
    res['laps'].append(row)

json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)

# ---- 要約
print('== 天井を超えたフレームの割合(AI・0.5 m/s 超)')
names = list(CANDS)
for row in res['runs']:
    print(row['no'], row['session'], row['setting'], 'p90', row['v_p90'], 'max', row['v_max'],
          ' '.join('%s=%.1f%%' % (n, 100 * row[n]['over_frac']) for n in ('none(4.5 only)', 'steer>=100', 'gz>=45',
                                                                         'steer>=100|gz>=45')))
print('== 出っ張り・LH: 天井が最初に掛かる時刻(当たる何秒前)')
for ev in res['events']:
    print(ev['kind'], ev['no'], ev['t_hit'], 'vmax', ev['v_max_2s'], 'v', ev['v_at_hit'], 'steer', ev['steer_last'],
          'gz', ev['gz5_last'], ' '.join('%s:%s' % (n, None if ev[n] is None else ev[n]['lead_s'])
                                          for n in ('none(4.5 only)', 'steer>=100', 'gz>=45', 'gz>=60',
                                                    'steer>=100|gz>=45')))
print('== 周ごとの失う時間(理想の天井・モデルの反応なし)')
for name in ('none(4.5 only)', 'steer>=100', 'gz>=45', 'steer>=100|gz>=45', 'steer>=100|gz>=60'):
    for key, f in (('No.9', lambda r: r['no'] == '9'), ('2回目', lambda r: r['session'] == '10/24 2回目'),
                   ('3回目', lambda r: r['session'] == '10/24 3回目'), ('10/23', lambda r: r['session'] == '10/23 公式'),
                   ('1回目', lambda r: r['session'] == '10/24 1回目')):
        L = [r[name]['loss_s'] for r in res['laps'] if f(r) and r.get(name)]
        T = [r[name]['touched_s'] for r in res['laps'] if f(r) and r.get(name)]
        if L:
            print(name, key, 'laps', len(L), 'loss mean %.3f max %.3f' % (np.mean(L), np.max(L)),
                  'touched mean %.2f s' % np.mean(T))
