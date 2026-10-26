"""fix/r1_bump.py — R1 の入口の上下の衝撃(3 軸)を、走行ごと・通過ごとに数える(反証 大 1 の確かめ)。
生の記録(data/raw の IMU・VEHICLE)と scorecard の通過(ta_s = R1 の起点・entry_erpm)だけを読む。
偽の起点(tt_data.json の false_end の終わり)は外す。立て直し中かどうかは見ない(上下の衝撃は衝突ではないので)。
動かし方: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I r1_bump.py
出力: /home/jetson/jetformer_reports/bench_20261024_report/fix_data/r1_bump.json
"""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, '..', 'verify'))
import rawload  # noqa: E402

REP = '/home/jetson/jetformer_reports/bench_20261024_report'
SC = {'20261024': '/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json',
      '20261023': '/home/jetson/jetformer_reports/bench_20261023_day1/nightly/scorecard/scorecard.json'}
W0, W1 = 0.0, 1.2      # R1 の起点からの窓(秒)
THR = 2.5              # 3 軸 |‖a‖−1G| のしきい値(scorecard の cmap と同じ形)

tt = json.load(open(os.path.join(REP, 'tt_data.json')))
runs = {r['log']: r for r in tt['runs']}
false_t = {}
for l in tt['laps']:
    if l.get('false_end'):
        false_t.setdefault(l['log'], []).append(l['t_end'])

out = {'window_s': [W0, W1], 'thr_g': THR, 'runs': [], 'passes': []}
for day, p in SC.items():
    sc = json.load(open(p))
    for lg, x in sorted(sc['per_log'].items()):
        if lg not in runs:
            continue
        ru = runs[lg]
        r = rawload.load(lg)
        it, acc = r['i_t'], r['i_acc']
        er = np.interp(it, r['v_t'], r['v_erpm'])
        st = np.abs(er) < 1
        b = np.median(acc[st], axis=0) if st.sum() > 50 else np.median(acc, axis=0)
        ah = np.hypot(acc[:, 0] - b[0], acc[:, 1] - b[1]) / rawload.G
        az = (acc[:, 2] - b[2]) / rawload.G
        a3 = np.abs(np.linalg.norm(acc, axis=1) / rawload.G - 1)
        rows = []
        for ps in x['passes']:
            if not ps.get('ai'):
                continue
            ta = ps['ta_s']
            if any(abs(ta - f) < 0.3 for f in false_t.get(lg, [])):
                continue
            m = (it >= ta + W0) & (it <= ta + W1)
            if m.sum() < 5:
                continue
            k = np.argmax(a3[m])
            idx = np.where(m)[0][k]
            row = dict(log=lg, no=ru['no'], session=ru['session'], setting=ru['setting'], ta_s=round(ta, 2),
                       entry_erpm=ps.get('entry_erpm'), a3_max=round(float(a3[m].max()), 2),
                       ah_max=round(float(ah[m].max()), 2), az_min=round(float(az[m].min()), 2),
                       az_max=round(float(az[m].max()), 2), t_peak_after=round(float(it[idx] - ta), 2),
                       bump=bool(a3[m].max() >= THR), horiz_hit=bool(ah[m].max() >= THR))
            rows.append(row)
            out['passes'].append(row)
        n = len(rows)
        nb = sum(rr['bump'] for rr in rows)
        nbv = sum(rr['bump'] and not rr['horiz_hit'] for rr in rows)
        er_med = float(np.median([rr['entry_erpm'] for rr in rows if rr['entry_erpm']])) if n else None
        a3_med = float(np.median([rr['a3_max'] for rr in rows])) if n else None
        out['runs'].append(dict(log=lg, no=ru['no'], session=ru['session'], setting=ru['setting'], passes=n,
                                bump=nb, bump_vertical_only=nbv, entry_erpm_median=er_med, a3_max_median=a3_med))
        print('%s No.%-3s %-10s %-22s 通過 %2d  3軸≥%.1fG %2d(上下だけ %2d)  進入 eRPM 中央 %s  3軸の最大の中央 %s' % (
            lg, ru['no'], ru['session'], ru['setting'], n, THR, nb, nbv,
            '%.0f' % er_med if er_med else '-', '%.2f' % a3_med if a3_med is not None else '-'))

# 進入の速さの帯ごと(B のすべての AI 通過)
P = [p for p in out['passes'] if p['entry_erpm']]
bands = [(0, 5000), (5000, 5500), (5500, 6000), (6000, 99999)]
out['bands'] = []
for lo, hi in bands:
    s = [p for p in P if lo <= p['entry_erpm'] < hi]
    nb = sum(p['bump'] for p in s)
    out['bands'].append(dict(lo=lo, hi=hi, passes=len(s), bump=nb,
                             a3_median=float(np.median([p['a3_max'] for p in s])) if s else None))
    print('進入 %5d〜%5d eRPM: 通過 %3d  3軸≥2.5G %3d (%.0f%%)  3軸の最大の中央 %.2f' % (
        lo, hi, len(s), nb, 100 * nb / max(len(s), 1), out['bands'][-1]['a3_median'] or 0))
x = np.array([p['entry_erpm'] for p in P], float)
y = np.array([p['a3_max'] for p in P], float)
out['corr_entry_erpm_a3'] = float(np.corrcoef(x, y)[0, 1])
print('相関(進入 eRPM と 3 軸の最大): %.2f(%d 通過)' % (out['corr_entry_erpm_a3'], len(P)))
no9 = [p for p in P if p['log'] == '20261024-162033']
x9 = np.array([p['entry_erpm'] for p in no9], float)
y9 = np.array([p['a3_max'] for p in no9], float)
out['corr_no9'] = float(np.corrcoef(x9, y9)[0, 1])
print('No.9 の中: 相関 %.2f(%d 通過)' % (out['corr_no9'], len(no9)))
for p in no9:
    print('  No.9 t=%.2f 進入 %s eRPM  3軸 %.2f  水平 %.2f  上下 %.2f〜%.2f  起点の %.2f 秒あと' % (
        p['ta_s'], p['entry_erpm'], p['a3_max'], p['ah_max'], p['az_min'], p['az_max'], p['t_peak_after']))
os.makedirs(os.path.join(REP, 'fix_data'), exist_ok=True)
json.dump(out, open(os.path.join(REP, 'fix_data', 'r1_bump.json'), 'w'), ensure_ascii=False, indent=1)
print('->', os.path.join(REP, 'fix_data', 'r1_bump.json'))
