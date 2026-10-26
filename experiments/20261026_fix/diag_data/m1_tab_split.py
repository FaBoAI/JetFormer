#!/usr/bin/env python3
"""原因 A の診断 M1(numpy だけ・読むだけ): 保存してある C3 の予測(desk/pred/*_tab.npz)を、
直線・低速・区間・回(10/10・10/23・10/24 2 回目)・速さの帯で分けて、候補 − B を出す。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I m1_tab_split.py

書くのは同じ場所の out/m1_tab_split.json だけ。
"""
import json
import sys
from pathlib import Path

import numpy as np

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
DESK = REP / 'desk'
RUNS = Path('/home/jetson/jetformer/data/runs')
OUT = Path(__file__).resolve().parent / 'out'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
ROW0, ROW2 = 0, 2
GAIN = 0.90
SPEED_PER_ERPM = 0.5e-3
LOC = {0: '?', 1: 'R1', 2: 'GM_in', 3: 'GM_out', 4: 'DE', 5: 'SW', 6: 'HS', 7: 'LH'}
SESS = {0: '10/10', 1: '10/23', 2: '10/24-2'}
BANDS = [(-9, 0.2, 'stop<0.2'), (0.2, 1.0, '0.2-1.0'), (1.0, 2.0, '1-2'), (2.0, 3.0, '2-3'), (3.0, 4.0, '3-4'),
         (4.0, 99, '>=4')]


def car_out_thr(raw_thr, ref_thr, g=GAIN):
    dt = raw_thr - ref_thr
    return np.clip(ref_thr + np.where(dt < 0, g * dt, dt), 1000.0, 2000.0)


def ref_of(run):
    c = json.loads((RUNS / run / 'model_config.json').read_text())
    return float(c['steer_center_us']), float(c['throttle_neutral_us'])


def main():
    S = np.load(DESK / 'sets_v2.npz')
    v = -S['erpm_label'].astype(np.float64) * SPEED_PER_ERPM
    sess, loc = S['sess'], S['loc_detail']
    runs = sorted(p.name[:-8] for p in (DESK / 'pred').glob('*_tab.npz'))
    P = {}
    for r in runs:
        with np.load(DESK / 'pred' / (r + '_tab.npz')) as z:
            P[r] = z['pred'].astype(np.float64)
    refB = ref_of(B_RUN)
    pb = P[B_RUN]
    tb = car_out_thr(pb[:, ROW2, 1], refB[1])
    ai = S['straight'] | S['stand'] | S['pre_r1'] | S['pre_de'] | S['pre_sw'] | S['de_pass'] | S['gb'] | S['gbref']
    out = {'n_frames': int(len(v)), 'runs': {}}
    for r in runs:
        if r == B_RUN:
            continue
        try:
            ref = ref_of(r)
        except Exception:
            continue
        pc = P[r]
        tc = car_out_thr(pc[:, ROW2, 1], ref[1])
        tc_bref = car_out_thr(pc[:, ROW2, 1], refB[1])          # B の基準で車に出したら
        d = tc - tb
        d_bref = tc_bref - tb
        d_raw = pc[:, ROW2, 1] - pb[:, ROW2, 1]
        st = S['straight']
        row = {'ref': ref,
               'straight_thr_vs_b_us': float(d[st].mean()),
               'straight_thr_vs_b_us_bref': float(d_bref[st].mean()),
               'straight_raw_row2_vs_b_us': float(d_raw[st].mean()),
               'straight_by_sess': {SESS[s]: [float(d[st & (sess == s)].mean()), int((st & (sess == s)).sum())]
                                    for s in (0, 1) if (st & (sess == s)).any()},
               'straight_by_band': {}, 'ai_by_sess_band': {}, 'ai_by_loc': {},
               'row0_brake_frac_lowspeed': {}, 'stand_row0_mean_vs_b': None}
        for lo, hi, name in BANDS:
            m = st & (v >= lo) & (v < hi)
            if m.sum() >= 20:
                row['straight_by_band'][name] = [float(d[m].mean()), int(m.sum())]
        for s in (0, 1, 2):
            for lo, hi, name in BANDS:
                m = ai & (sess == s) & (v >= lo) & (v < hi)
                if m.sum() >= 20:
                    row['ai_by_sess_band']['%s %s' % (SESS[s], name)] = [float(d_raw[m].mean()), int(m.sum())]
        for k, name in LOC.items():
            m = ai & (loc == k) & (v >= 1.0)
            if m.sum() >= 30:
                row['ai_by_loc'][name] = [float(d_raw[m].mean()), int(m.sum())]
        # 低速(< 1.0 m/s)で 行 0 の生のスロットル > 1598(門と同じ式)の割合。この表の AI のフレームで
        for s in (0, 1, 2):
            for lo, hi, name in ((-9, 0.3, '<0.3'), (0.3, 1.0, '0.3-1.0')):
                m = ai & (sess == s) & (v >= lo) & (v < hi)
                if m.sum() >= 20:
                    row['row0_brake_frac_lowspeed']['%s %s' % (SESS[s], name)] = [
                        float((pc[m, ROW0, 1] > 1598).mean()), float((pb[m, ROW0, 1] > 1598).mean()), int(m.sum())]
        m = S['stand']
        row['stand_row0_mean_vs_b'] = float((pc[m, ROW0, 1] - pb[m, ROW0, 1]).mean())
        out['runs'][r] = row
    OUT.mkdir(exist_ok=True)
    (OUT / 'm1_tab_split.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # 画面に要点
    print('%-58s %7s %7s %7s | %8s %8s' % ('run', 'st', 'st_bref', 'raw', '10/10', '10/23'))
    for r, row in out['runs'].items():
        bs = row['straight_by_sess']
        print('%-58s %7.2f %7.2f %7.2f | %8.2f %8.2f' % (r[:58], row['straight_thr_vs_b_us'], row['straight_thr_vs_b_us_bref'],
                                                     row['straight_raw_row2_vs_b_us'], bs.get('10/10', [np.nan])[0],
                                                     bs.get('10/23', [np.nan])[0]))


if __name__ == '__main__':
    sys.exit(main())
