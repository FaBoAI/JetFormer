"""反証 8-2 回目・係 safety: drive の別の道(速さの天井・自動合わせ・左 4.2・start_kick・発進の前進・ToF の欠け)と、リカバリー・
中立止め・人側が同じ走りの中で起きる形。1 次元のコース(fix8/decided_safety/h10.py と同じ考え: 角の手前で減速・角で舵・ToF は
次の角の壁を見る・舵と速さからヨー)で、2 つ目の角の壁に当たる(モデルが曲がりきれない)→ リカバリー → また押す …。
回の形 5 つ(kai1・kai1_ceil・kai2_auto42・kai3_lb_ceil・kai4_lb)× ESC の模型 6 つ × 乱れ(AI 側にしてからの 0.1〜1.5 秒 =
発進の前進・発進の補助の途中と、最初のリカバリーの段の移りに、カメラの止まり・人側・eRPM の跳ね・ToF の欠け・段差の衝撃・
車両の値の止まり)。ON と OFF を同じ場面で並べ、exp12 と同じ比べ(後ろの物の距離ごとの当たり・前の壁の押し・0 m の壁押し・
ON だけの中止)と記録の照らし。全部 OFF は写し f7b の OFF ともビット単位で比べる。
    python -I -B exp12c.py <i/n>  → out/exp12c_<i>of<n>.jsonl(1 組 1 行。流し直すと済んだ組は飛ばす)
"""
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS82_TAG', 'c_' + '_'.join(sys.argv[1:2]).replace('/', 'of'))
sys.path.insert(0, str(HERE))
import h12  # noqa: E402
import exp12 as E  # noqa: E402

h11 = h12.h11
C, N, A = h12.C, h12.N, h12.A
ESC6 = ('fit', 'cl', 'human', 'fit_lag0.1', 'fit_const', 'fit_coast3.0')
FORMS5 = ('kai1', 'kai1_ceil', 'kai2_auto42', 'kai3_lb_ceil', 'kai4_lb')
CORNERS = [8.0, 16.0, 24.0]
GAP = 0.6
XF = CORNERS[1] + GAP


def make_model(ta=1.0, vmax=6.0, vcors=(4.6, 2.8, 2.8), m0=60.0, ramp=0.8):
    """h10.make_model の写し(角は左 +1。角ごとの目標の速さ vcors: 1 つ目は左の天井 4.1/4.2 より速く曲がる)。直線の目標 6 m/s
    (全体の天井 4.5 より速い)。2 つ目の角の壁に当たる(1 次元なので曲がっても壁は前にある)"""
    corners = list(CORNERS)

    def f(t, car):
        if t < ta:
            return (C, N)
        el = t - ta
        x, v = car.x, car.v
        vt, st = vmax, C
        for c, vcor in zip(corners, vcors):
            if c - 2.5 <= x < c:
                vt = min(vt, vcor + (vmax - vcor) * max(0.0, (c - x - 1.0) / 1.5))
            if c <= x < c + 2.0:
                vt = min(vt, vcor)
                st = C - 300.0
        fwd = vt / 1.2 * 80.0 + 60.0 * (vt - v)
        fwd = float(np.clip(fwd, 0.0, 330.0))
        if el < ramp:
            fwd = min(fwd, m0 + (fwd - m0) * el / ramp) if fwd > m0 else fwd
        if v > vt + 0.6:
            return (st, N + 120.0)
        return (st, N - fwd)
    return f


def build(sd):
    sc = {'dur': sd.get('dur', 16.0), 'xf': XF, 'ai_from': 1.0, 'esc18': sd['esc'], 'veh_dt': 0.01,
          'corners': CORNERS, 'wall_gap': GAP, 'model': make_model(m0=sd.get('m0', 60.0))}
    for p in sd.get('pert', []):
        k, c = p[0], float(p[1])
        if k == 'drop':
            sc.setdefault('drops', []).append((c - 0.002, c - 0.002 + p[2] / 30.0))
        elif k == 'veh':
            sc.setdefault('veh_skip', []).append((c - 0.0005, c - 0.0005 + p[2]))
        elif k == 'tof':
            sc.setdefault('tof_skip', []).append((c - 0.0005, c - 0.0005 + p[2]))
        elif k == 'glitch':
            sc.setdefault('glitch', []).append((E.samp(c), int(p[2])))
        elif k == 'spike':
            sc.setdefault('spikes', []).append((round(E.samp(c) - 0.02, 2), float(p[2])))
        elif k == 'human':
            a = E.samp(c) - 0.0005
            sc.setdefault('human', []).append((a, a + float(p[2])))
    if sc.get('human'):
        sc['human_thr'] = N
    return sc


class _B(dict):
    pass


E.BASES['course'] = {'xf': XF, 'model': ('const', 0.0), 'dur': 16.0}


def run_pair(sd, idcheck=True):
    p = h12.base_params(**h12.FORMS[sd['form']])
    ron = h12.run(h12.ON(p), build(sd))
    roff = h12.run(h12.OFF(p), build(sd))
    sdm = dict(sd, base='course')
    mon, moff = E.metr(ron, sdm), E.metr(roff, sdm)
    for m, r in ((mon, ron), (moff, roff)):
        rows = [x for _, x in r[1].rows if not x['hold']]
        m['ceil_applied'] = sum(1 for x in rows if x.get('ceil_applied'))
        m['launch_rows'] = sum(1 for x in rows if x.get('launch_on'))
        m['launch_end'] = sorted({int(x.get('launch_end', 0)) for x in rows} - {0})
        m['auto_gain_last'] = round(float(rows[-1]['gains'][1]), 4) if rows else None
        m['first_hit'] = next((h for h in r[0].car.hits if h[1] == 'front'), None)
    row = {'on': mon, 'off': moff, 'worse': E.worse(mon, moff), 'same_on_off': h11.same_run(ron, roff)[0]}
    if idcheck:
        rb = h12.run(h12.OFF(p), build(sd), ver='f7b')
        ok, why = h11.same_run(roff, rb)
        row['off_eq_f7b'] = ok
        if not ok:
            row['off_f7b_diff'] = str(why)[:300]
    return row, ron


def perts(tl):
    out = []
    for dt in (0.1, 0.3, 0.6, 1.0, 1.5):
        c = round(0.15 + round((1.0 + dt - 0.15) * 30) / 30.0, 4)
        for pt in (('drop', 3), ('drop', 8), ('drop', 16), ('human', 0.5), ('glitch', 1322), ('tof', 0.2), ('spike', 3.0),
                   ('veh', 0.16)):
            out.append(('launch+%.1f' % dt, [(pt[0], c) + tuple(pt[1:])]))
    for ev in ('brake', 'reverse', 'coast', 'escape'):
        if ev not in tl:
            continue
        c = E.cam_of(tl[ev])
        for pt in (('drop', 5), ('drop', 16), ('veh', 0.16), ('human', 0.5), ('glitch', 1322)):
            out.append((ev, [(pt[0], c) + tuple(pt[1:])]))
    return out


def main(part='0/1'):
    h11.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    jl = E.JL(HERE / 'out' / ('exp12c_%s.jsonl' % part.replace('/', 'of')))
    n_new = n_worse = 0
    t_s = time.time()
    try:
        jobs = list(itertools.product(FORMS5, ESC6))
        only = os.environ.get('RS82_ESC')     # 10/28 10:0x: 短めに(ESC の模型を選ぶ)
        if only:
            jobs = [j for j in jobs if j[1] in only.split(',')]
        for ji, (form, esc) in enumerate(jobs):
            if ji % pn != pi:
                continue
            bsd = {'form': form, 'esc': esc}
            b = jl.bases.get(E.skey(bsd))
            if b is None:
                brow, bron = run_pair(dict(bsd, human_any=False))
                tl = E.timeline(bron[3])
                jl.add({'t': 'base', 'kind': 'c', 'sd': bsd, 'tl': tl, 'row': brow})
                print('base', bsd, tl, brow['worse'], brow.get('off_eq_f7b'), brow['on']['first_hit'], brow['on']['launch_end'],
                      brow['on']['ceil_applied'], round(time.time() - t_s), flush=True)
            else:
                tl = b['tl']
            for ev, pt in perts(tl):
                sd = dict(bsd, ev=ev, pert=pt, human_any=pt[0][0] == 'human')
                if E.skey(sd) in jl.done:
                    continue
                row, _ = run_pair(sd)
                row['sd'] = {k: v for k, v in sd.items() if k != 'human_any'}
                jl.add({'t': 'row', 'kind': 'c', 'sd': row['sd'], 'row': row})
                n_new += 1
                if row['worse'] or not row.get('off_eq_f7b', True) or row['on']['rows_bad'] or row['off']['rows_bad']:
                    n_worse += 1
                    print(n_new, round(time.time() - t_s), form, esc, ev, pt, row['worse'], row.get('off_eq_f7b'),
                          row['on']['rows_bad'][:1], row['off']['rows_bad'][:1], flush=True)
    finally:
        h12.cleanup()
    print('done c', part, 'new', n_new, 'flagged', n_worse, 'skipped(done before)', len(jl.done), round(time.time() - t_s), 's')


if __name__ == '__main__':
    main(*sys.argv[1:2])
