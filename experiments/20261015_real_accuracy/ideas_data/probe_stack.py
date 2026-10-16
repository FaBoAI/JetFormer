#!/usr/bin/env python3
"""下調べ 3(CPU だけ・学習なし): 学習のいらない 3 つの手(種のアンサンブル・時間方向のアンサンブル・電池の効きの補正)を
重ねたとき、足し算になるか。10/13 の判定の形(2 本 × best/last、閾値 0.626 / 1.322 µs)に当てはめた参考の値も出す。
(どれも保留を見たあとの下調べ。本番の判定は、規則を先に固めて新しい種・別の保留で行う)

    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \
        python experiments/20261015_real_accuracy/ideas_data/probe_stack.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_gain as pg  # noqa: E402

ROOT = pg.ROOT
ev = pg.ev
OUT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/ideas_data/probe_stack.json')
TH = {'best': 0.626, 'last': 1.322}
A = {'best': [49.146, 49.457], 'last': [49.235, 49.752]}


def main():
    tab = ev.pred_table()
    held = ev._heldout()
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    clean = np.isin(tab['scene'], held['human_clean_scenes']) & m0 & cols['fv'] & ~cols['impact']
    y = np.nan_to_num(cols['y'])
    sc, fr = tab['scene'], tab['frame']
    zp = ROOT / 'data/zoo/policy_zoo_20261012/pred'

    def ld(n):
        return np.load(zp / (n + '.npz'))['pred'].astype(np.float64)

    P = {ck: {} for ck in ('best', 'last')}
    for base in ('jetformer_v0_s0', 'jetformer_v0_s1', 'msfusion_s0', 'msfusion_s1', 'tcp_s0', 'tcp_s1', 'act_s0', 'act_s1'):
        P['best'][base] = ld(base)
        P['last'][base] = ld(base + '_last')
    P['best']['B'], P['last']['B'] = ld('B'), ld('B_last')

    def L(p, mask=clean):
        return float(np.abs(p - y)[mask].mean())

    # 時間方向(probe_cached と同じ。J=2・一様)
    row_of = {(s, int(i)): k for k, (s, i) in enumerate(zip(sc, fr)) if m0[k]}
    prev = np.full((len(sc), 3), -1, int)
    for k in np.nonzero(m0)[0]:
        for j in range(3):
            prev[k, j] = row_of.get((sc[k], int(fr[k]) - j), -1)

    def temporal(p, J=2):
        out = p.copy()
        for k in range(10):
            num, den = p[:, k, :].copy(), np.ones(len(p))
            for j in range(1, J + 1):
                if k + j > 9:
                    break
                ok = prev[:, j] >= 0
                num[ok] += p[prev[ok, j], k + j, :]
                den[ok] += 1
            out[:, k, :] = num / den[:, None]
        return out

    # 電池の効き(W=5 秒・clip 0.8〜2.0。probe_gain の既定)
    sp = json.loads((pg.CLEAN / 'splits.json').read_text())
    gs = {4: [], 5: []}
    for s in sp['train']:
        a = dict(np.load(pg.CLEAN / 'aligned' / (s + '.npz')))
        g = pg.ratio_series(a)
        for k in (4, 5):
            gs[k].append(g[(a['mode'] == k) & np.isfinite(g)])
    g_ref = {k: float(np.median(np.concatenate(v))) for k, v in gs.items()}
    gt = np.full(len(sc), np.nan)
    for s in sorted(set(sc[clean].tolist())):
        a = dict(np.load(pg.CLEAN / 'aligned' / (s + '.npz')))
        run = pg.running_median(pg.ratio_series(a), a['timestamps_us'].astype(np.float64), 5.0)
        rows = np.nonzero(m0 & (sc == s))[0]
        gt[rows] = run[fr[rows]]
    ref = np.array([g_ref.get(int(m), np.nan) for m in cols['mode']])
    scale = np.where(np.isfinite(gt / ref), np.clip(gt / ref, 0.8, 2.0), 1.0)

    def gain(p):
        q = p.copy()
        d = pg.NEUTRAL - q[:, :, 1]
        q[:, :, 1] = np.where(d > 0, pg.NEUTRAL - d * scale[:, None], q[:, :, 1])
        return q

    arms = {
        'single': lambda ck, s: P[ck]['jetformer_v0_s%d' % s],
        'T(time J2)': lambda ck, s: temporal(P[ck]['jetformer_v0_s%d' % s]),
        'G(battery)': lambda ck, s: gain(P[ck]['jetformer_v0_s%d' % s]),
        'T+G': lambda ck, s: gain(temporal(P[ck]['jetformer_v0_s%d' % s])),
        'E2 v0+msfusion (same seed)': lambda ck, s: 0.5 * (P[ck]['jetformer_v0_s%d' % s] + P[ck]['msfusion_s%d' % s]),
        'E2 v0+tcp (same seed)': lambda ck, s: 0.5 * (P[ck]['jetformer_v0_s%d' % s] + P[ck]['tcp_s%d' % s]),
        'E2 v0+act (same seed)': lambda ck, s: 0.5 * (P[ck]['jetformer_v0_s%d' % s] + P[ck]['act_s%d' % s]),
        'E2+T+G v0+msfusion': lambda ck, s: gain(temporal(0.5 * (P[ck]['jetformer_v0_s%d' % s] + P[ck]['msfusion_s%d' % s]))),
        'E4 v0+msfusion+tcp+act': lambda ck, s: np.mean([P[ck]['%s_s%d' % (n, s)] for n in ('jetformer_v0', 'msfusion', 'tcp', 'act')], 0),
        'E4+T+G': lambda ck, s: gain(temporal(np.mean([P[ck]['%s_s%d' % (n, s)] for n in ('jetformer_v0', 'msfusion', 'tcp', 'act')], 0))),
    }
    res = {'note': '保留を見たあとの下調べ(判定ではない)。arm の 2 本 = 種 0・種 1 の組。閾値は 10/13 の規則の値',
           'g_ref': g_ref, 'arms': {}}
    late = clean & (sc == '20261010-193638-0003')
    for name, fn in arms.items():
        v = {ck: [L(fn(ck, s)) for s in (0, 1)] for ck in ('best', 'last')}
        out = {'best': [round(x, 3) for x in v['best']], 'last': [round(x, 3) for x in v['last']]}
        verdict = []
        for ck in ('best', 'last'):
            gain_us = np.mean(A[ck]) - np.mean(v[ck])
            both = max(v[ck]) < min(A[ck])
            out['gain_%s' % ck] = round(float(gain_us), 3)
            verdict.append(both and gain_us > TH[ck])
        out['would_be_better(best&last)'] = bool(all(verdict))
        # 193638-0003 を除いた 10 シーンでの差(主指標ではない。どこから来たかの内訳)
        ex = clean & ~late
        out['gain_best_excluding_193638_0003'] = round(float(np.mean([L(P['best']['jetformer_v0_s%d' % s], ex) for s in (0, 1)])
                                                             - np.mean([L(fn('best', s), ex) for s in (0, 1)])), 3)
        res['arms'][name] = out
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
