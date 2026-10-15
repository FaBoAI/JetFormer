#!/usr/bin/env python3
"""Q2 の主指標の数え直し(protocol eval_Q2.independent_recount。CPU・numpy だけ)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_sim_diverse_train/q2/q2_recount.py

判定の道具(q2_evaluate.py・rl_evaluate._val_arrays・sim_cotrain.judge_primary)を使わずに、
  1. 正解と主指標のフレームを ai_mix_20261011 の aligned から作り直す(10/13 の review_check.primary_frames と同じ考え方)
  2. 予測の npz(パスはここに直に書く)から L1 を計算し、eval_real.json と ≤ 0.001 µs で同じか
  3. 判定の規則をここで書き直して、eval_real.json の判定と同じか
→ /home/jetson/jetformer_reports/sim_diverse_train_20261015/q2/recount.json
"""

import json
import os
import sys

import numpy as np

REPO = '/home/jetson/jetformer'
os.chdir(REPO)
sys.path.insert(0, REPO)
from jetformer.train_data import impact_mask  # noqa: E402

REPORT = '/home/jetson/jetformer_reports/sim_diverse_train_20261015'
AM = 'data/datasets/ai_mix_20261011'
REAL = 'data/datasets/clean_driving_20261011'
Z12 = 'data/zoo/policy_zoo_20261012/pred/'
Z13 = 'data/zoo/sim_cotrain_20261013/pred/'
Z15 = 'data/zoo/sim_diverse_train_20261015/pred/'
PRED = {'A_s0': Z12 + 'jetformer_v0_s0', 'A_s1': Z12 + 'jetformer_v0_s1', 'B': Z12 + 'B',
        'C_s0': Z13 + 'C_s0', 'C_s1': Z13 + 'C_s1', 'CDIV_s0': Z15 + 'CDIV_s0', 'CDIV_s1': Z15 + 'CDIV_s1'}
FLOOR = {'A': {'best': 0.313, 'last': 0.661}, 'C': {'best': 0.477, 'last': 0.661}}   # protocol で固定した s


def frames():
    """予測の表(part・scene・frame の順)は rl_evaluate.pred_table() から(表の作り方だけを借りる。正解・フレームの選び方はここで作る)"""
    sys.path.insert(0, REPO + '/tools')
    import rl_evaluate as ev
    tab = ev.pred_table()
    with open(REAL + '/splits.json') as f:
        val = set(json.load(f)['val'])
    n = len(tab['part'])
    y = np.full((n, 10, 2), np.nan)
    clean = np.zeros(n, bool)
    mode = np.zeros(n, int)
    cache = {}
    for i in np.nonzero((tab['part'] == 0) & np.isin(tab['scene'], list(val)))[0]:
        s = str(tab['scene'][i])
        if s not in cache:
            with np.load('%s/aligned/%s.npz' % (AM, s)) as z:
                a = {k: z[k] for k in ('timestamps_us', 'imu', 'future_actions', 'future_valid', 'vehicle_valid', 'mode')}
            a['_imp'] = impact_mask(a['timestamps_us'], a['imu'], 2.5)
            cache[s] = a
        a, fr = cache[s], int(tab['frame'][i])
        y[i], mode[i] = a['future_actions'][fr], a['mode'][fr]
        clean[i] = bool(a['future_valid'][fr] and a['vehicle_valid'][fr] and not a['_imp'][fr])
    return tab, y, clean, mode, sorted(val)


def judge(arm, ref, floor):
    arm, ref = np.asarray(arm, float), np.asarray(ref, float)
    s = max(ref.max() - ref.min(), floor)
    thr = 2.0 * s
    d = ref.mean() - arm.mean()
    if arm.max() < ref.min() and d > thr:
        return 'better', thr, -d
    if arm.min() > ref.max() and -d > thr:
        return 'worse', thr, -d
    return 'equivalent', thr, -d


def final(b, l):
    return ('better' if l == 'better' else 'equivalent') if b == 'better' else b


def main():
    tab, y, clean, mode, val = frames()
    ev_path = REPORT + '/eval_real.json'
    evj = json.load(open(ev_path, encoding='utf-8'))
    L, mism = {}, []
    for n, base in PRED.items():
        for ck in ('best', 'last'):
            p = base + ('' if ck == 'best' else '_last') + '.npz'
            if not os.path.exists(p):
                continue
            with np.load(p) as z:
                assert len(z['pred']) == len(tab['part'])
                pr = z['pred'].astype(np.float64)
                assert (z['genre'][clean] == mode[clean]).all(), (n, ck, 'ジャンルが aligned の mode と違う')
            L[(n, ck)] = float(np.abs(pr[clean] - y[clean]).mean())
            e = evj['primary']['L_us'][ck].get(n)
            if e is None or abs(round(L[(n, ck)], 3) - e) > 0.001:
                mism.append(['L', n, ck, L[(n, ck)], e])
    names = sorted({k[0] for k in L})
    cd = [n for n in ('CDIV_s0', 'CDIV_s1') if n in names]

    def vals(arm, ck):
        return [L[(n, ck)] for n in {'A': ['A_s0', 'A_s1'], 'C': ['C_s0', 'C_s1'], 'C-DIV': cd}[arm]]
    ver = {}
    for lab, arm, ref, key in (('CDIV_vs_A', 'C-DIV', 'A', 'CDIV_vs_A'), ('CDIV_vs_C', 'C-DIV', 'C', 'CDIV_vs_C'),
                               ('C_vs_A', 'C', 'A', 'C_vs_A')):
        r = {ck: judge(vals(arm, ck), vals(ref, ck), FLOOR[ref][ck]) for ck in ('best', 'last')}
        v = {'final': final(r['best'][0], r['last'][0]), 'best': r['best'][0], 'last': r['last'][0],
             'thr_best': round(r['best'][1], 4), 'thr_last': round(r['last'][1], 4),
             'diff_best': round(r['best'][2], 4), 'diff_last': round(r['last'][2], 4)}
        ver[lab] = v
        e = evj['verdict'].get(key)
        if e != v['final']:
            mism.append(['verdict', lab, v['final'], e])
    out = {'frames': {'human_clean': int(clean.sum()), 'genre4': int((clean & (mode == 4)).sum()),
                      'genre5': int((clean & (mode == 5)).sum()), 'val_scenes': len(val)},
           'L_us': {'%s_%s' % k: round(v, 4) for k, v in L.items()},
           'arm_mean_best_us': {a: round(float(np.mean(vals(a, 'best'))), 4) for a in ('A', 'C', 'C-DIV')},
           'verdicts': ver, 'seeds_C-DIV': cd, 'eval_real_created_at': evj.get('created_at'),
           'mismatch_vs_eval_real': mism, 'all_match': not mism and int(clean.sum()) == evj['frames']['human_clean']}
    with open(REPORT + '/q2/recount.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out['all_match'] else 1


if __name__ == '__main__':
    sys.exit(main())
