#!/usr/bin/env python3
"""desk_bench_20261022.json の数を、保存した予測から別のコードで数え直す(desk_lib・desk_lib2・desk_eval を読まない。numpy だけ)
あわせて、低速でブレーキ側を出したフレームのくわしい数(当日の注意の参考)を出す。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        /home/jetson/jetformer/experiments/20261022_prep/desk/recount2.py

入力: pred/desk_inputs_20261022.npz(予測・正解・集合・区間)と sanity_<id>.npz(B・SLOW は 10/21、新しい 3 本は 10/23)
結果: experiments/20261022_prep/desk/logs/recount2.json
"""
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

DESK = Path('/home/jetson/jetformer_reports/prep_20261022/desk')
OLD = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk/pred')
OUT = Path(__file__).resolve().parent / 'logs' / 'recount2.json'
TOL = {'l1_best_us': 0.002, 'l1_last_us': 0.002, 'l1_interv_us': 0.002, 'l1_steer_best_us': 0.002, 'l1_thr_best_us': 0.002,
       'r1_app_thr_diff_us': 0.002, 'steer_amp_ratio': 0.0002, 'brake_lowspeed_frac': 0.00002, 'nonzone_thr_diff_us': 0.002}


def sanity_path(cid):
    return (OLD if cid in ('B', 'SLOW') else DESK / 'pred') / ('sanity_%s.npz' % cid)


def main():
    z = dict(np.load(DESK / 'pred' / 'desk_inputs_20261022.npz'))
    bench = json.load(open(DESK / 'desk_bench_20261022.json', encoding='utf-8'))
    rows = {r['id']: r for r in bench['candidates']}
    y = np.where(np.isfinite(z['y']), z['y'], 0.0).astype(np.float64)
    hold = z['primary'].astype(bool)
    iv = z['interv'].astype(bool)
    r1 = z['r1_pre1s'].astype(bool)
    zone = z['zone'].astype(str)
    nz = hold & ((zone == 'none') | (zone == 'unknown'))
    speed = -z['erpm'].astype(np.float64) * 0.0005
    out, detail, ok_all = {}, {}, True
    for cid in [str(x) for x in z['ids']]:
        got = {}
        p = z['pred_%s_best' % cid].astype(np.float64)
        e = np.abs(p - y)
        got['l1_best_us'] = e[hold].mean()
        got['l1_steer_best_us'] = e[hold][:, :, 0].mean()
        got['l1_thr_best_us'] = e[hold][:, :, 1].mean()
        got['l1_interv_us'] = e[iv].mean()
        got['r1_app_thr_diff_us'] = (p[r1, 2, 1] - y[r1, 2, 1]).mean()
        got['steer_amp_ratio'] = p[hold][:, :, 0].std() / y[hold][:, :, 0].std()
        got['nonzone_thr_diff_us'] = (p[nz, 0, 1] - y[nz, 0, 1]).mean()
        low = hold & (speed < 1.0)
        k1, n1 = int((p[low, 0, 1] > 1598.0).sum()), int(low.sum())
        s = np.load(sanity_path(cid))
        v2 = -s['erpm'].astype(np.float64) * 0.0005
        u2 = s['pred'][:, 0, 1].astype(np.float64)
        low2 = (v2 < 1.0) & np.isfinite(u2)
        br = low2 & (u2 > 1598.0)
        k2, n2 = int(br.sum()), int(low2.sum())
        got['brake_lowspeed_frac'] = (k1 + k2) / (n1 + n2)
        got['l1_last_us'] = np.abs(z['pred_%s_last' % cid].astype(np.float64) - y)[hold].mean()
        res = {}
        for k, v in got.items():
            w = rows[cid].get(k)
            same = w is not None and abs(float(v) - float(w)) <= TOL[k]
            res[k] = {'recount': round(float(v), 5), 'desk_bench': w, 'same': bool(same)}
            ok_all &= bool(same)
        out[cid] = res
        logs = np.array([str(x)[:15] for x in s['scene']])
        detail[cid] = {'ai_low_frames': n2, 'ai_brake_frames': k2,
                       'brake_speed_median_mps': round(float(np.median(v2[br])), 3) if k2 else None,
                       'brake_speed_lt_0p2_share': round(float((v2[br] < 0.2).mean()), 3) if k2 else None,
                       'brake_thr_max_us': round(float(u2[br].max()), 1) if k2 else None,
                       'brake_by_log': {lg: int((br & (logs == lg)).sum()) for lg in sorted(set(logs[br]))},
                       'any_speed_thr_gt_1598': int((np.isfinite(u2) & (u2 > 1598.0)).sum()),
                       'heldout_low_brake': [k1, n1]}
    # A10 の C1 の決め方(別に書いた)
    cand = []
    for cid in ('SLOW', 'SLOW1', 'SLOW-SEC0', 'SLOW-SEC1'):
        r = rows.get(cid)
        if r and r.get('gates_pass') is True and r.get('r1_app_thr_diff_us') is not None and r['r1_app_thr_diff_us'] >= 20.0:
            q = float(Decimal(repr(float(r['nonzone_thr_diff_us']))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
            cand.append((q, -float(r['r1_app_thr_diff_us']), cid))
    pick = sorted(cand)[0][2] if cand else 'SLOW'
    a10_same = pick == bench['a10_c1']['pick']
    ok_all &= a10_same
    doc = {'pass': bool(ok_all), 'tolerance': TOL, 'candidates': out, 'a10_c1_recount': {'pick': pick, 'order': [c[2] for c in sorted(cand)],
                                                                                          'same': a10_same},
           'brake_detail_ai_1010': detail, 'inputs': str(DESK / 'pred' / 'desk_inputs_20261022.npz'),
           'desk_bench_created_at': bench.get('created_at')}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print('数え直し:', '一致' if ok_all else '食い違いあり', OUT)
    for cid, res in out.items():
        bad = [k for k, v in res.items() if not v['same']]
        if bad:
            print(' ', cid, bad)
    print(json.dumps(doc['a10_c1_recount'], ensure_ascii=False))
    print(json.dumps(detail, ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
