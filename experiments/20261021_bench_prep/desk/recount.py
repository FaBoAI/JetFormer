#!/usr/bin/env python3
"""desk_bench.json の数を、保存した予測から別のコードで数え直す(desk_lib.py・desk_eval.py を読まない。numpy だけ)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        /home/jetson/jetformer/experiments/20261021_bench_prep/desk/recount.py

入力: desk/pred/desk_inputs.npz(予測・正解・集合)と desk/pred/sanity_<id>.npz(10/10 の AI 走行の TensorRT の出力)、
engine_check.json は読まない(TensorRT の欄は数え直さない)。結果: experiments/20261021_bench_prep/desk/logs/recount.json
"""
import json
import sys
from pathlib import Path

import numpy as np

DESK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
OUT = Path(__file__).resolve().parent / 'logs' / 'recount.json'
TOL = {'l1_best_us': 0.002, 'l1_last_us': 0.002, 'l1_interv_us': 0.002, 'l1_steer_best_us': 0.002, 'l1_thr_best_us': 0.002,
       'r1_app_thr_diff_us': 0.002, 'steer_amp_ratio': 0.0002, 'brake_lowspeed_frac': 0.00002}


def main():
    z = dict(np.load(DESK / 'pred' / 'desk_inputs.npz'))
    bench = json.load(open(DESK / 'desk_bench.json', encoding='utf-8'))
    rows = {r['id']: r for r in bench['candidates']}
    y = np.where(np.isfinite(z['y']), z['y'], 0.0).astype(np.float64)
    hold = z['primary'].astype(bool)
    iv = z['interv'].astype(bool)
    r1 = z['r1_pre1s'].astype(bool)
    speed = -z['erpm'].astype(np.float64) * 0.0005
    out, ok_all = {}, True
    for cid in [str(x) for x in z['ids']]:
        got = {}
        kb, kl = 'pred_%s_best' % cid, 'pred_%s_last' % cid
        if kb in z:
            p = z[kb].astype(np.float64)
            e = np.abs(p - y)
            got['l1_best_us'] = e[hold].mean()
            got['l1_steer_best_us'] = e[hold][:, :, 0].mean()
            got['l1_thr_best_us'] = e[hold][:, :, 1].mean()
            got['l1_interv_us'] = e[iv].mean()
            got['r1_app_thr_diff_us'] = (p[r1, 2, 1] - y[r1, 2, 1]).mean()
            got['steer_amp_ratio'] = p[hold][:, :, 0].std() / y[hold][:, :, 0].std()
            low = hold & (speed < 1.0)
            k1, n1 = int((p[low, 0, 1] > 1598.0).sum()), int(low.sum())
            sp = DESK / 'pred' / ('sanity_%s.npz' % cid)
            if sp.exists():
                s = np.load(sp)
                v2 = -s['erpm'].astype(np.float64) * 0.0005
                u2 = s['pred'][:, 0, 1].astype(np.float64)
                low2 = (v2 < 1.0) & np.isfinite(u2)
                k2, n2 = int((low2 & (u2 > 1598.0)).sum()), int(low2.sum())
                got['brake_lowspeed_frac'] = (k1 + k2) / (n1 + n2) if (n1 + n2) else None
        if kl in z:
            p = z[kl].astype(np.float64)
            got['l1_last_us'] = np.abs(p - y)[hold].mean()
        res = {}
        for k, v in got.items():
            w = rows[cid].get(k)
            same = (v is None and w is None) or (v is not None and w is not None and abs(float(v) - float(w)) <= TOL[k])
            res[k] = {'recount': None if v is None else round(float(v), 5), 'desk_bench': w, 'same': bool(same)}
            ok_all &= bool(same)
        out[cid] = res
    doc = {'pass': bool(ok_all), 'tolerance': TOL, 'candidates': out,
           'inputs': str(DESK / 'pred' / 'desk_inputs.npz'), 'desk_bench_created_at': bench.get('created_at')}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print('数え直し:', '一致' if ok_all else '食い違いあり', OUT)
    for cid, res in out.items():
        bad = [k for k, v in res.items() if not v['same']]
        if bad:
            print(' ', cid, bad)
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
