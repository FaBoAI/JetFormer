#!/usr/bin/env python3
"""反証 V-replay: 天井が掛かっている間の出力の上下(記録の雑音のまま、開ループ)と、加速 a のならし方の比べ(2026-10-25)。読むだけ。
    jf-heavy --lock none --mem 1500M -- python -I rr_chatter.py
a の作り方: impl = 実装(0.1 秒前との差)/ a0.2 = 0.2 秒前との差(試し。実装は変えていない)/ 先読みなし(pred_s 0)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402

RUNS = ['20261024-162033', '20261024-160654', '20261024-160536', '20261024-160847', '20261024-162305',
        '20261023-211828', '20261023-212934', '20261023-213149', '20261023-211118']


def chatter(d, thr_out):
    lim = (thr_out != d['thr']) & d['ai']
    i, n, rev, dur, dfs = 0, len(lim), 0, 0.0, []
    while i < n:
        if lim[i]:
            j = i
            while j < n and lim[j]:
                j += 1
            if d['t'][j - 1] - d['t'][i] >= 0.5:
                df = np.diff(d['neutral'] - thr_out[i:j])
                dfs += list(np.abs(df))
                sg = np.sign(df[np.abs(df) > 2.0])
                rev += int(np.sum(sg[1:] != sg[:-1]))
                dur += float(d['t'][j - 1] - d['t'][i])
            i = j
        else:
            i += 1
    return {'runs_s': dur, 'rev_per_s': rev / dur if dur else None, 'df_med': float(np.median(dfs)) if dfs else None,
            'df_p90': float(np.percentile(dfs, 90)) if dfs else None}


def main():
    D = rr_cache.load()['runs']
    out = []
    for log in RUNS:
        d = D[log]
        v_old, j_old = C._median3(d['vt'], d['ve'], d['t'] - 0.2)
        a2 = np.where(j_old >= 0, (d['v'] - v_old) / 0.2, np.nan)
        row = {'log': log, 'no': d['no'], 'session': d['session']}
        for name, p, a in (('impl', C.P('impl'), d['a']), ('a0.2', C.P('impl'), a2),
                           ('pred0', C.P('impl', pred_s=0.0), d['a'])):
            dd = dict(d, a=a)
            ol = C.open_loop(dd, p)
            row[name] = chatter(d, ol['thr_out'])
        out.append(row)
        print(d['session'], d['no'], {k: row[k] for k in ('impl', 'a0.2', 'pred0')})
    C.jdump(out, os.path.join(C.HERE, 'rr_chatter.json'))


if __name__ == '__main__':
    main()
