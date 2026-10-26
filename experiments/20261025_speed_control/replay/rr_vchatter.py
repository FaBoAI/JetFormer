#!/usr/bin/env python3
"""反証 V-replay: 仮想の閉ループ(rr_ceiling.virtual。むだ時間 0.05 秒・記録の速さの跳ねを天井の入力に足す)で、
天井の出力の上下と「中立 ↔ 弱める」の入れ替わりを数える(2026-10-25)。読むだけ。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402
from rr_common import sc  # noqa: E402


def segments(d):
    act = d['active']
    i, n, out = 0, len(act), []
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j] and (j == i or d['t'][j] - d['t'][j - 1] <= 0.2):
                j += 1
            if j - i >= 30:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main():
    D = rr_cache.load()['runs']
    rows = []
    for log, d in D.items():
        b = RC.run_b(d)
        for var in ('impl', 'spec'):
            p = C.P(var)
            tot = {'frames': 0, 'lim': 0, 'neu': 0, 'toggle': 0, 'rev': 0, 'lim_s': 0.0, 'dfs': [], 'over_steady': [],
                   'jump100': 0}
            for i0, i1 in segments(d):
                o = RC.virtual(d, p, i0, i1, b, dead=0.05, keep=True)
                mode = o['mode']
                lim = o['changed']
                neu = mode == sc.CEIL_NEUTRAL
                f = o['f_out']
                tot['frames'] += i1 - i0
                tot['lim'] += int(lim.sum())
                tot['neu'] += int(neu.sum())
                tot['toggle'] += int(np.sum(neu[1:] != neu[:-1]))
                df = np.abs(np.diff(f))
                both = lim[1:] & lim[:-1]
                tot['dfs'] += list(df[both])
                tot['jump100'] += int(np.sum(df[both] >= 100))
                sg = np.sign(np.diff(f)[both & (df > 2.0)])
                tot['rev'] += int(np.sum(sg[1:] != sg[:-1]))
                tot['lim_s'] += float(lim.sum()) / 30.0
            dfs = np.array(tot.pop('dfs'))
            tot.pop('over_steady')
            rows.append({'log': log, 'no': d['no'], 'session': d['session'], 'variant': var, **tot,
                         'df_med': float(np.median(dfs)) if len(dfs) else None,
                         'df_p90': float(np.percentile(dfs, 90)) if len(dfs) else None,
                         'rev_per_lim_s': tot['rev'] / tot['lim_s'] if tot['lim_s'] > 0 else None,
                         'toggle_per_lim_s': tot['toggle'] / tot['lim_s'] if tot['lim_s'] > 0 else None})
    C.jdump(rows, os.path.join(C.HERE, 'rr_vchatter.json'))
    for r in rows:
        if r['lim'] > 30:
            print(r['session'], r['no'], r['variant'], 'lim %.1f%% neu %.1f%% lim_s %.1f toggles %d (%.2f/s) rev/s %.1f df med %.1f p90 %.1f jumps>=100µs %d' % (
                100 * r['lim'] / r['frames'], 100 * r['neu'] / r['frames'], r['lim_s'], r['toggle'], r['toggle_per_lim_s'],
                r['rev_per_lim_s'], r['df_med'], r['df_p90'], r['jump100']))


if __name__ == '__main__':
    main()
