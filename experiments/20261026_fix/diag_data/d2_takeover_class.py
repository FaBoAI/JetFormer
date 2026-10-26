#!/usr/bin/env python3
"""原因 A の診断 D2(numpy だけ・読むだけ): v2-H の新しい人の学習のサンプルを、直前の交代の種類(ラベルの events.takeovers の class)で分け、
正解のスロットル(前進の量・ブレーキ側の割合)を出す。書くのは out/d2_takeover_class.json だけ。"""
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path('/home/jetson/jetformer/data')
DS = DATA / 'datasets' / 'dataset_v2_20261025'
OUT = Path(__file__).resolve().parent / 'out'


def main():
    D = np.load(OUT / 'd1_samples.npz')
    new = D['group'] != 'B41(10/10 人)'
    idx = np.nonzero(new)[0]
    cls, dtc, since = [], [], []
    cache = {}
    for i in idx:
        s, f = str(D['scene'][i]), int(D['frame'][i])
        log = s[:15]
        if s not in cache:
            with np.load(DS / 'aligned' / (s + '.npz')) as z:
                cache[s] = z['timestamps_us']
        t = int(cache[s][f])
        ev = json.loads((DATA / 'labels' / (log + '.json')).read_text())['events']
        tks = [e for e in ev.get('takeovers', []) if e['t_us'] <= t]
        if tks:
            e = max(tks, key=lambda e: e['t_us'])
            cls.append(e.get('class', '?'))
            dtc.append(e.get('dt_since_collision_s'))
            since.append((t - e['t_us']) / 1e6)
        else:
            cls.append('交代なし')
            dtc.append(None)
            since.append(np.nan)
    cls = np.array(cls)
    since = np.array(since, float)
    out = {}
    for c in sorted(set(cls.tolist())):
        for win, wm in (('交代の後 3 秒以内', since < 3.0), ('3 秒より後', ~(since < 3.0))):
            m = (cls == c) & wm
            if not m.any():
                continue
            j = idx[m]
            mv = D['v'][j] >= 1.0
            out['%s / %s' % (c, win)] = {
                'n': int(m.sum()),
                'fwd_row2_moving_us': float(np.mean(1496.0 - D['thr2'][j][mv])) if mv.any() else None,
                'brake_frac_row0_moving': float(np.mean(D['thr0'][j][mv] > 1598)) if mv.any() else None,
                'brake_frac_row0_v<1': float(np.mean(D['thr0'][j][~mv] > 1598)) if (~mv).any() else None,
                'v_median': float(np.median(D['v'][j]))}
    (OUT / 'd2_takeover_class.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, r in out.items():
        print('%-36s %s' % (k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}))


if __name__ == '__main__':
    sys.exit(main())
