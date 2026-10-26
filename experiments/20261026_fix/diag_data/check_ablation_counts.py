#!/usr/bin/env python3
"""out/ablation の表で、train.py と同じ選び方の学習のサンプル数を数える(numpy だけ)。drop_v2h と same の配列が同じかも見る。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_labels as D1

HERE = Path(__file__).resolve().parent


def count(table):
    with np.load(table) as z:
        sc, fr, dd = z['scene'].astype(str), z['frame'], z['drop']
    tab = {}
    for s in np.unique(sc):
        m = sc == s
        tab[s] = (fr[m], dd[m])
    splits = json.loads((D1.DS_V2 / 'splits.json').read_text())
    n = 0
    for s in splits['train']:
        a = np.load(D1.DS_V2 / 'aligned' / (s + '.npz'))
        ok = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
        d = np.zeros(len(ok), bool)
        f, x = tab[s]
        d[f] = x
        ok &= ~d
        ok &= ~D1.impact_mask(a['timestamps_us'], a['imu'])
        ok &= (np.arange(len(ok)) % D1.STRIDE) == 0
        n += int(ok.sum())
    return n


def main():
    a, b = np.load(D1.TAB / 'drop_v2h.npz'), np.load(HERE / 'out' / 'ablation' / 'drop_v2h_same.npz')
    same = all(np.array_equal(a[k], b[k]) for k in ('scene', 'frame', 'drop', 'reason'))
    res = {'same_arrays_as_drop_v2h': same}
    for name in ('drop_v2h_same.npz', 'drop_v2h_noTK.npz', 'drop_v2h_noTK_noLB.npz'):
        res[name] = count(HERE / 'out' / 'ablation' / name)
    print(json.dumps(res, ensure_ascii=False))
    (HERE / 'out' / 'ablation' / 'counts.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
