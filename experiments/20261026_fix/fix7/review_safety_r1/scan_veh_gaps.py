"""反証 7-1 回目・safety: 本物の記録(data/raw、読むだけ)の車両の値のサンプルの間(timestamps_us の差)の分かれ方。
AI 側(select)・日ごと。0.15 秒より短い止まり(中立止めにならない)がどれだけあるか。出力 out/scan_veh_gaps.json"""
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = '/home/jetson/jetformer/data/raw'
EDGES = [0.0, 0.015, 0.025, 0.035, 0.05, 0.06, 0.08, 0.1, 0.15, 0.3, 1e9]


def main():
    res = defaultdict(lambda: {'ai_s': 0.0, 'n': 0, 'hist': Counter()})
    for vp in sorted(glob.glob(ROOT + '/2026*/*/VEHICLE.npz')):
        if os.path.getsize(vp) == 0:
            continue
        try:
            v = np.load(vp)
            t = v['timestamps_us'].astype(np.int64)
            sel = v['select'].astype(bool)
        except Exception:      # noqa: BLE001
            continue
        day = vp.split('/')[-3][:8]
        d = np.diff(t) / 1e6
        ai = sel[1:] & sel[:-1]
        r = res[day]
        r['ai_s'] += float(d[ai & (d < 1.0)].sum())
        r['n'] += int(ai.sum())
        for x in d[ai]:
            k = next(i for i in range(len(EDGES) - 1) if EDGES[i] <= x < EDGES[i + 1])
            r['hist']['%.3f-%.3f' % (EDGES[k], EDGES[k + 1])] += 1
    out = {day: {'ai_s': round(r['ai_s'], 1), 'n': r['n'], 'hist': dict(sorted(r['hist'].items()))} for day, r in res.items()}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_veh_gaps.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for day, r in out.items():
        print(day, r)


if __name__ == '__main__':
    main()
