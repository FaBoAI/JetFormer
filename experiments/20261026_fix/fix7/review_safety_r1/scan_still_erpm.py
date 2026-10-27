"""反証 7-1 回目・safety: 本物の記録(data/raw、読むだけ)で、車が止まっている(hall_count が直前 0.1 秒変わらない)のに
eRPM が後ろへ 300 を超えるサンプル(rec_back_guard の back_early・'gap' の数えが、動いていない車を「下がっている」と見る入り口)。
AI 側(select)と人側に分ける。出力 out/scan_still_erpm.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = '/home/jetson/jetformer/data/raw'


def main():
    res = {'files': 0, 'samples': 0, 'still': 0, 'still_gt300': 0, 'still_gt300_ai': 0, 'runs_gt300': 0, 'examples': []}
    for vp in sorted(glob.glob(ROOT + '/2026*/*/VEHICLE.npz')):
        if os.path.getsize(vp) == 0:
            continue
        try:
            v = np.load(vp)
            t = v['timestamps_us'].astype(np.int64)
            e = v['erpm'].astype(int)
            h = v['hall_count'].astype(np.int64)
            sel = v['select'].astype(bool)
        except Exception:      # noqa: BLE001
            continue
        res['files'] += 1
        res['samples'] += len(t)
        j = 0
        prev_bad = False
        for i in range(len(t)):
            while t[i] - t[j] > 100_000:
                j += 1
            if t[i] - t[j] < 80_000:
                prev_bad = False
                continue
            if h[i] != h[j]:
                prev_bad = False
                continue
            res['still'] += 1
            if e[i] > 300:
                res['still_gt300'] += 1
                res['still_gt300_ai'] += int(sel[i])
                if not prev_bad:
                    res['runs_gt300'] += 1
                    if len(res['examples']) < 30:
                        res['examples'].append({'file': vp.replace(ROOT + '/', ''), 'i': i, 'ai': bool(sel[i]),
                                                'erpm': [int(x) for x in e[max(0, i - 8):i + 8]]})
                prev_bad = True
            else:
                prev_bad = False
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_still_erpm.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'examples'}, ensure_ascii=False))
    for x in res['examples'][:12]:
        print(x)


if __name__ == '__main__':
    main()
