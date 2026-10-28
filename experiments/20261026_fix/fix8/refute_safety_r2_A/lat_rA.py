"""票 A: 本物の記録(10/23・10/24。読むだけ)の AI_CMD で、処理したフレームの latency_ms(書いた直後 − 撮像)・infer_ms と、
AI 側の処理したフレームの間の分け(0.1 秒より長い・0.13〜0.31 秒)を数える。
    python -I -B lat_rA.py → 標準出力"""
import glob
import sys

import numpy as np

lat, inf, gaps = [], [], []
for day in ('20261023', '20261024'):
    for f in sorted(glob.glob('/home/jetson/jetformer/data/raw/%s-*/*/AI_CMD.npz' % day)):
        try:
            z = np.load(f, allow_pickle=False)
        except Exception as e:   # noqa: BLE001
            print('skip', f, e)
            continue
        keys = set(z.files)
        if 'latency_ms' not in keys:
            continue
        hold = z['hold'].astype(bool) if 'hold' in keys else np.zeros(len(z['latency_ms']), bool)
        sel = z['select'].astype(bool) if 'select' in keys else np.ones(len(hold), bool)
        dry = z['dry_run'].astype(bool) if 'dry_run' in keys else np.zeros(len(hold), bool)
        ok = (~hold) & sel & (~dry)
        l = z['latency_ms'][ok]
        lat.extend(l[np.isfinite(l)].tolist())
        if 'infer_ms' in keys:
            i = z['infer_ms'][ok]
            inf.extend(i[np.isfinite(i)].tolist())
        tk = 'timestamps_us' if 'timestamps_us' in keys else None
        if tk is not None:
            t = z[tk][ok].astype(np.int64)
            if len(t) > 1:
                gaps.extend((np.diff(t) / 1e6).tolist())
print('keys example', sorted(keys))
lat, inf, gaps = np.array(lat), np.array(inf), np.array(gaps)
print('frames', len(lat), 'latency_ms p5/50/95/99', np.percentile(lat, [5, 50, 95, 99]).round(1))
if len(inf):
    print('infer_ms p5/50/95/99', np.percentile(inf, [5, 50, 95, 99]).round(1))
if len(gaps):
    g = gaps[(gaps > 0) & (gaps < 5)]
    print('gaps', len(g), '>0.1:', int((g > 0.1).sum()), '0.12-0.31:', int(((g > 0.12) & (g <= 0.31)).sum()),
          '>0.5:', int((g > 0.5).sum()))
