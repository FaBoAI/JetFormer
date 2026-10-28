"""票 A: 本物の記録(10/23・10/24。読むだけ)のリカバリーの行(assist 2)ごとに、時刻・出した値(out)・モデルの値(model_out)・
VEHICLE の eRPM(撮像時刻の前の最新)を並べる。後退側(1796)の終わりの前後で、モデルが前進側を出していたか(OFF で手順が
戻ったとき明けたフレームに出る値)を見る。 python -I -B model_rec_rA.py → 標準出力"""
import glob

import numpy as np

for f in sorted(glob.glob('/home/jetson/jetformer/data/raw/2026102[34]-*/*/AI_CMD.npz')):
    try:
        z = np.load(f, allow_pickle=False)
    except Exception:   # noqa: BLE001
        continue
    if 'assist' not in z.files:
        continue
    a = z['assist'].astype(int)
    m = (a == 2) & (~z['hold'].astype(bool))
    if not m.any():
        continue
    try:
        v = np.load(f.replace('AI_CMD.npz', 'VEHICLE.npz'), allow_pickle=False)
        vt = v['timestamps_us'].astype(np.int64)
        ve = v['erpm'] if 'erpm' in v.files else None
    except Exception:   # noqa: BLE001
        vt, ve = None, None
    t = z['timestamps_us'].astype(np.int64)
    idx = np.where(m)[0]
    i0, i1 = max(idx[0] - 2, 0), min(idx[-1] + 6, len(a) - 1)
    print('==', f.split('/')[-2])
    for i in range(i0, i1 + 1):
        e = None
        if ve is not None:
            k = np.searchsorted(vt, t[i], side='right') - 1
            e = int(ve[k]) if k >= 0 else None
        print('  %.3f a%d out %6.1f model %6.1f erpm %s hold %d' % ((t[i] - t[i0]) / 1e6, a[i], z['out'][i, 1], z['model_out'][i, 1],
                                                                 e, int(z['hold'][i])))
