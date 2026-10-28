"""票 A: 本物の記録(10/10・10/23・10/24。読むだけ)のリカバリーの惰性('coast' = assist 2 で out のスロットルが中立)の行で、
モデルの出力(model_out)のスロットルが前進側(中立 − 20 より小さい)だったか。OFF で中立止めが手順を戻したとき、明けたフレームに
出るのはこの値(ゲイン・ToF ガードのあと)。 python -I -B model_coast_rA.py → 標準出力"""
import glob
import json

import numpy as np

n_rows, fwd, neu, rev = 0, 0, 0, 0
vals = []
ep = 0
for day in ('20261010', '20261023', '20261024'):
    for f in sorted(glob.glob('/home/jetson/jetformer/data/raw/%s-*/*/AI_CMD.npz' % day)):
        try:
            z = np.load(f, allow_pickle=False)
        except Exception:   # noqa: BLE001
            continue
        if 'assist' not in z.files or 'model_out' not in z.files:
            continue
        a = z['assist'].astype(int)
        out = z['out']
        mo = z['model_out']
        hold = z['hold'].astype(bool)
        # 中立: このコードの中立は ref[1]。記録には無いので、assist 2 の行の out のスロットルで最も多い中立の帯(1478〜1538)の値
        m = (a == 2) & (~hold) & (out[:, 1] >= 1478) & (out[:, 1] <= 1538)
        if not m.any():
            continue
        nn = float(np.median(out[m, 1]))
        idx = np.where(m)[0]
        # 惰性の続き(連続する行)ごとに 1 つ目の行
        starts = [i for i in idx if i - 1 not in set(idx)]
        for i in starts:
            ep += 1
            j = i
            while j in set(idx):
                t = float(mo[j, 1])
                vals.append(t - nn)
                n_rows += 1
                if t < nn - 20:
                    fwd += 1
                elif t > nn + 40:
                    rev += 1
                else:
                    neu += 1
                j += 1
print('coast の続き', ep, '行', n_rows, 'model_out のスロットル: 前進側', fwd, '中立の帯', neu, '後退側', rev)
if vals:
    v = np.array(vals)
    print('model_out − 中立 [µs] p5/50/95', np.percentile(v, [5, 50, 95]).round(1))
