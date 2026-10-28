"""票 A: 本物の記録(10/10・10/23・10/24。読むだけ)のリカバリーで、後退側の行(assist 2・out > 中立 + 40)のすぐあとに続く
中立の行(assist 2・out = 中立。'gap' は中立 + 10 なので除く)= 'coast' の行の model_out(ゲインのあと・ToF ガードの前)の
スロットル。OFF で手順が戻ると、このフレームではこの値が出る(後退側なら Assist が中立にする)。
    python -I -B model_coast2_rA.py → 標準出力"""
import glob

import numpy as np

res = []
for day in ('20261010', '20261023', '20261024'):
    for f in sorted(glob.glob('/home/jetson/jetformer/data/raw/%s-*/*/AI_CMD.npz' % day)):
        try:
            z = np.load(f, allow_pickle=False)
        except Exception:   # noqa: BLE001
            continue
        if 'assist' not in z.files or 'model_out' not in z.files:
            continue
        a, out, mo, hold = z['assist'].astype(int), z['out'], z['model_out'], z['hold'].astype(bool)
        r2 = (a == 2) & (~hold)
        if not r2.any():
            continue
        vals, cnt = np.unique(np.round(out[r2, 1]), return_counts=True)
        # 中立 = escape(中立 − 130)の値 + 130
        esc = [v for v in vals if any(abs((v + 130) - w) < 1 and abs(w - 1500) < 40 for w in vals)]
        if not esc:
            continue
        nn = float(esc[0] + 130)
        i = 1
        while i < len(a):
            if r2[i] and abs(out[i, 1] - nn) < 0.6 and r2[i - 1] and out[i - 1, 1] > nn + 40:
                j = i
                seq = []
                while j < len(a) and r2[j] and abs(out[j, 1] - nn) < 0.6:
                    seq.append(round(float(mo[j, 1]) - nn, 1))
                    j += 1
                nxt = round(float(out[j, 1]) - nn, 1) if j < len(a) else None
                res.append((f.split('/')[-2], int(z['timestamps_us'][i] // 1000), seq, nxt))
                i = j
            else:
                i += 1
for r in res:
    print(r)
allv = np.array([v for r in res for v in r[2]])
print('coast の続き', len(res), '行', len(allv), '前進側(< −20)', int((allv < -20).sum()), '中立の帯', int(((allv >= -20) & (allv <= 40)).sum()),
      '後退側(> +40)', int((allv > 40).sum()))
first = np.array([r[2][0] for r in res])
print('各 coast の 1 行目: 前進側', int((first < -20).sum()), '/', len(first))
