"""safety2-1: 10/23・10/24 の AI 側の本の数・AI 側の秒・リカバリーの回数(率の目安)。読むだけ。0 バイトは飛ばす"""
import glob, os
import numpy as np
n_sc, ai_s, n_rec = 0, 0.0, 0
for sc in sorted(glob.glob('/home/jetson/jetformer/data/raw/2026102[34]-*/*/AI_CMD.npz')):
    if os.path.getsize(sc) == 0:
        continue
    try:
        z = np.load(sc); sel = z['select'].astype(bool); a = z['assist'].astype(int); t = z['timestamps_us'].astype(np.int64)
    except Exception:
        continue
    if sel.sum() < 30:
        continue
    n_sc += 1
    ai_s += sel.sum() / 30.0
    n_rec += int(((a[1:] == 2) & (a[:-1] != 2)).sum())
print('AI の本', n_sc, 'AI 側 約', round(ai_s / 60, 1), '分', 'リカバリー', n_rec)
