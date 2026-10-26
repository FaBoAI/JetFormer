"""verify/bump_check.py — No.9 で、3 軸の |a| − 1G が 2.5G 以上(scorecard の参考の決まり cmap)なのに、
水平 2.5G(レポートの決まり)では衝突にならないところの、水平・上下の成分を見る。"""
import os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import rawload
lg = sys.argv[1] if len(sys.argv) > 1 else '20261024-162033'
r = rawload.load(lg)
it, acc = r['i_t'], r['i_acc']
er = np.interp(it, r['v_t'], r['v_erpm']); st = np.abs(er) < 1
b = np.median(acc[st], axis=0)
ah = np.hypot(acc[:, 0] - b[0], acc[:, 1] - b[1]) / rawload.G
az = (acc[:, 2] - b[2]) / rawload.G
a3 = np.abs(np.linalg.norm(acc, axis=1) / rawload.G - 1)
ai = np.interp(it, r['v_t'], r['v_sel'].astype(float)) > 0.5
idx = np.where((a3 >= 2.5) & ai)[0]
ev = []
for k in idx:
    if ev and it[k] - ev[-1][-1] <= 0.6:
        ev[-1][1].append(k); ev[-1][-1] = it[k]
    else:
        ev.append([k, [k], it[k]])
print(lg, 'events |a|-1G>=2.5G (AI):', len(ev))
for e in ev:
    ks = e[1]; w = slice(max(ks[0] - 5, 0), ks[-1] + 6)
    print('  t=%.2f  max|a|-1G=%.2f  max水平=%.2f  上下(z−1G)の幅 %.2f〜%.2f  → %s' % (
        it[ks[0]], a3[w].max(), ah[w].max(), az[w].min(), az[w].max(), '水平でも衝突' if ah[w].max() >= 2.5 else '上下だけ(段差・スロープ)'))
