#!/usr/bin/env python3
"""代わりの指標(通過ごとの割合)の検出力: 片側 Fisher の正確検定 p<0.05、モンテカルロ 4000 回(CPU、読み取りなし)
   nice -n 15 python proxy_power.py > proxy_power.json"""
import json
import math
import sys

import numpy as np


def fisher_one_sided(a, n1, b, n2):
    """群1(基準) a/n1 と群2(新) b/n2。新のほうが少ない(b/n2 < a/n1)の片側 p"""
    k = a + b
    N = n1 + n2
    tot = math.comb(N, k)
    # 群2 の成功数が b 以下になる確率(周辺を固定)
    p = 0.0
    for x in range(0, b + 1):
        if k - x <= n1 and x <= n2:
            p += math.comb(n2, x) * math.comb(n1, k - x) / tot
    return p


rng = np.random.default_rng(0)
out = []
for name, p0, p1 in [('R1 進入 ≥6200 の割合 22%→5%', 0.22, 0.05), ('R1 進入 ≥6200 の割合 22%→10%', 0.22, 0.10),
                     ('DE 通過あたり衝突 65%→30%', 0.65, 0.30), ('DE 通過あたり衝突 65%→45%', 0.65, 0.45)]:
    row = {'metric': name}
    for n in (24, 43, 60):
        hits = 0
        for _ in range(4000):
            a = rng.binomial(n, p0)
            b = rng.binomial(n, p1)
            if fisher_one_sided(a, n, b, n) < 0.05:
                hits += 1
        row['power_n%d' % n] = round(hits / 4000, 2)
    out.append(row)
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
