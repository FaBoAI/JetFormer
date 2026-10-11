"""地点ごとの「通過あたりの衝突率」で A/B を判定するときの検出力(モンテカルロ、Fisher の正確検定・片側5%)。
基準の率は collision_map.md(006/007): R1 の強い衝突 9/36 通過 = 25%、DE の衝突 16/29 通過 = 55%(006 11/16 + 007 5/13)。
出力: pass_power.json(このフォルダ)
"""
import json, os, numpy as np
from math import comb
def fisher_greater(a, n1, b, n2):
    # 片側 p 値: 第1群の衝突数が a 以上になる確率(周辺度数を固定した超幾何分布)
    K = a + b; N = n1 + n2
    tot = comb(N, K)
    return sum(comb(n1, x) * comb(n2, K - x) for x in range(a, min(n1, K) + 1)) / tot
rng = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pass_power.json')
cases = {'R1 25%→5%': (0.25, 0.05), 'R1 25%→10%': (0.25, 0.10), 'DE 55%→25%': (0.55, 0.25), 'DE 55%→13%(人並み)': (0.55, 0.13)}
res = {}
for name, (p1, p2) in cases.items():
    res[name] = {}
    for n in (20, 30, 40, 60):
        hits = 0
        for _ in range(2000):
            a = rng.binomial(n, p1); b = rng.binomial(n, p2)
            p = fisher_greater(a, n, b, n)
            hits += p < 0.05
        res[name][n] = round(hits / 2000, 2)
    print(name, res[name])
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
