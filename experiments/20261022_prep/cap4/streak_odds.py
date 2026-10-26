"""A2 の目安: 007 のような走り(R1 で止まって取り返した衝突が 22 周のうち 8 周)で、その 8 周の並びが偶然だけで決まるとき、
1 ブロック(約 22 周 = AI 240 秒)で「同じ場所が k 周続く」が起きる見込み。並べ替えの模擬(標準ライブラリだけ・種 1)。
B が 1 日に 3 ブロック(1・4・8)走るときの「1 回以上止まる」「2 回以上止まる(A2: 比べる走りをやめる)」も出す(ブロックは独立と仮定)。
  python -I streak_odds.py [out.json]"""
import json, random, sys
from math import comb
random.seed(1)


def has_run(seq, k):
    r = 0
    for x in seq:
        r = r + 1 if x else 0
        if r >= k:
            return True
    return False


N = 200000
out = {}
for name, n, m in (('007_like', 22, 8), ('006_like', 16, 1)):
    base = [1] * m + [0] * (n - m)
    c = {2: 0, 3: 0, 4: 0}
    for _ in range(N):
        random.shuffle(base)
        for k in c:
            c[k] += has_run(base, k)
    out[name] = {}
    for k in c:
        q = c[k] / N
        out[name][str(k)] = {'per_block': round(q, 3), 'B_ge1_of_3': round(1 - (1 - q) ** 3, 3),
                             'B_ge2_of_3': round(sum(comb(3, j) * q ** j * (1 - q) ** (3 - j) for j in (2, 3)), 3)}
print(json.dumps(out, ensure_ascii=False, indent=1))
if len(sys.argv) > 1:
    json.dump({'laps': {'007_like': [22, 8], '006_like': [16, 1]}, 'trials': N, 'seed': 1, 'odds': out}, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
