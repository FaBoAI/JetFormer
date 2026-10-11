"""方策の学習データの「目標の偏り」(CPU、学習前に分かること)。

各方策が学習する目標(未来10フレームの操作の平均)が、場所ごとに (c) filtered BC の目標からどちらへずれるかを計算する。
  (b) AWR: 重み付き平均 − (b) のフレームの重みなし平均、および − (c) の平均
  (a)/(n): 「〜・良」で学習するフレーム(I=1 / I_neg=1)の平均 − (c) の平均
スロットルは µs が小さいほど前進(中立 約1496)、ステアは µs が大きいほど左。結果は仮説の材料で、方策の出力ではない。
出力: ../target_shift.json
"""
import json
import os
import sys

import numpy as np

J = '/home/jetson/jetformer'
sys.path.insert(0, J)
sys.path.insert(0, J + '/tools')
import rl_policies as rp  # noqa: E402

P = J + '/data/rl/ai_mix_20261011/policy/'
fr = np.load(J + '/data/rl/ai_mix_20261011/frames.npz', allow_pickle=False)
ad = np.load(J + '/data/rl/ai_mix_20261011/advantages.npz', allow_pickle=False)
dc = np.load(P + 'drop_clean.npz')
da = np.load(P + 'drop_all.npz')
w = np.load(P + 'awr_weights.npz')['weight'].astype(np.float64)
imp, _ = rp._impact_and_stride('ai_mix_20261011', fr['scene'], fr['frame'], 2.5)
st = (fr['frame'] % 3) == 0
used_c = ~dc['drop'] & ~imp & st
used_a = ~da['drop'] & ~(imp & ~da['keep_impact']) & st
fa = fr['future_actions'].mean(axis=1)            # [N,2] 未来10フレームの平均(ステア, スロットル)
LOC = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
out = {}
for split in ('train', 'val'):
    sp = fr['split'] == split
    rows = {}
    for li, l in enumerate(['all'] + LOC):
        lm = np.ones(len(st), bool) if l == 'all' else fr['loc_code'] == li - 1
        for g in (4, 5, 'all'):
            gm = np.ones(len(st), bool) if g == 'all' else ad['genre'] == g
            c = used_c & sp & lm & gm
            a = used_a & sp & lm & gm
            if c.sum() < 20:
                continue
            base = fa[c].mean(axis=0)
            e = {'n_c': int(c.sum()), 'n_all': int(a.sum()),
                 'c_steer': round(float(base[0]), 1), 'c_thr': round(float(base[1]), 1)}
            wa = w[a]
            mw = (fa[a] * wa[:, None]).sum(axis=0) / wa.sum()
            mu = fa[a].mean(axis=0)
            e['b_minus_c'] = [round(float(x), 1) for x in mw - base]
            e['b_weight_effect'] = [round(float(x), 1) for x in mw - mu]
            for v, lab in (('a', ad['I']), ('n', ad['I_neg'])):
                gd = a & (lab == 1)
                if gd.sum() >= 10:
                    e['%s_good_minus_c' % v] = [round(float(x), 1) for x in fa[gd].mean(axis=0) - base]
                    e['%s_good_n' % v] = int(gd.sum())
            rows['%s/%s' % (l, g)] = e
    out[split] = rows
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'target_shift.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('場所/ジャンル  n(c)  (c)の平均 ステア・スロットル | (b)−(c) | (b)重みの効果 | (a)良−(c) | (n)良−(c)   [µs、スロットル負=前進側、ステア正=左]')
for k, e in out['train'].items():
    if k.endswith('/all') or k.startswith('all/'):
        print('%-12s %5d  %6.0f %6.0f | %s | %s | %s | %s' % (k, e['n_c'], e['c_steer'], e['c_thr'], e['b_minus_c'],
                                                        e['b_weight_effect'], e.get('a_good_minus_c'), e.get('n_good_minus_c')))
