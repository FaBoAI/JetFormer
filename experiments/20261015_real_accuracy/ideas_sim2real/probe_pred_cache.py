#!/usr/bin/env python3
"""案出しの小さな確かめ(CPU だけ・学習なし・判定には使わない)。

保存済みの予測(data/zoo の npz)だけを使って、シミュレーターの混ぜ方の「どこが効いて、どこが悪くしたか」を分ける。

  1. ステアだけ・スロットルだけの L1(人のきれいな 4,670 フレーム、best.pth)
  2. 組み合わせ(仮想の腕): C-DIV のステア + A のスロットル(同じ種)
     → 「シミュレーターはステアだけ教える(スロットルの損失を外す)」案の上限に近い見積もり
  3. 平均(アンサンブル): A 種 0 + A 種 1、A + C-DIV など → シミュレーターの腕が「種の違い」以上の多様さを持つか
  4. ジャンル別・低速の直線(HS)を除いた値

出力: probe_pred_cache.json(このファイルの隣)と、標準出力の表。
使い方: cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/20261015_real_accuracy/ideas_sim2real/probe_pred_cache.py
"""
import json
import os
import sys

import numpy as np

JF = '/home/jetson/jetformer'
sys.path.insert(0, JF)
sys.path.insert(0, JF + '/tools')
import rl_evaluate as ev  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = JF + '/data/zoo'
FILES = {
    'A_s0': 'policy_zoo_20261012/pred/jetformer_v0_s0.npz', 'A_s1': 'policy_zoo_20261012/pred/jetformer_v0_s1.npz',
    'B': 'policy_zoo_20261012/pred/B.npz',
    'C_s0': 'sim_cotrain_20261013/pred/C_s0.npz', 'C_s1': 'sim_cotrain_20261013/pred/C_s1.npz',
    'CFT_s0': 'sim_cotrain_20261013/pred/CFT_s0.npz', 'CFT_s1': 'sim_cotrain_20261013/pred/CFT_s1.npz',
    'AFT_s0': 'sim_cotrain_20261013/pred/AFT_s0.npz', 'AFT_s1': 'sim_cotrain_20261013/pred/AFT_s1.npz',
    'CDIV_s0': 'sim_diverse_train_20261015/pred/CDIV_s0.npz', 'CDIV_s1': 'sim_diverse_train_20261015/pred/CDIV_s1.npz',
    'dinov2_s0': 'policy_zoo_20261012/pred/dinov2_s0.npz', 'dinov2_s1': 'policy_zoo_20261012/pred/dinov2_s1.npz',
    'act_s0': 'policy_zoo_20261012/pred/act_s0.npz', 'act_s1': 'policy_zoo_20261012/pred/act_s1.npz',
    'msfusion_s0': 'policy_zoo_20261012/pred/msfusion_s0.npz', 'msfusion_s1': 'policy_zoo_20261012/pred/msfusion_s1.npz',
    'tcp_s0': 'policy_zoo_20261012/pred/tcp_s0.npz', 'tcp_s1': 'policy_zoo_20261012/pred/tcp_s1.npz',
}


def r(x, nd=3):
    return round(float(x), nd)


def main():
    held = ev._heldout()
    tab = ev.pred_table()
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    y = np.nan_to_num(cols['y'].astype(np.float64))
    P = {}
    for k, p in FILES.items():
        path = os.path.join(ZOO, p)
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                P[k] = z['pred'].astype(np.float64)
    # 低速の直線(HS)の札: 10/15 の報告で差の 96% が出た場所
    fr = cols['frow']
    loc = np.full(len(fr), -1)
    loc[m0] = f['loc_code'][fr[m0]]
    hs = clean & (loc == ev.LOC_NAMES.index('HS'))
    g4 = clean & (cols['mode'] == 4)
    g5 = clean & (cols['mode'] == 5)

    def l1(pred, mask, ch=None):
        e = np.abs(pred - y)
        e = e.mean(axis=1)                       # [N,2] 10 ステップの平均
        if ch is None:
            return float(e[mask].mean())
        return float(e[mask, ch].mean())

    def row(pred):
        return {'all': r(l1(pred, clean)), 'steer': r(l1(pred, clean, 0)), 'throttle': r(l1(pred, clean, 1)),
                'g4': r(l1(pred, g4)), 'g5': r(l1(pred, g5)), 'no_HS': r(l1(pred, clean & ~hs)), 'HS': r(l1(pred, hs)),
                'HS_throttle_bias': r((pred[hs, :, 1] - y[hs, :, 1]).mean())}

    out = {'frames': {'clean': int(clean.sum()), 'HS': int(hs.sum()), 'g4': int(g4.sum()), 'g5': int(g5.sum())},
           'single': {k: row(v) for k, v in P.items()}}
    # 組み合わせ: ステアを X から、スロットルを A から(同じ種)
    hyb = {}
    for x in ('C', 'CDIV', 'CFT'):
        for s in (0, 1):
            a, b = P.get('A_s%d' % s), P.get('%s_s%d' % (x, s))
            if a is None or b is None:
                continue
            h = a.copy()
            h[:, :, 0] = b[:, :, 0]
            hyb['%s_steer+A_thr_s%d' % (x, s)] = row(h)
    out['hybrid'] = hyb
    # 平均(アンサンブル)
    ens = {}

    def add(name, keys):
        if all(k in P for k in keys):
            ens[name] = row(np.mean([P[k] for k in keys], axis=0))
    add('A_s0+A_s1', ['A_s0', 'A_s1'])
    add('A_s0+CDIV_s0', ['A_s0', 'CDIV_s0'])
    add('A_s1+CDIV_s1', ['A_s1', 'CDIV_s1'])
    add('A_s0+C_s0', ['A_s0', 'C_s0'])
    add('A_s0+A_s1+B', ['A_s0', 'A_s1', 'B'])
    add('A_s0+A_s1+CDIV_s0+CDIV_s1', ['A_s0', 'A_s1', 'CDIV_s0', 'CDIV_s1'])
    add('A_s0+A_s1+C_s0+C_s1', ['A_s0', 'A_s1', 'C_s0', 'C_s1'])
    add('A_s0+A_s1+AFT_s0+AFT_s1', ['A_s0', 'A_s1', 'AFT_s0', 'AFT_s1'])
    add('all_v0like_8', ['A_s0', 'A_s1', 'B', 'C_s0', 'C_s1', 'CDIV_s0', 'CDIV_s1', 'AFT_s0'])
    add('zoo_10', ['A_s0', 'A_s1', 'dinov2_s0', 'dinov2_s1', 'act_s0', 'act_s1', 'msfusion_s0', 'msfusion_s1',
                   'tcp_s0', 'tcp_s1'])
    out['ensemble'] = ens
    # 予測どうしの違い(L1 の距離): 種どうし vs 腕どうし
    dist = {}
    for a, b in (('A_s0', 'A_s1'), ('A_s0', 'B'), ('A_s0', 'CDIV_s0'), ('A_s1', 'CDIV_s1'), ('CDIV_s0', 'CDIV_s1'),
                 ('A_s0', 'C_s0'), ('C_s0', 'C_s1'), ('A_s0', 'dinov2_s0')):
        if a in P and b in P:
            d = np.abs(P[a] - P[b]).mean(axis=1)
            dist['%s|%s' % (a, b)] = {'all': r(d[clean].mean()), 'steer': r(d[clean, 0].mean()),
                                       'throttle': r(d[clean, 1].mean())}
    out['pred_distance'] = dist
    # ステップごとの誤差(A の平均): 0 ステップ目(今)と 9 ステップ目(0.33 秒先)
    if 'A_s0' in P:
        e = np.abs(P['A_s0'] - y)[clean]
        out['A_s0_per_step'] = {'steer': [r(v, 1) for v in e[:, :, 0].mean(0)], 'throttle': [r(v, 1) for v in e[:, :, 1].mean(0)]}
    with open(os.path.join(HERE, 'probe_pred_cache.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
