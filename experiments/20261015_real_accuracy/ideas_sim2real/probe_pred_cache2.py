#!/usr/bin/env python3
"""案出しの小さな確かめ 3(CPU だけ・学習なし・判定には使わない。保存済みの予測だけ)。

  1. 低速の直線(HS)の差は、ステアとスロットルのどちらから来たか(C-DIV − A)
  2. 間引きの位相: 主指標の 4,670 フレームを「フレーム番号 % 3」で 3 つに分けた誤差。学習は % 3 == 0 のフレームだけを見る
     (シミュレーターでは IMU の位相の 1 つしか見ないことが分かった)。0 だけが小さければ、位相の違いで損をしている
  3. 平均(アンサンブル)を 10/13 の規則の形で見る: 「種」= 重ならない組(E0 = A 種 0 + X 種 0、E1 = A 種 1 + X 種 1)。
     best・last の両方。ただし基準の A は 1 本のモデルなので、これは「アンサンブルは 1 本より良いか」で、シミュレーターの効き目ではない
  4. シーンごと: 2 本の平均が 1 本より良いシーンの数

出力: probe_pred_cache2.json(このファイルの隣)。
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
SRC = {'A': 'policy_zoo_20261012/pred/jetformer_v0_s%d%s.npz', 'C': 'sim_cotrain_20261013/pred/C_s%d%s.npz',
       'CDIV': 'sim_diverse_train_20261015/pred/CDIV_s%d%s.npz', 'AFT': 'sim_cotrain_20261013/pred/AFT_s%d%s.npz',
       'CFT': 'sim_cotrain_20261013/pred/CFT_s%d%s.npz'}
TH = {'best': 0.626, 'last': 1.322}


def r(x, nd=3):
    return round(float(x), nd)


def load(arm, s, ck):
    p = os.path.join(ZOO, SRC[arm] % (s, '' if ck == 'best' else '_last'))
    with np.load(p, allow_pickle=False) as z:
        return z['pred'].astype(np.float64)


def main():
    held = ev._heldout()
    tab = ev.pred_table()
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    clean = np.isin(tab['scene'], held['human_clean_scenes']) & m0 & cols['fv'] & ~cols['impact']
    y = np.nan_to_num(cols['y'].astype(np.float64))
    fr = cols['frow']
    loc = np.full(len(fr), -1)
    loc[m0] = f['loc_code'][fr[m0]]
    hs = clean & (loc == ev.LOC_NAMES.index('HS'))
    out = {}

    def err(p, mask):
        e = np.abs(p - y).mean(axis=1)
        return {'all': r(e[mask].mean()), 'steer': r(e[mask, 0].mean()), 'throttle': r(e[mask, 1].mean())}

    P = {(a, s, ck): load(a, s, ck) for a in SRC for s in (0, 1) for ck in ('best', 'last')}
    # 1. HS
    out['HS'] = {'frames': int(hs.sum())}
    for a in ('A', 'C', 'CDIV'):
        for s in (0, 1):
            out['HS']['%s_s%d' % (a, s)] = err(P[(a, s, 'best')], hs)
    # 2. 位相
    ph = tab['frame'] % 3
    out['phase'] = {'frames': {str(k): int((clean & (ph == k)).sum()) for k in range(3)}}
    for a in ('A', 'C', 'CDIV', 'AFT'):
        for s in (0, 1):
            out['phase']['%s_s%d' % (a, s)] = {str(k): err(P[(a, s, 'best')], clean & (ph == k))['all'] for k in range(3)}
    with np.load(os.path.join(ZOO, 'policy_zoo_20261012/pred/B.npz')) as z:
        pb = z['pred'].astype(np.float64)
    out['phase']['B'] = {str(k): err(pb, clean & (ph == k))['all'] for k in range(3)}
    # 3. アンサンブルを規則の形で
    ens = {}
    Lref = {ck: [err(P[('A', s, ck)], clean)['all'] for s in (0, 1)] for ck in ('best', 'last')}
    for x in ('A', 'C', 'CDIV', 'CFT', 'AFT'):
        row = {}
        for ck in ('best', 'last'):
            if x == 'A':
                vals = [err((P[('A', 0, ck)] + P[('A', 1, ck)]) / 2, clean)['all']]
            else:
                vals = [err((P[('A', s, ck)] + P[(x, s, ck)]) / 2, clean)['all'] for s in (0, 1)]
            gain = np.mean(Lref[ck]) - np.mean(vals)
            row[ck] = {'vals': [r(v) for v in vals], 'mean': r(np.mean(vals)), 'gain_vs_A_us': r(gain),
                       'all_below_A_min': bool(max(vals) < min(Lref[ck])), 'gain_gt_threshold': bool(gain > TH[ck])}
        ens['A+%s' % x] = row
    ens['note'] = ('A+A は 1 組だけ(A 種 0 + A 種 1)。A+X は E0 = A 種 0 + X 種 0、E1 = A 種 1 + X 種 1。'
                   'AFT・CFT は A・C から微調整したもので、元のモデルと似ている(独立な種ではない)')
    # 3 本の組(重ならない): E_s = A 種 s + C 種 s + C-DIV 種 s
    for name, arms in (('A+C+CDIV', ('A', 'C', 'CDIV')), ('A+C+CDIV+CFT', ('A', 'C', 'CDIV', 'CFT'))):
        row = {}
        for ck in ('best', 'last'):
            vals = [err(np.mean([P[(a, s, ck)] for a in arms], axis=0), clean)['all'] for s in (0, 1)]
            gain = np.mean(Lref[ck]) - np.mean(vals)
            row[ck] = {'vals': [r(v) for v in vals], 'mean': r(np.mean(vals)), 'gain_vs_A_us': r(gain),
                       'all_below_A_min': bool(max(vals) < min(Lref[ck])), 'gain_gt_threshold': bool(gain > TH[ck])}
        ens[name] = row
    out['ensemble_rule_form'] = ens
    # 4. シーンごと
    sc = {}
    for x in ('A', 'CDIV', 'C'):
        if x == 'A':
            pe = (P[('A', 0, 'best')] + P[('A', 1, 'best')]) / 2
        else:
            pe = (P[('A', 0, 'best')] + P[(x, 0, 'best')]) / 2
        better = 0
        n = 0
        for s in held['human_clean_scenes']:
            m = clean & (tab['scene'] == s)
            if not m.any():
                continue
            n += 1
            single = np.mean([err(P[('A', k, 'best')], m)['all'] for k in (0, 1)])
            better += int(err(pe, m)['all'] < single)
        sc['A+%s(種0)' % x if x != 'A' else 'A+A'] = '%d / %d シーンで 1 本(A の 2 本の平均)より小さい' % (better, n)
    out['per_scene'] = sc
    with open(os.path.join(HERE, 'probe_pred_cache2.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
