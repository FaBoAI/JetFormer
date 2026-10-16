#!/usr/bin/env python3
"""実機の精度の案出し(データ・学習の作り方の側)のための CPU だけの下調べ。学習はしない。

保存済みの予測(data/zoo/*/pred/*.npz、10/12〜10/15 の評価の道具が作ったもの。フレームの表 key 3d339512…)だけを読み、
10/13 の評価と同じ主指標(保留した人のきれいな走行 11 シーン・4,670 フレームの L1、10 ステップ × [ステア, スロットル])で
次を測る。どれも「後から見た下調べ」で、判定ではない(判定は案ごとに規則を先に固めた本番の学習で行う)。

  1. 種のアンサンブル(予測の平均・中央値)、best と last の平均(重みの平均の代わりの目安)
  2. 時間方向のアンサンブル(過去のフレームの予測のうち、同じ時刻を指すステップを平均。走行でも使える因果の形)
  3. 誤差の内訳(ジャンル × ステア/スロットル × ステップ)・一定のずれ(偏り)を引いたときの上限の目安
  4. フレーム番号 % 3(間引き 3 の位相)ごとの誤差 — 学習は位相 0 のフレームしか見ていない
  5. 種どうしの予測の食い違いと、人との誤差の比(アンサンブルで下げられる量の上限の目安)
  6. 人の操作の細かい揺れ(1 フレームごとのがたつき)の大きさ

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \
        python experiments/20261015_real_accuracy/ideas_data/probe_cached.py
出力: /home/jetson/jetformer_reports/real_accuracy_20261015/ideas_data/probe_cached.json
"""

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np  # noqa: E402

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import rl_evaluate as ev  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/ideas_data/probe_cached.json')
KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'


def r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def load_preds():
    out = {}
    for p in sorted(glob.glob(str(ROOT / 'data/zoo/*/pred/*.npz'))):
        name = Path(p).stem
        src = Path(p).parent.parent.name
        with np.load(p) as z:
            if str(z['key']) != KEY:
                continue
            ck = 'last' if name.endswith('_last') else 'best'
            base = name[:-5] if ck == 'last' else name
            out.setdefault(base, {})[ck] = z['pred'].astype(np.float64)
            out[base]['src'] = src
            out[base]['policy'] = str(z['policy'])
    return out


def main():
    tab = ev.pred_table()
    assert tab['key'] == KEY, tab['key']
    held = ev._heldout()
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    y = np.nan_to_num(cols['y'])
    n_clean = int(clean.sum())
    preds = load_preds()
    res = {'n_clean': n_clean, 'genre_counts': {int(g): int((clean & (cols['mode'] == g)).sum()) for g in (4, 5)},
           'models': {}}

    def L(p, mask=clean):
        return float(np.abs(p - y)[mask].mean())

    # ---- 1. 1 本ずつ
    for n, d in preds.items():
        res['models'][n] = {'src': d['src'], 'policy': d['policy'],
                            'best': r(L(d['best'])), 'last': r(L(d['last'])) if 'last' in d else None}
    # 名前(10/12: jetformer_v0_s0 など。10/13: AFT_s0 など。10/15: CDIV_s0)
    v0 = ['jetformer_v0_s0', 'jetformer_v0_s1']
    real_v0 = v0 + ['B', 'AFT_s0', 'AFT_s1']                      # 実機だけで学習した v0 の 5 本
    fam = real_v0 + ['C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'CDIV_s0', 'CDIV_s1']
    zoo = [n for n in preds if preds[n]['src'] == 'policy_zoo_20261012' and n != 'B']
    ens = {}

    def ens_of(names, ck='best', how='mean'):
        arr = np.stack([preds[n][ck] for n in names])
        return arr.mean(0) if how == 'mean' else np.median(arr, 0)

    ens['v0_s0+s1(best)'] = L(ens_of(v0))
    ens['v0_s0+s1(last)'] = L(ens_of(v0, 'last'))
    ens['v0_s0 best+last'] = L(0.5 * (preds[v0[0]]['best'] + preds[v0[0]]['last']))
    ens['v0_s1 best+last'] = L(0.5 * (preds[v0[1]]['best'] + preds[v0[1]]['last']))
    ens['v0 2 seeds x best+last (4)'] = L(np.mean([preds[n][c] for n in v0 for c in ('best', 'last')], 0))
    ens['real_v0 5 (A s0,s1,B,AFT s0,s1) mean'] = L(ens_of(real_v0))
    ens['real_v0 5 median'] = L(ens_of(real_v0, how='median'))
    ens['v0 family 11 (+C,CFT,CDIV) mean'] = L(ens_of(fam))
    ens['v0 family 11 median'] = L(ens_of(fam, how='median'))
    ens['zoo 10 policies mean'] = L(ens_of(zoo))
    ens['zoo 10 + family 11 mean'] = L(ens_of(sorted(set(zoo) | set(fam))))
    # 2 本のアンサンブル: 実機の v0 系の 5 本から作れる全部の組(独立の種の組だけでなく、A→AFT のように元が同じ組も入る)
    pairs = []
    for i in range(len(real_v0)):
        for j in range(i + 1, len(real_v0)):
            a, b = real_v0[i], real_v0[j]
            pairs.append({'pair': [a, b], 'single_mean': r(0.5 * (L(preds[a]['best']) + L(preds[b]['best']))),
                          'ens': r(L(0.5 * (preds[a]['best'] + preds[b]['best'])))})
    for p in pairs:
        p['gain'] = r(p['single_mean'] - p['ens'])
    res['ensembles'] = {k: r(v) for k, v in ens.items()}
    res['pair_ensembles_real_v0'] = pairs
    # 種どうしの食い違い(5)
    dis = {}
    for a, b in (('jetformer_v0_s0', 'jetformer_v0_s1'), ('jetformer_v0_s0', 'B'), ('AFT_s0', 'AFT_s1'),
                 ('jetformer_v0_s0', 'AFT_s0')):
        dis['%s|%s' % (a, b)] = r(float(np.abs(preds[a]['best'] - preds[b]['best'])[clean].mean()))
    res['seed_disagreement_L1'] = dis

    # ---- 2. 時間方向のアンサンブル(因果)。pred[t-j][k+j] は pred[t][k] と同じ時刻(t+k+1)を指す
    sc, fr = tab['scene'], tab['frame']
    row_of = {(s, int(i)): k for k, (s, i) in enumerate(zip(sc, fr)) if m0[k]}
    prev = np.full((len(sc), 10), -1, int)                        # prev[k, j] = フレーム t-j の行(無ければ -1)
    for k in np.nonzero(m0)[0]:
        for j in range(10):
            prev[k, j] = row_of.get((sc[k], int(fr[k]) - j), -1)

    def temporal(p, J, decay=0.0):
        out = p.copy()
        for k in range(10):
            num = p[:, k, :].copy()
            den = np.ones(len(p))
            for j in range(1, J + 1):
                if k + j > 9:
                    break
                rows = prev[:, j]
                ok = rows >= 0
                w = np.exp(-decay * j)
                num[ok] += w * p[rows[ok], k + j, :]
                den[ok] += w
            out[:, k, :] = num / den[:, None]
        return out

    tres = {}
    for n in v0 + ['B']:
        p = preds[n]['best']
        tres[n] = {'none': r(L(p))}
        for J in (1, 2, 3, 5, 9):
            tres[n]['J%d' % J] = r(L(temporal(p, J)))
        tres[n]['J9_decay0.3'] = r(L(temporal(p, 9, 0.3)))
        # 走行で使う lookahead 3(ステップ index 2)だけの L1
        tres[n]['k2_none'] = r(float(np.abs(p[:, 2] - y[:, 2])[clean].mean()))
        tres[n]['k2_J2'] = r(float(np.abs(temporal(p, 2)[:, 2] - y[:, 2])[clean].mean()))
        tres[n]['k2_J7'] = r(float(np.abs(temporal(p, 7)[:, 2] - y[:, 2])[clean].mean()))
    res['temporal_ensemble'] = tres

    # ---- 3. 誤差の内訳と偏り
    p = ens_of(v0)
    err = p - y
    dec = {}
    for g in (4, 5):
        mg = clean & (cols['mode'] == g)
        for c, cn in ((0, 'steer'), (1, 'throttle')):
            e = err[mg][:, :, c]
            dec['g%d_%s' % (g, cn)] = {'L1_by_step': [r(v, 1) for v in np.abs(e).mean(0)],
                                       'bias_by_step': [r(v, 1) for v in e.mean(0)],
                                       'median_err_by_step': [r(v, 1) for v in np.median(e, 0)],
                                       'L1': r(np.abs(e).mean(), 2)}
    res['decomposition_v0ens'] = dec
    # 全体への寄与(4,670 フレームの平均 L1 = Σ 部分)
    share = {}
    for g in (4, 5):
        mg = clean & (cols['mode'] == g)
        for c, cn in ((0, 'steer'), (1, 'throttle')):
            share['g%d_%s' % (g, cn)] = r(float(np.abs(err[mg][:, :, c]).sum() / (n_clean * 20)), 2)
    res['L1_share_v0ens_us'] = share
    # 一定のずれを引いた上限の目安: (a) 保留そのもので合わせた(楽観) (b) シーンを 1 つ抜いて合わせた(正直)
    scenes = sorted(set(sc[clean].tolist()))

    def shift_l1(p, by_scene=False, loso=False):
        q = p.copy()
        for g in (4, 5):
            mg = clean & (cols['mode'] == g)
            if by_scene:
                for s in scenes:
                    ms = mg & (sc == s)
                    if ms.any():
                        q[ms] -= np.median((p - y)[ms], axis=0)
            elif loso:
                for s in scenes:
                    ms = mg & (sc == s)
                    fit = mg & (sc != s)
                    if ms.any():
                        q[ms] -= np.median((p - y)[fit], axis=0)
            else:
                q[mg] -= np.median((p - y)[mg], axis=0)
        return L(q)

    res['shift_upper_bounds_v0ens'] = {'no_shift': r(L(p)), 'genre_step_channel_median_shift(in-sample)': r(shift_l1(p)),
                                       'same_but_leave_one_scene_out': r(shift_l1(p, loso=True)),
                                       'per_scene_shift(oracle)': r(shift_l1(p, by_scene=True))}
    # シーンごと
    res['per_scene'] = {s: {'n': int((clean & (sc == s)).sum()), 'genre': int(np.bincount(cols['mode'][clean & (sc == s)]).argmax()),
                            'v0_s0': r(L(preds['jetformer_v0_s0']['best'], clean & (sc == s)), 1),
                            'v0ens': r(L(p, clean & (sc == s)), 1),
                            'thr_bias_k2': r(float(err[clean & (sc == s)][:, 2, 1].mean()), 1),
                            'steer_bias_k2': r(float(err[clean & (sc == s)][:, 2, 0].mean()), 1)} for s in scenes}

    # ---- 4. フレーム番号 % 3 ごと
    ph = fr % 3
    phase = {}
    for n in v0 + ['B', 'AFT_s0', 'AFT_s1', 'C_s0', 'C_s1']:
        q = preds[n]['best']
        phase[n] = [r(L(q, clean & (ph == k)), 2) for k in range(3)]
    phase['_n'] = [int((clean & (ph == k)).sum()) for k in range(3)]
    # 誤差の差(位相 0 − 位相 1・2)のシーン単位のばらつき
    d_rows = []
    for s in scenes:
        q = ens_of(v0)
        a0 = L(q, clean & (sc == s) & (ph == 0))
        a12 = L(q, clean & (sc == s) & (ph != 0))
        d_rows.append(a12 - a0)
    phase['v0ens_phase12_minus_phase0_per_scene'] = [r(v, 2) for v in d_rows]
    phase['v0ens_phase12_minus_phase0_mean'] = r(float(np.mean(d_rows)), 3)
    res['phase_mod3'] = phase

    # ---- 6. 人の操作のがたつき(保留のきれいなフレーム): y[t, 0] と前後のフレームの平均との差
    jit = {}
    yy = y[:, 0, :]                     # 1 フレーム先の操作
    nxt = np.array([row_of.get((sc[k], int(fr[k]) + 1), -1) if m0[k] else -1 for k in range(len(sc))])
    prv = prev[:, 1]
    ok = clean & (nxt >= 0) & (prv >= 0)
    mid = 0.5 * (yy[prv[ok]] + yy[nxt[ok]])
    jit['L1_y_vs_neighbour_mean'] = [r(v, 2) for v in np.abs(yy[ok] - mid).mean(0)]
    jit['L1_consecutive_diff'] = [r(v, 2) for v in np.abs(yy[ok] - yy[prv[ok]]).mean(0)]
    # 10 ステップの正解の中で、今の値(1 フレーム先)からどれだけ動くか(人の操作の先の読みにくさの目安)
    jit['L1_step_k_vs_step0'] = [r(float(np.abs(y[clean][:, k] - y[clean][:, 0]).mean()), 1) for k in range(10)]
    # 今の人の操作(cur)を 10 ステップ全部の予測にしたときの L1(使えない基準。参考)
    cur = cols['cur'][:, None, :].repeat(10, 1)
    jit['L1_copy_current_human_action'] = r(L(cur))
    res['human_jitter'] = jit

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ('ensembles', 'temporal_ensemble', 'shift_upper_bounds_v0ens', 'phase_mod3',
                                          'seed_disagreement_L1', 'human_jitter', 'L1_share_v0ens_us')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
