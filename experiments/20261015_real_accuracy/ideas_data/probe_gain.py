#!/usr/bin/env python3
"""下調べ 2(CPU だけ・学習なし): 電池の減り(同じスロットルで出る速さの低下)を、過去のスロットルと eRPM から
因果に見積もって、予測の前進の強さを掛け直すと、保留の誤差はどれだけ変わるか。

ねらい: 保留のシーン 20261010-193638-0003(高速)は、後半で人がスロットルをほぼいっぱい(1044 µs)にしても eRPM が
2,000 まで落ちる(電池切れの形)。モデルは今の操作値を入れないので、この「車の効きの低下」を知らない。

やり方(結果を見る前に決めた既定値。感度は参考に並べる):
  - 効きの比 g = (中立 1498 − 今のスロットル) / (前進 eRPM / 1000)。前進 eRPM > 2000・衝突の前後でないフレームだけ
  - g_t = 同じシーンの過去 W 秒(既定 5 秒)の g の中央値(今のフレームを含む、未来は使わない)。フレームが 30 未満なら補正しない
  - g_ref = 学習の 41 シーンで同じ量の中央値(ジャンル別)
  - 予測のスロットルが前進側(< 中立)のときだけ: 中立 − (中立 − 予測) × clip(g_t / g_ref, 0.8, 2.0)
走行のときは「今のスロットル」= AI が実際に出した値になる(人の操作を入れるのではなく、車の効きの見積もりにだけ使う)。

    /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \
        python experiments/20261015_real_accuracy/ideas_data/probe_gain.py
"""

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
from jetformer.train_data import impact_mask  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/ideas_data/probe_gain.json')
CLEAN = ROOT / 'data/datasets/clean_driving_20261011'
NEUTRAL = 1498.0
MIN_ERPM = 2000.0


def ratio_series(a, impact_g=2.5):
    thr = a['vehicle'][:, 1].astype(np.float64)
    fwd = -a['vehicle'][:, 3].astype(np.float64)
    imp = impact_mask(a['timestamps_us'], a['imu'], impact_g)
    ok = a['vehicle_valid'] & (fwd > MIN_ERPM) & ~imp
    g = np.where(ok, (NEUTRAL - thr) / np.maximum(fwd, 1.0) * 1000.0, np.nan)
    return g


def running_median(g, t_us, W, min_n=30):
    out = np.full(len(g), np.nan)
    t = t_us / 1e6
    for i in range(len(g)):
        lo = np.searchsorted(t, t[i] - W)
        w = g[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_n:
            out[i] = np.median(w)
    return out


def main():
    sp = json.loads((CLEAN / 'splits.json').read_text())
    # g_ref(学習のシーンだけ)
    gs = {4: [], 5: []}
    for s in sp['train']:
        a = dict(np.load(CLEAN / 'aligned' / (s + '.npz')))
        g = ratio_series(a)
        m = a['mode']
        for k in (4, 5):
            v = g[(m == k) & np.isfinite(g)]
            gs[k].append(v)
    g_ref = {k: float(np.median(np.concatenate(v))) for k, v in gs.items()}

    tab = ev.pred_table()
    held = ev._heldout()
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    y = np.nan_to_num(cols['y'])
    sc, fr = tab['scene'], tab['frame']
    scenes = sorted(set(sc[clean].tolist()))
    zoo = ROOT / 'data/zoo'
    names = {'v0_s0': zoo / 'policy_zoo_20261012/pred/jetformer_v0_s0.npz', 'v0_s1': zoo / 'policy_zoo_20261012/pred/jetformer_v0_s1.npz',
             'B': zoo / 'policy_zoo_20261012/pred/B.npz'}
    preds = {n: np.load(p)['pred'].astype(np.float64) for n, p in names.items()}
    preds['v0_s0_last'] = np.load(zoo / 'policy_zoo_20261012/pred/jetformer_v0_s0_last.npz')['pred'].astype(np.float64)
    preds['v0_s1_last'] = np.load(zoo / 'policy_zoo_20261012/pred/jetformer_v0_s1_last.npz')['pred'].astype(np.float64)
    preds['v0ens'] = 0.5 * (preds['v0_s0'] + preds['v0_s1'])

    # 保留のシーンごとに g_t(因果)
    def gt_rows(W):
        gt = np.full(len(sc), np.nan)
        for s in scenes:
            a = dict(np.load(CLEAN / 'aligned' / (s + '.npz')))
            run = running_median(ratio_series(a), a['timestamps_us'].astype(np.float64), W)
            rows = np.nonzero(m0 & (sc == s))[0]
            gt[rows] = run[fr[rows]]
        return gt

    def L(p, mask=clean):
        return float(np.abs(p - y)[mask].mean())

    def correct(p, gt, lo=0.8, hi=2.0):
        q = p.copy()
        ref = np.array([g_ref.get(int(m), np.nan) for m in cols['mode']])
        scale = np.clip(gt / ref, lo, hi)
        scale = np.where(np.isfinite(scale), scale, 1.0)
        d = NEUTRAL - q[:, :, 1]
        fwd = d > 0
        q[:, :, 1] = np.where(fwd, NEUTRAL - d * scale[:, None], q[:, :, 1])
        return q, scale

    res = {'g_ref': {k: round(v, 2) for k, v in g_ref.items()}, 'n_clean': int(clean.sum()), 'variants': {}}
    target = clean & (sc == '20261010-193638-0003')
    others = clean & (sc != '20261010-193638-0003')
    for W in (3.0, 5.0, 10.0):
        gt = gt_rows(W)
        for lo, hi in ((0.8, 2.0), (1.0, 2.0), (0.9, 1.5)):
            key = 'W%.0f_clip%.1f-%.1f' % (W, lo, hi)
            v = {}
            for n, p in preds.items():
                q, scale = correct(p, gt, lo, hi)
                v[n] = {'before': round(L(p), 3), 'after': round(L(q), 3), 'gain': round(L(p) - L(q), 3),
                        'scene193638_0003_before_after': [round(L(p, target), 1), round(L(q, target), 1)],
                        'other_scenes_before_after': [round(L(p, others), 2), round(L(q, others), 2)]}
            v['scale_mean_by_scene'] = {s: round(float(np.nanmean(scale[clean & (sc == s)])), 3) for s in scenes}
            res['variants'][key] = v
    # 参考: 193638-0003 の後半を除いたときの全体(主指標の定義は変えない。床の内訳の参考)
    p = preds['v0ens']
    t_s = (fr.astype(np.float64)) / 30.0
    late = clean & (sc == '20261010-193638-0003') & (t_s >= 6.0)
    res['ref_v0ens_without_193638_0003_after6s'] = round(L(p, clean & ~late), 3)
    res['ref_v0ens_193638_0003_after6s_frames'] = int(late.sum())
    res['ref_v0ens_193638_0003_after6s_L1'] = round(L(p, late), 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
