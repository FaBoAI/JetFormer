#!/usr/bin/env python3
"""人どうしの差 D を「状態の近さ」ごとに測り、近さ → 0 へ外挿する(CPU だけ)

保留の 4,670 行それぞれについて、学習のシーン(+ --pool all で val の別のシーン)の同じジャンル・±0.30 m の中の別の周の
フレームと組を作り、|y − y'|(10 フレーム × 2 の平均)を、状態の距離 δ(標準化した 横ずれ・向きのずれ・速さ・ヨーレート・
曲率・100 ms 前からのヨーレートの変化・進みの差)ごとに平均する。δ → 0 の切片 D0 が「同じ状態の人どうしの差」。
底 B は D0/2 ≤ B ≤ D0(正規分布なら D0/√2、ラプラス分布なら D0/1.5)。
"""
import json
import os
import sys

import numpy as np

REPO = '/home/jetson/jetformer'
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, REPO)
import course_map as cm  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = 0.30


def main(pool_mode='train'):
    C = np.load(os.path.join(HERE, 'cache', 'heldout_cache.npz'), allow_pickle=False)
    Lc = np.load(os.path.join(HERE, 'cache', 'lapcoord.npz'), allow_pickle=False)
    z = np.load(rp.rl_dir(ev.DATASET) / 'frames.npz', allow_pickle=False)
    f = {k: z[k] for k in ('scene', 'frame', 'mode', 'future_actions', 'future_valid', 'vehicle_valid', 'near_collision',
                           'erpm', 'gz', 'split')}
    cl = cm.Centerline()
    L = float(Lc['L'])
    sp, off, h = Lc['sp'], Lc['off'], Lc['h']
    _, hcl = cl.at(np.nan_to_num(sp))
    dh = np.degrees(np.angle(np.exp(1j * (h - hcl))))
    good = np.isfinite(sp) & (np.abs(off) < 0.8) & (np.abs(dh) < 60)
    clean_scenes = set(x[:-4] for x in os.listdir(os.path.join(REPO, 'data/datasets/clean_driving_20261011/aligned')))
    isclean = np.isin(f['scene'], list(clean_scenes))
    spd = -f['erpm'] * 0.5e-3
    gz = f['gz'].astype(np.float64)
    # 100 ms 前(3 行前、同じシーンのとき)のヨーレート
    gz_prev = np.full(len(gz), np.nan)
    same = np.r_[np.zeros(3, bool), (f['scene'][3:] == f['scene'][:-3]) & (f['frame'][3:] == f['frame'][:-3] + 3)]
    gz_prev[same] = gz[np.nonzero(same)[0] - 3]
    curv = np.radians(gz) / np.maximum(spd, 0.5)
    feats = np.stack([off, np.radians(dh), spd, gz / 100.0, curv, (gz - gz_prev) / 100.0], 1)
    pool_ok = isclean & good & f['future_valid'] & f['vehicle_valid'] & ~f['near_collision'] & np.isfinite(feats).all(1)
    if pool_mode == 'train':
        pool_ok &= f['split'] == 'train'
    sd = np.nanstd(feats[pool_ok], 0)
    feats = feats / sd
    Y = np.nan_to_num(f['future_actions'].astype(np.float64))
    ev_rows = C['frow'][C['clean']]
    y_ev = np.nan_to_num(C['y'][C['clean']].astype(np.float64))
    names = [str(x).split('/')[-1] for x in C['names']]
    PA = C['preds'][:, C['clean']].astype(np.float64)
    eA = 0.5 * (np.abs(PA[names.index('jetformer_v0_s0')] - y_ev).mean(axis=(1, 2)) +
                np.abs(PA[names.index('jetformer_v0_s1')] - y_ev).mean(axis=(1, 2)))
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    acc = {g: {'sum': np.zeros(len(bins) - 1), 'n': np.zeros(len(bins) - 1), 'sum_st': np.zeros((len(bins) - 1, 2))} for g in (4, 5)}
    nn1 = np.full(len(ev_rows), np.nan)       # 最も近い 1 組の差
    nn1_d = np.full(len(ev_rows), np.nan)
    rows_used = np.zeros(len(ev_rows), bool)
    row_sum = np.zeros((len(ev_rows), len(bins) - 1))
    row_n = np.zeros((len(ev_rows), len(bins) - 1))
    for g in (4, 5):
        pidx = np.nonzero(pool_ok & (f['mode'] == g))[0]
        psp = sp[pidx]
        o = np.argsort(psp)
        pidx, psp = pidx[o], psp[o]
        for j in np.nonzero(f['mode'][ev_rows] == g)[0]:
            r = ev_rows[j]
            if not (good[r] and np.isfinite(feats[r]).all()):
                continue
            cand = []
            for shift in (-L, 0.0, L):
                a = np.searchsorted(psp, sp[r] + shift - WIN)
                b = np.searchsorted(psp, sp[r] + shift + WIN)
                cand.append(pidx[a:b])
            cand = np.concatenate(cand)
            cand = cand[f['scene'][cand] != f['scene'][r]]
            if not len(cand):
                continue
            dsp = np.abs(((sp[cand] - sp[r]) + L / 2) % L - L / 2) / 0.10       # 0.10 m = 1
            d = np.sqrt(((feats[cand] - feats[r]) ** 2).sum(1) + dsp ** 2)
            diff = np.abs(Y[cand] - y_ev[j])
            e = diff.mean(axis=(1, 2))
            k = np.digitize(d, bins) - 1
            ok = (k >= 0) & (k < len(bins) - 1)
            np.add.at(acc[g]['sum'], k[ok], e[ok])
            np.add.at(acc[g]['n'], k[ok], 1)
            np.add.at(acc[g]['sum_st'], k[ok], diff[ok].mean(axis=1))
            np.add.at(row_sum[j], k[ok], e[ok])
            np.add.at(row_n[j], k[ok], 1)
            i = np.argmin(d)
            nn1[j], nn1_d[j] = e[i], d[i]
            rows_used[j] = True
    res = {'pool': pool_mode, 'bins_delta': bins.tolist(), 'rows_used': int(rows_used.sum())}
    for g in (4, 5):
        m = acc[g]['n'] > 0
        mean = np.where(m, acc[g]['sum'] / np.maximum(acc[g]['n'], 1), np.nan)
        st = acc[g]['sum_st'] / np.maximum(acc[g]['n'], 1)[:, None]
        c = 0.5 * (bins[:-1] + bins[1:])
        # δ < 2 の 5 つのビンで直線を当てて δ=0 の切片(重み = 組の数)
        sel = m & (c < 2.0)
        w = np.sqrt(acc[g]['n'][sel])
        A = np.stack([np.ones(sel.sum()), c[sel]], 1)
        coef = np.linalg.lstsq(A * w[:, None], mean[sel] * w, rcond=None)[0]
        gm = rows_used & (f['mode'][ev_rows] == g)
        res['genre%d' % g] = {'D_by_delta': [None if not np.isfinite(x) else round(float(x), 1) for x in mean],
                              'D_steer_by_delta': [round(float(x), 1) for x in st[:, 0]],
                              'D_throttle_by_delta': [round(float(x), 1) for x in st[:, 1]],
                              'pairs_by_delta': acc[g]['n'].astype(int).tolist(),
                              'D0_intercept': round(float(coef[0]), 1), 'slope': round(float(coef[1]), 2),
                              'nearest_pair_D': round(float(np.nanmean(nn1[gm])), 1),
                              'nearest_pair_delta_median': round(float(np.nanmedian(nn1_d[gm])), 2),
                              'A_on_rows': round(float(eA[gm].mean()), 2)}
    # 2 つのジャンルを行数で重み付け
    n4 = int((rows_used & (f['mode'][ev_rows] == 4)).sum())
    n5 = int((rows_used & (f['mode'][ev_rows] == 5)).sum())
    D0 = (res['genre4']['D0_intercept'] * n4 + res['genre5']['D0_intercept'] * n5) / (n4 + n5)
    res['D0_all'] = round(D0, 1)
    res['floor_from_D0'] = {'range_[D0/2,D0]': [round(D0 / 2, 1), round(D0, 1)], 'gauss_D0/sqrt2': round(D0 / np.sqrt(2), 1),
                            'laplace_D0/1.5': round(D0 / 1.5, 1)}
    res['A_on_rows_all'] = round(float(eA[rows_used].mean()), 2)
    # シーン単位のブートストラップ(2,000 回)と、外挿のやり方の違い(δ<1.5・δ<3 の直線、δ<3 の 2 次)
    c = 0.5 * (bins[:-1] + bins[1:])
    vs = C['scene'][C['clean']]
    modes = f['mode'][ev_rows]

    def d0(m, cmax=2.0, deg=1):
        out = []
        ns = []
        for g in (4, 5):
            mm = m & (modes == g)
            S, N = row_sum[mm].sum(0), row_n[mm].sum(0)
            ok = (N > 0) & (c < cmax)
            mean = S[ok] / N[ok]
            w = np.sqrt(N[ok])
            A = np.stack([c[ok] ** p for p in range(deg + 1)], 1)
            out.append(np.linalg.lstsq(A * w[:, None], mean * w, rcond=None)[0][0])
            ns.append(int((rows_used & mm).sum()))
        return (out[0] * ns[0] + out[1] * ns[1]) / max(ns[0] + ns[1], 1)
    res['D0_variants'] = {'lin_delta<1.5': round(d0(rows_used, 1.5), 1), 'lin_delta<2': round(d0(rows_used, 2.0), 1),
                          'lin_delta<3': round(d0(rows_used, 3.0), 1), 'quad_delta<3': round(d0(rows_used, 3.0, 2), 1),
                          'quad_delta<5': round(d0(rows_used, 5.0, 2), 1)}
    rng = np.random.default_rng(0)
    sc = np.unique(vs[rows_used])
    bs, ba, bd = [], [], []
    for _ in range(2000):
        pick = rng.choice(sc, len(sc), replace=True)
        m = np.zeros(len(ev_rows), bool)
        w = np.zeros(len(ev_rows))
        for s_ in pick:
            w[vs == s_] += 1
        # 重みつき: 同じシーンが 2 回選ばれたら 2 倍
        rs, rn = row_sum, row_n
        tot = []
        nsum = 0
        for g in (4, 5):
            mm = rows_used & (modes == g) & (w > 0)
            S = (rs[mm] * w[mm, None]).sum(0)
            N = (rn[mm] * w[mm, None]).sum(0)
            ok = (N > 0) & (c < 2.0)
            if ok.sum() < 3:
                continue
            mean = S[ok] / N[ok]
            ww = np.sqrt(N[ok])
            A = np.stack([np.ones(ok.sum()), c[ok]], 1)
            nn = float(w[mm].sum())
            tot.append(np.linalg.lstsq(A * ww[:, None], mean * ww, rcond=None)[0][0] * nn)
            nsum += nn
        D0b = sum(tot) / nsum
        Ab = float((eA[rows_used] * w[rows_used]).sum() / w[rows_used].sum())
        bs.append(D0b)
        ba.append(Ab)
        bd.append(Ab - D0b / np.sqrt(2))
    res['bootstrap_scene_95ci'] = {'D0': [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                                   'A': [round(float(np.percentile(ba, 2.5)), 1), round(float(np.percentile(ba, 97.5)), 1)],
                                   'A_minus_gaussfloor': [round(float(np.percentile(bd, 2.5)), 1), round(float(np.percentile(bd, 97.5)), 1)]}
    json.dump(res, open(os.path.join(HERE, 'out', 'pairs_vs_match_%s.json' % pool_mode), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'train')
