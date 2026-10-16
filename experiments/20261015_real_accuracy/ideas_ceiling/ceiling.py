#!/usr/bin/env python3
"""天井(下げられない誤差の底)の見積もり(CPU だけ、学習なし)

同じ場所を走った人の別の周どうしで、未来 10 フレームの操作(ステア・スロットル)がどれだけ違うかを測り、
保留の 4,670 フレーム(主指標と同じ行)で次の「予測器」の L1 を出す。どれも人の別の周の値だけを使う(モデルなし)。

  F_loc    : 場所だけ。別の周それぞれで同じ場所(進み sp の差 < 0.10 m)のフレームを 1 つずつ取り、その中央値
  F_state  : 場所 + 車の状態。±0.30 m の中の別の周のフレームから、(横ずれ・向きのずれ・速さ・ヨーレート)が近い
             K 個(1 周から 1 つまで)の中央値。状態はどれもモデルの入力(画像・IMU・回転数)から原理的に分かる量
  D_state  : F_state で選んだ近い相手との差 |y - y'| の平均(2 人の人どうしの差)。同じ状態なら
             底 B は D/2 ≤ B ≤ D(正規分布なら D/√2、ラプラス分布なら D/1.5)
プールは「学習のシーン」(train の 41 シーン)だけ = 学習データから作れる予測器(公平)。--pool all で val の別の周も入れる。
不確かさ: 保留の 11 シーンを単位にしたブートストラップ(2,000 回)。

    cd ~/jetformer && python experiments/20261015_real_accuracy/ideas_ceiling/ceiling.py
"""
import argparse
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
KV = 0.5e-3 / 1.0          # eRPM → m/s(0.5e-3 m/s/eRPM)


def wrapd(a, L):
    return (a + L / 2) % L - L / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='train', choices=['train', 'all'])
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--win', type=float, default=0.30)
    ap.add_argument('--tol', type=float, default=0.10)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    C = np.load(os.path.join(HERE, 'cache', 'heldout_cache.npz'), allow_pickle=False)
    Lc = np.load(os.path.join(HERE, 'cache', 'lapcoord.npz'), allow_pickle=False)
    z = np.load(rp.rl_dir(ev.DATASET) / 'frames.npz', allow_pickle=False)
    f = {k: z[k] for k in ('scene', 'frame', 'mode', 'future_actions', 'future_valid', 'vehicle_valid', 'near_collision',
                           'erpm', 'gz', 'split', 'log')}
    cl = cm.Centerline()
    L = float(Lc['L'])
    sp, off, h = Lc['sp'], Lc['off'], Lc['h']
    # 向きのずれ(基準線の向きとの差)
    _, hcl = cl.at(np.nan_to_num(sp))
    dh = np.degrees(np.angle(np.exp(1j * (h - hcl))))
    good = np.isfinite(sp) & (np.abs(off) < 0.8) & (np.abs(dh) < 60)
    clean_scenes = set(x[:-4] for x in os.listdir(os.path.join(REPO, 'data/datasets/clean_driving_20261011/aligned')))
    isclean = np.isin(f['scene'], list(clean_scenes))
    spd = -f['erpm'] * KV
    # 保留の行(主指標の 4,670)→ frames.npz の行
    ev_rows = C['frow'][C['clean']]
    y_ev = np.nan_to_num(C['y'][C['clean']].astype(np.float64))
    names = [str(x).split('/')[-1] for x in C['names']]
    P = C['preds'][:, C['clean']].astype(np.float64)
    pa = {n: P[names.index(n)] for n in ('jetformer_v0_s0', 'jetformer_v0_s1', 'B')}
    val_scene = C['scene'][C['clean']]
    assert np.all(f['scene'][ev_rows] == val_scene)
    pool_ok = isclean & good & f['future_valid'] & f['vehicle_valid'] & ~f['near_collision']
    if args.pool == 'train':
        pool_ok &= f['split'] == 'train'
    lap = Lc['lapkey']
    res = {'args': vars(args), 'loc_quality': {
        'clean_rows_good_loc': round(float(good[isclean].mean()), 3),
        'eval_rows_good_loc': round(float(good[ev_rows].mean()), 3),
        'eval_abs_offset_median_m': round(float(np.nanmedian(np.abs(off[ev_rows]))), 3),
        'eval_abs_dheading_median_deg': round(float(np.nanmedian(np.abs(dh[ev_rows]))), 1)}}
    # 状態の標準化(プールの散らばり)
    feats = np.stack([off, np.radians(dh), spd, f['gz'] / 100.0], 1)
    sd = np.nanstd(feats[pool_ok], 0)
    feats = feats / sd
    Y = np.nan_to_num(f['future_actions'].astype(np.float64))
    n = len(ev_rows)
    pred_loc = np.full((n, 10, 2), np.nan)
    pred_st = np.full((n, 10, 2), np.nan)
    pair_st = np.full((n, 10, 2), np.nan)
    pair_loc = np.full((n, 10, 2), np.nan)
    nk_loc = np.zeros(n, int)
    for g in (4, 5):
        pidx = np.nonzero(pool_ok & (f['mode'] == g))[0]
        psp = sp[pidx]
        order = np.argsort(psp)
        pidx, psp = pidx[order], psp[order]
        for j in np.nonzero(f['mode'][ev_rows] == g)[0]:
            r = ev_rows[j]
            if not good[r]:
                continue
            # 窓の中のプール(周をまたぐ折り返しも見る)
            cand = []
            for shift in (-L, 0.0, L):
                a = np.searchsorted(psp, sp[r] + shift - args.win)
                b = np.searchsorted(psp, sp[r] + shift + args.win)
                cand.append(pidx[a:b])
            cand = np.concatenate(cand)
            cand = cand[(f['scene'][cand] != f['scene'][r])]          # 同じシーン(同じ周の前後)は使わない
            if not len(cand):
                continue
            dsp = np.abs(wrapd(sp[cand] - sp[r], L))
            # 場所だけ: 周ごとに最も近い 1 フレーム(< tol)
            lk = lap[cand]
            sel = []
            for u in np.unique(lk):
                m = np.nonzero(lk == u)[0]
                k = m[np.argmin(dsp[m])]
                if dsp[k] < args.tol:
                    sel.append(cand[k])
            if len(sel) >= 3:
                ys = Y[sel]
                pred_loc[j] = np.median(ys, 0)
                pair_loc[j] = np.abs(ys - y_ev[j]).mean(0)
                nk_loc[j] = len(sel)
            # 場所 + 状態: 近い順に、1 周から 1 つまで K 個
            dist = np.sqrt(((feats[cand] - feats[r]) ** 2).sum(1) + (dsp / args.win) ** 2)
            o = np.argsort(dist)
            seen, pick = set(), []
            for k in o:
                if lk[k] in seen:
                    continue
                seen.add(lk[k])
                pick.append(cand[k])
                if len(pick) >= args.K:
                    break
            if len(pick) >= 3:
                ys = Y[pick]
                pred_st[j] = np.median(ys, 0)
                pair_st[j] = np.abs(ys - y_ev[j]).mean(0)
    have = np.isfinite(pred_loc[:, 0, 0]) & np.isfinite(pred_st[:, 0, 0])
    res['eval_rows_used'] = int(have.sum())
    res['eval_rows_total'] = int(n)
    res['laps_per_eval_row_loc_median'] = float(np.median(nk_loc[have]))
    def l1(p, m):
        return float(np.abs(p - y_ev)[m].mean())
    rows = {'model_A_s0': pa['jetformer_v0_s0'], 'model_A_s1': pa['jetformer_v0_s1'], 'model_B': pa['B'],
            'F_loc': pred_loc, 'F_state': pred_st}
    out = {}
    for gname, gm in (('all', have), ('genre4', have & (f['mode'][ev_rows] == 4)), ('genre5', have & (f['mode'][ev_rows] == 5))):
        o = {k: round(l1(v, gm), 2) for k, v in rows.items()}
        o['A_mean'] = round((o['model_A_s0'] + o['model_A_s1']) / 2, 2)
        o['D_state_pair'] = round(float(pair_st[gm].mean()), 2)
        o['D_loc_pair'] = round(float(pair_loc[gm].mean()), 2)
        o['floor_range_from_D_state'] = [round(o['D_state_pair'] / 2, 1), round(o['D_state_pair'], 1)]
        o['floor_gauss_D_over_sqrt2'] = round(o['D_state_pair'] / np.sqrt(2), 1)
        o['floor_laplace_D_over_1.5'] = round(o['D_state_pair'] / 1.5, 1)
        o['n'] = int(gm.sum())
        for ch, cn in ((0, 'steer'), (1, 'throttle')):
            o['by_channel_' + cn] = {k: round(float(np.abs(v - y_ev)[gm][:, :, ch].mean()), 1) for k, v in rows.items()}
            o['by_channel_' + cn]['D_state_pair'] = round(float(pair_st[gm][:, :, ch].mean()), 1)
        o['per_k_total'] = {k: [round(float(x), 1) for x in np.abs(v - y_ev)[gm].mean(axis=(0, 2))] for k, v in
                            (('A_s0', rows['model_A_s0']), ('F_state', pred_st), ('F_loc', pred_loc))}
        o['per_k_total']['D_state_pair'] = [round(float(x), 1) for x in pair_st[gm].mean(axis=(0, 2))]
        out[gname] = o
    res['results'] = out
    # シーン別
    res['by_scene'] = {}
    for s in np.unique(val_scene):
        m = have & (val_scene == s)
        if m.sum() < 30:
            continue
        res['by_scene'][s] = {'n': int(m.sum()), 'A_s0': round(l1(rows['model_A_s0'], m), 1), 'F_state': round(l1(pred_st, m), 1),
                              'F_loc': round(l1(pred_loc, m), 1), 'D_state_pair': round(float(pair_st[m].mean()), 1)}
    # ブートストラップ(シーン単位)
    rng = np.random.default_rng(0)
    sc = np.unique(val_scene[have])
    eA = np.abs((rows['model_A_s0'] + rows['model_A_s1']) / 2 * 0 + rows['model_A_s0'] - y_ev).mean(axis=(1, 2)) * 0.5 + \
        np.abs(rows['model_A_s1'] - y_ev).mean(axis=(1, 2)) * 0.5        # 行ごとの A の 2 本の L1 の平均
    eS = np.abs(pred_st - y_ev).mean(axis=(1, 2))
    eLc = np.abs(pred_loc - y_ev).mean(axis=(1, 2))
    eD = pair_st.mean(axis=(1, 2))
    stats = {'A': [], 'F_state': [], 'F_loc': [], 'D_state_pair': [], 'A_minus_F_state': []}
    for _ in range(args.boot):
        pick = rng.choice(sc, len(sc), replace=True)
        m = np.concatenate([np.nonzero(have & (val_scene == s))[0] for s in pick])
        stats['A'].append(eA[m].mean())
        stats['F_state'].append(eS[m].mean())
        stats['F_loc'].append(eLc[m].mean())
        stats['D_state_pair'].append(eD[m].mean())
        stats['A_minus_F_state'].append(eA[m].mean() - eS[m].mean())
    res['bootstrap_95ci_scene'] = {k: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)] for k, v in stats.items()}
    tag = args.tag or ('pool-%s_K%d' % (args.pool, args.K))
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'out', 'ceiling_%s.json' % tag), 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(HERE, 'out', 'ceiling_rows_%s.npz' % tag), have=have, pred_state=pred_st.astype(np.float32),
                        pred_loc=pred_loc.astype(np.float32), pair_state=pair_st.astype(np.float32), eval_rows=ev_rows)
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
