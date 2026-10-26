#!/usr/bin/env python3
"""原因 A の診断 D1(numpy だけ・読むだけ): B(clean_driving_20261011 の学習 41 シーン)と v2-H の学習のサンプルの
正解のスロットルを、出どころ・区間・速さの帯・止まっているかで分けて比べる。作用の基準(action_reference)も数え直す。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I d1_labels.py

学習のサンプルの選び方は tools/train.py と同じ: future_valid & vehicle_valid & ジャンル != 0 & ~drop & ~衝突の前後(2.5G) & 間引き 3。
書くのは同じ場所の out/d1_labels.json・out/d1_samples.npz だけ。
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
DATA = REPO / 'data'
DS_V2 = DATA / 'datasets' / 'dataset_v2_20261025'
DS_B = DATA / 'datasets' / 'clean_driving_20261011'
TAB = DATA / 'derived' / 'dataset_v2_20261025'
LABELS = DATA / 'labels'
OUT = Path(__file__).resolve().parent / 'out'

GRAVITY = 9.80665
IMPACT_G, BEFORE_S, AFTER_S = 2.5, 1.0, 2.0
STRIDE = 3
SPEED_PER_ERPM = 0.5e-3
BRAKE_US = 1598.0                     # 門と同じ(中立 1498 + 100)
NEUTRAL_B = 1496.0
STRAIGHT_ERPM, STRAIGHT_GZ_RAD = -1000.0, np.radians(40.0)
LOC_DETAIL = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
BANDS = [(-9, 0.2, 'stop<0.2'), (0.2, 0.3, '0.2-0.3'), (0.3, 1.0, '0.3-1.0'), (1.0, 2.0, '1-2'), (2.0, 3.0, '2-3'),
         (3.0, 4.0, '3-4'), (4.0, 99, '>=4')]


def impact_mask(t_us, imu):
    t = np.asarray(t_us, np.float64) / 1e6
    acc = np.asarray(imu, np.float32)[:, :, :2]
    hg = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / GRAVITY
    m = np.zeros(len(t), bool)
    for ti in t[hg >= IMPACT_G]:
        m |= (t >= ti - BEFORE_S) & (t <= ti + AFTER_S)
    return m


def session_of(log):
    d, hm = log[:8], int(log[9:13])
    if d == '20261010':
        return '10/10 昼' if hm < 1900 else '10/10 夜'
    if d == '20261023':
        return '10/23'
    if d == '20261024':
        return '10/24-2'
    return d


def label_rows(log, cache={}):
    if log not in cache:
        p = LABELS / (log + '.npz')
        if not p.exists():
            cache[log] = None
        else:
            with np.load(p) as z:
                cache[log] = {k: z[k] for k in ('t_us', 't_s', 'ai', 'loc_detail', 'takeover_class', 'erpm')}
    return cache[log]


def move_median(x, half):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(x[max(0, i - half):i + half + 1])
    return out


def main():
    splits = json.loads((DS_V2 / 'splits.json').read_text())
    b_train = set(json.loads((DS_B / 'splits.json').read_text())['train'])
    with np.load(TAB / 'drop_v2h.npz') as z:
        dscene, dframe, ddrop, dreason = z['scene'].astype(str), z['frame'], z['drop'], z['reason']
    drop_tab = {}
    for s in np.unique(dscene):
        m = dscene == s
        drop_tab[s] = (dframe[m], ddrop[m], dreason[m])
    rows = {k: [] for k in ('scene', 'frame', 'group', 'sess', 'sub', 'thr0', 'thr2', 'steer0', 'v', 'gz', 'loc',
                            'straight', 'cur_thr', 'since_ai_s')}
    ref_rows = {'b41_all': [], 'v2_all_train_scenes': [], 'v2h_kept_samples': [], 'b41_kept_samples': [],
                'ai_logs_all': [], 'ai_logs_1023_1024': []}
    for s in splits['train']:
        a = np.load(DS_V2 / 'aligned' / (s + '.npz'))
        veh, vv = a['vehicle'], a['vehicle_valid']
        ev = a['erpm_valid'] if 'erpm_valid' in a.files else np.ones(len(vv), bool)
        stand = vv & ev & (np.abs(veh[:, 3]) < 1)
        log = s[:15]
        ref_rows['v2_all_train_scenes'].append(veh[stand, :2])
        if s in b_train:
            ref_rows['b41_all'].append(veh[stand, :2])
        else:
            ref_rows['ai_logs_all'].append(veh[stand, :2])
            if log.startswith('2026102'):
                ref_rows['ai_logs_1023_1024'].append(veh[stand, :2])
        n = len(vv)
        ok = a['future_valid'] & vv & (a['mode'] != 0)
        fr, dd, rr = drop_tab[s]
        drop = np.zeros(n, bool)
        reason = np.full(n, -1, np.int16)
        drop[fr] = dd
        reason[fr] = rr
        ok &= ~drop
        ok &= ~impact_mask(a['timestamps_us'], a['imu'])
        ok &= (np.arange(n) % STRIDE) == 0
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        ref_rows['v2h_kept_samples'].append(veh[idx][stand[idx], :2])
        if s in b_train:
            ref_rows['b41_kept_samples'].append(veh[idx][stand[idx], :2])
        fa = a['future_actions']
        erpm = veh[:, 3].astype(np.float64)
        gz = a['imu'][:, :, 5].mean(axis=1).astype(np.float64)       # rad/s(直前 10 サンプルの平均)
        straight = (erpm < STRAIGHT_ERPM) & (np.abs(gz) < STRAIGHT_GZ_RAD)
        t_us = a['timestamps_us']
        L = label_rows(log)
        loc = np.zeros(n, np.int8)
        since = np.full(n, np.nan)
        if L is not None:
            k = np.clip(np.searchsorted(L['t_us'], t_us), 1, len(L['t_us']) - 1)
            k = np.where(np.abs(L['t_us'][k - 1] - t_us) < np.abs(L['t_us'][k] - t_us), k - 1, k)
            close = np.abs(L['t_us'][k] - t_us) < 100000
            loc = np.where(close, L['loc_detail'][k], 0).astype(np.int8)
            tl = L['t_s']
            last_ai = np.maximum.accumulate(np.where(L['ai'], tl, -1e9))
            sl = tl - last_ai
            since = np.where(close, sl[k], np.nan)
        if s in b_train:
            group, sub = 'B41(10/10 人)', 'race'
        else:
            group = '新しい人 ' + session_of(log)
            fwd_med = move_median(-erpm, 30)
            sub = None
        for i in idx:
            rows['scene'].append(s)
            rows['frame'].append(int(i))
            rows['group'].append(group)
            rows['sess'].append(session_of(log))
            if s in b_train:
                rows['sub'].append('race')
            else:
                tk = np.isfinite(since[i]) and since[i] < 3.0 and since[i] >= 0
                rows['sub'].append('交代の直後3秒' if tk and not fwd_med[i] >= 3000 else
                                   ('走り+交代の直後' if tk else ('走りの区間' if fwd_med[i] >= 3000 else 'ほか')))
            rows['thr0'].append(float(fa[i, 0, 1]))
            rows['thr2'].append(float(fa[i, 2, 1]))
            rows['steer0'].append(float(fa[i, 0, 0]))
            rows['v'].append(float(-erpm[i] * SPEED_PER_ERPM))
            rows['gz'].append(float(gz[i]))
            rows['loc'].append(int(loc[i]))
            rows['straight'].append(bool(straight[i]))
            rows['cur_thr'].append(float(veh[i, 1]))
            rows['since_ai_s'].append(float(since[i]) if np.isfinite(since[i]) else np.nan)
    R = {k: np.asarray(v) for k, v in rows.items()}
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'd1_samples.npz', **R)
    out = {'n_samples': int(len(R['thr0'])), 'by_group': {}, 'by_sub': {}, 'action_reference': {}, 'bins': [],
           'straight': {}, 'loc_moving': {}}

    def summ(m):
        m = np.asarray(m, bool)
        if not m.any():
            return None
        v, t0, t2 = R['v'][m], R['thr0'][m], R['thr2'][m]
        low = v < 1.0
        mv = v >= 1.0
        st = R['straight'][m] & mv
        res = {'n': int(m.sum()), 'stop<0.2': float((v < 0.2).mean()), '<0.3': float((v < 0.3).mean()),
               '0.3-1.0': float(((v >= 0.3) & (v < 1.0)).mean()), 'v_median': float(np.median(v)),
               'fwd_row2_moving_us': float(np.mean(NEUTRAL_B - t2[mv])) if mv.any() else None,
               'fwd_row2_straight_us': float(np.mean(NEUTRAL_B - t2[st])) if st.any() else None,
               'n_straight': int(st.sum()),
               'brake_frac_row0_v<1': float((t0[low] > BRAKE_US).mean()) if low.any() else None,
               'brake_frac_row0_v<0.3': float((t0[v < 0.3] > BRAKE_US).mean()) if (v < 0.3).any() else None,
               'brake_frac_row0_v0.3-1': float((t0[(v >= 0.3) & low] > BRAKE_US).mean()) if ((v >= 0.3) & low).any() else None,
               'n_v<1': int(low.sum()),
               'brake_frac_row0_moving': float((t0[mv] > BRAKE_US).mean()) if mv.any() else None,
               'neutral_frac_row2_moving(|thr-1496|<=30)': float((np.abs(t2[mv] - NEUTRAL_B) <= 30).mean()) if mv.any() else None}
        return res

    for g in sorted(set(R['group'].tolist())):
        out['by_group'][g] = summ(R['group'] == g)
    new = R['group'] != 'B41(10/10 人)'
    out['by_group']['新しい人 全部'] = summ(new)
    out['by_group']['v2-H 全部'] = summ(np.ones(len(new), bool))
    for g in sorted(set(R['sub'][new].tolist())):
        out['by_sub'][g] = summ(new & (R['sub'] == g))
        for ss in sorted(set(R['sess'][new].tolist())):
            out['by_sub']['%s / %s' % (g, ss)] = summ(new & (R['sub'] == g) & (R['sess'] == ss))
    # 作用の基準
    for k, lst in ref_rows.items():
        x = np.concatenate(lst) if lst else np.zeros((0, 2))
        out['action_reference'][k] = {'rows': int(len(x)), 'steer_center_us': float(np.median(x[:, 0])) if len(x) else None,
                                      'throttle_neutral_us': float(np.median(x[:, 1])) if len(x) else None}
    # 区間 × 速さの帯(動いている)で、B41 と v2-H の正解の前進の量(row 2)と、混ぜたときの平均の動き
    for li, ln in enumerate(LOC_DETAIL):
        for lo, hi, bn in BANDS:
            m = (R['loc'] == li) & (R['v'] >= lo) & (R['v'] < hi)
            mb, mn = m & ~new, m & new
            if mb.sum() + mn.sum() < 20:
                continue
            fb = float(np.mean(NEUTRAL_B - R['thr2'][mb])) if mb.any() else None
            fn = float(np.mean(NEUTRAL_B - R['thr2'][mn])) if mn.any() else None
            fall = float(np.mean(NEUTRAL_B - R['thr2'][m]))
            out['bins'].append({'loc': ln, 'band': bn, 'n_B41': int(mb.sum()), 'n_new': int(mn.sum()),
                                'new_share': float(mn.sum() / m.sum()), 'fwd_B41_us': fb, 'fwd_new_us': fn,
                                'fwd_v2h_us': fall, 'shift_us(v2h−B41、+ = 緩める)': (fb - fall) if fb is not None else None})
    for lo, hi, bn in BANDS[3:]:
        m = R['straight'] & (R['v'] >= lo) & (R['v'] < hi)
        mb, mn = m & ~new, m & new
        out['straight'][bn] = {'n_B41': int(mb.sum()), 'n_new': int(mn.sum()),
                               'fwd_B41_us': float(np.mean(NEUTRAL_B - R['thr2'][mb])) if mb.any() else None,
                               'fwd_new_us': float(np.mean(NEUTRAL_B - R['thr2'][mn])) if mn.any() else None,
                               'new_share': float(mn.sum() / max(1, m.sum()))}
    (OUT / 'd1_labels.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({'n': out['n_samples'], 'action_reference': out['action_reference']}, ensure_ascii=False, indent=1))
    for g, r in list(out['by_group'].items()) + list(out['by_sub'].items()):
        if r:
            print('%-34s n=%5d stop<0.2=%.3f <0.3=%.3f 0.3-1=%.3f fwd_mv=%6.1f fwd_st=%6s brake<1=%s brake_mv=%.3f neutral_mv=%.3f' % (
                g, r['n'], r['stop<0.2'], r['<0.3'], r['0.3-1.0'], r['fwd_row2_moving_us'] or np.nan,
                '%.1f' % r['fwd_row2_straight_us'] if r['fwd_row2_straight_us'] is not None else '-',
                '%.3f' % r['brake_frac_row0_v<1'] if r['brake_frac_row0_v<1'] is not None else '-',
                r['brake_frac_row0_moving'] or 0, r['neutral_frac_row2_moving(|thr-1496|<=30)'] or 0))


if __name__ == '__main__':
    sys.exit(main())
