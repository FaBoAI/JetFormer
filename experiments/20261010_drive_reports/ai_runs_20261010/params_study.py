#!/usr/bin/env python3
"""AI走行のパラメータ比較(2026-10-10)。numpyだけで動く軽い集計。

  python params_study.py                 # 本日の全AI走行 + 人の走行を集計し params.json と fig_*.png をこのフォルダへ書く
  python params_study.py LOG [LOG ...]   # 新しいAI走行ログの指標だけを表示(チューニング手順の判定用、ファイルは書かない)

指標の定義は params.md の「指標の定義」を参照。
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
RAW = '/home/jetson/jetformer/data/raw'
HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.80665
CENTER, NEUTRAL = 1443.0, 1498.0
GAP = 1.0             # s: これより長いサンプル間隔 = データ欠損(シーン欠落など)。0.52 s のI2C停止は連続とみなす
COLL_G = 2.5          # 衝突しきい値 [g](水平 |ax| または |ay|)
EVENT_WIN = 0.6       # s: 同じ衝突とみなす範囲(最初のサンプルから)
STUCK_ERPM, STUCK_MIN_S, STUCK_WIN = 300, 0.5, 1.0
REV_HYST = 50.0       # us: 左右切り返しのヒステリシス(中心 ±50us を越えて反対側へ行ったら1回)
SECT = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]

AI_LOGS = [('20261010-155820', 'trial', 'I2C修正前(0.5秒停止あり)'),
           ('20261010-160306', 'run_001', 'I2C修正前(0.5秒停止あり)'),
           ('20261010-163247', 'run002', '電源断でシーン8-20欠落・21-23破損'),
           ('20261010-171639', 'ai_run_003', 'シーン間に20秒の欠落'),
           ('20261010-172505', 'ai_run_004', '20秒のみ'),
           ('20261010-172910', 'ai_run_006', '走行中にゲインを22回変更'),
           ('20261010-173445', 'ai_run_007', '走行中にゲインを14回変更')]
HUMAN_LOGS = ['20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707']
REF_LOG = '20261010-172910'   # 周回位置(方位の位相)の基準: 145秒連続のAI区間を持つ ai_run_006


# ------------------------------------------------------------------ 読み込み
def load(log):
    root = os.path.join(RAW, log)
    meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
    want = {'VEHICLE': ['timestamps_us', 'pwm', 'valid', 'select', 'erpm'],
            'IMU': ['timestamps_us', 'accel', 'gyro'],
            'AI_CMD': ['timestamps_us', 'out', 'raw', 'pred', 'hold', 'select', 'gains', 'lookahead', 'latency_ms']}
    parts = {k: {f: [] for f in v} for k, v in want.items()}
    skipped, used = [], []
    for s in sorted(x for x in os.listdir(root) if x.startswith(log + '-')):
        got = {}
        try:
            for name, keys in want.items():
                p = os.path.join(root, s, name + '.npz')
                if not os.path.exists(p):
                    if name == 'AI_CMD':
                        continue
                    raise IOError('missing ' + name)
                z = np.load(p, allow_pickle=True)
                got[name] = {k: z[k] for k in keys}
        except Exception as e:                     # 電源断で壊れたnpzなどは飛ばす
            skipped.append([s, repr(e)[:60]])
            continue
        used.append(s)
        for name, d in got.items():
            for k, v in d.items():
                parts[name][k].append(v)
    D = {name: {k: np.concatenate(v) for k, v in d.items()} for name, d in parts.items() if d['timestamps_us']}
    t0 = D['VEHICLE']['timestamps_us'][0]
    for name in D:
        D[name]['t'] = (D[name]['timestamps_us'] - t0) / 1e6
    D['meta'], D['skipped'], D['used'] = meta, skipped, used
    return D


def runs_of(mask):
    m = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.nonzero(m == 1)[0], np.nonzero(m == -1)[0]))


def split_gaps(t, a, b):
    cut = [a] + [i + 1 for i in range(a, b - 1) if t[i + 1] - t[i] > GAP] + [b]
    return list(zip(cut[:-1], cut[1:]))


def sample_dt(t):
    dt = np.diff(t, append=t[-1] + 0.01)
    return np.where(dt > GAP, 0.01, dt)


def lookup(ts, arr, t):
    return arr[np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)]


def poisson_ci(k, conf=0.95):
    """Garwood の正確な信頼区間(件数)"""
    a = (1 - conf) / 2

    def cdf(n, mu):
        return sum(math.exp(-mu + i * math.log(mu) - math.lgamma(i + 1)) for i in range(n + 1)) if mu > 0 else 1.0

    def solve(f, lo, hi):
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda mu: 1 - cdf(k - 1, mu) < a, 0.0, k + 10)
    upper = solve(lambda mu: cdf(k, mu) > a, k, k * 3 + 20)
    return lower, upper


# ------------------------------------------------------------------ 衝突・方位
def imu_events(D):
    im = D['IMU']; a = im['accel']; t = im['t']
    h = np.max(np.abs(a[:, :2]), axis=1) / G          # 水平
    v = np.abs(a[:, 2] - G) / G                        # 上下(1gを引く)
    gn = np.abs(np.linalg.norm(a, axis=1) - G) / G
    ev = []
    for i in np.nonzero((h >= COLL_G) | (v >= COLL_G))[0]:
        if ev and t[i] - ev[-1]['t'] <= EVENT_WIN:
            e = ev[-1]
        else:
            e = {'t': float(t[i]), 'h': 0.0, 'v': 0.0, 'ax': 0.0, 'ay': 0.0, 'idx': []}
            ev.append(e)
        e['idx'].append(i)
        if h[i] > e['h']:
            e['h'], e['ax'], e['ay'] = float(h[i]), float(a[i, 0] / G), float(a[i, 1] / G)
        e['v'] = max(e['v'], float(v[i]))
    for e in ev:
        sus = [((np.abs(t - t[i]) <= 0.05) & (gn > 1.0)).sum() <= 1 for i in e['idx']]   # 1サンプルだけの跳ね = センサーの誤り
        real_h = any(h[i] >= COLL_G and not s for i, s in zip(e['idx'], sus))
        e['suspect'] = bool(all(sus))
        e['kind'] = 'collision' if real_h else ('suspect' if e['suspect'] else 'vertical')
        e['contact'] = contact_side(e['ax'], e['ay'])
        del e['idx']
    return ev


def contact_side(ax_g, ay_g):
    """当たった方向。実測ではIMUの +y = 前方、+x = 右(旋回の遠心・加減速との相関で確認, params.md 参照)。
    感じた加速度の逆向きが接触点: 前から当たると -y(減速)、左側面に当たると右へ押されて +x"""
    a_fwd, a_left = ay_g, -ax_g
    ang = math.degrees(math.atan2(-a_left, -a_fwd))     # 0 = 前, +90 = 左, -90 = 右
    if abs(ang) <= 30:
        return 'front'
    if 30 < ang <= 75:
        return 'front-left'
    if -75 <= ang < -30:
        return 'front-right'
    if 75 < ang <= 105:
        return 'left'
    if -105 <= ang < -75:
        return 'right'
    return 'rear'


def imu_axes_check(D):
    """IMUの軸の向きを走行データから確認(加減速・旋回の遠心力との相関)"""
    im, v = D['IMU'], D['VEHICLE']
    k = np.ones(25) / 25
    sp = np.convolve(np.interp(im['t'], v['t'], -v['erpm'].astype(float)), k, 'same')
    ax = np.convolve(im['accel'][:, 0], k, 'same'); ay = np.convolve(im['accel'][:, 1], k, 'same')
    gz = np.convolve(im['gyro'][:, 2], k, 'same'); dsp = np.gradient(sp, im['t'])
    m = (np.abs(sp) > 500) & (np.abs(gz) < 30); m2 = sp > 2500
    return {'corr_ax_dspeed': float(np.corrcoef(ax[m], dsp[m])[0, 1]), 'corr_ay_dspeed': float(np.corrcoef(ay[m], dsp[m])[0, 1]),
            'corr_ax_yaw_x_speed': float(np.corrcoef(ax[m2], (gz * sp)[m2])[0, 1]),
            'corr_ay_yaw_x_speed': float(np.corrcoef(ay[m2], (gz * sp)[m2])[0, 1])}


def gyro_bias(D):
    v, im = D['VEHICLE'], D['IMU']
    er = lookup(v['t'], np.abs(v['erpm']), im['t'])
    gn = np.abs(np.linalg.norm(im['accel'], axis=1) - G) / G
    m = (er < 50) & (gn < 0.1) & (np.abs(im['gyro'][:, 2]) < 20)
    return float(np.median(im['gyro'][m, 2])) if m.sum() > 100 else 0.0


def yaw(D):
    return np.clip(D['IMU']['gyro'][:, 2], -990, 990) - gyro_bias(D)


def heading_laps(D, mask_imu):
    """マスク区間の方位を積分。周回 = 正味360度。新しい周回に入った瞬間(行き戻りを除く)の間隔をラップタイムに"""
    im = D['IMU']; t = im['t']; gz = yaw(D)
    # 台形積分(I2C修正前の 0.5-0.9 秒の欠損は前後の値で直線補間したことになる)
    inc = np.r_[0.5 * (gz[1:] + gz[:-1]) * np.diff(t), 0.0]
    inc[np.r_[np.diff(t) > GAP, True]] = 0.0
    net = float(np.sum(inc[mask_imu]))
    laps = []
    for a, b in runs_of(mask_imu):
        for a2, b2 in split_gaps(t, a, b):
            if b2 - a2 < 100:
                continue
            h = np.cumsum(inc[a2:b2])
            sgn = 1.0 if h[-1] >= 0 else -1.0
            k = np.maximum.accumulate(np.floor(h * sgn / 360.0))
            ct = t[a2 + 1 + np.nonzero(np.diff(k) > 0)[0]]
            laps += [float(c1 - c0) for c0, c1 in zip(ct[:-1], ct[1:])]
    return net, laps


def signature(hd, gz, nb=36):
    b = (np.mod(hd, 360) / (360 / nb)).astype(int) % nb
    return np.array([np.median(gz[b == k]) if (b == k).sum() > 5 else np.nan for k in range(nb)])


def phase_of(D, mask_imu, ref, min_len=12.0):
    """各IMUサンプルの周回位置(0-360度)。区間ごとに方位の位相を基準走行のヨーレート分布に合わせる(磁気センサーは無効)"""
    im = D['IMU']; t = im['t']; dt = sample_dt(t); gz = yaw(D)
    ph = np.full(len(t), np.nan); cor = np.full(len(t), np.nan)
    for a, b in runs_of(mask_imu):
        for a2, b2 in split_gaps(t, a, b):
            if t[b2 - 1] - t[a2] < min_len:
                continue
            hd = np.cumsum(gz[a2:b2] * dt[a2:b2])
            if abs(hd[-1]) < 400:
                continue
            s = signature(hd, gz[a2:b2]); best = (0, -1.0)
            for k in range(len(ref)):
                r = np.roll(s, k); m = np.isfinite(r) & np.isfinite(ref)
                c = np.corrcoef(r[m], ref[m])[0, 1] if m.sum() > 20 else -1.0
                if c > best[1]:
                    best = (k, c)
            ph[a2:b2] = np.mod(hd + best[0] * 360 / len(ref), 360); cor[a2:b2] = best[1]
    ph[~(cor >= 0.5)] = np.nan
    return ph


def reference_signature():
    D = load(REF_LOG); im = D['IMU']; v = D['VEHICLE']
    sel = lookup(v['t'], v['select'].astype(bool), im['t']).astype(bool)
    a, b = max(runs_of(sel), key=lambda r: r[1] - r[0])
    gz = yaw(D); hd = np.cumsum(gz[a:b] * sample_dt(im['t'])[a:b])
    return signature(hd, gz[a:b])


# ------------------------------------------------------------------ 指標
def reversals(x, thr=REV_HYST):
    s = n = 0
    for d in x - CENTER:
        if d > thr:
            n += s == -1; s = 1
        elif d < -thr:
            n += s == 1; s = -1
    return n


def throttle_stats(x):
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {'p5': p[0], 'p25': p[1], 'p50': p[2], 'p75': p[3], 'p95': p[4],
            'fwd_ratio': float(np.mean(x < NEUTRAL - 15)), 'neutral_ratio': float(np.mean(np.abs(x - NEUTRAL) <= 15)),
            'brake_rev_ratio': float(np.mean(x > NEUTRAL + 15)), 'strong_rev_ratio': float(np.mean(x >= 1900)),
            'fwd_us_p50': float(np.median(NEUTRAL - x)), 'fwd_us_p90': float(np.percentile(NEUTRAL - x, 90))}


def ai_metrics(D, t0=-1e9, t1=1e9, ref=None):
    """AI側(VEHICLE.select)の指標。t0-t1 で区切るとゲイン区間ごとの集計になる"""
    v, im, ai = D['VEHICLE'], D['IMU'], D['AI_CMD']
    vdt = sample_dt(v['t'])
    win = (v['t'] >= t0) & (v['t'] < t1)
    sel_all = v['select'].astype(bool)
    sel = sel_all & win
    total_s = float(vdt[win].sum()); ai_s = float(vdt[sel].sum()); ai_min = max(ai_s / 60, 1e-9)
    segs = []
    for a, b in runs_of(sel):
        for a2, b2 in split_gaps(v['t'], a, b):
            take = b2 == b and b < len(sel_all) and not sel_all[b] and v['t'][b] - v['t'][b - 1] <= GAP
            segs.append({'t0': float(v['t'][a2]), 'dur': float(vdt[a2:b2].sum()), 'end': 'takeover' if take else 'data_end'})
    takeovers = [s for s in segs if s['end'] == 'takeover']
    ev = [e for e in imu_events(D) if t0 <= e['t'] < t1]
    for e in ev:
        e['ai'] = bool(lookup(v['t'], sel_all, e['t']))
    coll = [e for e in ev if e['kind'] == 'collision' and e['ai']]
    vert = [e for e in ev if e['kind'] == 'vertical' and e['ai']]
    # 衝突後の停止(スタック)
    slow = [(v['t'][a], v['t'][b - 1]) for a, b in runs_of(np.abs(v['erpm']) < STUCK_ERPM)]
    stuck, used = [], set()
    for e in coll:
        for k, (s0, s1) in enumerate(slow):
            if s1 < e['t']:
                continue
            if s0 > e['t'] + STUCK_WIN:
                break
            st0 = max(s0, e['t'])
            if s1 - st0 >= STUCK_MIN_S and lookup(v['t'], sel_all, st0) and k not in used:
                used.add(k)
                to = any(st0 - 0.1 <= s['t0'] + s['dur'] <= s1 + 3.0 for s in takeovers)
                stuck.append({'t': e['t'], 'stuck_s': float(s1 - st0), 'takeover': bool(to)})
                break
    to_after = sum(1 for s in takeovers if any(0 <= s['t0'] + s['dur'] - e['t'] <= 2.5 for e in coll))
    er = np.abs(v['erpm'][sel]).astype(float)
    # AIの出力(AI側・中立保持以外)
    am = ai['select'].astype(bool) & ~ai['hold'].astype(bool) & (ai['t'] >= t0) & (ai['t'] < t1)
    asel = ai['select'].astype(bool) & (ai['t'] >= t0) & (ai['t'] < t1)
    o_s, o_t, r_s, r_t = ai['out'][am, 0], ai['out'][am, 1], ai['raw'][am, 0], ai['raw'][am, 1]
    rev_o = rev_r = 0; rtime = 0.0
    for a, b in runs_of(am):
        for a2, b2 in split_gaps(ai['t'], a, b):
            if b2 - a2 >= 5:
                rev_o += reversals(ai['out'][a2:b2, 0]); rev_r += reversals(ai['raw'][a2:b2, 0])
                rtime += float(ai['t'][b2 - 1] - ai['t'][a2])
    sel_imu = lookup(v['t'], sel_all, im['t']).astype(bool) & (im['t'] >= t0) & (im['t'] < t1)
    net, laps = heading_laps(D, sel_imu)
    nlap = abs(net) / 360.0
    gains = ai['gains'][am]
    c_lo, c_hi = poisson_ci(len(coll))
    r = {
        'gain_steer_mean': float(np.mean(gains[:, 0])), 'gain_throttle_mean': float(np.mean(gains[:, 1])),
        'gain_settings_n': len({(round(float(a), 2), round(float(b), 2)) for a, b in gains}),
        'lookahead': sorted({int(x) for x in ai['lookahead'][am]}),
        'total_s': total_s, 'ai_s': ai_s, 'ai_ratio': ai_s / max(total_s, 1e-9),
        'ai_segments': len(segs), 'ai_seg_median_s': float(np.median([s['dur'] for s in segs])) if segs else 0.0,
        'ai_seg_longest_s': float(max([s['dur'] for s in segs])) if segs else 0.0,
        'takeovers': len(takeovers), 'takeovers_per_ai_min': len(takeovers) / ai_min,
        'takeovers_within_2_5s_after_collision': to_after,
        'collisions': len(coll), 'collisions_per_ai_min': len(coll) / ai_min,
        'collisions_per_ai_min_ci95': [c_lo / ai_min, c_hi / ai_min],
        'collisions_by_contact': {k: sum(1 for e in coll if e['contact'] == k) for k in ('front', 'front-right', 'front-left', 'right', 'left', 'rear')},
        'vertical_bumps': len(vert), 'vertical_bumps_per_ai_min': len(vert) / ai_min,
        'suspect_events': sum(1 for e in ev if e['kind'] == 'suspect'),
        'stuck': len(stuck), 'stuck_per_ai_min': len(stuck) / ai_min, 'stuck_with_takeover': sum(s['takeover'] for s in stuck),
        'erpm_abs_p50': float(np.median(er)) if len(er) else 0.0, 'erpm_abs_p90': float(np.percentile(er, 90)) if len(er) else 0.0,
        'laps': nlap, 'direction': 'CCW' if net > 0 else 'CW',
        'collisions_per_lap': len(coll) / nlap if nlap > 0.25 else None,
        'lap_time_ai_s_per_lap': ai_s / nlap if nlap > 0.25 else None,
        'full_laps': len(laps), 'lap_time_median_s': float(np.median(laps)) if laps else None,
        'erev_per_lap': float(np.sum(er * vdt[sel]) / 60.0) / nlap if nlap > 0.25 else None,
        'hold_ratio': float(np.mean(ai['hold'][asel])) if asel.any() else 0.0,
        'steer_reversals_per_s_out': rev_o / max(rtime, 1e-9), 'steer_reversals_per_s_raw': rev_r / max(rtime, 1e-9),
        'steer_at_1000_right': float(np.mean(o_s <= 1001)), 'steer_at_2000_left': float(np.mean(o_s >= 1999)),
        'steer_out_absdev_p50': float(np.median(np.abs(o_s - CENTER))), 'steer_raw_absdev_p50': float(np.median(np.abs(r_s - CENTER))),
        'throttle_out': throttle_stats(o_t), 'throttle_raw': throttle_stats(r_t),
        'latency_ms_p50': float(np.nanmedian(ai['latency_ms'][am])),
        'imu_gaps_over_0_1s': int(np.sum(np.diff(im['t'][(im['t'] >= t0) & (im['t'] < t1)]) > 0.1)),
        'vehicle_gaps_over_0_1s': int(np.sum(np.diff(v['t'][win]) > 0.1)),
    }
    if ref is not None:
        ph = phase_of(D, lookup(v['t'], sel_all, im['t']).astype(bool), ref)
        hot = sum(1 for e in coll if 0 <= ph[min(np.searchsorted(im['t'], e['t']), len(ph) - 1)] < 120)
        r['collisions_in_hot_sector_0_120'] = hot
    return r, coll, stuck


def gain_blocks(D, min_ai_s=7.0):
    ai = D['AI_CMD']; g = np.round(ai['gains'], 2); t = ai['t']
    ch = np.r_[0, np.nonzero(np.any(np.diff(g, axis=0) != 0, axis=1))[0] + 1, len(g)]
    out = []
    for a, b in zip(ch[:-1], ch[1:]):
        t0 = float(t[a]); t1 = float(t[b]) if b < len(t) else float(t[-1] + 0.05)
        m, _, _ = ai_metrics(D, t0, t1)
        if m['ai_s'] >= min_ai_s:
            m.update({'gs': float(g[a, 0]), 'gt': float(g[a, 1]), 'la': int(ai['lookahead'][a]), 't0': t0, 't1': t1})
            out.append(m)
    return out


def human_metrics(D, ref):
    v, im = D['VEHICLE'], D['IMU']
    vdt = sample_dt(v['t']); total = float(vdt.sum())
    ev = imu_events(D)
    coll = [e for e in ev if e['kind'] == 'collision']
    allm = np.ones(len(im['t']), bool)
    net, laps = heading_laps(D, allm)
    ok = v['valid'][:, 0] & v['valid'][:, 1]
    st = v['pwm'][:, 0].astype(float); th = v['pwm'][:, 1].astype(float)
    rev = 0; rt = 0.0
    for a, b in split_gaps(v['t'], 0, len(v['t'])):
        rev += reversals(st[a:b]); rt += float(v['t'][b - 1] - v['t'][a])
    er = np.abs(v['erpm']).astype(float); mv = er > STUCK_ERPM
    nlap = abs(net) / 360
    ph = phase_of(D, allm, ref)
    return {'total_s': total, 'collisions': len(coll), 'collisions_per_min': len(coll) / (total / 60),
            'collisions_by_contact': {k: sum(1 for e in coll if e['contact'] == k) for k in ('front', 'front-right', 'front-left', 'right', 'left', 'rear')},
            'vertical_bumps': sum(1 for e in ev if e['kind'] == 'vertical'),
            'erpm_abs_p50': float(np.median(er)), 'erpm_abs_p90': float(np.percentile(er, 90)),
            'erpm_abs_p50_moving': float(np.median(er[mv])), 'laps': nlap, 'direction': 'CCW' if net > 0 else 'CW',
            'collisions_per_lap': len(coll) / nlap if nlap > 0.25 else None,
            'lap_time_s_per_lap': total / nlap if nlap > 0.25 else None,
            'full_laps': len(laps), 'lap_time_median_s': float(np.median(laps)) if laps else None,
            'erev_per_lap': float(np.sum(er * vdt) / 60.0) / nlap if nlap > 0.25 else None,
            'steer_reversals_per_s': rev / rt, 'steer_absdev_p50': float(np.median(np.abs(st[ok & mv] - CENTER))),
            'steer_p5_p95': [float(np.percentile(st[ok], 5)), float(np.percentile(st[ok], 95))],
            'steer_near_right_lock_le1010': float(np.mean(st[ok] <= 1010)), 'steer_near_left_max_ge1880': float(np.mean(st[ok] >= 1880)),
            'throttle': throttle_stats(th[ok]),
            'collisions_in_hot_sector_0_120': sum(1 for e in coll if 0 <= ph[min(np.searchsorted(im['t'], e['t']), len(ph) - 1)] < 120)}, coll, ph


# ------------------------------------------------------------------ 衝突の分類・周回位置
def classify(D, e, gz):
    ai = D.get('AI_CMD'); v = D['VEHICLE']; im = D['IMU']
    w = (v['t'] >= e['t'] - 0.5) & (v['t'] < e['t'] - 0.1)
    if ai is not None and lookup(v['t'], v['select'].astype(bool), e['t']):
        wa = (ai['t'] >= e['t'] - 0.5) & (ai['t'] < e['t'] - 0.1)
        steer = float(np.mean(ai['out'][wa, 0])) if wa.any() else float('nan')
        raw = float(np.mean(ai['raw'][wa, 0])) if wa.any() else float('nan')
    else:
        steer = raw = float(np.mean(v['pwm'][w, 0]))
    wi = (im['t'] >= e['t'] - 0.5) & (im['t'] < e['t'] - 0.1)
    yb = float(np.mean(gz[wi])); eb = float(np.median(np.abs(v['erpm'][w])))
    wa2 = (v['t'] >= e['t'] + 0.3) & (v['t'] < e['t'] + 1.0)
    ea = float(np.median(np.abs(v['erpm'][wa2]))) if wa2.any() else float('nan')
    if steer <= 1120 and yb < -60:
        kind = 'flip_right_lock'       # 左コーナー出口で右いっぱいへ切り返し→前から壁に当たる→止まる
    elif steer >= 1850:
        kind = 'left_lock_apex'        # 左いっぱいのまま曲がり切れず右前が当たる
    elif abs(steer - CENTER) <= 100:
        kind = 'straight_scrape'       # ほぼ直進のまま右前・正面が仕切りに当たる(速度は残る)
    else:
        kind = 'other'
    return {'steer_out': steer, 'steer_raw': raw, 'yaw_before': yb, 'erpm_before': eb, 'erpm_after': ea, 'type': kind}


# ------------------------------------------------------------------ lookahead(予測の先読み)
def lookahead_stats(D):
    ai = D['AI_CMD']; im = D['IMU']; v = D['VEHICLE']
    am = ai['select'].astype(bool) & ~ai['hold'].astype(bool)
    P = ai['pred'][:, :, 0]; t = ai['t']
    runs = [(a, b) for a, b in runs_of(am) if b - a >= 90 and np.all(np.diff(t[a:b]) < 0.2)]
    if not runs:
        return None
    lead, jit = [], []
    for k in range(P.shape[1]):
        best = (0, -2.0)
        for lag in range(16):
            x = np.concatenate([P[a:b - lag, k] for a, b in runs]); y = np.concatenate([P[a + lag:b, 0] for a, b in runs])
            c = np.corrcoef(x, y)[0, 1]
            if c > best[1]:
                best = (lag, c)
        lead.append(best[0])
        jit.append(float(np.median(np.concatenate([np.abs(np.diff(P[a:b, k])) for a, b in runs]))))
    gz = np.interp(t, im['t'], np.clip(im['gyro'][:, 2], -990, 990))
    er = np.interp(t, v['t'], np.abs(v['erpm']).astype(float))
    best = (0, -2.0)
    for lag in range(21):
        x = np.concatenate([ai['out'][a:b - lag, 0][er[a:b - lag] > 2000] for a, b in runs])
        y = np.concatenate([gz[a + lag:b][er[a:b - lag] > 2000] for a, b in runs])
        c = np.corrcoef(x, y)[0, 1]
        if c > best[1]:
            best = (lag, c)
    return {'lead_frames': lead, 'jitter_us_per_frame': jit, 'out_to_yaw_lag_frames': best[0], 'out_to_yaw_r': float(best[1])}


def zero_yaw_steer(st_t, st, D, mask):
    """ヨーレートが0になるステア値(サーボの実際の直進点)"""
    im, v = D['IMU'], D['VEHICLE']
    gz = np.interp(st_t + 0.10, im['t'], np.clip(im['gyro'][:, 2], -990, 990))
    er = np.interp(st_t + 0.10, v['t'], np.abs(v['erpm']).astype(float))
    m = mask & (er > 2500) & (np.abs(st - CENTER) < 200)
    k = gz[m] / (er[m] / 1000.0); x = st[m]
    pts = [(b + 12.5, np.median(k[(x >= b) & (x < b + 25)])) for b in np.arange(CENTER - 200, CENTER + 200, 25)
           if ((x >= b) & (x < b + 25)).sum() > 30]
    if len(pts) < 4:
        return None
    bx, by = np.array(pts).T
    p = np.polyfit(bx - CENTER, by, 1)
    return float(CENTER - p[1] / p[0])


# ------------------------------------------------------------------ 図
def figures(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 10,
                         'axes.edgecolor': '#c9c8c3', 'axes.linewidth': 0.8, 'axes.labelcolor': '#52514e',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e8e7e3',
                         'grid.linewidth': 0.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb', 'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    S1, S2, S3, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
    blocks = res['gain_blocks']

    # 1) スロットルゲインとモデルの補償
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    hum = [h['throttle']['fwd_us_p50'] for h in res['human'].values()]
    for b in blocks:
        ax[0].plot([b['gt'], b['gt']], [b['throttle_raw']['fwd_us_p50'], b['throttle_out']['fwd_us_p50']], color='#d6d5d0', lw=1, zorder=1)
    ax[0].scatter([b['gt'] for b in blocks], [b['throttle_raw']['fwd_us_p50'] for b in blocks], s=50, color=S1,
                  edgecolor='#fcfcfb', linewidth=2, label='モデルの生出力 raw(ゲイン前)', zorder=3)
    ax[0].scatter([b['gt'] for b in blocks], [b['throttle_out']['fwd_us_p50'] for b in blocks], s=50, color=S2,
                  edgecolor='#fcfcfb', linewidth=2, label='実際の出力 out = ゲイン × raw', zorder=3)
    ax[0].axhspan(min(hum), max(hum), color='#e8e7e3', alpha=0.6, lw=0, zorder=0)
    ax[0].text(2.02, (min(hum) + max(hum)) / 2, '人の走行\n(5ログ)', va='center', fontsize=8, color='#52514e')
    ax[0].set_xlabel('throttle_gain'); ax[0].set_ylabel('前進スロットル 中央値 [us](1498 − 値)')
    ax[0].set_title('ゲインを上げるとモデルが生出力を下げる', fontsize=11, loc='left')
    ax[0].legend(loc='upper right', fontsize=8); ax[0].set_ylim(0, 300); ax[0].set_xlim(0.7, 2.25)
    ax[1].scatter([b['gt'] for b in blocks], [b['erpm_abs_p90'] for b in blocks], s=50, color=S3, edgecolor='#fcfcfb', linewidth=2, label='|eRPM| p90', zorder=3)
    ax[1].scatter([b['gt'] for b in blocks], [b['erpm_abs_p50'] for b in blocks], s=50, color=S1, edgecolor='#fcfcfb', linewidth=2, label='|eRPM| 中央値', zorder=3)
    hp = [h['erpm_abs_p50_moving'] for h in res['human'].values()]
    ax[1].axhspan(min(hp), max(hp), color='#e8e7e3', alpha=0.6, lw=0, zorder=0)
    ax[1].text(2.02, (min(hp) + max(hp)) / 2, '人の\n中央値', va='center', fontsize=8, color='#52514e')
    ax[1].set_xlabel('throttle_gain'); ax[1].set_ylabel('|eRPM|(AI側)'); ax[1].set_ylim(0, 8000); ax[1].set_xlim(0.7, 2.25)
    ax[1].set_title('速度はほとんど変わらない', fontsize=11, loc='left'); ax[1].legend(loc='lower right', fontsize=8)
    fig.text(0.01, 0.01, '点 = ゲイン一定のAI区間(7秒以上, 15区間)。ai_run_006/007 は同じ走行の中でゲインだけ変えている。', fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(HERE, 'fig_throttle_gain.png'), dpi=110); plt.close(fig)

    # 2) 周回位置ごとの衝突率
    sec = res['sectors']; ref = np.array(res['ref_signature'])
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), gridspec_kw={'height_ratios': [1, 1.4]}, sharex=True)
    xs = (np.arange(len(ref)) + 0.5) * 360 / len(ref)
    ax[0].plot(xs, ref, color=S1, lw=2); ax[0].axhline(0, color='#c9c8c3', lw=0.8)
    ax[0].set_ylabel('ヨーレート [deg/s]\n(+ = 左旋回)'); ax[0].set_title('周回位置(方位の位相)とコース形状: 0-50° が最もきつい左コーナー', fontsize=11, loc='left')
    for i, (lo, hi) in enumerate(SECT):
        ai_r = sec['ai'][i][1] / (sec['ai'][i][0] / 60) if sec['ai'][i][0] else 0
        hu_r = sec['human'][i][1] / (sec['human'][i][0] / 60) if sec['human'][i][0] else 0
        c = (lo + hi) / 2
        ax[1].bar(c - 7, ai_r, width=12, color=S1, label='AI' if i == 0 else None)
        ax[1].bar(c + 7, hu_r, width=12, color=S2, label='人' if i == 0 else None)
        if lo < 120:
            ax[1].text(c - 7, ai_r + 0.3, '%d件' % sec['ai'][i][1], ha='center', fontsize=8, color='#0b0b0b')
    ax[1].set_ylabel('衝突 / 分'); ax[1].set_xlabel('周回位置 [deg](60°ごと)'); ax[1].set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax[1].legend(loc='upper right', fontsize=9); ax[1].set_xlim(0, 360)
    ax[1].set_title('AIの衝突の %d%% が 0-120° に集中(AIの滞在時間は %d%%)' % (round(100 * res['hot_share_ai']), round(100 * res['hot_time_share_ai'])), fontsize=11, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_collision_sectors.png'), dpi=110); plt.close(fig)

    # 3) ゲイン区間ごとの衝突率と95%信頼区間
    fig, ax = plt.subplots(figsize=(10, 4))
    order = sorted(range(len(blocks)), key=lambda i: (blocks[i]['la'], blocks[i]['gs'], blocks[i]['gt']))
    col = {1: S3, 3: S1, 5: S2}
    for j, i in enumerate(order):
        b = blocks[i]; lo, hi = b['collisions_per_ai_min_ci95']
        ax.plot([j, j], [lo, hi], color=col[b['la']], lw=2, solid_capstyle='round')
        ax.scatter([j], [b['collisions_per_ai_min']], s=50, color=col[b['la']], edgecolor='#fcfcfb', linewidth=2, zorder=3)
    for la, c in col.items():
        ax.scatter([], [], s=50, color=c, label='lookahead %d' % la)
    hr = res['human_pooled_collisions_per_min']
    ax.axhline(hr, color=MUTED, lw=1); ax.text(len(order) - 0.5, hr + 0.4, '人(5ログ合算) %.1f/分' % hr, ha='right', fontsize=8, color='#52514e')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['%s\n%.2f/%.2f\n%.0fs' % (blocks[i]['label'].replace('ai_run_', ''), blocks[i]['gs'], blocks[i]['gt'], blocks[i]['ai_s']) for i in order], fontsize=7)
    ax.set_ylabel('衝突 / AI分(95%信頼区間)'); ax.set_ylim(0, 32)
    ax.set_title('ゲイン区間ごとの衝突率: 区間が短く、どの差も信頼区間が重なる', fontsize=11, loc='left')
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    fig.text(0.01, 0.01, 'x軸: 走行 / steer_gain/throttle_gain / AI時間。trial・run_001 はI2C修正前。', fontsize=8, color='#52514e')
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(HERE, 'fig_collision_ci.png'), dpi=110); plt.close(fig)

    # 4) スタックの典型例
    D = load('20261010-173445'); ai = D['AI_CMD']; v = D['VEHICLE']; im = D['IMU']; tc = res['stuck_example_t']
    w = (ai['t'] > tc - 3) & (ai['t'] < tc + 3); wv = (v['t'] > tc - 3) & (v['t'] < tc + 3); wi = (im['t'] > tc - 3) & (im['t'] < tc + 3)
    fig, ax = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    ax[0].plot(ai['t'][w] - tc, ai['out'][w, 0], color=S2, lw=2, label='出力 out')
    ax[0].plot(ai['t'][w] - tc, ai['raw'][w, 0], color=S1, lw=2, label='モデル raw')
    ax[0].axhline(CENTER, color='#c9c8c3', lw=0.8); ax[0].set_ylabel('ステア [us]\n(上 = 左)'); ax[0].legend(loc='lower left', fontsize=8, ncol=2)
    ax[0].text(-2.95, 1800, '左いっぱい(2000)から\n0.5秒で右いっぱい(1000)へ', fontsize=8, color='#0b0b0b', va='center')
    ax[1].plot(im['t'][wi] - tc, im['gyro'][wi, 2], color=S1, lw=2); ax[1].axhline(0, color='#c9c8c3', lw=0.8)
    ax[1].set_ylabel('ヨーレート\n[deg/s]')
    ax[2].plot(ai['t'][w] - tc, ai['out'][w, 1], color=S2, lw=2, label='出力 out')
    ax[2].plot(ai['t'][w] - tc, ai['raw'][w, 1], color=S1, lw=2, label='モデル raw')
    ax[2].axhline(NEUTRAL, color='#c9c8c3', lw=0.8); ax[2].set_ylabel('スロットル [us]\n(上 = ブレーキ/後退)'); ax[2].legend(loc='upper left', fontsize=8, ncol=2)
    ax[2].text(0.15, 1250, '後退 1930-1990us を出し続けるが、中立を挟まないので\nESCはブレーキのまま(車は動かない)', fontsize=8, color='#0b0b0b')
    ax[3].plot(v['t'][wv] - tc, np.abs(v['erpm'][wv]), color=S1, lw=2); ax[3].set_ylabel('|eRPM|')
    sel = v['select'][wv].astype(bool)
    if (~sel).any():
        x0 = (v['t'][wv][~sel][0]) - tc
        for a in ax:
            a.axvline(x0, color=MUTED, lw=1)
        ax[3].text(x0 + 0.05, 5500, '人が交代', fontsize=8, color='#52514e')
    for a in ax:
        a.axvline(0, color='#0b0b0b', lw=1)
    ax[0].text(0.05, 1900, '衝突(前後 −6g)', fontsize=8, color='#0b0b0b')
    ax[3].set_xlabel('衝突からの時間 [s](ai_run_007, t = %.1f s, steer_gain 1.4 / throttle_gain 2.0)' % tc)
    ax[0].set_title('スタックの典型例: コーナー出口で右へ急に切り返し、正面から仕切りに当たって停止', fontsize=11, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_stuck_example.png'), dpi=110); plt.close(fig)


def frames_figure():
    """集中区間での衝突の直前のカメラ画像(-1.5〜0秒)"""
    import cv2
    cases = [('20261010-172910', 95.2, 'ai_run_006 直進のまま右前が当たる'), ('20261010-172910', 113.3, 'ai_run_006 直進のまま右前が当たる'),
             ('20261010-173445', 58.2, 'ai_run_007 右へ切り返して正面から(スタック)'), ('20261010-173445', 143.2, 'ai_run_007 右へ切り返して正面から(スタック)')]
    rows = []
    for log, tc, _ in cases:
        root = os.path.join(RAW, log); ts, fs = [], []
        for s in sorted(x for x in os.listdir(root) if x.startswith(log + '-')):
            try:
                z = np.load(os.path.join(root, s, 'CAM_FRONT.npz'), allow_pickle=True)
            except Exception:
                continue
            ts.append(z['timestamps_us']); fs += [os.path.join(root, s, str(f)) for f in z['files']]
        t = (np.concatenate(ts) - load(log)['VEHICLE']['timestamps_us'][0]) / 1e6
        row = []
        for o in (-1.5, -1.0, -0.5, -0.2, 0.0):
            im = cv2.imread(fs[int(np.argmin(np.abs(t - (tc + o))))])
            cv2.putText(im, '%+.1fs' % o, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            row.append(im)
        rows.append(np.hstack(row))
    cv2.imwrite(os.path.join(HERE, 'fig_hotspot_frames.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------------------ main
def rnd(x, n=3):
    if isinstance(x, float):
        return float(round(float(x), n))
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (np.floating,)):
        return round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main_full():
    ref = reference_signature()
    res = {'definitions': {
        'ai_side': 'VEHICLE.select == True(プロポCH3でAI側)。時間は100Hzサンプル間隔の合計(%.1f秒超の欠損は除く)' % GAP,
        'takeover': 'AI区間がselect=Falseに切り替わって終わった回数(データ末尾・欠損は除く)',
        'collision': '水平 |ax| または |ay| >= %.1fg、1サンプルだけの跳ね(suspect)を除く、%.1f秒以内は1件、AI側のみ' % (COLL_G, EVENT_WIN),
        'vertical_bump': '|az-1g| >= %.1fg で水平は %.1fg 未満' % (COLL_G, COLL_G),
        'stuck': 'AI側の衝突から%.1f秒以内に |eRPM|<%d が%.1f秒以上続いた回数' % (STUCK_WIN, STUCK_ERPM, STUCK_MIN_S),
        'laps': 'AI側のジャイロzを積分した正味方位/360。lap_time_median_s は1区間内で連続した完全な周回の中央値',
        'steer_reversals': '出力ステアが中心±%.0fusを越えて左右反対側へ移った回数/秒' % REV_HYST,
        'forward_throttle': '前進量 = 1498 - throttle_us(大きいほど速い指令)',
        'imu_axes': '実測: IMU +y = 前方, +x = 右, +z = 上(config.CALIBRATIONの回転[1,0,0,0]とは90度違う)。接触方向は感じた加速度の逆向き',
        'collisions_per_lap': '衝突件数 / AI側の周回数(速度が違う設定どうしを比べるときはこちら)',
    }, 'runs': {}, 'gain_blocks': [], 'human': {}, 'collisions': [], 'stuck_events': []}
    sec_ai = np.zeros((len(SECT), 2)); sec_h = np.zeros((len(SECT), 2))
    for log, label, note in AI_LOGS:
        D = load(log)
        m, coll, stuck = ai_metrics(D, ref=ref)
        m.update({'log': log, 'label': label, 'note': note, 'params_log_json': D['meta']['drive']['params'],
                  'scenes_used': len(D['used']), 'scenes_skipped': D['skipped']})
        m['lookahead_analysis'] = lookahead_stats(D)
        ai = D['AI_CMD']; am = ai['select'].astype(bool) & ~ai['hold'].astype(bool)
        m['zero_yaw_steer_us'] = zero_yaw_steer(ai['t'][am], ai['out'][am, 0].astype(float), D,
                                                lookup(D['VEHICLE']['t'], D['VEHICLE']['select'].astype(bool), ai['t'][am]).astype(bool))
        res['runs'][label] = m
        for b in gain_blocks(D):
            b.update({'log': log, 'label': label}); res['gain_blocks'].append(b)
        # 周回位置
        im = D['IMU']; v = D['VEHICLE']; gz = yaw(D); dt = sample_dt(im['t'])
        sel_imu = lookup(v['t'], v['select'].astype(bool), im['t']).astype(bool)
        ph = phase_of(D, sel_imu, ref)
        for i, (lo, hi) in enumerate(SECT):
            sec_ai[i, 0] += float(np.sum(dt[sel_imu & (ph >= lo) & (ph < hi)]))
        for e in coll:
            p = ph[min(np.searchsorted(im['t'], e['t']), len(ph) - 1)]
            c = classify(D, e, gz)
            gi = max(np.searchsorted(ai['t'], e['t']) - 1, 0)
            row = {'run': label, 't': e['t'], 'h_g': e['h'], 'ax_g': e['ax'], 'ay_g': e['ay'], 'contact': e['contact'],
                   'phase_deg': None if np.isnan(p) else float(p), 'gains': [float(x) for x in ai['gains'][gi]],
                   'lookahead': int(ai['lookahead'][gi]), 'stuck': any(abs(s['t'] - e['t']) < 1e-6 for s in stuck)}
            row.update(c); res['collisions'].append(row)
            if not np.isnan(p):
                sec_ai[int(p // 60), 1] += 1
        for s in stuck:
            s.update({'run': label}); res['stuck_events'].append(s)
    for log in HUMAN_LOGS:
        D = load(log)
        h, coll, ph = human_metrics(D, ref)
        res['human'][log] = h
        im = D['IMU']; dt = sample_dt(im['t']); gz = yaw(D)
        for i, (lo, hi) in enumerate(SECT):
            sec_h[i, 0] += float(np.sum(dt[(ph >= lo) & (ph < hi)]))
        for e in coll:
            p = ph[min(np.searchsorted(im['t'], e['t']), len(ph) - 1)]
            if not np.isnan(p):
                sec_h[int(p // 60), 1] += 1
            c = classify(D, e, gz)
            res['collisions'].append({'run': 'human ' + log, 't': e['t'], 'h_g': e['h'], 'ax_g': e['ax'], 'ay_g': e['ay'], 'contact': e['contact'],
                                      'phase_deg': None if np.isnan(p) else float(p), **c})
    res['ref_signature'] = ref.tolist()
    res['sectors'] = {'deg': SECT, 'ai': sec_ai.tolist(), 'human': sec_h.tolist(), 'columns': ['aligned_time_s', 'collisions']}
    res['hot_share_ai'] = float(sec_ai[:2, 1].sum() / max(sec_ai[:, 1].sum(), 1))
    res['hot_time_share_ai'] = float(sec_ai[:2, 0].sum() / max(sec_ai[:, 0].sum(), 1e-9))
    res['hot_share_human'] = float(sec_h[:2, 1].sum() / max(sec_h[:, 1].sum(), 1))
    res['hot_time_share_human'] = float(sec_h[:2, 0].sum() / max(sec_h[:, 0].sum(), 1e-9))
    res['human_pooled_collisions_per_min'] = sum(h['collisions'] for h in res['human'].values()) / (sum(h['total_s'] for h in res['human'].values()) / 60)
    # 人の zero-yaw
    zs = {}
    for log in HUMAN_LOGS:
        D = load(log); v = D['VEHICLE']; ok = v['valid'][:, 0]
        zs[log] = zero_yaw_steer(v['t'][ok], v['pwm'][ok, 0].astype(float), D, np.ones(ok.sum(), bool))
    res['human_zero_yaw_steer_us'] = zs
    res['imu_axes_check'] = {log: imu_axes_check(load(log)) for log in ('20261010-172910', '20261010-173445', '20261010-152707')}
    frames_figure()
    st_ex = [s for s in res['stuck_events'] if s['run'] == 'ai_run_007']
    res['stuck_example_t'] = st_ex[1]['t'] if len(st_ex) > 1 else (st_ex[0]['t'] if st_ex else 58.2)
    res = rnd(res)
    with open(os.path.join(HERE, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    figures(res)
    return res


def main_quick(logs):
    try:
        ref = np.array(json.load(open(os.path.join(HERE, 'params.json'), encoding='utf-8'))['ref_signature'])
    except (OSError, ValueError, KeyError):
        ref = reference_signature()
    keys = ['ai_s', 'ai_ratio', 'ai_seg_median_s', 'ai_seg_longest_s', 'takeovers_per_ai_min', 'collisions', 'collisions_per_ai_min',
            'collisions_per_lap',
            'collisions_per_ai_min_ci95', 'collisions_in_hot_sector_0_120', 'vertical_bumps', 'stuck', 'stuck_per_ai_min',
            'erpm_abs_p50', 'erpm_abs_p90', 'laps', 'direction', 'lap_time_median_s', 'lap_time_ai_s_per_lap', 'erev_per_lap',
            'steer_reversals_per_s_out', 'steer_at_1000_right', 'steer_at_2000_left', 'steer_out_absdev_p50', 'hold_ratio']
    for log in logs:
        D = load(log)
        m, _, _ = ai_metrics(D, ref=ref)
        print('==', log, D['meta'].get('description'), 'skipped:', D['skipped'])
        for k in keys:
            print('  %-34s %s' % (k, rnd(m[k], 2)))
        print('  throttle_out                       ', rnd(m['throttle_out'], 1))
        for b in gain_blocks(D):
            print('  block %.2f/%.2f la%d %5.1fs  coll %.1f/min  laps %.1f  eRPM p50 %.0f  raw fwd %.0f -> out fwd %.0f  limit R/L %.2f/%.2f' % (
                b['gs'], b['gt'], b['la'], b['ai_s'], b['collisions_per_ai_min'], b['laps'], b['erpm_abs_p50'],
                b['throttle_raw']['fwd_us_p50'], b['throttle_out']['fwd_us_p50'], b['steer_at_1000_right'], b['steer_at_2000_left']))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_quick(sys.argv[1:])
    else:
        main_full()
