#!/usr/bin/env python3
"""区間ごとのタイム・速度・横G・フルロック時の曲率・空転・ステアの速さを、人と AI の周で比べる(読み取りのみ、CPU)

    nice -n 15 python sections.py > ../sections.json

周の区切り: R1(コースで唯一の大きな右ヘアピン)の開始。ヨー(5点平滑)が −60deg/s を下回り始めた時刻で、
その右旋回の最小ヨー ≤ −170deg/s かつ旋回量 ≤ −90deg(collision_map.md §1.3 と同じ考え方)。
区間(R1 開始からの向き ψ、+ = 左): R1 = 開始〜ψ最小、GM = ψ最小〜ψ −20、DE = −20〜+40、SW = +40〜+175、
HS = +175〜+215、LH = +215〜次の R1 開始。しきい値は人と AI で共通(絶対値よりも比較に使う)。
1周と数える条件: 途切れ(>0.3 s)なし、5〜16 s、ψ の正味の変化 300〜420deg、全部人か全部 AI。
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import (AI_LOGS, G, HUMAN_LOGS, KV, STEER_C, THR_N, impacts, load, prev_index, smooth)  # noqa: E402

SECS = ['R1', 'GM', 'DE', 'SW', 'HS', 'LH']
BOUNDS = [('DE', -20.0), ('SW', 40.0), ('HS', 175.0), ('LH', 215.0)]   # (区間名, その区間に入る ψ)


def analyse(log, who):
    meta, d = load(log)
    imu, veh, cmd = d['IMU'], d['VEHICLE'], d['AI_CMD']
    t = imu['timestamps_us'].astype(np.int64)
    ts = (t - t[0]) / 1e6
    gz = np.clip(imu['gyro'][:, 2].astype(np.float64), -990, 990)
    acc = imu['accel'].astype(np.float64)
    kv = prev_index(veh['timestamps_us'], t)
    erpm = veh['erpm'][kv].astype(np.float64)
    sel = veh['select'][kv].astype(bool)
    v = -erpm * KV
    # 操作: 人は受信機の PWM、AI は AI_CMD.out(AI 側のとき)
    steer = veh['pwm'][kv, 0].astype(np.float64)
    thr = veh['pwm'][kv, 1].astype(np.float64)
    if cmd is not None and len(cmd['timestamps_us']):
        kc = prev_index(cmd['timestamps_us'], t)
        ok = sel & (cmd['timestamps_us'][kc] <= t) & (t - cmd['timestamps_us'][kc] < 100_000)
        steer = np.where(ok, cmd['out'][kc, 0], steer)
        thr = np.where(ok, cmd['out'][kc, 1], thr)
    still = (np.abs(erpm) < 50) & (np.abs(gz) < 5)
    bias = float(np.median(gz[still])) if still.sum() > 50 else 0.0
    g = gz - bias
    dt = np.diff(ts, prepend=ts[0])
    gap = dt > 0.3
    mov = np.abs(erpm) > 200
    psi = np.cumsum(np.where(mov & (dt < 0.2), g * dt, 0.0))
    gs = smooth(g, 5)
    ev = impacts(t, acc)
    ev_t = np.array([(e[0] - t[0]) / 1e6 for e in ev])
    ev_g = np.array([e[1] for e in ev])
    # R1 の開始を探す
    starts = []
    i, n = 0, len(ts)
    right = gs < -60
    while i < n:
        if right[i]:
            j = i
            while j < n and right[j] and (j == i or not gap[j]):
                j += 1
            seg = slice(i, j)
            if gs[seg].min() <= -170 and psi[j - 1] - psi[i] <= -90:
                if not starts or ts[i] - starts[-1][1] > 3.0:
                    starts.append((i, ts[i], j))
            i = j
        else:
            i += 1
    laps = []
    for (i0, t0, _), (i1, t1, _) in zip(starts[:-1], starts[1:]):
        dur = t1 - t0
        if not (5.0 <= dur <= 16.0) or gap[i0 + 1:i1 + 1].any():
            continue
        dpsi = psi[i1] - psi[i0]
        if not (300 <= dpsi <= 420):
            continue
        s = sel[i0:i1]
        if s.all():
            drv = 'ai'
        elif not s.any():
            drv = 'human'
        else:
            continue
        rel = psi[i0:i1] - psi[i0]
        tt = ts[i0:i1]
        k_min = int(np.argmin(rel[: max(1, int(min(len(rel), 250)))]))
        bnd = {'R1': 0}
        cur = k_min
        okb = True
        bnd['GM'] = k_min
        for name, th in BOUNDS:
            idx = np.nonzero(rel[cur:] >= th)[0]
            if not len(idx):
                okb = False
                break
            cur = cur + int(idx[0])
            bnd[name] = cur
        if not okb:
            continue
        # GM は ψ最小から始まる
        edges = [0, bnd['GM'], bnd['DE'], bnd['SW'], bnd['HS'], bnd['LH']]
        edges.append(len(rel))
        sec = {}
        vv = v[i0:i1]
        gg = g[i0:i1]
        ax = smooth(acc[i0:i1, 0], 10)
        st = steer[i0:i1] - STEER_C
        for k, name in enumerate(SECS):
            a, b = edges[k], edges[k + 1]
            if b <= a:
                sec[name] = None
                continue
            seg = slice(a, b)
            alat_kin = np.abs(np.deg2rad(gg[seg]) * vv[seg])
            sec[name] = {
                't_s': round(float((tt[b] if b < len(tt) else t1) - tt[a]), 3),
                'v_mean': round(float(np.mean(vv[seg])), 2), 'v_max': round(float(np.max(vv[seg])), 2),
                'alat_kin_p90': round(float(np.percentile(alat_kin, 90)), 2),
                'alat_imu_p90': round(float(np.percentile(np.abs(ax[seg]), 90)), 2),
                'steer_abs_p90': round(float(np.percentile(np.abs(st[seg]), 90)), 1),
            }
        w = (ts >= t0 - 0.4) & (ts <= t0)
        r1_entry = float(np.max(-erpm[w])) if w.any() else None
        imp = [(round(float(x - t0), 2), round(float(gv), 1)) for x, gv in zip(ev_t, ev_g) if t0 <= x < t1]
        imp_sec = []
        for x, gv in imp:
            k = int(np.searchsorted(tt - t0, x, side='right')) - 1
            sname = SECS[int(np.searchsorted(edges, k, side='right')) - 1] if k >= 0 else 'R1'
            imp_sec.append(sname)
        lap = {'log': log, 'who': who, 'driver': drv, 't0': round(float(t0), 2), 'lap_s': round(float(dur), 3),
               'dpsi': round(float(dpsi), 1), 'r1_entry_erpm': r1_entry, 'impacts': imp, 'impact_secs': imp_sec,
               'n_imp': len(imp), 'sec': sec, 'v_max_lap': round(float(np.max(vv)), 2)}
        if drv == 'ai' and cmd is not None:
            kc0 = prev_index(cmd['timestamps_us'], t[i0:i1])
            lap['gains'] = [float(np.median(cmd['gains'][kc0, 0])), float(np.median(cmd['gains'][kc0, 1]))]
            lap['lookahead'] = int(np.median(cmd['lookahead'][kc0]))
        laps.append(lap)

    # 加速時の空転: d(v_eRPM)/dt と IMU 前後加速度(+y = 前)の差(0.2 s 平滑)
    vs = smooth(v, 20)
    dv = np.gradient(vs, ts)
    ay = smooth(acc[:, 1], 20)
    # ステアの速さ(µs/s): 人は 100Hz の PWM(3点平滑)、AI は AI_CMD.out(30Hz)
    out = {'log': log, 'who': who, 'desc': meta.get('description', ''), 'bias_dps': round(bias, 3), 'laps': laps,
           'n_r1': len(starts)}
    # サンプル単位の特徴(区間つき)を返す
    samp = {'ts': ts, 'v': v, 'g': g, 'ax': smooth(acc[:, 0], 10), 'steer': steer, 'thr': thr, 'sel': sel,
            'dv': dv, 'ay': ay}
    seclab = np.full(len(ts), '', dtype=object)
    for lp in laps:
        i0 = int(np.searchsorted(ts, lp['t0']))
        acc_t = 0.0
        tt0 = ts[i0]
        # 区間の時刻境界から各サンプルに区間名を付ける
        bounds = [tt0]
        for name in SECS:
            acc_t += lp['sec'][name]['t_s'] if lp['sec'][name] else 0.0
            bounds.append(tt0 + acc_t)
        i1 = int(np.searchsorted(ts, tt0 + lp['lap_s']))
        for k, name in enumerate(SECS):
            a = int(np.searchsorted(ts, bounds[k]))
            b = int(np.searchsorted(ts, bounds[k + 1]))
            seclab[a:min(b, i1)] = name
    samp['sec'] = seclab
    return out, samp


def main():
    res = {'logs': [], 'definitions': __doc__}
    pools = {'human': [], 'ai': []}
    for log in HUMAN_LOGS:
        o, s = analyse(log, 'human')
        res['logs'].append(o)
        pools['human'].append(s)
    for log, name in AI_LOGS.items():
        o, s = analyse(log, name)
        res['logs'].append(o)
        pools['ai'].append(s)
    # フルロック時の曲率・横G(区間 × 速度帯)、人と AI(AI 側)を合わせて区間ごと
    lock = {}
    for grp, ss in pools.items():
        for s in ss:
            ok = s['sec'] != ''
            if grp == 'ai':
                ok &= s['sel']
            else:
                ok &= ~s['sel']
            dev = s['steer'] - STEER_C
            for side, m in (('left', dev >= 380), ('right', dev <= -380)):
                mm = ok & m & (s['v'] > 1.0)
                for sec in SECS:
                    q = mm & (s['sec'] == sec)
                    if q.sum() < 10:
                        continue
                    kap = np.deg2rad(s['g'][q]) / s['v'][q]
                    for lo, hi in ((1.0, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 9.0)):
                        b = (s['v'][q] >= lo) & (s['v'][q] < hi)
                        if b.sum() < 10:
                            continue
                        key = '%s|%s|%s|%.1f-%.1f' % (grp, side, sec, lo, hi)
                        e = lock.setdefault(key, [])
                        e.extend(kap[b].tolist())
    lock_sum = {k: {'n': len(v), 'kappa_med': round(float(np.median(v)), 3), 'kappa_p90abs': round(
        float(np.percentile(np.abs(v), 90)), 3)} for k, v in sorted(lock.items())}
    res['full_lock_curvature'] = lock_sum
    # 横G(運動学 v*ω)の p95(区間ごと、速度 > 1 m/s、全サンプル)
    alat = {}
    for grp, ss in pools.items():
        for sec in SECS:
            vals = []
            for s in ss:
                ok = (s['sec'] == sec) & (s['v'] > 1.0) & (s['sel'] if grp == 'ai' else ~s['sel'])
                vals.extend(np.abs(np.deg2rad(s['g'][ok]) * s['v'][ok]).tolist())
            if len(vals) > 20:
                alat['%s|%s' % (grp, sec)] = {'n': len(vals), 'p50': round(float(np.percentile(vals, 50)), 2),
                                              'p90': round(float(np.percentile(vals, 90)), 2),
                                              'p99': round(float(np.percentile(vals, 99)), 2)}
    res['lateral_accel_kin'] = alat
    # 空転の目安: 加速中(スロットル前進 > 80us、1 < v < 4)の d(v_eRPM)/dt − IMU 前後加速度
    slip = {}
    for grp, ss in pools.items():
        for sec in SECS:
            ex, dvs, ays = [], [], []
            for s in ss:
                ok = (s['sec'] == sec) & (s['v'] > 1.0) & (s['v'] < 4.5) & (s['thr'] < THR_N - 80) & (s['dv'] > 0.3)
                ok &= (s['sel'] if grp == 'ai' else ~s['sel'])
                ex.extend((s['dv'][ok] - s['ay'][ok]).tolist())
                dvs.extend(s['dv'][ok].tolist())
                ays.extend(s['ay'][ok].tolist())
            if len(ex) > 20:
                slip['%s|%s' % (grp, sec)] = {'n': len(ex), 'dv_erpm_med': round(float(np.median(dvs)), 2),
                                              'ay_imu_med': round(float(np.median(ays)), 2),
                                              'excess_med': round(float(np.median(ex)), 2),
                                              'excess_p90': round(float(np.percentile(ex, 90)), 2)}
    res['wheelspin_proxy'] = slip
    # 減速の能力: 前進中(v > 1)のブレーキ指令(> 中立+40)の区間の −dv/dt(IMU と eRPM)
    brake = {}
    for grp, ss in pools.items():
        vals_e, vals_i, coast_i = [], [], []
        for s in ss:
            own = s['sel'] if grp == 'ai' else ~s['sel']
            ok = own & (s['v'] > 1.0) & (s['thr'] > THR_N + 40)
            vals_e.extend((-s['dv'][ok]).tolist())
            vals_i.extend((-s['ay'][ok]).tolist())
            okc = own & (s['v'] > 1.0) & (np.abs(s['thr'] - THR_N) <= 20)
            coast_i.extend((-s['ay'][okc]).tolist())
        if vals_e:
            brake[grp] = {'n_brake': len(vals_e), 'decel_erpm_p50': round(float(np.median(vals_e)), 2),
                          'decel_erpm_p90': round(float(np.percentile(vals_e, 90)), 2),
                          'decel_imu_p50': round(float(np.median(vals_i)), 2),
                          'decel_imu_p90': round(float(np.percentile(vals_i, 90)), 2),
                          'n_coast': len(coast_i),
                          'coast_decel_imu_p50': round(float(np.median(coast_i)), 2) if coast_i else None}
    res['braking'] = brake
    # ステアの速さ(|Δsteer|/Δt、µs/s)。人 = 受信機 PWM、AI = AI_CMD.out。区間ごとの p50/p95/p99
    slew = {}
    for grp, ss in pools.items():
        for sec in SECS + ['ALL']:
            vals = []
            for s in ss:
                own = s['sel'] if grp == 'ai' else ~s['sel']
                st = smooth(s['steer'], 3)
                # 30Hz 程度に間引いて比べる(AI の更新周期)
                idx = np.arange(0, len(st), 3)
                d = np.abs(np.diff(st[idx])) / np.maximum(np.diff(s['ts'][idx]), 1e-3)
                ok = own[idx][1:] & (s['v'][idx][1:] > 0.5)
                if sec != 'ALL':
                    ok &= s['sec'][idx][1:] == sec
                vals.extend(d[ok].tolist())
            if len(vals) > 20:
                slew['%s|%s' % (grp, sec)] = {'n': len(vals), 'p50': round(float(np.percentile(vals, 50)), 0),
                                              'p95': round(float(np.percentile(vals, 95)), 0),
                                              'p99': round(float(np.percentile(vals, 99)), 0)}
    res['steer_slew_us_per_s'] = slew
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
