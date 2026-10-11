#!/usr/bin/env python3
"""ToF ガードの「予測コース」と「接近の確認」を変えたときの、捕捉率・誤警報・警告時間・ラップ損失の目安(オフライン、CPU)

    nice -n 15 python tof_variants.py > ../tof_variants.json

使うもの: 生ログの TOF_FRONT(42x54)、VEHICLE(eRPM・select)、IMU(gyro z・加速度)、AI_CMD(out・pred・gains・lookahead)。
障害物画素の抽出は jetformer/tof_guard.py の ToFGuard.mask と同じ(床基準 data/tof_floor_ref.npy)。

予測コース(帯 = 半幅 0.10 + 0.05 m)の作り方:
  U0  今の実装: ヨーの円弧 ∪ ステア指令の円弧
  L   遅れを入れた1本の道: 最初の v*TAU [m] は今のヨーの曲率、その先はステア指令の曲率(TAU = 0.13 s =
      撮像→出力 28 ms + 指令→ヨー応答 約 100 ms。tof_predictor.md §8)
  P   計画を使う1本の道: 最初の v*TAU は今のヨー、その先はモデルの 10 フレーム予測(ゲイン後)を時刻順に使う
  LP  L ∪ P
接近の確認(+C): 警告(TTC < しきい値)は、0.1 s 前にもすでに帯の中に物があり(3 m 未満)、前方距離 x_free が
  自車速度の 50% 以上の速さで縮んでいるときだけ出す(壁と平行に走るときは x_free がほぼ一定なので警告しない)。
  ブレーキ段(TTC < 0.3)は確認なし。

評価の定義(overnight runs/summary.json に合わせた):
  衝突 = AI 側、水平 hypot(ax,ay) ≥ 2.5 G を 0.6 s でまとめ、直前 0.2 s の速度 > 0.5 m/s
  捕捉(0.3) = 警告の始まりが衝撃の 0.3 s 以上前で、衝撃前 0.3 s 以内まで警告が続いている
  誤警報 = 警告の始まりから 1.5 s 以内に 2.0 G 以上の出来事がない。率は AI 前進時間(2 G 以上の後 1 s を除く)あたり
  警告時間率 = AI 前進フレーム(衝撃の 1.5 s 前以内を除く)のうち警告が出ている割合
  ラップ損失(目安) = 警告中は前進を中立へ(切迫度 u に比例して最大 1.0 m/s^2)、ブレーキ段は 2.8 m/s^2、反応まで 0.15 s、
      その後 1.5 m/s^2 で元の速さへ戻る、として失った距離 ÷ 速度 を AI の周(ヨーの向き 360 度)あたりにした開ループの値
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from jetformer.tof_guard import ToFGuard, KV, K_LEFT, K_RIGHT, K_MAX_L, K_MAX_R  # noqa: E402
from common import G, load, prev_index, impacts  # noqa: E402

TAU = 0.13
DT_F = 1 / 30.0
W = 0.15
FLOOR = np.load('/home/jetson/jetformer/data/tof_floor_ref.npy')
LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007',
        '20261010-203924': 'ai_run_009', '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}


def kappa_cmd(steer_us, center=1443.0):
    s = np.asarray(steer_us, dtype=np.float64) - center
    return np.clip(np.where(s > 0, s * K_LEFT, s * K_RIGHT), -K_MAX_R, K_MAX_L)


def frame_points(guard, raw):
    raw = np.asarray(raw)[guard.ROWS]
    d = (raw & 0x1FFF).astype(np.float32)
    d[d == 0x1FFF] = 0
    d *= 0.001
    conf = raw >> 13
    m = guard.mask(d, conf)
    x = (d * guard.fwd)[m]
    y = (d * guard.left)[m] + guard.Y_OFF
    return x, y, int(m.sum())


def kth_x(x, inpath, k=3):
    xs = x[inpath]
    return float(np.partition(xs, k - 1)[k - 1]) if xs.size >= k else 9.0


def path_latency(x, v, kg, kc):
    x0 = v * TAU
    y = np.where(x <= x0, 0.5 * kg * x * x, 0.5 * kg * x0 * x0 + kg * x0 * (x - x0) + 0.5 * kc * (x - x0) ** 2)
    return y


def path_plan(x, v, kg, kplan):
    """kplan: 10 個の曲率(予測の 1..10 フレーム先、ゲイン後)。距離 x で効いている曲率を時刻で引く"""
    grid = np.linspace(0, 2.2, 89)
    tt = grid / max(v, 0.3)
    j = np.floor((tt - TAU) / DT_F).astype(int)
    kap = np.where(tt < TAU, kg, kplan[np.clip(j, 0, len(kplan) - 1)])
    dx = grid[1] - grid[0]
    th = np.concatenate([[0.0], np.cumsum(kap[:-1] * dx)])
    yy = np.concatenate([[0.0], np.cumsum(np.sin(th[:-1]) * dx)])
    return np.interp(x, grid, yy)


def episodes(t, flag, merge=0.3):
    eps = []
    on = None
    last = None
    for ti, f in zip(t, flag):
        if f:
            if on is None:
                on = ti
            elif last is not None and ti - last > merge:
                eps.append([on, last])
                on = ti
            last = ti
        elif on is not None and last is not None and ti - last > merge:
            eps.append([on, last])
            on, last = None, None
    if on is not None:
        eps.append([on, last])
    return eps


def evaluate(t, fwd, warn, brake, u, v, coll, ev2, laps):
    """t: フレーム時刻 [s]、fwd: AI 前進フレーム、warn/brake: 判定、coll: 衝突時刻、ev2: 2 G 以上の時刻"""
    w = warn & fwd
    eps = episodes(t[w], np.ones(w.sum(), bool))
    caught03 = caught015 = 0
    leads = []
    for tc in coll:
        best = None
        for a, b in eps:
            if b >= tc - 0.3 and a <= tc:
                if best is None or a < best:
                    best = a
        if best is not None:
            if tc - best >= 0.3:
                caught03 += 1
            if tc - best >= 0.15:
                caught015 += 1
            leads.append(tc - best)
    fa = sum(1 for a, b in eps if not np.any((ev2 >= a) & (ev2 <= a + 1.5)))
    # 曝露時間: AI 前進 − 2 G 以上の後 1 s
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 0.1)
    excl = np.zeros(len(t), bool)
    for e in ev2:
        excl |= (t >= e) & (t <= e + 1.0)
    expo = float(dt[fwd & ~excl].sum()) / 60.0
    pre = np.zeros(len(t), bool)
    for tc in coll:
        pre |= (t >= tc - 1.5) & (t <= tc)
    base = fwd & ~pre & ~excl
    wt = float((w & base).sum()) / max(int(base.sum()), 1)
    # ラップ損失の目安(全警告)
    lost_t = 0.0
    lost_fa = 0.0
    for a, b in eps:
        is_fa = not np.any((ev2 >= a) & (ev2 <= a + 1.5))
        idx = np.nonzero((t >= a) & (t <= b + 1e-6) & w)[0]
        if not len(idx):
            continue
        v0 = float(np.median(v[idx]))
        D, area, tcur = 0.0, 0.0, a
        k = 0
        while True:
            tcur += 1 / 30.0
            if tcur <= b + 0.15:
                j = idx[min(k, len(idx) - 1)]
                k += 1
                acc = 0.0
                if tcur - a >= 0.15:
                    acc = 2.8 if brake[j] else 1.0 * float(u[j])
                D = min(v0, D + acc / 30.0)
            else:
                D = max(0.0, D - 1.5 / 30.0)
                if D <= 0:
                    break
            area += D / 30.0
        lost_t += area / max(v0, 0.5)
        if is_fa:
            lost_fa += area / max(v0, 0.5)
    return {'n_coll': len(coll), 'caught_0.3': caught03, 'caught_0.15': caught015,
            'lead_med_s': round(float(np.median(leads)), 2) if leads else None, 'episodes': len(eps), 'fa': fa,
            'expo_min': round(expo, 3), 'fa_per_min': round(fa / max(expo, 1e-6), 1), 'warn_time_frac': round(wt, 3),
            'lap_cost_s_per_lap': round(lost_t / max(laps, 1e-6), 2),
            'lap_cost_fa_only_s_per_lap': round(lost_fa / max(laps, 1e-6), 2)}


def run_log(log, floor):
    guard = ToFGuard(floor)
    _, d = load(log, ('TOF_FRONT', 'VEHICLE', 'IMU', 'AI_CMD'))
    tof, veh, imu, cmd = d['TOF_FRONT'], d['VEHICLE'], d['IMU'], d['AI_CMD']
    tt = tof['timestamps_us']
    t0 = cmd['timestamps_us'][0]
    kv = prev_index(veh['timestamps_us'], tt)
    ki = prev_index(imu['timestamps_us'], tt)
    kc = prev_index(cmd['timestamps_us'], tt)
    erpm = veh['erpm'][kv].astype(float)
    sel = veh['select'][kv].astype(bool) & cmd['select'][kc].astype(bool)
    v = np.maximum(-erpm * KV, 0.0)
    gz_all = imu['gyro'][:, 2].astype(float)
    still = (np.abs(prev_erpm := veh['erpm'][prev_index(veh['timestamps_us'], imu['timestamps_us'])]) < 50) & (np.abs(gz_all) < 5)
    bias = float(np.median(gz_all[still])) if still.sum() > 50 else 0.0
    gz = np.clip(gz_all[ki] - bias, -990, 990)
    steer = cmd['out'][kc, 0].astype(float)
    gains = cmd['gains'][kc]
    pred = cmd['pred'][kc]                          # [N,10,2] µs(ゲイン前)
    la = cmd['lookahead'][kc].astype(int)
    t = (tt - t0) / 1e6
    fwd = sel & (v > 0.5)
    # 衝突と 2 G 以上の出来事(IMU)
    it = (imu['timestamps_us'] - t0) / 1e6
    ev = impacts(imu['timestamps_us'], imu['accel'].astype(float), 2.5)
    ev2 = np.array([(e[0] - t0) / 1e6 for e in impacts(imu['timestamps_us'], imu['accel'].astype(float), 2.0)])
    kvi = prev_index(veh['timestamps_us'], imu['timestamps_us'])
    v_imu = np.maximum(-veh['erpm'][kvi].astype(float) * KV, 0)
    sel_imu = veh['select'][kvi].astype(bool)
    coll = []
    for e in ev:
        te = (e[0] - t0) / 1e6
        i = int(np.searchsorted(it, te))
        w = (it >= te - 0.2) & (it < te)
        if sel_imu[min(i, len(it) - 1)] and w.any() and v_imu[w].max() > 0.5:
            coll.append(te)
    coll = np.array(coll)
    # AI の周(ヨーの向き)
    dti = np.clip(np.diff(it, prepend=it[0]), 0, 0.2)
    mov = v_imu > 0.1
    laps = float(np.sum(np.where(sel_imu & mov, (np.clip(gz_all, -990, 990) - bias) * dti, 0.0))) / 360.0
    # フレームごとの特徴
    n = len(tt)
    feats = {k: np.full(n, 9.0) for k in ('U0', 'L', 'P', 'LP', 'Y', 'C')}
    nobs = np.zeros(n, int)
    for i in range(n):
        if not fwd[i]:
            continue
        x, y, nobs[i] = frame_points(guard, tof['depth'][i])
        if x.size == 0:
            continue
        vi = v[i]
        kg = float(np.clip(np.deg2rad(gz[i]) / max(vi, 0.5), -2.0, 2.0))
        kc_ = float(kappa_cmd(steer[i]))
        g_s = float(gains[i, 0])
        plan_us = 1443.0 + g_s * (pred[i, :, 0] - 1443.0)
        kplan = kappa_cmd(np.clip(plan_us, 1000, 2000))[max(la[i] - 1, 0):]
        if kplan.size == 0:
            kplan = np.array([kc_])
        in_y = np.abs(y - 0.5 * kg * x * x) < W
        in_c = np.abs(y - 0.5 * kc_ * x * x) < W
        in_l = np.abs(y - path_latency(x, vi, kg, kc_)) < W
        in_p = np.abs(y - path_plan(x, vi, kg, kplan)) < W
        feats['U0'][i] = kth_x(x, in_y | in_c)
        feats['Y'][i] = kth_x(x, in_y)
        feats['C'][i] = kth_x(x, in_c)
        feats['L'][i] = kth_x(x, in_l)
        feats['P'][i] = kth_x(x, in_p)
        feats['LP'][i] = kth_x(x, in_l | in_p)
    out = {'log': log, 'name': LOGS.get(log, log), 'ai_heading_laps': round(laps, 2), 'n_coll': int(len(coll)),
           'frames_fwd': int(fwd.sum()), 'variants': {}}
    # 区間(sections.py のジャイロの向きによる判定。周として数えた区間だけ)をフレームに付ける
    try:
        from sections import analyse
        _, smp = analyse(log, LOGS.get(log, log))
        imu_abs = imu['timestamps_us']
        kk = np.clip(np.searchsorted(imu_abs, tt), 0, len(imu_abs) - 1)
        sec_f = smp['sec'][kk]
    except Exception:
        sec_f = np.full(n, '', dtype=object)
    for name, xf in feats.items():
        ttc = xf / np.maximum(v, 0.3)
        # 接近の確認: 0.1 s 前(3 フレーム)からの x_free の変化
        prev = np.concatenate([np.full(3, 9.0), xf[:-3]])
        # 前のフレームでもすでに帯の中に物があり(3 m 未満)、そこから縮んでいる(帯に入った最初のフレームでは出さない)
        closing = (prev < 3.0) & ((prev - xf) >= 0.5 * v * 0.1)
        for thr in (0.5, 0.4):
            for gate in (False, True):
                warn = (ttc < thr) & ((closing | (ttc < 0.3)) if gate else True)
                brake = ttc < 0.3
                u = np.clip((thr - ttc) / max(thr - 0.3, 1e-3), 0, 1)
                key = '%s%s|TTC<%.1f' % (name, '+C' if gate else '', thr)
                out['variants'][key] = evaluate(t, fwd, warn, brake, u, v, coll, ev2, laps)
                if key in ('U0|TTC<0.5', 'C|TTC<0.5', 'L|TTC<0.5', 'C+C|TTC<0.5', 'L+C|TTC<0.4'):
                    # 誤警報と捕捉の区間別(警告の始まりのフレームの区間)
                    w = warn & fwd
                    eps = episodes(t[w], np.ones(w.sum(), bool))
                    by = {}
                    for a, b in eps:
                        i0 = int(np.searchsorted(t, a))
                        sname = sec_f[i0] or '?'
                        fa_ = not np.any((ev2 >= a) & (ev2 <= a + 1.5))
                        e = by.setdefault(sname, {'onsets': 0, 'fa': 0})
                        e['onsets'] += 1
                        e['fa'] += int(fa_)
                    # 前進フレームの区間別の時間
                    for sname in set(sec_f[fwd]):
                        by.setdefault(sname or '?', {'onsets': 0, 'fa': 0})['fwd_s'] = round(float(((sec_f == sname) & fwd).sum()) / 30.0, 1)
                    out['variants'][key]['by_section'] = by
    return out


def main():
    res = {'definitions': __doc__, 'logs': []}
    for log in LOGS:
        res['logs'].append(run_log(log, FLOOR))
        sys.stderr.write('done %s\n' % log)
    # まとめ: 006+007 と 009+010+101
    for grp, logs in (('006+007', ['ai_run_006', 'ai_run_007']), ('009+010+101', ['ai_run_009', 'ai_run_010', 'ai_run_101'])):
        agg = {}
        rows = [l for l in res['logs'] if l['name'] in logs]
        laps = sum(l['ai_heading_laps'] for l in rows)
        for key in rows[0]['variants']:
            vs = [l['variants'][key] for l in rows]
            n_coll = sum(x['n_coll'] for x in vs)
            fa = sum(x['fa'] for x in vs)
            expo = sum(x['expo_min'] for x in vs)
            agg[key] = {'caught_0.3': '%d/%d' % (sum(x['caught_0.3'] for x in vs), n_coll),
                        'caught_0.15': '%d/%d' % (sum(x['caught_0.15'] for x in vs), n_coll),
                        'fa_per_min': round(fa / max(expo, 1e-6), 1), 'fa': fa, 'expo_min': round(expo, 2),
                        'warn_time_frac_wavg': round(sum(x['warn_time_frac'] * x['expo_min'] for x in vs) / max(expo, 1e-6), 3),
                        'lap_cost_s_per_lap': round(sum(x['lap_cost_s_per_lap'] * l['ai_heading_laps'] for x, l in zip(vs, rows)) / max(laps, 1e-6), 2),
                        'lap_cost_fa_only_s_per_lap': round(sum(x['lap_cost_fa_only_s_per_lap'] * l['ai_heading_laps'] for x, l in zip(vs, rows)) / max(laps, 1e-6), 2)}
        res['pooled_' + grp] = {'ai_heading_laps': round(laps, 2), 'variants': agg}
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
