#!/usr/bin/env python3
"""シミュレーター較正の下調べ: 実車のログのうち、車の動きの同定に使える量を数える(2026-10-16、設計の段)

    python -I inventory.py     # → /home/jetson/jetformer_reports/sim_calibration_20261016/inventory.json

読み取りだけ(CPU、数分)。生ログ(data/raw)とラベル(data/labels)は変えない。実車・PCA9685・I2C には触れない。

数えるもの(100 Hz の VEHICLE の時刻で):
  - 操作者(人 = CH3 が人側、AI = AI 側)ごと・区間(data/labels の loc_code)ごとの「走っている秒数」(速さ > 0.5 m/s、きれいな区間だけ)
  - 舵の範囲: 中心からのずれ(µs)の帯 × 速さの帯の秒数(左右別)。舵いっぱい(人 |ずれ| ≥ 430、AI の 1000/2000 張り付き)
  - スロットル: 惰性(中立 ±12 µs を 0.3 秒以上)の秒数(直線・曲がりながら)、ブレーキの出来事(中立 +60 µs 以上を 0.1 秒以上、始まりで 1 m/s 以上)
  - 再生の窓: 1 秒・2 秒の重ならない窓のうち、全部がきれい(衝突の 1 秒前〜2 秒後・基板の再起動 ±0.5 秒・交代/戻し ±0.3 秒・hold・補助・
    ガード・dry_run・指令の欠け を含まない)で、操作者が同じで、始まりの速さ ≥ 0.5 m/s のものの数(区間は窓の始まりの区間)
  - AI の遅れ: AI_CMD.latency_ms(撮像 → PWM 書き込み)の分布
  - 電池の代わりの量: シーンごとの「駆動の効き」g = (1498 − スロットル) / (前進 eRPM / 1000)(直線・一定のスロットル・加速 < 0.5 m/s²)

指令: 人 = VEHICLE.pwm(基板が受信機から読んだ値、100 Hz)。AI = AI_CMD.out を「撮像の時刻 + latency_ms」から次の指令まで保持
  (100 ms より古ければ欠け)。AI が運転中の VEHICLE.pwm は送信機のスティックなので使わない。
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/inventory.json')
LABELS = JF / 'data' / 'labels'
STEER_C_HUMAN, STEER_C_AI, THR_N = 1442.0, 1443.0, 1498.0
KV = 0.5e-3
LOC = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
SPEED_BANDS = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 9.0)]
STEER_BANDS = [(0, 100), (100, 300), (300, 430), (430, 700)]


def movavg(x, n):
    if n <= 1 or len(x) < n:
        return x
    return np.convolve(x, np.ones(n) / n, mode='same')


def runs(mask):
    return le.runs(mask)


def interval_mask(t_us, centers_us, before_s, after_s):
    m = np.zeros(len(t_us), bool)
    for c in centers_us:
        m |= (t_us >= c - before_s * 1e6) & (t_us <= c + after_s * 1e6)
    return m


def span_mask(t_s, spans, pad_s=0.0):
    m = np.zeros(len(t_s), bool)
    for a, b in spans:
        m |= (t_s >= a - pad_s) & (t_s <= b + pad_s)
    return m


def analyse_log(log):
    info = json.loads((LABELS / (log + '.json')).read_text(encoding='utf-8'))
    lab = dict(np.load(LABELS / (log + '.npz')))
    L = le.load_log(log)
    V, I, A = L['VEHICLE'], L['IMU'], L['AI_CMD']
    t0 = info['t0_us']
    res = {'log': log, 'ai_log': bool(info['ai_log']), 'mode_id': info['mode_id'], 'description': info.get('description'),
           'problems': L['problems'][:10], 'scenes': len(L['scenes'])}
    if V is None or I is None or len(V['timestamps_us']) < 200:
        res['skipped'] = 'VEHICLE/IMU が足りない'
        return res, None
    vt_us = V['timestamps_us'].astype(np.int64)
    vt = (vt_us - t0) / 1e6
    sel = le.clean_select(vt, V['select'].astype(bool), V['valid'][:, 2].astype(bool))
    v = -V['erpm'].astype(float) * KV
    it_us = I['timestamps_us'].astype(np.int64)
    gz = np.interp(vt_us, it_us, I['gyro'][:, 2].astype(float)) * np.pi / 180.0
    dt = float(np.median(np.diff(vt)))
    # ---- きれいでない時刻
    ev = info['events']
    bad = interval_mask(vt_us, [c['t_us'] for c in ev.get('collisions', [])], 1.0, 2.0)
    bad |= interval_mask(vt_us, [c['t_us'] for c in ev.get('takeovers', [])], 0.3, 0.3)
    bad |= interval_mask(vt_us, [c['t_us'] for c in ev.get('handbacks', [])], 0.3, 0.3)
    bad |= span_mask(vt, [(r['t_last_s'], r['t_after_s']) for r in ev.get('board_reboots', [])], 0.5)
    for k in ('esc_assists', 'recoveries', 'giveups', 'holds'):
        bad |= span_mask(vt, [(r['t_s'], r['end_s']) for r in ev.get(k, [])], 0.2)
    gaps = np.concatenate([[False], np.diff(vt) > 0.05])
    bad |= gaps
    bump = interval_mask(vt_us, [c['t_us'] for c in ev.get('bumps', [])], 0.1, 0.3)
    # ---- 指令(100 Hz)
    steer = np.full(len(vt), np.nan)
    thr = np.full(len(vt), np.nan)
    human = ~sel
    hv = human & V['valid'][:, 0].astype(bool) & V['valid'][:, 1].astype(bool)
    steer[hv] = V['pwm'][hv, 0]
    thr[hv] = V['pwm'][hv, 1]
    lat = None
    if A is not None and len(A['timestamps_us']):
        lat = A['latency_ms'].astype(float)
        lat_ok = np.isfinite(lat)
        med = float(np.median(lat[lat_ok])) if lat_ok.any() else 30.0
        t_eff = A['timestamps_us'].astype(np.int64) + (np.where(lat_ok, lat, med) * 1000).astype(np.int64)
        order = np.argsort(t_eff, kind='stable')
        t_eff = t_eff[order]
        out = A['out'][order].astype(float)
        # ラベルのフレーム(AI_CMD と同じ並び)から、使えない AI のフレーム(hold・補助・ガード・dry_run)を引く
        lt = lab['t_us'].astype(np.int64)
        ki = np.clip(np.searchsorted(lt, A['timestamps_us'][order]), 0, len(lt) - 1)
        same = lt[ki] == A['timestamps_us'][order]
        okf = same & ~lab['hold'][ki] & (lab['assist'][ki] <= 0) & ~lab['guard_applied'][ki] & ~lab['dry_run'][ki]
        k = np.searchsorted(t_eff, vt_us, side='right') - 1
        has = (k >= 0)
        kk = np.clip(k, 0, len(t_eff) - 1)
        fresh = has & ((vt_us - t_eff[kk]) <= 100_000) & okf[kk]
        av = sel & fresh
        steer[av] = out[kk[av], 0]
        thr[av] = out[kk[av], 1]
        res['latency_ms'] = {'n': int(lat_ok.sum()), 'p50': round(float(np.percentile(lat[lat_ok], 50)), 1) if lat_ok.any() else None,
                             'p90': round(float(np.percentile(lat[lat_ok], 90)), 1) if lat_ok.any() else None,
                             'p99': round(float(np.percentile(lat[lat_ok], 99)), 1) if lat_ok.any() else None,
                             'max': round(float(np.max(lat[lat_ok])), 1) if lat_ok.any() else None}
        cam_dt = np.diff(A['timestamps_us']) / 1e3
        res['ai_cmd_interval_ms'] = {'p50': round(float(np.median(cam_dt)), 1), 'p99': round(float(np.percentile(cam_dt, 99)), 1)}
    op = np.where(sel, 1, 0)                 # 0 = 人、1 = AI
    cmd_ok = np.isfinite(steer) & np.isfinite(thr)
    clean = cmd_ok & ~bad
    # ---- 区間(30 Hz のラベル → 100 Hz)
    lt = lab['t_us'].astype(np.int64)
    ki = np.clip(np.searchsorted(lt, vt_us), 1, len(lt) - 1)
    ki = np.where(np.abs(lt[ki - 1] - vt_us) < np.abs(lt[ki] - vt_us), ki - 1, ki)
    loc = np.where(np.abs(lt[ki] - vt_us) <= 100_000, lab['loc_code'][ki], 0).astype(int)
    vs = movavg(v, 9)
    acc = movavg(np.gradient(vs, vt), 9)
    center = np.where(op == 1, STEER_C_AI, STEER_C_HUMAN)
    off = steer - center
    rec = {'t': vt, 'op': op, 'clean': clean, 'loc': loc, 'v': v, 'vs': vs, 'acc': acc, 'gz': gz, 'steer': steer, 'thr': thr,
           'off': off, 'bump': bump, 'dt': dt, 'scene': V['scene'] if 'scene' in V else np.zeros(len(vt), int)}
    return res, rec


def summarize(res, r):
    dt = r['dt']
    names = ['human', 'ai']
    moving = r['clean'] & (r['v'] > 0.5)
    out = {}
    # 走っている秒数(操作者 × 区間)
    out['moving_s'] = {names[o]: {LOC[c]: round(float(((r['op'] == o) & moving & (r['loc'] == c)).sum() * dt), 1) for c in range(7)}
                       for o in (0, 1)}
    # 舵の帯 × 速さの帯(左右)
    st = {}
    for o in (0, 1):
        for side, sgn in (('left', 1), ('right', -1)):
            for lo, hi in STEER_BANDS:
                for vlo, vhi in SPEED_BANDS:
                    m = (r['op'] == o) & moving & (sgn * r['off'] >= lo) & (sgn * r['off'] < hi) & (r['v'] >= vlo) & (r['v'] < vhi)
                    st['%s|%s|%d-%d|%.1f-%.1f' % (names[o], side, lo, hi, vlo, vhi)] = round(float(m.sum() * dt), 2)
    out['steer_band_s'] = st
    # AI の張り付き(1000 / 2000)
    out['ai_saturated_s'] = {'left_2000': round(float(((r['op'] == 1) & moving & (r['steer'] >= 1999)).sum() * dt), 2),
                             'right_1000': round(float(((r['op'] == 1) & moving & (r['steer'] <= 1001)).sum() * dt), 2)}
    # 惰性(中立 ±12 を 0.3 秒以上)
    n3 = int(round(0.3 / dt))
    neutral = np.abs(r['thr'] - THR_N) <= 12
    coast = np.zeros(len(neutral), bool)
    for a, b in runs(neutral & r['clean']):
        if b - a + 1 >= n3:
            coast[a + n3 - 1:b + 1] = True
    coast &= r['vs'] > 1.0
    out['coast_s'] = {names[o]: {'straight': round(float(((r['op'] == o) & coast & (np.abs(r['off']) < 100)).sum() * dt), 2),
                                 'turning': round(float(((r['op'] == o) & coast & (np.abs(r['off']) > 250)).sum() * dt), 2)} for o in (0, 1)}
    # ブレーキの出来事
    br = []
    brake = (r['thr'] >= THR_N + 60) & np.isfinite(r['thr'])
    for a, b in runs(brake):
        if b - a + 1 < int(round(0.1 / dt)) or r['vs'][a] < 1.0:
            continue
        seg = slice(a, b + 1)
        dur = (b - a + 1) * dt
        v0, v1 = float(r['vs'][a]), float(r['vs'][b])
        dec = float(-np.mean(r['acc'][seg]))
        br.append({'t_s': round(float(r['t'][a]), 2), 'op': names[int(r['op'][a])], 'dur_s': round(dur, 2), 'v0': round(v0, 2), 'v1': round(v1, 2),
                   'mean_decel': round(dec, 2), 'peak_decel': round(float(-np.min(r['acc'][seg])), 2),
                   'thr_off_med': round(float(np.median(r['thr'][seg] - THR_N)), 0), 'steer_off_med': round(float(np.median(np.abs(r['off'][seg]))), 0),
                   'clean': bool(r['clean'][seg].all()), 'loc': LOC[int(r['loc'][a])]})
    out['brake_events'] = br
    # 再生の窓
    win = {}
    for W in (1.0, 2.0):
        n = int(round(W / dt))
        cnt = {}
        for o in (0, 1):
            for c in range(7):
                cnt[(o, c)] = 0
        for j0 in range(0, len(r['t']) - n, n):
            seg = slice(j0, j0 + n)
            if not r['clean'][seg].all() or r['v'][j0] < 0.5:
                continue
            o = r['op'][j0]
            if (r['op'][seg] != o).any():
                continue
            cnt[(int(o), int(r['loc'][j0]))] += 1
        win['%.0fs' % W] = {names[o]: {LOC[c]: cnt[(o, c)] for c in range(7)} for o in (0, 1)}
    out['replay_windows'] = win
    # 駆動の効き(シーンごと)
    n1 = int(round(1.0 / dt))
    thr = r['thr']
    sd = np.array([np.nanstd(thr[max(0, j - n1):j + 1]) for j in range(len(thr))])
    m = r['clean'] & (thr < THR_N - 60) & (sd < 10) & (np.abs(r['off']) < 120) & (np.abs(r['acc']) < 0.5) & (r['vs'] > 1.0)
    eff = []
    for s in np.unique(r['scene']):
        ms = m & (r['scene'] == s)
        if ms.sum() < 30:
            continue
        g = (THR_N - thr[ms]) / (r['vs'][ms] * 2.0)
        eff.append({'scene': int(s), 'n': int(ms.sum()), 'op': names[int(np.round(np.mean(r['op'][ms])))],
                    'g_us_per_kerpm': round(float(np.median(g)), 1), 'v_med': round(float(np.median(r['vs'][ms])), 2)})
    out['drive_efficiency'] = eff
    return out


def main():
    logs = sorted(os.path.basename(p) for p in glob.glob(str(JF / 'data' / 'raw' / '2026*')) if len(os.path.basename(p)) == 15)
    allres = []
    tot = {}
    brakes = []
    for log in logs:
        if not (LABELS / (log + '.json')).exists():
            allres.append({'log': log, 'skipped': 'ラベルが無い'})
            continue
        res, rec = analyse_log(log)
        if rec is not None:
            res.update(summarize(res, rec))
            for e in res['brake_events']:
                brakes.append(dict(e, log=log))
        allres.append(res)
        print(log, res.get('moving_s'), len(res.get('brake_events', [])), res.get('latency_ms'), flush=True)
    # 合計
    def add(d, path, x):
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = round(d.get(path[-1], 0) + x, 2)
    for r in allres:
        for o, dd in (r.get('moving_s') or {}).items():
            for c, x in dd.items():
                add(tot, ['moving_s', o, c], x)
        for k, x in (r.get('steer_band_s') or {}).items():
            add(tot, ['steer_band_s', k], x)
        for k, x in (r.get('ai_saturated_s') or {}).items():
            add(tot, ['ai_saturated_s', k], x)
        for o, dd in (r.get('coast_s') or {}).items():
            for c, x in dd.items():
                add(tot, ['coast_s', o, c], x)
        for W, dd in (r.get('replay_windows') or {}).items():
            for o, d2 in dd.items():
                for c, x in d2.items():
                    add(tot, ['replay_windows', W, o, c], x)
    lat_all = [r['latency_ms'] for r in allres if r.get('latency_ms')]
    tot['brake_events'] = {'n': len(brakes), 'n_clean': sum(b['clean'] for b in brakes),
                           'by_op': {o: sum(b['op'] == o for b in brakes) for o in ('human', 'ai')},
                           'clean_by_op': {o: sum(b['op'] == o and b['clean'] for b in brakes) for o in ('human', 'ai')}}
    OUT.write_text(json.dumps({'schema': 'jf-sim-calib-inventory/v1', 'logs': allres, 'totals': tot, 'brake_events': brakes,
                               'latency_logs': lat_all}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(tot, ensure_ascii=False)[:6000])


if __name__ == '__main__':
    main()
