#!/usr/bin/env python3
"""2026-10-10 の全走行ログ(AI走行・人の走行)を生ログから集計する(読み取りのみ、CPU)

    nice -n 15 python runs_table.py > ../runs_table.json

定義(tools/drive_report.py / jetformer.laps.LapCounter / overnight runs/summary.json に合わせた):
  - AI側: AI_CMD.select(= その時の VEHICLE.select)。AI時間 = select=True のフレーム間隔の合計(0.1秒で打ち切り)
  - 衝突: IMU の水平 hypot(ax, ay) / g >= 2.5、0.6秒以内はまとめる。AI側かは VEHICLE.select で判定。強い衝突 >= 4G
  - 周回(方位): AI側・走行中(|eRPM|>200)の gyro z を積分(静止時の値でバイアス補正)÷ 360
  - LapCounter: jetformer.laps.LapCounter(drive_report.py と同じ入力)。ただしシーンの欠けで 1秒以上
    データが途切れたところで数え直す(途切れをまたいだ周のタイムが伸びないように)
  - 交代: AI区間の終わり(select True→False)のうち、ログの終わりでも基板の再起動でもないもの
  - 基板の再起動: VEHICLE.board_t_us が前のサンプルより小さくなった回数
  - 中立で止めた(HOLD): AI側で hold=True のフレームが始まった回数
"""

import json
import os
import sys
import datetime

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.laps import LapCounter  # noqa: E402

RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665
IMPACT_G = 2.5
MERGE_S = 0.6
GAP_RESET_S = 1.0


def load_log(log):
    root = os.path.join(RAW, log)
    meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
    scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    ok, broken, sj = [], [], {}
    for s in scenes:
        p = os.path.join(root, s, 'scene.json')
        try:
            sj[s] = json.load(open(p))
            ok.append(s)
        except Exception:
            broken.append(s)
    data = {}
    for k in ('AI_CMD', 'VEHICLE', 'IMU'):
        parts = []
        for s in ok:
            p = os.path.join(root, s, k + '.npz')
            if os.path.exists(p) and os.path.getsize(p) > 0:
                parts.append(dict(np.load(p, allow_pickle=True)))
        data[k] = parts
    nums = [int(s.split('-')[-1]) for s in scenes]
    missing = [n for n in range(1, max(nums) + 1) if n not in nums] if nums else []
    return meta, ok, broken, missing, sj, data


def cat(parts, key):
    parts = [p for p in parts if key in p and np.ndim(p[key]) > 0]
    return np.concatenate([p[key] for p in parts]) if parts else None


def impacts_from(imu_t, acc, veh_t, veh_sel):
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G
    ev = []
    for i in np.nonzero(hg >= IMPACT_G)[0]:
        t = imu_t[i]
        if ev and t - ev[-1]['t_us'] <= MERGE_S * 1e6:
            ev[-1]['g'] = max(ev[-1]['g'], float(hg[i]))
            continue
        ai = None
        if veh_t is not None and len(veh_t):
            k = int(np.searchsorted(veh_t, t, side='right')) - 1
            ai = bool(veh_sel[max(k, 0)])
        ev.append({'t_us': int(t), 'g': float(hg[i]), 'ai': ai})
    return ev, hg


def flag_events(ev, imu_t, hg, veh_t, veh_erpm):
    """各衝突に、走っていたか(直前0.5秒の |eRPM| 最大 >= 500)と、1サンプルだけの跳ね(±50ms に 1G 超が1つだけ)の印を付ける"""
    for e in ev:
        t = e['t_us']
        w = (imu_t >= t - 50000) & (imu_t <= t + 50000)
        e['spike'] = bool(np.sum(hg[w] > 1.0) < 2)
        m = (veh_t >= t - 500000) & (veh_t <= t)
        e['moving'] = bool(m.any() and np.max(np.abs(veh_erpm[m])) >= 500)
    return ev


def crossing_laps(imu_t, gz, veh_t, veh_erpm, veh_sel, ev, ai_only):
    """params.md と同じ「完全周」: 途切れのない区間(AI走行なら AI側が続く区間)の中で、方位が新しく 360度の倍数を
    越えた時刻の間隔を1周とする(区間の始まりからの最初の越えは発進を含むので数えない)。各周に衝突(>=2.5G)の数"""
    k = np.clip(np.searchsorted(veh_t, imu_t, side='right') - 1, 0, len(veh_t) - 1)
    erpm, sel = veh_erpm[k], veh_sel[k]
    still = (np.abs(erpm) < 50) & (np.abs(gz) < 5)
    bias = float(np.median(gz[still])) if still.sum() > 20 else 0.0
    t = imu_t / 1e6
    ev_t = np.array([e['t_us'] / 1e6 for e in ev]) if ev else np.zeros(0)
    laps = []
    h, cross = 0.0, []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        brk = dt > 0.3 or (ai_only and not sel[i])
        if brk:
            for a, b in zip(cross[:-1], cross[1:]):
                laps.append((a, b))
            h, cross = 0.0, []
            continue
        if abs(erpm[i]) > 200 and dt > 0:
            h += (gz[i] - bias) * dt
            if h >= 360.0 * (len(cross) + 1):
                cross.append(t[i])
    for a, b in zip(cross[:-1], cross[1:]):
        laps.append((a, b))
    out = []
    for a, b in laps:
        n = int(np.sum((ev_t >= a) & (ev_t < b)))
        out.append({'time_s': round(b - a, 2), 'impacts': n})
    times = [x['time_s'] for x in out]
    clean = [x['time_s'] for x in out if not x['impacts']]

    def st(v):
        if not v:
            return None
        return {'n': len(v), 'median': round(float(np.median(v)), 2), 'best': round(float(min(v)), 2),
                'iqr': [round(float(np.percentile(v, 25)), 2), round(float(np.percentile(v, 75)), 2)]}
    return {'all': st(times), 'no_impact': st(clean), 'laps': out}


def heading_laps(imu_t, gz, veh_t, veh_erpm, veh_sel, ai_side=True):
    """AI側(ai_side=True)または全区間(None)で、走行中の方位の積分 ÷ 360(左回り+)"""
    k = np.clip(np.searchsorted(veh_t, imu_t, side='right') - 1, 0, len(veh_t) - 1)
    erpm = veh_erpm[k]
    sel = veh_sel[k]
    still = (np.abs(erpm) < 50) & (np.abs(gz) < 5)
    bias = float(np.median(gz[still])) if still.sum() > 20 else 0.0
    t = imu_t / 1e6
    dt = np.diff(t)
    use = (dt > 0) & (dt <= 0.2) & (np.abs(erpm[1:]) > 200)
    if ai_side is True:
        use &= sel[1:]
    h = float(np.sum((gz[1:] - bias)[use] * dt[use]))
    return h / 360.0


def lapcounter(imu_t, gz, hg, veh_t, veh_erpm, veh_sel, force_ai=None):
    """drive_report.py と同じ入力で LapCounter を回す。1秒以上の途切れで数え直す"""
    laps, summaries = [], []
    lc = LapCounter()
    last = None
    seg = 0
    for ti, g, h in zip(imu_t / 1e6, gz, hg):
        if last is not None and ti - last > GAP_RESET_S:
            if lc.laps:
                summaries.append(lc.summary())
                laps.extend(dict(x, segment=seg) for x in lc.laps)
            lc = LapCounter()
            seg += 1
        last = ti
        k = max(int(np.searchsorted(veh_t / 1e6, ti, side='right')) - 1, 0)
        ai = bool(veh_sel[k]) if force_ai is None else force_ai
        lc.add(float(ti), float(g), float(veh_erpm[k]), ai, float(h))
    if lc.laps:
        laps.extend(dict(x, segment=seg) for x in lc.laps)
    clean = [x['time_s'] for x in laps if x['clean']]
    ai_all = [x['time_s'] for x in laps if x['ai_only'] and not x['first']]

    def med(v):
        return round(float(np.median(v)), 2) if v else None
    return {
        'laps': len(laps), 'segments': seg + 1,
        'clean_laps': len(clean), 'clean_median_s': med(clean), 'clean_best_s': min(clean) if clean else None,
        'clean_iqr_s': [round(float(np.percentile(clean, 25)), 2), round(float(np.percentile(clean, 75)), 2)] if len(clean) >= 4 else None,
        'ai_only_laps': len(ai_all), 'ai_only_median_s': med(ai_all),
        'lap_times': [{k: x[k] for k in ('n', 'segment', 'time_s', 'ai_only', 'takeovers', 'impacts', 'recoveries', 'holds', 'clean', 'first')}
                      for x in laps],
    }


def gains_table(ai):
    sel = ai['select'].astype(bool)
    t = ai['timestamps_us'] / 1e6
    dt = np.clip(np.diff(t, append=t[-1]), 0, 0.1)
    combos = {}
    for i in np.nonzero(sel)[0]:
        key = (round(float(ai['gains'][i, 0]), 2), round(float(ai['gains'][i, 1]), 2), int(ai['lookahead'][i]))
        combos[key] = combos.get(key, 0.0) + float(dt[i])
    rows = sorted(({'steer_gain': k[0], 'throttle_gain': k[1], 'lookahead': k[2], 'ai_s': round(v, 1)}
                   for k, v in combos.items()), key=lambda r: -r['ai_s'])
    return rows


def analyze(log):
    meta, ok, broken, missing, sj, data = load_log(log)
    ai = {k: cat(data['AI_CMD'], k) for k in ('timestamps_us', 'select', 'hold', 'gains', 'lookahead', 'latency_ms',
                                               'infer_ms', 'assist', 'guard', 'mode_id', 'raw', 'out')}
    veh = {k: cat(data['VEHICLE'], k) for k in ('timestamps_us', 'select', 'erpm', 'board_t_us', 'pwm')}
    imu = {k: cat(data['IMU'], k) for k in ('timestamps_us', 'accel', 'gyro')}
    first = min((sj[s]['start_us'] for s in ok), default=None)
    r = {'log': log, 'description': meta.get('description', ''), 'mode_id_log': meta.get('mode_id'),
         'start_local': datetime.datetime.fromtimestamp(first / 1e6).strftime('%H:%M:%S') if first else None,
         'scenes_present': len(ok), 'scenes_broken': [s[-4:] for s in broken], 'scene_numbers_missing': missing,
         'recorded_s': round(sum(sj[s]['duration_s'] for s in ok), 1),
         'scene_modes': sorted({int(sj[s]['mode_id']) for s in ok})}
    drive = meta.get('drive')
    if drive:
        r['model_run'] = drive.get('run')
        r['params_at_start'] = drive.get('params')
    vt, vsel, verpm = veh['timestamps_us'], veh['select'].astype(bool), veh['erpm'].astype(float)
    it, acc, gz = imu['timestamps_us'], imu['accel'], imu['gyro'][:, 2]
    # 基板の再起動
    bt = veh['board_t_us']
    r['board_reboots'] = int(np.sum(np.diff(bt) < 0)) if bt is not None else None
    # 衝突
    ev, hg = impacts_from(it, acc, vt, vsel)
    flag_events(ev, it, hg, vt, verpm)
    if ai['timestamps_us'] is not None and drive:
        at = ai['timestamps_us']
        sel = ai['select'].astype(bool)
        hold = ai['hold'].astype(bool)
        t = (at - at[0]) / 1e6
        dt = np.clip(np.diff(t, append=t[-1]), 0, 0.1)
        r['duration_s'] = round(float(t[-1]), 1)
        r['ai_time_s'] = round(float(dt[sel].sum()), 1)
        r['ai_ratio'] = round(float(dt[sel].sum() / max(t[-1], 1e-6)), 3)
        # AI区間と交代(基板の再起動で切れた区間は交代に数えない)
        ends = np.nonzero(sel[:-1] & ~sel[1:])[0]
        reboots_us = vt[1:][np.diff(bt) < 0] if bt is not None else np.zeros(0)
        starts = np.nonzero(~sel[:-1] & sel[1:])[0] + 1
        tk = []
        for i in ends:
            te = at[i + 1]
            nxt = [at[j] for j in starts if at[j] >= te]
            resumed = bool(nxt and nxt[0] - te < 0.5e6)       # 0.5秒以内にAI側へ戻った = 基板の再起動などの読み取りの切れ目
            near_reboot = bool(len(reboots_us) and np.min(np.abs(reboots_us - te)) < 0.5e6)
            tk.append({'t_s': round(float(t[i + 1]), 2), 'resumed_immediately': resumed, 'near_board_reboot': near_reboot})
        r['ai_segments'] = int(np.sum(~np.r_[False, sel[:-1]] & sel))
        r['takeovers'] = sum(1 for x in tk if not x['resumed_immediately'])
        r['takeovers_right_after_board_reboot'] = sum(1 for x in tk if x['near_board_reboot'] and not x['resumed_immediately'])
        r['segment_breaks_not_takeover'] = sum(1 for x in tk if x['resumed_immediately'])
        r['takeover_times_s'] = [x['t_s'] for x in tk if not x['resumed_immediately']]
        hs = hold & sel
        r['hold_events_ai'] = int(np.sum(~np.r_[False, hs[:-1]] & hs))
        r['hold_frames_ai'] = int(hs.sum())
        r['hold_s_ai'] = round(float(dt[hs].sum()), 1)
        lat = ai['latency_ms'][np.isfinite(ai['latency_ms'])]
        r['latency_ms_p50_p99'] = [round(float(np.median(lat)), 1), round(float(np.percentile(lat, 99)), 1)] if len(lat) else None
        inf = ai['infer_ms'][np.isfinite(ai['infer_ms'])]
        r['infer_ms_p50'] = round(float(np.median(inf)), 1) if len(inf) else None
        r['camera_fps'] = round(float((len(t) - 1) / t[-1]), 1)
        r['gain_settings_ai'] = gains_table(ai)
        r['gain_settings_count'] = len(r['gain_settings_ai'])
        if ai['assist'] is not None:
            a = ai['assist'].astype(int)
            on = lambda code: int(np.sum((a[1:] == code) & (a[:-1] != code)) + int(a[0] == code))
            r['assist_onsets'] = {'esc_reverse': on(1), 'recovery': on(2), 'giveup': on(3)}
        if ai['guard'] is not None:
            g = ai['guard'].astype(int)
            gs = (g >= 1) & sel
            r['tof_guard_onsets_ai'] = {'warn_or_brake': int(np.sum(~np.r_[False, gs[:-1]] & gs)),
                                        'brake': int(np.sum(~np.r_[False, ((g >= 2) & sel)[:-1]] & (g >= 2) & sel))}
        # 速度
        ve = np.abs(verpm[vsel])
        mv = ve[ve > 200]
        r['erpm_ai_moving_p50_p90'] = [int(np.median(mv)), int(np.percentile(mv, 90))] if len(mv) else None
        r['ai_t0_us'] = int(at[0])
        t0 = at[0]
    else:
        t0 = it[0]
        r['duration_s'] = round(float((it[-1] - it[0]) / 1e6), 1)
        ve = np.abs(verpm)
        mv = ve[ve > 200]
        r['erpm_moving_p50_p90'] = [int(np.median(mv)), int(np.percentile(mv, 90))] if len(mv) else None
    imp_ai = [e for e in ev if e['ai']]
    imp_h = [e for e in ev if e['ai'] is False]
    r['impacts_ai'] = len(imp_ai)
    r['impacts_ai_ge4g'] = sum(1 for e in imp_ai if e['g'] >= 4.0)
    r['impacts_human_side'] = len(imp_h)
    r['impacts_all'] = len(ev)
    r['impacts_all_ge4g'] = sum(1 for e in ev if e['g'] >= 4.0)
    filt = [e for e in ev if e['moving'] and not e['spike']]
    r['impacts_filtered_moving_nospike'] = {'all': len(filt), 'ai': sum(1 for e in filt if e['ai'])}
    r['impact_list'] = [{'t_s': round((e['t_us'] - t0) / 1e6, 2), 'g': round(e['g'], 1), 'ai': e['ai'],
                         'moving': e['moving'], 'spike': e['spike']} for e in ev]
    if drive:
        hl = heading_laps(it, gz, vt, verpm, vsel, True)
        r['ai_heading_laps'] = round(hl, 2)
        r['impacts_per_ai_heading_lap'] = round(len(imp_ai) / hl, 2) if hl > 0.5 else None
        r['impacts_per_ai_min'] = round(len(imp_ai) / (r['ai_time_s'] / 60), 1) if r['ai_time_s'] > 5 else None
        r['takeovers_per_ai_min'] = round(r['takeovers'] / (r['ai_time_s'] / 60), 1) if r['ai_time_s'] > 5 else None
        r['lapcounter'] = lapcounter(it, gz, hg, vt, verpm, vsel)
        r['crossing_laps_ai'] = crossing_laps(it, gz, vt, verpm, vsel, imp_ai, True)
    else:
        hl = heading_laps(it, gz, vt, verpm, vsel, None)
        r['heading_laps'] = round(hl, 2)
        r['impacts_per_heading_lap'] = round(len(ev) / hl, 2) if hl > 0.5 else None
        r['impacts_per_min'] = round(len(ev) / (r['recorded_s'] / 60), 2)
        # 人の走行: 運転者を「AI側」とみなして LapCounter を回す(clean = ぶつからず・最初の周でない)
        r['lapcounter_as_driver'] = lapcounter(it, gz, hg, vt, verpm, vsel, force_ai=True)
        r['crossing_laps'] = crossing_laps(it, gz, vt, verpm, vsel, ev, False)
        r['impacts_filtered_per_heading_lap'] = round(len(filt) / hl, 2) if hl > 0.5 else None
    return r


def main():
    logs = sys.argv[1:] or sorted(os.listdir(RAW))
    out = []
    for log in logs:
        try:
            out.append(analyze(log))
        except Exception as e:      # noqa: BLE001
            out.append({'log': log, 'error': repr(e)})
        print(log, file=sys.stderr)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
