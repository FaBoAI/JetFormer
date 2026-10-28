"""(反証 8-2 rehearsal: 8-1 の gen_course_r8.py の写しに recov(assist 2 = リカバリーの途中)と scene.json(本物の recorder と同じ欄)を足した。8-1 rehearsal の写し。もと experiments/20261026_fix/fix7/tools_r1/probes/gen_course.py。変えていない)
反証 7-1 回目・係 tools: 作り物の記録(距離で決めたコース。止まって・その場から走り直す形も作れる)。

tests/test_standing_3lap.write_standing_log と同じ形の npz(AI_CMD・VEHICLE・IMU・CAM_FRONT・log.json・events.jsonl)を書く。
違い: コースの曲がりを「距離」で決める(1 周 18 m。R1 右 180°・GM 左・SW 左・LH 左 = 左回り 360°。write_standing_log の 9 秒の周を
2 m/s で走ったときと同じ)。速さ v(t) は (時刻, 速さ) の点を線でつなぐ。止まっている間はヨーも 0。だから「コースの途中で止まり、
その場から AI 側で走り直す」記録が作れる(write_standing_log の時刻で決めたヨーではできない)。

    write_log(raw, log, dur, vkeys, ai_iv, ...)
      vkeys   [(秒, m/s)]。最初の点の前・最後の点のあとは端の値
      ai_iv   [(始め, 終わり)] CH3 が AI 側の間(VEHICLE の select・AI_CMD の select)
      stall   [(始め, 終わり)] カメラが止まった間(AI_CMD・CAM_FRONT のフレームが無い)
      giveup  [(始め, 終わり)] リカバリー中止(assist = 3・出力は中立)
      launch  [(始め, 長さ)] 発進の前進(launch_on・assist = 5・events.jsonl の start)
      hits    [(秒, 水平 G)]
      events  events.jsonl に足す行(dict)
置いた場所 = 距離 0。R1 の起点 = 5 m・23 m・41 m …(START_TO_R1 + 18k)。
"""
import json
from pathlib import Path

import numpy as np

REF = (1441.0, 1496.0)
T0_US = 1_800_000_000_000_000
G = 9.80665
KV = 0.5e-3
LAP_M = 18.0
START_TO_R1 = 5.0
# R1 の起点から: (長さ m, ヨー deg/m)。2 m/s で test_lap_anchor.LAP(秒, deg/s)と同じ
SEGS = [(1.8, -100.0), (0.2, 0.0), (3.0, 60.0), (3.0, 0.0), (4.0, 45.0), (3.0, 0.0), (3.0, 60.0)]
EDGES = np.cumsum([0.0] + [L for L, _ in SEGS])
KAPS = np.array([k for _, k in SEGS])


def kappa(s):
    """距離 s [m](置いた場所 = 0)のヨー [deg/m]"""
    x = np.mod(np.asarray(s, float) - START_TO_R1, LAP_M)
    i = np.clip(np.searchsorted(EDGES, x, side='right') - 1, 0, len(KAPS) - 1)
    return KAPS[i]


def in_iv(t, ivs):
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for a, b in ivs or ():
        m |= (t >= a - 1e-9) & (t < b - 1e-9)
    return m


def write_log(raw, log, dur, vkeys, ai_iv, params, description, stall=(), giveup=(), launch=(), hits=(), events=(),
              run='runB', scene_s=20.0, recov=()):
    d = Path(raw) / log
    d.mkdir(parents=True)
    meta = {'description': description, 'tags': ['ai-drive'], 'mode_id': 5,
            'drive': {'run': run, 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1], 'dry_run': False, 'params': params}}
    (d / 'log.json').write_text(json.dumps(meta), encoding='utf-8')
    tk, vk = np.array([x[0] for x in vkeys], float), np.array([x[1] for x in vkeys], float)
    tg = np.arange(0.0, dur + 0.5, 0.001)
    vg = np.interp(tg, tk, vk)
    sg = np.concatenate([[0.0], np.cumsum(0.5 * (vg[1:] + vg[:-1]) * np.diff(tg))])
    ag = np.gradient(vg, tg)
    ft = np.arange(0, dur, 1 / 30.0)
    vt = np.arange(0, dur, 0.01)
    it = np.arange(0, dur, 0.01) + 0.003
    v_v = np.interp(vt, tg, vg)
    s_i, v_i = np.interp(it, tg, sg), np.interp(it, tg, vg)
    gz = kappa(s_i) * v_i                                   # deg/s
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    acc[:, 1] = np.interp(it, tg, ag)
    for th, g in hits:
        k = int(np.searchsorted(it, th))
        acc[k:k + 3, 1] += [-0.5 * g * G, -g * G, -0.3 * g * G]
    e = -v_v / KV
    sel_v = in_iv(vt, ai_iv)
    keep = ~in_iv(ft, stall)
    sel_f = in_iv(ft, ai_iv)
    s_f = np.interp(ft + 0.1, tg, sg)
    kap = np.radians(kappa(s_f))                             # rad/m
    steer = REF[0] + np.clip(kap * 1e4 / 40.0, -480, 480)
    thr = np.full(len(ft), REF[1] - 200.0)
    assist = np.zeros(len(ft), np.int8)
    rc = in_iv(ft, recov)
    assist[rc] = 2
    thr[rc] = REF[1] + 150.0
    gu = in_iv(ft, giveup)
    assist[gu] = 3
    steer[gu] = REF[0]
    thr[gu] = REF[1]
    lon = np.zeros(len(ft), bool)
    lend = np.zeros(len(ft), np.int8)
    ev_lines = []
    for a0, ln in launch:
        m = (ft >= a0 - 1e-9) & (ft < a0 + ln - 1e-9)
        lon |= m
        assist[m] = 5
        thr[m] = REF[1] - 250.0
        k_end = int(np.searchsorted(ft, a0 + ln - 1e-9))
        if k_end < len(lend):
            lend[k_end] = 1
        ev_lines.append({'kind': 'launch_boost', 'event': 'start', 'us': 250.0, 'cam_t_us': int(T0_US + round(a0 * 1e6)),
                         't_us': int(T0_US + round(a0 * 1e6)) + 500})
    ev_lines += list(events)
    if ev_lines:
        (d / 'events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in ev_lines), encoding='utf-8')
    out = np.stack([steer, thr], axis=1)
    us = lambda x: (T0_US + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    nsc = int(np.ceil(dur / scene_s))
    for si in range(nsc):
        a, b = si * scene_s, min((si + 1) * scene_s, dur)
        s = d / ('%s-%04d' % (log, si + 1))
        s.mkdir()
        fm = (ft >= a) & (ft < b) & keep
        vm, im = (vt >= a) & (vt < b), (it >= a) & (it < b)
        n = int(fm.sum())
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), raw=out[fm].astype(np.float32), out=out[fm].astype(np.float32),
                 gains=np.tile([1.0, float(params.get('throttle_gain', 0.9))], (n, 1)).astype(np.float32),
                 trim=np.zeros(n, np.float32), lookahead=np.full(n, 3, np.int16), hold=np.zeros(n, bool), select=sel_f[fm],
                 dry_run=np.zeros(n, bool), assist=assist[fm], model_out=out[fm].astype(np.float32),
                 launch_on=lon[fm], launch_end=lend[fm])
        nv = int(vm.sum())
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=np.tile([1441, 1496, 2000], (nv, 1)).astype(np.int32),
                 valid=np.ones((nv, 3), bool), select=sel_v[vm], erpm=np.round(e[vm]).astype(np.int32),
                 board_t_us=(us(vt[vm]) - T0_US + 10_000_000))
        gyro = np.zeros((int(im.sum()), 3))
        gyro[:, 2] = gz[im]
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32), gyro=gyro.astype(np.float32))
        np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * n))
        # 本物の recorder._save_scene と同じ欄の scene.json(説明 = log.json の説明。8-2 で足した)
        cts = ft[fm]
        sj = {'token': 'tok-%s' % s.name, 'name': s.name, 'log_token': 'tok-%s' % log, 'log': log, 'description': description,
              'tags': ['ai-drive'], 'mode_id': 5, 'mode_name': '低速走行',
              'mode_segments': [{'mode_id': 5, 'start_s': 0.0, 'end_s': round(float(cts[-1] - cts[0]), 3) if n else 0.0}],
              'start_us': int(us(np.array([a]))[0]), 'end_us': int(us(np.array([b]))[0]), 'duration_s': round(b - a, 2),
              'complete': bool(b - a >= scene_s * 0.95), 'run_index': None, 'planned_s': None,
              'counts': {'CAM_FRONT': n, 'TOF_FRONT': n, 'IMU': int(im.sum()), 'VEHICLE': nv, 'AI_CMD': n},
              'rates': {'CAM_FRONT': 30.0, 'TOF_FRONT': 30.0, 'IMU': 100.0, 'VEHICLE': 100.0},
              'stats': {'camera_max_gap_ms': 33.3}, 'vehicle_quality': 'v2',
              'drive': {'run': run, 'dry_run': False, 'params': params}, 'created_at': '2026-11-03T08:00:00'}
        (s / 'scene.json').write_text(json.dumps(sj, ensure_ascii=False, indent=1), encoding='utf-8')
    return d


def launch_keys(t_ai, d_move, ramp, v=2.0, until=None):
    """止まった所から: t_ai + d_move に動き出し、ramp 秒で v m/s。until があればそこまで(そのあとは呼び手が足す)"""
    k = [(0.0, 0.0), (t_ai + d_move, 0.0), (t_ai + d_move + ramp, v)]
    if until is not None:
        k.append((until, v))
    return k


def time_at_distance(vkeys, s_target, dur):
    """vkeys の速さで走って距離 s_target に届く時刻"""
    tk, vk = np.array([x[0] for x in vkeys], float), np.array([x[1] for x in vkeys], float)
    tg = np.arange(0.0, dur + 0.5, 0.001)
    vg = np.interp(tg, tk, vk)
    sg = np.concatenate([[0.0], np.cumsum(0.5 * (vg[1:] + vg[:-1]) * np.diff(tg))])
    return float(np.interp(s_target, sg, tg))
