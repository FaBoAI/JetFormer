"""止まってから 3 周の時間(tools/standing_3lap.py、2026-10-26 fix2)の試験。実機・GPU 不要(作り物の記録)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -m pytest -q tests/test_standing_3lap.py

作り物の記録: 1 周 9 秒の左回り(tests/test_lap_anchor.py の course_profile)。R1 の起点 4・13・22・31・40 秒。
0〜t_ai 秒は人(CH3)、t_move 秒まで止まっていて、そこから ramp_s 秒で 2 m/s(1 周 18 m)。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_lap_anchor import REF, T0_US, course_profile  # noqa: E402

G = le.G
V = 2.0                     # 走りながらの速さ [m/s]


def write_standing_log(root, log, dur=45.0, t_ai=1.0, t_move=1.5, ramp_s=0.0, first_anchor=4.0, false_laps=(),
                       false_hit_g=6.0, hits=(), human_after=None, stop_after=None, params=None, launch=None,
                       imu_scale=1.0, zero_scene=False, description='ai_no=3 test', scene_s=20.0, run='runX',
                       moving_at_ai=False):
    """止まった状態から AI で走り出す記録。hits: [(時刻, 水平の G)]。human_after: この時刻から人(CH3)。
    stop_after: この時刻から止まる(eRPM 0・ヨー 0)。launch: (始め, 長さ, 終わりの番号) を AI_CMD の launch_on・
    launch_end・assist = 5 に入れ、events.jsonl に始めた出来事を書く。imu_scale: IMU の前の加速 = 車輪の加速 × これ
    (< 1 で空転)。zero_scene: 最後に 0 バイトの npz だけのシーンを足す。moving_at_ai: 人のうちから走っている"""
    d = Path(root) / log
    d.mkdir(parents=True)
    meta = {'description': description, 'tags': ['ai-drive'], 'mode_id': 5,
            'drive': {'run': run, 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1], 'dry_run': False,
                      'params': params if params is not None else {'throttle_gain': 0.9}}}
    (d / 'log.json').write_text(json.dumps(meta), encoding='utf-8')
    ft = np.arange(0, dur, 1 / 30.0)
    vt = np.arange(0, dur, 0.01)
    it = np.arange(0, dur, 0.01) + 0.003
    segs, anchors, false_at = course_profile(dur, first_anchor, false_laps)
    gz = np.zeros(len(it))
    for a, b, g in segs:
        gz[(it >= a) & (it < b)] = g

    def speed(x):
        if moving_at_ai:
            return np.full(np.shape(x), V)
        if ramp_s > 0:
            return np.clip((np.asarray(x) - t_move) / ramp_s, 0.0, 1.0) * V
        return np.where(np.asarray(x) >= t_move - 1e-9, V, 0.0)

    def accel(x):
        if moving_at_ai or ramp_s <= 0:
            return np.zeros(np.shape(x))
        x = np.asarray(x)
        return np.where((x >= t_move) & (x < t_move + ramp_s), V / ramp_s, 0.0)
    vv = speed(vt)
    if not moving_at_ai:
        gz[it < t_move] = 0.0
    if stop_after is not None:
        vv[vt >= stop_after] = 0.0
        gz[it >= stop_after] = 0.0
    e = -vv / le.KV
    sel_v = vt >= t_ai
    sel_f = ft >= t_ai
    if human_after is not None:
        sel_v &= vt < human_after
        sel_f &= ft < human_after
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    acc[:, 1] = imu_scale * accel(it)
    for ta in false_at:
        if false_hit_g is not None:
            k = int(np.searchsorted(it, ta + 0.02))
            acc[k:k + 3, 1] += [-0.5 * false_hit_g * G, -false_hit_g * G, -0.3 * false_hit_g * G]
    for th, g in hits:
        k = int(np.searchsorted(it, th))
        acc[k:k + 3, 1] += [-0.5 * g * G, -g * G, -0.3 * g * G]
    v_f = np.maximum(np.interp(ft, vt, vv), 0.1)
    kap = np.radians(np.interp(ft + 0.1, it, gz)) / v_f
    steer = REF[0] + np.clip(kap * 1e4 / 40.0, -480, 480)
    thr = np.full(len(ft), REF[1] - 200.0)
    out = np.stack([steer, thr], axis=1)
    assist = np.zeros(len(ft), np.int8)
    lon = np.zeros(len(ft), bool)
    lend = np.zeros(len(ft), np.int8)
    if launch is not None:
        a0, ln, code = launch
        lon = (ft >= a0 - 1e-9) & (ft < a0 + ln - 1e-9)
        assist[lon] = 5
        k_end = int(np.searchsorted(ft, a0 + ln - 1e-9))
        lend[k_end] = code
        out[lon, 1] = REF[1] - 250.0
        ev = {'kind': 'launch_boost', 'event': 'start', 'us': 250.0, 'cam_t_us': int(T0_US + round(a0 * 1e6)),
              't_us': int(T0_US + round(a0 * 1e6)) + 500}
        (d / 'events.jsonl').write_text(json.dumps(ev) + '\n', encoding='utf-8')
    nsc = int(np.ceil(dur / scene_s))
    us = lambda x: (T0_US + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    for si in range(nsc):
        a, b = si * scene_s, min((si + 1) * scene_s, dur)
        s = d / ('%s-%04d' % (log, si + 1))
        s.mkdir()
        fm, vm, im = (ft >= a) & (ft < b), (vt >= a) & (vt < b), (it >= a) & (it < b)
        n = int(fm.sum())
        extra = {}
        if launch is not None:
            extra = {'launch_on': lon[fm], 'launch_end': lend[fm]}
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), raw=out[fm].astype(np.float32), out=out[fm].astype(np.float32),
                 gains=np.tile([1.0, 0.9], (n, 1)).astype(np.float32), trim=np.zeros(n, np.float32),
                 lookahead=np.full(n, 3, np.int16), hold=np.zeros(n, bool), select=sel_f[fm], dry_run=np.zeros(n, bool),
                 assist=assist[fm], model_out=out[fm].astype(np.float32), **extra)
        nv = int(vm.sum())
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=np.tile([1441, 1496, 2000], (nv, 1)).astype(np.int32),
                 valid=np.ones((nv, 3), bool), select=sel_v[vm], erpm=np.round(e[vm]).astype(np.int32),
                 board_t_us=(us(vt[vm]) - T0_US + 10_000_000))
        gyro = np.zeros((int(im.sum()), 3))
        gyro[:, 2] = gz[im]
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32), gyro=gyro.astype(np.float32))
        np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * n))
    if zero_scene:
        s = d / ('%s-%04d' % (log, nsc + 1))
        s.mkdir()
        for nm in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT', 'TOF_FRONT'):
            (s / (nm + '.npz')).write_bytes(b'')
    return d, anchors, false_at


def analyze(tmp_path, log, **kw):
    return s3.analyze_log(log, tmp_path, tmp_path / 'no_labels', {'runX': 'B'}, kw.pop('line_ahead_m', 0.0))


def valid_rows(res):
    return [s for s in res['standing'] if s.get('to_r1_s') is not None]


# ------------------------------------------------------------------ 止まってから 3 周
def test_standing_three_laps_from_move_and_from_ai(tmp_path):
    log = '20261103-090000'
    _, anchors, _ = write_standing_log(tmp_path, log)
    assert anchors[:5] == [4.0, 13.0, 22.0, 31.0, 40.0]
    r = analyze(tmp_path, log)
    rows = valid_rows(r)
    assert len(rows) == 1
    s = rows[0]
    assert s['valid'] and s['method'] == 'anchor' and s['why'] is None
    assert s['t_ai'] == pytest.approx(1.0, abs=0.04) and s['t_move'] == pytest.approx(1.5, abs=0.011)
    assert s['d_pre_m'] == pytest.approx(5.0, abs=0.1)                     # 2 m/s × (4.0 − 1.5)
    assert [round(x, 1) for x in s['anchors']] == [4.0, 13.0, 22.0, 31.0]   # 偽でない起点 A1〜A4(5 つ目は使わない)
    # 3 周目の終わり = A4 の 5 m 手前 = 28.5 秒
    assert s['end_s'] == pytest.approx(28.5, abs=0.03)
    assert s['from_move_s'] == pytest.approx(27.0, abs=0.03)
    assert s['from_ai_s'] == pytest.approx(27.5, abs=0.05)
    assert s['laps_line_s'] == pytest.approx([9.0, 9.0, 9.0], abs=0.03)
    assert s['laps_anchor_s'] == pytest.approx([9.0, 9.0, 9.0], abs=0.03)
    assert s['to_a4_from_move_s'] == pytest.approx(29.5, abs=0.05)         # 参考: 3 周目の終わりの起点まで
    # 距離の 3 周(analysis_tt の作り方): t_ai から 3 × 18 m
    assert r['lap_dist_median_m'] == pytest.approx(18.0, abs=0.1)
    assert s['dist3_from_ai_s'] == pytest.approx(27.5, abs=0.05)
    assert s['collisions']['n'] == 0 and s['collisions_by_lap'] == [0, 0, 0]
    assert s['launch']['recorded'] is False and s['launch']['kick_frames'] == 0
    assert r['zero_byte'] == [] and r['labels'] == 'computed'


def test_ramp_launch_loses_time_against_flying(tmp_path):
    """1 秒で 0 → 2 m/s: 動いた(0.2 m/s = 1.6 秒)から A4 の d_pre 手前まで。走りながらより約 0.4 秒遅い"""
    log = '20261103-090100'
    write_standing_log(tmp_path, log, ramp_s=1.0)
    s = valid_rows(analyze(tmp_path, log))[0]
    assert s['t_move'] == pytest.approx(1.6, abs=0.011)
    assert s['d_pre_m'] == pytest.approx(3.99, abs=0.1)                    # 加速の 1 m + 2 m/s × 1.5 秒 − 0.01 m
    assert s['valid'] and s['from_move_s'] == pytest.approx(27.405, abs=0.03)
    assert s['from_ai_s'] == pytest.approx(28.005, abs=0.05)
    assert s['laps_line_s'][1:] == pytest.approx([9.0, 9.0], abs=0.03)
    assert s['laps_line_s'][0] == pytest.approx(9.405, abs=0.03)
    sl = s['launch_slip']
    assert sl['ratio'] == pytest.approx(1.0, abs=0.05) and 'note' not in s


def test_false_anchor_is_not_the_end_of_a_lap(tmp_path):
    """2 周目の SW の出口で壁に当たって右へ回る(偽の起点 19.5 秒・6G)。外さないと 3 周目の終わりの起点を取り違える"""
    log = '20261103-090200'
    _, anchors, false_at = write_standing_log(tmp_path, log, false_laps=(1,))
    assert anchors[:4] == [4.0, 13.0, 26.0, 35.0] and false_at == [pytest.approx(19.5)]
    r = analyze(tmp_path, log)
    assert r["r1_anchors_rejected"] == [pytest.approx(19.5, abs=0.06)]
    s = valid_rows(r)[0]
    assert [round(x, 1) for x in s['anchors']] == [4.0, 13.0, 26.0, 35.0]
    assert s['laps_anchor_s'] == pytest.approx([9.0, 13.0, 9.0], abs=0.03)
    assert s['valid'] and s['from_move_s'] == pytest.approx(31.0, abs=0.03)   # 35 − 2.5 − 1.5
    assert s['collisions']['n'] == 1 and s['collisions']['strong'] == 1 and s['collisions']['gmax'] >= 5.5
    assert s['collisions_by_lap'] == [0, 1, 0]
    # 距離の 3 周は 1 周の距離を 18 m のまま数えるので、遅い 2 周目(26 m)を短く数える(線の方法との違い)
    assert s['dist3_from_ai_s'] == pytest.approx(27.5, abs=0.05)


def test_a4_missing_uses_a3_plus_lap_distance(tmp_path):
    """3 周目の線を越えた(29.0 秒)あとで人に替えて止めた → A4 が無い。A3 + (1 周の距離 − d_pre)"""
    log = '20261103-090300'
    write_standing_log(tmp_path, log, human_after=29.3, stop_after=29.4)
    r = analyze(tmp_path, log)
    s = valid_rows(r)[0]
    assert len(s['anchors']) == 3
    assert s['method'] == 'A3+dist' and s['valid']
    assert s['from_move_s'] == pytest.approx(27.0, abs=0.05)


def test_human_before_the_line_is_not_valid(tmp_path):
    """3 周目の線(28.5 秒)の前に人に替えた → 数えない(valid = False・理由つき)"""
    log = '20261103-090400'
    write_standing_log(tmp_path, log, human_after=27.9)
    s = valid_rows(analyze(tmp_path, log))[0]
    assert not s['valid'] and '人側' in s['why']


def test_line_ahead_measures_from_crossing_the_line(tmp_path):
    """スタートの線が置いた場所の 1 m 先: 線を越えてから 3 周 = 走りながらの 27.0 秒(加速 1 秒・加速で 1 m)"""
    log = '20261103-090500'
    write_standing_log(tmp_path, log, ramp_s=1.0)
    s = valid_rows(analyze(tmp_path, log, line_ahead_m=0.99))[0]
    assert s['t_line0'] == pytest.approx(2.5, abs=0.02)
    assert s['from_line_s'] == pytest.approx(27.0, abs=0.03)
    assert s['from_move_s'] == pytest.approx(27.9, abs=0.03)


def test_moving_at_ai_is_not_a_standing_start(tmp_path):
    log = '20261103-090600'
    write_standing_log(tmp_path, log, moving_at_ai=True)
    r = analyze(tmp_path, log)
    assert r['standing'] == []
    assert r['flying']['n'] >= 1                                             # 走りながらの窓は出す


def test_slip_note(tmp_path):
    """IMU の加速が車輪の半分(空転)→ 空転の比 2 → note"""
    log = '20261103-090700'
    write_standing_log(tmp_path, log, ramp_s=1.0, imu_scale=0.5)
    s = valid_rows(analyze(tmp_path, log))[0]
    assert s['launch_slip']['ratio'] == pytest.approx(2.0, abs=0.1)
    assert '空転' in s['note']


# ------------------------------------------------------------------ 走りながらの 3 周
def test_flying_windows_best_and_clean(tmp_path):
    """周 4 本(4→13→22→31→40)。3 周の窓は 2 つ(どちらも 27.0 秒)。2 周目に 3G の接触 → 2 つとも「軽い接触だけ」"""
    log = '20261103-090800'
    write_standing_log(tmp_path, log, hits=[(16.0, 3.0)])
    r = analyze(tmp_path, log)
    F = r['flying']
    assert F['n'] == 2 and F['best_s'] == pytest.approx(27.0, abs=0.05)
    assert [w['kind'] for w in r['windows']] == ['軽い接触だけ', '軽い接触だけ']
    assert F['best_clean_s'] is None and F['n_le27'] == sum(1 for w in r['windows'] if w['total_s'] <= 27.0)
    s = valid_rows(r)[0]
    assert s['collisions']['n'] == 1 and s['collisions']['strong'] == 0 and s['collisions_by_lap'] == [0, 1, 0]
    log2 = '20261103-090900'
    write_standing_log(tmp_path, log2, hits=[(34.0, 5.0)])                 # 4 周目だけ強い衝突
    r2 = analyze(tmp_path, log2)
    assert [w['kind'] for w in r2['windows']] == ['きれい', '強い衝突あり']
    assert r2['flying']['best_clean_s'] == pytest.approx(27.0, abs=0.05)
    assert r2['flying']['best_clean']['t_start'] == pytest.approx(4.0, abs=0.06)


# ------------------------------------------------------------------ 設定・発進の前進の記録
def test_settings_and_launch_record(tmp_path):
    log = '20261103-091000'
    params = {'throttle_gain': 0.95, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1,
              'speed_auto': 1, 'auto_target_mps': 4.27, 'launch_boost': 1, 'launch_us': 250.0, 'launch_mps': 2.5,
              'launch_max_s': 1.2, 'zone_caps': 0, 'start_kick': 1}
    write_standing_log(tmp_path, log, params=params, launch=(1.0, 0.5, 1),
                       description='ai_no=7 day=2026-11-03 block=13 cand=EXP model=runX memo=x')
    r = analyze(tmp_path, log)
    S = r['settings']
    for k, v in params.items():
        assert S[k] == v
    assert S['model'] == 'B' and S['run'] == 'runX' and S['throttle_gain_frames'] == pytest.approx(0.9)
    assert r['settings_text'] == 'B・0.95・天井 4.1・自動 4.27・発進 250'
    assert r['cand'] == 'EXP' and r['ai_no'] == '7' and r['block'] == 13
    lr = valid_rows(r)[0]['launch']
    assert lr['recorded'] and lr['on_frames'] == 15 and lr['assist5_frames'] == 15
    assert lr['on_s'] == pytest.approx(0.5, abs=0.04)
    assert lr['end_code'] == 1 and lr['end_reason'] == '速さに届いた' and lr['end_s'] == pytest.approx(0.5, abs=0.04)
    assert lr['events'] == [[pytest.approx(1.0, abs=0.01), 'start', '']]
    assert '0.50 秒・速さに届いた' in s3.launch_text(lr)


def test_old_record_without_new_fields(tmp_path):
    """10/24 までの記録: 天井・自動合わせ・発進の前進の欄が無い → None(表は「—」で、OFF と同じ)"""
    log = '20261103-091100'
    write_standing_log(tmp_path, log, params={'throttle_gain': 0.9, 'steer_gain': 1.1})
    r = analyze(tmp_path, log)
    S = r['settings']
    assert S['speed_ceiling'] is None and S['launch_boost'] is None and S['auto_target_mps'] is None
    assert r['settings_text'] == 'B・0.90・ステア 1.10'
    assert s3.launch_text(valid_rows(r)[0]['launch']) == '—'


# ------------------------------------------------------------------ 0 バイト・壊れた記録
def test_zero_byte_npz_is_skipped(tmp_path):
    log = '20261103-091200'
    write_standing_log(tmp_path, log, zero_scene=True)
    r = analyze(tmp_path, log)
    zs = '%s-0004' % log
    assert r['zero_byte'] == ['%s/%s.npz' % (zs, n) for n in ('AI_CMD', 'CAM_FRONT', 'IMU', 'TOF_FRONT', 'VEHICLE')]
    assert any(zs in p and '読めません' in p for p in r['problems'])
    s = valid_rows(r)[0]
    assert s['valid'] and s['from_move_s'] == pytest.approx(27.0, abs=0.03)
    md = s3.to_markdown([r])
    assert '0 バイトで飛ばした npz' in md and zs in md


def test_day_logs_skip_unreadable_log_json_and_dry_run(tmp_path):
    write_standing_log(tmp_path, '20261103-091300')
    bad = tmp_path / '20261103-091400'
    bad.mkdir()
    (bad / 'log.json').write_bytes(b'')
    dry = tmp_path / '20261103-091500'
    dry.mkdir()
    (dry / 'log.json').write_text(json.dumps({'drive': {'run': 'x', 'dry_run': True}}), encoding='utf-8')
    hum = tmp_path / '20261103-091600'
    hum.mkdir()
    (hum / 'log.json').write_text(json.dumps({'description': '収集'}), encoding='utf-8')
    logs, skipped = s3.day_logs('2026-11-03', tmp_path)
    assert logs == ['20261103-091300']
    assert dict(skipped) == {'20261103-091400': 'log.json が読めない(0 バイトか壊れている)', '20261103-091500': 'dry run'}
    assert s3.day_logs('20261103', tmp_path, since='0914')[0] == []
    r = s3.analyze_log('20261103-091400', tmp_path, tmp_path / 'no_labels', {})
    assert r['error'] == 'log.json が読めない'


def test_main_writes_json_and_markdown(tmp_path):
    write_standing_log(tmp_path, '20261103-091700', description='ai_no=4 cand=EXP')
    pj, pm = tmp_path / 'o.json', tmp_path / 'o.md'
    rc = s3.main(['--day', '20261103', '--raw', str(tmp_path), '--labels', str(tmp_path / 'no_labels'),
                  '--bench', str(tmp_path / 'none.json'), '--json', str(pj), '--md', str(pm)])
    assert rc == 0
    j = json.loads(pj.read_text(encoding='utf-8'))
    assert j['schema'] == s3.SCHEMA and len(j['results']) == 1
    s = [x for x in j['results'][0]['standing'] if x.get('valid')][0]
    assert s['from_move_s'] == pytest.approx(27.0, abs=0.03)
    md = pm.read_text(encoding='utf-8')
    assert '| 4 | 20261103-091700 | runX・0.90 |' in md and '**27.00**' in md
    assert s3.main(['--day', '20261104', '--raw', str(tmp_path)]) == 1


# ------------------------------------------------------------------ 決まりの細かいところ(配列で直接)
def _arrays(anchors, heads, dur=45.0):
    t = np.arange(0, dur, 1 / 30.0)
    ai = t >= 1.0
    vt = np.arange(0, dur, 0.01)
    verpm = np.where(vt >= 1.5, -V / le.KV, 0.0)
    hd = np.interp(vt, [0.0] + list(anchors) + [dur], [0.0] + list(heads) + [heads[-1]])
    return t, ai, vt, hd, vt, verpm


def test_heading_change_outside_one_lap_is_not_valid():
    """起点 → 起点の向きの変化が 300〜420° でない(偽の起点の決まりで外れなかった半周など)→ valid = False"""
    A = [4.0, 13.0, 22.0, 31.0, 40.0]
    t, ai, it, hd, vt, verpm = _arrays(A, [0, 360, 640, 1000, 1360])          # 13 → 22 秒は 280°
    s = s3.standing_starts(t, ai, it, hd, vt, verpm, A, [], Lrun=18.0)[0]
    assert s['dpsi_anchor_deg'] == pytest.approx([360, 280, 360], abs=0.5)
    assert not s['valid'] and '280°' in s['why']
    t, ai, it, hd, vt, verpm = _arrays(A, [0, 360, 720, 1080, 1440])
    s = s3.standing_starts(t, ai, it, hd, vt, verpm, A, [], Lrun=18.0)[0]
    assert s['valid'] and s['from_move_s'] == pytest.approx(27.0, abs=0.02) and 'note' not in s


def test_too_fast_or_odd_distance_lap_gets_a_note_but_counts():
    """6.5 秒の周(13 m、中央値 18 m から 28% 短い)→ note に「確かめる」。valid はそのまま"""
    A = [4.0, 10.5, 19.5, 28.5, 37.5]
    t, ai, it, hd, vt, verpm = _arrays(A, [0, 360, 720, 1080, 1440])
    s = s3.standing_starts(t, ai, it, hd, vt, verpm, A, [], Lrun=18.0)[0]
    assert s['valid'] and s['laps_anchor_s'] == pytest.approx([6.5, 9.0, 9.0], abs=0.02)
    assert s['note'].startswith('確かめる: 6.50 秒') and '13.0 m' in s['note']
