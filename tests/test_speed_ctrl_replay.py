"""速さの天井・自動合わせを 10/24 の実車の記録で確かめる(2026-10-25。記録があるときだけ。無ければ skip)。
仕様: jetformer_reports/speed_control_20261025/spec.md 6.1 節の D-c2・D-o2(記録の入力の列)。実機・GPU・torch は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_speed_ctrl_replay.py

  - D-c2: 10/24 No.6〜10 の記録の入力(t・v・a・s・gz5・車に出したスロットル)を ceil_step に通すと、決まり(出力は入力以上・
    中立以下・前進でない入力はそのまま)が全フレームで成り立つ。走行ループのつなぎ(drive._SpeedCeil)に記録の時刻どおり
    バッファへ入れて回した値が、純粋な関数の値と全フレームで同じ
  - D-o2: 10/24 No.7・No.9 の記録の入力の列(予測・車両・IMU)で走行ループ(偽物)を回すと、OFF の出力・今の記録の欄が
    変更前の drive.py(写し)とビット単位で同じ。ON の出力は純粋な関数の値
"""
import collections
import glob
import json
import os

import numpy as np
import pytest

from jetformer import drive
from jetformer import safety_caps as sc
from jetformer.sensors.base import Sample

from test_speed_ctrl import T0, assert_before_fields_equal, load_before, run

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
LOGS = {6: '20261024-160536', 7: '20261024-160654', 8: '20261024-160847', 9: '20261024-162033', 10: '20261024-162305'}


def _chunks(log):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, log, log + '-*'))):
        ok = True
        for n in ('AI_CMD', 'VEHICLE', 'IMU'):
            try:
                with np.load(os.path.join(d, n + '.npz')) as z:
                    ok &= 'timestamps_us' in z.files and len(z['timestamps_us']) > 0
            except Exception:       # noqa: BLE001  壊れたチャンク(10/24 No.10 の後ろ)は使わない
                ok = False
        if ok:
            out.append(d)
    return out


def load(log):
    if not os.path.isdir(os.path.join(RAW, log)):
        pytest.skip('記録がありません: %s' % log)
    cat = collections.defaultdict(list)
    keys = {'AI_CMD': ('timestamps_us', 'out', 'select', 'dry_run', 'hold', 'model_pred'),
            'VEHICLE': ('timestamps_us', 'erpm', 'pwm', 'valid', 'select', 'moving'),
            'IMU': ('timestamps_us', 'accel', 'gyro')}
    for d in _chunks(log):
        for n, ks in keys.items():
            with np.load(os.path.join(d, n + '.npz')) as z:
                for k in ks:
                    cat[(n, k)].append(z[k])
    r = {k: np.concatenate(v) for k, v in cat.items()}
    for n in keys:
        o = np.argsort(r[(n, 'timestamps_us')], kind='stable')
        for k in keys[n]:
            r[(n, k)] = r[(n, k)][o]
    meta = json.load(open(os.path.join(RAW, log, 'log.json')))
    drv = meta.get('drive') or {}
    r['center'] = float(drv.get('steer_center_us') or 1441.0)
    r['neutral'] = float(drv.get('throttle_neutral_us') or 1496.0)
    r['trim'] = float((drv.get('params') or {}).get('steer_trim_us', 0.0) or 0.0)
    return r


def reference_inputs(r):
    """記録の配列から天井の入力(drive.speed_inputs と同じ決まりを、配列の添字で作り直したもの)"""
    at = r[('AI_CMD', 'timestamps_us')].astype(np.int64) * 1000
    vt = r[('VEHICLE', 'timestamps_us')].astype(np.int64) * 1000
    erpm = r[('VEHICLE', 'erpm')]
    it = r[('IMU', 'timestamps_us')].astype(np.int64) * 1000
    gz = [float(x) for x in np.degrees(np.deg2rad(r[('IMU', 'gyro')][:, 2]))]

    def med3(t):
        j = int(np.searchsorted(vt, t, side='right')) - 1
        if j < 0:
            return np.nan, None
        xs = [-float(erpm[k]) * sc.KV for k in range(j, max(j - 3, -1), -1)]
        while len(xs) < 3:
            xs.append(xs[-1])
        return sorted(xs)[1], int(vt[j])

    out = []
    for t in at:
        v, tv = med3(t)
        v_old, t_old = med3(t - 100_000_000)
        a = (v - v_old) / 0.1 if t_old is not None else np.nan
        k = int(np.searchsorted(it, t, side='right')) - 1
        if k < 0:
            g5 = np.nan
        else:
            xs = [gz[q] for q in range(k, max(k - 5, -1), -1)]
            while len(xs) < 5:
                xs.append(xs[-1])
            xs.reverse()
            g5 = sum(xs) / 5.0
        out.append((int(t), v, a, g5, tv))
    return out


@pytest.mark.parametrize('no', sorted(LOGS))
def test_pure_rules_and_glue_on_real_logs(no):
    r = load(LOGS[no])
    A = {k[1]: v for k, v in r.items() if isinstance(k, tuple) and k[0] == 'AI_CMD'}
    out = A['out'].astype(np.float64)
    active = A['select'].astype(bool) & ~A['dry_run'].astype(bool) & ~A['hold'].astype(bool)
    ref = reference_inputs(r)
    p = sc.ceil_params(drive.DEFAULT_PARAMS['ceil_all_mps'], drive.DEFAULT_PARAMS['ceil_left_mps'])   # 走行の既定(左 4.1)
    st = sc.CeilState()
    pure = []
    origin = ref[0][0]
    modes = collections.Counter()
    for i, (t, v, a, g5, tv) in enumerate(ref):
        thr = float(out[i, 1])
        fresh = tv is not None and t - tv < drive.VEH_FRESH_NS
        steer = float(out[i, 0]) - (r['center'] + r['trim'])
        st, o = sc.ceil_step(st, (t - origin) / 1e9, v, a, steer, g5, thr, r['neutral'], bool(active[i]) and fresh, p)
        assert o.thr_us >= thr and (o.thr_us <= r['neutral'] or o.thr_us == thr)
        if not thr < r['neutral']:
            assert o.thr_us == thr
        pure.append(o)
        modes[o.mode] += 1
    assert modes[sc.CEIL_PASS] > 100
    # 走行ループのつなぎ: 記録の時刻どおりバッファへ(撮像の少しあとの IMU も届いている)
    glue = drive._SpeedCeil(dict(drive.DEFAULT_PARAMS))
    ib, vb = collections.deque(maxlen=60), collections.deque(maxlen=50)
    I_t = r[('IMU', 'timestamps_us')].astype(np.int64) * 1000
    I_a, I_g = r[('IMU', 'accel')], r[('IMU', 'gyro')]
    V_t = r[('VEHICLE', 'timestamps_us')].astype(np.int64) * 1000
    V_e = r[('VEHICLE', 'erpm')]
    ki = kv = 0
    n_same = 0
    for i, (t, v, a, g5, tv) in enumerate(ref):
        while ki < len(I_t) and I_t[ki] <= t + 2_000_000:
            ib.append((int(I_t[ki]), np.concatenate([I_a[ki], np.deg2rad(I_g[ki])])))
            ki += 1
        while kv < len(V_t) and V_t[kv] <= t:
            vb.append((int(V_t[kv]), {'erpm': int(V_e[kv])}))
            kv += 1
        if A['hold'][i]:                                   # 中立で止めた行: どちらも状態を戻す(純粋な関数は active False)
            glue.hold()
            continue
        c = glue.frame(t, (float(out[i, 0]), float(out[i, 1])), r['center'] + r['trim'], r['neutral'], bool(active[i]),
                       ib, vb)
        assert c is not None, glue.failed
        for got, want in ((glue.inputs[0], ref[i][1]), (glue.inputs[1], ref[i][2]), (glue.inputs[3], ref[i][3])):
            assert got == want or (np.isnan(got) and np.isnan(want)), i
        assert (c.mode, c.thr_us, c.allow_us, c.cap) == (pure[i].mode, pure[i].thr_us, pure[i].allow_us, pure[i].cap) or (
            c.mode == pure[i].mode == sc.CEIL_OFF and c.thr_us == pure[i].thr_us), i
        n_same += 1
    assert n_same == int((~A['hold'].astype(bool)).sum())
    assert np.mean(glue.ms) < 1.0


# ---------------------------------------------------------------- 走行ループ(偽物)に記録の入力の列
def sc_from_log(r):
    """記録 → test_speed_ctrl.Feeder の入力(フレーム・IMU・車両)。時刻は合成の T0 に合わせて動かす"""
    A_t = r[('AI_CMD', 'timestamps_us')].astype(np.int64) * 1000
    shift = T0 + 150_000_000 - int(A_t[0])
    pred = r[('AI_CMD', 'model_pred')].astype(np.float64)
    frames = []
    for i in range(len(A_t)):
        p = pred[i]
        if not np.all(np.isfinite(p)):
            p = np.tile([[1441.0, 1496.0]], (10, 1))
        frames.append({'cam_t': int(A_t[i]) + shift, 'pred': p})
    I_t = r[('IMU', 'timestamps_us')].astype(np.int64) * 1000 + shift
    acc, gyr = r[('IMU', 'accel')], r[('IMU', 'gyro')]
    imu = [(int(I_t[k]), float(gyr[k, 2]), 0.0, float(acc[k, 0]), float(acc[k, 1])) for k in range(len(I_t))]
    V_t = r[('VEHICLE', 'timestamps_us')].astype(np.int64) * 1000 + shift
    veh = [(int(V_t[k]), {'pwm': [int(x) for x in r[('VEHICLE', 'pwm')][k]],
                          'valid': [bool(x) for x in r[('VEHICLE', 'valid')][k]],
                          'select': bool(r[('VEHICLE', 'select')][k]), 'moving': bool(r[('VEHICLE', 'moving')][k]),
                          'erpm': int(r[('VEHICLE', 'erpm')][k])}) for k in range(len(V_t))]
    return {'frames': frames, 'imu': imu, 'veh': veh}


PARAMS_B = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'steer_trim_us': 0.0, 'lookahead': 3, 'osc_percent': 0.0,
            'esc_reverse': 1, 'recovery': 1, 'rev_us': 300.0, 'rev_s': 0.4, 'tof_guard': 1, 'speed_cap': 0.0,
            'speed_scale': 1.0, 'tcp_mode': 0, 'ensemble_j': 0, 'ensemble_m': 0.5, 'zone_caps': 0, 'start_kick': 1}


@pytest.mark.parametrize('no', [7, 9])
def test_loop_on_real_log_off_identical_and_on_pure(no, monkeypatch, tmp_path):
    r = load(LOGS[no])
    sc_ = sc_from_log(r)
    before = load_before()
    _, pca0, rec0 = run(before, monkeypatch, tmp_path, PARAMS_B, sc_)
    _, pca1, rec1 = run(drive, monkeypatch, tmp_path, PARAMS_B, sc_)
    assert len(pca0.pairs) > 1000
    assert pca0.pairs == pca1.pairs and pca0.single == pca1.single
    assert_before_fields_equal(rec0.rows, rec1.rows)
    shadow = collections.Counter(x['ceil_mode'] for _, x in rec1.rows)
    # ON: 書いた値 = 純粋な関数の値。前進を強めない・舵は変えない
    calls = []
    real = sc.ceil_step

    def spy(*a, **kw):
        res = real(*a, **kw)
        calls.append((a, res))
        return res
    monkeypatch.setattr(sc, 'ceil_step', spy)
    d2, pca2, rec2 = run(drive, monkeypatch, tmp_path, dict(PARAMS_B, speed_ceiling=1), sc_)
    monkeypatch.setattr(sc, 'ceil_step', real)
    rows_on = [x for _, x in rec2.rows if not x['hold']]
    rows_off = [x for _, x in rec1.rows if not x['hold']]
    assert len(rows_on) == len(calls) == len(rows_off)
    n_app = 0
    for (args, (_, o)), a, b in zip(calls, rows_on, rows_off):
        assert real(*args)[1].thr_us == o.thr_us
        assert a['ceil_in_us'] == b['out'][1] and a['out'][0] == b['out'][0] and a['out'][1] >= b['out'][1]
        assert a['out'][1] == (o.thr_us if a['ceil_applied'] else b['out'][1])
        n_app += a['ceil_applied']
    print('No.%d: フレーム %d・影の出し方 %s・ON で出力を変えたフレーム %d(%.1f%%)' % (
        no, len(rows_on), dict(shadow), n_app, 100.0 * n_app / max(len(rows_on), 1)))
    assert d2.st['stats']['ceil_applied'] == n_app and d2.st['speed_ctrl_error'] is None
