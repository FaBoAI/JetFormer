#!/usr/bin/env python3
"""desk_lib.py の試験(numpy だけ・torch と GPU を使わない)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -m pytest -q \\
        /home/jetson/jetformer/experiments/20261021_bench_prep/desk/test_desk_lib.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import desk_lib as D  # noqa: E402


def _row(**kw):
    r = {'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 3.0, 'steer_amp_ratio': 1.0, 'l1_best_us': 50.0,
         'l1_steer_best_us': 51.0, 'brake_lowspeed_frac': 0.0}
    r.update(kw)
    return r


BASE = {'l1_best_us': 49.0, 'l1_steer_best_us': 50.0}


def test_gates_pass_and_each_rule():
    assert D.gates(_row(), BASE, 'arch') == (True, [])
    assert D.gates(_row(engine_ok=False), BASE, 'arch') == (False, ['engine_ok'])
    assert D.gates(_row(sanity_ok=False), BASE, 'arch') == (False, ['sanity_ok'])
    assert D.gates(_row(trt_p99_ms=10.0), BASE, 'arch')[0] is True          # ≤ 10.0 は合格
    assert D.gates(_row(trt_p99_ms=10.01), BASE, 'arch') == (False, ['trt_p99_ms'])
    assert D.gates(_row(steer_amp_ratio=0.80), BASE, 'arch')[0] is True
    assert D.gates(_row(steer_amp_ratio=1.25), BASE, 'arch')[0] is True
    assert D.gates(_row(steer_amp_ratio=0.79), BASE, 'arch') == (False, ['steer_amp_ratio'])
    assert D.gates(_row(steer_amp_ratio=1.26), BASE, 'arch') == (False, ['steer_amp_ratio'])
    # 5 番目: 差 < 10.0(ちょうど 10.0 は不合格)
    assert D.gates(_row(l1_best_us=58.99), BASE, 'arch')[0] is True
    assert D.gates(_row(l1_best_us=59.0), BASE, 'arch') == (False, ['l1_best_us'])


def test_gates_missing_and_nan():
    ok, why = D.gates(_row(engine_ok=None, trt_p99_ms=float('nan'), steer_amp_ratio=None, l1_best_us=None), BASE, 'robust')
    assert not ok
    assert why == ['missing:engine_ok', 'missing:trt_p99_ms', 'missing:steer_amp_ratio', 'missing:l1_best_us']
    r = _row()
    del r['sanity_ok']
    assert D.gates(r, BASE, 'ens') == (False, ['missing:sanity_ok'])
    assert D.gates(_row(), {}, 'ens') == (False, ['missing:B.l1_best_us'])


def test_gates_label_r1_amendment_a1():
    # SLOW: 全体の L1 が 14 µs 悪くても、ステアの L1 の差が 10 未満なら 5 番目は合格(A1)
    r = _row(l1_best_us=63.3, l1_steer_best_us=54.6, brake_lowspeed_frac=0.0)
    assert D.gates(r, BASE, 'label_r1') == (True, [])
    assert D.gates(r, BASE, 'arch') == (False, ['l1_best_us'])        # ほかの家族は今までどおり
    assert D.gates(_row(l1_steer_best_us=60.0), BASE, 'label_r1') == (False, ['l1_steer_best_us'])
    # 足した門: brake_lowspeed_frac ≤ 0.01
    assert D.gates(_row(brake_lowspeed_frac=0.01), BASE, 'label_r1')[0] is True
    assert D.gates(_row(brake_lowspeed_frac=0.0101), BASE, 'label_r1') == (False, ['brake_lowspeed_frac'])
    assert D.gates(_row(brake_lowspeed_frac=None), BASE, 'label_r1') == (False, ['missing:brake_lowspeed_frac'])
    assert D.gates(_row(brake_lowspeed_frac=0.5), BASE, 'weight')[0] is True     # label_r1 でなければ見ない


def test_family_by_run():
    assert D.family_by_run('bench_20261021-slow-s0', 'robust') == 'label_r1'
    assert D.family_by_run('bench_20261021-slow-s1', None) == 'label_r1'
    assert D.family_by_run('bench_20261021-wgt-s0', None) == 'weight'
    assert D.family_by_run('bench_20261021-aug-s0', None) == 'robust'
    assert D.family_by_run('bench_20261021-ens-p2', None) == 'ens'
    assert D.family_by_run('clean_driving_20261011-tcp-s0-nocmd-stride3-e30', 'arch') == 'arch'
    for cid, _, run, fam in D.CANDIDATES[1:]:
        assert D.family_by_run(run, fam) == fam


def test_l1_parts_and_nan_labels():
    pred = np.full((3, 10, 2), 1500.0)
    y = np.full((3, 10, 2), 1500.0)
    y[:, :, 0] += 10.0          # ステアだけ 10 µs ずれ
    y[2] = np.nan               # 正解が NaN の行は 0 として数える(realacc_lib.l1_rows と同じ nan_to_num)
    out = D.l1_parts(pred, y, np.array([True, True, False]))
    assert out == {'l1': 5.0, 'l1_steer': 10.0, 'l1_thr': 0.0, 'n': 2}
    assert D.l1_parts(pred, y, np.zeros(3, bool))['l1'] is None


def test_steer_amp_ratio():
    rng = np.random.default_rng(0)
    y = np.zeros((500, 10, 2))
    y[:, :, 0] = 1443 + rng.normal(0, 100, (500, 10))
    pred = y.copy()
    pred[:, :, 0] = 1443 + (y[:, :, 0] - 1443) * 0.9
    m = np.ones(500, bool)
    assert math.isclose(D.steer_amp_ratio(pred, y, m), 0.9, rel_tol=1e-9)
    assert math.isclose(D.steer_amp_ratio(pred, y, m, row=2), 0.9, rel_tol=1e-9)
    y0 = np.zeros((5, 10, 2))
    assert D.steer_amp_ratio(y0, y0, np.ones(5, bool)) is None


def test_pre_anchor_mask_and_thr_diff():
    t = np.arange(0, 5, 0.1)
    m = D.pre_anchor_mask(t, [2.0, 4.05])
    assert m[np.isclose(t, 1.0)].all() and m[np.isclose(t, 1.9)].all()
    assert not m[np.isclose(t, 2.0)].any()            # ta は入れない([ta − 1, ta))
    assert not m[np.isclose(t, 0.9)].any()
    assert m[np.isclose(t, 4.0)].all()
    pred = np.full((4, 10, 2), 1400.0)
    y = np.full((4, 10, 2), 1380.0)
    pred[:, 2, 1] = 1410.0                            # 先読み 3 の行(index 2)
    mm = np.array([True, True, False, False])
    assert D.thr_diff(pred, y, mm) == 30.0            # 正 = 人より緩める
    assert math.isclose(D.thr_diff(pred, y, mm, None), (9 * 20.0 + 30.0) / 10)
    assert D.thr_diff(pred, y, np.zeros(4, bool)) is None


def test_brake_lowspeed():
    u = np.array([1600.0, 1598.0, 1700.0, 1700.0, 1450.0])
    erpm = np.array([-1000.0, -1000.0, -3000.0, np.nan, 0.0])     # 0.5・0.5・1.5 m/s・不明・0
    v = D.speed_mps(erpm)
    assert np.allclose(v[:3], [0.5, 0.5, 1.5])
    assert D.brake_lowspeed(u, v) == (1, 3)               # 1598 ちょうどは数えない(> 1598)・1.5 m/s と NaN は分母に入れない
    assert D.brake_lowspeed(u, v, np.array([False, True, True, True, True])) == (0, 2)
    assert D.frac(1, 3) == 1 / 3 and D.frac(0, 0) is None


def test_engine_current_and_ok():
    c = {'engine': {'size': 10, 'mtime_ns': 5}, 'accuracy_ok': True, 'actions_max_us': 2.5}
    assert D.engine_current(c, 10, 5) and not D.engine_current(c, 11, 5) and not D.engine_current(c, 10, 6)
    assert not D.engine_current(None, 10, 5)
    assert D.engine_ok_of(c, True) is True
    assert D.engine_ok_of(c, False) is False
    assert D.engine_ok_of(dict(c, actions_max_us=2.51), True) is False
    assert D.engine_ok_of(dict(c, accuracy_ok=False), True) is False
    assert D.engine_ok_of(dict(c, actions_max_us=None), True) is False


def test_sanity_of():
    good = {'check_io_ok': True, 'frames': 100, 'nonfinite_frames': 0, 'out_of_range_values': 0}
    assert D.sanity_of(good) is True
    assert D.sanity_of(dict(good, nonfinite_frames=1)) is False
    assert D.sanity_of(dict(good, out_of_range_values=3)) is False
    assert D.sanity_of(dict(good, check_io_ok=False)) is False
    assert D.sanity_of(dict(good, frames=0)) is False
    assert D.sanity_of(dict(good, error='x')) is False
    assert D.sanity_of(None) is False


def test_impact_mask_matches_window():
    t_us = (np.arange(100) * 1e5).astype(np.int64)            # 0.1 秒おき
    imu = np.zeros((100, 10, 6), np.float32)
    imu[50, 3, 0] = 3.0 * 9.80665                             # 5.0 秒に 3 G
    m = D.impact_mask(t_us, imu, 2.5)
    t = t_us / 1e6
    assert m[(t >= 4.0) & (t <= 7.0)].all() and not m[(t < 3.95) | (t > 7.05)].any()


def test_anchor_ids_and_cluster_boot():
    t = np.arange(0, 10, 0.1)
    ids = D.anchor_ids(t, [2.0, 6.0], start=3)
    assert set(ids[(t >= 1.0) & (t < 1.95)]) == {3} and set(ids[(t >= 5.0) & (t < 5.95)]) == {4}
    assert (ids[t < 0.95] == -1).all()
    v = np.where(ids == 3, 1.0, np.where(ids == 4, 3.0, 99.0))
    mean, lo, hi, k = D.cluster_boot_ci(v, ids, n_boot=500, seed=0)
    assert k == 2 and math.isclose(mean, 2.0) and 1.0 <= lo <= mean <= hi <= 3.0
    assert D.cluster_boot_ci(v, np.full(len(v), -1)) == (None, None, None, 0)
