"""オフライン強化学習の机上の評価(tools/rl_evaluate.py)の numpy の規則のテスト。GPU・データ不要(CPU、1秒未満)

    cd ~/jetformer && python -m pytest -q tests/test_rl_evaluate.py
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import rl_evaluate as ev  # noqa: E402
from jetformer import drive  # noqa: E402


def test_drive_outputs_matches_drive_apply_gains():
    """走行の出力の式が jetformer/drive.py の apply_gains と同じ(先読み 3 = 3番目の予測、ゲインは中心の周り、
    速度のゲインは前進側だけ、上下限で切る)"""
    rng = np.random.default_rng(0)
    pred = rng.uniform(1000, 2000, size=(50, 10, 2))
    pred[0, 2] = [1990, 1100]                     # ステアが上限を超える・前進側
    pred[1, 2] = [1440, 1700]                     # ブレーキ側(ゲインを掛けない)
    ref = (1442.0, 1498.0)
    p = {'steer_gain': 1.3, 'throttle_gain': 1.6, 'steer_trim_us': 5.0, 'lookahead': 3}
    got = ev.drive_outputs(pred, ref, 1.3, 1.6, 3, 5.0)
    for i in range(len(pred)):
        want = drive.apply_gains((pred[i, 2, 0], pred[i, 2, 1]), ref, p, 1000, 2000)
        assert got[i] == pytest.approx(want)
    assert got[1, 1] == pytest.approx(1700)


def test_drive_outputs_lookahead_is_clamped():
    pred = np.zeros((1, 10, 2))
    pred[0, :, 0] = np.arange(10) + 1500
    pred[0, :, 1] = 1498
    assert ev.drive_outputs(pred, (1500, 1498), 1.0, 1.0, 1)[0, 0] == 1500
    assert ev.drive_outputs(pred, (1500, 1498), 1.0, 1.0, 99)[0, 0] == 1509


def _frames_two_logs():
    """ログ L1(val)と L2(train)。30fps で 4秒ずつ"""
    t = (np.arange(120) * 1e6 / 30).astype(np.int64)
    log = np.array(['L1'] * 120 + ['L2'] * 120)
    t_us = np.concatenate([t + 1000, t + 5000])
    split = np.array(['val'] * 120 + ['train'] * 120)
    return log, t_us, split


def test_heldout_passes_keeps_only_val_windows():
    log, t_us, split = _frames_two_logs()
    passes = [{'log': 'L1', 't0_us': 1000, 'ta_s': 1.0, 'tg_s': 2.0},       # R1・DE とも val
              {'log': 'L2', 't0_us': 5000, 'ta_s': 1.0, 'tg_s': 2.0},       # train のログ
              {'log': 'L1', 't0_us': 1000, 'ta_s': 1.0, 'tg_s': None},      # DE が無い
              {'log': 'L3', 't0_us': 0, 'ta_s': 1.0, 'tg_s': 2.0}]          # フレームが無いログ
    r1, de = ev.heldout_passes(passes, log, t_us, split)
    assert [p['pass_id'] for p in r1] == [0, 2]
    assert [p['pass_id'] for p in de] == [0]
    # R1 の窓 = 起点の 0.4秒手前〜起点(30fps で 13フレーム)、DE の窓 = 出口〜0.8秒後(25フレーム)
    assert len(r1[0]['rows']) == 13 and r1[0]['rows'][-1] == 30
    assert len(de[0]['rows']) == 25 and de[0]['rows'][0] == 60


def test_heldout_passes_drops_window_touching_train():
    log, t_us, split = _frames_two_logs()
    split = split.copy()
    split[25] = 'train'                            # R1 の窓の中の1フレームが train
    r1, _ = ev.heldout_passes([{'log': 'L1', 't0_us': 1000, 'ta_s': 1.0, 'tg_s': None}], log, t_us, split)
    assert r1 == []


def test_straight_mask():
    loc = np.array([ev.LOC_NAMES.index(x) for x in ('HS', 'DE', 'R1', 'HS', 'HS', 'HS')])
    gz = np.array([5, -10, 0, 30, 0, 0])
    erpm = np.array([-5000, -5000, -5000, -5000, -1000, -5000])
    near = np.array([0, 0, 0, 0, 0, 1], bool)
    assert ev.straight_mask(loc, gz, erpm, near).tolist() == [True, True, False, False, False, False]


def test_bootstrap_mean_interval_contains_mean():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    m, lo, hi = ev.bootstrap_mean(d, n_boot=500)
    assert m == pytest.approx(2.5) and lo <= m <= hi
    assert ev.bootstrap_mean(np.ones(5) * 3)[1:] == [3.0, 3.0]
    assert ev.bootstrap_mean([]) == [None, None, None]


def test_cluster_bootstrap_ratio_of_sums():
    x = np.array([1, 1, 2, 2, 3.0])
    r = np.array([1, 1, 1, 1, 1.0])
    cl = np.array([0, 0, 1, 1, 2])
    point, lo, hi = ev.cluster_bootstrap({'x': x, 'r': r}, cl, lambda s, c: s['x'] / s['r'], n_boot=300)
    assert point == pytest.approx(9 / 5)
    assert lo <= point <= hi
    # 平均(和 / 数)
    mean = ev.cluster_bootstrap({'d': x}, cl, lambda s, c: s['d'] / c, n_boot=300)
    assert mean[0] == pytest.approx(x.mean())


def test_steady_samples_needs_constant_window():
    u = np.concatenate([np.full(20, 1400.0), np.linspace(1400, 1300, 20)])
    e = np.full(40, 4000.0)
    ok = np.ones(40, bool)
    x, y = ev.steady_samples(u, e, ok, np.zeros(40), win=15, max_range_us=20)
    # 一定の区間で終わる窓 6個 + 傾き(1フレーム 5.3µs)に入って幅が 20µs 以内の窓 4個
    assert len(x) == 6 + 4
    assert np.all(y == 4000)
    ok[16] = False
    x2, _ = ev.steady_samples(u, e, ok, np.zeros(40), win=15, max_range_us=20)
    assert len(x2) < len(x)


def test_fit_erpm_map_is_monotone_and_interpolates():
    rng = np.random.default_rng(1)
    u = rng.uniform(1250, 1450, 5000)
    e = 4000 + (1400 - u) * 30 + rng.normal(0, 50, 5000)
    e[(u > 1300) & (u < 1310)] += 3000           # 外れたビンがあっても単調にする
    emap = ev.fit_erpm_map(u, e, bin_us=10, min_n=30)
    assert np.all(np.diff(emap[0]) > 0) and np.all(np.diff(emap[1]) <= 0)
    assert ev.erpm_from_throttle(emap, 1400) == pytest.approx(4000, abs=200)
    assert ev.erpm_from_throttle(emap, 2000) == emap[1][-1]       # 表の外は端の値


def test_entry_estimate_shifts_by_map_difference():
    emap = (np.array([1300.0, 1400.0]), np.array([7000.0, 4000.0]))     # 1µs 遅く = 30 eRPM 下がる
    est = ev.entry_estimate([6000.0], [1350.0], [1330.0], emap)
    assert est[0] == pytest.approx(6000 - 20 * 30)


def test_rolling_sum_per_group():
    x = np.arange(6, dtype=float)
    g = np.array([0, 0, 0, 1, 1, 1])
    r = ev.rolling_sum(x, g, 2)
    assert np.isnan(r[0]) and r[1] == 1 and r[2] == 3 and np.isnan(r[3]) and r[4] == 7 and r[5] == 9


def test_judges():
    assert ev.judge_r1([3.0, 0.5, 6.0], 0.8, 0.7)
    assert not ev.judge_r1([3.0, -0.5, 6.0], 0.8, 0.7)           # 区間が 0 を含む
    assert not ev.judge_r1([3.0, 0.5, 6.0], 0.7, 0.7)            # 5500 未満の割合が上がらない
    assert ev.judge_de([2.0, 0.1, 4.0]) and not ev.judge_de([2.0, None, None])
    ok, parts = ev.judge_collapse(0.04, 1.05, -10)
    assert ok and all(parts.values())
    ok, parts = ev.judge_collapse(0.08, 1.05, -40)
    assert not ok and parts == {'straight': False, 'l1': True, 'fwd_bias': False}
    ok, parts = ev.judge_labels({'spin': {'n': 10, 'good_share': 0.50, 'ref_good_share': 0.35},
                                 'scrape': {'n': 0, 'good_share': None, 'ref_good_share': 0.35},
                                 'human_reverse': {'n': 10, 'good_share': 0.05, 'ref_good_share': 0.35}})
    assert not ok and parts == {'spin': False, 'scrape': True, 'human_reverse': True}
    ok, parts = ev.judge_labels({'human_reverse': {'n': 10, 'good_share': 0.2, 'ref_good_share': 0.35}})
    assert not ok


def test_genres_map_and_substitute():
    g, sub = ev._genres([4, 5, 0, 5], {5: 6, 4: 7}, [4, 5, 6, 7])
    assert g.tolist() == [7, 6, 4, 6] and sub.tolist() == [False, False, True, False]
    g, sub = ev._genres([4, 5], None, [5])
    assert g.tolist() == [5, 5] and sub.tolist() == [True, False]
