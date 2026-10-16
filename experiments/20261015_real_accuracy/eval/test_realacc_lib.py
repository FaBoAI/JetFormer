#!/usr/bin/env python3
"""realacc_lib.py のテスト(numpy だけ。GPU なし)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q experiments/20261015_real_accuracy/eval/test_realacc_lib.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/tools')
import realacc_lib as L  # noqa: E402


def test_judge_same_as_sim_cotrain():
    import sim_cotrain as sc
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = 49 + rng.normal(0, 1.5, 2)
        r = 49 + rng.normal(0, 1.5, 2)
        fl = abs(rng.normal(0, 0.5))
        assert L.judge_primary(a, r, 2.0, fl, 10.0) == sc.judge_primary(a, r, 2.0, fl, 10.0)
    b, l_ = {'class': 'better'}, {'class': 'equivalent'}
    assert L.final_verdict(b, l_) == sc.final_verdict(b, l_) == 'equivalent'
    assert L.final_verdict(b, b) == 'better'
    assert L.final_verdict({'class': 'worse'}, b) == 'worse'


def test_judge_protocol_thresholds():
    # A = 49.146・49.457(best)、s_A = 0.313 → 閾値 0.626
    A = [49.146, 49.457]
    j = L.judge_primary([48.5, 48.6], A, 2.0, 0.313)
    assert j['class'] == 'better' and abs(j['threshold_us'] - 0.626) < 1e-9
    assert L.judge_primary([48.9, 48.95], A, 2.0, 0.313)['class'] == 'equivalent'     # 改善 0.38 < 0.626
    assert L.judge_primary([48.0, 49.2], A, 2.0, 0.313)['class'] == 'equivalent'     # 1 本が A の最小より上
    assert L.judge_primary([50.2, 50.3], A, 2.0, 0.313)['class'] == 'worse'
    c = L.compare({'best': [48.5, 48.6], 'last': [48.5, 48.6]}, {'best': A, 'last': [49.235, 49.752]},
                  {'best': 0.313, 'last': 0.661})
    assert c['final_class'] == 'equivalent'            # last の閾値 1.322 に届かない
    c = L.compare({'best': [48.0, 48.1], 'last': [48.0, 48.1]}, {'best': A, 'last': [49.235, 49.752]},
                  {'best': 0.313, 'last': 0.661})
    assert c['final_class'] == 'better'


def test_gain_correct_matches_probe_formula():
    rng = np.random.default_rng(1)
    p = 1498 + rng.normal(-150, 120, (50, 10, 2))
    gt = rng.uniform(20, 90, 50)
    gt[::7] = np.nan
    ref = np.where(rng.random(50) < 0.5, 36.41, 36.86)
    ref[3] = np.nan
    # probe_gain.py の correct(写し)
    q = p.copy()
    scale = np.clip(gt / ref, 0.8, 2.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    d = 1498.0 - q[:, :, 1]
    q[:, :, 1] = np.where(d > 0, 1498.0 - d * scale[:, None], q[:, :, 1])
    s2 = L.gain_scale(gt, ref)
    assert np.allclose(s2, scale)
    assert np.array_equal(L.gain_correct(p, s2), q)
    assert np.array_equal(L.gain_correct(p, s2)[:, :, 0], p[:, :, 0])      # ステアは変えない


def test_gain_label_inverse():
    rng = np.random.default_rng(2)
    y = 1498 + rng.normal(-150, 120, (40, 10, 2))
    r = rng.uniform(0.8, 2.0, 40)
    lab = L.gain_label(y, r)
    assert np.allclose(L.gain_correct(lab, r), y)                     # 後処理で元のラベルに戻る
    assert np.array_equal(L.gain_label(y, np.ones(40)), y)            # r = 1 なら元のまま
    assert np.array_equal(lab[:, :, 0], y[:, :, 0])
    back = y[:, :, 1] >= 1498
    assert np.array_equal(lab[:, :, 1][back], y[:, :, 1][back])       # 中立以上(後ろ・止まる)は変えない


def test_temporal_ensemble_matches_ensembler():
    from jetformer.policies.ensemble import TemporalEnsembler
    rng = np.random.default_rng(3)
    frames = np.array([0, 1, 2, 3, 5, 6, 7, 10, 11, 12])      # 抜けあり
    n = len(frames)
    part = np.zeros(n, int)
    scene = np.array(['s'] * n)
    pred = 1500 + rng.normal(0, 100, (n, 10, 2))
    prev = L.prev_index(part, scene, frames, 2)
    out = L.temporal_ensemble(pred, prev, 0.0)
    te = TemporalEnsembler()
    for k in range(n):
        if k > 0 and frames[k] - frames[k - 1] > 2:
            te.reset()          # 時刻で数えるので、3 フレーム以上離れた前の予測は使われない(同じ結果)
        got = te.push(int(round(frames[k] * 1e9 / 30)), pred[k], 2, 0.0)
        assert np.allclose(got, out[k], atol=1e-3), k
    # j_max = 0 は元のまま
    assert np.allclose(L.temporal_ensemble(pred, L.prev_index(part, scene, frames, 0)), pred)


def test_prev_index_scene_and_part():
    part = np.array([0, 0, 1, 1, 0])
    scene = np.array(['a', 'a', 'a', 'a', 'b'])
    frame = np.array([4, 5, 4, 5, 5])
    p = L.prev_index(part, scene, frame, 2)
    assert p[1].tolist() == [1, 0, -1] and p[3].tolist() == [3, 2, -1] and p[4].tolist() == [4, -1, -1]


def test_size_reading():
    sl = L.size_slopes(53.0, 51.0, 49.3, 1682, 3046, 5912)
    assert sl['s1'] == pytest.approx(2.0 / np.log2(3046 / 1682))
    assert sl['s2'] == pytest.approx(1.7 / np.log2(5912 / 3046))
    assert L.size_reading(sl) in ('middle', 'high_value')
    assert L.size_reading(L.size_slopes(50, 49.8, 49.3, 1682, 3046, 5912)) == 'low_value'      # L(H) − L(全部) ≤ 0.626
    hi = L.size_slopes(60, 54, 49.3, 1682, 3046, 5912)
    assert hi['G2_pow'] >= 2 and L.size_reading(hi) == 'high_value'
    neg = L.size_slopes(49, 51, 49.3, 1682, 3046, 5912)
    assert neg['G2_pow'] == 0.0


def test_scene_bootstrap_shapes():
    out = L.scene_bootstrap({'x': [10.0, 20.0, 30.0]}, [1, 2, 3], n_boot=50, seed=0)
    assert out['x'].shape == (50,)
    assert np.allclose(L.scene_bootstrap({'x': [10.0, 20.0]}, [1, 2], n_boot=20)['x'], 10.0)


def test_metric_values():
    y = np.full((4, 10, 2), 1500.0)
    p = y.copy()
    p[:, 2, 0] += 6
    p[0] += 1
    sets = {'all': np.ones(4, bool), 'first': np.array([1, 0, 0, 0], bool)}
    v = L.metric_values(p, y, sets, np.ones(4, bool))
    assert v['S2_k3_steer'] == pytest.approx((7 + 6 * 3) / 4)
    assert v['first'] == pytest.approx(1 + 6 / 20)
