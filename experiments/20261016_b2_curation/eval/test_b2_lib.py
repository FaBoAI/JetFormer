#!/usr/bin/env python3
"""b2_lib.py・b2_eval.py の試験(CPU・数秒)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q \
        experiments/20261016_b2_curation/eval/test_b2_lib.py
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import b2_lib as L  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import sim_cotrain as sc  # noqa: E402


def _summ(D_seeds, D_A, ci_lo=1.0, D_last=None):
    s = {'D': float(np.mean(list(D_seeds.values()))), 'D_seeds': D_seeds, 'ci95': [ci_lo, 99.0], 'D_A': D_A}
    if D_last is not None:
        s['D_last'] = D_last
    return s


def test_window_means_and_nan():
    out = np.array([[1400, 1300], [1450, 1310], [1500, 1320.0]])
    v = L.window_means(out, [[0, 1], [2], []], 1)
    assert v[0] == pytest.approx(1305.0) and v[1] == pytest.approx(1320.0) and np.isnan(v[2])


def test_boot_ci_deterministic():
    d = np.arange(25, dtype=float)
    a, b = L.boot_ci(d), L.boot_ci(d)
    assert a == b and a[0] < d.mean() < a[1]


def test_set_summary_seed_mean():
    s = L.set_summary({0: [10.0, 30.0], 1: [20.0, 40.0]}, {'A_s0': [1.0, 3.0]}, {0: [5.0, 5.0], 1: [15.0, 15.0]}, n_boot=200)
    assert s['D'] == pytest.approx(25.0)
    assert s['D_seeds'] == {'0': 20.0, '1': 30.0}
    assert s['D_A'] == {'A_s0': 2.0}
    assert s['D_last'] == pytest.approx(10.0)
    assert s['per_pass_mean_delta'] == [15.0, 35.0]


def test_gate_r1_held_parts():
    ok = _summ({'0': 32.0, '1': 34.0}, {'A_s0': 5.0, 'A_s1': 10.0}, 2.0, 25.0)
    assert L.gate_r1_held(ok)['pass']
    assert not L.gate_r1_held(_summ({'0': 25.0, '1': 34.0}, {'A_s0': 5.0}, 2.0, 25.0))['pass']       # D 29.5 < 30
    assert not L.gate_r1_held(_summ({'0': 19.0, '1': 45.0}, {'A_s0': 5.0}, 2.0, 25.0))['parts']['each_seed_ge_20']
    assert not L.gate_r1_held(_summ({'0': 32.0, '1': 34.0}, {'A_s0': 5.0}, -0.1, 25.0))['parts']['ci_low_gt_0']
    assert not L.gate_r1_held(_summ({'0': 32.0, '1': 34.0}, {'A_s0': 18.5}, 2.0, 25.0))['parts']['minus_maxA_ge_15']
    assert not L.gate_r1_held(_summ({'0': 32.0, '1': 34.0}, {'A_s0': 5.0}, 2.0, 19.9))['parts']['last_ge_20']
    assert not L.gate_r1_held(_summ({'0': 32.0, '1': 34.0}, {'A_s0': 5.0}, 2.0, None))['parts']['last_ge_20']


def test_gate_r1_train_and_slow():
    assert L.gate_r1_train(_summ({'0': 30.0, '1': 30.0}, {'A_s0': 15.0}))['pass']
    assert not L.gate_r1_train(_summ({'0': 30.0, '1': 30.0}, {'A_s0': 15.1}))['pass']
    assert L.gate_r1_slow({'D': -10.0})['pass'] and not L.gate_r1_slow({'D': -10.01})['pass']


def test_gate_de_threshold():
    st = {'D': -15.5, 'D_A': {'A_s0': 0.8, 'A_s2': -10.7}}
    th = {'D': -14.0, 'D_A': {'A_s0': 2.0}}
    g = L.gate_de(st, th)
    assert g['threshold_steer_us'] == pytest.approx(-15.7) and g['threshold_throttle_us'] == pytest.approx(-15.0)
    assert g['pass']
    assert not L.gate_de({'D': -15.8, 'D_A': {'A_s2': -10.7}}, th)['pass']


def test_gate_straight_and_collapse():
    g = L.gate_straight([200.0, 204.0], 212.0)
    assert g['drop'] == pytest.approx(1 - 202.0 / 212.0) and g['pass']
    assert not L.gate_straight([200.0, 201.0], 212.0)['pass']           # 5.4% の低下
    c = L.gate_no_collapse([-31.0, -28.0], [0.90, 0.92], 0.92)
    assert c['pass']                                                    # −29.5 ≥ −30、0.91 ≥ 0.91
    assert not L.gate_no_collapse([-31.0, -30.0], [0.90, 0.92], 0.92)['pass']
    assert not L.gate_no_collapse([0.0, 0.0], [0.90, 0.905], 0.92)['pass']


def test_compare_is_sim_cotrain_rule():
    arm = {'best': [50.5, 50.7], 'last': [50.9, 51.0]}
    ref = {'best': [49.146, 49.457], 'last': [49.235, 49.752]}
    c = L.compare(arm, ref, {'best': 0.313, 'last': 0.661})
    j = sc.judge_primary(arm['best'], ref['best'], 2.0, 0.313, 10.0)
    assert c['best'] == j and c['final_class'] == 'worse'
    assert c['worsening_best_us'] == pytest.approx(np.mean(arm['best']) - np.mean(ref['best']))
    # vs B(基準 1 本): ばらつき = spread_floor
    cb = L.compare({'best': [49.2, 49.3], 'last': [49.0, 49.1]}, {'best': [49.144], 'last': [49.091]},
                   {'best': 0.313, 'last': 0.661})
    assert cb['best']['threshold_us'] == pytest.approx(0.626) and cb['final_class'] == 'equivalent'


def _cmp(cls, worse):
    return {'final_class': cls, 'worsening_best_us': worse}


def test_gate_l1_and_reading():
    assert L.gate_l1(_cmp('worse', 1.5), _cmp('equivalent', 0.3))['pass']
    assert L.gate_l1(_cmp('worse', 3.0), _cmp('worse', 2.7))['pass']             # rest の悪化が帯の内
    assert not L.gate_l1(_cmp('worse', 3.5), _cmp('worse', 2.9))['pass']         # rest が帯の外
    assert not L.gate_l1(_cmp('worse', 10.0), _cmp('equivalent', 0.0))['pass']   # 実用上の差
    assert L.l1_reading(_cmp('equivalent', 0.2), _cmp('equivalent', 0.1)).startswith('同等')
    assert '種のばらつきの内' in L.l1_reading(_cmp('worse', 1.0), _cmp('equivalent', 0.1))
    assert '期待どおり' in L.l1_reading(_cmp('worse', 3.0), _cmp('equivalent', 0.5))
    assert L.l1_reading(_cmp('worse', 4.0), _cmp('worse', 3.0)).startswith('no-go')


def test_gate_intervention():
    assert L.gate_intervention(_cmp('worse', 5.8))['pass']
    assert not L.gate_intervention(_cmp('worse', 5.9))['pass']
    assert L.gate_intervention(_cmp('equivalent', 9.0))['pass']


def test_desk_go_and_genre():
    gates = {k: {'pass': True} for k in L.GATE_ORDER}
    assert L.desk_go(gates) == (True, [])
    gates['G3-straight'] = {'pass': False}
    assert L.desk_go(gates) == (False, ['G3-straight'])
    assert L.drive_genre(True, True) == 5 and L.drive_genre(False, True) == 4 and L.drive_genre(False, False) is None


def test_gate_engine():
    e = {'model_info': {'trained_modes': [4, 5], 'drivable': True, 'steer_center_us': 1441.0, 'throttle_neutral_us': 1496.0},
         'latency_ms': {'run_numpy_median': 2.5, 'run_numpy_p99': 3.1}, 'accuracy': {'actions_max_us': 1.2, 'finite': True},
         'peak_mem_mb': 2100.0, 'listed': True}
    assert L.gate_engine(e, 2.377, 5)['pass']
    assert not L.gate_engine(e, 1.9, 5)['parts']['median_le_B_plus_0p5']
    assert not L.gate_engine(dict(e, listed=False), 2.377, 5)['pass']
    assert not L.gate_engine(e, 2.377, 3)['pass']
    e2 = dict(e, accuracy={'actions_max_us': 2.6, 'finite': True})
    assert not L.gate_engine(e2, 2.377, 5)['pass']


def test_entry_estimate_matches_rl_evaluate():
    x = np.array([1300.0, 1340.0, 1400.0])
    y = np.array([5300.0, 4450.0, 3000.0])
    a = L.entry_estimate([6000.0, 5000.0], [1320.0, 1350.0], [1310.0, 1330.0], x, y)
    b = ev.entry_estimate(np.array([6000.0, 5000.0]), np.array([1320.0, 1350.0]), np.array([1310.0, 1330.0]), (x, y))
    assert np.allclose(a, b)


def test_drive_outputs_gain1_is_raw_k3():
    rng = np.random.default_rng(0)
    pred = 1450 + rng.normal(0, 60, size=(20, 10, 2))
    o = ev.drive_outputs(pred, (1441.0, 1496.0), 1.0, 1.0, 3, 0.0)
    assert np.allclose(o[:, 1], np.clip(pred[:, 2, 1], 1000, 2000)) and np.allclose(o[:, 0], np.clip(pred[:, 2, 0], 1000, 2000))


def test_eval_tool_jobs_and_key():
    import b2_eval as E
    k1 = E.table_key([0, 1], ['a', 'b'], [3, 4])
    assert k1 == E.table_key(np.array([0, 1]), np.array(['a', 'b']), np.array([3, 4])) and len(k1) == 40
    jobs = E.predict_jobs()
    need = [('B2_s0', 'pred', 'best', 'own'), ('B2_s1', 'pred', 'last', 'own'), ('B2_s0', 'pred', 'best', 'alignedrows'),
            ('B', 'win', 'best', 'g5'), ('A_s3', 'win', 'best', 'g5'), ('B2_s1', 'win', 'last', 'g5'),
            ('B2_s0', 'win', 'best', 'g4')]
    for j in need:
        assert j in jobs
    assert len(jobs) == len(set(jobs))


@pytest.mark.skipif(not os.path.exists('/home/jetson/jetformer/data/zoo/b2_curation_20261016/windows.json'),
                    reason='窓の表がまだ無い')
def test_real_window_counts_match_protocol():
    wj = json.load(open('/home/jetson/jetformer/data/zoo/b2_curation_20261016/windows.json'))
    proto = json.load(open('/home/jetson/jetformer_reports/b2_curation_20261016/protocol.json'))
    ce = proto['evaluation']['window_sets']['counts_expected']
    for kind in ('r1', 'de'):
        for st, want in ce[kind].items():
            es = [e for e in wj['entries'] if e['kind'] == kind and e['set'] == st]
            assert len(es) == want['n']
            assert sum(e['entry_erpm'] >= 5500 for e in es) == want['ge5500']
    assert len(wj['straight_rows']) == ce['straight_frames']
    held_fast = [e for e in wj['entries'] if e['kind'] == 'r1' and e['set'] in ('H', 'AI') and e['entry_erpm'] >= 5500]
    assert len(held_fast) == 25
    for e in wj['entries']:
        assert set(e['rows_w1']) <= set(e['rows_w2']) and set(e['rows_w04']) <= set(e['rows_w1'])
