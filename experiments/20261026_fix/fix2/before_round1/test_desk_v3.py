#!/usr/bin/env python3
"""desk_v3.py・eval_v3.py の試験(pytest。numpy だけ。torch・GPU を読まない。車・装置・走行の API に触れない)。

    cd /home/jetson/jetformer/experiments/20261026_fix/fix2
    /home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 600 -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B \\
        -m pytest -q -p no:cacheprovider --basetemp=/home/jetson/jetformer/experiments/20261026_fix/fix2/tmp_desk/pytest test_desk_v3.py
    rm -rf /home/jetson/jetformer/experiments/20261026_fix/fix2/tmp_desk

照らす相手(写しではなく本物): drive.apply_gains(AST で関数だけ)・gpu_queue_v3.py(run_dir・物差しの出力の名前・
predicted の約束)・protocol_v3.json / .md の数・protocol_numbers_v3.json(段 0 の 8)・protocol の自己試験。
書くのは pytest の tmp(--basetemp)の下だけ。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
C3 = Path('/home/jetson/jetformer/experiments/20261026_fix/c3')
sys.path.insert(0, str(C3))
import desk_v3 as D  # noqa: E402
import eval_v3 as E  # noqa: E402

REPO = Path('/home/jetson/jetformer')
FIX = REPO / 'experiments' / '20261026_fix'
PROTO = json.loads(D.PROTOCOL.read_text(encoding='utf-8'))
PROTO_MD = (D.REP3 / 'protocol_v3.md').read_text(encoding='utf-8')
T = E.thresholds(PROTO)
PROTECTED = [D.PROTOCOL, D.PROTOCOL_SHA, D.NUMBERS, D.REP3 / 'protocol_v3.md', D.PROTOCOL2, D.SETS_NPZ,
             FIX / 'gpu_queue_v3.py', FIX / 'gpu_queue_v3.sh', FIX / 'gpu_queue_v3.tsv', REPO / 'tools' / 'train.py',
             REPO / 'jetformer' / 'train_data.py', REPO / 'jetformer' / 'dataset.py', D.PROBE_SET, D.PROBE_CHECK]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


BEFORE = {str(p): sha(p) for p in PROTECTED + [D.DESK_JSON]}          # decisions_v3.json・lock2 はほかの係が書く


def func_from(path, fname):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname]
    ns = {'np': np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    return ns[fname]


def queue_module():
    spec = importlib.util.spec_from_file_location('gq3_for_test', str(FIX / 'gpu_queue_v3.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 1. 式

def test_car_out_matches_drive_apply_gains():
    apply_gains = func_from(REPO / 'jetformer' / 'drive.py', 'apply_gains')
    rng = np.random.default_rng(0)
    raw = np.stack([rng.uniform(900, 2100, 500), rng.uniform(900, 2100, 500)], 1)
    for ref in ((1433.0, 1500.0), (1441.0, 1496.0)):
        for g in (0.90, 1.00):
            want = np.array([apply_gains(r, ref, {'steer_trim_us': 0.0, 'steer_gain': 1.0, 'throttle_gain': g}, 1000.0, 2000.0)
                             for r in raw])
            assert np.allclose(D.car_out(raw, ref, g), want, atol=1e-9)


def test_back_and_go_frac():
    ref = (1440.0, 1496.0)
    p = np.zeros((6, 10, 2))
    p[:, 0, 1] = [1496 + 31, 1496 + 30, 1496 + 100, 1496 - 31, 1496 - 40, 1496]   # 後ろ側 2 つ(> +30)・前 2 つ(< −30 × 0.9 → −27.9・−36)
    m = np.ones(6, bool)
    assert D.back_frac(p, ref, m) == pytest.approx(2 / 6)
    # 前側はゲイン 0.90 のあと: −31 → −27.9(前へ出さない)・−40 → −36(出す)
    assert D.go_frac(p, ref, m) == pytest.approx(1 / 6)
    assert D.back_frac(p, ref, np.zeros(6, bool)) is None


def synth_sets(n=40, seed=1):
    rng = np.random.default_rng(seed)
    S = {k: np.zeros(n, bool) for k in ('pre_r1', 'pre_de', 'pre_sw', 'de_pass', 'straight', 'stand')}
    S['straight'][:20] = True
    S['sess'] = np.array([0] * 10 + [1] * 10 + [2] * 20, np.int8)
    S['stand'][20:30] = True
    S['pre_de'][30:35] = True
    pb = rng.uniform(1300, 1600, (n, 10, 2))
    return S, pb


def test_row_fields_split_by_session():
    S, pb = synth_sets()
    ref = (1433.0, 1500.0)
    pc = pb.copy()
    pc[:10, D.ROW2, 1] = 1550.0                       # 10/10 の直線だけ後ろ側へ
    pb2 = pb.copy()
    pb2[:10, D.ROW2, 1] = 1450.0
    r = D.row_fields(pc, pb2, ref, ref, S)
    want10 = float(np.mean(D.car_out(pc[:10, D.ROW2], ref, 0.9)[:, 1] - D.car_out(pb2[:10, D.ROW2], ref, 0.9)[:, 1]))
    assert r['straight_thr_vs_b_us_1010'] == pytest.approx(want10)
    assert r['straight_thr_vs_b_us_1023'] == pytest.approx(0.0)
    assert r['straight_thr_vs_b_us'] == pytest.approx(want10 / 2)
    assert r['n'] == {'straight': 20, 'straight_1010': 10, 'straight_1023': 10, 'stand': 10}
    rb = D.row_fields(pb, pb, ref, ref, S)
    assert rb['straight_thr_vs_b_us'] == 0 and rb['de_pre_thr_vs_b_us'] == 0


def test_probe_from_tab_formula():
    pb = np.full((10, 10, 2), 1500.0)
    pc = pb.copy()
    pc[0:4, 2, 1] += 8.0
    pc[4:8, 0, 1] = [1529, 1528, 1465, 1466]
    rows = {'st10': np.arange(0, 2), 'st23': np.arange(2, 4), 'stand': np.arange(4, 8)}
    r = D.probe_from_tab(pc, pb, rows)
    assert r['st10_thr_vs_b_us'] == pytest.approx(8.0) and r['st23_thr_vs_b_us'] == pytest.approx(8.0)
    assert r['stand_brake_side_frac'] == pytest.approx(0.25)       # 1529 > 1528 だけ
    assert r['stand_go_frac'] == pytest.approx(0.25)               # 1465 < 1466 だけ
    assert r['low10_brake_frac'] is None
    assert D.low10_of([1598.0, 1598.5, 1700, 1500]) == pytest.approx(0.5)


# ---------------------------------------------------------------- 2. 閾値・門・拒否権(protocol の文と照らす)

def test_thresholds_are_protocol_numbers():
    assert (T['g1010'], T['g1023'], T['g_back']) == (10.0, 12.138, 0.064)
    assert (T['p_st10'], T['p_st23'], T['p_stand'], T['p_low10']) == (11.111, 14.123, 0.064, 0.01)
    assert T['iv_steer_thr']['vs5_n4'] == 4.111 and T['iv_steer_thr']['vs5_n3'] == 4.475 and T['iv_steer_thr']['vs5_n2'] == 5.127
    assert T['iv_steer_mean_protocol'] == pytest.approx(158.3852)
    assert (T['fix_st23_k4'], T['fix_back_k4']) == (7.835, 0.0241)
    assert T['human_throttle'] == pytest.approx(15.964)
    assert T['vs5']['n4']['steer_bias_best'] == 3.913
    for s in ('**4.111**', '±10.0', '**±12.138**', '**≤ 0.0640**', '**±11.111**', '**±14.123**', '**≤ 0.01**', '**−7.835**',
              '**−0.0241**', '158.385', '15.964'):
        assert s in PROTO_MD, s
    v2g = json.dumps(PROTO['gates_v3']['v2_gates'], ensure_ascii=False)
    for s in ('0.80〜1.25', '< 10 µs', '≤ 0.01', 'B の値 − 0.10', '300 フレーム', '1.3 ×', '15.964', '≤ 10 µs', '≤ 2.5 µs'):
        assert s in v2g, s
    assert (E.AMP, E.L1_VS_B_US, E.BRAKE_FRAC, E.STANDSTILL_GO_DROP, E.STANDSTILL_MIN_FRAMES, E.INFER_FACTOR, E.STRAIGHT_US) == \
        ((0.80, 1.25), 10.0, 0.01, 0.10, 300, 1.3, 10.0)
    assert '42.4' in json.dumps(PROTO['futility'], ensure_ascii=False) and T['futility_st23'] == 42.369


def good_row(**kw):
    r = {'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 2.5, 'b_trt_p99_ms_same_day': 2.5, 'steer_amp_ratio': 1.0,
         'l1_best_us': 49.0, 'brake_lowspeed_frac': 0.005, 'standstill_go_frac': 0.2, 'thr_bias_us': 12.0,
         'straight_thr_vs_b_us': 0.0, 'straight_thr_vs_b_us_1010': 0.0, 'straight_thr_vs_b_us_1023': 0.0,
         'standstill_back_frac': 0.02, 'probe_best': {'st10_thr_vs_b_us': 0.0, 'st23_thr_vs_b_us': 0.0,
                                                      'stand_brake_side_frac': 0.02, 'low10_brake_frac': 0.005}}
    r.update(kw)
    return r


def test_gates_v3_edges():
    b = good_row()
    assert E.gates_pass(E.gates_v3(good_row(), b, T, 1115)) is True
    edge = good_row(straight_thr_vs_b_us_1010=-10.0, straight_thr_vs_b_us_1023=12.138, standstill_back_frac=0.064,
                    probe_best={'st10_thr_vs_b_us': 11.111, 'st23_thr_vs_b_us': -14.123, 'stand_brake_side_frac': 0.064,
                                'low10_brake_frac': 0.01}, trt_p99_ms=1.3 * 2.5, brake_lowspeed_frac=0.01,
                    standstill_go_frac=0.1, thr_bias_us=15.96, straight_thr_vs_b_us=10.0, steer_amp_ratio=0.80)
    g = E.gates_v3(edge, b, T, 1115)
    assert E.gates_pass(g) is True, g
    for k, v in (('straight_thr_vs_b_us_1010', 10.001), ('straight_thr_vs_b_us_1023', -12.14), ('standstill_back_frac', 0.0641),
                 ('brake_lowspeed_frac', 0.0101), ('standstill_go_frac', 0.0999), ('thr_bias_us', 15.97),
                 ('straight_thr_vs_b_us', 10.01), ('l1_best_us', 59.0), ('steer_amp_ratio', 1.26), ('trt_p99_ms', 3.26),
                 ('engine_ok', False), ('sanity_ok', False)):
        g = E.gates_v3(good_row(**{k: v}), b, T, 1115)
        assert E.gates_pass(g) is False, (k, g)
    for k, v in (('st10_thr_vs_b_us', -11.2), ('st23_thr_vs_b_us', 14.2), ('stand_brake_side_frac', 0.065),
                 ('low10_brake_frac', 0.011)):
        p = dict(good_row()['probe_best'], **{k: v})
        assert E.gates_pass(E.gates_v3(good_row(probe_best=p), b, T, 1115)) is False, k


def test_gates_missing_is_undecided_and_small_standstill_is_reference():
    b = good_row()
    g = E.gates_v3(good_row(probe_best=dict(good_row()['probe_best'], low10_brake_frac=None)), b, T, 1115)
    assert g['probe_low10'] is None and E.gates_pass(g) is None
    g = E.gates_v3(good_row(trt_p99_ms=None), b, T, 1115)
    assert g['inference'] is None and E.gates_pass(g) is None
    g = E.gates_v3(good_row(probe_best=None), b, T, 1115)
    assert all(g[k] is None for k in ('probe_st10', 'probe_st23', 'probe_stand_brake_side', 'probe_low10'))
    g = E.gates_v3(good_row(standstill_go_frac=0.0), b, T, 299)
    assert g['standstill'] == '参考' and E.gates_pass(g) is True
    assert list(E.gates_v3(good_row(), b, T, 1115)) == list(E.GATE_ORDER)


def test_veto_interv_steer():
    m = 158.385
    v = E.veto_interv_steer({0: m + 4.2, 1: m + 4.3, 2: m + 4.0, 3: m + 4.5}, m, T['iv_steer_thr'])
    assert v['veto'] is True and v['thr'] == 4.111
    v = E.veto_interv_steer({0: m + 6.0, 1: m + 6.0, 2: m + 6.0, 3: m - 0.1}, m, T['iv_steer_thr'])
    assert v['veto'] is False                                # 1 本が平均より小さい
    v = E.veto_interv_steer({0: m + 5.0, 1: m + 5.0}, m, T['iv_steer_thr'])
    assert v['veto'] is False and v['thr'] == 5.127          # 途中でやめたとき n = 2
    assert E.veto_interv_steer({}, m, T['iv_steer_thr'])['veto'] is None


def test_veto_bias():
    assert E.veto_bias({s: 0.132 + 3.9 for s in range(4)}, 0.132, T['vs5'])['veto'] is False
    assert E.veto_bias({s: 0.132 - 3.92 for s in range(4)}, 0.132, T['vs5'])['veto'] is True
    assert E.veto_bias({0: 5.0, 1: 5.0}, 0.132, T['vs5'])['thr'] == T['vs5']['n2']['steer_bias_best']


# ---------------------------------------------------------------- 3. E1〜E3・途中でやめる・削った種・直しが効いた

def arm(seeds, l1=49.0, iv=150.0, bias=0.0, l1_last=None, expected=(0, 1, 2, 3)):
    rows = {s: {'l1_best_us': l1, 'l1_last_us': l1 if l1_last is None else l1_last, 'l1_interv_steer_best_us': iv,
                'steer_bias_us': bias} for s in seeds}
    five = [49.6] * 5
    return {'seeds': list(seeds), 'expected': list(expected), 'rows': rows,
            'complete': all(s in rows for s in expected),
            'vs5': E.vs5({s: r['l1_best_us'] for s, r in rows.items()}, {s: r['l1_last_us'] for s, r in rows.items()},
                         five, [49.9] * 5, T['vs5']),
            'veto': {'interv_steer': E.veto_interv_steer({s: iv for s in seeds}, 158.385, T['iv_steer_thr']),
                     'bias': E.veto_bias({s: bias for s in seeds}, 0.132, T['vs5'])}}


def name_of(a, s):
    return 'v3_20261026-%s-aug2-e60-s%d' % (a, s)


def test_e1_e2_e3_order_and_candidates():
    arms = {'v3n': arm((0, 1, 2, 3), l1=49.3), 'v3d': arm((0, 1, 2, 3), l1=49.0), 'v3b': arm((0, 1, 2, 3), iv=170.0)}
    e1 = E.decide_e1(arms)
    assert e1['kept'] == ['v3n', 'v3d'] and e1['arms']['v3b']['keep'] is False and e1['final']
    gp = {'v3n': {0: True, 1: False, 2: True, 3: True}, 'v3d': {0: False, 1: True, 2: True, 3: False}}
    e2 = E.decide_e2(e1, arms, gp)
    assert e2['order'] == ['v3n', 'v3d'] and e2['final']               # 合格 3 本 > 2 本(L1 は v3d が小さくても)
    e3 = E.decide_e3(e2, gp, name_of)
    assert (e3['V1'], e3['V2']) == (name_of('v3n', 0), name_of('v3n', 2))
    gp2 = {'v3n': {0: True, 1: True, 2: False, 3: False}, 'v3d': {0: True, 1: True, 2: False, 3: False}}
    e2b = E.decide_e2(e1, arms, gp2)
    assert e2b['order'] == ['v3d', 'v3n']                                # 同じ数 → H1 の L1(best)の平均が小さい順


def test_e3_v2_from_second_arm_and_none():
    arms = {'v3n': arm((0, 1, 2, 3)), 'v3d': arm((0, 1, 2, 3), l1=49.1), 'v3b': arm((0, 1, 2, 3), l1=49.2)}
    e1 = E.decide_e1(arms)
    gp = {'v3n': {0: False, 1: False, 2: True, 3: False}, 'v3d': {0: False, 1: True, 2: False, 3: False},
          'v3b': {0: False, 1: False, 2: False, 3: False}}
    e2 = E.decide_e2(e1, arms, gp)
    e3 = E.decide_e3(e2, gp, name_of)
    assert e2['order'] == ['v3n', 'v3d', 'v3b']
    assert (e3['V1'], e3['V2']) == (name_of('v3n', 2), name_of('v3d', 1))
    gp0 = {a: {s: False for s in range(4)} for a in arms}
    e3 = E.decide_e3(E.decide_e2(e1, arms, gp0), gp0, name_of)
    assert e3['V1'] is None and e3['V2'] is None


def test_e_undecided_when_seeds_or_gates_missing():
    arms = {'v3n': arm((0, 1)), 'v3d': arm((0, 1, 2, 3)), 'v3b': arm((0, 1, 2, 3))}
    e1 = E.decide_e1(arms)
    assert e1['arms']['v3n']['keep'] is None and not e1['final']
    arms = {'v3n': arm((0, 1), expected=(0, 1)), 'v3d': arm((0, 1, 2, 3)), 'v3b': arm((0, 1, 2, 3))}
    e1 = E.decide_e1(arms)
    assert e1['arms']['v3n']['keep'] is True and e1['final']             # 途中でやめた腕は n = 2 で決まる
    gp = {'v3n': {0: None, 1: True}, 'v3d': {s: True for s in range(4)}, 'v3b': {s: True for s in range(4)}}
    e2 = E.decide_e2(e1, arms, gp)
    assert not e2['final'] and E.decide_e3(e2, gp, name_of)['final'] is False
    assert not E.decide_e1({'v3n': arm((0, 1, 2, 3)), 'v3d': arm((0, 1, 2, 3))})['final']   # 腕がそろっていない


def test_futility_and_cut_seeds():
    hi = {'st23_thr_vs_b_us': 50.0, 'stand_brake_side_frac': 0.0}
    lo = {'st23_thr_vs_b_us': 5.0, 'stand_brake_side_frac': 0.0}
    rows = {0: {'probe_best': hi}, 1: {'probe_best': dict(lo, stand_brake_side_frac=0.2)}}
    exp, f = E.expected_seeds(rows, T)
    assert exp == [0, 1] and f['protocol'] is True and f['queue'] is True
    exp, f = E.expected_seeds({0: {'probe_best': hi}, 1: {'probe_best': lo}}, T)
    assert exp == [0, 1, 2, 3] and f['protocol'] is False
    exp, f = E.expected_seeds({0: {'probe_best': hi}}, T, cut_seeds=(3,))
    assert exp == [0, 1, 2] and f['protocol'] is None
    edge = {'st23_thr_vs_b_us': 42.38, 'stand_brake_side_frac': 0.0}    # protocol 42.369 と キュー 42.4 のあいだ
    _, f = E.expected_seeds({0: {'probe_best': edge}, 1: {'probe_best': edge}}, T)
    assert f['protocol'] is True and f['queue'] is False


def test_queue_cut_reads_commented_rows(tmp_path):
    p = tmp_path / 'q.tsv'
    p.write_text('# コメント\n1\tv3_20261026-v3n-aug2-e60-s2\tcmd a\n# 1\tv3_20261026-v3n-aug2-e60-s3\tcmd b\n', encoding='utf-8')
    assert D.queue_cut(p) == {'v3_20261026-v3n-aug2-e60-s3'}
    assert D.queue_cut(FIX / 'gpu_queue_v3.tsv') == {n for _, n, _, c in queue_module().queue_rows(include_commented=True) if c}


def test_fix_confirmed():
    base = {s: {'straight_thr_vs_b_us_1023': 40.0, 'standstill_back_frac': 0.45, 'l1_best_us': 48.5, 'l1_last_us': 48.7}
            for s in range(4)}
    good = {s: {'straight_thr_vs_b_us_1023': 30.0, 'standstill_back_frac': 0.40, 'l1_best_us': 48.6, 'l1_last_us': 48.8}
            for s in range(4)}
    f = E.fix_confirmed(good, base, T, 5.5401, 0.01703)
    assert f['straight_1023']['ok'] and f['standstill_back']['ok'] and f['h1_not_worse'] and len(f['say']) == 2
    mixed = dict(good)
    mixed[3] = dict(good[3], straight_thr_vs_b_us_1023=41.0)
    assert E.fix_confirmed(mixed, base, T, 5.5401, 0.01703)['straight_1023']['ok'] is False    # 1 組が > 0
    f2 = E.fix_confirmed({s: good[s] for s in (0, 1)}, base, T, 5.5401, 0.01703)
    assert f2['k'] == 2 and 'note' in f2 and f2['straight_1023']['thr'] == pytest.approx(-round(2 * 5.5401, 3))


def v3row(a, s, **kw):
    r = good_row(run=name_of(a, s), family='v3', arm=a, seed=s, l1_last_us=49.2, l1_interv_steer_best_us=150.0,
                 steer_bias_us=0.0, mark_too_long=False, mark_rise=False, probe_last=None)
    r.update(kw)
    return r


def test_judge_end_to_end_synthetic():
    five = {rid: good_row(run=run, l1_best_us=49.6, l1_last_us=49.9, l1_interv_steer_best_us=158.3852, steer_bias_us=0.132)
            for rid, run, *_ in D.FIVE}
    bad_probe = dict(good_row()['probe_best'], st23_thr_vs_b_us=50.0)
    rows = [v3row('v3n', s) for s in range(4)]
    rows[1]['brake_lowspeed_frac'] = 0.02                                  # 種 1 だけ門に落ちる
    rows += [v3row('v3d', s, probe_best=bad_probe) for s in range(3)]     # 種 0・1 で途中でやめる → 種 2 は数えない
    rows += [v3row('v3b', s, l1_best_us=48.9) for s in range(3)]          # 種 3 は表で削った
    J = E.judge(rows, five, T, 1115, cut=['v3_20261026-v3b-aug2-e60-s3'])
    a = J['arms']
    assert a['v3d']['expected'] == [0, 1] and a['v3d']['extra_seeds'] == [2] and a['v3d']['seeds'] == [0, 1]
    assert a['v3b']['expected'] == [0, 1, 2] and a['v3b']['complete']
    assert a['v3n']['gates_pass'] == {0: True, 1: False, 2: True, 3: True}
    assert a['v3d']['gates_pass'] == {0: False, 1: False}
    e = J['decisions']
    assert e['E1']['kept'] == ['v3n', 'v3d', 'v3b'] and e['E1']['final']
    assert e['E2']['order'] == ['v3b', 'v3n', 'v3d'] and e['E2']['final']    # 3 本どうしは L1 の小さい v3b が先
    assert (e['E3']['V1'], e['E3']['V2']) == (name_of('v3b', 0), name_of('v3b', 1)) and e['E3']['final']
    assert any('数に入れない' in n for n in J['notes'])


def test_loosen_at_end_mark():
    ok = good_row()['probe_best']
    bad = dict(ok, st23_thr_vs_b_us=30.0)
    assert E.loosen_at_end(ok, bad, T) is True
    assert E.loosen_at_end(ok, ok, T) is False
    assert E.loosen_at_end(ok, None, T) is None


# ---------------------------------------------------------------- 4. 学習中の物差しの出どころ

def test_pick_probe_prefers_result_json():
    rb = {'st10_thr_vs_b_us': 1.0, 'st23_thr_vs_b_us': 2.0, 'stand_brake_side_frac': 0.01, 'low10_brake_frac': 0.0, 'epoch': 40}
    q = dict(rb)
    q['st23_thr_vs_b_us'] = 2.005
    pk = D.pick_probe({'probe_best': rb, 'probe_last': rb, 'probe_history': [{'epoch': 1}]}, q, None, {'st10_thr_vs_b_us': 9})
    assert pk['best'] is rb and pk['source'] == 'result.json' and not pk['notes'] and pk['history'] == [{'epoch': 1}]
    q2 = dict(rb, st23_thr_vs_b_us=2.5)
    assert D.pick_probe({'probe_best': rb}, q2, None, None)['notes']           # 照らして違えば書く
    pk = D.pick_probe({}, q, None, {'x': 1})
    assert pk['best'] is q and 'キュー' in pk['source']
    pk = D.pick_probe(None, None, None, {'x': 1})
    assert pk['best'] == {'x': 1} and '保存した予測' in pk['source']
    assert D.pick_probe(None, None, None, None)['best'] is None


def test_queue_probe_file_names_and_run_dirs_match_the_queue():
    Q = queue_module()
    assert Q.PROBE_DIR == D.QUEUE_PROBE
    for name in D.protocol_runs(PROTO):
        if D.family(name) != 'v3':
            continue
        assert Q.run_dir(name) == D.run_dir_of(name, PROTO) == D.RUNS / name, name   # キューの tsv の --out と同じ
        assert Q.probe_file(name, 'best') == D.QUEUE_PROBE / ('%s_best.json' % name)
    # キューは desk_v3/eval_v3.py を見る(名前は同じ eval_v3.py・場所は違う。報告の課題に書いた)
    assert Q.EVAL_V3.name == 'eval_v3.py' and Q.EVAL_V3.parent.name == 'desk_v3'
    src = (FIX / 'gpu_queue_v3.sh').read_text(encoding='utf-8')
    assert '"$EVAL_V3" predict --runs "$name"' in src and "'predicted', a[0]" in (FIX / 'gpu_queue_v3.py').read_text(encoding='utf-8')
    # 実際の物差しの出力(段 0)があれば、同じ読み方で読める
    for p in sorted(D.QUEUE_PROBE.glob('*_best.json')):
        v = D.queue_probe(p.name[:-len('_best.json')], 'best')
        assert v is not None and all(k in v for k in D.PROBE_KEYS)
        assert Q.probe_result(p.name[:-len('_best.json')], 'best') == v


def test_eval_v3_cli_matches_queue_contract():
    ap_src = (C3 / 'eval_v3.py').read_text(encoding='utf-8')
    assert "sub.add_parser('predicted'" in ap_src and "q.add_argument('name')" in ap_src
    assert "sub.add_parser('predict'" in ap_src and "p.add_argument('--runs', nargs='*')" in ap_src


# ---------------------------------------------------------------- 5. 予測の印(今の重みか)・0 バイト・封

def test_predicted_needs_current_weights(tmp_path, monkeypatch):
    name = 'v3_20261026-v3n-aug2-e60-s0'
    run = tmp_path / 'run'
    run.mkdir()
    for ck in ('best', 'last'):
        (run / ('%s.pth' % ck)).write_bytes(b'w' * 10)
    proto = {'runs': [{'name': name, 'command': 'x --out %s' % run}]}
    assert D.run_dir_of(name, proto) == run
    monkeypatch.setattr(E, 'PRED', tmp_path / 'pred')
    monkeypatch.setattr(D, 'PRED', tmp_path / 'tab')
    monkeypatch.setattr(D, 'TRT', tmp_path / 'trt')
    ok, parts = E.is_predicted(name, proto)
    assert not ok and parts == {'h1_best': False, 'h1_last': False, 'tab': False, 'trt': False}
    (tmp_path / 'pred').mkdir()
    for ck in ('best', 'last'):
        st = D.weights_stamp(run / ('%s.pth' % ck))
        np.savez(E.h1_file(name, ck), key=np.array(D.TABLE_KEY), ckpt=np.array(ck), pred=np.zeros((10985, 10, 2), np.float32),
                 weights_size=np.array(st['size']), weights_mtime_ns=np.array(st['mtime_ns']))
    st = D.weights_stamp(run / 'best.pth')
    (tmp_path / 'tab').mkdir()
    np.savez(tmp_path / 'tab' / ('%s_tab.npz' % name), pred=np.zeros((5, 10, 2), np.float32),
             weights_size=np.array(st['size']), weights_mtime_ns=np.array(st['mtime_ns']))
    eng = run / D.ENGINE
    eng.write_bytes(b'e')
    es = eng.stat()
    (tmp_path / 'trt').mkdir()
    D.write_json(tmp_path / 'trt' / '20261027-010101.json', {'schema': 'jf-c3-trt/v3', 'job': '20261027-010101', 'day': '20261027',
                                                             'engines': {name: {'engine': {'size': es.st_size,
                                                                                           'mtime_ns': es.st_mtime_ns}}}})
    ok, parts = E.is_predicted(name, proto)
    assert ok, parts
    (run / 'best.pth').write_bytes(b'w' * 11)                 # 学習し直した(--resume など)→ 予測し直す
    ok, parts = E.is_predicted(name, proto)
    assert not ok and not parts['h1_best'] and not parts['tab'] and parts['h1_last']


def test_zero_byte_npz_is_skipped(tmp_path, monkeypatch):
    z = tmp_path / 'AI_CMD.npz'
    z.write_bytes(b'')
    assert D.npz_ok(z) is False and D.npz_ok(tmp_path / 'none.npz') is False
    assert D.load_h1_pred(z) is None
    monkeypatch.setattr(D, 'PRED', tmp_path)
    monkeypatch.setattr(D, 'PRED2', tmp_path)
    (tmp_path / 'x_tab.npz').write_bytes(b'')
    assert D.load_tab('x', 'k') is None


def test_seal_guard_and_sets_have_no_sealed_logs():
    with pytest.raises(SystemExit):
        D.seal_guard(['20261024-134607-0001'], PROTO['seal']['logs'])
    D.seal_guard(['20261023-210840'], PROTO['seal']['logs'])
    S = D.load_sets(PROTO)                                      # 封の記録があれば止まる
    assert len(S['frame']) == 16621 and int(S['straight'].sum()) == 2043 and int(S['stand'].sum()) == 1115


def test_names_and_families():
    assert D.family('v3_20261026-v3d-aug2-e60-s3') == 'v3' and D.arm_seed('v3_20261026-v3d-aug2-e60-s3') == ('v3d', 3)
    assert D.family('bfast2_20261026-e15-s1') == 'bfast2' and D.family(D.B_RUN) == 'ref'
    assert D.family('v2_20261025-v2h-aug2-e60-s0') == 'v2' and D.arm_seed('v2_20261025-v2h-aug2-e60-s0') == ('v2h-aug2-e60', 0)
    assert D.allowed_run('v3_20261026-v3n-aug2-e60-s0', PROTO) and not D.allowed_run('v3_20261026-v3x-aug2-e60-s0', PROTO)
    with pytest.raises(SystemExit):
        E.check_names(['realacc_20261015-v0-s2'], PROTO, True)          # 5 本は保存した予測を使う


def test_fields_used_by_eval_exist_in_desk():
    src = (C3 / 'eval_v3.py').read_text(encoding='utf-8')
    used = set(re.findall(r"r\.get\('([a-z0-9_]+)'\)", src))
    meta = {'rid', 'run', 'family', 'arm', 'seed', 'notes', 'desk_notes', 'probe_tab', 'n', 'tab_sha256'}   # n・tab_sha256 は確かめだけ
    desk_side = {k for k in used if k not in E.FIELDS_H1 and not k.startswith('mark_') and k not in meta}
    assert {'straight_thr_vs_b_us_1023', 'standstill_back_frac', 'probe_best', 'b_trt_p99_ms_same_day'} <= desk_side
    assert desk_side <= set(D.FIELDS_DESK), desk_side - set(D.FIELDS_DESK)


# ---------------------------------------------------------------- 6. 本物の保存した予測で(段 0 の 8・空回し)

def test_stage0_check8_real(tmp_path):
    rc = E.cmd_check(argparse.Namespace(out=str(tmp_path)))
    o = json.loads((tmp_path / 'check_v3.json').read_text(encoding='utf-8'))
    assert rc == 0 and o['all_ok'], o['ng'][:5]
    s = o['summary']
    for part in ('1 5 本の H1・介入', '2 新しい欄(5 本)', '2 新しい欄(v2 の 22 本)', '3 物差し(保存した予測)',
                 '4 v2 の 22 本の介入', '8 閾値の数え直し', '自己試験(protocol の数え方)'):
        assert s[part]['n'] > 0 and s[part]['ng'] == 0, part
    st = o['self_test_new_gates']
    assert all(st[run] == 'pass' for run in D.FIVE_RUNS)
    v2 = [r for r in st if r.startswith('v2_')]
    assert len(v2) == 19 and all(st[r] != 'pass' for r in v2)
    assert all(o['full_gates'][r]['pass'] is False for r in v2)
    assert o['full_gates'][D.B_RUN]['pass'] is True
    assert sorted(m['run'] for m in o['missing'] if m['what'].startswith('probe の low10')) == sorted(E.LOW10_MISSING)


def test_probe_check_gpu_merges_extra(tmp_path, monkeypatch):
    extra = tmp_path / 'probe_low10_ref.json'
    D.write_json(extra, {'/home/jetson/jetformer/data/runs/realacc_20261015-v0-s2': {'low10_brake_frac': 0.5},
                         D.B_RUN: {'low10_brake_frac': 0.9}})
    monkeypatch.setattr(D, 'PROBE_EXTRA', extra)
    pc = D.probe_check_gpu()
    assert pc['realacc_20261015-v0-s2']['low10_brake_frac'] == 0.5
    assert pc[D.B_RUN]['low10_brake_frac'] == 0.0052          # probe_check.json が先


def test_dry_run_judge_with_b(tmp_path):
    D.cmd_report(argparse.Namespace(out=str(tmp_path / 'desk'), runs=None))
    E.cmd_judge(argparse.Namespace(desk=str(tmp_path / 'desk' / 'desk_v3.json'), out=str(tmp_path / 'j'), dry_run_b=True))
    p = json.loads((tmp_path / 'j' / 'decisions_v3_proposed.json').read_text(encoding='utf-8'))
    assert p['E1']['kept'] == ['v3n', 'v3d', 'v3b'] and p['E1']['final']
    assert p['E3']['V1'] == 'v3_20261026-v3n-aug2-e60-s0' and p['E3']['V2'] == 'v3_20261026-v3n-aug2-e60-s1'
    E.cmd_judge(argparse.Namespace(desk=str(tmp_path / 'desk' / 'desk_v3.json'), out=str(tmp_path / 'k'), dry_run_b=False))
    p = json.loads((tmp_path / 'k' / 'decisions_v3_proposed.json').read_text(encoding='utf-8'))
    assert p['E1']['kept'] == [] and not p['E1']['final'] and p['E3']['V1'] is None


def test_protected_files_unchanged():
    after = {str(p): sha(p) for p in PROTECTED + [D.DESK_JSON]}
    assert after == BEFORE


# ---------------------------------------------------------------- 7. GPU の手順を作り物で通す(torch・GPU・TensorRT を読まない)

import types  # noqa: E402

NAME = 'v3_20261026-v3n-aug2-e60-s0'


def fake_modules(monkeypatch, n_h1=10985):
    """torch・policy_zoo・rl_evaluate・runtime・desk_gpu・rl_policies・cv2・jetformer.* の作り物を sys.modules に入れる"""
    calls = {'predict_frames': 0, 'getitem': 0, 'run_numpy': 0}
    torch = types.ModuleType('torch')
    torch.from_numpy = lambda a: a
    torch.zeros = lambda *a, **k: np.zeros(1)
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None)
    pz = types.ModuleType('policy_zoo')

    def _predict_frames(model, fs, genres, num_modes, batch, workers):
        calls['predict_frames'] += 1
        fs[0]                                   # __getitem__ を 1 回通す
        calls['getitem'] += 1
        return {'actions': np.full((len(fs), 10, 2), 1500.0, np.float32)}
    pz._predict_frames = _predict_frames
    ev = types.ModuleType('rl_evaluate')
    ev._genres = lambda modes, gm, trained: (np.asarray(modes, np.int64), np.zeros(len(modes), bool))
    rt = types.ModuleType('jetformer.policies.runtime')
    rt.load_run = lambda rd, ck, device='cuda': (object(), {}, None, None)
    td = types.ModuleType('jetformer.train_data')
    td.normalize_inputs = lambda img, tof, imu, veh, uc: (np.zeros((3, 4, 4), np.float32), np.zeros(4, np.float32),
                                                          np.zeros(6, np.float32), np.zeros(4, np.float32))
    cv2 = types.ModuleType('cv2')
    cv2.imread = lambda p: np.zeros((8, 8, 3), np.uint8)
    rp = types.ModuleType('rl_policies')
    rp._check_gpu_free = lambda min_gb=2.0: None
    rp._save_npz = lambda path, **a: (Path(path).parent.mkdir(parents=True, exist_ok=True), np.savez(path, **a))
    dg = types.ModuleType('desk_gpu')
    dg.web_busy = lambda: []
    dg.check_ready = lambda: None
    dg.table = lambda: {'key': D.TABLE_KEY, 'scene': np.array(['20261010-155820-0001'])}

    def predict_one(run_dir, ck, tab, batch=32, workers=3):
        return ({'actions': np.full((n_h1, 10, 2), 1500.0, np.float32)}, np.zeros(n_h1, np.int64), np.zeros(n_h1, bool),
                {'key': np.array(D.TABLE_KEY), 'run': np.array(Path(run_dir).name), 'ckpt': np.array(ck)})
    dg.predict_one = predict_one
    dg.ai_frames = lambda: [('ai_scene-0001', i, 1000 + i) for i in range(4)]
    # TensorRT の側
    cfg = types.ModuleType('jetformer.config')
    cfg.DRIVE = {'min_us': 1000, 'max_us': 2000}
    cfg.CAMERA = {'fps': 30}
    drv = types.ModuleType('jetformer.drive')
    drv.DEFAULT_PARAMS = {k: 0 for k in D.BASE_PARAMS_OVERRIDE}
    drv.model_info = lambda n: {'drivable': True, 'why_not': None, 'num_modes': 10, 'steer_center_us': 1433.0,
                                'throttle_neutral_us': 1500.0, 'command_input': False}
    pol = types.ModuleType('jetformer.policies')
    pol.check_io = lambda i, o, nm: (True, '')
    post = types.ModuleType('jetformer.policies.post')

    class PostError(Exception):
        pass

    class Stage(object):
        def __init__(self, info, ref, fps=30):
            pass

        def reset(self):
            pass

        def model_pred(self, res):
            return np.full((10, 2), 1500.0, np.float32)

        def apply(self, res, mp, speed, params, t_ns):
            return mp, None
    post.Stage, post.PostError, post.speed_of = Stage, PostError, (lambda e: 0.0)
    pol.post, pol.runtime = post, rt
    trt = types.ModuleType('jetformer.trt_runtime')

    class TRTRunner(object):
        def __init__(self, path):
            self.inputs, self.outputs = ['image', 'tof', 'imu', 'state', 'mode'], ['actions']
            self.shapes = {'image': (1, 3, 4, 4), 'tof': (1, 4), 'imu': (1, 6), 'state': (1, 4), 'mode': (1, 10),
                           'actions': (1, 10, 2)}

        def run_numpy(self, **feed):
            calls['run_numpy'] += 1
            return {'actions': np.zeros((1, 10, 2), np.float32)}
    trt.TRTRunner = TRTRunner
    jf = types.ModuleType('jetformer')
    jf.config, jf.drive, jf.policies, jf.train_data, jf.trt_runtime = cfg, drv, pol, td, trt
    for name, mod in (('torch', torch), ('policy_zoo', pz), ('rl_evaluate', ev), ('rl_policies', rp), ('desk_gpu', dg),
                      ('cv2', cv2), ('jetformer', jf), ('jetformer.config', cfg), ('jetformer.drive', drv),
                      ('jetformer.policies', pol), ('jetformer.policies.post', post), ('jetformer.policies.runtime', rt),
                      ('jetformer.train_data', td), ('jetformer.trt_runtime', trt)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def fake_world(tmp_path, monkeypatch):
    """作り物の run・データセット・保存の場所。D・E の場所の定数を tmp に向ける"""
    runs = tmp_path / 'runs'
    for n in (NAME, D.B_RUN):
        d = runs / n
        d.mkdir(parents=True)
        for f in ('best.pth', 'last.pth', D.ENGINE):
            (d / f).write_bytes(b'x' * 8)
        D.write_json(d / 'model_config.json', {'trained_modes': [5], 'command_input': False, 'num_modes': 10,
                                               'steer_center_us': 1433.0, 'throttle_neutral_us': 1500.0})
    ds = tmp_path / 'datasets'
    for dsname, scene in ((D.DS_1010, 'ai_scene-0001'), (D.DS_V2, 'v2_scene-0001')):
        (ds / dsname / 'aligned').mkdir(parents=True)
        np.savez(ds / dsname / 'aligned' / (scene + '.npz'), cam_files=np.array(['c/%d.jpg' % i for i in range(4)]),
                 tof=np.zeros((4, 4)), imu=np.zeros((4, 6)), vehicle=np.zeros((4, 5)), mode=np.full(4, 5))
    (ds / D.DS_V2 / 'aligned' / 'zero-0001.npz').write_bytes(b'')        # 0 バイトの aligned(飛ばす)
    S = {'scene': np.array(['v2_scene-0001'] * 3), 'frame': np.array([0, 1, 2]), 'key': 'k3',
         'san_scene': np.array(['v2_scene-0001', 'zero-0001', 'v2_scene-0001']), 'san_frame': np.array([0, 0, 3]),
         'san_t_us': np.array([1, 2, 3])}
    pred2 = tmp_path / 'pred2'
    pred2.mkdir()
    np.savez(pred2 / ('%s_tab.npz' % D.B_RUN), key=np.array('k3'), pred=np.full((3, 10, 2), 1500.0, np.float32))
    zoo_b = tmp_path / 'zoo_B.npz'
    np.savez(zoo_b, key=np.array(D.TABLE_KEY), pred=np.full((10985, 10, 2), 1500.0, np.float32), genre=np.zeros(10985, np.int8))
    monkeypatch.setattr(D, 'run_dir_of', lambda n, proto=None: runs / n)
    monkeypatch.setattr(D, 'RUNS', runs)
    monkeypatch.setattr(D, 'DATASETS', ds)
    monkeypatch.setattr(D, 'PRED', tmp_path / 'tab')
    monkeypatch.setattr(D, 'PRED2', pred2)
    monkeypatch.setattr(D, 'TRT', tmp_path / 'trt')
    monkeypatch.setattr(D, 'WORK', tmp_path / 'work')
    monkeypatch.setattr(D, 'load_sets', lambda proto: S)
    monkeypatch.setattr(D, 'FIVE', D.FIVE[:-1] + [('B', D.B_RUN, zoo_b, zoo_b, None)])
    monkeypatch.setattr(E, 'PRED', tmp_path / 'h1')
    monkeypatch.setattr(E, 'CROSS_H1', tmp_path / 'h1' / 'crosscheck_B_h1.json')
    return runs, S


def test_gpu_paths_with_fakes(tmp_path, monkeypatch):
    calls = fake_modules(monkeypatch)
    runs, S = fake_world(tmp_path, monkeypatch)
    proto = PROTO
    # ① AI の記録の集合(desk_v3 predict): B の予測し直し → 候補
    D.cmd_predict(argparse.Namespace(runs=[NAME], force=False, batch=4, workers=0, allow_before_lock2=True))
    cj = json.loads((tmp_path / 'work' / 'crosscheck_B_tab.json').read_text(encoding='utf-8'))
    assert cj['pass'] and cj['max_abs_us'] == 0
    assert D.tab_current(NAME, proto) and calls['predict_frames'] == 2 and calls['getitem'] == 2
    with np.load(D.PRED / ('%s_tab.npz' % NAME)) as z:
        assert z['pred'].shape == (3, 10, 2) and str(z['key']) == 'k3' and int(z['weights_size']) == 8
    D.cmd_predict(argparse.Namespace(runs=[NAME], force=False, batch=4, workers=0, allow_before_lock2=True))
    assert calls['predict_frames'] == 2                          # そろっていれば飛ばす(B の照らしも済み)
    # ② H1・介入(eval_v3 predict-h1): B の予測し直し → best・last
    E.cmd_predict_h1(argparse.Namespace(runs=[NAME], force=False, allow_before_lock2=True))
    assert json.loads(E.CROSS_H1.read_text(encoding='utf-8'))['pass']
    assert E.h1_current(NAME, 'best', proto) and E.h1_current(NAME, 'last', proto)
    # ③ TensorRT(desk_v3 trt): B → 候補 → B。0 バイトの aligned は飛ばす
    D.cmd_trt(argparse.Namespace(runs=[NAME], max_runs=2, limit=0, allow_before_lock2=True))   # --limit の試しは欄に使わない
    jobs = sorted((tmp_path / 'trt').glob('*.json'))
    assert len(jobs) == 1
    j = json.loads(jobs[0].read_text(encoding='utf-8'))
    assert j['schema'] == 'jf-c3-trt/v3' and set(j['engines']) == {D.B_RUN, NAME}
    assert j['skipped_zero_byte_frames'] == 1 and j['frames'] == 7 and j['limit'] is None
    for st in j['engines'].values():
        assert st['sanity_ok'] is True and st['frames'] == 6 and st['nonfinite_frames'] == 0, st
        assert st['run_numpy_p99_ms'] is not None and Path(st['file']).exists()
    assert D.trt_current(NAME, proto)
    ok, parts = E.is_predicted(NAME, proto)
    assert ok, parts
    # TensorRT の欄が読める(同じ日の B・低速のブレーキ)
    T_ = D.trt_results((D.TRT,))
    sv = np.full((10985, 10, 2), 1500.0)
    tf = D.trt_fields(NAME, runs / NAME, T_, {'erpm': np.zeros(10985), 'primary': np.ones(10985, bool)}, sv)
    assert tf['sanity_ok'] and tf['b_trt_p99_ms_same_day'] is not None and tf['inference_ok_same_day'] is not None
    assert tf['brake_lowspeed_frac'] == 0.0


def test_predict_orchestration_runs_missing_parts_only(tmp_path, monkeypatch):
    runs, S = fake_world(tmp_path, monkeypatch)
    seen = []
    real_run = E.subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[0] != sys.executable:                      # sha256sum -c などは本物
            return real_run(cmd, *a, **k)
        seen.append(cmd[2:])
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(E.subprocess, 'run', fake_run)
    monkeypatch.setattr(D, 'LOCK2', tmp_path / 'lock2.json')
    D.write_json(tmp_path / 'lock2.json', {})
    assert E.cmd_predict(argparse.Namespace(runs=[NAME], allow_before_lock2=False)) == 0
    kinds = [(Path(c[0]).name, c[1]) for c in seen]
    assert kinds == [('eval_v3.py', 'predict-h1'), ('desk_v3.py', 'predict'), ('desk_v3.py', 'trt')]
    assert all(c[-1] == NAME for c in seen)
    (runs / NAME / D.ENGINE).unlink()
    seen.clear()
    assert E.cmd_predict(argparse.Namespace(runs=[NAME], allow_before_lock2=False)) == 3     # エンジンが無い → TensorRT は飛ばす
    assert [c[1] for c in seen] == ['predict-h1', 'predict']
