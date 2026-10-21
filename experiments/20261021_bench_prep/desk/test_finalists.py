"""finalists.py の門(A1 入り)・risk・desk_key・家族の試験(標準ライブラリ + pytest。torch・GPU なし)

    jf-heavy --lock none --mem 1500M -- python -m pytest -q test_finalists.py
"""
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('finalists', str(HERE / 'finalists.py'))
fz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fz)

B = {'id': 'B', 'l1_best_us': 49.144, 'l1_steer_best_us': 50.89}
POOLF = {'MSF': 'arch', 'TCP': 'arch', 'ACT': 'arch', 'SIM': 'robust', 'AUG': 'robust', 'ENS': 'ens', 'SLOW': 'label_r1', 'WGT': 'weight'}


def row(**k):
    r = dict(id='X', run='x', engine_ok=True, sanity_ok=True, trt_p99_ms=3.0, steer_amp_ratio=1.0, l1_best_us=50.0,
             l1_steer_best_us=51.0, brake_lowspeed_frac=0.0, l1_interv_us=125.0, r1_app_thr_diff_us=0.0)
    r.update(k)
    return r


def test_gate5_boundary_and_missing():
    assert fz.gates(row(l1_best_us=49.144 + 9.999), B, 'arch')[0] == []
    assert fz.gates(row(l1_best_us=49.144 + 10.0), B, 'arch')[0] == ['l1_best_us']
    assert fz.gates(row(l1_best_us=None), B, 'arch')[0] == ['missing:l1_best_us']
    assert fz.gates(row(trt_p99_ms=float('nan')), B, 'arch')[0] == ['missing:trt_p99_ms']
    assert fz.gates(row(engine_ok=False), B, 'arch')[0] == ['engine_ok']
    assert fz.gates(row(steer_amp_ratio=1.26), B, 'arch')[0] == ['steer_amp_ratio']
    assert fz.gates(row(steer_amp_ratio=0.80, trt_p99_ms=10.0), B, 'arch')[0] == []


def test_a1_only_for_label_r1():
    slow = row(id='SLOW', run='bench_20261021-slow-s0', l1_best_us=63.4, l1_steer_best_us=54.6, brake_lowspeed_frac=0.0035)
    assert fz.gates(slow, B, 'label_r1')[0] == []                         # A1: ステアの L1 で比べる
    assert fz.gates(slow, B, 'arch')[0] == ['l1_best_us']                 # A1 を当てなければ落ちる
    assert fz.gates(dict(slow, brake_lowspeed_frac=0.0101), B, 'label_r1')[0] == ['brake_lowspeed_frac']
    assert fz.gates(dict(slow, brake_lowspeed_frac=0.01), B, 'label_r1')[0] == []
    assert fz.gates(dict(slow, brake_lowspeed_frac=None), B, 'label_r1')[0] == ['missing:brake_lowspeed_frac']
    assert fz.gates(dict(slow, l1_steer_best_us=60.89), B, 'label_r1')[0] == ['l1_steer_best_us']
    # ほかの家族は brake_lowspeed_frac を門にしない(SIM 0.080・WGT 0.055 でも通る)
    assert fz.gates(row(brake_lowspeed_frac=0.08), B, 'robust')[0] == []


def test_family_by_run_name():
    assert fz.family({'id': 'AUG', 'run': 'bench_20261021-slow-s1'}, POOLF) == 'label_r1'
    assert fz.family({'id': 'ENS', 'run': 'bench_20261021-ens-p2'}, POOLF) == 'ens'
    assert fz.family({'id': 'SIM', 'run': 'simreal_mix_20261013-cotrain-ft-s0'}, POOLF) == 'robust'


def test_risk_rule():
    assert fz.risk(row(), False) == 'low'
    assert fz.risk(row(), True) == 'mid'                                   # 新しい方策のコード(ENS)
    assert fz.risk(row(r1_app_thr_diff_us=-10.0), False) == 'low'
    assert fz.risk(row(r1_app_thr_diff_us=-10.01), False) == 'mid'
    assert fz.risk(row(r1_app_thr_diff_us=-20.01), False) == 'high'
    assert fz.risk(row(steer_amp_ratio=0.849), False) == 'high'
    assert fz.risk(row(steer_amp_ratio=1.15), False) == 'mid'
    assert fz.risk(row(r1_app_thr_diff_us=None), False) == 'mid'
    assert fz.risk(row(trt_p99_ms=8.01), False) == 'mid'


def test_desk_key_rounding_and_order():
    a = row(id='A', l1_interv_us=123.56, r1_app_thr_diff_us=0.0)
    b = row(id='Bb', l1_interv_us=123.649, r1_app_thr_diff_us=5.0)      # 123.6 で同点 → 2 番目の鍵(大きい順)で Bb が先
    assert sorted([a, b], key=fz.desk_key)[0]['id'] == 'Bb'
    c = row(id='C', l1_interv_us=None)
    assert sorted([c, a], key=fz.desk_key)[0]['id'] == 'A'                # null は最後
    d1, d2 = row(id='D1'), row(id='D2')
    assert sorted([d2, d1], key=fz.desk_key)[0]['id'] == 'D1'             # 最後は id
    assert fz.rnd(123.555, 1) == 123.6 and fz.rnd(-8.25, 1) == -8.3
    assert math.isclose(fz.desk_key(row(steer_amp_ratio=0.9822))[3][1], 0.018)
