#!/usr/bin/env python3
"""desk_lib2.py の試験(numpy だけ・torch と GPU を使わない)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -m pytest -q \\
        /home/jetson/jetformer/experiments/20261022_prep/desk/test_desk_lib2.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import desk_lib2 as D2  # noqa: E402


def test_register_adds_three_once_and_family():
    D = D2.register()
    D2.register()
    for cid, _, run, fam in D2.NEW:
        assert D.IDS.count(cid) == 1
        assert D.RUN_OF[cid] == run
        assert D.family_by_run(run, None) == 'label_r1' == fam
    assert D.RUN_OF['SLOW'] == 'bench_20261021-slow-s0'


def test_nonzone_mask_and_row0():
    n = 6
    pred = np.full((n, 10, 2), 1400.0)
    y = np.full((n, 10, 2), 1400.0)
    pred[:, 0, 1] = [1410, 1420, 1430, 1440, 1450, 1460]     # 1 行目: 人より +10..+60
    pred[:, 2, 1] = 1300                                     # 3 行目は違う値(1 行目だけを使うか)
    zone = np.array(['none', 'unknown', 'R1_app', 'GM', 'none', 'DE'])
    primary = np.array([True, True, True, True, False, True])
    m = D2.nonzone_mask(primary, zone)
    assert m.tolist() == [True, True, False, False, False, False]
    assert D2.nonzone_thr_diff(pred, y, primary, zone) == 15.0
    assert D2.nonzone_thr_diff(pred, y, primary, zone, row=2) == -100.0
    z = D2.zone_thr_diff(pred, y, primary, zone)
    assert z['GM'] == {'n': 1, 'thr_diff_us': 40.0} and z['R1_turn']['n'] == 0 and z['R1_turn']['thr_diff_us'] is None


def test_nonzone_empty_is_none():
    pred = np.zeros((2, 10, 2))
    assert D2.nonzone_thr_diff(pred, pred, [True, True], ['GM', 'DE']) is None


def _r(gp=True, r1=30.0, nz=10.0):
    return {'gates_pass': gp, 'r1_app_thr_diff_us': r1, 'nonzone_thr_diff_us': nz}


def test_c1_rule_order_ties_and_exclusions():
    rows = {'SLOW': _r(nz=30.04), 'SLOW1': _r(nz=12.26, r1=25.0), 'SLOW-SEC0': _r(nz=12.34, r1=28.0),
            'SLOW-SEC1': _r(nz=5.0, r1=19.99)}
    c = D2.c1_rule_a10(rows)
    # SEC1 は r1 < 20 で対象外。SLOW1 12.26 → 12.3、SEC0 12.34 → 12.3 で同点 → r1 が大きい SEC0
    assert c['pick'] == 'SLOW-SEC0'
    assert [t['id'] for t in c['targets']] == ['SLOW-SEC0', 'SLOW1', 'SLOW']
    assert c['excluded'] == {'SLOW-SEC1': 'r1_app_thr_diff_us<20.0'}
    # 同点で r1 も同じ → id の文字の順
    rows2 = {'SLOW-SEC0': _r(nz=7.0, r1=30.0), 'SLOW1': _r(nz=7.0, r1=30.0)}
    assert D2.c1_rule_a10(rows2)['pick'] == 'SLOW-SEC0'
    assert sorted(['SLOW1', 'SLOW-SEC0'])[0] == 'SLOW-SEC0'


def test_c1_rule_gate_missing_and_fallback():
    rows = {'SLOW': _r(gp=False), 'SLOW1': _r(r1=None), 'SLOW-SEC0': _r(nz=None)}
    c = D2.c1_rule_a10(rows)
    assert c['pick'] == 'SLOW' and c['empty_fallback'] is True
    assert c['excluded']['SLOW'] == 'gates_pass'
    assert c['excluded']['SLOW1'] == 'missing:r1_app_thr_diff_us'
    assert c['excluded']['SLOW-SEC0'] == 'missing:nonzone_thr_diff_us'
    assert c['excluded']['SLOW-SEC1'] == 'no_row'


def test_c1_rule_r1_boundary_inclusive():
    c = D2.c1_rule_a10({'SLOW1': _r(r1=20.0, nz=1.0), 'SLOW': _r(r1=40.0, nz=2.0)})
    assert c['pick'] == 'SLOW1'


def test_round1_half_up():
    assert D2._round1(12.25) == 12.3
    assert D2._round1(12.35) == 12.4
    assert D2._round1(-0.05) == -0.1
    assert D2._round1(3.04) == 3.0
