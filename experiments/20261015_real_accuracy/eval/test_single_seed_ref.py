#!/usr/bin/env python3
"""single_seed_ref.py のテスト(numpy・GPU なし)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q experiments/20261015_real_accuracy/eval/test_single_seed_ref.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_seed_ref as S  # noqa: E402


def _row(p, s1=40.0, s3=120.0, bat=90.0):
    return {'primary': p, 'S1_no_battery_scene': s1, 'S3_intervention_part1': s3, 'battery_scene_only': bat}


def _pv(**kw):
    return {k: {'best': _row(v[0]), 'last': _row(v[1])} for k, v in kw.items()}


def test_gainl_inside_threshold():
    pv = _pv(**{'GAIN(gainl-s0)': (44.45, 44.576), 'GAIN(A_s0)': (44.935, 44.979), 'gainl-s0': (49.688, 49.861),
                'A_s0': (49.146, 49.235)})
    r = S.single_seed_reference(pv)
    g = r['GAINL_s0']
    assert g['verdict'] is None
    assert abs(g['post']['best']['primary']['diff'] - (44.45 - 44.935)) < 1e-9
    assert g['post']['best']['primary']['within_threshold'] and g['post']['last']['primary']['within_threshold']
    assert '閾値(0.626・1.322)の中' in g['text']
    assert r['SIZE_EQ_s0'] is None              # sizeHeq が無ければ出さない


def test_gainl_outside_threshold():
    pv = _pv(**{'GAIN(gainl-s0)': (43.0, 43.0), 'GAIN(A_s0)': (44.935, 44.979), 'gainl-s0': (49.0, 49.0),
                'A_s0': (49.146, 49.235)})
    g = S.single_seed_reference(pv)['GAINL_s0']
    assert not g['post']['best']['primary']['within_threshold']
    assert '判定はしない' in g['text']


def test_size_eq_formula_and_reading():
    pv = _pv(**{'sizeH-s0': (52.41, 52.554), 'sizeHeq-s0': (51.828, 52.108), 'A_s0': (49.146, 49.235)})
    r = S.single_seed_reference(pv)['SIZE_EQ_s0']
    d = math.log2(5912 / 3046)
    row = r['by_metric']['primary/best']
    assert abs(row['s2_s0'] - (52.41 - 49.146) / d) < 1e-9
    assert abs(row['s2_eq_s0'] - (51.828 - 49.146) / d) < 1e-9
    assert abs(row['share_of_gap_from_steps'] - (52.41 - 51.828) / (52.41 - 49.146)) < 1e-9
    assert r['text'].startswith('ステップ数をそろえてもデータの量の効果が残る')
    # s2_eq < 1 なら「ステップ数の差で説明できる部分が大きい」(21:55 に決めた読み方)
    pv2 = _pv(**{'sizeH-s0': (52.41, 52.554), 'sizeHeq-s0': (49.6, 49.7), 'A_s0': (49.146, 49.235)})
    assert S.single_seed_reference(pv2)['SIZE_EQ_s0']['text'].startswith('データの量の効果は、ステップ数の差で')


def test_gainl_s3_worse_is_reported():
    pv = _pv(**{'GAIN(gainl-s0)': (44.45, 44.576), 'GAIN(A_s0)': (44.935, 44.979), 'gainl-s0': (49.688, 49.861),
                'A_s0': (49.146, 49.235)})
    pv['GAIN(gainl-s0)']['last']['S3_intervention_part1'] = 130.24
    pv['GAIN(A_s0)']['last']['S3_intervention_part1'] = 124.763
    g = S.single_seed_reference(pv)['GAINL_s0']
    assert '介入の集合(S3、拒否権の指標)では悪い向きで閾値を超えた: last +5.48' in g['text']
    assert 'best' not in g['text'].split('閾値を超えた:')[1]
