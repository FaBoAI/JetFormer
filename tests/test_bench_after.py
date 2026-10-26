"""10/23 の課題の直し(2026-10-24): web/bench.py の
- plan_check: ブロックの計画と違うモデル・設定で走り始めたら cand=EXP・official=0 にして memo に違いを書く
- after_description・POST /api/bench/after: 走り終わりのメモ(電池・電圧・止めた理由)をシーンの説明に書く
夜の道具(tools/section_scorecard.py の parse_description・effective_description)がそれを読めること

    cd ~/jetformer && python -m pytest -q tests/test_bench_after.py
"""

import json
import os
import sys

import pytest
from fastapi import HTTPException

from jetformer import config
from jetformer.web import bench

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import section_scorecard  # noqa: E402

B = bench.B_RUN
WGT = 'bench_20261021-wgt-s0'
DESC = ('ai_no=8 bench=real_bench_20261023 day=2026-10-23 block=1 cand=B model=%s battery=#__ V_before=__ V_after=__ '
        'check=__ official=1 plan_laps=3 memo=' % B)


def data():
    base = bench.fallback_baseline()
    return {'ok': True, 'baseline': base,
            'candidates': [{'id': 'B', 'run': B}, {'id': 'WGT', 'run': WGT}],
            'days': [{'date': '2026-10-23', 'blocks': [{'block': 1, 'id': 'B', 'run': B}, {'block': 7, 'id': 'WGT', 'run': WGT}]}]}


def params(**kw):
    p = dict(bench.fallback_baseline()['params'])
    p.update(kw)
    return p


def test_same_as_plan_is_unchanged():
    out, diffs = bench.plan_check(DESC, B, 5, params(), data())
    assert diffs == [] and out == DESC


def test_other_model_becomes_exp():
    out, diffs = bench.plan_check(DESC, 'clean_driving_20261011-dinov2-s0-nocmd-stride3-e30', 5, params(), data())
    d = section_scorecard.parse_description(out)
    assert d['cand'] == 'EXP' and d['official'] == '0' and d['block'] == 1
    assert '元 cand=B' in d['memo'] and 'dinov2' in d['memo']
    assert len(diffs) == 1


def test_gain_and_guard_become_exp_but_steer_only_is_recorded():
    out, diffs = bench.plan_check(DESC, B, 5, params(throttle_gain=1.25, steer_gain=1.3), data())
    d = section_scorecard.parse_description(out)
    assert d['cand'] == 'EXP' and d['steer'] == '1.3' and any('throttle_gain' in x for x in diffs)
    out, diffs = bench.plan_check(DESC, B, 5, params(tof_guard=3), data())
    assert section_scorecard.parse_description(out)['cand'] == 'EXP'
    out, diffs = bench.plan_check(DESC, B, 5, params(steer_gain=1.2), data())      # この日のステアは試しにしない
    d = section_scorecard.parse_description(out)
    assert diffs == [] and d['cand'] == 'B' and d['steer'] == '1.2' and d['official'] == '1'


def test_not_bench_or_already_exp_is_untouched():
    for desc in ('AI走行 B', '', 'ai_no=3 bench=x day=2026-10-23 block=1 cand=EXP memo=試し'):
        assert bench.plan_check(desc, 'other', 5, params(throttle_gain=1.4), data()) == (desc, [])


def test_block_id_differs_from_plan():
    desc = DESC.replace('block=1', 'block=7')                 # ブロック 7 は WGT の計画なのに cand=B
    out, diffs = bench.plan_check(desc, B, 5, params(), data())
    assert section_scorecard.parse_description(out)['cand'] == 'EXP' and 'WGT' in diffs[0]


def test_after_description_fills_fields_and_keeps_memo():
    out = bench.after_description(DESC.replace('memo=', 'memo=慣らし'), battery='3', v_before='8.2', v_after=7.61,
                                  reason='安全で中止', memo='DE で 2 回')
    d = section_scorecard.parse_description(out)
    assert d['battery'] == '3' and d['V_before'] == '8.2' and d['V_after'] == '7.61'
    assert d['memo'].startswith('安全で中止・DE で 2 回') and '慣らし' in d['memo'] and d['cand'] == 'B'
    assert bench.after_description(DESC) == DESC
    with pytest.raises(ValueError):
        bench.after_description(DESC, v_after='abc')


def test_api_writes_scene_descriptions_read_by_nightly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    log = '20261024-200000'
    (tmp_path / log).mkdir()
    (tmp_path / log / 'log.json').write_text(json.dumps({'description': DESC}), encoding='utf-8')
    for i in (1, 2):
        sd = tmp_path / log / ('%s-%04d' % (log, i))
        sd.mkdir()
        (sd / 'scene.json').write_text(json.dumps({'description': DESC, 'mode_id': 5}), encoding='utf-8')
    r = bench.api_bench_after({'log': log, 'battery': '#4', 'V_after': '7.5', 'reason': '回れない'})
    assert r['scenes'] == 2
    desc, info = section_scorecard.effective_description(log, DESC, tmp_path)
    d = section_scorecard.parse_description(desc)
    assert d['battery'] == '4' and d['V_after'] == '7.5' and '回れない' in d['memo'] and not info['conflicts']
    with pytest.raises(HTTPException):
        bench.api_bench_after({'log': '../x', 'reason': '回れない'})
    with pytest.raises(HTTPException):
        bench.api_bench_after({'log': log, 'reason': 'なんとなく'})
    with pytest.raises(HTTPException):
        bench.api_bench_after({'log': '20261024-999999', 'reason': '回れない'})
