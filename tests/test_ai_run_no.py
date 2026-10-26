"""AI 走行の番号(2026-10-23「AI走行にNoを付与、1から開始」)。web/bench.py の next_ai_no・commit_ai_no・with_ai_no と、
説明欄の道具(tools/section_scorecard.parse_description)が ai_no= を読めること

    cd ~/jetformer && python -m pytest -q tests/test_ai_run_no.py
"""

import json
import os
import sys

from jetformer.web import bench

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import section_scorecard  # noqa: E402


def test_first_is_one_and_counts_up(tmp_path):
    p = tmp_path / 'ai_run_no.json'
    assert bench.next_ai_no(p, '2026-10-23') == 1
    bench.commit_ai_no(1, p, '2026-10-23')
    assert bench.next_ai_no(p, '2026-10-23') == 2
    bench.commit_ai_no(2, p, '2026-10-23')
    assert json.loads(p.read_text()) == {'date': '2026-10-23', 'last': 2}


def test_new_day_starts_at_one(tmp_path):
    p = tmp_path / 'ai_run_no.json'
    bench.commit_ai_no(7, p, '2026-10-23')
    assert bench.next_ai_no(p, '2026-10-24') == 1


def test_broken_file_starts_at_one(tmp_path):
    p = tmp_path / 'ai_run_no.json'
    p.write_text('{oops')
    assert bench.next_ai_no(p, '2026-10-23') == 1
    p.write_text('[1, 2]')
    assert bench.next_ai_no(p, '2026-10-23') == 1


def test_peek_does_not_write(tmp_path):
    p = tmp_path / 'ai_run_no.json'
    assert bench.next_ai_no(p, '2026-10-23') == 1
    assert not p.exists()


def test_description_keeps_bench_fields():
    tpl = ('bench=real_bench_20261023 day=2026-10-23 block=2 cand=SLOW-SEC1 model=bench_20261022-slow-sec-s1 '
           'battery=#3 V_before=8.2 V_after=__ check=__ memo=慣らし 2 周')
    d = section_scorecard.parse_description(bench.with_ai_no(tpl, 1))
    assert d['ai_no'] == '1'
    assert d['bench'] == 'real_bench_20261023' and d['block'] == 2 and d['cand'] == 'SLOW-SEC1'
    assert d['battery'] == '3' and d['memo'] == '慣らし 2 周'
    assert bench.with_ai_no('', 3) == 'ai_no=3'
