"""tools/bench_day2.py(10/23 の夜に 2 日目のブロックを data/bench_candidates.json へ書く。2026-10-21 Fix-all)の試験

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_bench_day2.py

確かめること:
  - 並び(protocol day2): 2 本 B・T1・T2・B・T2・T1・B / 1 本 B・T1・B・T1・B / 0 本 B・B・B
  - A 案: 最後に B+cap・B。B+cap の候補の行(run = B)を足す。条件の確かめ(--a-conditions-ok)が無ければ書かない
  - B 案: T のブロックだけ counted_ai_s 300。bench_report の block_budget がその値を読む
  - 1 日目・基準・ほかの候補は変えない。--write なしでは書かない。書くときは控えを残す
  - 画面のパネル(jetformer/web/bench.load_candidates)が書いたファイルを読め、B+cap のブロックが走れる(B の学習結果があるとき)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import bench_day2 as bd  # noqa: E402

B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'


def bench_doc():
    return {'version': 1, 'bench': 'real_bench_20261023', 'protocol_sha256': 'x', 'created_at': '2026-10-21T13:00:00',
            'baseline': {'mode_id': 5, 'params': {'steer_gain': 1.0}},
            'candidates': [{'id': 'B', 'label': 'モデルB(基準)', 'run': B_RUN, 'risk': 'low'},
                           {'id': 'SLOW', 'label': '遅く入る型', 'run': 'bench_20261021-slow-s0', 'risk': 'low'},
                           {'id': 'ENS', 'label': '2本の平均', 'run': 'bench_20261021-ens-p2', 'risk': 'mid'}],
            'days': [{'date': '2026-10-23', 'blocks': [{'block': 1, 'id': 'B'}, {'block': 2, 'id': 'SLOW'}, {'block': 3, 'id': 'ENS'},
                                                      {'block': 4, 'id': 'B'}]},
                     {'date': '2026-10-24', 'blocks': [], 'note': '10/23 の夜に決める'}],
            'description_template': 'bench=real_bench_20261023 day={date} block={block} cand={id} model={run} memo='}


def ids(doc, date='2026-10-24'):
    return [b['id'] for d in doc['days'] if d['date'] == date for b in d['blocks']]


def test_sequences_and_plan_b():
    doc = bench_doc()
    new, _ = bd.build_day2(doc, 'SLOW', 'ENS', 'B')
    assert ids(new) == ['B', 'SLOW', 'ENS', 'B', 'ENS', 'SLOW', 'B']
    day = [d for d in new['days'] if d['date'] == '2026-10-24'][0]
    assert [b.get('counted_ai_s') for b in day['blocks']] == [None, 300, 300, None, 300, 300, None]
    assert day['plan'] == 'B' and 'T1 = SLOW' in day['note']
    assert new['days'][0] == doc['days'][0] and new['candidates'] == doc['candidates'] and new['baseline'] == doc['baseline']
    assert ids(doc) == []                                                     # 元の dict は変えない
    assert ids(bd.build_day2(doc, 'ENS', None, 'B')[0]) == ['B', 'ENS', 'B', 'ENS', 'B']
    assert ids(bd.build_day2(doc, None, None, 'B')[0]) == ['B', 'B', 'B']
    import bench_report as br
    assert br.block_budget(new, '2026-10-24', 2) == 300.0 and br.block_budget(new, '2026-10-24', 1) == 240.0


def test_plan_a_adds_cap_row():
    new, msgs = bd.build_day2(bench_doc(), 'SLOW', 'ENS', 'A')
    assert ids(new) == ['B', 'SLOW', 'ENS', 'B', 'ENS', 'SLOW', 'B', 'B+cap', 'B']
    cap = [c for c in new['candidates'] if c['id'] == 'B+cap']
    assert len(cap) == 1 and cap[0]['run'] == B_RUN
    assert all('counted_ai_s' not in b for d in new['days'] for b in d['blocks'])
    again, _ = bd.build_day2(new, 'SLOW', 'ENS', 'A')                          # 2 回目でも行は 1 つ
    assert len([c for c in again['candidates'] if c['id'] == 'B+cap']) == 1


def test_bad_ids_refused():
    for t1, t2 in (('NOPE', None), ('B', 'SLOW'), ('SLOW', 'SLOW'), (None, 'ENS'), ('B+cap', None)):
        with pytest.raises(ValueError):
            bd.build_day2(bench_doc(), t1, t2, 'B')


def test_main_write_and_guard(tmp_path, capsys):
    p = tmp_path / 'bench_candidates.json'
    p.write_text(json.dumps(bench_doc(), ensure_ascii=False), encoding='utf-8')
    before = p.read_text(encoding='utf-8')
    assert bd.main(['--bench', str(p), '--t1', 'SLOW', '--t2', 'ENS', '--plan', 'B']) == 0
    assert p.read_text(encoding='utf-8') == before                            # 見るだけ
    assert bd.main(['--bench', str(p), '--t1', 'SLOW', '--t2', 'ENS', '--plan', 'A', '--write']) == 2      # A 案の確かめが無い
    assert p.read_text(encoding='utf-8') == before
    assert bd.main(['--bench', str(p), '--t1', 'SLOW', '--t2', 'ENS', '--plan', 'A', '--a-conditions-ok', '--write']) == 0
    new = json.loads(p.read_text(encoding='utf-8'))
    assert ids(new)[-2:] == ['B+cap', 'B']
    baks = list(tmp_path.glob('bench_candidates.json.*.bak'))
    assert len(baks) == 1 and baks[0].read_text(encoding='utf-8') == before
    # report.json から
    rep = tmp_path / 'report.json'
    rep.write_text(json.dumps({'days': [{'day': '2026-10-23', 'ranking': {'T1': 'ENS', 'T2': None, 'lambda': 1.0}}]}), encoding='utf-8')
    assert bd.main(['--bench', str(p), '--report', str(rep), '--plan', 'B', '--write']) == 0
    assert ids(json.loads(p.read_text(encoding='utf-8'))) == ['B', 'ENS', 'B', 'ENS', 'B']
    assert bd.main(['--bench', str(p), '--report', str(rep), '--t1', 'SLOW', '--plan', 'B']) == 2


def test_panel_reads_written_file(tmp_path):
    from jetformer import drive
    from jetformer.web import bench
    if drive.model_info(B_RUN) is None:
        pytest.skip('モデル B の学習結果がありません')
    p = tmp_path / 'bench_candidates.json'
    new, _ = bd.build_day2(bench_doc(), 'SLOW', 'ENS', 'A')
    p.write_text(json.dumps(new, ensure_ascii=False), encoding='utf-8')
    out = bench.load_candidates(p)
    assert out['ok'] and not out['warnings']
    day2 = [d for d in out['days'] if d['date'] == '2026-10-24'][0]
    cap = [b for b in day2['blocks'] if b['id'] == 'B+cap'][0]
    assert cap['drivable'] and cap['run'] == B_RUN and 'cand=B+cap' in cap['description']
