"""実際の設定で分ける早い集計(tools/quick_eval_by_params.py、2026-10-26 fix2)の試験。実機・GPU 不要(作り物の記録)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -m pytest -q tests/test_quick_eval_by_params.py
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

P1 = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
      'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0}


def p(**kw):
    return dict(P1, **kw)


@pytest.fixture(scope='module')
def day(tmp_path_factory):
    """11/3 の形: 全部 cand=EXP。P1・天井 4.1・天井 + 自動・発進 ON(出た)・発進 ON(出ず)・P1(もう 1 本)・0 バイトの log.json"""
    root = tmp_path_factory.mktemp('raw')
    spec = [('20261103-100000', 1, p(), None, {}),
            ('20261103-100500', 2, p(speed_ceiling=1), None, {}),
            ('20261103-101000', 3, p(speed_ceiling=1, speed_auto=1), None, {}),
            ('20261103-101500', 4, p(launch_boost=1), (1.0, 0.5, 1), {}),
            ('20261103-102000', 5, p(launch_boost=1), None, {}),
            ('20261103-102500', 6, p(), None, {'hits': [(16.0, 5.0)]}),
            ('20261103-103000', 7, p(speed_ceiling=1, ceil_left_mps=4.2), None, {})]
    for log, no, prm, launch, kw in spec:
        write_standing_log(root, log, params=prm, launch=launch, run='runB',
                           description='ai_no=%d day=2026-11-03 block=%d cand=EXP model=runB tune=t%d' % (no, no, no), **kw)
    bad = root / '20261103-103500'
    bad.mkdir()
    (bad / 'log.json').write_bytes(b'')
    bench = root / 'bench.json'
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}]}), encoding='utf-8')
    logs, skipped = s3.day_logs('20261103', root)
    res = s3.run(logs, root, root / 'no_labels', str(bench))
    return root, bench, res, skipped


def test_groups_by_actual_settings_not_by_cand(day):
    root, bench, res, skipped = day
    assert [r['cand'] for r in res] == ['EXP'] * 7
    assert dict(skipped) == {'20261103-103500': 'log.json が読めない(0 バイトか壊れている)'}
    G = {g['text']: g for g in qp.group_rows(res)}
    assert set(G) == {'B・0.90・天井 OFF・発進 OFF', 'B・0.90・天井 4.1・発進 OFF', 'B・0.90・天井 4.1・自動 4.27・発進 OFF',
                      'B・0.90・天井 OFF・発進 250', 'B・0.90・天井 OFF・発進 ON・出ず', 'B・0.90・天井 4.2・発進 OFF'}
    g = G['B・0.90・天井 OFF・発進 OFF']
    assert g['runs'] == 2 and g['ai_no'] == ['1', '6'] and g['cand'] == ['EXP']
    assert g['passes'] == 10 and g['clean_laps'] == 7 and g['strong'] == 1      # 6 番の 2 周目に 5G
    assert g['standing_n'] == 2 and g['standing_best_s'] == pytest.approx(27.0, abs=0.03)
    assert g['standing_clean_n'] == 1 and g['standing_le25'] == 0
    assert g['window_best_clean_s'] == pytest.approx(27.0, abs=0.05)
    assert G['B・0.90・天井 OFF・発進 250']['runs'] == 1


def test_ignore_merges_parts(day):
    _, _, res, _ = day
    G = {g['text']: g for g in qp.group_rows(res, ignore=('launch',))}
    assert G['B・0.90・天井 OFF']['runs'] == 4                                # 発進の ON/OFF をまとめた
    G = {g['text']: g for g in qp.group_rows(res, ignore=('launch', 'ceiling', 'auto'))}
    assert list(G) == ['B・0.90'] and G['B・0.90']['runs'] == 7


def test_launch_state_and_flags(day):
    _, _, res, _ = day
    by = {r['ai_no']: r for r in res}
    assert qp.launch_state(by['1']) == 'off' and qp.launch_state(by['4']) == 'on'
    assert qp.launch_state(by['5']) == 'on_not_started'
    assert '発進の前進が出ていない' in qp.flags_of(by['5'])
    assert qp.flags_of(by['1']) == []
    # フレームのゲインが log.json と違う(自動合わせ OFF): 作り物の記録のフレームは 0.90
    r = dict(by['1'], settings=dict(by['1']['settings'], throttle_gain=0.95))
    assert 'フレームのゲイン 0.900' in qp.flags_of(r)
    r = dict(r, settings=dict(r['settings'], speed_auto=1))
    assert not any(f.startswith('フレームのゲイン') for f in qp.flags_of(r))     # 自動合わせはゲインを変える


def test_old_records_count_as_off(tmp_path):
    """10/24 までの記録(天井・発進の欄が無い)は OFF と同じ鍵"""
    write_standing_log(tmp_path, '20261024-100000', params={'throttle_gain': 0.9, 'steer_gain': 1.0}, run='runB')
    write_standing_log(tmp_path, '20261024-100500', params=p(), run='runB')
    res = s3.run(['20261024-100000', '20261024-100500'], tmp_path, tmp_path / 'no_labels', str(tmp_path / 'none.json'))
    G = qp.group_rows(res)
    assert len(G) == 1 and G[0]['runs'] == 2 and G[0]['text'] == 'runB・0.90・天井 OFF・発進 OFF'


def test_markdown_and_main(day, capsys):
    root, bench, res, _ = day
    md, runs, groups = qp.to_markdown(res, '20261103', None)
    assert '## 2. 実際の設定ごと' in md and '| B・0.90・天井 OFF・発進 ON・出ず | 1 | 5 | EXP |' in md
    assert len(runs) == 7 and runs[0]['settings'] == 'B・0.90・天井 OFF・発進 OFF'
    out = root / 'out'
    rc = qp.main(['0000', '--day=20261103', '--raw', str(root), '--labels', str(root / 'no_labels'), '--bench', str(bench),
                  '--out', str(out)])
    assert rc == 0
    j = json.loads((out / 'by_params.json').read_text(encoding='utf-8'))
    assert j['schema'] == qp.SCHEMA and len(j['runs']) == 7 and len(j['groups']) == 6
    assert (out / 'by_params.md').read_text(encoding='utf-8').startswith('# 実際の設定で分けた早い集計')
    assert '飛ばした記録: 20261103-103500' in capsys.readouterr().out
    assert qp.main(['--day', '20261103', '--since', '1100', '--raw', str(root)]) == 1
    assert qp.main(['--day', '20261103', '--raw', str(root), '--ignore', 'nope']) == 2
