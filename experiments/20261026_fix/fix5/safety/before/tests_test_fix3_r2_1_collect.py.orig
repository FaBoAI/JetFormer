"""直す 2 回目・1 組目(2026-10-27 fix3。反証 2 回目の指摘 r2-rec2-1)の試験。夜の道具 ⑤ experiments/20261026_fix/diag_launch/collect.py。

  r2-rec2-1: まとめの組の鍵にゲイン(自動合わせ OFF はフレームのゲイン)が入る。手順書 4 章 3 は毎本ゲインを電圧の表で入れ直すので、
    回 3 の 8 本でゲインが変わると組が分かれ、「ON − OFF」の行が出ない・1 本対 1 本の答えが別々に出た(手順書 6 章の
    「4 本対 4 本の中央値」にならない)。
  直し: --ignore-gain(組の鍵からゲインを外す。組の行にゲインの幅)。--ignore-gain なしで、同じ回・同じモデル・同じスイッチなのに
    ゲインの違う組が 2 つ以上あり、その中に発進の前進 ON の本があれば、それぞれの組に「注意: ゲインが本ごとに違うので組が分かれた」。
    その場面の無い日は、出力(JSON・標準出力)が直す前の写し tests/fixtures/fix3_r2_1_before/collect.py とまるごと同じ。

作り物の記録(tests/test_fix3_r1_2.Day。反証の probe_gain_split.py と同じ作り)を tmp に書いて流す。data/raw・labels は読まない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix3_r2_1_collect.py
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from test_fix3_r1_2 import Day, ON, PRM

JF = Path(__file__).resolve().parents[1]
BEFORE = JF / 'tests' / 'fixtures' / 'fix3_r2_1_before' / 'collect.py'


def test_before_copy():
    sums = dict(reversed(x.split()) for x in (BEFORE.parent / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert hashlib.sha256(BEFORE.read_bytes()).hexdigest() == sums['collect.py']


def set_gain(d, g):
    for p in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(p))
        z['gains'][:, 1] = g
        np.savez(p, **z)


def day(tmp, spec):
    """spec: [(params, フレームのゲイン or None, kw)]。block 12 + i(3-1〜3-8)"""
    D = Day(tmp)
    logs = []
    for i, (params, gf, kw) in enumerate(spec):
        lg, ld = D.add(12 + i, params, **kw)
        set_gain(ld, gf if gf is not None else params['throttle_gain'])
        logs.append(lg)
    return D, logs


def P(i, g, **x):
    """3-1 から OFF・ON の交互(回 3)"""
    return (dict(ON if i % 2 else PRM, throttle_gain=g, **x), None, {})


def _no_path(r):
    """標準出力の最後の「出力 <JSON の場所>」の行を外す(場所は流すたびに違う)"""
    j, out, txt, m = r
    return j, '\n'.join(x for x in out.splitlines() if not x.startswith('出力 ')), txt, m


def run3(D, name, args=()):
    """今の collect.py・--ignore-gain・直す前の写し"""
    new = D.run(name='n_' + name, args=args)
    ig = D.run(name='i_' + name, args=list(args) + ['--ignore-gain'])
    old = D.run(path=BEFORE, name='o_' + name, args=args)
    return _no_path(new), _no_path(ig), _no_path(old)


def strip_split(j):
    for g in j['summary']:
        g.pop('gain_split', None)
    return j


def compare_lines(out):
    return [x for x in out.splitlines() if 'ON − OFF' in x]


def test_same_gain_unchanged(tmp_path):
    """対照: 8 本とも 0.90。組は 1 つ(4 本対 4 本)。直す前の写しと JSON・標準出力がまるごと同じ。--ignore-gain でも比べは同じ"""
    D, _ = day(tmp_path, [P(i, 0.90) for i in range(8)])
    (jn, on, tn, _), (ji, oi, ti, _), (jo, oo, to, _) = run3(D, 'a')
    assert tn == to and on == oo
    assert len(jn['summary']) == 1 and jn['summary'][0]['by_launch']['on']['n'] == 4 and jn['summary'][0]['by_launch']['off']['n'] == 4
    assert 'gain_split' not in jn['summary'][0] and '注意: ゲインが本ごとに違う' not in on
    g = ji['summary'][0]
    assert g['gain'] is None and g['gain_ignored'] is True and g['gain_range'] == [0.9, 0.9] and ji['summary_ignore_gain'] is True
    assert g['compare'] == jn['summary'][0]['compare'] and g['by_launch'] == jn['summary'][0]['by_launch']
    assert '11/03 回3 B g0.90(ゲインをまとめた --ignore-gain)' in oi
    assert ji['launches'] == jn['launches']                       # 発進ごとの行は変えない


def test_one_voltage_step_split_warned_and_merged(tmp_path):
    """電圧の表で 1 回だけ段が変わる(3-1〜3-4 0.90、3-5〜3-8 0.92)。直す前: 2 組・2 本対 2 本の答えが別々。今: 同じ 2 組に
    「注意: ゲインが本ごとに違うので組が分かれた(0.90・0.92)」(ほかは前と同じ)。--ignore-gain: 1 組・4 本対 4 本"""
    D, _ = day(tmp_path, [P(i, 0.90 if i < 4 else 0.92) for i in range(8)])
    (jn, on, _, _), (ji, oi, _, _), (jo, oo, _, _) = run3(D, 'b')
    assert [g['gain'] for g in jo['summary']] == [0.9, 0.92] and len(compare_lines(oo)) == 2           # 指摘の再現
    assert [g['gain_split'] for g in jn['summary']] == [[0.9, 0.92], [0.9, 0.92]]
    assert strip_split(jn) == jo
    assert on.count('注意: ゲインが本ごとに違うので組が分かれた(0.90・0.92)') == 2 and '--ignore-gain' in on
    assert [x for x in on.splitlines() if '注意: ゲインが本ごとに違う' not in x] == oo.splitlines()
    assert len(ji['summary']) == 1
    g = ji['summary'][0]
    assert g['by_launch']['on']['n'] == 4 and g['by_launch']['off']['n'] == 4 and g['gain_range'] == [0.9, 0.92]
    assert 'compare' in g and len(compare_lines(oi)) == 1 and 'g0.90〜0.92(ゲインをまとめた --ignore-gain)' in oi
    assert '注意: ゲインが本ごとに違う' not in oi


def test_each_run_different_gain(tmp_path):
    """毎本 +0.01(0.90〜0.97)。直す前: 8 組・どれも OFF だけか ON だけで「ON − OFF」の行が 1 つも出ない。--ignore-gain: 1 組で出る"""
    D, _ = day(tmp_path, [P(i, round(0.90 + 0.01 * i, 2)) for i in range(8)])
    (jn, on, _, _), (ji, oi, _, _), (jo, oo, _, _) = run3(D, 'c')
    assert len(jo['summary']) == 8 and compare_lines(oo) == []                                      # 指摘の再現
    assert all(len(g['gain_split']) == 8 for g in jn['summary']) and on.count('注意: ゲインが本ごとに違う') == 8
    assert len(ji['summary']) == 1 and len(compare_lines(oi)) == 1
    assert ji['summary'][0]['gain_range'] == [0.9, 0.97]


def test_on_hit_in_one_gain_group_fails_merged(tmp_path):
    """B に、0.92 の組の ON(3-6)だけ R1 の前に 6G で当たる。直す前: 0.90 は「使ってよい」・0.92 は「満たさない」と食い違う。
    --ignore-gain: 1 組で「満たさない」(当たった ON は R1 に届かなかった本として数える)"""
    spec = [P(i, 0.90 if i < 4 else 0.92) for i in range(8)]
    spec[5] = (spec[5][0], None, dict(hits=[(3.0, 6.0)], stop_after=3.1, human_after=3.6))
    D, _ = day(tmp_path, spec)
    (jn, on, _, _), (ji, oi, _, _), (jo, oo, _, _) = run3(D, 'd')
    assert [(g['compare'] or {}).get('all_ok') for g in jo['summary']] == [True, False]               # 指摘の再現
    g = ji['summary'][0]
    assert len(ji['summary']) == 1 and g['compare']['all_ok'] is False and g['not_reached']['on']['collisions'] == 1
    assert '→ 満たさない' in oi and '→ 使ってよい' not in oi


@pytest.mark.parametrize('slider', [(1.0, 1.0), (1.0, 1.03)])
def test_auto_tuning(tmp_path, slider):
    """自動合わせ ON(天井 ON)。スライダーが同じなら前から 1 組(gain_key)で、注意も出ない(前と同じ)。途中でスライダーを 1.00 →
    1.03 にしたら 2 組に分かれ、注意が出る。--ignore-gain で 1 組"""
    spec = [(dict(ON if i % 2 else PRM, throttle_gain=slider[0] if i < 4 else slider[1], speed_ceiling=1, speed_auto=1),
             0.965 + 0.002 * i, {}) for i in range(8)]
    D, _ = day(tmp_path, spec)
    (jn, on, tn, _), (ji, oi, _, _), (jo, oo, to, _) = run3(D, 'e')
    if slider[0] == slider[1]:
        assert len(jn['summary']) == 1 and tn == to and on == oo
    else:
        assert len(jo['summary']) == 2 and all(g['gain_split'] == [1.0, 1.03] for g in jn['summary'])
        assert strip_split(jn) == jo
    assert len(ji['summary']) == 1 and ji['summary'][0]['gain_range'] == [1.0, slider[1]]
    assert '(自動合わせ ON: 組はスライダーのゲイン %s。' % ('1.00' if slider[0] == slider[1] else '1.00〜1.03') in oi


def test_off_only_day_with_gain_changes_unchanged(tmp_path):
    """発進の前進 ON の本が無い日(10/23・10/24 のような日)は、ゲインで組が分かれても注意を出さない(出力は前と同じ)"""
    D, _ = day(tmp_path, [(dict(PRM, throttle_gain=g), None, {}) for g in (0.90, 0.92, 0.95, 0.90)])
    (jn, on, tn, _), _, (jo, oo, to, _) = run3(D, 'f')
    assert len(jn['summary']) == 3 and tn == to and on == oo


def test_first_only_and_skip_with_ignore_gain(tmp_path):
    """--first-only・--skip と一緒に使える(まとめだけ変わる)"""
    D, logs = day(tmp_path, [P(i, 0.90 if i < 4 else 0.92) for i in range(8)])
    (jn, on, _, _), (ji, oi, _, _), _ = run3(D, 'g', args=('--first-only', '--skip', logs[7]))
    g = ji['summary'][0]
    assert len(ji['summary']) == 1 and g['by_launch']['on']['n'] == 3 and g['by_launch']['off']['n'] == 4
    assert ji['summary_first_only'] is True and ji['summary_skip'] == [logs[7]]
