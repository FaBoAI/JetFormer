"""critic(2026-10-26 夜)が直した夜の道具の試験。作り物の記録・作り物の行だけ。本物の data/raw・装置・GPU は使わない。

- 反証 1 回目 R1・tools:F1: 自動合わせ ON の回 3 でも collect.py の summarize が OFF と ON を同じ組にして比べを出す
- tools:F4: ON に launch_us の違う本が混ざったら注意の行
- 反証 1 回目 R2・tools:F2: standing_3lap で A4 が無い(A3+dist)発進に「確かめる」の印
- 反証 1 回目 R4: runs_1103 が読めない log.json の記録の名前を出す
- tools:F8: トリム・後ろの見張りが standing_3lap の設定の文と quick_eval_by_params の鍵に入る(0 なら今までと同じ)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> experiments/20261026_fix/fix2/critic/test_critic_tools.py
"""
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def collect():
    return load_mod('collect_critic', JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py')


def _row(log, state, gain, to_r1, auto=1, launch_us=250.0, tg=0.95, t_ai=1.0, hits=()):
    S = {'throttle_gain': tg, 'speed_ceiling': 1, 'ceil_left_mps': 4.1, 'speed_auto': auto, 'auto_target_mps': 4.27,
         'launch_boost': 0 if state == 'off' else 1, 'launch_us': launch_us}
    return {'log': log, 'no': log[-2:], 'session': '11/03 回3', 'model': 'B', 'gain': gain, 'settings': S, 't_ai': t_ai,
            'to_r1_s': to_r1, 'loss_s': to_r1 - 2.25, 'move_0p2_s': 0.3, 'slip': {'ratio': 1.01}, 'v_launch_at_lh': 2.0,
            'collisions': [], 'collisions_first_lap': list(hits), 'launch_on_s': 0.5 if state == 'on' else None,
            'launch_end_reason': '速さに届いた' if state == 'on' else None, 'launch_state': state, 'start_no': 1}


# ------------------------------------------------------------------ R1・tools:F1
def test_auto_on_round3_groups_off_and_on_by_slider_gain(collect):
    """回 3 を天井 + 自動合わせ ON で走る: フレームのゲインは電圧で本ごとに違う(0.965〜0.987)。前は 4 つの組になり比べが出なかった"""
    rows = [_row('20261103-130000', 'off', 0.965, 3.013), _row('20261103-130500', 'on', 0.972, 2.713),
            _row('20261103-131000', 'off', 0.979, 3.013), _row('20261103-131500', 'on', 0.987, 2.713)]
    s = collect.summarize(rows)
    assert len(s) == 1
    g = s[0]
    assert g['gain'] == 0.95 and set(g['by_launch']) == {'off', 'on'}
    assert g['compare']['d_to_r1_s'] == pytest.approx(-0.30) and g['compare']['all_ok'] is True
    assert g['by_launch']['on']['frame_gain'] == [0.972, 0.987]
    txt = '\n'.join(collect.summary_lines(s))
    assert '使ってよい' in txt and '組はスライダーのゲイン 0.95' in txt


def test_auto_off_still_groups_by_frame_gain(collect):
    """自動合わせ OFF は今まで通りフレームのゲインで組にする(スライダーが違えば別の組)"""
    rows = [_row('20261103-120000', 'off', 0.90, 3.0, auto=0, tg=0.90), _row('20261103-120500', 'on', 0.95, 2.7, auto=0, tg=0.95)]
    s = collect.summarize(rows)
    assert len(s) == 2 and all('compare' not in g for g in s)
    assert collect.gain_key({'speed_auto': 0, 'throttle_gain': 0.95}, 0.9) == 0.9
    assert collect.gain_key({'speed_auto': 1, 'throttle_gain': 0.95}, 0.97) == 0.95
    assert collect.gain_key({'speed_auto': 1, 'throttle_gain': None}, 0.97) == 0.97
    assert collect.gain_key(None, 0.97) == 0.97


def test_auto_on_compare_still_fails_when_first_lap_hit(collect):
    """組がまとまっても判定の決まりは同じ: ON の最初の周に衝突があれば「満たさない」"""
    rows = [_row('20261103-130000', 'off', 0.965, 3.013), _row('20261103-130500', 'on', 0.972, 2.713, hits=[[5.0, 4.2, 'LH']])]
    g = collect.summarize(rows)[0]
    assert g['compare']['checks']['first_lap_collisions_0'] is False and g['compare']['all_ok'] is False


@pytest.mark.parametrize('what, change, key', [
    ('起点まで −0.10 秒(−0.15 に届かない)', {'to_r1': 2.913}, 'to_r1'),
    ('空転の比 1.5', {'slip': 1.5}, 'slip'),
    ('LH の入口 +0.5 m/s', {'v_lh': 2.5}, 'lh_speed'),
])
def test_compare_fails_in_each_direction(collect, what, change, key):
    """反証 1 回目 tools:F3(判定の 4 条件の落ちる向きに試験が無い)。どれか 1 つが外れたら「満たさない」"""
    on = _row('20261103-130500', 'on', 0.972, change.get('to_r1', 2.713))
    if 'slip' in change:
        on['slip'] = {'ratio': change['slip']}
    if 'v_lh' in change:
        on['v_launch_at_lh'] = change['v_lh']
    g = collect.summarize([_row('20261103-130000', 'off', 0.965, 3.013), on])[0]
    assert g['compare']['checks'][key] is False, what
    assert all(v for k, v in g['compare']['checks'].items() if k != key), what
    assert g['compare']['all_ok'] is False and '満たさない' in '\n'.join(collect.summary_lines([g]))


# ------------------------------------------------------------------ tools:F4
def test_mixed_launch_us_gets_a_warning(collect):
    rows = [_row('20261103-130000', 'off', 0.95, 3.0, auto=0), _row('20261103-130500', 'on', 0.95, 2.7, auto=0),
            _row('20261103-131000', 'on', 0.95, 2.8, auto=0, launch_us=200.0)]
    s = collect.summarize(rows)
    assert len(s) == 1 and s[0]['mixed_launch_us'] == [200.0, 250.0]
    assert s[0]['by_launch']['off']['launch_us'] == []
    assert '注意: ON に launch_us の違う本が混ざっている(200・250 µs)' in '\n'.join(collect.summary_lines(s))
    s1 = collect.summarize(rows[:2])
    assert 'mixed_launch_us' not in s1[0] and '注意' not in '\n'.join(collect.summary_lines(s1))


# ------------------------------------------------------------------ R2・tools:F2
def test_a3dist_start_gets_a_check_note(tmp_path):
    import standing_3lap as s3
    from test_standing_3lap import write_standing_log, analyze, valid_rows
    log = '20261103-090300'
    write_standing_log(tmp_path, log, human_after=29.3, stop_after=29.4)
    s = valid_rows(analyze(tmp_path, log))[0]
    assert s['method'] == 'A3+dist' and s['valid'] and s['counted']          # 数えるのはそのまま
    assert s['note'] == s3.A3DIST_NOTE and 'ストップウォッチ' in s['note']
    line = [x for x in s3.standing_table([{'log': log, 'standing': [s], 'settings_text': 'B'}]) if log in x][0]
    assert 'ok(A3+dist)・確かめる: A4 が無い' in line
    log2 = '20261103-090310'                                                  # A4 まで AI 側 → 印なし
    write_standing_log(tmp_path, log2)
    s2 = valid_rows(analyze(tmp_path, log2))[0]
    assert s2['method'] == 'anchor' and 'A4 が無い' not in (s2.get('note') or '')


# ------------------------------------------------------------------ tools:F8
def test_trim_and_back_guard_in_settings_and_key(tmp_path):
    import standing_3lap as s3
    qe = load_mod('qebp_critic', JF / 'tools' / 'quick_eval_by_params.py')
    base = {'model': 'B', 'throttle_gain': 0.9, 'steer_gain': 1.0}
    assert s3.settings_text(dict(base, steer_trim_us=0.0, rec_back_guard=0)) == 'B・0.90'
    assert s3.settings_text(dict(base, steer_trim_us=15.0, rec_back_guard=1)) == 'B・0.90・トリム +15・後ろの見張り'
    k0, t0 = qe.key_of({'settings': dict(base, steer_trim_us=0.0)})
    k1, t1 = qe.key_of({'settings': dict(base, steer_trim_us=15.0)})
    kn, tn = qe.key_of({'settings': dict(base)})                               # 古い記録(欄なし)= 0 と同じ
    assert k0 == kn and k0 != k1 and 'トリム +15' in t1 and 'トリム' not in t0
    assert qe.key_of({'settings': dict(base, steer_trim_us=15.0)}, ignore=('trim',))[0] == \
        qe.key_of({'settings': dict(base)}, ignore=('trim',))[0]
    assert 'trim' in qe.PARTS and 'steer_trim_us' in s3.SETTING_KEYS and 'rec_back_guard' in s3.SETTING_KEYS


# ------------------------------------------------------------------ R4
def test_runs_1103_names_unreadable_log_json(tmp_path, monkeypatch):
    m = load_mod('runs_1103_critic', JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
    raw = tmp_path / 'raw'
    (raw / '20261103-100000').mkdir(parents=True)
    (raw / '20261103-100000' / 'log.json').write_text('')                    # 電源断の 0 バイト
    (raw / '20261103-100500').mkdir()
    (raw / '20261103-100500' / 'log.json').write_text('{壊れた')
    (raw / '20261103-101000').mkdir()
    (raw / '20261103-101000' / 'log.json').write_text(json.dumps({'drive': {'run': 'r', 'params': {'throttle_gain': 0.9}},
                                                                  'description': 'ai_no=1 day=2026-11-03 block=1'}))
    m.RAW = raw
    monkeypatch.setattr(sys, 'argv', ['runs_1103.py', '--day', '20261103', '--json'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main()
    out = buf.getvalue().strip().splitlines()
    rows = json.loads(out[-1])                                                # --json の行は今まで通り最後
    assert [r['log'] for r in rows] == ['20261103-101000']
    line = [x for x in out if x.startswith('読めない記録')][0]
    assert '2 本' in line and '20261103-100000(log.json が 0 バイト)' in line and '20261103-100500(JSONDecodeError)' in line
