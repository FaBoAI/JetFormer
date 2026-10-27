"""夜の道具の直す 3 回目(2026-10-26 fix3・係 tools)の試験: tools/standing_3lap.py・tools/quick_eval_by_params.py。
作り物の記録・作り物の行だけ(tmp)。本物の data/raw・labels は読まない・書かない。GPU・torch は使わない。

- 反証 1 回目 tools:F5  standing_3lap の印(確かめる)の付いた止まってから 3 周・3 周の窓が、quick_eval_by_params の
                        1 章の印に出て、2 章の最速・中央値・25 秒以下に入らない(「(確かめる N: 値)」で数え分ける)
- 反証 1 回目 tools:F7  発進の前進の ON/OFF を発進ごとに: 同じ本の中の「出ず」の発進を「発進 250」の組に数えない
- 反証 1 回目 tools:F6  試験の効かなかった分かれ道(standing_3lap 9 か所・quick_eval_by_params 8 か所)に 1 つずつ場面

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_tools_fix3.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import label_events as le  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_standing_3lap import analyze, write_standing_log  # noqa: E402

V = 2.0
S0 = {'model': 'B', 'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'ceil_all_mps': 4.5,
      'speed_auto': 0, 'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2,
      'zone_caps': 0, 'mode_id': 5}


# ------------------------------------------------------------------ 作り物の行(quick_eval_by_params)
def start(t_ai, from_move, valid=True, coll_before=0, note=None, launched=None, to_r1=3.0, move_after=0.5, ncoll=0):
    """standing_3lap の発進の行。launched: None = 記録なし / True = 出た / False = 出ず(前が近い)"""
    lr = {'recorded': launched is not None, 'on_frames': 15 if launched else 0, 'on_s': 0.5 if launched else None,
          'end_reason': '速さに届いた' if launched else None,
          'events': [[t_ai, 'start', '']] if launched else ([[t_ai, 'skip', '前が近い']] if launched is False else [])}
    s = {'t_ai': t_ai, 'valid': valid, 'counted': valid and coll_before == 0, 'coll_before': coll_before,
         'from_move_s': from_move, 'to_r1_s': to_r1, 'move_after_s': move_after, 'collisions': {'n': ncoll, 'strong': 0},
         'launch': lr, 'start_no': 1}
    if note:
        s['note'] = note
    return s


def fake_res(log, S=None, standing=(), windows=(), clean=(), ai_no='1'):
    clean = list(clean)
    return {'log': log, 'ai_no': ai_no, 'cand': 'EXP', 'block': 1, 'tune': None, 'settings': dict(S0, **(S or {})),
            'standing': list(standing), 'windows': list(windows), 'zero_byte': [], 'midrun': [], 'speed_ctrl_flags': [],
            'laps': {'passes': len(clean), 'laps': len(clean), 'clean': len(clean), 'clean_laps_s': clean,
                     'best_clean_s': min(clean) if clean else None, 'median_clean_s': float(np.median(clean)) if clean else None,
                     'r1_entry_median_erpm': None, 'entries_erpm': []},
            'collisions': {'all': 0, 'strong': 0, 'max_g': 0.0, 'n_ge8': 0, 'by_place': {}},
            'flying': {'best_s': min([w['total_s'] for w in windows], default=None), 'best_clean_s': None},
            'ai_s': 30.0, 'takeovers_failure': 0, 'vbat': {}}


def win(total, suspect=False, kind='きれい'):
    return {'t_start': 4.0, 'laps_s': [total / 3] * 3, 'total_s': total, 'kind': kind, 'suspect': suspect, 'gmax': 0.0}


# ------------------------------------------------------------------ tools:F5
def test_f5_marks_reach_flags_and_stay_out_of_best_and_le25():
    """10/23 No.17 の形: 印の付いた止まってから 3 周(24.50。A4 が無い)と、7 秒より速い周を含む窓(24.00)。
    前は 1 章の印が無く、2 章の最速・25 秒以下に入った(24.50 が「25 秒以下 1」に見えた)"""
    r = fake_res('20261103-100000', standing=[start(1.0, 24.5, note=s3.A3DIST_NOTE), start(40.0, 26.0)],
                 windows=[win(24.0, suspect=True), win(26.2)], clean=[8.7, 8.8])
    fl = qp.flags_of(r)
    assert '確かめる(止まってから 3 周 24.50)' in fl and '確かめる(3 周の窓 24.00: 7 秒より速い周)' in fl
    g = qp.group_rows([r])[0]
    assert g['standing_n'] == 1 and g['standing_best_s'] == 26.0 and g['standing_le25'] == 0 and g['standing_list_s'] == [26.0]
    assert g['standing_check_s'] == [24.5] and g['standing_clean_n'] == 1
    assert g['windows'] == 1 and g['window_best_s'] == 26.2 and g['window_best_clean_s'] == 26.2 and g['window_check_s'] == [24.0]
    md, _, _ = qp.to_markdown([r], '20261103', None)
    row = [x for x in md.splitlines() if x.startswith('| B・0.90')][0]
    assert '1・26.20・26.20・26.20(確かめる 1: 24.00)' in row and '1・26.00・26.00・1・0・0(確かめる 1: 24.50)' in row
    assert '(確かめる N: 値)= standing_3lap の印のある値' in md


def test_f5_slip_note_also_counts_as_check_and_no_marks_unchanged():
    """空転の比の印も「確かめる」に数える。印が無ければ新しい欄も文も出ない(前の出力と同じ形)"""
    r = fake_res('20261103-100500', standing=[start(1.0, 26.5, note='空転の比 1.45(> 1.3): d_pre が長く出て…')])
    assert qp.group_rows([r])[0]['standing_check_s'] == [26.5]
    r0 = fake_res('20261103-101000', standing=[start(1.0, 26.5)], windows=[win(26.0)])
    g = qp.group_rows([r0])[0]
    assert not {'standing_check_s', 'window_check_s', 'starts_from_other_runs'} & set(g)
    assert not any(f.startswith('確かめる') for f in qp.flags_of(r0))
    md, _, _ = qp.to_markdown([r0], '20261103', None)
    assert '確かめる' not in md and 'ほかの本の発進' not in md


def test_f5_end_to_end_a3dist_start_is_checked(tmp_path):
    """作り物の記録で: 3 周目の線のあと R1 の前で人側(A4 が無い)→ standing_3lap の印 → by_params の 2 章に入らない"""
    log = '20261103-090300'
    write_standing_log(tmp_path, log, human_after=29.3, stop_after=29.4, run='runB')
    r = analyze(tmp_path, log)
    s = [x for x in r['standing'] if x.get('valid')][0]
    assert s['method'] == 'A3+dist' and s3.counted(s)
    g = qp.group_rows([r])[0]
    assert g['standing_n'] == 0 and g['standing_check_s'] == [s['from_move_s']]
    assert any(f.startswith('確かめる(止まってから 3 周') for f in qp.flags_of(r))


# ------------------------------------------------------------------ tools:F7
LB = {'launch_boost': 1}


def test_f7_not_started_start_in_an_on_record_goes_to_the_not_started_group():
    """同じ本の中で 1 回目の発進は出た・2 回目(人が戻したあと)は出ず。前は 2 回目の 28.20 も「発進 250」の組に入った"""
    a = fake_res('20261103-110000', LB, standing=[start(1.0, 27.0, launched=True), start(40.0, 28.2, launched=False)], ai_no='1')
    b = fake_res('20261103-110500', LB, standing=[start(1.0, 29.0, launched=False)], ai_no='2')
    assert qp.launch_state(a) == 'on' and qp.launch_state(b) == 'on_not_started'
    assert [qp.start_launch_state(a, s) for s in a['standing']] == ['on', 'on_not_started']
    G = {g['text']: g for g in qp.group_rows([a, b])}
    on, ns = G['B・0.90・天井 OFF・発進 250'], G['B・0.90・天井 OFF・発進 ON・出ず']
    assert on['runs'] == 1 and on['standing_list_s'] == [27.0] and on['launch_n'] == 1
    assert ns['runs'] == 1 and ns['standing_list_s'] == [28.2, 29.0] and ns['launch_n'] == 2
    assert ns['starts_from_other_runs'] == ['20261103-110000@40.00'] and 'starts_from_other_runs' not in on
    assert '発進の前進が混ざっている(出た 1・出ず 1)' in qp.flags_of(a)
    assert not any(f.startswith('発進の前進が混ざっている') for f in qp.flags_of(b))
    md, _, _ = qp.to_markdown([a, b], '20261103', None)
    assert '| B・0.90・天井 OFF・発進 ON・出ず・ほかの本の発進 1 | 1 | 2 |' in md


def test_f7_moved_start_makes_its_own_group_and_ignore_launch_merges():
    """出ずの組の本が無いときは、その発進だけの組(本数 0)。--ignore launch なら今まで通り 1 つにまとまる"""
    a = fake_res('20261103-110000', LB, standing=[start(1.0, 27.0, launched=True), start(40.0, 28.2, launched=False)])
    G = {g['text']: g for g in qp.group_rows([a])}
    assert G['B・0.90・天井 OFF・発進 ON・出ず']['runs'] == 0 and G['B・0.90・天井 OFF・発進 ON・出ず']['standing_list_s'] == [28.2]
    assert G['B・0.90・天井 OFF・発進 250']['standing_list_s'] == [27.0]
    G = qp.group_rows([a], ignore=('launch',))
    assert len(G) == 1 and G[0]['standing_list_s'] == [27.0, 28.2] and 'starts_from_other_runs' not in G[0]
    off = fake_res('20261103-111000', standing=[start(1.0, 27.5, launched=False)])     # launch_boost 0 は発進ごとでも off
    assert qp.start_launch_state(off, off['standing'][0]) == 'off' and len(qp.group_rows([off])) == 1


# ------------------------------------------------------------------ tools:F6 quick_eval_by_params の鍵と数え方
@pytest.mark.parametrize('part, a, b', [
    ('gain', {'throttle_gain': 0.90}, {'throttle_gain': 0.95}),
    ('steer', {'steer_gain': 1.0}, {'steer_gain': 1.1}),
    ('zone', {'zone_caps': 0}, {'zone_caps': 1}),
    ('mode', {'mode_id': 5}, {'mode_id': 6}),
    ('auto', {'speed_auto': 1, 'auto_target_mps': 4.27}, {'speed_auto': 1, 'auto_target_mps': 4.5}),
    ('launch', {'launch_boost': 1, 'launch_us': 250.0}, {'launch_boost': 1, 'launch_us': 200.0}),
    ('ceiling', {'speed_ceiling': 1, 'ceil_left_mps': 4.1}, {'speed_ceiling': 1, 'ceil_left_mps': 4.2}),
])
def test_f6_key_separates_each_part(part, a, b):
    sa = [start(1.0, 27.0, launched=True)] if 'launch_boost' in a else []
    ra, rb = fake_res('a', a, standing=sa), fake_res('b', b, standing=sa)
    assert qp.key_of(ra)[0] != qp.key_of(rb)[0], part
    assert qp.key_of(ra, ignore=(part,))[0] == qp.key_of(rb, ignore=(part,))[0], part
    assert len(qp.group_rows([ra, rb])) == 2 and len(qp.group_rows([ra, rb], ignore=(part,))) == 1
    assert qp.key_of(ra)[0] == qp.key_of(fake_res('c', a, standing=sa))[0]


def test_f6_only_counted_starts_and_fast_lap_threshold():
    """valid でない発進(3 周目の線の前に人側など。from_move_s はある)は止まってから 3 周に入れない。9 秒未満 = 9.0 秒未満"""
    r = fake_res('x', standing=[start(1.0, 24.0, valid=False), start(40.0, 26.0), start(80.0, 25.5, coll_before=1)],
                 clean=[8.5, 9.0, 9.5, 8.999])
    g = qp.group_rows([r])[0]
    assert g['standing_n'] == 1 and g['standing_list_s'] == [26.0] and g['standing_le25'] == 0
    assert g['standing_excluded_n'] == 1 and g['fast_laps'] == 2


# ------------------------------------------------------------------ tools:F6 standing_3lap の分かれ道
def arrays(anchors, heads=None, dur=45.0, t_ai=1.0, t_move=1.5, v_of=None, drop_frames=(), drop_vehicle=()):
    """配列で直接: フレーム t・ai、IMU の時刻 it と向き hd、車両の時刻 vt と eRPM。v_of: 車両の時刻 → 速さ [m/s]"""
    heads = heads if heads is not None else [360.0 * k for k in range(len(anchors))]
    t = np.arange(0, dur, 1 / 30.0)
    for a, b in drop_frames:
        t = t[~((t > a) & (t < b))]
    it = np.arange(0, dur, 0.01) + 0.003
    hd = np.interp(it, [0.0] + list(anchors) + [dur], [0.0] + list(heads) + [heads[-1]])
    vt = np.arange(0, dur, 0.01)
    for a, b in drop_vehicle:
        vt = vt[~((vt > a) & (vt < b))]
    v = v_of(vt) if v_of else np.where(vt >= t_move, V, 0.0)
    return t, t >= t_ai, it, hd, vt, -v / le.KV


def one(anchors, **kw):
    t, ai, it, hd, vt, verpm = arrays(anchors, **kw)
    rows = s3.standing_starts(t, ai, it, hd, vt, verpm, anchors, [], Lrun=18.0)
    assert len(rows) == 1
    return rows[0]


A4 = [4.0, 13.0, 22.0, 31.0, 40.0]


def test_f6_baseline_valid():
    s = one(A4)
    assert s['valid'] and s['why'] is None and s['from_move_s'] == pytest.approx(27.0, abs=0.02)


def test_f6_frame_gap_is_not_valid():
    s = one(A4, drop_frames=[(10.0, 10.3)])
    assert not s['valid'] and 'フレームの途切れ 0.3' in s['why']


def test_f6_vehicle_gap_is_not_valid():
    s = one(A4, drop_vehicle=[(12.0, 12.4)])
    assert not s['valid'] and '車両データの途切れ 0.4' in s['why']


def test_f6_line_lap_outside_5_16_is_not_valid():
    """起点 → 起点は 7・9・9 秒(5〜16 秒)だが、A2 の手前 5 m を 1 m/s で走ったので線 → 線の 1 周目が 4.5 秒"""
    v_of = lambda x: np.where(x < 1.5, 0.0, np.where((x >= 6.0) & (x < 11.0), 1.0, V))      # noqa: E731
    s = one([4.0, 11.0, 20.0, 29.0, 38.0], v_of=v_of)
    assert s['laps_line_s'][0] == pytest.approx(4.5, abs=0.03) and s['laps_anchor_s'] == pytest.approx([7.0, 9.0, 9.0], abs=0.03)
    assert not s['valid'] and s['why'] == '線 → 線の周が 5〜16 秒でない'


def test_f6_anchor_chain_stops_at_a_long_gap():
    """起点の間が 17 秒(止まった・人が運転した)→ そこで切る → 3 周に届かない"""
    s = one([4.0, 13.0, 30.0, 39.0], heads=[0, 360, 720, 1080])
    assert s['anchors'] == pytest.approx([4.0, 13.0], abs=0.01)
    assert s['why'].startswith('3 周に届かない(続いた起点 2')


def test_f6_first_anchor_after_15_s_is_not_used():
    s = one([17.0, 26.0, 35.0, 44.0], dur=50.0)
    assert s['why'] == '15 秒以内(AI の区間の中)に R1 の起点が無い' and s.get('to_r1_s') is None


def test_f6_flying_windows_need_continuous_laps():
    laps = [{'t_start': 4.0 + 9.0 * k, 'lap_s': 9.0, 'clean': True} for k in range(4)]
    assert len(s3.flying_windows(laps, [])) == 2
    laps[2] = dict(laps[2], t_start=22.5)                                     # 2 周目の終わり 22.0 → 3 周目の始まり 22.5
    laps[3] = dict(laps[3], t_start=31.5)
    assert s3.flying_windows(laps, []) == []


def test_f6_dry_run_frames_are_not_ai(tmp_path):
    """AI 側でも dry run のフレーム(車は動かさない)は発進に数えない"""
    log = '20261103-091900'
    d, _, _ = write_standing_log(tmp_path, log)
    assert [x for x in analyze(tmp_path, log)['standing'] if x.get('valid')]
    log2 = '20261103-092000'
    d2, _, _ = write_standing_log(tmp_path, log2)
    for p in sorted(d2.glob('*/AI_CMD.npz')):
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        a['dry_run'] = np.ones(len(a['timestamps_us']), bool)
        np.savez(p, **a)
    assert analyze(tmp_path, log2)['standing'] == []


def test_f6_lap_distance_median_uses_only_full_turns():
    """起点 → 起点で向きの変化が 300〜420° の周だけ(半周 180° の 9 m は入れない)"""
    anchors = np.array([4.0, 13.0, 22.0])
    it = np.arange(0, 30, 0.01)
    hd = np.interp(it, [0, 4, 13, 22, 30], [0, 0, 360, 540, 540])
    vt = np.arange(0, 30, 0.01)
    v = np.where(vt < 13.0, 2.0, 1.0)
    _, Df = le.cumulative_distance(vt, -v / le.KV)
    assert s3.lap_dist_median(anchors, it, hd, vt, Df) == pytest.approx(18.0, abs=0.05)


def test_f6_best_standing_is_the_fastest_counted():
    res = {'standing': [start(1.0, 28.0), start(40.0, 27.0), start(80.0, 26.0, coll_before=1), start(120.0, 25.0, valid=False)]}
    assert s3.best_standing(res)['from_move_s'] == 27.0
