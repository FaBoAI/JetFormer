"""確認 V-tools の major 4 つの直し(段 Fix-tools、2026-10-21)の試験

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_bench_tools_fix.py

1. 基板の再起動/4 分: ブロックの時間(AI の走り始め → 数える区間の終わり)の再起動を全部数える。周 0 本でも出す。None は判定できない
2. 「R1 の旋回中のブレーキ」: 層の区間 R1_turn ではなく、後から見た起点 ta → ta + 0.5 秒の時刻で数える
3. 事前登録の直し A1: 家族 label_r1 は門 5 を l1_steer_best_us で比べ、brake_lowspeed_frac ≤ 0.01 を足す
4. 説明欄を記録・計画と照らし合わせる(候補の id・日・計画のブロック・drive.run)
GPU・torch は使わない。10/10 の記録が要る試験は、無ければ飛ばす。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import section_scorecard as ss  # noqa: E402
import bench_report as br  # noqa: E402

LABELS = ROOT / 'data' / 'labels'
RAW = ROOT / 'data' / 'raw'
L181504, L155820 = '20261010-181504', '20261010-155820'
HAVE_RB = all((LABELS / (lg + '.json')).exists() and (RAW / lg).exists() for lg in (L181504, L155820))
need_rb = pytest.mark.skipif(not HAVE_RB, reason='10/10 の再起動のある記録・ラベルが無い')


def _rows(n, lap=9.5, inc=0, reboot_at=()):
    return [{'ta': 10.0 * i, 'tg': 10.0 * i + 2, 'incidents': inc, 'collisions': inc, 'strong': 0, 'tk_failure': 0, 'tk_all': 0,
             'tk_stuck_preemptive': 0, 'r1_strong': False, 'r1_coll': False, 'gm_coll': False, 'de_coll': False, 'clean': True,
             'lap_time_s': lap, 'reboot': i in reboot_at, 'n_reboots': int(i in reboot_at), 'collision_locs': [],
             'entry_erpm': 5000.0, 't_r1gm_s': 2.0, 't_rest_s': 7.0, 'holds': 0, 'ai_s_at_mid': 10.0 * i + 5, 'recoveries': 0}
            for i in range(n)]


# ---------------------------------------------------------------- 1. 基板の再起動
def test_reboot_basis_counts_without_laps():
    win = [{'ai_s_block': 60.0, 'reboots_block': 3, 'reboots_in_window': 1}, {'ai_s_block': 60.0, 'reboots_block': 1}]
    b = ss.reboot_basis(win)
    assert b['reboots'] == 4 and b['ai_s'] == 120.0 and b['in_window'] == 1
    m = ss.protocol_metrics([[]], ai_s=0.0, n_boot=50, reboot_basis=b)
    assert m['laps_counted'] == 0 and m['reboots'] == 4 and m['reboots_per_4min'] == 8.0      # 周 0 本でも出す
    assert ss.reboot_basis([{}, None]) is None
    m0 = ss.protocol_metrics([[]], n_boot=50, reboot_basis=None)
    assert m0['reboots_per_4min'] is None                                                      # 元が無い = 判定できない
    # 周の行の外(慣らし・半端な周)の再起動も入る: 周の行の中 1 回、ブロックでは 5 回
    rows = _rows(15, reboot_at=(3,))
    m = ss.protocol_metrics([rows], ai_s=150.0, n_boot=50, reboot_basis={'reboots': 5, 'ai_s': 240.0, 'in_window': 2})
    assert m['reboots_in_laps'] == 1 and m['reboots'] == 5 and m['reboots_per_4min'] == 5.0 and m['reboots_basis'] == 'ai_all'


@need_rb
def test_score_log_counts_all_reboots_of_the_block():
    """181504(再起動 12 回)を B のブロック(K = 3)として数えると、周 0 本でも再起動は 12 回・None にならない(V-tools 4 章 1)"""
    s = ss.score_log(L181504, LABELS, RAW, 'bench=x day=2026-10-23 block=1 cand=B', None, 3, 240)
    w = s['window']
    assert w['reboots_block'] == 12 and w['ai_s_block'] > 0
    agg = ss.aggregate([s], n_boot=50)
    P = agg['protocol']
    assert P['reboots'] == 12 and P['reboots_per_4min'] == round(12 / (w['ai_s_block'] / 240.0), 3)
    s2 = ss.score_log(L155820, LABELS, RAW, 'bench=x day=2026-10-24 block=1 cand=B', None, 3, 240)
    assert s2['window']['reboots_block'] >= 8                                 # AI 側 8 回(区間の中は 1 回だけ)


def test_judge_day2_reboots_none_is_undecided():
    good = {'metrics': {'laps_counted': 20, 'r1_strong_per_pass': {'value': 0.1}, 'incidents_per_lap': {'value': 0.5},
                        'reboots_per_4min': None},
            'diff': {'d_incidents_per_lap': {'value': -0.3, 'ci90': [-0.6, 0.0]}},
            'S': {'1.0': {'value': -0.6, 'ci90': [-1.0, -0.1]}}, 'blocks': [2], 'flags': []}
    cmp_ = {'reference': {'pooled_B': {'r1_strong_per_pass': {'value': 0.2}, 'reboots_per_4min': 0.5}}, 'cands': {'T1': good}}
    j = br.judge_day2(cmp_, None)
    assert j['rows'][0]['conditions']['5_reboots_not_1_more'] is None and not j['rows'][0]['better_than_B']
    good['metrics']['reboots_per_4min'] = 1.2                                  # B 0.5 + 0.7 < 1 回 → 満たす
    assert br.judge_day2(cmp_, None)['verdict']['best'] == 'T1'
    good['metrics']['reboots_per_4min'] = 1.6                                  # B より 1.1 回多い → 満たさない
    assert br.judge_day2(cmp_, None)['rows'][0]['conditions']['5_reboots_not_1_more'] is False


# ---------------------------------------------------------------- 2. R1 の旋回中のブレーキ
def test_turn_brake_counted_by_time_not_by_layer_zone():
    """層の位相が遅れ、実際の旋回中(起点 5.0 秒の後)も区間を R1_app のままにしてブレーキを出した → 時刻で数えると 6 フレーム。
    層の区間 R1_turn で数えると、作りの上で 0(V-tools 4 章 2)"""
    t = np.round(np.arange(0, 10, 0.05), 3)
    ai = np.ones(len(t), bool)
    zone = np.zeros(len(t), int)
    zone[(t >= 4.0) & (t < 5.3)] = 2           # R1_app(遅れて 5.3 秒まで)
    zone[(t >= 5.3) & (t < 6.5)] = 3           # R1_turn
    mode = np.zeros(len(t), int)
    mode[(t >= 5.0) & (t <= 5.25)] = ss.CAP_MODE_BRAKE        # 5.00〜5.25 秒 = 6 フレーム
    A = {'timestamps_us': (t * 1e6).astype(np.int64), 'cap_mode': mode, 'cap_zone': zone, 'cap_on': np.ones(len(t), bool),
         'cap_applied': mode > 0}
    c = ss.cap_counts(t, A, t, ai, 2.0, 9.0, np.array([5.0]), np.array([7.0]))
    assert c['on_brake_frames_in_r1_turn'] == 6 and c['on_brake_starts_in_r1_turn'] == 1
    assert c['on_brake_frames_in_r1_turn_zone'] == 0 and c['shadow_brake_in_r1_turn_zone'] == 0
    assert c['on_brake_frames_in_r1_turn_to_tg'] == 6 and c['shadow_brake_frames_in_r1_turn'] == 6
    # 慣らし(数える区間の前)のブレーキも数える
    c2 = ss.cap_counts(t, A, t, ai, 6.0, 9.0, np.array([5.0]), np.array([7.0]))
    assert c2['on_brake_frames_in_r1_turn'] == 6 and c2['on_brake_frames_in_r1_turn_window'] == 0
    # 区間が無い(慣らしで終わった)ブロックでも数える
    c3 = ss.cap_counts(t, A, t, ai, None, None, np.array([5.0]), np.array([7.0]))
    assert c3['on_brake_frames_in_r1_turn'] == 6 and c3['ai_frames'] == 0
    # OFF の走行(cap_on = 0)は 0。ブレーキが R1_app(起点の前)だけなら 0
    A_off = dict(A, cap_on=np.zeros(len(t), bool))
    assert ss.cap_counts(t, A_off, t, ai, 2.0, 9.0, np.array([5.0]), np.array([7.0]))['on_brake_frames_in_r1_turn'] == 0
    mode2 = np.zeros(len(t), int)
    mode2[(t >= 4.5) & (t < 5.0)] = ss.CAP_MODE_BRAKE
    A2 = dict(A, cap_mode=mode2)
    assert ss.cap_counts(t, A2, t, ai, 2.0, 9.0, np.array([5.0]), np.array([7.0]))['on_brake_frames_in_r1_turn'] == 0


def test_cap_codes_follow_safety_caps():
    sys.path.insert(0, str(ROOT))
    from jetformer import safety_caps as sc
    assert ss.CAP_ZONE_R1_TURN == sc.ZONES.index('R1_turn') and ss.CAP_MODE_BRAKE == sc.CAP_MODES.index('brake')


def test_report_cap_metric_uses_time_based_count():
    w = {'ai_s': 100.0, 'ai_s_block': 100.0, 'reboots_block': 0,
         'cap': {'ai_frames': 100, 'on_brake_frames_in_r1_turn': 3, 'shadow_brake_in_r1_turn_zone': 0}}
    e = {'score': {'window': w}, 'rows': _rows(12)}
    m, _ = br.metrics_and_samples([e], '2026-10-24', 'x', 50)
    assert m['cap_brake_in_r1_turn'] == 3


# ---------------------------------------------------------------- 3. A1
def test_recompute_gates_applies_amendment_a1():
    base = {'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 5.0, 'steer_amp_ratio': 1.0}
    rows = {'B': {'l1_best_us': 49.0, 'l1_steer_best_us': 28.0},
            'SLOW': dict(base, l1_best_us=63.3, l1_steer_best_us=30.0, brake_lowspeed_frac=0.0, gates_pass=True),
            'S1': dict(base, l1_best_us=63.3, l1_steer_best_us=30.0, brake_lowspeed_frac=0.2),
            'S2': dict(base, l1_best_us=50.0),
            'MSF': dict(base, l1_best_us=63.3, l1_steer_best_us=30.0)}
    fam = {'SLOW': br.family_of('SLOW', 'bench_20261021-slow-s0'), 'S1': br.family_of('S1', 'bench_20261021-slow-s1'),
           'S2': 'label_r1', 'MSF': br.family_of('MSF', 'clean_driving_20261011-msfusion-s0-nocmd-stride3-e30',
                                                 proto=json.loads(br.PROTOCOL.read_text(encoding='utf-8')) if br.PROTOCOL.exists() else None)}
    assert fam['SLOW'] == fam['S1'] == 'label_r1'
    g = br.recompute_gates(rows, fam)
    assert g['SLOW']['gates_pass'] and g['SLOW']['agrees_with_desk'] and g['SLOW']['amendment'] == 'A1'   # 食い違いと書かない
    assert g['S1']['gate_fail_reasons'] == ['brake_lowspeed_frac']
    assert set(g['S2']['gate_fail_reasons']) == {'missing:l1_steer_best_us', 'missing:brake_lowspeed_frac'}
    assert g['MSF']['gate_fail_reasons'] == ['l1_best_us'] and g['MSF']['amendment'] is None              # ほかの家族はそのまま
    if br.PROTOCOL.exists():
        assert fam['MSF'] == 'arch'
    # 家族を渡さなければ、今までと同じ(全部 l1_best_us)
    assert 'l1_best_us' in br.recompute_gates(rows)['SLOW']['gate_fail_reasons']


def test_family_of_by_run_name_then_table():
    bench = {'candidates': [{'id': 'X', 'run': 'foo', 'family': 'robust'}, {'id': 'AUG', 'run': 'bench_20261021-slow-s1'}]}
    assert br.family_of('AUG', None, bench) == 'label_r1'          # AUG の枠に slow-s1 が入った日(run の名前が先)
    assert br.family_of('X', None, bench) == 'robust'
    assert br.family_of('ENS', 'bench_20261021-ens-p2') == 'ens' and br.family_of('Q', 'zzz') is None


# ---------------------------------------------------------------- 4. 説明欄の照らし合わせ
BENCH = {'bench': 'rb', 'candidates': [{'id': 'B', 'run': 'runB'}, {'id': 'SLOW', 'run': 'runS'}, {'id': 'MSF', 'run': 'runM'}],
         'days': [{'date': '2026-10-23', 'blocks': [{'block': 1, 'id': 'B'}, {'block': 2, 'id': 'SLOW'}, {'block': 3, 'id': 'MSF'}]},
                  {'date': '2026-10-24', 'blocks': [{'block': 1, 'id': 'B'}, {'block': 2, 'id': 'SLOW'}]}]}


def _fake_logs(tmp, items):
    raw = tmp / 'raw'
    for lg, desc, run in items:
        (raw / lg).mkdir(parents=True)
        (raw / lg / 'log.json').write_text(json.dumps({'description': desc, 'drive': {'run': run}}), encoding='utf-8')
    (tmp / 'labels').mkdir(exist_ok=True)
    return tmp / 'labels', raw


def test_collect_checks_descriptions(tmp_path):
    t = 'bench=rb day=%s block=%d cand=%s'
    items = [('20261023-100000', t % ('2026-10-23', 1, 'B'), 'runB'),          # 正しい
             ('20261023-101000', t % ('2026-10-23', 2, 'SLWO'), 'runS'),       # id の書き違い
             ('20261023-102000', t % ('2026-10-23', 3, 'b'), 'runB'),          # 小文字
             ('20261023-103000', t % ('2026-10-23', 3, 'MSF'), 'runB'),        # 走ったモデルが違う
             ('20261024-100000', t % ('2026-10-23', 2, 'SLOW'), 'runS'),       # 前の日の説明欄(10/23 の 2)を 10/24 に使った
             ('20261024-101000', t % ('2026-10-24', 1, 'SLOW'), 'runS'),       # 計画では B
             ('20261024-102000', t % ('2026-10-24', 9, 'B'), 'runB')]          # 計画に無いブロック
    labels, raw = _fake_logs(tmp_path, items)
    es, un = br.collect([x[0] for x in items], BENCH, None, labels, raw)
    assert not un
    by = {e['log']: e for e in es}
    kinds = {lg: sorted(k for k, _ in e['checks']) for lg, e in by.items()}
    assert kinds['20261023-100000'] == []
    assert kinds['20261023-101000'] == ['cand', 'plan'] and by['20261023-101000']['unknown_cand']   # 計画(SLOW)とも違う
    assert 'SLOW' in dict(by['20261023-101000']['checks'])['cand']
    assert 'cand' in kinds['20261023-102000'] and 'B' in dict(by['20261023-102000']['checks'])['cand']
    assert kinds['20261023-103000'] == ['run']
    e = by['20261024-100000']
    assert e['day'] == '2026-10-24' and kinds['20261024-100000'] == ['day']        # 記録の日付で数える。計画(10/24 の 2 = SLOW)とは合う
    assert kinds['20261024-101000'] == ['plan']
    assert kinds['20261024-102000'] == ['plan']
    dv = br.check_deviations(es)
    assert any('SLWO' in x for x in dv) and any('runB' in x and 'cand=MSF' in x for x in dv) and any('day=2026-10-23' in x for x in dv)
    # --assign の説明欄は人の直し: その day を使い、日の食い違いと言わない
    es2, _ = br.collect(['20261024-100000'], BENCH, {'20261024-100000': t % ('2026-10-23', 2, 'SLOW')}, labels, raw)
    assert es2[0]['day'] == '2026-10-23' and es2[0]['checks'] == [] and es2[0]['assigned']


def test_unknown_id_is_not_ranked():
    def ent(log, block, cand, rows, unknown=False):
        return {'log': log, 'block': block, 'cand': cand, 'day': '2026-10-23', 'desc': {'block': block, 'cand': cand}, 'K': 2,
                'flags': [], 'rerun': False, 'unknown_cand': unknown, 'checks': [],
                'score': {'window': {'ai_s': 10.0 * len(rows), 'ai_s_block': 10.0 * len(rows), 'reboots_block': 0}, 'laps_protocol': rows}}
    es = [ent('a', 1, 'B', _rows(15, inc=1)), ent('b', 2, 'SLWO', _rows(15, lap=9.0), True), ent('c', 3, 'MSF', _rows(15, lap=9.4)),
          ent('d', 4, 'B', _rows(15, inc=1))]
    for e in es:
        e['rows'] = e['score']['laps_protocol']
    cm = br.compare_day('2026-10-23', es, n_boot=200)
    rk = br.rank_day1(cm, '1.0', min_laps=12)
    marks = {r['id']: r['marks'] for r in rk['rows']}
    assert '計画に無い id' in marks['SLWO'] and rk['T1'] == 'MSF' and rk['T2'] is None


def test_drift_cannot_judge_when_last_b_block_has_no_laps():
    def ent(log, block, cand, rows):
        return {'log': log, 'block': block, 'cand': cand, 'day': '2026-10-23', 'desc': {'block': block, 'cand': cand}, 'K': 2,
                'flags': [], 'rerun': False, 'checks': [], 'rows': rows,
                'score': {'window': {'ai_s': 10.0 * max(len(rows), 1), 'ai_s_block': 30.0, 'reboots_block': 4 if not rows else 0},
                          'laps_protocol': rows}}
    es = [ent('a', 1, 'B', _rows(15, lap=9.0)), ent('b', 2, 'C1', _rows(15, lap=9.5)), ent('c', 3, 'B', [])]
    cm = br.compare_day('2026-10-23', es, n_boot=200)
    dr = cm['reference']['drift']
    assert dr['flag'] is False and dr['undecided'] is True and '判定できない' in dr['note']
    assert cm['cands']['C1']['ref'] == 'pooled_B'
    assert cm['blocks'][3]['metrics']['reboots_per_4min'] == 32.0               # 周 0 本の B のブロックの再起動も出る


# ---------------------------------------------------------------- 模擬のベンチマーク(端から端まで、直したところ)
MOCK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/build/mock_bench')
HAVE_MOCK = (MOCK / 'mock_assign.json').exists() and all(
    (LABELS / (lg + '.json')).exists() for lg in json.loads((MOCK / 'mock_assign.json').read_text(encoding='utf-8'))) \
    if (MOCK / 'mock_assign.json').exists() else False


@pytest.mark.skipif(not HAVE_MOCK, reason='模擬の割り当て・10/10 のラベルが無い')
def test_mock_report_shows_the_fixes(tmp_path):
    assign = json.loads((MOCK / 'mock_assign.json').read_text(encoding='utf-8'))
    rep = br.build_report([], MOCK / 'mock_bench_candidates.json', MOCK / 'mock_desk_bench.json', br.PROTOCOL, LABELS, RAW,
                          assign, None, None, n_boot=200, min_laps=3, mock=True, logs=sorted(assign), k_first=1, k_later=1)
    rep['command'] = 'test'
    br.write_report(rep, tmp_path)
    md = (tmp_path / 'README.md').read_text(encoding='utf-8')
    dv = (tmp_path / 'deviations.md').read_text(encoding='utf-8')
    d1 = rep['days'][0]['compare']
    # 1. 周 0 本の B#8(160306、再起動 4 回)も再起動/4 分が出る
    b8 = d1['blocks'][8]['metrics']
    assert b8['laps_counted'] == 0 and b8['reboots'] == 4 and b8['reboots_per_4min'] is not None
    assert '再起動/4分' in md
    # 時間のずれ: 最後の B の周が 0 → 「なし」と言わず「判定できない」
    assert d1['reference']['drift'].get('undecided') and '判定できない' in md
    # 3. A1 の欄が 3 章にある。SLOW(label_r1)は A1 の欄が無いので missing になる(模擬の机上表には無い)
    assert 'l1_steer_best_us' in md and 'brake_lowspeed_frac' in md
    assert rep['families'].get('SLOW') == 'label_r1' and rep['gates_recomputed']['SLOW']['amendment'] == 'A1'
    # 4. 模擬は全部 first_trial で走った記録 → モデルの食い違いが deviations に出る
    assert '説明欄の照らし合わせ' in dv and 'first_trial' in dv and '--assign で説明欄を上書きした記録が 11 本' in dv
    rj = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert rj['days'][0]['laps'][0]['drive_run'].startswith('first_trial')


# ---------------------------------------------------------------- minor の直し
def test_per_pass_rates_show_exact_interval():
    rows = _rows(34)
    for i in range(8):
        rows[i * 4]['r1_strong'] = True
    m = ss.protocol_metrics([rows], ai_s=340.0, n_boot=200)
    lo, hi = ss.cp_ci(8, 34)
    assert m['r1_strong_per_pass']['ci90_exact'] == [round(lo, 4), round(hi, 4)]
    assert '正確な二項' in ss._pm(m['r1_strong_per_pass'], True, '%.0f')
    assert 'ci90_exact' not in m['incidents_per_lap']                     # 周あたりの率には付けない


def test_brake_fit_is_provisional_with_few_strong_events():
    import brake_table as bt
    ev = [{'clean': True, 'op': 'human', 'surface': '床', 'strength': '弱' if off < 150 else '中', 'thr_off_med': off,
           'mean_decel': 0.6 + 0.004 * max(0, off - 100), 'alat_mean': 0.0, 'v0': 2.0} for off in (70, 80, 90, 100, 110, 120, 200, 260)]
    T = bt.table(ev, coast={'c0': 0.58, 'k_s': 0.40, 'source': 'test'}, ops=('human',))
    f = T['fits']['床']
    assert f['n_ge150us'] == 2 and f['provisional'] and '仮' in f['note']
    ev += [dict(ev[-1], thr_off_med=o, mean_decel=0.6 + 0.004 * (o - 100)) for o in (300, 330, 360)]
    f2 = bt.table(ev, coast={'c0': 0.58, 'k_s': 0.40, 'source': 'test'}, ops=('human',))['fits']['床']
    assert f2['n_ge150us'] == 5 and not f2.get('provisional')


def _no_records(*days):
    """記録の無い日(本物の data/raw/<日>-* があると bench_nightly が最後まで回ってしまう: 重く、試験のねらいからも外れる)"""
    for d in days:
        assert not list(RAW.glob(d + '-*')), '%s の記録がある(試験は記録の無い日で行う)' % d


def test_nightly_takes_day1_from_bench(tmp_path):
    """--day1 を忘れても bench_candidates.json の days から 1 日目を決める(V-tools 4 章 11: 2 日目の初めてのブロックの K を誤らない)。
    記録の無い日(2000 年)で試す(2026-10-26 fix2: 10/23・10/24 は本物の記録ができて、ここで止まらずに落ちていた)"""
    import subprocess
    _no_records('20001023', '20001024', '20001001', '20001002')
    b = tmp_path / 'bench.json'
    b.write_text(json.dumps({'bench': 'x', 'days': [{'date': '2000-10-23', 'blocks': []}, {'date': '2000-10-24', 'blocks': []}]}))
    r = subprocess.run(['bash', str(ROOT / 'tools' / 'bench_nightly.sh'), '20001024', '--bench', str(b), '--out', str(tmp_path / 'n'),
                        '--report', str(tmp_path / 'r')], capture_output=True, text=True, timeout=120)
    head = (tmp_path / 'n' / 'nightly.log').read_text(encoding='utf-8').splitlines()[0]
    assert '1 日目 20001023、bench_candidates.json の days から' in head      # --day1 を忘れても K を誤らない
    assert r.returncode == 1 and 'data/raw/20001024-* がありません' in r.stdout   # 記録が無いのでここで止まる
    # 1 日目そのもの(days[0])は「1 日目 なし」
    b1 = tmp_path / 'bench1.json'
    b1.write_text(json.dumps({'bench': 'x', 'days': [{'date': '2000-10-01', 'blocks': []}, {'date': '2000-10-02', 'blocks': []}]}))
    r1 = subprocess.run(['bash', str(ROOT / 'tools' / 'bench_nightly.sh'), '20001001', '--bench', str(b1), '--out', str(tmp_path / 'n1'),
                         '--report', str(tmp_path / 'r1')], capture_output=True, text=True, timeout=120)
    assert r1.returncode == 1 and '1 日目 なし' in (tmp_path / 'n1' / 'nightly.log').read_text(encoding='utf-8').splitlines()[0]


def test_timeline_and_report_handle_zero_second_block(tmp_path):
    def ent(log, block, cand, rows, ai_s):
        return {'log': log, 'block': block, 'cand': cand, 'day': '2026-10-23', 'desc': {'block': block, 'cand': cand}, 'K': 2,
                'flags': [], 'rerun': False, 'checks': [], 'rows': rows,
                'score': {'window': {'ai_s': ai_s, 'ai_s_block': ai_s, 'reboots_block': 0}, 'laps_protocol': rows}}
    es = [ent('a', 1, 'B', _rows(15), 150.0), ent('b', 2, 'TCP', [], 0.0), ent('c', 3, 'ACT', _rows(15, lap=9.3), 150.0),
          ent('d', 4, 'B', _rows(15), 150.0)]
    cm = br.compare_day('2026-10-23', es, n_boot=100)
    rep = {'days': [{'day': '2026-10-23', 'entries': es, 'compare': cm, 'battery_fit': {'used': False}, 'ranking': br.rank_day1(cm)}]}
    made = br.make_figs(rep, tmp_path)
    assert 'f7_timeline_20261023.png' in made and (tmp_path / 'f7_timeline_20261023.png').exists()


def test_uncorrected_S_when_drift_interpolation_is_used():
    def ent(log, block, cand, lap):
        rows = _rows(15, lap=lap)
        return {'log': log, 'block': block, 'cand': cand, 'day': '2026-10-23', 'desc': {'block': block, 'cand': cand}, 'K': 2,
                'flags': [], 'rerun': False, 'checks': [], 'rows': rows,
                'score': {'window': {'ai_s': 150.0, 'ai_s_block': 150.0, 'reboots_block': 0}, 'laps_protocol': rows}}
    es = [ent('a', 1, 'B', 9.0), ent('b', 2, 'C1', 9.3), ent('c', 3, 'B', 10.0)]
    cm = br.compare_day('2026-10-23', es, n_boot=200)
    assert cm['reference']['drift']['flag'] and cm['cands']['C1']['ref'] == 'interpolated'
    u = br.uncorrected_S('2026-10-23', es, 200)
    # 内挿の基準(ブロック 2 = 9.5)では S = −0.2、B のプール(中央値 9.5)でも同じ −0.2。値が出ることと、内挿をしないことを見る
    assert u['C1'] == pytest.approx(cm['cands']['C1']['S']['1.0']['value'], abs=0.06)
    c0 = br.compare_day('2026-10-23', es, n_boot=200, use_drift=False)
    assert c0['reference']['drift']['flag'] is False and c0['reference']['drift']['flag_ignored'] is True and c0['cands']['C1']['ref'] == 'pooled_B'
