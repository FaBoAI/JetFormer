"""実車のベンチマークの道具(段 S3、2026-10-21)の試験: 区間の成績表・ブレーキの表・ベンチマークレポート

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_bench_tools.py

10/10 の実車の記録(data/raw/20261010-*・data/labels)が無い機械では、その試験だけ飛ばす。GPU・torch は使わない。
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import section_scorecard as ss  # noqa: E402
import brake_table as bt  # noqa: E402
import bench_report as br  # noqa: E402

LABELS = ROOT / 'data' / 'labels'
RAW = ROOT / 'data' / 'raw'
L006, L007 = '20261010-172910', '20261010-173445'
INVENTORY = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/inventory.json')
HAVE_1010 = all((LABELS / (lg + '.json')).exists() and (RAW / lg).exists() for lg in (L006, L007))
need_1010 = pytest.mark.skipif(not HAVE_1010, reason='10/10 の記録・ラベルが無い')


# ---------------------------------------------------------------- 統計の小道具
def test_poisson_ci_known_values():
    lo, hi = ss.poisson_ci(0)
    assert lo == 0.0 and abs(hi - 2.9957) < 1e-3           # −ln(0.05)
    lo, hi = ss.poisson_ci(10)
    assert abs(lo - 5.4254) < 2e-3 and abs(hi - 16.9622) < 2e-3


def test_clopper_pearson_known_values():
    lo, hi = ss.cp_ci(6, 9)
    assert abs(lo - 0.3449) < 2e-3 and abs(hi - 0.9015) < 2e-3
    lo, hi = ss.cp_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2589) < 2e-3


def test_weighted_median_matches_numpy():
    rng = np.random.default_rng(1)
    vals = rng.normal(10, 1, 11)
    vals[3] = np.nan
    C = rng.integers(0, 3, size=(200, 11)).astype(float)
    got = ss._weighted_median(vals, C)
    for i in range(200):
        ok = np.isfinite(vals)
        rep = np.repeat(vals[ok], C[i, ok].astype(int))
        if len(rep):
            assert abs(got[i] - np.median(rep)) < 1e-12
        else:
            assert np.isnan(got[i])


def test_parse_description_template():
    d = ss.parse_description('bench=real_bench_20261023 day=2026-10-23 block=4 cand=SLOW model=bench_20261021-slow-s0 '
                             'battery=#3 V_before=8.21 V_after=__ check=bumper7_tof_ok memo=R1 で 2 回 当たった')
    assert d['bench'] == 'real_bench_20261023' and d['day'] == '2026-10-23' and d['block'] == 4 and d['cand'] == 'SLOW'
    assert d['battery'] == '3' and d['V_before'] == '8.21' and d['V_after'] is None
    assert d['memo'] == 'R1 で 2 回 当たった'
    assert ss.parse_description('') == {}
    assert ss.parse_description('ai_run_006') == {}


def test_bootstrap_chunks_and_seed_are_deterministic():
    rows = [{'incidents': i % 3, 'collisions': i % 3, 'strong': 0, 'tk_failure': 0, 'tk_all': 0, 'r1_strong': i % 4 == 0,
             'r1_coll': False, 'gm_coll': False, 'de_coll': i % 2 == 0, 'clean': i % 3 == 0, 'tg': 1.0,
             'lap_time_s': 9.5 + 0.1 * i, 'reboot': i == 5, 'collision_locs': [], 'entry_erpm': 5000 + 10 * i,
             't_r1gm_s': 2.0, 't_rest_s': 7.5, 'holds': 0} for i in range(20)]
    a = ss.protocol_metrics([rows], ai_s=200, seed=20261023, n_boot=500)
    b = ss.protocol_metrics([rows], ai_s=200, seed=20261023, n_boot=500)
    assert a == b
    assert a['laps_counted'] == 19 and a['reboot_laps'] == 1          # 再起動の周は分子も分母も外す
    k = sum(r['incidents'] for r in rows if not r['reboot'])
    assert a['incidents_per_lap']['value'] == round(k / 19, 4)
    lo, hi = a['incidents_per_lap']['ci90']
    assert lo <= a['incidents_per_lap']['value'] <= hi
    assert a['reboots_per_4min'] == round(1 / (200 / 240), 3)
    z = ss.protocol_metrics([[dict(r, collisions=0, incidents=0) for r in rows]], ai_s=200, seed=1, n_boot=200)
    assert z['collisions_per_lap']['upper95_zero'] == round(3.0 / 19, 4)


# ---------------------------------------------------------------- 区間の成績表(10/10 の 006/007)
@pytest.fixture(scope='module')
def sc0607():
    return ss.build([L006, L007], LABELS, RAW, n_boot=500, warmup=1)


@need_1010
def test_scorecard_reproduces_collision_map_locations(sc0607):
    """collision_map.md の 006/007 の衝突(回復中を除く 45 件): R1 13・GM 8・DE 20・その他 4(SW 2・LH 2)"""
    A = sc0607['groups']['all']
    cm = A['collisions_cmap_by_loc']
    assert (cm['R1'], cm['GM'], cm['DE']) == (13, 8, 20)
    assert cm['LH'] == 2 and cm['other'] == 2 and cm['unknown'] == 0
    assert sum(cm.values()) == 45


@need_1010
def test_scorecard_reproduces_r1_entry_bins(sc0607):
    """week_20261016/README.md 1 章: 5,500 未満 1/22・5,500〜6,200 2/7・6,200 以上 6/9"""
    bins = sc0607['groups']['all']['r1_entry_bins']
    assert bins['<5500']['text'] == '1/22'
    assert bins['5500-6200']['text'] == '2/7'
    assert bins['>=6200']['text'] == '6/9'


@need_1010
def test_scorecard_whole_log_numbers(sc0607):
    A = sc0607['groups']['all']
    assert 38.0 <= A['lap_equiv'] <= 39.2                    # 周相当(10/10 の数え方 38.59。172910 のシーン 0001 が消えた分だけ少ない)
    assert A['collisions_per_lap']['n'] == 45 and abs(A['collisions_per_lap']['per'] - 1.17) < 0.02
    assert A['passes_ai'] == 38
    assert A['collisions_restart'] == 2                      # 007 の「回復」(100.99 GM・106.20 R1)
    assert sc0607['groups']['by_log'][L007]['lapcounter']['clean_best_s'] == 9.68   # 画面の LapCounter と同じ


@need_1010
def test_scorecard_protocol_laps(sc0607):
    s7 = sc0607['per_log'][L007]
    w = s7['window']
    assert w['warmup_anchors'] == 1 and w['start_s'] is not None
    rows = s7['laps_protocol']
    assert len(rows) >= 18
    for r in rows:
        assert r['t_from'] < r['t_to'] and r['incidents'] == r['collisions'] + r['tk_stuck_preemptive']
        if r['lap_time_s'] is not None:
            assert r['lap_time_s'] > 0
    P = sc0607['groups']['all']['protocol']
    assert P['laps_counted'] == sum(len(sc0607['per_log'][lg]['laps_protocol']) for lg in (L006, L007))
    assert 0.8 < P['incidents_per_lap']['value'] < 1.8


@need_1010
def test_scorecard_writes_json_and_markdown(sc0607, tmp_path):
    pj, pm = ss.write(sc0607, tmp_path)
    d = json.loads(pj.read_text(encoding='utf-8'))
    assert d['schema'] == ss.SCHEMA
    md = pm.read_text(encoding='utf-8')
    for s in ('## 0. 手順の指標', '## 2. 場所ごとの衝突', '## 3. R1 の進入', '## 4. 区間の時間', '## ログ ごと'):
        assert s in md
    assert '%%' not in md


@need_1010
def test_scorecard_groups_by_candidate_and_battery(tmp_path):
    assign = {L006: 'bench=x day=2026-10-23 block=1 cand=B battery=#1 V_before=8.3 V_after=8.0',
              L007: 'bench=x day=2026-10-23 block=2 cand=SLOW battery=#1 V_before=8.0 V_after=7.8'}
    sc = ss.build([L006, L007], LABELS, RAW, assign=assign, n_boot=200)
    g = sc['groups']
    assert set(g['by_cand']) == {'B', 'SLOW'} and set(g['by_battery']) == {'1'}
    assert set(g['by_block']) == {'2026-10-23#1', '2026-10-23#2'}


# ---------------------------------------------------------------- ブレーキの表
@pytest.fixture(scope='module')
def brakes1010():
    logs = sorted(d.name for d in RAW.glob('20261010-*') if (d / 'log.json').exists() and (LABELS / (d.name + '.json')).exists())
    return bt.build(logs, LABELS, RAW)


@need_1010
def test_brake_table_reproduces_inventory(brakes1010):
    if not INVENTORY.exists():
        pytest.skip('inventory.json が無い')
    ref = json.loads(INVENTORY.read_text(encoding='utf-8'))['brake_events']
    key = ('log', 't_s', 'op', 'dur_s', 'v0', 'v1', 'mean_decel', 'thr_off_med', 'clean', 'loc')
    got = [tuple(e[k] for k in key) for e in brakes1010['events']]
    assert got == [tuple(e[k] for k in key) for e in ref]
    assert len(got) == 23 and sum(e['clean'] for e in brakes1010['events']) == 16


@need_1010
def test_brake_table_006_nine_ai_brakes(brakes1010):
    """10/10 の 006 の AI のブレーキ 9 回(critique.md の「使えるブレーキは 9 回」)で表が作れる"""
    ev = [e for e in brakes1010['events'] if e['log'] == L006 and e['op'] == 'ai' and e['clean']]
    assert len(ev) == 9
    T = bt.table(ev)
    assert T['n_clean'] == 9
    assert sum(r['n'] for r in T['rows']) == 9
    assert all(r['n'] == 0 or r['decel_median'] > 0 for r in T['rows'])
    assert 's_b' in T['fits']['床']
    md = bt.to_markdown(dict(brakes1010, table=T))
    assert '| 床 | 弱 |' in md


def test_brake_surface_from_description():
    assert bt.surface_of_description('bench=x cand=BRAKE surface=芝 memo=') == '芝'
    assert bt.surface_of_description('bench=x memo=床で強いブレーキ') == '床'
    assert bt.surface_of_description('bench=x memo=床と芝') is None
    assert bt.surface_of_description('') is None


def test_fit_brake_recovers_line():
    off = np.array([100, 150, 200, 250, 300, 350.0])
    ab = 0.02 * np.maximum(0, off - 120)
    s, db, sse = bt.fit_brake(off, ab)
    assert abs(s - 0.02) < 1e-6 and db == 120 and sse < 1e-9


# ---------------------------------------------------------------- ベンチマークレポート(順位の決まり)
def _cand(laps=20, S=0.0, d_inc_ci=(-0.3, 0.3), r1=0.2, fail=0.3, blocks=(2,), flags=()):
    return {'metrics': {'laps_counted': laps, 'r1_strong_per_pass': {'value': r1}, 'failure_takeovers_per_lap': {'value': fail},
                        'incidents_per_lap': {'value': 1.0}, 'reboots_per_4min': 0.0},
            'diff': {'d_incidents_per_lap': {'value': 0.0, 'ci90': list(d_inc_ci)}},
            'S': {'1.0': {'value': S, 'ci90': [S - 0.5, S + 0.5]}, '0.5': {'value': S, 'ci90': [S - 0.5, S + 0.5]},
                  '2.0': {'value': S, 'ci90': [S - 0.5, S + 0.5]}},
            'blocks': list(blocks), 'flags': list(flags)}


def test_rank_day1_rules():
    cmp_ = {'reference': {'pooled_B': {'r1_strong_per_pass': {'value': 0.30, 'k': 6}, 'reboots_per_4min': 0.0}},
            'cands': {'MSF': _cand(S=-0.20, blocks=(2,)), 'TCP': _cand(S=-0.21, r1=0.25, blocks=(3,)),
                      'ACT': _cand(S=0.40, r1=0.10, blocks=(5,)), 'WGT': _cand(laps=8, S=-0.9, blocks=(6,)),
                      'SIM': _cand(S=0.1, d_inc_ci=(0.05, 0.6), blocks=(7,)), 'AUG': _cand(S=-0.5, flags=('safety_stop',))}}
    rk = br.rank_day1(cmp_)
    marks = {r['id']: r['marks'] for r in rk['rows']}
    assert '足りない' in marks['WGT'] and 'はっきり悪い' in marks['SIM'] and '安全で中止' in marks['AUG']
    # MSF と TCP は 0.05 刻みで同じ S(−0.20)→ R1 の強い衝突の少ない MSF が先
    assert rk['T1'] == 'MSF'
    # 上位 2 本のどちらも R1 が B の半分(0.15)以下でなく、B の R1 は 6 回 → R1 の枠: ACT(0.10)が 2 位に入る
    assert rk['T2'] == 'ACT' and rk['r1_slot_swap'] == {'out': 'TCP', 'in': 'ACT'}


def test_judge_day2_rules():
    good = _cand(S=-0.6, d_inc_ci=(-0.5, 0.05), r1=0.1)
    good['S']['1.0']['ci90'] = [-1.0, -0.1]
    bad = _cand(S=-0.3, d_inc_ci=(-0.5, 0.3), r1=0.1)
    cmp_ = {'reference': {'pooled_B': {'r1_strong_per_pass': {'value': 0.2}, 'reboots_per_4min': 0.0}},
            'cands': {'T1': good, 'T2': bad}}
    j = br.judge_day2(cmp_, None)
    assert j['verdict'] == {'result': 'better', 'best': 'T1', 'text': 'T1 が B より良い'}
    cmp_['cands'] = {'T2': bad}
    j = br.judge_day2(cmp_, None)
    assert j['verdict']['result'] == 'keep_B' and j['verdict']['next_to_try'] == 'T2'


def test_recompute_gates():
    rows = {'B': {'l1_best_us': 49.0},
            'OK': {'l1_best_us': 50.0, 'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 5.0, 'steer_amp_ratio': 1.0, 'gates_pass': True},
            'SLOW': {'l1_best_us': 59.5, 'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 5.0, 'steer_amp_ratio': 1.3, 'gates_pass': True},
            'NOENG': {'l1_best_us': 50.0, 'sanity_ok': True, 'trt_p99_ms': 12.0, 'steer_amp_ratio': 1.0}}
    g = br.recompute_gates(rows)
    assert g['OK']['gates_pass'] and g['OK']['agrees_with_desk']
    assert set(g['SLOW']['gate_fail_reasons']) == {'steer_amp_ratio', 'l1_best_us'} and not g['SLOW']['agrees_with_desk']
    assert 'missing:engine_ok' in g['NOENG']['gate_fail_reasons'] and 'trt_p99_ms' in g['NOENG']['gate_fail_reasons']


def test_t_quantile_close_to_table():
    assert abs(br.t_quantile(0.95, 10) - 1.8125) < 0.01
    assert abs(br.t_quantile(0.95, 30) - 1.6973) < 0.005


# ---------------------------------------------------------------- 模擬のベンチマーク(端から端まで)
MOCK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/build/mock_bench')


@need_1010
def test_mock_bench_end_to_end(tmp_path):
    if not (MOCK / 'mock_assign.json').exists():
        pytest.skip('模擬の割り当てが無い')
    assign = json.loads((MOCK / 'mock_assign.json').read_text(encoding='utf-8'))
    rep = br.build_report([], MOCK / 'mock_bench_candidates.json', MOCK / 'mock_desk_bench.json', br.PROTOCOL, LABELS, RAW,
                          assign, None, None, n_boot=300, min_laps=3, mock=True, logs=sorted(assign), k_first=1, k_later=1)
    rep['command'] = 'test'
    br.write_report(rep, tmp_path)
    md = (tmp_path / 'README.md').read_text(encoding='utf-8')
    for s in ('## 0. 結論', '## 2. 条件', '## 3. 候補の表', '## 4. 比べ方', '## 5. 結果の順位表', '## 6. 区間ごと', '## 7. 電池ごと',
              '## 8. 時間のずれ', '## 10. 図', '## 11. 再現のコマンド', '## 12. 限界', '模擬のベンチマーク'):
        assert s in md, s
    rj = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert rj['schema'] == br.SCHEMA and [d['day'] for d in rj['days']] == ['2026-10-23', '2026-10-24']
    d1 = rj['days'][0]
    assert [b['id'] for b in d1['blocks']] == ['B', 'SLOW', 'ENS', 'B', 'MSF', 'TCP', 'ACT', 'B']
    assert d1['ranking']['T1'] is not None
    assert rj['days'][1]['day2']['verdict']['result'] in ('better', 'keep_B', 'insufficient')
    assert any('K = 1' in x for x in rj['deviations'])
    assert (tmp_path / 'fig' / 'f1_rank_day1.png').exists() and (tmp_path / 'deviations.md').exists()
    # 人が書いたことは書き直しても残る
    dv = tmp_path / 'deviations.md'
    dv.write_text(dv.read_text(encoding='utf-8') + '- 10/23 ブロック 3 の前に電池を替えた\n', encoding='utf-8')
    br.write_report(rep, tmp_path)
    assert '- 10/23 ブロック 3 の前に電池を替えた' in dv.read_text(encoding='utf-8')


# ---------------------------------------------------------------- 電池の直し・時間のずれ・走り直し(合成の周)
def _rows(n, lap, inc=0, start=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lt = lap + rng.normal(0, 0.05) if not callable(lap) else lap(i)
        out.append({'ta': start + 10 * i, 'tg': start + 10 * i + 2, 'incidents': inc if i % 2 == 0 else 0, 'collisions': inc if i % 2 == 0 else 0,
                    'strong': 0, 'tk_failure': 0, 'tk_all': 0, 'tk_stuck_preemptive': 0, 'r1_strong': False, 'r1_coll': False,
                    'gm_coll': False, 'de_coll': False, 'clean': True, 'lap_time_s': round(float(lt), 3), 'reboot': False,
                    'n_reboots': 0, 'collision_locs': [], 'entry_erpm': 5000.0, 't_r1gm_s': 2.0, 't_rest_s': 7.0, 'holds': 0,
                    'ai_s_at_mid': 10.0 * i + 5, 'recoveries': 0})
    return out


def _entry(log, block, cand, rows, vb=None, va=None, day='2026-10-23'):
    desc = {'block': block, 'cand': cand, 'day': day}
    if vb is not None:
        desc.update(V_before=str(vb), V_after=str(va), battery='1')
    return {'log': log, 'block': block, 'cand': cand, 'day': day, 'desc': desc, 'K': 2, 'flags': [], 'rerun': False,
            'score': {'window': {'ai_s': 10.0 * len(rows)}, 'laps_protocol': rows}}


def test_drift_interpolation_uses_neighbouring_b_blocks():
    es = [_entry('a', 1, 'B', _rows(15, 9.0)), _entry('b', 2, 'C1', _rows(15, 9.5, seed=1)), _entry('c', 3, 'B', _rows(15, 10.0, seed=2))]
    fit = br.battery_fit(es)
    assert fit['used'] is False and fit['missing_voltage_logs'] == ['a', 'b', 'c']
    br.apply_battery(es, fit)
    cm = br.compare_day('2026-10-23', es, n_boot=300)
    assert cm['reference']['drift']['flag'] is True                     # B の最初と最後でラップが 1 秒違う
    c = cm['cands']['C1']
    assert c['ref'] == 'interpolated' and abs(c['ref_point']['lap_median_s'] - 9.5) < 0.05
    assert abs(c['diff']['d_lap_median_s']['value']) < 0.06
    # 直し A7(10/21 Fix-all): ラップは「差の 90% 区間が 0 を含まない」で判定する。周のばらつき(σ 0.5 秒)に比べて
    # 小さい差(0.05 秒)は「ずれなし」。protocol.json のままの判定(差 > 0.3 秒)は flag_protocol_v1 に残る
    noisy = lambda m, sd: (lambda i, r=np.random.default_rng(sd): m + r.normal(0, 0.5))   # noqa: E731
    es2 = [_entry('a', 1, 'B', _rows(15, noisy(9.0, 0))), _entry('b', 2, 'C1', _rows(15, 9.5, seed=1)),
           _entry('c', 3, 'B', _rows(15, noisy(9.05, 2)))]
    br.apply_battery(es2, {'used': False})
    cm2 = br.compare_day('2026-10-23', es2, n_boot=300)
    assert cm2['reference']['drift']['flag'] is False and cm2['cands']['C1']['ref'] == 'pooled_B'
    assert cm2['reference']['drift']['rule'] == 'A7' and 'flag_protocol_v1' in cm2['reference']['drift']


def test_battery_correction_when_slope_is_clearly_negative():
    # B の周: 電圧が 8.4 → 7.6 V に下がると、ラップが遅くなる(b < 0)
    def lap_of(i, n=20):
        v = 8.4 - 0.8 * (10.0 * i + 5) / (10.0 * n)
        return 9.5 - 1.0 * (v - 8.0)
    es = [_entry('a', 1, 'B', _rows(20, lambda i: lap_of(i)), 8.4, 7.6), _entry('b', 2, 'B', _rows(20, lambda i: lap_of(i)), 8.4, 7.6)]
    fit = br.battery_fit(es)
    assert fit['used'] is True and abs(fit['b_s_per_V'] + 1.0) < 1e-6 and fit['ci90'][1] < 0
    br.apply_battery(es, fit)
    laps = [r['lap_time_s'] for e in es for r in e['rows']]
    assert max(laps) - min(laps) < 0.01                                   # 電池の分を直すと同じラップになる
    assert all('lap_time_raw_s' in r for r in es[0]['rows'])


@need_1010
def test_rerun_of_same_block_uses_k2_and_remaining_budget():
    bench = {'bench': 'x', 'candidates': [{'id': 'B', 'run': 'r'}], 'days': [{'date': '2026-10-23', 'blocks': [{'block': 1, 'id': 'B'}]}]}
    assign = {L006: 'bench=x day=2026-10-23 block=1 cand=B', L007: 'bench=x day=2026-10-23 block=1 cand=B'}
    es, un = br.collect([L006, L007], bench, assign, LABELS, RAW)
    assert not un and [e['log'] for e in es] == [L006, L007]
    es = br.score_entries(es, bench, LABELS, RAW)
    w1, w2 = es[0]['score']['window'], es[1]['score']['window']
    assert es[0]['K'] == 3 and not es[0]['rerun'] and es[1]['K'] == 2 and es[1]['rerun']
    assert abs(w1['ai_s'] + w2['ai_s'] - 240.0) < 0.5 or not w2['budget_reached']
    assert w2['counted_ai_s'] == pytest.approx(240.0 - w1['ai_s'], abs=0.01)
