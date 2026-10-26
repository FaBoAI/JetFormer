"""車の確認(tools/car_check.py、spec 5.2・6.3 T-4〜T-6)と、成績表・夜のレポートの速さの制御・電圧の要約の試験。
実機・GPU 不要(合成の記録。10/23・10/24 の記録と 10/24 の台本の数があるときだけ、それと合うかも見る)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -m pytest -q tests/test_car_check.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_check as cc  # noqa: E402
import section_scorecard as ss  # noqa: E402
from test_lap_anchor import T0_US, write_lap_log  # noqa: E402


# ---------------------------------------------------------------- T-5: 判定の境
def _m(ratio=0.95, s0=0.0, a3_med=3.0, a3_max=5.0, passes=5, v0=8.3, n_l=100, n_r=100, n_s0=500, p90=4.0, sw=None, ceil=None):
    return {'turn': {'ratio': ratio, 'n_left': n_l, 'n_right': n_r}, 's0': {'s0_us': s0, 'n': n_s0},
            'r1_bump': {'passes': passes, 'a3_median': a3_med, 'a3_max': a3_max},
            'vbat': {'start_v': v0, 'source': 'board'}, 'laps': {'p90_median_mps': p90, 'sw_max_median_mps': sw},
            'speed_ctrl': {'speed_ceiling': ceil}}


def _st(j, item):
    return next(x['status'] for x in j['items'] if x['item'] == item)


@pytest.mark.parametrize('ratio,status,code', [(0.90, 'ok', 0), (0.8999, 'fix', 1), (0.85, 'fix', 1), (0.8499, 'fail', 2),
                                               (0.96, 'ok', 0), (0.69, 'fail', 2)])
def test_judge_ratio_boundaries(ratio, status, code):
    j = cc.judge(_m(ratio=ratio))
    assert _st(j, 'turn') == status and j['code'] == code


@pytest.mark.parametrize('s0,status,code', [(10.0, 'ok', 0), (-10.0, 'ok', 0), (10.1, 'fix', 1), (15.0, 'fix', 1),
                                            (15.1, 'fail', 2), (-10.1, 'caution', 1), (-17.8, 'caution', 1), (22.0, 'fail', 2)])
def test_judge_s0_boundaries(s0, status, code):
    j = cc.judge(_m(s0=s0))
    assert _st(j, 's0') == status and j['code'] == code


@pytest.mark.parametrize('med,mx,status', [(4.49, 6.99, 'ok'), (4.5, 5.0, 'fail'), (3.9, 7.0, 'fail'), (3.93, 6.38, 'ok')])
def test_judge_r1_board_boundaries(med, mx, status):
    j = cc.judge(_m(a3_med=med, a3_max=mx))
    assert _st(j, 'r1_bump') == status and j['code'] == (2 if status == 'fail' else 0)
    if status == 'fail':
        assert 'ゲイン −0.05' in next(x['text'] for x in j['items'] if x['item'] == 'r1_bump')


@pytest.mark.parametrize('v0,status,code', [(7.70, 'ok', 0), (7.69, 'caution', 1), (None, 'na', 0)])
def test_judge_voltage(v0, status, code):
    j = cc.judge(_m(v0=v0))
    assert _st(j, 'vbat') == status and j['code'] == code


def test_judge_insufficient_and_priority():
    assert cc.judge(_m(n_l=29))['code'] == 3
    assert cc.judge(_m(n_r=29))['code'] == 3
    assert cc.judge(_m(n_s0=199))['code'] == 3
    assert cc.judge(_m(passes=2))['code'] == 3
    j = cc.judge(_m(n_l=30, n_r=30, n_s0=200, passes=3))
    assert j['code'] == 0 and j['headline'] == '本番の設定へ進めます'
    # 不合格は足りないより先(板が 7G を超えたら、ほかが足りなくても止める)
    assert cc.judge(_m(n_l=10, a3_max=7.5))['code'] == 2
    # 足りないは直す・注意より先
    assert cc.judge(_m(n_s0=10, ratio=0.87))['code'] == 3
    assert cc.judge(_m(ratio=None))['code'] == 3
    assert '判定に足りません(もう 1 本)' == cc.judge(_m(passes=0))['headline']


@pytest.mark.parametrize('p90,sw,ceil,p1,p2', [(3.7499, None, None, '+0.05', '+0.05'), (3.75, None, None, '+0.03', '+0.03'),
                                               (3.90, None, None, 'そのまま', 'そのまま'), (4.0999, None, None, 'そのまま', 'そのまま'),
                                               (4.10, None, None, '−0.03', 'そのまま'), (4.25, None, None, '−0.05', '−0.05'),
                                               (4.0, 4.3, None, '−0.05', '−0.05'), (4.0, 4.29, None, 'そのまま', 'そのまま')])
def test_speed_advice_table(p90, sw, ceil, p1, p2):
    a = cc.speed_advice(p90, sw, ceil)
    assert (a['p1'], a['p2']) == (p1, p2)
    j = cc.judge(_m(p90=p90, sw=sw))
    assert _st(j, 'speed') == 'info' and j['code'] == 0          # 速さの直しは番号に入れない
    assert '天井 ON' in cc.speed_advice(4.15, None, True)['text']
    assert cc.speed_advice(None)['p1'] is None


# ---------------------------------------------------------------- 合成の記録で通しで
def test_check_log_synthetic(tmp_path):
    log = '20261103-120000'
    write_lap_log(tmp_path, log, left_gain=36.0, right_gain=40.0, s0_us=0.0, vbat=(8.30, 8.10), r1_bump_g=3.0)
    r = cc.check_log(log, tmp_path / 'no_labels', tmp_path)
    assert r['labels'] == 'computed' and not r.get('error')
    assert r['turn']['ratio'] == pytest.approx(0.90, abs=0.03) and r['turn']['n_left'] >= 30 and r['turn']['n_right'] >= 30
    assert r['s0']['n'] >= 200 and abs(r['s0']['s0_us']) < 8
    assert r['top_speed']['mps_p90'] == pytest.approx(2.0, abs=0.01)
    assert r['laps']['n'] == 4 and r['laps']['p90_median_mps'] == pytest.approx(2.0, abs=0.01)
    assert r['r1_bump']['passes'] == 5 and r['r1_bump']['a3_median'] == pytest.approx(3.0, abs=0.05)
    assert r['vbat']['source'] == 'board' and r['vbat']['start_v'] == pytest.approx(8.30, abs=0.01)
    assert r['vbat']['end_v'] == pytest.approx(8.10, abs=0.01) and r['vbat']['min_v'] == pytest.approx(8.10, abs=0.01)
    assert len(r['reference']['r1_anchors_rejected']) == 1          # 2 周目の偽の起点は外した
    assert r['judge']['code'] in (0, 1) and r['seconds'] < 30
    txt = cc.to_text(r)
    assert '車の確認' in txt and '左右の旋回の効き' in txt


def test_check_log_steer_offset_and_ratio_follow_the_car(tmp_path):
    """左に曲がりにくい車(比 0.75)・右へ流れる車(s0 +20 µs)→ 不合格。左へ流れる(−15 µs)→ 注意。
    (左右の効きが違うと、1 本の直線で当てはめる s0 はずれる = steer_offset.py の作り。s0 は左右をそろえた車で見る)"""
    write_lap_log(tmp_path, '20261103-120000', left_gain=30.0, right_gain=40.0, r1_bump_g=3.0)
    r = cc.check_log('20261103-120000', tmp_path / 'no_labels', tmp_path)
    assert r['turn']['ratio'] == pytest.approx(0.75, abs=0.03) and r['judge']['code'] == 2
    assert {x['item']: x['status'] for x in r['judge']['items']}['turn'] == 'fail'
    write_lap_log(tmp_path, '20261103-120100', left_gain=40.0, right_gain=40.0, s0_us=20.0, r1_bump_g=3.0)
    r = cc.check_log('20261103-120100', tmp_path / 'no_labels', tmp_path)
    assert r['s0']['s0_us'] == pytest.approx(20.0, abs=6) and r['judge']['code'] == 2
    write_lap_log(tmp_path, '20261103-120200', left_gain=40.0, right_gain=40.0, s0_us=-15.0, r1_bump_g=3.0, vbat=(7.6, 7.5))
    r = cc.check_log('20261103-120200', tmp_path / 'no_labels', tmp_path)
    assert r['s0']['s0_us'] < -10 and r['judge']['code'] == 1
    st = {x['item']: x['status'] for x in r['judge']['items']}
    assert st['s0'] == 'caution' and st['vbat'] == 'caution'


def test_main_exit_code_json_and_latest(tmp_path, capsys):
    raw = tmp_path / 'raw'
    raw.mkdir()
    write_lap_log(raw, '20261103-130000', r1_bump_g=7.5)                    # 板 7.5G → 不合格
    write_lap_log(raw, '20261103-131000', r1_bump_g=3.0, dry_run=True)        # dry run は --latest で選ばない
    out = tmp_path / 'cc.json'
    code = cc.main(['20261103-130000', '--raw', str(raw), '--labels', str(tmp_path / 'nl'), '--json', str(out)])
    assert code == 2
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['schema'] == cc.SCHEMA and d['judge']['code'] == 2 and d['r1_bump']['a3_max'] >= 7.0
    assert '不合格' in capsys.readouterr().out
    assert cc.latest_log(raw, '20261103') == '20261103-130000'
    assert cc.main(['--latest', '--day', '20261103', '--raw', str(raw), '--labels', str(tmp_path / 'nl'), '--no-json']) == 2
    assert cc.main(['--latest', '--day', '20261104', '--raw', str(raw), '--no-json']) == 3
    assert cc.main(['20261103-999999', '--raw', str(raw), '--no-json']) == 3


def test_very_short_record_is_insufficient_not_an_error(tmp_path):
    """4 フレームの記録(10/23 の 203740 で窓の計算が落ちた)→ 足りない(3)。落ちない"""
    log = '20261103-135900'
    write_lap_log(tmp_path, log, dur=0.13, false_laps=())
    r = cc.check_log(log, tmp_path / 'no_labels', tmp_path)
    assert not r.get('error') and r['judge']['code'] == 3
    assert len(cc.conv_same(np.ones(4), 61)) == 4
    x = np.random.default_rng(0).random(100)
    assert np.allclose(cc.conv_same(x, 61), np.convolve(x, np.ones(61), 'same'))


def test_old_firmware_voltage_from_description(tmp_path):
    log = '20261103-140000'
    write_lap_log(tmp_path, log, description='ai_no=2 battery=#1 V_before=8.21 V_after=7.95V memo=x', r1_bump_g=3.0)
    r = cc.check_log(log, tmp_path / 'no_labels', tmp_path)
    assert r['vbat']['source'] == 'description' and r['vbat']['start_v'] == 8.21 and r['vbat']['end_v'] == 7.95
    # 基板の値(Vb_before)があればそちらを使う
    vb = ss.vbat_summary(None, None, None, None, ss.parse_description('V_before=8.30 Vb_before=8.27 Vb_after=8.01 Vb_min=7.80'))
    assert (vb['start_v'], vb['end_v'], vb['min_v']) == (8.27, 8.01, 7.80)
    assert ss.vbat_summary(None, None, None, None, {})['source'] is None


def test_scene_stats_voltage(tmp_path):
    log = '20261103-140100'
    d, _, _ = write_lap_log(tmp_path, log, r1_bump_g=3.0)
    for i, sj in enumerate(sorted(d.glob('*-000[0-9]'))):
        (sj / 'scene.json').write_text(json.dumps({'description': 'ai_no=2 test', 'stats': {
            'vbat': {'first': 8.3 - 0.1 * i, 'last': 8.25 - 0.1 * i, 'min': 8.2 - 0.1 * i, 'median': 8.27 - 0.1 * i}}}),
            encoding='utf-8')
    vb = ss.vbat_summary(None, T0_US, log, tmp_path, {})
    assert len(list(d.glob('*/scene.json'))) == 3                    # 44 秒 = 20 秒のシーン 3 つ
    assert vb['source'] == 'scene' and vb['start_v'] == 8.3 and vb['end_v'] == pytest.approx(8.05) and vb['min_v'] == pytest.approx(8.0)
    assert vb['median_v'] == pytest.approx(8.17)


# ---------------------------------------------------------------- 速さの天井・自動合わせの記録の要約
def _ai_cmd(n=600, dt=1 / 30.0):
    at = np.arange(n) * dt
    return at, {'timestamps_us': (T0_US + np.round(at * 1e6)).astype(np.int64), 'dry_run': np.zeros(n, bool)}


def test_speed_ctrl_summary_ceiling_fractions_and_switch_flag():
    at, A = _ai_cmd()
    mode = np.zeros(len(at), np.int8)
    mode[60:120] = 1                     # 2 秒弱めた
    mode[120:150] = 2                    # 1 秒中立
    mode[300:330:2] = 1                  # 1 秒に 15 回の入れ替わり(0.5 秒の中で 1 と 0 が交互)
    A.update(ceil_mode=mode, ceil_on=np.zeros(len(at), bool), ceil_applied=np.zeros(len(at), bool),
             ceil_left=np.r_[np.ones(150, bool), np.zeros(len(at) - 150, bool)])
    s = ss.speed_ctrl_summary(A, T0_US, at, np.ones(len(at), bool), {'speed_ceiling': 0, 'speed_auto': 0})
    c = s['ceil']
    assert s['ceil_recorded'] and not s['auto_recorded'] and s['speed_ceiling'] == 0
    assert c['applied_frac'] == 0 and c['on_frames'] == 0                 # OFF の日は影だけ
    assert c['weak_frac'] == pytest.approx((60 + 15) / 600, abs=1e-3) and c['neutral_frac'] == pytest.approx(30 / 600, abs=1e-3)
    assert c['left_frac'] == pytest.approx(0.25, abs=1e-3)
    assert c['switch_max_per_s'] > 10 and 'ceil_switch_gt10_per_s' in s['flags']
    # 入れ替わりが少なければ印なし。計算の失敗(−2)は印
    A['ceil_mode'] = np.where(np.arange(len(at)) < 590, mode * 0, -2).astype(np.int8)
    s = ss.speed_ctrl_summary(A, T0_US, at, np.ones(len(at), bool), {})
    assert s['ceil']['switch_max_per_s'] <= 1 and s['flags'] == ['ceil_failed']
    # 欄の無い古い記録
    _, B = _ai_cmd()
    s = ss.speed_ctrl_summary(B, T0_US, at, np.ones(len(at), bool), {'speed_ceiling': 1})
    assert not s['ceil_recorded'] and not s['auto_recorded'] and s['speed_ceiling'] == 1 and s['flags'] == []


def test_speed_ctrl_summary_auto_edge_and_flip_flags():
    at, A = _ai_cmd(n=1200)
    n = len(at)
    close = [100, 300, 500, 700, 900, 1100]                 # 周を閉じたフレーム
    mfb_after = [1.03, 1.06, 1.09, 1.10, 1.10, 1.10]
    mfb = np.ones(n)
    p90 = np.full(n, np.nan)
    ev = np.zeros(n, np.int8)
    for i, (k, v) in enumerate(zip(close, mfb_after)):
        mfb[k:] = v
        p90[k] = 3.8 + 0.01 * i
        ev[k] = 1 if i < 3 else 3
    A.update(sa_on=np.ones(n, bool), sa_m=mfb * 1.02, sa_m_ff=np.full(n, 1.02), sa_m_fb=mfb, sa_gain=mfb * 0.918,
             sa_src=np.ones(n, np.int8), sa_v=np.full(n, 8.2), sa_event=ev, sa_lap_p90=p90, sa_lap_s=np.where(np.isfinite(p90), 8.6, np.nan),
             sa_applied=np.ones(n, bool))
    s = ss.speed_ctrl_summary(A, T0_US, at, np.ones(n, bool), {'speed_ceiling': 1, 'speed_auto': 1})
    a = s['auto']
    assert a['m_fb_end'] == pytest.approx(1.10) and a['laps_used'] == 6 and a['laps_skipped'] == 0
    assert a['edge_run_max'] == 3 and 'auto_edge_3laps' in s['flags'] and a['src'] == 1 and a['v_first'] == pytest.approx(8.2)
    assert a['events'] == {'corrected': 3, 'no_raise_ceiling': 3}
    assert [x['m_fb'] for x in a['laps']] == pytest.approx(mfb_after)
    # 揺れ: 上げ下げの向きが 3 回続けて入れ替わる
    mfb2 = np.ones(n)
    for k, v in zip(close, [1.03, 1.00, 1.03, 1.00, 1.03, 1.00]):
        mfb2[k:] = v
    A['sa_m_fb'] = mfb2
    s = ss.speed_ctrl_summary(A, T0_US, at, np.ones(n, bool), {})
    assert s['auto']['flip_run_max'] >= 3 and 'auto_flip_3laps' in s['flags'] and 'auto_edge_3laps' not in s['flags']


def test_scorecard_and_bench_report_show_controls(tmp_path):
    """成績表の 5 章・夜のレポートの 13 章に、外した起点・天井・自動合わせ・電圧・車の確認が出る"""
    import bench_report as br
    log = '20261103-150000'
    write_lap_log(tmp_path, log, vbat=(8.3, 8.1), r1_bump_g=3.0)
    sc = ss.score_log(log, tmp_path / 'nl', tmp_path)
    assert sc['vbat']['source'] == 'board' and sc['speed_ctrl']['ceil_recorded'] is False
    assert sc['vbat']['start_v'] == pytest.approx(8.3, abs=0.01) and sc['vbat']['min_v'] == pytest.approx(8.1, abs=0.01)
    md = '\n'.join(ss.per_log_ctrl_table({log: sc}))
    assert '## 5. 外した R1 の起点' in md and log in md and 'heading+impact' in md and '(基板)' in md
    e = {'log': log, 'block': 1, 'cand': 'B', 'score': sc}
    br.car_checks([e], tmp_path / 'nl', tmp_path)
    assert e['car_check']['code'] in (0, 1, 2, 3) and e['car_check']['ratio'] is not None
    md2 = '\n'.join(br.ctrl_section({'days': [{'day': '2026-11-03', 'entries': [e]}]}))
    assert '## 13. 速さの制御・電圧・車の確認' in md2 and e['car_check']['label'] in md2
    # 壊れた記録でもレポートは止まらない
    bad = {'log': '20261103-000000', 'block': 1, 'cand': 'B', 'score': {}}
    br.car_checks([bad], tmp_path / 'nl', tmp_path)
    assert bad['car_check']['code'] is None and bad['car_check']['error']


# ---------------------------------------------------------------- T-4・T-6: 10/24 の記録(あるときだけ)
REP = Path('/home/jetson/jetformer_reports/bench_20261024_report')
COND = REP / 'cond_data' / 'cond_metrics.json'
S0 = Path('/home/jetson/jetformer/experiments/20261024_report/col/steer_offset.json')
BUMP = REP / 'fix_data' / 'r1_bump.json'
LABELS = Path('/home/jetson/jetformer/data/labels')
RAW = Path('/home/jetson/jetformer/data/raw')
LOGS = {'9': '20261024-162033', '11': '20261024-183123', '12': '20261024-183437', '18': '20261024-185703'}


def _need():
    if not (COND.exists() and S0.exists() and BUMP.exists() and all((RAW / lg).exists() for lg in LOGS.values())):
        pytest.skip('10/24 の記録・台本の数が無い')


@pytest.mark.parametrize('no', sorted(LOGS))
def test_records_match_1024_scripts(no):
    _need()
    lg = LOGS[no]
    cond = {r['log']: r for r in json.loads(COND.read_text(encoding='utf-8'))['runs']}[lg]
    s0 = {r['log']: r for r in json.loads(S0.read_text(encoding='utf-8'))}[lg]
    bump = json.loads(BUMP.read_text(encoding='utf-8'))
    brun = {r['log']: r for r in bump['runs']}[lg]
    bmax = max(p['a3_max'] for p in bump['passes'] if p['log'] == lg)
    r = cc.check_log(lg)
    assert r['turn']['ratio'] == pytest.approx(cond['turn_gain_left'] / cond['turn_gain_right'], abs=0.01)
    assert r['turn']['left'] == pytest.approx(cond['turn_gain_left'], abs=0.01)
    assert r['turn']['right'] == pytest.approx(cond['turn_gain_right'], abs=0.01)
    assert r['turn']['n_left'] == cond['turn_n_left'] and r['turn']['n_right'] == cond['turn_n_right']
    assert r['s0']['s0_us'] == pytest.approx(s0['s0_us'], abs=0.5) and r['s0']['n'] == s0['frames']
    assert r['r1_bump']['a3_median'] == pytest.approx(brun['a3_max_median'], abs=0.05)
    assert r['r1_bump']['a3_max'] == pytest.approx(bmax, abs=0.05) and r['r1_bump']['passes'] == brun['passes']
    assert r['top_speed']['erpm_p90'] == pytest.approx(cond['erpm_p90_ai'], abs=1)
    assert r['top_speed']['erpm_p90_HS'] == pytest.approx(cond['erpm_p90_HS'], abs=1)


def test_records_repair_before_after_split():
    """修理の前(1・2 回目)と後(3 回目)で、左右の比が重ならない・s0 が約 +20 µs ずれる(10/24 の解析 B・衝突の解析)"""
    _need()
    cond = json.loads(COND.read_text(encoding='utf-8'))['runs']
    before, after, s0b, s0a = [], [], [], []
    for c in cond:
        if not (RAW / c['log']).exists():
            continue
        r = cc.check_log(c['log'])
        if r.get('error'):
            continue
        ok = r['turn']['n_left'] >= 30 and r['turn']['n_right'] >= 30
        if c['session'] in (1, 2):
            before += [r['turn']['ratio']] if ok else []
            s0b += [r['s0']['s0_us']] if r['s0']['s0_us'] is not None else []
        else:
            after += [r['turn']['ratio']] if ok else []
            s0a += [r['s0']['s0_us']] if r['s0']['s0_us'] is not None else []
    assert len(before) >= 7 and len(after) >= 8
    assert min(before) > max(after)
    assert 0.82 <= min(before) and max(before) <= 0.96 and 0.69 <= min(after) and max(after) <= 0.81
    shift = float(np.median(s0a) - np.median(s0b))
    assert 15 <= shift <= 30


def test_records_time_within_30s(tmp_path):
    """T-6: 60 秒より長い記録(No.9、137 秒)で、ラベルのファイルがあってもその場で作っても 30 秒以内"""
    _need()
    lg = LOGS['9']
    t = time.time()
    r1 = cc.check_log(lg)
    t1 = time.time() - t
    t = time.time()
    r2 = cc.check_log(lg, tmp_path / 'no_labels')
    t2 = time.time() - t
    assert r1['labels'] == 'file' and r2['labels'] == 'computed'
    assert r1['duration_s'] >= 60 and t1 < 30 and t2 < 30
    assert r2['turn']['ratio'] == pytest.approx(r1['turn']['ratio'], abs=0.01)
    assert r2['judge']['code'] == r1['judge']['code'] == 1                # No.9: s0 −17.8 µs は「注意」
