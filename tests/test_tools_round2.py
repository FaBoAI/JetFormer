"""夜の道具の直す 2 回目(2026-10-26 fix2)の試験。作り物の記録だけ(tmp)。本物の data/raw・labels は読まない・書かない。

- R2-rec-1  car_check・quick_eval_by_params に 1 本ごとの衝突の最大 G・8G 以上・場所
- R2-rec-2・state2-2  走行中に天井・自動合わせを OFF にした本・設定を変えた本に印(ON の組にも OFF の組にも入れない)
- R2-rec-3  同じ本の中で、衝突のあとに人が止めて戻した発進を「止まってから 3 周(衝突 0)」に数えない
- R2-rec-4  quick_eval.py が読めない本(scorecard の per_log が error だけ)で止まらない
- R2-rec-5  car_check の天井・自動合わせの行に自動合わせの電圧の源
- R2-rec-6  runs_1103 が回 4 の「ON なのに始まらない」「回の形が違う」に印
- 発進の前進の出来事の時刻(cam_epoch_us)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_tools_round2.py
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import car_check as cc  # noqa: E402
import label_events as le  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import section_scorecard as ss  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_lap_anchor import REF, T0_US, course_profile, write_lap_log  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

JF = Path(__file__).resolve().parents[1]
G = le.G
P_SC = {'throttle_gain': 0.95, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 1,
        'auto_target_mps': 4.27, 'launch_boost': 0}


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runs_1103(raw, day='20261103'):
    m = load_mod('runs_1103_t', JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
    m.RAW = raw
    argv, buf = sys.argv, io.StringIO()
    sys.argv = ['runs_1103.py', '--day', day, '--json']
    try:
        with contextlib.redirect_stdout(buf):
            m.main()
    finally:
        sys.argv = argv
    return json.loads(buf.getvalue().strip().splitlines()[-1])


# ------------------------------------------------------------------ R2-rec-2・state2-2 走行中に OFF
def make_ceil_log(raw, log, off_at, desc='ai_no=8 bench=x day=2026-11-03 block=8 cand=EXP tune=2-3'):
    """天井 4.1 + 自動合わせ ON の記録。off_at 秒から ceil_on・sa_on が False(画面のスイッチで OFF にした形)"""
    d, anchors, _ = write_lap_log(raw, log, dur=60.0, false_laps=(), params=P_SC, description=desc)
    for s in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(s))
        t = (z['timestamps_us'] - T0_US) / 1e6
        on = t < off_at
        n = len(t)
        g = np.where(on, 0.95 * 1.04, 0.95)
        z['gains'] = np.stack([np.ones(n), g], axis=1).astype(np.float32)
        z.update(ceil_on=on, ceil_mode=np.zeros(n, np.int8), ceil_applied=np.zeros(n, bool), ceil_left=np.zeros(n, bool),
                 sa_on=on, sa_m=np.where(on, 1.04, np.nan).astype(np.float32), sa_gain=g.astype(np.float32),
                 sa_src=np.ones(n, np.int8), sa_event=np.zeros(n, np.int8), launch_on=np.zeros(n, bool),
                 launch_end=np.zeros(n, np.int8))
        np.savez(s, **z)
    return d


def test_midrun_off_is_flagged_and_grouped_apart(tmp_path):
    raw = tmp_path / 'raw'
    make_ceil_log(raw, '20261103-130000', 1e9, desc='ai_no=8 bench=x day=2026-11-03 block=8 cand=EXP tune=2-3')
    make_ceil_log(raw, '20261103-140000', 25.0, desc='ai_no=9 bench=x day=2026-11-03 block=9 cand=EXP tune=2-4')
    a = s3.analyze_log('20261103-130000', raw, tmp_path / 'labels', {'x': 'B'})
    b = s3.analyze_log('20261103-140000', raw, tmp_path / 'labels', {'x': 'B'})
    sa = ss.score_log('20261103-130000', tmp_path / 'labels', raw)['speed_ctrl']
    sb = ss.score_log('20261103-140000', tmp_path / 'labels', raw)['speed_ctrl']
    assert 'ceil_off_midrun' not in sa['flags'] and 'auto_off_midrun' not in sa['flags']
    assert {'ceil_off_midrun', 'auto_off_midrun'} <= set(sb['flags'])
    assert sb['ceil']['off_at_s'] == pytest.approx(25.0, abs=0.05)
    c = ss.ctrl_cells({'speed_ctrl': sb, 'vbat': {}})
    assert c[0].startswith('ON→途中で OFF(25.0 秒)') and c[1].startswith('ON→途中で OFF(25.0 秒)')
    assert 'ceil_off_midrun' in c[3]
    assert not a['midrun'] and [m['key'] for m in b['midrun']] == ['speed_ceiling', 'speed_auto']
    assert '途中で変えた(天井 1→0 25.0 秒・自動 1→0 25.0 秒)' in b['settings_text']
    ka, ta = qp.key_of(a)
    kb, tb = qp.key_of(b)
    assert ka != kb and '途中で変えた' in tb and '途中で変えた' not in ta
    assert any(f.startswith('途中で変えた') for f in qp.flags_of(b)) and not any(f.startswith('途中で変えた') for f in qp.flags_of(a))
    assert len(qp.group_rows([a, b])) == 2                           # ON の組に入れない
    rows = runs_1103(raw)
    by = {r['log']: r for r in rows}
    assert by['20261103-130000']['midrun'] == []
    assert by['20261103-140000']['midrun'] == ['speed_ceiling 1→0', 'speed_auto 1→0']
    assert any(x.startswith('途中で変えた') for x in by['20261103-140000']['diffs'])


def test_midrun_from_server_events_and_params_end(tmp_path):
    """新しいサーバーの記録の形: events.jsonl の params・log.json の params_end(速度のスライダーを走行中に動かした)"""
    raw = tmp_path / 'raw'
    log = '20261103-150000'
    d, _, _ = write_standing_log(raw, log, params={'throttle_gain': 0.95}, description='ai_no=5 day=2026-11-03 block=5 cand=EXP tune=1-5')
    (d / 'events.jsonl').write_text(json.dumps({'kind': 'params', 'changed': {'throttle_gain': [0.95, 1.0]},
                                                'epoch_us': T0_US + 12_000_000, 't_us': 5}) + '\n', encoding='utf-8')
    j = json.loads((d / 'log.json').read_text(encoding='utf-8'))
    j['drive']['params_end'] = {'throttle_gain': 1.0}
    (d / 'log.json').write_text(json.dumps(j), encoding='utf-8')
    res = s3.analyze_log(log, raw, tmp_path / 'labels', {'runX': 'B'})
    assert res['midrun'] == [{'key': 'throttle_gain', 'from': 0.95, 'to': 1.0, 't_s': pytest.approx(12.0, abs=0.01), 'src': 'event'}]
    assert '速さ 0.95→1 12.0 秒' in res['settings_text']
    r = {x['log']: x for x in runs_1103(raw)}[log]
    assert r['midrun'] == ['throttle_gain 0.95→1.0']


def test_frames_gain_change_is_midrun_when_auto_off(tmp_path):
    """古いサーバーの記録(出来事なし): 自動合わせ OFF でフレームのゲインが途中で変わった = スライダーを動かした"""
    raw = tmp_path / 'raw'
    log = '20261103-160000'
    d, _, _ = write_standing_log(raw, log, params={'throttle_gain': 0.9})
    for s in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(s))
        t = (z['timestamps_us'] - T0_US) / 1e6
        z['gains'] = np.stack([np.ones(len(t)), np.where(t < 20.0, 0.9, 1.0)], axis=1).astype(np.float32)
        np.savez(s, **z)
    res = s3.analyze_log(log, raw, tmp_path / 'labels', {'runX': 'B'})
    assert [(m['key'], m['from'], m['to'], m['src']) for m in res['midrun']] == [('throttle_gain', 0.9, 1.0, 'frames')]


# ------------------------------------------------------------------ R2-rec-3 2 回目の発進
def write_restart_log(raw, log, hit_g=6.0):
    """1 回目: 1 秒に AI 側・1.5 秒に動く・8.0 秒に hit_g で当たる・10 秒に人が止める。2 回目: 14 秒に AI 側(止まった状態)
    ・14.5 秒に動く・そのあと 3 周きれい(反証 restart_credit.py と同じ形)"""
    d = raw / log
    d.mkdir(parents=True)
    (d / 'log.json').write_text(json.dumps({'description': 'ai_no=20 day=2026-11-03 block=20 cand=EXP tune=4-1', 'mode_id': 5,
                                            'drive': {'run': 'runX', 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1],
                                                      'dry_run': False, 'params': {'throttle_gain': 0.9}}}), encoding='utf-8')
    dur, V = 47.0, 2.0
    ft, vt = np.arange(0, dur, 1 / 30.0), np.arange(0, dur, 0.01)
    it = vt + 0.003
    seg1, seg2 = (1.5, 10.0), 14.5
    g1, _, _ = course_profile(dur, 4.0, ())
    g2, _, _ = course_profile(dur, 2.5, ())
    gz = np.zeros(len(it))
    for a, b, g in g1:
        gz[(it >= a) & (it < b) & (it >= seg1[0]) & (it < seg1[1])] = g
    for a, b, g in g2:
        gz[(it >= a + seg2) & (it < b + seg2) & (it >= seg2)] = g
    vv = np.where(((vt >= seg1[0]) & (vt < seg1[1])) | (vt >= seg2), V, 0.0)
    e = -vv / le.KV
    sel_v = ((vt >= 1.0) & (vt < 10.0)) | (vt >= 14.0)
    sel_f = ((ft >= 1.0) & (ft < 10.0)) | (ft >= 14.0)
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    if hit_g:
        k = int(np.searchsorted(it, 8.0))
        acc[k:k + 3, 1] += [-0.5 * hit_g * G, -hit_g * G, -0.3 * hit_g * G]
    v_f = np.maximum(np.interp(ft, vt, vv), 0.1)
    kap = np.radians(np.interp(ft + 0.1, it, gz)) / v_f
    steer = REF[0] + np.clip(kap * 1e4 / 40.0, -480, 480)
    out = np.stack([steer, np.full(len(ft), REF[1] - 200.0)], axis=1)
    us = lambda x: (T0_US + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    for si in range(3):
        a, b = si * 20.0, min((si + 1) * 20.0, dur)
        s = d / ('%s-%04d' % (log, si + 1))
        s.mkdir()
        fm, vm, im = (ft >= a) & (ft < b), (vt >= a) & (vt < b), (it >= a) & (it < b)
        n = int(fm.sum())
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), raw=out[fm].astype(np.float32), out=out[fm].astype(np.float32),
                 gains=np.tile([1.0, 0.9], (n, 1)).astype(np.float32), trim=np.zeros(n, np.float32),
                 lookahead=np.full(n, 3, np.int16), hold=np.zeros(n, bool), select=sel_f[fm], dry_run=np.zeros(n, bool),
                 assist=np.zeros(n, np.int8), model_out=out[fm].astype(np.float32), launch_on=np.zeros(n, bool),
                 launch_end=np.zeros(n, np.int8))
        nv = int(vm.sum())
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=np.tile([1441, 1496, 2000], (nv, 1)).astype(np.int32),
                 valid=np.ones((nv, 3), bool), select=sel_v[vm], erpm=np.round(e[vm]).astype(np.int32),
                 board_t_us=(us(vt[vm]) - T0_US + 10_000_000))
        gy = np.zeros((int(im.sum()), 3))
        gy[:, 2] = gz[im]
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32), gyro=gy.astype(np.float32))
        np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * n))


def test_restart_after_collision_is_not_counted(tmp_path):
    raw = tmp_path / 'raw'
    write_restart_log(raw, '20261103-170000')
    res = s3.analyze_log('20261103-170000', raw, tmp_path / 'labels', {'runX': 'B'})
    assert res['collisions']['all'] == 1
    st2 = [s for s in res['standing'] if s.get('valid')]
    assert len(st2) == 1 and st2[0]['start_no'] == 2 and st2[0]['coll_before'] == 1 and st2[0]['counted'] is False
    assert s3.best_standing(res) is None                             # 前は 27.0 秒(衝突 0)として出ていた
    r = qp.run_rows([res])[0]
    assert r['standing_best_s'] is None and any(f.startswith('数えない発進 1') for f in r['flags'])
    g = qp.group_rows([res])[0]
    assert g['standing_n'] == 0 and g['standing_clean_n'] == 0 and g['standing_le25'] == 0 and g['standing_excluded_n'] == 1
    tbl = '\n'.join(s3.standing_table([res]))
    assert '数えない(この本の前の衝突 1)' in tbl and '2 回目の発進' in tbl


def test_restart_without_earlier_collision_is_counted(tmp_path):
    """対照: 前に衝突が無ければ 2 回目の発進も数える(印は付く)"""
    raw = tmp_path / 'raw'
    write_restart_log(raw, '20261103-171000', hit_g=None)
    res = s3.analyze_log('20261103-171000', raw, tmp_path / 'labels', {'runX': 'B'})
    b = s3.best_standing(res)
    assert b is not None and b['start_no'] == 2 and b['coll_before'] == 0 and b['counted'] is True
    assert '2 回目の発進' in '\n'.join(s3.standing_table([res]))


def test_first_start_unchanged(tmp_path):
    """1 回だけの発進(今までの形)は counted = valid で、best_standing も前と同じ"""
    raw = tmp_path / 'raw'
    write_standing_log(raw, '20261103-172000', hits=[(30.0, 5.0)])
    res = s3.analyze_log('20261103-172000', raw, tmp_path / 'labels', {'runX': 'B'})
    v = [s for s in res['standing'] if s.get('valid')]
    assert v and all(s['start_no'] == 1 and s['coll_before'] == 0 and s['counted'] for s in v)
    assert s3.best_standing(res) is v[0]


# ------------------------------------------------------------------ R2-rec-1 衝突の最大 G・場所
def test_collision_summary_and_car_check_text(tmp_path):
    raw = tmp_path / 'raw'
    write_standing_log(raw, '20261103-173000', hits=[(12.0, 9.0), (21.0, 3.0), (30.5, 5.0)])
    res = cc.check_log('20261103-173000', tmp_path / 'labels', raw)
    cs = res['collisions']
    assert cs['n'] == 3 and cs['n_ge8'] == 1 and cs['strong'] == 2 and cs['max_g'] >= 8.0
    assert sum(cs['by_place'].values()) == 3 and set(cs['by_place']) == set(ss.PLACES)
    txt = cc.to_text(res)
    line = [x for x in txt.splitlines() if '衝突(AI・' in x]
    assert line and line[0].startswith('!') and '8G 以上 1' in line[0] and '最大' in line[0]
    sa = s3.analyze_log('20261103-173000', raw, tmp_path / 'labels', {'runX': 'B'})
    r = qp.run_rows([sa])[0]
    assert r['coll_ge8'] == 1 and r['coll_max_g'] >= 8.0 and r['coll_places']
    md, _, _ = qp.to_markdown([sa], '20261103', None)
    assert '最大 G・8G 以上・場所' in md


def test_place_of_names():
    assert ss.place_of({'loc_detail': 'GM_out', 'loc': 'GM'}) == 'GM'
    assert ss.place_of({'loc_detail': 'SW', 'loc': 'other'}) == 'SW'
    assert ss.place_of({'loc_detail': None, 'loc': 'other'}) == '?'
    assert ss.place_of({'loc': 'LH'}) == 'LH'
    assert ss.collision_text(ss.collision_summary([])) == '0 回'


# ------------------------------------------------------------------ R2-rec-5 自動合わせの電圧の源
@pytest.mark.parametrize('src,name', [(0, 'なし'), (1, '基板'), (2, '説明欄')])
def test_car_check_shows_auto_voltage_source(tmp_path, src, name):
    raw = tmp_path / 'raw'
    log = '20261103-131000'
    d, _, _ = write_lap_log(raw, log, dur=60.0, false_laps=(), params=dict(P_SC), description='ai_no=9 day=2026-11-03 V_before=8.31')
    for s in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(s))
        n = len(z['timestamps_us'])
        z.update(ceil_on=np.ones(n, bool), ceil_mode=np.zeros(n, np.int8), ceil_applied=np.zeros(n, bool),
                 ceil_left=np.zeros(n, bool), sa_on=np.ones(n, bool), sa_m=np.ones(n, np.float32),
                 sa_gain=np.full(n, 0.95, np.float32), sa_src=np.full(n, src, np.int8), sa_event=np.zeros(n, np.int8))
        np.savez(s, **z)
    res = cc.check_log(log, tmp_path / 'labels', raw)
    line = [x for x in cc.to_text(res).splitlines() if '自動合わせ' in x and '天井' in x]
    assert line and ('電圧の源 %s' % name) in line[0]


# ------------------------------------------------------------------ R2-rec-4 quick_eval の読めない本
def test_quick_eval_skips_unreadable_rows(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    day = '20001103'
    write_standing_log(raw, day + '-100000', description='ai_no=1 cand=EXP tune=1-1', params={'throttle_gain': 0.9})
    bad = raw / (day + '-101500')
    (bad / (bad.name + '-0001')).mkdir(parents=True)
    (bad / 'log.json').write_text(json.dumps({'description': 'ai_no=2 cand=EXP tune=1-2', 'mode_id': 5,
                                              'drive': {'run': 'runX', 'dry_run': False, 'params': {'throttle_gain': 0.9}}}),
                                  encoding='utf-8')
    for nm in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT', 'TOF_FRONT'):
        (bad / (bad.name + '-0001') / (nm + '.npz')).write_bytes(b'')
    write_standing_log(raw, day + '-103000', description='ai_no=3 cand=EXP tune=1-3', params={'throttle_gain': 0.9})
    rep = tmp_path / 'reports'
    out = rep / ('bench_%s_tune' % day) / 'since_0000'
    with contextlib.redirect_stdout(io.StringIO()):
        ss.main(['--day', day, '--raw', str(raw), '--labels', str(tmp_path / 'labels'), '--out', str(out / 'nightly' / 'scorecard'),
                 '--boot', '20'])
    sc = json.loads((out / 'nightly' / 'scorecard' / 'scorecard.json').read_text(encoding='utf-8'))
    assert 'error' in sc['per_log'][day + '-101500']
    qe = load_mod('quick_eval_t', JF / 'experiments' / '20261024_tune' / 'quick_eval.py')
    qe.RAW = raw
    real = qe.Path
    qe.Path = lambda s_, *a: real(str(s_).replace('/home/jetson/jetformer_reports', str(rep)), *a)
    monkeypatch.setattr(sys, 'argv', ['quick_eval.py', '0000', '--day=' + day, '--no-run'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        qe.main()                                                    # 前は KeyError で止まった
    txt = buf.getvalue()
    assert '(読めない: ' in txt and '| 3 | 1-3 |' in txt and '段(モデル・速さ)' in txt


# ------------------------------------------------------------------ R2-rec-6 runs_1103 の回 4
def _rec(raw, name, ai_no, block, tune, params, events):
    base = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1,
            'speed_auto': 0, 'launch_us': 250, 'launch_mps': 2.5, 'launch_max_s': 1.2}
    d = raw / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    desc = 'ai_no=%d bench=x day=2026-11-03 block=%d cand=EXP tune=%s battery=#A V_before=8.30 memo=' % (ai_no, block, tune)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
        'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'dry_run': False, 'params': dict(base, **params)}},
        ensure_ascii=False), encoding='utf-8')
    (d / 'events.jsonl').write_text(''.join(json.dumps(dict(e, kind='launch_boost'), ensure_ascii=False) + '\n' for e in events),
                                    encoding='utf-8')


def test_runs_1103_round4_launch_not_started_and_form(tmp_path):
    bench = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    blocks = {b['block']: b for d in bench['days'] if d['date'] == '2026-11-03' for b in d['blocks']}
    if not all(blocks.get(k, {}).get('run_id') == '4-%d' % (k - 19) for k in (20, 21, 22)):
        pytest.skip('11/3 の回 4 のブロックが前提と違う')
    raw = tmp_path / 'raw'
    skip = [{'event': 'skip', 'why': 'near', 'reason': '前が近い(TTC の中央値 < 3.3 秒)'}]
    start = [{'event': 'start'}, {'event': 'end', 'why': 'mps', 'reason': '速さ 2.5 m/s'}]
    _rec(raw, '20261103-110000', 2, 20, '4-1', {'speed_ceiling': 1, 'launch_boost': 1}, start)
    _rec(raw, '20261103-111000', 3, 21, '4-2', {'speed_ceiling': 1, 'launch_boost': 1}, skip)
    _rec(raw, '20261103-112000', 4, 22, '4-3', {'speed_ceiling': 0, 'launch_boost': 0, 'throttle_gain': 1.0}, [])
    by = {r['run_id']: r for r in runs_1103(raw)}
    assert by['4-1']['diffs'] == []
    assert by['4-2']['diffs'] and by['4-2']['diffs'][0].startswith('発進の前進が始まっていない')
    assert any(x.startswith('回 4 の形が 4-1(No.2)と違う') and 'launch_boost 0(4-1 は 1)' in x and 'speed_ceiling 0(4-1 は 1)' in x
               for x in by['4-3']['diffs'])


# ------------------------------------------------------------------ 発進の前進の出来事の時刻
def test_launch_events_prefer_epoch_and_drop_monotonic(tmp_path):
    d = tmp_path / 'L'
    d.mkdir()
    lines = [{'kind': 'launch_boost', 'event': 'start', 'cam_t_us': 36_000_000_000, 'cam_epoch_us': T0_US + 1_000_000},
             {'kind': 'launch_boost', 'event': 'skip', 'cam_t_us': 36_100_000_000, 'reason': 'x'},     # 単調時計だけ(古い形)
             {'kind': 'launch_boost', 'event': 'end', 'cam_t_us': T0_US + 2_000_000, 'reason': 'y'}]  # 試験の形(UNIX µs)
    (d / 'events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines), encoding='utf-8')
    ev = s3.launch_events('L', tmp_path)
    assert ev == [(T0_US + 1_000_000, 'start', ''), (None, 'skip', 'x'), (T0_US + 2_000_000, 'end', 'y')]


# ------------------------------------------------------------------ collect.py の 1 回目の発進だけ
def test_collect_summary_first_only():
    col = load_mod('collect_t', JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py')
    base = {'session': '3', 'model': 'B', 'gain': 0.9, 'settings': {}, 'launch_state': 'off', 'to_r1_s': 6.0, 'loss_s': 0.6,
            'move_0p2_s': 0.3, 'collisions': [], 'collisions_first_lap': [], 'log': 'L', 't_ai': 1.0, 'no': 1}
    rows = [dict(base, start_no=1), dict(base, start_no=2, to_r1_s=9.0, t_ai=20.0)]
    a = col.summarize(rows)[0]['by_launch']['off']
    b = col.summarize(rows, first_only=True)[0]['by_launch']['off']
    assert a['n'] == 2 and a['restarts'] == 1 and a['to_r1_median_s'] == 7.5
    assert b['n'] == 1 and b['restarts'] == 0 and b['to_r1_median_s'] == 6.0
