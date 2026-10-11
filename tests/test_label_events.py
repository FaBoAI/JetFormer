"""ラベル付け(tools/label_events.py)のテスト。実機・GPU不要(合成した配列と、小さな合成ログ)

    cd ~/jetformer && python -m pytest -q tests/test_label_events.py
"""

import json
import os
import sys

import numpy as np
import pytest

from jetformer import drive

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import label_events as le  # noqa: E402

REF = (1443.0, 1498.0)
G = le.G


# ---------------------------------------------------------------- ゲインの割り戻し
def _random_frames(n=400, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.stack([rng.uniform(1100, 1800, n), rng.uniform(1250, 1750, n)], axis=1)
    gains = np.stack([rng.uniform(0.8, 1.8, n), rng.uniform(0.5, 2.0, n)], axis=1)
    trim = rng.uniform(-30, 30, n)
    return raw, gains, trim


def test_forward_matches_drive_apply_gains():
    """配列版の apply_gains は drive.apply_gains(走行で使っている計算)と同じ"""
    raw, gains, trim = _random_frames()
    out = le.apply_gains_vec(raw, gains, trim, REF)
    for i in range(len(raw)):
        p = {'steer_gain': gains[i, 0], 'throttle_gain': gains[i, 1], 'steer_trim_us': trim[i]}
        assert out[i] == pytest.approx(drive.apply_gains(raw[i], REF, p, 1000, 2000), abs=1e-6)


@pytest.mark.parametrize('forward_only', [True, False])
def test_invert_gains_recovers_raw(forward_only):
    """割り戻すと raw に戻る(上下限に張り付いたフレームを除く)。速度のゲインの規則は2通り"""
    raw, gains, trim = _random_frames(seed=1)
    out = le.apply_gains_vec(raw, gains, trim, REF, forward_only=forward_only).astype(np.float32)
    back = le.invert_gains(out, gains.astype(np.float32), trim.astype(np.float32), REF, forward_only)
    ok = ~le.saturated(out).any(axis=1)
    assert ok.sum() > 200
    assert np.abs(back[ok] - raw[ok]).max() < 0.01
    # ブレーキ・後退側(raw > 中立)のスロットル: 新しい規則ではゲインを掛けない → out がそのまま
    brake = ok & (raw[:, 1] > REF[1])
    if forward_only:
        assert np.allclose(out[brake, 1], raw[brake, 1], atol=1e-3)
    # 違う規則で割り戻すと、ゲイン≠1 のブレーキ側は合わない
    wrong = le.invert_gains(out, gains, trim, REF, not forward_only)
    m = brake & (np.abs(gains[:, 1] - 1) > 0.1)
    assert np.mean(np.abs(wrong[m, 1] - raw[m, 1]) < 0.5) < 0.05


def test_invert_gains_zero_gain_is_nan():
    out = np.array([[1500.0, 1400.0], [1500.0, 1600.0]])
    lab = le.invert_gains(out, np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), REF, True)
    assert np.isnan(lab[0, 0]) and np.isnan(lab[0, 1])
    assert lab[1, 0] == pytest.approx(1500.0)
    assert lab[1, 1] == pytest.approx(1600.0)          # ブレーキ側はゲインを掛けていないのでそのまま


@pytest.mark.parametrize('forward_only,expect', [(True, 'forward_only'), (False, 'both')])
def test_detect_gain_rule(forward_only, expect):
    raw, gains, trim = _random_frames(seed=2)
    out = le.apply_gains_vec(raw, gains, trim, REF, forward_only=forward_only)
    r = le.detect_gain_rule(raw, out, gains, trim, REF, np.ones(len(raw), bool))
    assert r['rule'] == expect and r['source'] == 'evidence' and r['n_checked'] > 50


def test_detect_gain_rule_without_evidence():
    raw, _, trim = _random_frames(seed=3)
    ones = np.ones((len(raw), 2))
    out = le.apply_gains_vec(raw, ones, trim, REF)
    r = le.detect_gain_rule(raw, out, ones, trim, REF, np.ones(len(raw), bool))
    assert r['source'] == 'irrelevant'                  # 速度のゲインがずっと 1 ならどちらでも同じ
    gains = np.column_stack([np.ones(len(raw)), np.full(len(raw), 1.5)])
    fwd = raw.copy()
    fwd[:, 1] = np.minimum(fwd[:, 1], REF[1] - 10)      # 前進側だけ: 規則の違いが出ない
    out = le.apply_gains_vec(fwd, gains, trim, REF)
    old = le.detect_gain_rule(fwd, out, gains, trim, REF, np.ones(len(raw), bool), le.datetime(2026, 10, 10, 17, 0))
    new = le.detect_gain_rule(fwd, out, gains, trim, REF, np.ones(len(raw), bool), le.datetime(2026, 10, 10, 20, 0))
    assert (old['rule'], old['source']) == ('both', 'time')
    assert (new['rule'], new['source']) == ('forward_only', 'time')


# ---------------------------------------------------------------- 交代の分類
def _vehicle(erpm_of_t, t1=10.0):
    vt = np.arange(0.0, t1, 0.01)
    return vt, np.array([erpm_of_t(t) for t in vt], float)


@pytest.mark.parametrize('case,coll,reboot,erpm,expect', [
    ('衝突の2秒後', [5.0], [], lambda t: 0.0, 'after_collision'),
    ('衝突は3秒より前・止まっている', [3.5], [], lambda t: 0.0, 'stuck'),
    ('止まっていない(遅い)', [], [], lambda t: -800.0, 'operational'),
    ('走行中に先回り', [], [], lambda t: -4500.0, 'preemptive'),
    ('1秒の途中まで動いていた', [], [], lambda t: -1000.0 if t < 6.5 else 0.0, 'operational'),
    ('基板の再起動の直後に止まっている', [], [6.6], lambda t: 0.0, 'operational'),
    ('再起動より衝突を優先', [6.0], [6.6], lambda t: 0.0, 'after_collision'),
])
def test_classify_takeover(case, coll, reboot, erpm, expect):
    vt, e = _vehicle(erpm)
    cls, info = le.classify_takeover(7.0, coll, vt, e, reboot)
    assert cls == expect, case
    if reboot and expect == 'operational':
        assert info['cause'] == 'reboot'
    if expect == 'after_collision':
        assert info['dt_since_collision_s'] == pytest.approx(7.0 - coll[0])


# ---------------------------------------------------------------- 衝突・CH3・再起動
def test_detect_collisions_merge_and_spike():
    t = np.arange(0, 10, 0.01)
    hg = np.full(len(t), 0.2)
    hg[100] = 5.0                       # 1サンプルだけ(前後 50ms に 1G を超えるものが無い)= ノイズ
    hg[300:303] = [1.5, 4.0, 2.0]       # 本物(前後にも大きな値)
    hg[340] = 3.0                       # 0.4 秒後: 同じ衝突にまとめる
    hg[341] = 1.2
    hg[500:502] = [2.6, 1.5]            # 2秒後: 別の衝突
    ev = le.detect_collisions(t, hg)
    assert [round(e['t'], 2) for e in ev] == [1.0, 3.01, 5.0]
    assert [e['suspect'] for e in ev] == [True, False, False]
    assert ev[1]['g'] == pytest.approx(4.0) and ev[1]['n'] == 2


def test_horizontal_and_dynamic_g():
    a = np.zeros((100, 3))
    a[:, 2] = G
    a[:, 0] = 0.3                       # 取り付けの傾き(止まっていても x に 0.3m/s^2)
    a[50] = [0.3 + 2.5 * G, 0.0, G]     # 水平 2.5G
    a[60] = [0.3, 0.0, G + 3 * G]       # 上下だけ 3G
    still = np.ones(100, bool)
    hg, bias = le.horizontal_g(a, still)
    dyn = le.dynamic_g(a, still)
    assert bias[0] == pytest.approx(0.3)
    assert hg[50] == pytest.approx(2.5) and hg[60] == pytest.approx(0.0) and hg[0] == pytest.approx(0.0)
    assert dyn[50] == pytest.approx(2.5) and dyn[60] == pytest.approx(3.0)


def test_clean_select():
    t = np.arange(0, 3, 0.01)
    sel = np.ones(len(t), bool)
    valid = np.ones(len(t), bool)
    sel[100:102] = False                 # 20ms だけ人側(電波の揺れ)
    sel[150:153] = False                 # 基板の再起動直後: CH3 が無効で 0
    valid[150:152] = False
    sel[200:] = False                    # 本物の交代
    s = le.clean_select(t, sel, valid)
    assert s[:200].all() and not s[200:].any()


def test_board_reboots():
    vt = np.array([0, 10, 20, 30, 40, 50], np.int64) * 1000
    bt = np.array([5e9, 5.01e9, 3e5, 3.1e5, 3.2e5, 1e5])
    assert le.board_reboots(vt, bt) == [(10000, 20000), (40000, 50000)]


# ---------------------------------------------------------------- 区間とリターン
def test_segments_and_returns():
    t = np.arange(10) / 30.0
    t[7:] += 0.5                         # 0.2秒を超える途切れ
    ai = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], bool)
    ok = np.ones(10, bool)
    ok[6] = False                        # マスク
    seg = le.make_segments(ai, ok, t)
    assert seg.tolist() == [0, 0, 0, 1, 1, 2, -1, 3, 3, 3]
    r = np.ones(10)
    ret = le.mc_returns(r, seg, 0.5)
    assert ret[:3].tolist() == [1.75, 1.5, 1.0]          # 区間の終わりで打ち切り
    assert ret[3:5].tolist() == [1.5, 1.0]
    assert ret[5] == 1.0 and np.isnan(ret[6])
    assert ret[7:].tolist() == [1.75, 1.5, 1.0]


def test_cumulative_distance():
    vt = np.arange(0, 2.0, 0.01)
    erpm = np.where(vt < 1.0, -4000.0, 2000.0)          # 1秒前進(2m/s)→ 1秒後退(1m/s)
    erpm[100:] = 2000.0
    d, df = le.cumulative_distance(vt, erpm)
    assert d[99] == pytest.approx(1.98, abs=0.02)
    assert d[-1] == pytest.approx(1.98 + (-1.0 * 0.99) - 1.5 * 0.01, abs=0.03)
    assert df[-1] == pytest.approx(1.99, abs=0.02)      # 前進だけ


def test_locate_rules():
    """起点 10s・緑マット出口 12.5s・次の起点 20s の1周(向きは単純な直線補間)"""
    t = np.arange(0, 30, 0.01)
    hd = np.interp(t, [10, 11, 12.5, 14, 18, 19, 20], [0, -140, 0, 10, 195, 200, 360])
    hd = np.where(t > 20, 360 + (t - 20) * 0, hd)
    anchors, gm = np.array([10.0, 20.0]), np.array([12.5, np.inf])
    names = [le.LOC_DETAIL[c] for c in le.locate([10.5, 11.5, 12.6, 13.5, 15.0, 18.5, 19.5], anchors, gm, t, hd)]
    assert names == ['R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
    # 前の起点が無い 5.0s は、次の起点(5秒後)までの向きの変化が 0° → LH(R1 の直前)
    # 最後の起点の後で緑マット出口が無い 25s は、決め手が無いので unknown
    assert [le.LOC_DETAIL[c] for c in le.locate([5.0, 25.0], anchors, gm, t, hd)] == ['LH', '?']


def test_parse_collision_map(tmp_path):
    p = tmp_path / 'cm.md'
    p.write_text('| ラン | t (s) | 地点 | |a| |\n|---|---|---|---|\n'
                 '| 006 | 28.97 | DE | 6.8 | −7.0 |\n| 007 | 100.17 | 回復 | 2.6 | x |\n'
                 '| 007 | 35.12 | **R1** | 4.1 | x |\n', encoding='utf-8')
    rows = le.parse_collision_map(p)
    assert [(r['log'][-6:], r['t'], r['recovery']) for r in rows] == [
        ('172910', 28.97, False), ('173445', 100.17, True), ('173445', 35.12, False)]
    assert le.map_loc('**R1**') == 'R1' and le.map_loc('R1(手前・軽)') == 'R1' and le.map_loc('GM出口') == 'GM'
    assert le.map_loc('SW') == 'other' and le.map_loc('回復') == 'unknown'


# ---------------------------------------------------------------- 小さな合成ログで通しで
def _write_log(root, log, old_rule=False, reboot=False):
    """10秒・2シーンの AI 走行。4.0s に衝突 → 5.0s に人へ交代 → 7.0s に AI へ戻す。
    1.0〜1.3s は hold、2.0〜2.5s はリカバリー(assist=2)。
    reboot: 衝突の直後に基板が再起動する(4.1〜4.6s は車両データなし、4.6s から board_t_us が戻る)"""
    d = root / log
    d.mkdir(parents=True)
    (d / 'log.json').write_text(json.dumps({'description': 'test', 'tags': ['ai-drive'], 'mode_id': 5, 'drive': {
        'run': 'x', 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1], 'params': {}}}), encoding='utf-8')
    t0 = 1_800_000_000_000_000
    ft = np.arange(0, 10, 1 / 30.0)
    vt = np.arange(0, 10, 0.01)
    it = np.arange(0, 10, 0.01) + 0.003
    sel_v = ~((vt >= 5.0) & (vt < 7.0))
    sel_f = ~((ft >= 5.0) & (ft < 7.0))
    rng = np.random.default_rng(0)
    raw = np.stack([1443 + 80 * np.sin(ft), 1498 + 120 * np.sin(ft * 1.7)], axis=1)
    gains = np.tile([1.3, 1.5], (len(ft), 1))
    out = le.apply_gains_vec(raw, gains, np.zeros(len(ft)), REF, forward_only=not old_rule)
    hold = (ft >= 1.0) & (ft < 1.3)
    assist = np.where((ft >= 2.0) & (ft < 2.5), 2, 0).astype(np.int8)
    out[hold] = REF
    raw[hold] = np.nan
    out[assist == 2] = [REF[0], REF[1] + 300]
    erpm = np.where((vt > 4.05) & (vt < 5.5), 0, -3000).astype(np.int32)
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    acc[:, :2] = rng.normal(0, 0.2, (len(it), 2))
    k = np.searchsorted(it, 4.0)
    acc[k:k + 3, 1] = [-3.0 * G, -6.0 * G, -2.0 * G]      # 正面から 6G
    gyro = np.zeros((len(it), 3))
    for si, (a, b) in enumerate([(0, 5.0), (5.0, 10.0)]):
        s = d / ('%s-%04d' % (log, si + 1))
        s.mkdir()
        fm, vm, im = (ft >= a) & (ft < b), (vt >= a) & (vt < b), (it >= a) & (it < b)
        us = lambda x: (t0 + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), pred=np.zeros((fm.sum(), 10, 2), np.float32),
                 raw=raw[fm].astype(np.float32), out=out[fm].astype(np.float32), gains=gains[fm].astype(np.float32),
                 trim=np.zeros(fm.sum(), np.float32), lookahead=np.full(fm.sum(), 3, np.int16),
                 mode_id=np.full(fm.sum(), 5, np.int16), hold=hold[fm], select=sel_f[fm],
                 latency_ms=np.zeros(fm.sum(), np.float32), infer_ms=np.zeros(fm.sum(), np.float32),
                 dry_run=np.zeros(fm.sum(), bool), assist=assist[fm], model_out=out[fm].astype(np.float32))
        bt = us(vt) - t0 + 10_000_000
        if reboot:
            vm &= ~((vt > 4.1) & (vt < 4.6))
            bt = np.where(vt >= 4.6, us(vt) - us(np.array([4.6]))[0] + 100_000, bt)
        pwm = np.tile([1440, 1500, 2000], (vm.sum(), 1)).astype(np.int32)
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=pwm, valid=np.ones((vm.sum(), 3), bool),
                 select=sel_v[vm], erpm=erpm[vm], board_t_us=bt[vm])
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32),
                 gyro=gyro[im].astype(np.float32))
        np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * fm.sum()))
    return raw


@pytest.mark.parametrize('old_rule', [False, True])
def test_label_log_end_to_end(tmp_path, old_rule):
    raw = _write_log(tmp_path, '20261010-120000' if old_rule else '20261010-210000', old_rule)
    log = '20261010-120000' if old_rule else '20261010-210000'
    info, a = le.label_log(log, tmp_path)
    ev, c = info['events'], info['counts']
    # 衝突 1件(AI側・止まった)、交代 1件(衝突の1秒後 = 失敗)、戻し 1件、リカバリー 1件、hold 1件
    assert c['collisions'] == 1 and ev['collisions'][0]['ai'] and ev['collisions'][0]['outcome'] == 'stop'
    assert ev['collisions'][0]['side'] == 'front'
    tk = ev['takeovers']
    assert len(tk) == 1 and tk[0]['class'] == 'after_collision' and tk[0]['failure']
    assert tk[0]['dt_since_collision_s'] == pytest.approx(1.0, abs=0.02)
    assert len(ev['handbacks']) == 1 and ev['handbacks'][0]['human_s'] == pytest.approx(2.0, abs=0.02)
    assert len(ev['recoveries']) == 1 and ev['recoveries'][0]['result'] == 'returned'
    assert len(ev['holds']) == 1
    assert info['gain_rule']['rule'] == ('both' if old_rule else 'forward_only')
    # 行動ラベル: assist=0・hold なしの AI フレームでは raw と一致
    ch = info['gain_rule']['check_ai']
    assert ch['n'] > 150 and ch['match_0.5us'] == [1.0, 1.0]
    va, ai, t = a['valid_action'], a['ai'], a['t_s']
    assert np.allclose(a['action_label'][va & ai], raw[va & ai], atol=0.01)
    assert np.allclose(a['action_label'][va & ~ai], [1440, 1500])          # 人はスティックの値
    # マスク: hold・リカバリー・交代と戻しの前後 0.3秒
    assert not va[(t >= 1.0) & (t < 1.3)].any() and not va[(t >= 2.0) & (t < 2.5)].any()
    assert not va[np.abs(t - 5.0) <= 0.29].any() and not va[np.abs(t - 7.0) <= 0.29].any()
    assert va[(t > 3.0) & (t < 3.9)].all()
    # 報酬: 衝突 −1、失敗による交代 −2(交代直前の AI のフレーム)、リカバリー開始 −0.5
    k_tk = tk[0]['frame']
    assert ai[k_tk] and not ai[k_tk + 1] and a['failure_takeover'][k_tk]
    assert a['reward'][k_tk] == pytest.approx(a['progress_fwd_m'][k_tk] - 2.0, abs=1e-4)
    kc = ev['collisions'][0]['frame']
    assert a['reward'][kc] == pytest.approx(a['progress_fwd_m'][kc] - 1.0, abs=1e-4)
    kr = ev['recoveries'][0]['frame']
    assert a['recovery_start'][kr] and a['reward'][kr] == pytest.approx(a['progress_fwd_m'][kr] - 0.5, abs=1e-4)
    # 進み: 3000 eRPM = 1.5 m/s
    assert a['progress_m'][(t > 0.5) & (t < 0.9)] == pytest.approx(1.5 / 30, rel=0.05)
    # 区間: hold で切れる。交代の区間の終わりは「失敗による交代」、人の区間の終わりは「戻し」、最後は「ログの終わり」
    seg = a['segment_id']
    assert (seg[(t >= 1.0) & (t < 1.3)] == -1).all()
    ends = {int(a['end_code'][i]) for i in np.nonzero(a['segment_end'])[0]}
    assert {1, 3, 4, 5} <= ends
    assert a['end_code'][k_tk] == 1 and a['segment_end'][k_tk] and a['time_to_end_s'][k_tk] == 0
    i = k_tk - 30
    assert a['ret'][i] == pytest.approx(sum(0.99 ** j * a['reward'][i + j] for j in range(31)), rel=1e-4)
    assert np.isnan(a['ret'][seg < 0]).all()


# ---------------------------------------------------------------- 区間の外に落ちた罰を移す
def test_carry_penalties_rules():
    """区間 0(0〜9)の後がマスク(10〜29)。直前の区間の最後のフレーム 9 へ移すのは、0.6秒以内の衝突と、
    同じ区間へ衝突を移した後の失敗による交代だけ。人をまたぐもの・遠いものは移さない"""
    t = np.arange(40) / 30.0
    seg = np.full(40, -1)
    seg[:10] = 0
    seg[30:] = 1
    ai = np.ones(40, bool)
    coll = np.zeros(40, bool)
    ftk = np.zeros(40, bool)
    coll[14] = True                        # 区間の終わり(9)から 5フレーム = 0.17秒
    ftk[28] = True                         # 0.63秒(> 0.6)だが、同じ区間へ衝突を移したので移す
    carried, fail_end, moves = le.carry_penalties(t, seg, ai, coll, ftk)
    assert carried[9] == pytest.approx(-3.0) and carried[14] == pytest.approx(1.0) and carried[28] == pytest.approx(2.0)
    assert carried.sum() == pytest.approx(0.0)
    assert fail_end[9] and fail_end.sum() == 1
    assert [(m['kind'], m['from_frame'], m['to_frame']) for m in moves] == [
        ('collision', 14, 9), ('failure_takeover', 28, 9)]
    # 衝突なしで 0.63秒離れた交代は移さない(再起動で止まってからの交代は AI の操作の結果とは言えない)
    c2, f2, m2 = le.carry_penalties(t, seg, ai, np.zeros(40, bool), ftk)
    assert not c2.any() and not f2.any() and not m2
    # 遠い衝突(0.67秒)も移さない
    far = np.zeros(40, bool)
    far[29] = True
    assert not le.carry_penalties(t, seg, ai, far, np.zeros(40, bool))[2]
    # 間に人のフレームがあれば移さない
    ai2 = ai.copy()
    ai2[12] = False
    assert not le.carry_penalties(t, seg, ai2, coll, ftk)[2]
    # 区間の中の罰はそのまま(移さない)
    inside = np.zeros(40, bool)
    inside[5] = True
    assert not le.carry_penalties(t, seg, ai, inside, inside)[2]


def test_label_log_carries_penalty_over_board_reboot(tmp_path):
    """4.0s の衝突の直後に基板が再起動 → 衝突(4.0s)と交代直前の AI のフレーム(4.97s)が再起動のマスク(3.6〜5.1s)に入る。
    罰は直前の区間の最後のフレーム(3.57s)へ移り、その区間は「失敗による交代」で終わる"""
    log = '20261010-210000'
    _write_log(tmp_path, log, reboot=True)
    info, a = le.label_log(log, tmp_path)
    ev, c = info['events'], info['counts']
    assert c['board_reboots'] == 1 and c['collisions'] == 1
    tk = ev['takeovers']
    assert len(tk) == 1 and tk[0]['failure'] and tk[0]['class'] == 'after_collision'
    seg, t = a['segment_id'], a['t_s']
    kc, kt = ev['collisions'][0]['frame'], tk[0]['frame']
    assert seg[kc] < 0 and seg[kt] < 0                      # どちらもマスクの中
    j = int(np.nonzero((seg >= 0) & (t < t[kc]))[0][-1])
    assert t[j] == pytest.approx(3.567, abs=0.02)
    assert a['reward'][j] == pytest.approx(a['progress_fwd_m'][j] - 3.0, abs=1e-4)
    assert a['penalty_carried'][j] == pytest.approx(-3.0)
    assert a['segment_end'][j] and a['end_code'][j] == 1
    assert a['reward'][kc] == pytest.approx(a['progress_fwd_m'][kc], abs=1e-4)     # 元のフレームからは外す
    assert c['collisions_on_masked_frames'] == 0 and c['failure_takeovers_on_masked_frames'] == 0
    assert c['penalties_carried'] == {'collision': 1, 'failure_takeover': 1}
    assert [m['kind'] for m in ev['carried_penalties']] == ['collision', 'failure_takeover']
    # 報酬の合計は変わらない(移しただけ)。リターンにも入る
    total = a['progress_fwd_m'].astype(np.float64).sum() - 1.0 - 2.0 - 0.5 * a['recovery_start'].sum()
    assert a['reward'].astype(np.float64).sum() == pytest.approx(total, abs=1e-3)
    assert a['ret'][j] == pytest.approx(a['reward'][j], abs=1e-5)
    i = j - 20
    assert a['ret'][i] == pytest.approx(sum(0.99 ** k * a['reward'][i + k] for k in range(21)), rel=1e-4)
