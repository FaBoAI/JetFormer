"""周の数え方の直し(偽の R1 の起点を外す、spec 5.1・6.3 T-1〜T-3)の試験。実機・GPU 不要(合成した配列と小さな合成の記録。
10/23・10/24 の記録があるときだけ、その数え直しも見る)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -m pytest -q tests/test_lap_anchor.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import label_events as le  # noqa: E402
import section_scorecard as ss  # noqa: E402

G = le.G
REF = (1441.0, 1496.0)
LAP_S = 9.0
T0_US = 1_800_000_000_000_000


# ---------------------------------------------------------------- 合成の記録(tests/test_car_check.py も使う)
# 1 周 9 秒の左回り: R1 右 −180°・GM 左 +180°・SW 左 +180°・LH 左 +180° = +360°((秒, ヨー deg/s) の並び)
LAP = [(0.9, -200.0), (0.1, 0.0), (1.5, 120.0), (1.5, 0.0), (2.0, 90.0), (1.5, 0.0), (1.5, 120.0)]
# 偽の起点の周: SW の出口で壁に当たって右へ回り(−100°)、左へ戻り(+100°)、遅くなって 3.1 秒よけいにかかる(周 13 秒)
FALSE_INSERT = [(0.4, -250.0), (0.5, 200.0), (3.1, 0.0)]


def course_profile(dur, first_anchor=2.0, false_laps=(1,)):
    """[(始め, 終わり, ヨー)]・本物の R1 の起点・偽の起点(右へ回り始めた時刻)。周の番号 0 = first_anchor から"""
    segs, anchors, false = [], [], []
    t, k = first_anchor - LAP_S, -1
    while t < dur:
        lap = LAP[:5] + [(0.5, 0.0)] + FALSE_INSERT + [(1.0, 0.0)] + LAP[6:] if k in false_laps else LAP
        if k >= 0:
            anchors.append(round(t, 6))
        tt = t
        for j, (dd, g) in enumerate(lap):
            if k in false_laps and j == 6:
                false.append(round(tt, 6))
            segs.append((tt, tt + dd, g))
            tt += dd
        t, k = tt, k + 1
    return segs, anchors, false


def write_lap_log(root, log, dur=44.0, first_anchor=2.0, false_laps=(1,), false_hit_g=6.0, human=(), stops=(),
                  left_gain=40.0, right_gain=40.0, s0_us=0.0, erpm=-4000, ceil_applied=(), cap_applied=(), vbat=None,
                  r1_bump_g=None, description='ai_no=1 test', params=None, dry_run=False, scene_s=20.0):
    """左回りの周回の記録(AI_CMD 30 fps・VEHICLE/IMU 100 Hz)。既定: R1 の起点 2・11・24・33・42 秒、
    偽の起点 17.5 秒(2 周目の SW の出口で壁に当たって右へ回る。0.02 秒後に false_hit_g の水平の衝突。None なら衝突なし)。
    human: [(t0, t1)] の間は人(CH3)。stops: [(t0, t1)] の間は止まる(eRPM 0)
    舵: out = 中心 + s0_us + κ × 1e4 ÷ gain(左は left_gain、右は right_gain)。κ = 0.1 秒後のヨー[rad/s] ÷ 速さ
    ceil_applied・cap_applied: [(t0, t1)] の間のフレームの記録の欄を True にする
    vbat: (始め, 終わり) [V] を VEHICLE に入れる(None なら欄なし = 古いファーム)
    r1_bump_g: 本物の R1 の起点の 0.3 秒後に、上下の衝撃(3 軸 |‖a‖/G − 1| がこの値)を入れる。
    返り値 (記録のフォルダ, 本物の起点, 偽の起点)"""
    d = Path(root) / log
    d.mkdir(parents=True)
    meta = {'description': description, 'tags': ['ai-drive'], 'mode_id': 5,
            'drive': {'run': 'x', 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1], 'dry_run': dry_run,
                      'params': params or {'throttle_gain': 0.9}}}
    (d / 'log.json').write_text(json.dumps(meta), encoding='utf-8')
    ft = np.arange(0, dur, 1 / 30.0)
    vt = np.arange(0, dur, 0.01)
    it = np.arange(0, dur, 0.01) + 0.003
    segs, anchors, false_at = course_profile(dur, first_anchor, false_laps)
    gz = np.zeros(len(it))
    for a, b, g in segs:
        gz[(it >= a) & (it < b)] = g
    e = np.full(len(vt), float(erpm))
    for a, b in stops:
        e[(vt >= a) & (vt < b)] = 0.0
        gz[(it >= a) & (it < b)] = 0.0
    sel_v = np.ones(len(vt), bool)
    sel_f = np.ones(len(ft), bool)
    for a, b in human:
        sel_v[(vt >= a) & (vt < b)] = False
        sel_f[(ft >= a) & (ft < b)] = False
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    for ta in false_at:
        if false_hit_g is not None:
            k = int(np.searchsorted(it, ta + 0.02))
            acc[k:k + 3, 1] = [-0.5 * false_hit_g * G, -false_hit_g * G, -0.3 * false_hit_g * G]
    if r1_bump_g is not None:
        for ta in anchors:
            k = int(np.searchsorted(it, ta + 0.3))
            if k + 1 < len(it):
                acc[k:k + 2, 2] = G * (1.0 + r1_bump_g)
    v = -np.interp(ft, vt, e) * le.KV
    kap = np.radians(np.interp(ft + 0.1, it, gz)) / np.maximum(v, 0.1)      # 舵はヨーより 0.1 秒先に出る
    steer_rel = np.where(kap > 0, kap * 1e4 / left_gain, kap * 1e4 / right_gain)
    steer = REF[0] + s0_us + np.clip(steer_rel, -480, 480)
    thr = np.full(len(ft), REF[1] - 200.0)
    out = np.stack([steer, thr], axis=1)
    raw = out.copy()
    ca = np.zeros(len(ft), bool)
    for a, b in ceil_applied:
        ca |= (ft >= a) & (ft < b)
    pa = np.zeros(len(ft), bool)
    for a, b in cap_applied:
        pa |= (ft >= a) & (ft < b)
    out[ca | pa, 1] = REF[1] - 120.0               # 天井・上限で弱めた
    nsc = int(np.ceil(dur / scene_s))
    us = lambda x: (T0_US + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    for si in range(nsc):
        a, b = si * scene_s, min((si + 1) * scene_s, dur)
        s = d / ('%s-%04d' % (log, si + 1))
        s.mkdir()
        fm, vm, im = (ft >= a) & (ft < b), (vt >= a) & (vt < b), (it >= a) & (it < b)
        n = int(fm.sum())
        extra = {}
        if ceil_applied:
            extra['ceil_applied'] = ca[fm]
        if cap_applied:
            extra['cap_applied'] = pa[fm]
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), raw=raw[fm].astype(np.float32), out=out[fm].astype(np.float32),
                 gains=np.ones((n, 2), np.float32), trim=np.zeros(n, np.float32), lookahead=np.full(n, 3, np.int16),
                 hold=np.zeros(n, bool), select=sel_f[fm], dry_run=np.full(n, dry_run), assist=np.zeros(n, np.int8),
                 model_out=out[fm].astype(np.float32), **extra)
        vx = {}
        if vbat is not None:
            vx['vbat_v'] = np.interp(vt[vm], [0, dur], list(vbat)).astype(np.float32)
            vx['vbat_ok'] = np.ones(int(vm.sum()), bool)
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=np.tile([1441, 1496, 2000], (int(vm.sum()), 1)).astype(np.int32),
                 valid=np.ones((int(vm.sum()), 3), bool), select=sel_v[vm], erpm=e[vm].astype(np.int32),
                 board_t_us=(us(vt[vm]) - T0_US + 10_000_000), **vx)
        gyro = np.zeros((int(im.sum()), 3))
        gyro[:, 2] = gz[im]
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32), gyro=gyro.astype(np.float32))
        np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * n))
    return d, anchors, false_at


# ---------------------------------------------------------------- T-1: 決まり(合成の配列)
def _track(anchor_dpsi, dur=60.0, t_s=None):
    """向き hd [deg] を、起点の時刻 → その時刻の向き の表で作る(線形)。車両データはずっと 2 m/s・AI"""
    it = np.arange(0, dur, 0.01) if t_s is None else t_s
    ts, hs = zip(*anchor_dpsi)
    hd = np.interp(it, ts, hs)
    vt = np.arange(0, dur, 0.01)
    erpm = np.full(len(vt), -4000.0)
    ft = np.arange(0, dur, 1 / 30.0)
    return it, hd, vt, erpm, ft, np.ones(len(ft), bool)


def test_heading_rule_drops_half_lap_and_keeps_next_real_anchor():
    """衝突で右へ回った起点(前の起点から 180°)は外れる。その次の本物の起点は、外した起点からは 180° だが、
    最後に使った起点からは 360° なので残る(すぐ前の起点から測ると本物まで外れてしまう)"""
    A = [10.0, 16.5, 19.0, 28.0]
    it, hd, vt, erpm, ft, sel = _track([(0, 0), (10, 0), (16.5, 180), (19, 360), (28, 720), (60, 720)])
    keep, rej = le.reject_false_anchors(A, it, hd, vt, erpm, ft, sel, [])
    assert keep.tolist() == [True, False, True, True]
    assert len(rej) == 1 and rej[0]['t_s'] == 16.5 and rej[0]['reason'] == 'heading'
    assert rej[0]['dpsi_deg'] == pytest.approx(180, abs=0.5) and rej[0]['prev_kept_s'] == 10.0
    assert rej[0]['heading_rule'] == 'used'
    # すぐ前の起点(16.5)から 19.0 は 180°: 外した起点を a_p にしていれば 19.0 も外れていた
    assert np.interp(19.0, it, hd) - np.interp(16.5, it, hd) < le.FALSE_ANCHOR_DPSI_DEG


def test_impact_is_evidence_only_by_default(monkeypatch):
    """(I) 起点の −0.05〜+0.25 秒に水平 4.0G 以上の衝突。既定では (G) の理由に足すだけで、それだけでは外さない
    (10/10 の本物の R1 の起点に、0.03 秒後・0.25 秒後の R1 の衝突がある)。FALSE_ANCHOR_IMPACT_ALONE で窓と値を確かめる"""
    A = [10.0, 19.0]
    it, hd, vt, erpm, ft, sel = _track([(0, 0), (10, 0), (19, 360), (60, 360)])
    keep, rej = le.reject_false_anchors(A, it, hd, vt, erpm, ft, sel, [{'t_s': 19.02, 'g': 8.0}])
    assert keep.all() and not rej                                  # ふつうの周 + 衝突 → 残す
    it2, hd2, *_ = _track([(0, 0), (10, 0), (16.5, 180), (60, 180)])
    keep, rej = le.reject_false_anchors([10.0, 16.5], it2, hd2, vt, erpm, ft, sel, [{'t_s': 16.52, 'g': 8.0}])
    assert keep.tolist() == [True, False] and rej[0]['reason'] == 'heading+impact' and rej[0]['g'] == 8.0
    monkeypatch.setattr(le, 'FALSE_ANCHOR_IMPACT_ALONE', True)
    cases = [(19.0 + 0.25, 4.0, False), (19.0 - 0.05, 6.0, False), (19.0 + 0.26, 8.0, True), (19.0 - 0.06, 8.0, True),
             (19.0 + 0.10, 3.99, True), (19.0 + 0.02, 10.6, False)]
    for tc, g, kept in cases:
        keep, rej = le.reject_false_anchors(A, it, hd, vt, erpm, ft, sel, [{'t_s': tc, 'g': g}])
        assert keep.tolist() == [True, kept], (tc, g)
        if not kept:
            assert rej[0]['reason'] == 'impact' and rej[0]['g'] == pytest.approx(g)
            assert rej[0]['hit_dt_s'] == pytest.approx(tc - 19.0, abs=1e-3)
    # 上下の衝撃(bump)は (I) に使わない。(t, g) の組でも渡せる
    keep, _ = le.reject_false_anchors(A, it, hd, vt, erpm, ft, sel, [{'t_s': 19.05, 'g': 9.0, 'kind': 'bump'}])
    assert keep.all()
    keep, rej = le.reject_false_anchors(A, it, hd, vt, erpm, ft, sel, [(19.05, 9.0)])
    assert keep.tolist() == [True, False]
    # 最初の起点(前の起点が無い)でも効き、その次の起点は「前の起点なし」で (G) を使わない
    keep, rej = le.reject_false_anchors(A, it, hd, vt, erpm, ft, sel, [(10.01, 8.0)])
    assert keep.tolist() == [False, True] and rej[0]['prev_kept_s'] is None


def test_heading_rule_not_used_when_long_human_stopped_or_gap():
    """16 秒を超えた・人が運転した・1 秒以上止まった・IMU が途切れた間は (G) を使わない(残す)"""
    base = [(0, 0), (10, 0), (16.5, 180), (60, 180)]
    it, hd, vt, erpm, ft, sel = _track(base)
    A = [10.0, 16.5]
    # 人
    s2 = sel.copy()
    s2[(ft >= 12.0) & (ft < 13.0)] = False
    keep, rej = le.reject_false_anchors(A, it, hd, vt, erpm, ft, s2, [])
    assert keep.all() and not rej
    # 止まった(1.0 秒)/ 0.9 秒なら止まったと見ない
    for dur_stop, kept in ((1.0, True), (0.9, False)):
        e2 = erpm.copy()
        e2[(vt >= 12.0) & (vt < 12.0 + dur_stop)] = 0.0
        keep, _ = le.reject_false_anchors(A, it, hd, vt, e2, ft, sel, [])
        assert keep[1] == kept, dur_stop
    # IMU の途切れ(0.3 秒)
    m = ~((it > 12.0) & (it < 12.3))
    keep, _ = le.reject_false_anchors(A, it[m], hd[m], vt, erpm, ft, sel, [])
    assert keep.all()
    # 16 秒を超える
    it, hd, vt, erpm, ft, sel = _track([(0, 0), (10, 0), (26.5, 180), (60, 180)])
    keep, _ = le.reject_false_anchors([10.0, 26.5], it, hd, vt, erpm, ft, sel, [])
    assert keep.all()
    # 衝突があっても (G) が使えなければ外さない(既定)。(I) だけで外す設定なら、使えなかった理由が行に残る
    keep, _ = le.reject_false_anchors([10.0, 26.5], it, hd, vt, erpm, ft, sel, [(26.5, 5.0)])
    assert keep.all()
    le_alone = le.FALSE_ANCHOR_IMPACT_ALONE
    try:
        le.FALSE_ANCHOR_IMPACT_ALONE = True
        _, rows = le.reject_false_anchors([10.0, 26.5], it, hd, vt, erpm, ft, sel, [(26.5, 5.0)])
        assert rows[0]['t_s'] == 26.5 and rows[0]['reason'] == 'impact' and rows[0]['heading_rule'] == 'long'
        _, rows = le.reject_false_anchors([10.0, 16.5], *_track(base)[:4], ft, s2, [(16.5, 5.0)])
        assert rows[0]['reason'] == 'impact' and rows[0]['heading_rule'] == 'not_ai'
    finally:
        le.FALSE_ANCHOR_IMPACT_ALONE = le_alone


def test_checked_anchors_keeps_gm_exit_aligned():
    """checked_anchors: 外した起点の緑マット出口も一緒に外れる(残りの値は外す前と同じ)"""
    it = np.arange(0, 44, 0.01) + 0.003
    segs, anchors, false = course_profile(44.0)
    assert anchors == [2.0, 11.0, 24.0, 33.0, 42.0] and false == [17.5]
    gz = np.zeros(len(it))
    for a, b, g in segs:
        gz[(it >= a) & (it < b)] = g
    hd, gzu = le.heading_deg(it, gz)
    vt = np.arange(0, 44, 0.01)
    erpm = np.full(len(vt), -4000.0)
    a0, gm0, _ = le.course_anchors(it, hd, gzu, vt, erpm)
    assert np.round(a0, 1).tolist() == [2.0, 11.0, 17.5, 24.0, 33.0, 42.0]
    assert np.isfinite(gm0[[0, 1, 3, 4]]).all()
    ft = np.arange(0, 44, 1 / 30.0)
    a1, gm1, _, rej = le.checked_anchors(it, hd, gzu, vt, erpm, ft, np.ones(len(ft), bool), [(17.52, 6.0)])
    assert np.round(a1, 1).tolist() == [2.0, 11.0, 24.0, 33.0, 42.0]
    assert gm1.tolist() == np.delete(gm0, 2).tolist()
    assert [r['reason'] for r in rej] == ['heading+impact']


# ---------------------------------------------------------------- T-1: 合成の記録で通しで(label_log・score_log)
def test_label_log_and_scorecard_drop_false_anchor(tmp_path):
    log = '20261103-100000'
    write_lap_log(tmp_path, log)
    info, a = le.label_log(log, tmp_path)
    assert info['anchor_rule'] == le.ANCHOR_RULE
    assert info['r1_anchors'] == 5
    rj = info['r1_anchors_rejected']
    assert len(rj) == 1 and rj[0]['t_s'] == pytest.approx(17.5, abs=0.05) and rj[0]['reason'] == 'heading+impact'
    assert rj[0]['dpsi_deg'] == pytest.approx(180, abs=5) and rj[0]['g'] == pytest.approx(6.0, abs=0.3)
    # 衝突の場所: 偽の起点のままなら R1(起点から 1 秒以内)。外すと SW の出口の向き(150〜215° = HS)
    c = info['events']['collisions']
    assert len(c) == 1 and c[0]['loc'] != 'R1' and c[0]['loc_detail'] in ('SW', 'HS')
    # 成績表(ラベルはその場で作ったもの)。偽の起点で切れた 6.5 秒の周は無く、2 周目は 13 秒の 1 周(衝突あり)
    sc = ss.score_log(log, tmp_path / 'no_labels', tmp_path)
    assert sc['labels'] == 'computed' and sc['r1_anchors'] == 5 and len(sc['passes']) == 5
    assert [round(x['lap_s'], 1) for x in sc['laps']] == [9.0, 13.0, 9.0, 9.0]
    assert [x['clean'] for x in sc['laps']] == [True, False, True, True]          # 2 周目に衝突
    assert all(330 <= x['dpsi_deg'] <= 390 for x in sc['laps'])
    assert [r['t_s'] for r in sc['r1_anchors_rejected']] == [r['t_s'] for r in rj]
    assert not sc['loc_relabeled']                                                 # 新しいラベルなので付け直さない


def test_scorecard_relabels_locations_of_old_label_files(tmp_path, monkeypatch):
    """外す前に作ったラベルのファイル(anchor_rule が無い)でも、成績表は外したあとの起点で場所を付け直す"""
    log = '20261103-100100'
    write_lap_log(tmp_path, log)
    keep_all = lambda A, *a, **k: (np.ones(len(A), bool), [])      # noqa: E731
    with monkeypatch.context() as m:
        m.setattr(le, 'reject_false_anchors', keep_all)
        info, arr = le.label_log(log, tmp_path)
    assert info['r1_anchors'] == 6 and info['events']['collisions'][0]['loc'] == 'R1'
    info.pop('anchor_rule')
    lab_dir = tmp_path / 'labels'
    lab_dir.mkdir()
    (lab_dir / (log + '.json')).write_text(json.dumps(info, default=le._json_default), encoding='utf-8')
    np.savez(lab_dir / (log + '.npz'), **arr)
    sc = ss.score_log(log, lab_dir, tmp_path)
    assert sc['labels'] == 'file' and sc['loc_relabeled']
    assert [round(x['lap_s'], 1) for x in sc['laps']] == [9.0, 13.0, 9.0, 9.0]
    assert sc['collision_events'][0]['loc'] != 'R1'
    assert not any(p['r1_collision'] for p in sc['passes'])
    # 外さなかったとき(古い道具)は 6.5 秒の偽の周があった
    with monkeypatch.context() as m:
        m.setattr(le, 'reject_false_anchors', keep_all)
        old = ss.score_log(log, lab_dir, tmp_path)
    assert [round(x['lap_s'], 1) for x in old['laps']] == [9.0, 6.5, 6.5, 9.0, 9.0]


def test_human_before_weak_false_anchor_keeps_next_real_anchor(tmp_path):
    """人が運転したあと(交代 → 戻し)に弱い衝突(3G)で偽の起点ができると、前の起点から (G) を使えず、(I) も効かない。
    その偽の起点を a_p にすると次の本物の起点(24 秒)が 180° で外れてしまう → 次の起点(33 秒)とつじつまが合う
    24 秒を残し、偽の 17.5 秒の方を外す(heading_next)"""
    log = '20261103-100200'
    write_lap_log(tmp_path, log, false_hit_g=3.0, human=[(13.0, 14.0)])
    info, _ = le.label_log(log, tmp_path)
    rj = info['r1_anchors_rejected']
    assert info['r1_anchors'] == 5
    assert [round(r['t_s'], 1) for r in rj] == [17.5] and rj[0]['reason'] == 'heading_next'
    assert rj[0]['heading_rule'] == 'not_ai' and rj[0]['next_s'] == pytest.approx(24.0, abs=0.05)
    sc = ss.score_log(log, tmp_path / 'no_labels', tmp_path)
    assert [round(x['lap_s'], 1) for x in sc['laps']] == [9.0, 9.0, 9.0]          # 2 周目は人がいたので周にしない
    log2 = '20261103-100300'
    write_lap_log(tmp_path, log2, false_hit_g=3.0)                  # 人なし → 向きだけで外れる
    info2, _ = le.label_log(log2, tmp_path)
    assert info2['r1_anchors'] == 5 and info2['r1_anchors_rejected'][0]['reason'] == 'heading'


def test_next_anchor_decides_only_when_previous_is_unverified():
    """a_p が確かめられていない(人のあと)ときだけ次の起点で決める。次の起点が無ければ (G) のとおり a_k を外す"""
    it, hd, vt, erpm, ft, sel = _track([(0, 0), (10, 0), (16.5, 180), (23, 360), (32, 720), (60, 720)])
    s2 = sel.copy()
    s2[(ft >= 12.0) & (ft < 13.0)] = False
    keep, rej = le.reject_false_anchors([10.0, 16.5, 23.0, 32.0], it, hd, vt, erpm, ft, s2, [])
    assert keep.tolist() == [True, False, True, True] and rej[0]['reason'] == 'heading_next'
    keep, rej = le.reject_false_anchors([10.0, 16.5, 23.0], it, hd, vt, erpm, ft, s2, [])
    assert keep.tolist() == [True, True, False] and rej[0]['reason'] == 'heading'
    # a_p が確かめられている(人なし)なら、次の起点を見ずに (G) のとおり
    keep, rej = le.reject_false_anchors([1.0, 10.0, 16.5, 23.0, 32.0], *_track(
        [(0, -360), (1, -360), (10, 0), (16.5, 180), (23, 360), (32, 720), (60, 720)])[:4], ft, sel, [])
    assert keep.tolist() == [True, True, False, True, True] and rej[0]['reason'] == 'heading'


# ---------------------------------------------------------------- T-3: 行動ラベルのマスク
def test_ceiling_and_cap_frames_are_masked(tmp_path):
    log = '20261103-100400'
    write_lap_log(tmp_path, log, false_laps=(), ceil_applied=[(5.0, 6.0)], cap_applied=[(8.0, 8.5)])
    info, a = le.label_log(log, tmp_path)
    t, va, bits = a['t_s'], a['valid_action'], a['mask_bits']
    assert le.MASK_BITS['cap'] == 256
    for lo, hi in ((5.0, 6.0), (8.0, 8.5)):
        m = (t >= lo) & (t < hi)
        assert m.sum() > 10 and not va[m].any() and ((bits[m] & 256) > 0).all()
    other = ~(((t >= 5.0) & (t < 6.0)) | ((t >= 8.0) & (t < 8.5)))
    assert va[other].all() and not ((bits[other] & 256) > 0).any()
    assert info['mask_counts_ai']['cap'] == int((~other).sum())
    # 割り戻しの検算(label_err)は、天井・上限で出力を変えたフレームを使わない → ずれ 0
    ch = info['gain_rule']['check_ai']
    assert ch['n'] == int(other.sum()) and ch['match_0.5us'] == [1.0, 1.0]
    # 欄の無い古い記録はマスクしない
    log2 = '20261103-100500'
    write_lap_log(tmp_path, log2, false_laps=())
    info2, a2 = le.label_log(log2, tmp_path)
    assert a2['valid_action'].all() and info2['mask_counts_ai']['cap'] == 0


# ---------------------------------------------------------------- T-2: 10/23・10/24 の記録(あるときだけ)
OLD_SC = {'20261023': Path('/home/jetson/jetformer_reports/bench_20261023_day1/nightly/scorecard/scorecard.json'),
          '20261024': Path('/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json')}
ANCHOR_CHECK = Path('/home/jetson/jetformer_reports/bench_20261024_report/anchor_check.json')
# spec 5.1 の 5 つ(anchor_check.json の 127 起点の中)
FALSE_5 = {('20261023-211020', 14.24), ('20261023-211118', 33.146), ('20261024-160654', 33.898),
           ('20261024-160847', 14.724), ('20261024-162305', 22.804)}
# anchor_check に入っていない 10/23 の公式でない走行(No.4・6・17・19・23)で外れるもの。どれも衝突で右へ回った半周
# (183〜209°。No.4・17 は LH の壁に当たった 0.07・0.31 秒後で (I) の窓の外 → 向きだけで外れる)
FALSE_OTHER = {('20261023-205309', 12.526), ('20261023-205458', 21.719), ('20261023-211828', 177.989),
               ('20261023-212242', 15.006), ('20261023-212934', 53.874)}
FALSE_LAPS = {('20261023-211020', 7.33): 6.91, ('20261023-211118', 27.255): 5.891, ('20261024-160654', 27.547): 6.351,
              ('20261024-160847', 8.53): 6.194, ('20261024-162305', 16.073): 6.731}
NEED = [p for p in OLD_SC.values()] + [ANCHOR_CHECK, Path('/home/jetson/jetformer/data/labels/20261024-162033.json')]


@pytest.fixture(scope='module')
def recount():
    if not all(p.exists() for p in NEED):
        pytest.skip('10/23・10/24 の記録・成績表が無い')
    out = {}
    for p in OLD_SC.values():
        old = json.loads(p.read_text(encoding='utf-8'))['per_log']
        for lg, o in old.items():
            if 'error' in o:
                continue
            out[lg] = (o, ss.score_log(lg))
    return out


def test_records_drop_exactly_the_known_false_anchors(recount):
    ac = json.loads(ANCHOR_CHECK.read_text(encoding='utf-8'))
    assert sum(len(v) for v in ac.values()) == 127
    got = {(lg, r['t_s']) for lg, (o, n) in recount.items() for r in n.get('r1_anchors_rejected') or []}
    assert {x for x in got if x[0] in ac} == FALSE_5
    assert got == FALSE_5 | FALSE_OTHER
    for lg, (o, n) in recount.items():
        for r in n.get('r1_anchors_rejected') or []:
            assert 170 <= r['dpsi_deg'] <= 210 and r['heading_rule'] == 'used'
            if (lg, r['t_s']) in FALSE_5:
                assert r['reason'] == 'heading+impact' and 0.0 <= r['hit_dt_s'] <= 0.05 and 5.4 <= r['g'] <= 10.7


def test_records_normal_laps_unchanged_and_false_laps_gone(recount):
    ac = json.loads(ANCHOR_CHECK.read_text(encoding='utf-8'))
    n_same = 0
    for lg, (o, n) in recount.items():
        if lg not in ac:
            continue
        old = {round(x['t_start'], 3): x['lap_s'] for x in o.get('laps', [])}
        new = {round(x['t_start'], 3): x['lap_s'] for x in n.get('laps', [])}
        for k, v in old.items():
            if (lg, k) in FALSE_LAPS:
                assert k not in new and v == pytest.approx(FALSE_LAPS[(lg, k)], abs=1e-3)
            else:
                assert new[k] == pytest.approx(v, abs=0.001)
                n_same += 1
        assert set(new) <= set(old)               # つながった周は無い(偽の起点の衝突のあと人へ交代している)
    assert n_same == 85
    # No.9 の 11 周はそのまま
    o9, n9 = recount['20261024-162033']
    assert [x['lap_s'] for x in n9['laps']] == [x['lap_s'] for x in o9['laps']] and len(n9['laps']) == 11


def test_records_no7_2433_window_is_gone(recount):
    def windows(laps):
        out = []
        for a, b, c in zip(laps, laps[1:], laps[2:]):
            if abs(a['t_start'] + a['lap_s'] - b['t_start']) <= 0.02 and abs(b['t_start'] + b['lap_s'] - c['t_start']) <= 0.02:
                out.append(round(a['lap_s'] + b['lap_s'] + c['lap_s'], 2))
        return out
    o7, n7 = recount['20261024-160654']
    assert 24.33 in windows(o7['laps']) and windows(n7['laps']) == []
    # 偽の起点の衝突は、R1 ではなく SW の出口の向き(HS)の場所になる
    locs = {round(c['t_s'], 2): c['loc_detail'] for c in n7['collision_events']}
    assert locs[33.91] == 'HS' and n7['loc_relabeled']


def test_records_1010_real_r1_anchors_with_hits_are_kept():
    """10/10 の 173445: 本物の R1 の起点で、0.03 秒後(106.17 秒、人が戻した直後)・0.25 秒後(270.86 秒、向き 347° の
    ふつうの周)に R1 の衝突(collision_map.md で目で確かめたもの)。どちらも外さない(衝撃だけでは外さない)"""
    if not Path('/home/jetson/jetformer/data/labels/20261010-173445.json').exists():
        pytest.skip('10/10 の記録が無い')
    s = ss.score_log('20261010-173445')
    assert s['r1_anchors_rejected'] == [] and s['r1_anchors'] == 22
    ta = [p['ta_s'] for p in s['passes']]
    assert any(abs(x - 106.17) < 0.01 for x in ta) and any(abs(x - 270.86) < 0.01 for x in ta)
    assert ss.score_log('20261010-172910')['r1_anchors_rejected'] == []


def test_records_label_log_agrees_with_scorecard(recount):
    for lg in ('20261023-211020', '20261024-160654', '20261024-162033'):
        info, _ = le.label_log(lg)
        n = recount[lg][1]
        assert [r['t_s'] for r in info['r1_anchors_rejected']] == [r['t_s'] for r in n['r1_anchors_rejected']]
        assert info['r1_anchors'] == n['r1_anchors']
