"""反証 1 回目・safety(2026-10-26 fix2)。走行ループ(drive._loop)を偽物のセンサー・PCA9685・エンジンで回して、
モーターに出す値の組み合わせの場面を確かめる。装置は開かない(偽物は tests/test_launch_drive.py・test_drive_caps.py と同じ)。
直さない: 起きることを記録するだけ(結果は out/probe_safety1.json)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider \
        --basetemp experiments/20261026_fix/fix2/tmp_safety1/bt experiments/20261026_fix/fix2/safety1/test_probe_safety1.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests'))

from jetformer import assist as A          # noqa: E402
from jetformer import drive                # noqa: E402
from test_launch_drive import C, N, PARAMS, build, run, tof_near   # noqa: E402

OUT = Path(__file__).resolve().parent / 'out' / 'probe_safety1.json'
RESULTS = {}


def _save(key, val):
    RESULTS[key] = val
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old.update(RESULTS)
    OUT.write_text(json.dumps(old, ensure_ascii=False, indent=1))


def _rows(rec, frames):
    """記録の行 + PCA9685 に書いた値(書いた順)。中立止めの行は書いた値 = 中立"""
    s_of = {f['cam_t']: f['s'] for f in frames}
    return [(round(s_of[t], 3), r) for t, r in rec.rows]


# ---------------------------------------------------------------- 1. 中立止め 1 回でリカバリーが消える
def _stuck_scene(dur, hold_at=None, hold_len=0.2):
    """人側 1 秒 → AI 側。1.0〜2.0 秒は前へ 2 m/s(リカバリーが armed になる)、2.0 秒からは壁に当たって止まったまま
    (下がらない)。モデルはずっと弱い前進(中立 − 120)。hold_at があれば、その時刻から hold_len 秒は車両の値が来ない
    (0.15 秒を超えて古い = 中立止め。走行中の I2C の途切れ・処理の遅れ・カメラの途切れと同じ形の 1 回)"""
    return build(dur, sel=lambda s: s >= 1.0, v=lambda s: 2.0 if 1.0 <= s < 2.0 else 0.0, pred=(C, N - 120.0),
                 veh_skip=(lambda s: hold_at <= s < hold_at + hold_len) if hold_at is not None else False)


def _timeline(rows):
    """(s, assist, hold, 出したスロットル)"""
    return [(s, int(r['assist']), bool(r['hold']), round(float(r['out'][1]), 1)) for s, r in rows]


def test_1a_giveup_is_cleared_by_one_hold(monkeypatch, tmp_path):
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, launch_boost=0, tof_guard=1)
    frames, imu = _stuck_scene(24.0)
    _, pca0, rec0 = run(drive, monkeypatch, tmp_path, p, frames, imu)
    tl0 = _timeline(_rows(rec0, frames))
    g = [s for s, a, h, _ in tl0 if a == A.ASSIST_GIVEUP]
    assert g, 'リカバリー中止まで行かない場面'
    g0 = g[0]
    hold_at = g0 + 1.5
    frames1, imu1 = _stuck_scene(24.0, hold_at=hold_at)
    _, pca1, rec1 = run(drive, monkeypatch, tmp_path, p, frames1, imu1)
    tl1 = _timeline(_rows(rec1, frames1))
    after0 = [x for x in tl0 if x[0] >= hold_at + 0.4]
    after1 = [x for x in tl1 if x[0] >= hold_at + 0.4]
    # 中立止めなし: 中止のまま最後まで中立(人の交代を待つ)
    assert all(a == A.ASSIST_GIVEUP and t == N for _, a, h, t in after0)
    # 中立止め 1 回(0.2 秒)のあと: AI 側のまま、止まった車にモデルの前進が出続け、リカバリーも二度と出ない
    fwd = [x for x in after1 if not x[2] and x[3] < N]
    _save('1a_giveup_cleared_by_hold', {
        'giveup_first_s': g0, 'hold_at_s': round(hold_at, 3), 'hold_len_s': 0.2,
        'no_hold_after': collections.Counter((a, t) for _, a, h, t in after0).most_common(3),
        'with_hold_after': collections.Counter((a, t) for _, a, h, t in after1 if not h).most_common(3),
        'with_hold_forward_frames': len(fwd), 'with_hold_forward_s': round(fwd[-1][0] - fwd[0][0], 2) if fwd else 0,
        'with_hold_recovery_frames_after': sum(1 for x in after1 if x[1] in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP)),
        'pca_last_throttle_no_hold': pca0.pairs[-1][1], 'pca_last_throttle_with_hold': pca1.pairs[-1][1]})
    assert fwd and all(a == A.ASSIST_NONE and t == N - 120.0 for _, a, h, t in after1 if not h)


def test_1b_recovery_in_progress_is_cancelled_and_never_rearmed(monkeypatch, tmp_path):
    """リカバリーの途中(後退を出している間)に中立止め 1 回 → リカバリーが消え、armed も消える。車は止まったまま
    (|eRPM| < 1000)なので、リカバリーはもう始まらない。モデルの前進が人の交代まで出続ける"""
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, launch_boost=0, tof_guard=1)
    frames, imu = _stuck_scene(12.0)
    _, _, rec0 = run(drive, monkeypatch, tmp_path, p, frames, imu)
    tl0 = _timeline(_rows(rec0, frames))
    r0 = next(s for s, a, h, t in tl0 if a == A.ASSIST_RECOVERY)
    hold_at = r0 + 0.45                         # ブレーキ 0.25 秒 → 中立 0.15 秒 → 後退を出している間
    frames1, imu1 = _stuck_scene(12.0, hold_at=hold_at)
    _, _, rec1 = run(drive, monkeypatch, tmp_path, p, frames1, imu1)
    tl1 = _timeline(_rows(rec1, frames1))
    after1 = [x for x in tl1 if x[0] >= hold_at + 0.4]
    n_rec0 = sum(1 for x in tl0 if x[1] == A.ASSIST_RECOVERY)
    n_rec1_after = sum(1 for x in after1 if x[1] in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP))
    _save('1b_recovery_cancelled_by_hold', {
        'recovery_first_s': r0, 'hold_at_s': round(hold_at, 3),
        'no_hold_recovery_frames_total': n_rec0,
        'with_hold_recovery_frames_after_hold': n_rec1_after,
        'with_hold_after_out': collections.Counter((a, t) for _, a, h, t in after1 if not h).most_common(3),
        'with_hold_after_s': round(after1[-1][0] - after1[0][0], 2)})
    assert n_rec1_after == 0 and all(t == N - 120.0 for _, a, h, t in after1 if not h)


# ---------------------------------------------------------------- 2. 発進の前進が「前が近い」で出さないのに、発進の補助が前へ出す
def _near_scene(d_mm, floor=True):
    """人側 1 秒(止まっている)→ AI 側。車は止まったまま、前の真ん中 d_mm に立った物。モデルは中立(止まっている)"""
    return build(4.0, sel=lambda s: s >= 1.0, v=0.0, pred=(C, N), tof=tof_near(d_mm))


def test_2_launch_near_skip_but_start_kick_pushes_forward(monkeypatch, tmp_path):
    out = {}
    for d_mm in (350, 500, 900):
        frames, imu = _near_scene(d_mm)
        for sk in (0, 1):
            p = dict(PARAMS, launch_boost=1, start_kick=sk, tof_guard=1)
            d, pca, rec = run(drive, monkeypatch, tmp_path, p, frames, imu)
            rows = [(s, r) for s, r in _rows(rec, frames) if r['select'] and not r['hold']]
            skips = [e[1]['why'] for e in rec.events if e[0] == 'launch_boost']
            fwd = [(s, float(r['out'][1]), float(r['ttc'])) for s, r in rows if float(r['out'][1]) < N]
            out['d%d_kick%d' % (d_mm, sk)] = {
                'launch_skip': skips, 'ttc_first_ai': round(float(rows[0][1]['ttc']), 2),
                'forward_frames': len(fwd), 'forward_s': round(fwd[-1][0] - fwd[0][0] + 1 / 30, 2) if fwd else 0.0,
                'forward_thr': sorted(set(round(x[1], 1) for x in fwd)), 'assist': sorted(set(int(r['assist']) for _, r in rows))}
    # ToF ガードの床の基準が無い(ガードなし = 発進の前進は「ToF が無い」で出さない)
    frames, imu = build(4.0, sel=lambda s: s >= 1.0, v=0.0, pred=(C, N), tof=tof_near(350))
    d, pca, rec = run(drive, monkeypatch, tmp_path, dict(PARAMS, launch_boost=1, start_kick=1), frames, imu, floor=False)
    rows = [(s, r) for s, r in _rows(rec, frames) if r['select'] and not r['hold']]
    fwd = [x for x in rows if float(x[1]['out'][1]) < N]
    out['no_floor_kick1'] = {'launch_skip': [e[1]['why'] for e in rec.events if e[0] == 'launch_boost'],
                             'forward_frames': len(fwd), 'forward_thr': sorted(set(round(float(r['out'][1]), 1) for _, r in fwd))}
    _save('2_launch_skip_then_start_kick', out)
    assert out['d500_kick1']['launch_skip'] == ['near'] and out['d500_kick1']['forward_frames'] > 30
    assert out['d500_kick0']['forward_frames'] == 0
    assert out['no_floor_kick1']['launch_skip'] == ['tof'] and out['no_floor_kick1']['forward_frames'] > 30
