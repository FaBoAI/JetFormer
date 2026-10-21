"""走行タブの「AI の時間」(drive.py の stats['ai_s'])の試験(2026-10-21 Fix-all、V-protocol の major 2)。
実機・GPU・torch は使わない(tests/test_drive_caps.py の偽物と決まった時刻の列で _loop を回す)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_drive_ai_time.py

確かめること:
  - AI の時間 = CH3 が AI 側で dry run でないフレームから次のフレームまでの時間の和(0.2 秒を超える間は 1 フレーム分)。
    道具(tools/label_events の dt_s・ai と section_scorecard の ai_s)と同じ決め方で、別に数えた値と一致する
  - 人の操縦のフレーム・dry run は数えない。中立で止めたフレーム(センサーの途切れなど)でも CH3 が AI 側なら数える
  - 数えるだけで、車への出力と記録の欄は変わらない(区間の速度上限 OFF のまま変更前 30b71e8 とビット単位で同じ。
    tests/test_drive_caps.py の test_off_is_bit_identical_to_before もこの変更のあとに回す)
  - status() の stats に出る
"""
import numpy as np

from jetformer import drive

from test_drive_caps import PARAMS, assert_rows_equal_before, load_before, run_det, scenario

FPS = 30.0


def expected_ai_s(frames, dry_run=False):
    """別の数え方: 道具と同じ。各フレームの CH3 は、そのフレームまでに届いた最新の車両の値(届かなかったフレームは前の値)"""
    if dry_run:
        return 0.0
    t = np.array([f['cam_t'] for f in frames], np.float64) / 1e9
    sel, last = [], False
    for f in frames:
        if not f['veh_skip']:
            last = bool(f['veh']['select'])
        sel.append(last)
    dt = np.diff(t)
    dt = np.where((dt < 0) | (dt > drive.AI_GAP_S), 1 / FPS, dt)
    return float(np.sum(dt[np.asarray(sel[:-1])]))


def test_ai_time_matches_tool_rule(monkeypatch, tmp_path):
    frames, imu = scenario()
    d, _, rec = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    want = expected_ai_s(frames)
    got = d.st['stats']['ai_s']
    assert want > 10.0
    assert abs(got - want) <= 0.06, (got, want)                 # 画面は 0.1 秒に丸める
    # 人の操縦の 15 フレームは入らない(全部を足した時間より約 0.5 秒短い)
    t = np.array([f['cam_t'] for f in frames]) / 1e9
    assert got < (t[-1] - t[0]) - 0.4
    assert d.status()['stats']['ai_s'] == got


def test_ai_time_zero_in_dry_run(monkeypatch, tmp_path):
    frames, imu = scenario()
    d, _, _ = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames[:150], imu, dry_run=True)
    assert d.st['stats']['frames'] > 100 and d.st['stats']['ai_s'] == 0.0


def test_ai_time_gap_counts_one_frame(monkeypatch, tmp_path):
    """カメラが 1 秒止まった間は 1 フレーム分(道具と同じ)。止まった時間を AI の時間に入れない"""
    frames, imu = scenario()
    cut = frames[:100] + frames[130:200]                       # 100 → 130 で 1 秒空く(CH3 はずっと AI 側)
    d, _, _ = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), cut, imu)
    want = expected_ai_s(cut)
    assert abs(d.st['stats']['ai_s'] - want) <= 0.06
    assert abs(want - (len(cut) - 1) / FPS) < 1e-6


def test_ai_time_human_only_is_zero(monkeypatch, tmp_path):
    frames, imu = scenario()
    for f in frames:
        f['veh'] = dict(f['veh'], select=False, pwm=[1441, 1496, 1000])
    d, _, _ = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames[:120], imu)
    assert d.st['stats']['ai_s'] == 0.0


def test_counting_does_not_change_output(monkeypatch, tmp_path):
    """数えるだけ: OFF の出力と記録の欄は変更前(30b71e8)と同じ(人・AI・途切れ・中立が混ざった列)"""
    before = load_before()
    frames, imu = scenario()
    _, pca0, rec0 = run_det(before, monkeypatch, tmp_path, PARAMS, frames, imu)
    d1, pca1, rec1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    assert pca0.pairs == pca1.pairs and pca0.single == pca1.single
    assert_rows_equal_before(rec0.rows, rec1.rows)
    assert d1.st['stats']['ai_s'] > 10.0
