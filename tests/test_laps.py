"""周回の計測"""

import os

import numpy as np
import pytest

from jetformer import config
from jetformer.laps import LapCounter


def test_counts_laps_from_constant_turn():
    c = LapCounter()
    laps = []
    t = 0.0
    for _ in range(200):                     # 止まっている(バイアス 0.5deg/s を覚える)
        c.add(t, 0.5, 0)
        t += 0.01
    for _ in range(int(25 / 0.01)):          # 40deg/s(+バイアス)で左回り → 9秒で1周
        lap = c.add(t, 40.5, -4000, ai=True)
        if lap:
            laps.append(lap)
        t += 0.01
    assert len(laps) == 2
    assert laps[1]['time_s'] == pytest.approx(9.0, abs=0.05)
    assert laps[0]['first'] and not laps[1]['first'] and laps[1]['best']
    assert c.summary()['direction'] == '左回り' and c.summary()['best_ai_s'] == pytest.approx(9.0, abs=0.05)


def test_marks_laps_with_human_and_recovery():
    c = LapCounter()
    t, laps = 0.0, []
    for i in range(int(40 / 0.01)):
        ai = not (12 < t < 13)               # 2周目の途中で人が交代
        if abs(t - 22.0) < 0.005:
            c.note('recovery')               # 3周目でリカバリー
        lap = c.add(t, -45.0, -4000, ai=ai)  # 右回り 8秒/周
        if lap:
            laps.append(lap)
        t += 0.01
    assert c.direction == -1
    assert [x['ai_only'] for x in laps[:3]] == [True, False, True]
    assert laps[1]['takeovers'] == 1 and laps[2]['recoveries'] == 1
    s = c.summary()
    assert s['best_ai_s'] == pytest.approx(8.0, abs=0.05)     # 人・リカバリーの周は除く


@pytest.mark.skipif(not os.path.exists(config.RAW_DIR / '20261010-172910'), reason='実走のログが無い')
def test_replay_ai_run_006():
    """2026-10-10 ai_run_006: 約16周、AIだけの周は 8〜11.5秒"""
    root = config.RAW_DIR / '20261010-172910'
    scenes = sorted(s for s in os.listdir(root) if s.startswith('20261010-172910-'))
    imu = [np.load(root / s / 'IMU.npz') for s in scenes]
    veh = [np.load(root / s / 'VEHICLE.npz') for s in scenes]
    it = np.concatenate([a['timestamps_us'] for a in imu]) / 1e6
    gz = np.concatenate([a['gyro'][:, 2] for a in imu])
    vt = np.concatenate([v['timestamps_us'] for v in veh]) / 1e6
    er = np.concatenate([v['erpm'] for v in veh])
    sel = np.concatenate([v['select'] for v in veh])
    c = LapCounter()
    for t, g in zip(it, gz):
        k = max(np.searchsorted(vt, t, side='right') - 1, 0)
        c.add(t, float(g), float(er[k]), bool(sel[k]))
    s = c.summary()
    assert 15 <= s['laps'] <= 17 and s['direction'] == '左回り'
    assert 7.5 < s['best_ai_s'] < 9.0 and 8.5 < s['median_ai_s'] < 10.5
