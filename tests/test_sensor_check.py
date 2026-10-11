"""センサーの向きの点検(ロードマップ S0)"""

import numpy as np

from jetformer import sensor_check as sc


def _image(bumper_rows):
    img = np.full((224, 224, 3), 90, np.uint8)                 # 灰色の床
    if bumper_rows:
        img[224 - bumper_rows:, :] = (200, 120, 20)              # BGR の青
    return img


def _tof(row9_mm, row18_mm):
    d = np.full((42, 54), 1500, np.uint16)
    d[9, :] = row9_mm
    d[18, :] = row18_mm
    return (d | (4 << 13)).astype(np.uint16)


def _ref(row9_mm, row18_mm):
    """基準は距離(mm)だけ(data/tof_check_ref.npy と同じ形)"""
    d = np.full((42, 54), 1500, np.uint16)
    d[9, :] = row9_mm
    d[18, :] = row18_mm
    return d


def test_bumper_rows_counts_blue_rows_from_bottom():
    assert sc.bumper_rows(_image(0)) == 0
    assert sc.bumper_rows(_image(7)) == 7
    assert sc.bumper_rows(_image(10)) == 10


def test_evaluate_ok_and_tilted():
    ref = _ref(2000, 850)
    ok = sc.evaluate([_image(7)] * 5, [_tof(2010, 852)] * 5, [0.0], ref)
    assert ok['ok'] and ok['bumper_ok'] and ok['tof_ok'] and abs(ok['tof_floor_pct']['9'] - 0.5) < 0.2
    tilted = sc.evaluate([_image(10)] * 5, [_tof(1700, 800)] * 5, [0.0], ref)   # 下を向いた: 遠い床が15%近い
    assert not tilted['ok'] and tilted['tof_ok'] is False and not tilted['bumper_ok']
    assert any('下' in n for n in tilted['notes'])


def test_evaluate_warns_when_moving_or_no_reference():
    moving = sc.evaluate([_image(7)], [_tof(2000, 850)], [-3000.0, -3100.0], _ref(2000, 850))
    assert moving['moving'] and not moving['ok']
    noref = sc.evaluate([_image(7)], [_tof(2000, 850)], [0.0], None)
    assert noref['tof_floor_pct'] is None and noref['ok']
