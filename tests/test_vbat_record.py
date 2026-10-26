"""電池の電圧・速さの制御の記録(2026-10-25、speed_control_20261025/spec.md 2.6・3.9・4.2 節、6.2 節 V-3)の試験。
recorder の保存(_save_scene)を偽のシーンで回す(センサー・カメラに触れない)。torch を読まない

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_vbat_record.py
"""

import json
import os
import sys

import numpy as np
import pytest

from jetformer import clock, recorder as rec_mod
from jetformer.recorder import CEIL_FIELDS, SA_FIELDS, Recorder, _SceneBuffer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import label_events  # noqa: E402


class _Veh(object):
    vbat_scale = 1.0125


class _Hub(object):
    channels = ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE')

    def __getitem__(self, name):
        return _Veh()


def _veh(i, v=None, legacy=False):
    d = {'pwm': [1441, 1496, 1980], 'valid': [True, True, True], 'select': True, 'moving': False, 'hall_state': 1,
         'erpm': -100 * i, 'hall_count': i, 'hall_errors': 0, 'seq': i, 'board_t_us': 10000 * i,
         'quality': 'snapshot'}
    if not legacy:
        d.update(vbat_v=v if v is not None else float('nan'), vbat_ok=v is not None)
    return d


def _ai(i, new=True):
    row = {'pred': np.full((10, 2), 1450.0, np.float32), 'raw': (1450.0, 1400.0), 'out': (1450.0, 1380.0 + i),
           'gains': (1.0, 0.9), 'trim': 0.0, 'lookahead': 3, 'mode_id': 5, 'hold': False, 'select': True,
           'latency_ms': 28.0, 'infer_ms': 9.0, 'dry_run': False}
    if new:
        row.update(ceil_on=False, ceil_left=bool(i % 2), ceil_cap=4.0 if i % 2 else 4.5, ceil_v=3.9, ceil_vpred=4.05,
                   ceil_steer=120.0, ceil_gz=50.0, ceil_mode=1, ceil_in_us=1380.0 + i, ceil_thr_us=1390.0,
                   ceil_applied=False, sa_on=False, sa_m=1.031, sa_m_ff=1.031, sa_m_fb=1.0, sa_gain=0.928, sa_src=1,
                   sa_v=8.10, sa_eff=np.nan, sa_event=6 if i == 3 else 0, sa_lap_p90=4.1 if i == 3 else np.nan,
                   sa_lap_s=8.3 if i == 3 else np.nan, sa_applied=False)
        if i == 4:
            row['ceil_mode'] = None                                  # None も既定にする
    return row


def _scene(tmp_path, name, veh_rows, ai_rows):
    r = Recorder(_Hub())
    t0 = clock.now_ns()
    buf = _SceneBuffer(0, name, tmp_path / name.rsplit('-', 1)[0] / name, t0, t0 + int(20e9))
    for i in range(5):
        buf.cam.append((t0 + i * 33_000_000, i, 'CAM_FRONT/%d.jpg' % i))
    buf.veh = [(t0 + i * 10_000_000, d) for i, d in enumerate(veh_rows)]
    buf.ai = [(t0 + i * 33_000_000, d) for i, d in enumerate(ai_rows)]
    buf.imu = [(t0 + i * 10_000_000, np.array([0.0, 0.0, 9.8]), np.zeros(3), np.zeros(3), 30.0) for i in range(5)]
    log = {'token': 't', 'name': name.rsplit('-', 1)[0], 'description': 'ai_no=1 V_before=8.31 V_src=board memo=',
           'tags': [], 'mode_events': [(t0, 5)], 'drive': None}
    r._save_scene(buf, log, partial=True)
    return buf.path


def test_vehicle_npz_has_vbat_and_scene_stats(tmp_path):
    rows = [_veh(i, v) for i, v in enumerate([8.31, None, 8.29, 8.10, 8.30])]
    p = _scene(tmp_path, '20261103-100000-0001', rows, [_ai(i) for i in range(5)])
    veh = np.load(p / 'VEHICLE.npz')
    assert veh['vbat_v'].dtype == np.float32 and veh['vbat_ok'].dtype == bool
    assert np.isnan(veh['vbat_v'][1]) and not veh['vbat_ok'][1] and veh['vbat_ok'].sum() == 4
    assert veh['vbat_v'][0] == pytest.approx(8.31) and float(veh['vbat_scale']) == pytest.approx(1.0125)
    sc = json.loads((p / 'scene.json').read_text())
    assert sc['stats']['vbat'] == {'first': 8.31, 'last': 8.3, 'min': 8.1, 'median': 8.295, 'n': 4}
    json.dumps(sc, allow_nan=False)                                   # scene.json に NaN を書かない(/api/scenes が読む)


def test_old_firmware_and_old_rows_record_nan_and_none(tmp_path):
    rows = [_veh(i) for i in range(3)] + [_veh(3, legacy=True)]        # 電圧なし + 古い形の dict(欄が無い)
    p = _scene(tmp_path, '20261103-100100-0001', rows, [_ai(i, new=False) for i in range(3)])
    veh = np.load(p / 'VEHICLE.npz')
    assert np.isnan(veh['vbat_v']).all() and not veh['vbat_ok'].any()
    sc = json.loads((p / 'scene.json').read_text())
    assert sc['stats']['vbat'] is None
    ai = np.load(p / 'AI_CMD.npz')
    for k, t, d in CEIL_FIELDS + SA_FIELDS:                           # 古い行(欄が無い)は既定
        assert k in ai.files, k
        assert len(ai[k]) == 3
        if k == 'ceil_in_us':
            assert np.allclose(ai[k], ai['out'][:, 1])
        elif isinstance(d, float) and np.isnan(d):
            assert np.isnan(ai[k]).all(), k
        else:
            assert (ai[k] == d).all(), k
        assert ai[k].dtype == np.dtype(t), k


def test_ai_cmd_ceil_and_sa_fields(tmp_path):
    p = _scene(tmp_path, '20261103-100200-0001', [_veh(i, 8.3) for i in range(5)], [_ai(i) for i in range(5)])
    ai = np.load(p / 'AI_CMD.npz')
    names = [k for k, _, _ in CEIL_FIELDS + SA_FIELDS]
    assert names == ['ceil_on', 'ceil_left', 'ceil_cap', 'ceil_v', 'ceil_vpred', 'ceil_steer', 'ceil_gz', 'ceil_mode',
                     'ceil_in_us', 'ceil_thr_us', 'ceil_applied', 'sa_on', 'sa_m', 'sa_m_ff', 'sa_m_fb', 'sa_gain', 'sa_src',
                     'sa_v', 'sa_eff', 'sa_event', 'sa_lap_p90', 'sa_lap_s', 'sa_applied']     # 仕様 2.6・3.9 節
    assert ai['ceil_left'].tolist() == [False, True, False, True, False]
    assert ai['ceil_cap'].tolist() == [4.5, 4.0, 4.5, 4.0, 4.5]
    assert ai['ceil_mode'].tolist() == [1, 1, 1, 1, -1]                # None → 既定 -1
    assert ai['ceil_in_us'].tolist() == [1380.0, 1381.0, 1382.0, 1383.0, 1384.0]
    assert ai['sa_src'].dtype == np.int8 and (ai['sa_src'] == 1).all() and ai['sa_event'].tolist() == [0, 0, 0, 6, 0]
    assert ai['sa_lap_p90'][3] == pytest.approx(4.1) and np.isnan(ai['sa_lap_p90'][0])
    assert ai['sa_m'][0] == pytest.approx(1.031) and ai['ceil_on'].dtype == bool
    # 今までの欄は変わらない
    assert ai['out'].shape == (5, 2) and ai['gains'][0].tolist() == pytest.approx([1.0, 0.9]) and len(ai['cap_applied']) == 5


def test_old_reader_concat_keeps_working(tmp_path):
    """古い読み手(label_events.load_log → _concat)で、電圧のある新しいシーンと古いシーンがつながる"""
    log = '20261103-100300'
    root = tmp_path / log
    root.mkdir()
    (root / 'log.json').write_text(json.dumps({'description': 'x', 'logfile': log}))
    _scene(tmp_path, log + '-0001', [_veh(i, 8.3) for i in range(5)], [_ai(i) for i in range(5)])
    _scene(tmp_path, log + '-0002', [_veh(i, legacy=True) for i in range(5)], [_ai(i, new=False) for i in range(5)])
    d = label_events.load_log(log, tmp_path)
    assert not d['problems'] and len(d['VEHICLE']['timestamps_us']) == 10 and len(d['AI_CMD']['timestamps_us']) == 10
    assert 'vbat_v' in d['VEHICLE'] and 'ceil_mode' in d['AI_CMD']    # どちらのシーンにも入る(古いシーンは既定)
    assert np.isnan(d['VEHICLE']['vbat_v']).sum() == 5 and 'vbat_scale' not in d['VEHICLE']   # 0 次元はつながない


def test_vbat_stats_helper():
    assert rec_mod.vbat_stats([]) is None
    assert rec_mod.vbat_stats([{'vbat_v': float('nan'), 'vbat_ok': False}, {'erpm': 0}]) is None
    s = rec_mod.vbat_stats([{'vbat_v': 8.0, 'vbat_ok': True}, {'vbat_v': 7.6, 'vbat_ok': True}])
    assert s == {'first': 8.0, 'last': 7.6, 'min': 7.6, 'median': 7.8, 'n': 2}
