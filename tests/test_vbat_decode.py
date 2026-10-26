"""電池の電圧(2026-10-25、speed_control_20261025/spec.md 4 章・6.2 節 V-1・V-2)の解読の試験。
基板・I2C には触れない(偽物の 28 バイトを CRC を計算して作る。偽物のバス)。torch を読まない

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_vbat_decode.py
"""

import json
import math

import pytest

from jetformer.sensors import vehicle as vs


def snap(code=0, flags=0x07, seq=1, erpm=0, ver=vs.SNAP_VERSION, magic=vs.SNAP_MAGIC, bt=1000, crc_ok=True):
    """28 バイトのスナップショット(バイト 26 = code、flags は bit0-2 受信中が既定)"""
    d = bytearray(vs.SNAP.pack(magic, ver, seq & 0xFFFF, bt, 1441, 1496, 1980, flags, 3, erpm, 1234, 5, code, 0))
    d[-1] = vs.crc8(bytes(d[:-1])) ^ (0 if crc_ok else 0x55)
    return bytes(d)


# ---------------------------------------------------------------- V-1 解読

def test_layout_is_unchanged():
    assert vs.SNAP.size == vs.SNAP_LEN == 28 and vs.SNAP_VERSION == 1
    assert vs.FLAG_VBAT == 0x20 and vs.VBAT_LSB_V == 0.05
    assert vs.FLAG_VBAT not in (vs.FLAG_SELECT, vs.FLAG_MOVING, 1, 2, 4)        # 空きのビット(bit5)


def test_bit5_off_means_no_voltage_even_with_a_code():
    d = vs.decode_snapshot(snap(code=168, flags=0x07))
    assert math.isnan(d['vbat_v']) and d['vbat_ok'] is False


@pytest.mark.parametrize('code,volts', [(168, 8.40), (1, 0.05), (154, 7.70), (120, 6.00), (180, 9.00), (254, 12.70),
                                        (255, 12.75)])
def test_bit5_on_gives_code_times_lsb(code, volts):
    """値は code × 0.05 V。使うのは 6.0〜9.0 V(2S)の中だけ(fix.md: 0.05 V・12.75 V のような分圧の外れ・張り付きで
    走行のゲインを上げない)。外れた値は vbat_ok False・vbat_v NaN・vbat_bad True・vbat_raw_v に値"""
    d = vs.decode_snapshot(snap(code=code, flags=0x07 | vs.FLAG_VBAT))
    assert d['vbat_raw_v'] == pytest.approx(volts, abs=1e-9)
    if 6.0 <= volts <= 9.0:
        assert d['vbat_ok'] is True and d['vbat_v'] == pytest.approx(volts, abs=1e-9) and d['vbat_bad'] is False
    else:
        assert d['vbat_ok'] is False and math.isnan(d['vbat_v']) and d['vbat_bad'] is True


def test_range_constants_match_the_drive_side():
    from jetformer import speed_auto as sa
    p = sa.AutoParams()
    assert vs.VBAT_RANGE_V == (6.0, 9.0) == (p.vboard_lo, p.vboard_hi) == (p.vdesc_lo, p.vdesc_hi)
    assert vs.vbat_in_range(6.0) and vs.vbat_in_range(9.0) and not vs.vbat_in_range(5.95)
    assert not vs.vbat_in_range(float('nan')) and not vs.vbat_in_range(None)


def test_bit5_on_with_code_zero_is_no_voltage():
    d = vs.decode_snapshot(snap(code=0, flags=0x07 | vs.FLAG_VBAT))
    assert math.isnan(d['vbat_v']) and d['vbat_ok'] is False and d['vbat_bad'] is False and math.isnan(d['vbat_raw_v'])


def test_other_versions_and_bad_crc_still_raise():
    for bad in (snap(ver=2, code=168, flags=0x27), snap(magic=0x5A), snap(crc_ok=False, code=168, flags=0x27)):
        with pytest.raises(ValueError):
            vs.decode_snapshot(bad)
    with pytest.raises(ValueError):
        vs.decode_snapshot(snap()[:27])
    # バイト 26 は CRC に入る(電圧のバイトだけ壊れても見つかる)
    d = bytearray(snap(code=168, flags=0x27))
    d[26] = 167
    with pytest.raises(ValueError):
        vs.decode_snapshot(bytes(d))


def test_calibration_scale_is_applied():
    d = vs.decode_snapshot(snap(code=168, flags=0x27), vbat_scale=1.012)
    assert d['vbat_v'] == pytest.approx(8.40 * 1.012)
    d = vs.decode_snapshot(snap(code=168, flags=0x07), vbat_scale=1.012)
    assert math.isnan(d['vbat_v'])                                           # 無いときは係数に関係なく無し


def test_load_and_save_calibration(tmp_path):
    p = tmp_path / 'vbat_calib.json'
    assert vs.load_vbat_scale(p) == (1.0, '')                                # 無ければ 1.0
    doc = vs.save_vbat_scale(1.0125, meter_v=8.31, board_v=8.21, path=p)
    assert doc['scale'] == 1.0125 and vs.load_vbat_scale(p)[0] == pytest.approx(1.0125)
    for bad in ('{"scale": 2.0}', '{"scale": "x"}', '[1]', '{bad json'):
        p.write_text(bad)
        x, note = vs.load_vbat_scale(p)
        assert x == 1.0 and note                                             # 範囲の外・壊れている → 1.0 と理由
    with pytest.raises(ValueError):
        vs.save_vbat_scale(0.5, path=p)
    with pytest.raises(ValueError):
        vs.save_vbat_scale(float('nan'), path=p)


def test_level_thresholds():
    assert vs.vbat_level(8.40) == 'ok' and vs.vbat_level(7.90) == 'ok'
    assert vs.vbat_level(7.89) == 'warn' and vs.vbat_level(7.70) == 'warn'
    assert vs.vbat_level(7.69) == 'bad' and vs.vbat_level(None) is None and vs.vbat_level(float('nan')) is None


# ---------------------------------------------------------------- V-2 古いファーム(バイト 26 = 0・bit5 = 0)

OLD_KEYS = ('pwm', 'valid', 'select', 'moving', 'hall_state', 'erpm', 'hall_count', 'hall_errors', 'seq', 'board_t_us',
            'quality')


def test_old_firmware_frame_decodes_as_before():
    raw = snap(code=0, flags=0x1F, seq=77, erpm=-5123, bt=987654)        # 受信中・SELECT・回転中
    d = vs.decode_snapshot(raw)
    want = {'pwm': [1441, 1496, 1980], 'valid': [True, True, True], 'select': True, 'moving': True, 'hall_state': 3,
            'erpm': -5123, 'hall_count': 1234, 'hall_errors': 5, 'seq': 77, 'board_t_us': 987654, 'quality': 'snapshot'}
    assert {k: d[k] for k in OLD_KEYS} == want
    assert set(d) == set(OLD_KEYS) | {'vbat_v', 'vbat_ok', 'vbat_raw_v', 'vbat_bad'}   # 足したのは電圧の 4 つだけ
    assert math.isnan(d['vbat_v']) and d['vbat_ok'] is False and d['vbat_bad'] is False
    # 電圧のビットを立てても、ほかの値は同じ
    d2 = vs.decode_snapshot(snap(code=166, flags=0x1F | vs.FLAG_VBAT, seq=77, erpm=-5123, bt=987654))
    assert {k: d2[k] for k in OLD_KEYS} == want and d2['vbat_v'] == pytest.approx(8.30)


class _Bus(object):
    def __init__(self, frames):
        self.frames = iter(frames)

    def read_i2c_block_data(self, addr, reg, n):
        return list(next(self.frames))


def _sensor(tmp_path, scale=None):
    path = tmp_path / 'vbat_calib.json'
    if scale is not None:
        path.write_text(json.dumps({'scale': scale}))
    return vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot', 'vbat_calib_path': path})


def test_sensor_status_old_firmware_is_none_and_json_safe(tmp_path):
    s = _sensor(tmp_path)
    s.bus = _Bus([snap(seq=i) for i in range(1, 4)])
    samples = [x for _ in range(3) for x in s._poll_snapshot()]
    assert len(samples) == 3 and all(math.isnan(x.data['vbat_v']) for x in samples)
    s._publish(samples[-1])
    st = s.status()
    assert st['vbat_v'] is None and st['vbat_ok'] is False and st['vbat_age_s'] is None and st['vbat_supported'] is False
    json.dumps(st, allow_nan=False)                                          # 画面の JSON に NaN を出さない


def test_sensor_status_with_voltage_and_scale(tmp_path):
    s = _sensor(tmp_path, scale=1.01)
    s.bus = _Bus([snap(seq=1, code=168, flags=0x27), snap(seq=2, code=166, flags=0x27), snap(seq=3, code=0, flags=0x07)])
    out = [x for _ in range(3) for x in s._poll_snapshot()]
    assert out[0].data['vbat_v'] == pytest.approx(8.40 * 1.01) and out[1].data['vbat_v'] == pytest.approx(8.30 * 1.01)
    s._publish(out[-1])
    st = s.status()
    assert st['vbat_supported'] is True and st['vbat_v'] == pytest.approx(8.30 * 1.01, abs=1e-3)   # 最後に来た値
    assert st['vbat_ok'] is False and st['vbat_age_s'] is not None and st['vbat_scale'] == pytest.approx(1.01)
    json.dumps(st, allow_nan=False)
    s.set_vbat_scale(1.0)
    s.bus = _Bus([snap(seq=4, code=168, flags=0x27)])
    assert s._poll_snapshot()[0].data['vbat_v'] == pytest.approx(8.40)
    assert st['vbat_bad_v'] is None and st['vbat_range_v'] == [6.0, 9.0]


def test_sensor_status_broken_voltage_is_not_used(tmp_path):
    """0.5 V(分圧の外れ)・12.75 V(張り付き): 使わない(vbat_v は前のよい値のまま・無ければ None)。vbat_bad_v に出す"""
    s = _sensor(tmp_path)
    s.bus = _Bus([snap(seq=1, code=10, flags=0x27), snap(seq=2, code=255, flags=0x27)])
    out = [x for _ in range(2) for x in s._poll_snapshot()]
    assert all(x.data['vbat_ok'] is False and x.data['vbat_bad'] is True for x in out)
    s._publish(out[-1])
    st = s.status()
    assert st['vbat_v'] is None and st['vbat_supported'] is False and st['vbat_bad_v'] == pytest.approx(12.75)
    json.dumps(st, allow_nan=False)
    tr = vs.VbatTracker()
    for k, x in enumerate(out):
        tr.on_sample('VEHICLE', x)
    assert tr.seen is False and tr.bad_seen is True
    t_last = out[-1].t_ns / 1e9
    assert tr.bad_latest(t_last) == pytest.approx(12.75) and tr.bad_latest(t_last + 3.0) is None
    tr.add(t_last + 0.01, 0.5, 0)                       # 古い解読・偽物で範囲の外が来ても数えない
    assert tr.seen is False and tr.latest(t_last + 0.02)[0] is None


def test_legacy_protocol_has_no_voltage(tmp_path, monkeypatch):
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'legacy',
                          'vbat_calib_path': tmp_path / 'none.json'})

    class Msg(object):                                   # smbus2 の i2c_msg の偽物(バスに触れない)
        write = staticmethod(lambda addr, data: None)
        read = staticmethod(lambda addr, n: bytes(n))

    class LegacyBus(object):
        def i2c_rdwr(self, *msgs):
            pass
    monkeypatch.setattr(vs, 'i2c_msg', Msg)
    s.bus = LegacyBus()
    d = s._poll_legacy()[0].data
    assert math.isnan(d['vbat_v']) and d['vbat_ok'] is False and d['quality'] == 'legacy'


# ---------------------------------------------------------------- 止まっているときの中央値・1 秒ごとの最小

def _feed(tr, t0, t1, v, erpm=0, hz=100):
    n = int(round((t1 - t0) * hz))
    for i in range(n):
        tr.add(t0 + i / hz, v(t0 + i / hz) if callable(v) else v, erpm(t0 + i / hz) if callable(erpm) else erpm)


def test_tracker_still_median_rules():
    tr = vs.VbatTracker()
    _feed(tr, 0.0, 3.0, 8.31)
    assert tr.still_median(3.0, 2.0, 20) == pytest.approx(8.31)
    assert tr.seen
    v, why = tr.still_check(3.0 + 1.0, 2.0, 20)                              # 値が古い
    assert v is None and '来ていません' in why
    tr2 = vs.VbatTracker()
    _feed(tr2, 0.0, 3.0, 8.31, erpm=lambda t: 3000 if t > 1.5 else 0)       # 動いている
    v, why = tr2.still_check(3.0, 2.0, 20)
    assert v is None and '止まっていません' in why
    tr3 = vs.VbatTracker()
    _feed(tr3, 0.0, 3.0, float('nan'))                                       # 古いファーム
    v, why = tr3.still_check(3.0, 2.0, 20)
    assert v is None and '電圧がありません' in why and not tr3.seen
    tr4 = vs.VbatTracker()
    _feed(tr4, 0.0, 0.5, 8.3)                                                # 2 秒ぶん無い
    assert tr4.still_median(0.5, 2.0, 20) is None
    tr5 = vs.VbatTracker()
    _feed(tr5, 0.0, 3.0, lambda t: 8.3 if int(t * 100) % 20 == 0 else float('nan'))   # 使える値が 2 秒で 10 個だけ
    v, why = tr5.still_check(3.0, 2.0, 20)
    assert v is None and '足りません' in why


def test_tracker_bin_min_and_latest():
    tr = vs.VbatTracker()
    _feed(tr, 0.0, 10.0, lambda t: 8.3 - 0.05 * t + (-0.4 if 4.0 <= t < 4.3 else 0.0), erpm=-6000)
    assert tr.bin_min(0.0, 10.0) == pytest.approx(8.3 - 0.05 * 9.5, abs=0.01)     # 0.3 秒の下がりは 1 秒の中央値に出ない
    assert tr.bin_min(0.0, 3.0) == pytest.approx(8.3 - 0.05 * 2.5, abs=0.01)
    assert tr.bin_min(20.0, 30.0) is None
    v, age = tr.latest(10.0)
    assert v == pytest.approx(8.3 - 0.05 * 9.75, abs=0.02) and age < 0.1
    assert tr.latest(15.0) == (None, 5.0)
