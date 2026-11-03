"""tools/as5601_logger.py(AS5601 を読むだけの記録係。2026-10-29 ユーザーの決め)の試験。I2C・装置には触れない(偽物の bus と時計)

- 読むのは 0x36 だけ・読みだけ(偽物の bus には書く関数が無い = 書こうとすると落ちる)
- 1 秒に hz 回・STATUS などは health_s ごと・2 秒ごとに fsync・失敗は数えて続ける(5 回続いたら 0.2 秒休む)
- 欄・単位(epoch_us は UNIX µs・deg = raw × 360 / 4096)

    cd ~/jetformer && python -m pytest -q tests/test_as5601_logger.py
"""
import csv
import importlib.util
import json
import os

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_spec = importlib.util.spec_from_file_location('as5601_logger', os.path.join(ROOT, 'tools', 'as5601_logger.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


class Clock(object):
    def __init__(self, t=1761740000.0):
        self.t = t
        self.m = 100.0
        self.sleeps = []

    def time(self):
        return self.t

    def monotonic(self):
        return self.m

    def sleep(self, dt):
        self.sleeps.append(dt)
        self.t += dt
        self.m += dt


class Bus(object):
    """読むだけの偽物(write_* は持たない)。fail_angle = 失敗させる角度の読みの番号の集まり"""

    def __init__(self, raw=978, status=0x33, agc=128, mag=350, fail_angle=()):
        self.raw, self.status, self.agc, self.mag = raw, status, agc, mag
        self.fail_angle = set(fail_angle)
        self.calls = []
        self.n_angle = 0

    def read_byte_data(self, addr, reg):
        self.calls.append(('byte', addr, reg))
        assert addr == 0x36
        return {0x0B: self.status, 0x1A: self.agc}[reg]

    def read_i2c_block_data(self, addr, reg, n):
        self.calls.append(('block', addr, reg, n))
        assert addr == 0x36 and n == 2
        if reg == 0x0C:
            i = self.n_angle
            self.n_angle += 1
            if i in self.fail_angle:
                raise OSError(121, 'Remote I/O error')
            v = (self.raw + i) % 4096
        elif reg == 0x1B:
            v = self.mag
        else:
            raise AssertionError(reg)
        return [(v >> 8) & 0x0F, v & 0xFF]


def _rows(path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def test_bits_and_values():
    assert L.raw12([0x03, 0xD2]) == 978 and L.raw12([0xF3, 0xD2]) == 978      # 上の 4 ビットは捨てる
    assert L.status_bits(0x33) == (1, 1, 0) and L.status_bits(0x28) == (1, 0, 1) and L.status_bits(0) == (0, 0, 0)
    assert L.ADDR == 0x36


def test_run_rate_fields_and_meta(tmp_path, monkeypatch):
    syncs = []
    monkeypatch.setattr(L.os, 'fsync', lambda fd: syncs.append(fd))
    clk, bus = Clock(), Bus()
    s = L.run(L.Reader(bus), str(tmp_path), hz=50, duration=3.0, fsync_s=2.0, health_s=1.0, clock=clk, printer=lambda *a: None)
    rows = _rows(s['path'])
    assert list(rows[0].keys()) == L.FIELDS
    assert 148 <= len(rows) == s['samples'] <= 152                              # 3 秒 × 50 回
    assert s['errors'] == 0 and s['raw_min'] == 978 and s['raw_max'] == 978 + len(rows) - 1
    assert float(rows[0]['deg']) == pytest.approx(978 * 360 / 4096, abs=1e-3)
    assert int(rows[0]['epoch_us']) == int(1761740000.0 * 1e6)
    health = [r for r in rows if r['status'] != '']
    assert len(health) == 3 and health[0]['md'] == '1' and health[0]['ml'] == '1' and health[0]['agc'] == '128'
    assert all(c[1] == 0x36 for c in bus.calls)                                 # 0x36 だけ
    assert {c[0] for c in bus.calls} == {'byte', 'block'}                      # 読みだけ(偽物に書く関数は無い)
    n_angle = sum(1 for c in bus.calls if c[0] == 'block' and c[2] == 0x0C)
    assert n_angle == len(rows)                                                 # 角度は 1 行に 1 回だけ読む
    meta = json.loads(open(os.path.join(str(tmp_path), os.path.basename(s['path'])[:-4] + '.json'), encoding='utf-8').read())
    assert meta['addr'] == '0x36' and meta['hz'] == 50 and meta['fields'] == L.FIELDS
    assert len(syncs) >= 3                                                      # 始めの json・2 秒ごと・終わり


def test_errors_are_counted_and_do_not_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(L.os, 'fsync', lambda fd: None)
    clk, bus = Clock(), Bus(fail_angle=range(10, 17))
    msgs = []
    s = L.run(L.Reader(bus), str(tmp_path), hz=50, duration=2.0, clock=clk, printer=msgs.append)
    rows = _rows(s['path'])
    bad = [r for r in rows if r['raw'] == '-1']
    assert len(bad) == 7 == s['errors'] and [r['err'] for r in bad] == [str(i) for i in range(1, 8)]
    assert rows[[r['raw'] for r in rows].index('-1') + 7]['err'] == '0'         # 戻ったら数え直す
    assert any(dt == pytest.approx(0.2) for dt in clk.sleeps)                    # 5 回続いたら 0.2 秒休む
    assert len(msgs) == 1 and '失敗' in msgs[0]                                  # 知らせは 5 秒に 1 回まで


def test_stop_and_show(tmp_path, monkeypatch):
    monkeypatch.setattr(L.os, 'fsync', lambda fd: None)
    clk, bus = Clock(), Bus()
    msgs = []
    s = L.run(L.Reader(bus), str(tmp_path), hz=50, stop=lambda: clk.m >= 101.0, show=True, clock=clk, printer=msgs.append)
    assert 48 <= s['samples'] <= 52 and len(msgs) >= 2 and msgs[0].startswith('角度 ')


def test_cli_has_no_address_option():
    src = open(os.path.join(ROOT, 'tools', 'as5601_logger.py'), encoding='utf-8').read()
    assert "'--addr'" not in src and 'write_' not in src                       # 0x36 のほかは読まない・書かない


# ---------------------------------------------------------------- 2026-10-30: 起動のサービス用(待つ・区切る・同じ秒で上書きしない)
class DeadBus(Bus):
    """最初の n 回は応答しない(車の電池をつなぐ前など)"""

    def __init__(self, dead, **kw):
        super().__init__(**kw)
        self.dead = dead

    def read_byte_data(self, addr, reg):
        if self.dead > 0:
            self.dead -= 1
            raise OSError(121, 'Remote I/O error')
        return super().read_byte_data(addr, reg)


def test_wait_ready_waits_then_returns_health():
    clk, msgs = Clock(), []
    h = L.wait_ready(L.Reader(DeadBus(3)), True, lambda: False, clock=clk, printer=msgs.append, every_s=10.0)
    assert h == (0x33, 128, 350) and len(msgs) == 1 and '待ちます' in msgs[0]
    assert clk.m - 100.0 == pytest.approx(30.0, abs=0.5)                       # 10 秒ごとに 3 回待った


def test_wait_ready_without_wait_or_with_stop_returns_none():
    msgs = []
    assert L.wait_ready(L.Reader(DeadBus(1)), False, lambda: False, clock=Clock(), printer=msgs.append) is None
    assert '応答しません' in msgs[0] and '待ちます' not in msgs[0]
    clk = Clock()
    assert L.wait_ready(L.Reader(DeadBus(99)), True, lambda: clk.m > 105.0, clock=clk, printer=lambda *a: None) is None


def test_same_second_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(L.os, 'fsync', lambda fd: None)
    clk = Clock()
    a = L.run(L.Reader(Bus()), str(tmp_path), hz=50, duration=0.0, clock=clk, printer=lambda *x: None)
    b = L.run(L.Reader(Bus()), str(tmp_path), hz=50, duration=0.0, clock=clk, printer=lambda *x: None)
    assert a['path'] != b['path'] and b['path'].endswith('-2.csv')
    assert os.path.exists(b['path'][:-4] + '.json')


def test_cli_options_for_the_service():
    src = open(os.path.join(ROOT, 'tools', 'as5601_logger.py'), encoding='utf-8').read()
    for opt in ("'--fsync-s'", "'--rotate-s'", "'--wait'"):
        assert opt in src


# ---------------------------------------------------------------- 2026-10-30 反証 S3: main() をそのまま動かす(偽物の bus・時計)
import errno  # noqa: E402
import signal as _signal  # noqa: E402


class FakeSMBus(object):
    """smbus2.SMBus の代わり(読む 2 つの関数と with だけ。書く関数は持たない)"""
    made = []

    def __init__(self, n, inner):
        self.n, self.inner = n, inner
        FakeSMBus.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read_byte_data(self, addr, reg):
        assert (addr, reg) in {(0x36, 0x0B), (0x36, 0x1A)}
        return self.inner.read_byte_data(addr, reg)

    def read_i2c_block_data(self, addr, reg, n):
        assert (addr, reg, n) in {(0x36, 0x0C, 2), (0x36, 0x1B, 2)}
        return self.inner.read_i2c_block_data(addr, reg, n)


def _main(monkeypatch, tmp_path, args, inner, clock=None):
    handlers = {}
    monkeypatch.setattr(L.signal, 'signal', lambda sig, fn: handlers.__setitem__(sig, fn))   # 本物のシグナルは入れない
    monkeypatch.setattr(L.os, 'fsync', lambda fd: None)
    monkeypatch.setattr(L.os, 'nice', lambda inc: 10)
    msgs = []
    clk = clock or Clock()
    rc = L.main(['--out', str(tmp_path)] + args, clock=clk, bus_factory=lambda n: FakeSMBus(n, inner), printer=msgs.append)
    return rc, msgs, handlers, clk


def test_main_rotates_and_stops_at_duration(monkeypatch, tmp_path):
    rc, msgs, handlers, _ = _main(monkeypatch, tmp_path, ['--duration', '25', '--rotate-s', '10', '--fsync-s', '1'], Bus())
    files = sorted(p for p in os.listdir(str(tmp_path)) if p.endswith('.csv'))
    assert rc == 0 and len(files) == 3                                          # 10 + 10 + 5 秒
    assert sum(1 for m in msgs if m.startswith('区切りました')) == 2 and msgs[-1].startswith('止めました')
    assert set(handlers) == {_signal.SIGINT, _signal.SIGTERM}
    rows = [r for f in files for r in _rows(os.path.join(str(tmp_path), f))]
    assert 1240 <= len(rows) <= 1260                                            # 25 秒 × 50 回
    meta = json.load(open(os.path.join(str(tmp_path), files[0][:-4] + '.json'), encoding='utf-8'))
    assert 'boot_id' in meta and isinstance(meta['rt_minus_mono_us'], int)
    assert open(os.path.join(str(tmp_path), files[0]), 'rb').read().count(b'\r') == 0      # 行末は \n だけ


def test_main_waits_for_the_sensor_then_records(monkeypatch, tmp_path):
    rc, msgs, _, clk = _main(monkeypatch, tmp_path, ['--duration', '5', '--wait'], DeadBus(2))
    assert rc == 0 and any('待ちます' in m for m in msgs) and any(m.startswith('止めました') for m in msgs)
    assert len([p for p in os.listdir(str(tmp_path)) if p.endswith('.csv')]) == 1


def test_main_without_wait_returns_2(monkeypatch, tmp_path):
    rc, msgs, _, _ = _main(monkeypatch, tmp_path, ['--duration', '5'], DeadBus(1))
    assert rc == 2 and '応答しません' in msgs[0] and not os.listdir(str(tmp_path))


def test_main_sigterm_ends_with_a_complete_file(monkeypatch, tmp_path):
    class StopBus(Bus):
        def read_i2c_block_data(self, addr, reg, n):
            if reg == 0x0C and self.n_angle == 120:
                hs[_signal.SIGTERM](_signal.SIGTERM, None)                        # 120 行目で systemd の止め
            return super().read_i2c_block_data(addr, reg, n)
    hs = {}
    monkeypatch.setattr(L.signal, 'signal', lambda sig, fn: hs.__setitem__(sig, fn))
    monkeypatch.setattr(L.os, 'fsync', lambda fd: None)
    monkeypatch.setattr(L.os, 'nice', lambda inc: 10)
    rc = L.main(['--out', str(tmp_path), '--rotate-s', '1800'], clock=Clock(),
                bus_factory=lambda n: FakeSMBus(n, StopBus()), printer=lambda m: None)
    f = [p for p in os.listdir(str(tmp_path)) if p.endswith('.csv')]
    assert rc == 0 and len(f) == 1
    data = open(os.path.join(str(tmp_path), f[0]), encoding='utf-8').read()
    assert data.endswith('\n') and 119 <= len(_rows(os.path.join(str(tmp_path), f[0]))) <= 122


def test_main_disk_errors_wait_instead_of_crashing(monkeypatch, tmp_path):
    real_run, calls = L.run, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real_run(*a, **k)
    monkeypatch.setattr(L, 'run', flaky)
    rc, msgs, _, clk = _main(monkeypatch, tmp_path, ['--duration', '100', '--rotate-s', '30'], Bus())
    assert rc == 0 and any(m.startswith('書けません') for m in msgs) and len(calls) >= 2
    monkeypatch.setattr(L, 'run', real_run)
    monkeypatch.setattr(L, 'free_bytes', lambda p: 1024 ** 3)                    # 空き 1 GB: 書かずに待つ
    rc, msgs, _, _ = _main(monkeypatch, tmp_path / 'b', ['--duration', '130', '--rotate-s', '30'], Bus())
    assert rc == 0 and sum('空きが' in m for m in msgs) == 1 and not os.path.exists(str(tmp_path / 'b'))


def test_health_failures_do_not_slow_the_angle(monkeypatch, tmp_path):
    class NoStatus(Bus):
        def read_byte_data(self, addr, reg):
            raise OSError(121, 'Remote I/O error')
    monkeypatch.setattr(L.os, 'fsync', lambda fd: None)
    s = L.run(L.Reader(NoStatus()), str(tmp_path), hz=50, duration=3.0, clock=Clock(), printer=lambda *a: None)
    rows = _rows(s['path'])
    assert 148 <= len(rows) <= 152 and s['errors'] == 0 and s['health_errors'] == 3
    assert [r['status'] for r in rows if r['status']] == ['E', 'E', 'E'] and all(r['err'] == '0' for r in rows)


def test_lower_priority_does_not_double(monkeypatch):
    calls = []
    monkeypatch.setattr(L.os, 'nice', lambda inc: calls.append(inc) or 10)
    L.lower_priority(10)
    assert calls == [0]                                                        # すでに 10 なら足さない


def test_cal_skips_a_cut_last_line(tmp_path):
    spec = importlib.util.spec_from_file_location('as5601_cal', os.path.join(ROOT, 'tools', 'as5601_cal.py'))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    p = tmp_path / 'x.csv'
    p.write_text('epoch_us,mono_us,raw,deg,status,md,ml,mh,agc,mag,err\n1,1,1000,87.9,51,1,1,0,128,350,0\n2,2,1001,88.0,,,,,,,0\n3,3,10')
    t, a, st = C.load(str(p))
    assert len(t) == 2 and len(st) == 1
