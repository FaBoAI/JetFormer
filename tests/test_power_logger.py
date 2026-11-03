"""tools/power_logger.py(Jetson の電源と温度を読むだけの記録係。2026-10-30 ユーザーの決め)の試験。本物の sysfs には触れない

- INA3221 の欄(VDD_IN・VDD_CPU_GPU_CV・VDD_SOC)を名前で見つける・温度は thermal_zone の type で
- 1 秒に hz 回・温度は 1 秒に 1 回・1 秒ごとに fsync・読めないファイルは数えて続ける・同じ秒の名前で上書きしない

    cd ~/jetformer && python -m pytest -q tests/test_power_logger.py
"""
import csv
import importlib.util
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_spec = importlib.util.spec_from_file_location('power_logger', os.path.join(ROOT, 'tools', 'power_logger.py'))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


class Clock(object):
    def __init__(self, t=1761780000.0):
        self.t, self.m = t, 50.0

    def time(self):
        return self.t

    def monotonic(self):
        return self.m

    def sleep(self, dt):
        self.t += dt
        self.m += dt


def _fake_sysfs(tmp_path):
    hw = tmp_path / 'hwmon'
    for i, name in enumerate(['pwmfan', 'ina3221']):
        d = hw / ('hwmon%d' % i)
        d.mkdir(parents=True)
        (d / 'name').write_text(name + '\n')
    ina = hw / 'hwmon1'
    for n, (lab, mv, ma) in enumerate([('VDD_IN', 5000, 800), ('VDD_CPU_GPU_CV', 4992, 160), ('VDD_SOC', 5000, 256)], 1):
        (ina / ('in%d_label' % n)).write_text(lab + '\n')
        (ina / ('in%d_input' % n)).write_text('%d\n' % mv)
        (ina / ('curr%d_input' % n)).write_text('%d\n' % ma)
    (ina / 'in7_label').write_text('sum of shunt voltages\n')
    th = tmp_path / 'thermal'
    for i, (name, mc) in enumerate([('cpu-thermal', 47812), ('gpu-thermal', 46000)]):
        z = th / ('thermal_zone%d' % i)
        z.mkdir(parents=True)
        (z / 'type').write_text(name + '\n')
        (z / 'temp').write_text('%d\n' % mc)
    return str(hw), str(th), ina


def test_finds_channels_and_zones(tmp_path):
    hw, th, _ = _fake_sysfs(tmp_path)
    s = P.Sensors(hw, th)
    assert s.ina.endswith('hwmon1')
    assert sorted(s.ch) == sorted(['vin_mv', 'iin_ma', 'vcpu_mv', 'icpu_ma', 'vsoc_mv', 'isoc_ma'])
    vals, err = s.power()
    assert vals['vin_mv'] == 5000 and vals['iin_ma'] == 800 and err == 0
    assert s.temps() == 'cpu-thermal=47.8;gpu-thermal=46.0'


def test_run_rate_temps_fsync_and_meta(tmp_path, monkeypatch):
    syncs = []
    monkeypatch.setattr(P.os, 'fsync', lambda fd: syncs.append(fd))
    hw, th, _ = _fake_sysfs(tmp_path)
    s = P.run(P.Sensors(hw, th), str(tmp_path / 'out'), hz=10, duration=3.0, clock=Clock())
    rows = list(csv.DictReader(open(s['path'], encoding='utf-8')))
    assert list(rows[0].keys()) == P.FIELDS and 29 <= len(rows) <= 31 and s['vin_min_mv'] == 5000
    assert sum(1 for r in rows if r['temps']) == 3 and rows[0]['vin_mv'] == '5000' and rows[0]['err'] == '0'
    assert len(syncs) >= 4                                                     # 始めの json・1 秒ごと・終わり
    meta = json.load(open(s['path'][:-4] + '.json', encoding='utf-8'))
    assert meta['fields'] == P.FIELDS and 'vin_mv' in meta['channels'] and meta['zones'] == ['cpu-thermal', 'gpu-thermal']


def test_unreadable_files_are_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(P.os, 'fsync', lambda fd: None)
    hw, th, ina = _fake_sysfs(tmp_path)
    se = P.Sensors(hw, th)
    (ina / 'curr1_input').unlink()                                            # 途中で読めなくなっても止まらない
    s = P.run(se, str(tmp_path / 'out'), hz=10, duration=1.0, clock=Clock())
    rows = list(csv.DictReader(open(s['path'], encoding='utf-8')))
    assert all(r['err'] == '1' and r['iin_ma'] == '' and r['vin_mv'] == '5000' for r in rows)


def test_same_second_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(P.os, 'fsync', lambda fd: None)
    hw, th, _ = _fake_sysfs(tmp_path)
    clk = Clock()
    a = P.run(P.Sensors(hw, th), str(tmp_path / 'out'), hz=10, duration=0.0, clock=clk)
    b = P.run(P.Sensors(hw, th), str(tmp_path / 'out'), hz=10, duration=0.0, clock=clk)
    assert a['path'] != b['path'] and b['path'].endswith('-2.csv')


def test_reads_only_sysfs():
    src = open(os.path.join(ROOT, 'tools', 'power_logger.py'), encoding='utf-8').read()
    assert 'smbus' not in src and '/dev/i2c' not in src                         # I2C の装置に直に触れない


# ---------------------------------------------------------------- 2026-10-30 反証 S3・S4: main() をそのまま動かす(偽物の sysfs・時計)
import errno  # noqa: E402


def _main(monkeypatch, tmp_path, args, se):
    monkeypatch.setattr(P.signal, 'signal', lambda sig, fn: None)
    monkeypatch.setattr(P.os, 'fsync', lambda fd: None)
    monkeypatch.setattr(P.os, 'nice', lambda inc: 10)
    msgs = []
    rc = P.main(['--out', str(tmp_path / 'out')] + args, clock=Clock(), sensors=se, printer=msgs.append)
    return rc, msgs


def test_main_rotates(monkeypatch, tmp_path):
    hw, th, _ = _fake_sysfs(tmp_path)
    rc, msgs = _main(monkeypatch, tmp_path, ['--duration', '25', '--rotate-s', '10'], P.Sensors(hw, th))
    files = [p for p in os.listdir(str(tmp_path / 'out')) if p.endswith('.csv')]
    assert rc == 0 and len(files) == 3 and msgs[-1].startswith('止めました')
    assert open(os.path.join(str(tmp_path / 'out'), files[0]), 'rb').read().count(b'\r') == 0


def test_main_disk_errors_wait(monkeypatch, tmp_path):
    hw, th, _ = _fake_sysfs(tmp_path)
    real_run, calls = P.run, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.EROFS, 'Read-only file system')
        return real_run(*a, **k)
    monkeypatch.setattr(P, 'run', flaky)
    rc, msgs = _main(monkeypatch, tmp_path, ['--duration', '100', '--rotate-s', '30'], P.Sensors(hw, th))
    assert rc == 0 and any(m.startswith('書けません') for m in msgs) and len(calls) >= 2
