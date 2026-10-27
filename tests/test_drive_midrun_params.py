"""走行中に変えた設定の記録(2026-10-26 fix2 state2-2・R2-rec-2 のサーバー側)。

指摘: 走行中に天井・自動合わせ・発進の前進を OFF にしても、log.json の drive.params は走行開始の値のまま、events.jsonl にも
何も書かれない。夜の道具は ON の本として数えていた。
直したこと: set_params が走行中に値を変えたら events.jsonl に params(changed・epoch_us)。止めるとき log.json の drive に
params_end・params_changes。変えなかった走行の log.json は今までと同じ。
本物の drive.Driver と本物の recorder.Recorder(偽のハブ。記録の置き場・設定ファイルは tmp)。車・装置・サーバーは使わない。
"""
import json
import threading

import pytest

from jetformer import clock, config, drive, recorder


class FakeCh(object):
    connected = True
    count = 0

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


class FakeHub(object):
    channels = ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE')

    def __init__(self):
        self.ch = {n: FakeCh() for n in self.channels}

    def __getitem__(self, k):
        return self.ch[k]

    def start(self):
        pass


@pytest.fixture
def rig(monkeypatch, tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    monkeypatch.setattr(config, 'RAW_DIR', raw)
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    rec = recorder.Recorder(FakeHub())
    rec._warm_up = lambda timeout=8.0: None
    d = drive.Driver(hub=FakeHub(), recorder=rec)
    d.params = d._launch_rules(d._speed_ctrl_rules(d._clean_params(
        {'throttle_gain': 0.95, 'speed_ceiling': 1, 'ceil_left_mps': 4.1, 'speed_auto': 1, 'auto_target_mps': 4.27,
         'launch_boost': 0}), False), False)
    stop = threading.Event()
    th = threading.Thread(target=stop.wait, daemon=True)               # 「走行中」(スレッドが生きている)
    th.start()
    d.thread = th
    d.st = dict(d.st, run='clean_driving_20261011-nocmd-s3-20261011-024626')
    drive_info = {'run': d.st['run'], 'dataset': 'x', 'dry_run': False, 'params': dict(d.params)}
    name = rec.start(description='day=2026-11-03 block=8 cand=EXP tune=2-3', tags=[], mode_id=5, drive=drive_info)
    yield d, rec, raw / name, name
    stop.set()
    th.join(1)
    if rec.recording:
        rec.stop()


def events(log_dir):
    p = log_dir / 'events.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []


def test_midrun_off_is_recorded(rig):
    d, rec, log_dir, name = rig
    t0 = clock.to_epoch_us(clock.now_ns())
    d.set_params({'speed_ceiling': 0, 'speed_auto': 0})
    assert d.params['speed_ceiling'] == 0 and d.params['speed_auto'] == 0
    with pytest.raises(ValueError):
        d.set_params({'speed_ceiling': 1})                             # ON に戻すのは前と同じく断る(記録も増えない)
    ev = [e for e in events(log_dir) if e['kind'] == 'params']
    assert len(ev) == 1 and ev[0]['changed'] == {'speed_ceiling': [1, 0], 'speed_auto': [1, 0]}
    assert ev[0]['epoch_us'] >= t0 and ev[0]['run'] == d.st['run']
    assert d._params_end_to_log(name) is True                         # 止めるとき(drive._run の後始末と同じ)
    rec.stop(only_log=name)
    lj = json.loads((log_dir / 'log.json').read_text(encoding='utf-8'))
    assert lj['drive']['params']['speed_ceiling'] == 1 and lj['drive']['params']['speed_auto'] == 1   # 走行開始の値のまま
    assert lj['drive']['params_end']['speed_ceiling'] == 0 and lj['drive']['params_end']['speed_auto'] == 0
    assert [c['changed'] for c in lj['drive']['params_changes']] == [ev[0]['changed']]


def test_slider_change_midrun_is_recorded(rig):
    d, rec, log_dir, name = rig
    d.set_params({'throttle_gain': 1.0})
    d.set_params({'throttle_gain': 1.0})                               # 同じ値をもう一度: 記録しない
    d.set_params({'steer_trim_us': 15})
    ev = [e['changed'] for e in events(log_dir) if e['kind'] == 'params']
    assert ev == [{'throttle_gain': [0.95, 1.0]}, {'steer_trim_us': [0.0, 15.0]}]


def test_no_change_keeps_log_json_bytes(rig):
    d, rec, log_dir, name = rig
    before = (log_dir / 'log.json').read_bytes()
    d.set_params({'throttle_gain': 0.95, 'speed_ceiling': 1})          # 値は同じ
    assert d._params_end_to_log(name) is False
    rec.stop(only_log=name)
    assert (log_dir / 'log.json').read_bytes() == before               # 変えなかった走行は今までと同じ
    assert not [e for e in events(log_dir) if e['kind'] == 'params']


def test_not_running_is_not_recorded(rig):
    d, rec, log_dir, name = rig
    d.thread = None                                                    # 止まっている
    d.set_params({'speed_ceiling': 0})
    assert not [e for e in events(log_dir) if e['kind'] == 'params'] and d.params_changes == []


def test_update_drive_only_for_that_log(rig):
    d, rec, log_dir, name = rig
    assert rec.update_drive('20000101-000000', params_end={}) is False
    assert rec.update_drive(name, params_end={'x': 1}) is True
    lj = json.loads((log_dir / 'log.json').read_text(encoding='utf-8'))
    assert lj['drive']['params_end'] == {'x': 1} and 'params' in lj['drive']


def test_params_changed_helper():
    assert drive.params_changed({'a': 1, 'b': 0.5}, {'a': 1.0, 'b': 0.5000000001}) == {}
    assert drive.params_changed({'a': 1}, {'a': 0, 'c': 2}) == {'a': [1, 0], 'c': [None, 2]}
