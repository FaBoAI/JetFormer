"""走行ごとに区切る収集(カウントダウン・終了の合図・連続収集)のテスト。音は鳴らさず、時刻だけ確かめる"""

import threading
import time

import numpy as np
import pytest

from jetformer import clock, config, dataset, session, sound
from jetformer.recorder import Recorder
from jetformer.sensors.base import Sample


class FakeHub(object):
    channels = ('CAM_FRONT',)

    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


class FakePlayer(object):
    """play_clip は on_start をすぐ別スレッドで呼ぶ(再生の始まり)。鳴らした合図を記録する"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.clips = []
        self.tts = None
        self.start_times = []

    def play_clip(self, event, mono, on_start=None):
        if not self.enabled:
            return False
        self.clips.append(event)
        if on_start:
            t = time.monotonic()
            self.start_times.append(t)
            threading.Thread(target=on_start, args=(t,)).start()
        return True


@pytest.fixture
def rec(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(session, 'LIMITS', {'runs': (1, 10), 'run_seconds': (0.1, 10), 'pause_s': (0, 10)})
    monkeypatch.setattr(session, 'SILENT_COUNTDOWN_S', 0.2)
    monkeypatch.setattr(sound, 'countdown_clip', lambda tts, n: (np.zeros(100, np.float32), 0.2))
    monkeypatch.setattr(sound, 'end_clip', lambda tts, n: np.zeros(100, np.float32))
    r = Recorder(FakeHub())
    r._warm_up = lambda: None
    r.events = []
    r.listeners.append(lambda ev, **kw: r.events.append((ev, kw)))
    feeding = {'on': True}

    def feed():                       # 30fpsのカメラの代わり
        i = 0
        while feeding['on']:
            if r.recording:
                r._on_sample('CAM_FRONT', Sample(clock.now_ns(), {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))
                i += 1
            time.sleep(1 / 30)
    th = threading.Thread(target=feed, daemon=True)
    th.start()
    yield r
    feeding['on'] = False


def wait_done(s, timeout=15):
    t0 = time.time()
    while s.active and time.time() - t0 < timeout:
        time.sleep(0.05)
    assert not s.active


def test_recorder_runs_mode_records_only_inside_runs(rec):
    log = rec.start(runs_mode=True)
    time.sleep(0.3)                                       # 走行の前: 記録しない
    rec.begin_run(5.0)
    time.sleep(0.5)
    assert rec.end_run() == 1
    time.sleep(0.4)                                       # 走行の後: 記録しない
    rec.stop()
    assert rec.saved == ['%s-0001' % log]
    meta = dataset.load_scene_meta('%s-0001' % log)
    assert meta['run_index'] == 1 and meta['complete'] and 0.3 < meta['duration_s'] < 0.6
    assert 10 <= meta['counts']['CAM_FRONT'] <= 18


def test_session_runs_repeat_with_pause_and_announcements(rec):
    p = FakePlayer()
    s = session.CollectSession(rec, p)
    s.start(runs=2, run_seconds=0.6, pause_s=0.5, mode_id=4)
    wait_done(s)
    assert s.status()['done'] == 2 and not s.status().get('error')
    assert p.clips == ['run', 'run', 'run', 'run']                    # カウントダウン・終了 × 2回
    evs = [e for e, _ in rec.events if e in ('record_start', 'record_stop')]
    assert evs == ['record_start', 'record_stop']
    stop_kw = [kw for e, kw in rec.events if e == 'record_stop'][0]
    assert stop_kw == {'scenes': 2, 'runs': True}
    metas = [dataset.load_scene_meta(n) for n in rec.saved]
    assert [m['run_index'] for m in metas] == [1, 2] and all(m['complete'] for m in metas)
    gap = (metas[1]['start_us'] - metas[0]['end_us']) / 1e6           # 停止秒数 + カウントダウンの間は記録しない
    assert gap > 0.6


def test_recording_starts_at_the_beep(rec):
    p = FakePlayer()
    s = session.CollectSession(rec, p)
    s.start(runs=1, run_seconds=0.5, pause_s=0)
    wait_done(s)
    meta = dataset.load_scene_meta(rec.saved[0])
    beep_epoch_us = clock.to_epoch_us(int((p.start_times[0] + sound.LEAD_S + 0.2) * 1e9))
    assert abs(meta['start_us'] - beep_epoch_us) < 70_000                # 最初のフレームはピから1フレーム以内


def test_run_lasts_exactly_from_start_beep_to_end_beep(rec):
    p = FakePlayer()
    s = session.CollectSession(rec, p)
    s.start(runs=1, run_seconds=1.5, pause_s=0)
    wait_done(s)
    start_beep = p.start_times[0] + sound.LEAD_S + 0.2
    end_beep = p.start_times[1] + sound.LEAD_S
    assert end_beep - start_beep == pytest.approx(1.5, abs=0.06)       # ピーからピーまでがちょうど走行時間
    meta = dataset.load_scene_meta(rec.saved[0])
    assert meta['duration_s'] == pytest.approx(1.5, abs=0.08)          # 記録もその間だけ


def test_skip_pause_end_run_early_and_stop(rec):
    p = FakePlayer()
    s = session.CollectSession(rec, p)
    s.start(runs=3, run_seconds=5.0, pause_s=5.0)
    t0 = time.time()
    while s.status()['phase'] != 'running':
        time.sleep(0.02)
    time.sleep(0.3)
    s.end_run()                                                       # 5秒待たずに終える
    while s.status()['phase'] != 'pause':
        time.sleep(0.02)
    s.next_run()                                                      # 5秒の停止を飛ばす
    while s.status()['phase'] != 'running' or s.status()['run'] != 2:
        time.sleep(0.02)
    time.sleep(0.3)
    s.stop()                                                          # 2回目の途中で止める
    wait_done(s)
    assert time.time() - t0 < 4.0
    metas = [dataset.load_scene_meta(n) for n in rec.saved]
    assert [m['complete'] for m in metas] == [True, False]             # 打ち切った走行は complete=False
    assert s.status()['done'] == 1


def test_silent_mode_still_counts_down(rec):
    s = session.CollectSession(rec, FakePlayer(enabled=False))
    s.start(runs=1, run_seconds=0.3, pause_s=0)
    wait_done(s)
    assert s.status()['done'] == 1 and len(rec.saved) == 1


def test_invalid_settings_are_rejected(rec):
    s = session.CollectSession(rec, FakePlayer())
    with pytest.raises(ValueError):
        s.start(runs=0)
    with pytest.raises(ValueError):
        s.start(runs=1, run_seconds=100)


def test_countdown_clip_layout():
    class T(object):
        def speech(self, text):
            return np.full(int(0.4 * sound.RATE), 0.1, np.float32)
    mono, beep_at = sound.countdown_clip(T(), 3)
    assert beep_at == pytest.approx(0.4 + 0.35 + 3.0, abs=1e-3)       # 文 + 間 + 3・2・1(1秒ずつ)
    tail = mono[int(beep_at * sound.RATE):]
    assert len(tail) / sound.RATE == pytest.approx(0.5, abs=0.01) and np.abs(tail).max() > 0.3     # 「ピー」
    assert sound.phrase('record_start') == ('データセット収集を開始します', 'start')
    assert sound.phrase('record_stop', scenes=2, runs=True)[0] == 'データセット収集を終了します。2回の走行を保存しました'
