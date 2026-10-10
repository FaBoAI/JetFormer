"""車のスピーカーでの案内(音声合成・WAV・設定・出来事の通知)のテスト。実際の再生(aplay)はしない"""

import time
import wave

import numpy as np
import pytest

from jetformer import config, sound
from jetformer.recorder import Recorder
from jetformer.sensors.base import Sample

import test_drive


class FakeTTS(object):
    available = True

    def speech(self, text):
        return np.full(4800, 0.5, np.float32)


def make_player(monkeypatch, tmp_path, **settings):
    monkeypatch.setattr(sound, 'SETTINGS_PATH', tmp_path / 'sound.json')
    played = []
    p = sound.SoundPlayer(tts=FakeTTS(), play=lambda path: played.append(wave.open(path).getnframes()))
    if settings:
        p.update(settings)
    return p, played


def test_phrases():
    assert sound.phrase('record_start', mode_name='高速走行') == ('データセット収集を開始します', 'start')
    assert '3 シーン' in sound.phrase('record_stop', scenes=3)[0]
    assert sound.phrase('record_stop', scenes=3, runs=True)[0] == 'データセット収集を終了します。3回の走行を保存しました'
    assert sound.phrase('handover', select=True) == ('AIが操縦します', 'ai')
    assert sound.phrase('handover', select=False)[1] == 'human'
    assert sound.phrase('drive_stop', reason='走行を始めた画面との通信が 3 秒途絶えたため')[1] == 'warn'
    assert sound.phrase('drive_stop', error=True)[1] == 'warn'
    assert sound.phrase('drive_start', dry_run=True)[0] == '推論の確認を開始します'
    assert sound.speakable('駐車(1)で AI走行') == '駐車 1 で エーアイ走行'


def test_chimes_are_short_and_within_range():
    for name, make in sound.CHIMES.items():
        x = make()
        assert 0.05 < len(x) / sound.RATE < 1.0 and np.abs(x).max() <= 1.0, name


def test_wav_format_matches_i2s_and_volume(tmp_path):
    p = tmp_path / 'a.wav'
    sound.write_wav(p, np.full(4800, 0.5, np.float32), 100)
    with wave.open(str(p)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 4, 48000)
        x = np.frombuffer(w.readframes(w.getnframes()), '<i4').reshape(-1, 2)
    assert np.array_equal(x[:, 0], x[:, 1]) and x.max() == pytest.approx(0.5 * 2 ** 31, rel=1e-3)
    sound.write_wav(p, np.full(4800, 0.5, np.float32), 0)
    with wave.open(str(p)) as w:
        assert not np.frombuffer(w.readframes(w.getnframes()), '<i4').any()


def test_settings_and_event_switches(monkeypatch, tmp_path):
    p, played = make_player(monkeypatch, tmp_path)
    assert sound.announce(p, 'record_start', mode_name='高速走行')
    assert not sound.announce(p, 'scene_saved')                       # 既定でOFF
    p.update({'events': {'scene': True}, 'volume': 150})
    assert p.settings['volume'] == 100 and sound.announce(p, 'scene_saved')
    p.update({'enabled': False})
    assert not sound.announce(p, 'drive_start')
    assert sound.wait_idle(p) and len(played) == 2
    p2, _ = make_player(monkeypatch, tmp_path)                        # 設定は保存される
    assert p2.settings['enabled'] is False and p2.settings['events']['scene'] is True


def test_urgent_announcement_jumps_the_queue(monkeypatch, tmp_path):
    p, played = make_player(monkeypatch, tmp_path)
    p._play = lambda path: time.sleep(0.3)
    for _ in range(5):
        p.enqueue('案内', 'info')
    time.sleep(0.05)
    sound.announce(p, 'rc_lost')
    assert p.q.qsize() == 1                                           # たまっていた案内は捨てて警告だけ


def test_open_jtalk_speech_is_cached(tmp_path):
    tts = sound.TTS(cache_dir=tmp_path)
    if not tts.available:
        pytest.skip('Open JTalk がありません')
    x = tts.speech('データ収集を開始します')
    assert len(x) / sound.RATE > 0.8 and 0.5 < np.abs(x).max() <= 0.81
    assert len(list(tmp_path.glob('*.npy'))) == 1
    assert np.array_equal(tts.speech('データ収集を開始します'), x)


class FakeHub(object):
    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


def test_recorder_emits_events(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
    rec = Recorder(FakeHub())
    rec._warm_up = lambda: None
    events = []
    rec.listeners.append(lambda ev, **kw: events.append((ev, kw)))
    rec.start(mode_id=4)
    rec.set_mode(5)
    rec.set_mode(5)                                                   # 同じジャンルなら知らせない
    t0 = rec.log['start_ns']
    for i in range(5):
        rec._on_sample('CAM_FRONT', Sample(t0 + i * 33_000_000, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))
    rec._io.join()
    rec.stop()
    names = [e for e, _ in events]
    assert names == ['record_start', 'mode', 'scene_saved', 'record_stop']
    assert events[0][1]['mode_name'] and events[-1][1]['scenes'] == 1


def test_drive_emits_start_handover_and_rc_lost(monkeypatch, tmp_path):
    d = test_drive.make_driver(monkeypatch, tmp_path)
    events = []
    d.listeners.append(lambda ev, **kw: events.append((ev, kw)))

    def veh(i):            # 人 → AI → AI(CH3の信号断)→ 人
        if i < 6:
            return test_drive.veh(False)
        if i < 12:
            return test_drive.veh(True)
        if i < 18:
            return test_drive.veh(True, valid=(False, False, False))
        return test_drive.veh(False)
    runner, pca = test_drive.FakeRunner(np.tile([[1500.0, 1500.0]], (10, 1))), test_drive.FakePCA()
    assert test_drive.run_loop(d, runner, pca, veh, frames=24) is None
    seq = [(e, kw.get('select')) for e, kw in events]
    assert seq == [('drive_start', None), ('handover', True), ('rc_lost', None), ('handover', False)]


def test_lap_phrases():
    assert sound.phrase('lap', time_s=8.24, ai_only=True, best=True) == ('ベスト、8.2秒', 'info')
    assert sound.phrase('lap', time_s=9.66, ai_only=True, best=False) == ('9.7秒', None)
    assert sound.phrase('lap', time_s=12.0, ai_only=False) == ('', None)
