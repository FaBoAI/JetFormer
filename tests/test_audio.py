"""マイク(I2S MEMS)の処理のテスト。実機のI2Sは使わず、合成した信号を流す"""

import io
import wave

import numpy as np
import pytest

from jetformer import config, dataset
from jetformer.recorder import Recorder
from jetformer.sensors.audio import AudioSensor
from jetformer.sensors.base import Sample

RATE_IN = config.AUDIO['capture_rate']


def make_sensor(chunks):
    """chunks: [N,2] int32 の配列のリスト(48kHz)を順に返す偽の入力"""
    s = AudioSensor(config.AUDIO)
    s._reset()
    it = iter(chunks)
    s._read_chunk = lambda timeout=1.0: next(it).astype('<i4').tobytes()
    return s


def tone(freq, dbfs, seconds, channel, dc=0.0):
    n = int(RATE_IN * seconds)
    amp = 10 ** (dbfs / 20) * 2 ** 31
    x = np.zeros((n, 2))
    x[:, channel] = amp * np.sin(2 * np.pi * freq * np.arange(n) / RATE_IN) + dc
    return [x[i:i + 960] for i in range(0, n - 959, 960)]


def run(sensor, n):
    return [sensor._poll()[0].data for _ in range(n)]


def test_picks_the_channel_with_signal_and_measures_level():
    s = make_sensor(tone(1000, -20, 1.0, channel=1))
    out = run(s, 50)
    assert all(d['channel'] == 1 for d in out[5:])
    assert out[-1]['rms_dbfs'] == pytest.approx(-23.0, abs=0.5)     # 正弦波のRMS = 振幅 - 3dB
    assert len(out[-1]['pcm']) == 320 and out[-1]['pcm'].dtype == np.int32 and out[-1]['rate'] == 16000
    pcm = np.concatenate([d['pcm'] for d in out[10:]]).astype(float)
    freqs = np.fft.rfftfreq(len(pcm), 1 / 16000)
    assert freqs[np.argmax(np.abs(np.fft.rfft(pcm)))] == pytest.approx(1000, abs=5)


def test_antialias_filter_removes_tones_above_8khz():
    s = make_sensor(tone(12000, -6, 0.5, channel=0))
    out = run(s, 25)
    assert out[-1]['rms_dbfs'] < -45                                  # 12kHz は16kHzに間引くと折り返すので落とす


def test_dc_offset_is_removed():
    s = make_sensor(tone(500, -30, 2.0, channel=0, dc=0.05 * 2 ** 31))
    out = run(s, 100)
    pcm = np.concatenate([d['pcm'] for d in out[-20:]]).astype(float) / 2 ** 31
    assert abs(pcm.mean()) < 1e-3
    assert out[-1]['rms_dbfs'] == pytest.approx(-33.0, abs=1.0)


def test_all_zero_input_reports_no_signal():
    s = make_sensor([np.zeros((960, 2))] * 30)
    out = run(s, 30)
    assert out[-1]['zero'] and s.status()['no_signal']


class FakeHub(object):
    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


def test_scene_saves_audio_and_exports_wav(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
    rec = Recorder(FakeHub())
    rec._warm_up = lambda: None
    log = rec.start()
    t0 = rec.log['start_ns']
    s = make_sensor(tone(440, -12, 0.9, channel=0))
    for i, d in enumerate(run(s, 45)):
        t = t0 + (i + 1) * 20_000_000
        rec._on_sample('AUDIO', Sample(t, d))
        if i % 2 == 0:
            rec._on_sample('CAM_FRONT', Sample(t, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))
    rec._io.join()
    rec._finalize(0, partial=False)
    rec.stop()
    name = '%s-0001' % log
    meta = dataset.load_scene_meta(name)
    assert meta['counts']['AUDIO'] == 45 and not meta['stats']['audio']['no_signal']
    a = np.load(tmp_path / log / name / 'AUDIO.npz')
    assert len(a['pcm']) == 45 * 320 and int(a['sample_rate']) == 16000
    with wave.open(io.BytesIO(dataset.scene_audio_wav(name))) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 45 * 320 and w.getsampwidth() == 2
    assert dataset.scene_timeseries(name)['audio']['rms_dbfs']
