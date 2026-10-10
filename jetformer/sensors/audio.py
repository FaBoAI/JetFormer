"""マイク: JetFormer基板のI2S MEMSマイク(Jetsonの40ピンヘッダ I2S2: pin12=SCLK / pin35=LRCLK / pin38=DIN)

ALSAでI2S2をADMAIF1につなぎ(Jetsonがクロックを出すマスター)、arecord で 48kHz・2ch・32bit を受け取る。
I2S MEMSマイクはL/R選択ピンで決まる片方のチャネルにだけ出すので、信号のあるチャネルを自動で選ぶ。
16kHz・モノラルに間引き(ローパスFIR)、直流分を除いて、24bitの精度のまま int32 で配信・保存する。

40ピンのI2S2が無効(ピンがGPIOのまま)だとクロックが出ず、値は全て0になる → scripts/i2s_mic/enable.sh
"""

import os
import select
import subprocess

import numpy as np

from .. import clock
from .base import Sample, SensorThread

FULL_SCALE = 2.0 ** 31


def lowpass_taps(rate_in, cutoff_hz, n=63):
    """窓付きsincのローパスFIR(間引き前のエイリアス除去用)"""
    m = np.arange(n) - (n - 1) / 2.0
    h = np.sinc(2.0 * cutoff_hz / rate_in * m) * np.hamming(n)
    return h / h.sum()


def dbfs(x):
    return float(20.0 * np.log10(max(x, 1e-12)))


class AudioSensor(SensorThread):
    name = 'AUDIO'

    def __init__(self, cfg):
        super(AudioSensor, self).__init__()
        self.cfg = cfg
        self.proc = None
        self.rate_in = cfg['capture_rate']
        self.rate = cfg['rate']
        self.decim = self.rate_in // self.rate
        self.chunk_frames = self.rate_in * cfg['chunk_ms'] // 1000          # 20ms = 960フレーム(48kHz)
        self.chunk_bytes = self.chunk_frames * 2 * 4
        self.taps = lowpass_taps(self.rate_in, 0.45 * self.rate)
        self.zero_chunks = 0
        self.level_db = None

    def _open(self):
        for name, value in self.cfg['mixer']:
            subprocess.run(['amixer', '-q', '-c', self.cfg['card'], 'cset', 'name=%s' % name, value],
                           check=True, timeout=5, capture_output=True)
        self.proc = subprocess.Popen(
            ['arecord', '-q', '-D', self.cfg['device'], '-c', '2', '-r', str(self.rate_in), '-f', 'S32_LE', '-t', 'raw',
             '--period-time=%d' % (self.cfg['chunk_ms'] * 1000), '--buffer-time=200000'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._reset()

    def _reset(self):
        self._buf = bytearray()
        self.hist = np.zeros(len(self.taps) - 1)
        self.dc = None
        self.energy = np.zeros(2)
        self.channel = {'left': 0, 'right': 1}.get(self.cfg['channel'])
        self.zero_chunks = 0

    def _close(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def _read_chunk(self, timeout=1.0):
        fd = self.proc.stdout.fileno()
        while len(self._buf) < self.chunk_bytes:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise RuntimeError('マイクのデータが来ません(I2Sのクロックが出ていない?)')
            data = os.read(fd, 65536)
            if not data:
                err = self.proc.stderr.read().decode('utf-8', 'replace').strip()
                raise RuntimeError('arecord が止まりました: %s' % (err or '理由不明'))
            self._buf += data
        chunk = bytes(self._buf[:self.chunk_bytes])
        del self._buf[:self.chunk_bytes]
        return chunk

    def _poll(self):
        raw = np.frombuffer(self._read_chunk(), dtype='<i4').reshape(-1, 2)
        t_ns = clock.now_ns()                       # このチャンク(20ms)の末尾の受信時刻
        x = raw.astype(np.float64)
        if self.cfg['channel'] == 'auto':           # 信号のあるチャネル(マイクのL/R選択ピンで決まる)を選ぶ
            self.energy = 0.9 * self.energy + 0.1 * x.var(axis=0)
            best = int(np.argmax(self.energy))
            if self.channel is None or self.energy[best] > 4.0 * self.energy[self.channel]:
                self.channel = best
        y = x[:, self.channel]
        self.dc = y.mean() if self.dc is None else 0.98 * self.dc + 0.02 * y.mean()   # MEMSマイクの直流分を除く
        buf = np.concatenate([self.hist, y - self.dc])
        self.hist = buf[-(len(self.taps) - 1):]
        z = np.convolve(buf, self.taps, mode='valid')[::self.decim]
        pcm = np.clip(np.round(z), -FULL_SCALE, FULL_SCALE - 1).astype(np.int32)
        rms = float(np.sqrt(np.mean(z * z))) / FULL_SCALE
        self.zero_chunks = self.zero_chunks + 1 if not raw.any() else 0
        self.level_db = dbfs(rms)
        return [Sample(t_ns, {'pcm': pcm, 'rate': self.rate, 'rms_dbfs': round(self.level_db, 1),
                              'peak_dbfs': round(dbfs(float(np.abs(z).max()) / FULL_SCALE), 1),
                              'channel': self.channel, 'zero': not raw.any()})]

    def status(self):
        st = super(AudioSensor, self).status()
        st.update(channel=['L', 'R'][self.channel] if self.channel is not None else None,
                  level_dbfs=round(self.level_db, 1) if self.level_db is not None else None,
                  no_signal=self.zero_chunks >= 25)     # 0.5秒以上ずっと0 = クロックが出ていない・マイクが無い
        return st
