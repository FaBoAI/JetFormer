"""VL53L9 ToF(JetFormer基板 ESP32-P4 / csi_ranging ファームのUSB CDC)

ストリーム(リトルエンディアン、ファーム公式の l9proto で解析):
    L9FR(旧): 'L9FR' | seq u32 | width u16 | height u16 | zones u16 * (width*height) | checksum u32
    L9FE(新): 'L9FE' の後ろを全てバイトスタッフィング(0xFF,0x0F,0x16,0x14,0x7D → 0x7D, 値^0x20)
    checksum = seq + ゾーンの全バイトの和 (u32)
フレームの間に '#ST rx=.. center=.. drops=..' 形式のステータス行が混ざる。
binning=2 の 54x42 ゾーンを約32fpsで受信する。ゾーン値の解釈は tof_format を参照
(下位13bit=距離mm、上位3bit=信頼度)。ここでは生の値のまま配信・保存する。
"""

import re

import numpy as np
import serial

from .. import clock
from .base import Sample, SensorThread
from .l9proto import FrameParser

STATUS_RE = re.compile(rb'#ST ([^\r\n]*)\r?\n')


class TofSensor(SensorThread):
    name = 'TOF_FRONT'

    def __init__(self, cfg):
        super(TofSensor, self).__init__()
        self.cfg = cfg
        self.serial = None
        self.parser = FrameParser()
        self._text = bytearray()          # ステータス行の取り出し用(フレームはパーサが処理)
        self.skipped_seq = 0
        self.device_status = {}
        self._last_seq = None

    def _open(self):
        self.serial = serial.Serial(self.cfg['port'], self.cfg['baudrate'], timeout=0.2)
        self.serial.reset_input_buffer()
        self.parser = FrameParser()
        self._text = bytearray()
        self._last_seq = None

    def _close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def _parse_status(self, chunk):
        self._text += chunk
        for m in STATUS_RE.finditer(self._text):
            status = {}
            for item in m.group(1).decode('ascii', 'replace').split():
                if '=' in item:
                    k, v = item.split('=', 1)
                    try:
                        status[k] = int(v)
                    except ValueError:
                        status[k] = v
            self.device_status = status
        del self._text[:max(len(self._text) - 64, 0)]   # 行の途中が残る分だけ保持

    def _poll(self):
        chunk = self.serial.read(max(self.serial.in_waiting, 1))
        t_ns = clock.now_ns()
        if not chunk:
            return []
        self._parse_status(chunk)
        samples = []
        for seq, w, h, payload in self.parser.feed(chunk):
            # 接続直後はESP32側に溜まった古いフレームで番号が大きく飛ぶので、小さい欠落だけ数える
            if self._last_seq is not None and self._last_seq + 1 < seq < self._last_seq + 1000:
                self.skipped_seq += seq - self._last_seq - 1
            self._last_seq = seq
            zones = np.frombuffer(payload, dtype='<u2').reshape(h, w)
            samples.append(Sample(t_ns, {'depth': zones, 'seq': seq}))
        return samples

    def status(self):
        st = super(TofSensor, self).status()
        st.update(bad_frames=self.parser.crc_errors, skipped=self.skipped_seq, device=self.device_status,
                  unit=self.cfg.get('unit', ''))
        return st
