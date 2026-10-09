"""VL53L9 ToF(ESP32ブリッジ経由のUSB CDC)

ストリーム形式(リトルエンディアン、実機の受信データから解読):
    'L9FR' | seq u32 | width u16 | height u16 | distance u16 * (width*height) | checksum u32
    checksum = (seq + sum(distance)) & 0xFFFFFFFF
フレームの間に '#ST rx=.. center=.. drops=..\\r\\n' 形式のステータス行が混ざる。
binning=2 の 54x42 ゾーンを約32fpsで受信する。

同じストリームに車両パケット(提案中の 'JFVH'、docs/firmware_protocol.md)が流れてきたら
解析して vehicle_feed に渡す(VEHICLE の protocol='usb' のとき):
    'JFVH' | seq u32 | t_us u32 | pwm1 u16 | pwm2 u16 | pwm3 u16 | m1_us u16 | m1_pct u16 | flags u16 | checksum u32
    checksum = (seq + t_us + 各u16の和) & 0xFFFFFFFF
"""

import re
import struct
import time

import numpy as np
import serial

from .. import clock
from .base import Sample, SensorThread

MAGIC = b'L9FR'
HEADER = struct.Struct('<4sIHH')
VEH_MAGIC = b'JFVH'
VEH_PACKET = struct.Struct('<4sII6HI')      # 28バイト
STATUS_RE = re.compile(rb'#ST ([^\r\n]*)\r?\n')


class TofSensor(SensorThread):
    name = 'TOF_FRONT'

    def __init__(self, cfg):
        super(TofSensor, self).__init__()
        self.cfg = cfg
        self.serial = None
        self._buf = bytearray()
        self.bad_frames = 0
        self.skipped_seq = 0
        self.device_status = {}
        self._last_seq = None
        self.vehicle_feed = None       # 車両パケットの受け渡し先(SensorHubが設定)
        self.vehicle_packets = 0
        self.vehicle_bad = 0

    def _open(self):
        self.serial = serial.Serial(self.cfg['port'], self.cfg['baudrate'], timeout=0.2)
        self.serial.reset_input_buffer()
        self._buf = bytearray()
        self._last_seq = None

    def _close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def _parse_status(self, text):
        status = {}
        for item in text.decode('ascii', 'replace').split():
            if '=' in item:
                k, v = item.split('=', 1)
                try:
                    status[k] = int(v)
                except ValueError:
                    status[k] = v
        self.device_status = status

    def _poll(self):
        chunk = self.serial.read(max(self.serial.in_waiting, 1))
        t_ns = clock.now_ns()
        if not chunk:
            return []
        self._buf += chunk
        samples = []
        while True:
            start = self._buf.find(MAGIC)
            vstart = self._buf.find(VEH_MAGIC)
            if vstart >= 0 and (start < 0 or vstart < start):
                if vstart > 0:
                    del self._buf[:vstart]
                if len(self._buf) < VEH_PACKET.size:
                    break
                self._parse_vehicle(bytes(self._buf[:VEH_PACKET.size]), t_ns)
                del self._buf[:VEH_PACKET.size]
                continue
            if start < 0:
                # ステータス行だけ拾って、マジックの断片が残りうる末尾3バイト以外は捨てる
                for m in STATUS_RE.finditer(self._buf):
                    self._parse_status(m.group(1))
                del self._buf[:max(len(self._buf) - 3, 0)]
                break
            if start > 0:
                for m in STATUS_RE.finditer(self._buf[:start]):
                    self._parse_status(m.group(1))
                del self._buf[:start]
            if len(self._buf) < HEADER.size:
                break
            _, seq, width, height = HEADER.unpack_from(self._buf)
            n = width * height
            if not (0 < width <= 256 and 0 < height <= 256):
                self.bad_frames += 1
                del self._buf[:4]
                continue
            total = HEADER.size + n * 2 + 4
            if len(self._buf) < total:
                break
            depth = np.frombuffer(bytes(self._buf[HEADER.size:HEADER.size + n * 2]), dtype='<u2')
            checksum, = struct.unpack_from('<I', self._buf, HEADER.size + n * 2)
            if (seq + int(depth.sum(dtype=np.uint64))) & 0xFFFFFFFF != checksum:
                self.bad_frames += 1
                del self._buf[:4]      # 偽のマジックの可能性があるので1つ先から探し直す
                continue
            del self._buf[:total]
            # 接続直後はESP32側に溜まった古いフレームで番号が大きく飛ぶので、小さい欠落だけ数える
            if self._last_seq is not None and self._last_seq + 1 < seq < self._last_seq + 1000:
                self.skipped_seq += seq - self._last_seq - 1
            self._last_seq = seq
            samples.append(Sample(t_ns, {'depth': depth.reshape(height, width), 'seq': seq}))
        return samples

    def _parse_vehicle(self, packet, t_ns):
        _, seq, t_us, p1, p2, p3, m1_us, m1_pct, flags, checksum = VEH_PACKET.unpack(packet)
        if (seq + t_us + p1 + p2 + p3 + m1_us + m1_pct + flags) & 0xFFFFFFFF != checksum:
            self.vehicle_bad += 1
            return
        self.vehicle_packets += 1
        if self.vehicle_feed is not None:
            self.vehicle_feed(t_ns, {'pwm': [p1, p2, p3], 'motor_us': m1_us, 'motor_pct': m1_pct, 'flags': flags,
                                     'seq': seq, 'board_t_ms': t_us // 1000, 'quality': 'usb',
                                     'valid': [bool(flags & (1 << k)) for k in range(4)]})

    def status(self):
        st = super(TofSensor, self).status()
        st.update(bad_frames=self.bad_frames, skipped=self.skipped_seq, device=self.device_status,
                  vehicle_packets=self.vehicle_packets, vehicle_bad=self.vehicle_bad,
                  unit=self.cfg.get('unit', ''))
        return st
