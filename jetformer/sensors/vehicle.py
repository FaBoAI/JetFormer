"""車両データ: JetFormerコントロール基板(ESP32-S3, I2Cスレーブ 0x28)

プロポ受信機のPWM(ステアリング/スロットル/補助)とモーター回転数を読む。
3つの読み出し方式に対応する(docs/firmware_protocol.md 参照):

usb    (推奨・修正版ファーム)
    ToFと同じUSBストリームに 'JFVH' パケットを流してもらい、ToFスレッドから受け取る。
    ToFと操作値が同じ基板の時計で刻まれ、I2Cバスも使わない。

frame  (推奨・修正版ファーム)
    基板は24バイトのスナップショットフレームをTX FIFOに詰め続ける:
      'JF' | seq u16 | t_ms u32 | pwm1 u16 | pwm2 u16 | pwm3 u16 | m1_us u16 | m1_pct u16
           | flags u16 | reserved u16 | checksum u16 (=先頭22バイトの和 & 0xFFFF)
    Jetsonは48バイト読み、チェックサムの合う最も新しいフレームを採用する。
    FIFOに古いデータが残っていても壊れない。

legacy (現行ファーム main/leracer.c)
    レジスタIDを1バイトwrite → int32(LE)を4バイトread。
    ファームが「最後に選択したレジスタの値」を詰め続けるため、レジスタを切り替えると
    前のレジスタの値がFIFOに残り、バイト位置もずれる。妥当な値の範囲から位置ずれを
    補正するが、値の取り違えは防げないため quality='legacy' として記録する。
"""

import queue
import struct
import time

from smbus2 import SMBus, i2c_msg

from .. import clock
from .base import Sample, SensorThread

FRAME_LEN = 24
FRAME_MAGIC = b'JF'
FRAME_REGISTER = 0x20
FRAME = struct.Struct('<2sHIHHHHHHHH')     # 22バイト + checksum u16

PLAUSIBLE = {
    'pwm': lambda v: v == 0 or 500 <= v <= 2600,
    'pct': lambda v: 0 <= v <= 100,
}


def parse_frames(data):
    """bytes中の正しいフレームを全て返す [(seq, t_ms, values_dict), ...]"""
    frames = []
    i = data.find(FRAME_MAGIC)
    while 0 <= i <= len(data) - FRAME_LEN:
        chunk = data[i:i + FRAME_LEN]
        checksum, = struct.unpack_from('<H', chunk, 22)
        if sum(chunk[:22]) & 0xFFFF == checksum:
            (_, seq, t_ms, p1, p2, p3, m1_us, m1_pct, flags, _r) = FRAME.unpack_from(chunk)
            frames.append((seq, t_ms, {'pwm': [p1, p2, p3], 'motor_us': m1_us,
                                       'motor_pct': m1_pct, 'flags': flags}))
        i = data.find(FRAME_MAGIC, i + 1)
    return frames


def decode_legacy(raw, kind):
    """位置がずれた繰り返しint32列から、妥当な値を復元する。復元できなければNone"""
    if len(raw) < 4:
        return None
    ok = PLAUSIBLE[kind]
    candidates = []
    for p in range(4):
        word = raw[p:p + 4] if p + 4 <= len(raw) else None
        if word is None or len(word) < 4:
            continue
        for rot in range(4):
            b = word[rot:] + word[:rot]
            v = struct.unpack('<i', b)[0]
            if ok(v):
                candidates.append(v)
    if not candidates:
        return None
    # 0は「データ無し」とも区別できないので、0以外の候補があればそちらを優先
    nonzero = [v for v in candidates if v != 0]
    return max(set(nonzero), key=nonzero.count) if nonzero else 0


class VehicleSensor(SensorThread):
    name = 'VEHICLE'

    def __init__(self, cfg):
        super(VehicleSensor, self).__init__()
        self.cfg = cfg
        self.bus = None
        self.addr = cfg['address']
        self.period = 1.0 / cfg['rate_hz']
        self.protocol = cfg['protocol']
        self._next = 0.0
        self._last_seq = None
        self.stale = 0
        self.invalid = 0
        self._fed = queue.Queue(maxsize=1000)

    def feed(self, t_ns, data):
        """protocol='usb': ToFと同じUSBストリームで届いた車両パケットを受け取る"""
        try:
            self._fed.put_nowait(Sample(t_ns, data))
        except queue.Full:
            pass

    def _open(self):
        if self.protocol == 'usb':
            return
        self.bus = SMBus(self.cfg['bus'])
        # 応答確認(0x28が居ること)
        self.bus.i2c_rdwr(i2c_msg.read(self.addr, 1))
        if self.protocol == 'frame':
            self.bus.i2c_rdwr(i2c_msg.write(self.addr, [FRAME_REGISTER]))
        self._next = time.monotonic()

    def _close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def _read(self, n):
        msg = i2c_msg.read(self.addr, n)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def _poll_frame(self):
        data = self._read(FRAME_LEN * 2)
        t_ns = clock.now_ns()
        frames = parse_frames(data)
        if not frames:
            self.invalid += 1
            return []
        seq, t_ms, values = max(frames, key=lambda f: f[0])
        if seq == self._last_seq:
            self.stale += 1
            return []
        self._last_seq = seq
        flags = values['flags']
        values.update(seq=seq, board_t_ms=t_ms, quality='frame',
                      valid=[bool(flags & (1 << k)) for k in range(4)])
        return [Sample(t_ns, values)]

    def _poll_legacy(self):
        ch = self.cfg['channels']
        pwm = []
        valid = []
        for name in ('steering', 'throttle', 'aux'):
            self.bus.i2c_rdwr(i2c_msg.write(self.addr, [ch[name]]))
            v = decode_legacy(self._read(8), 'pwm')
            pwm.append(v or 0)
            valid.append(bool(v))
        self.bus.i2c_rdwr(i2c_msg.write(self.addr, [ch['motor_pct']]))
        m = decode_legacy(self._read(8), 'pct')
        valid.append(m is not None)
        t_ns = clock.now_ns()
        if not any(valid):
            self.invalid += 1
        return [Sample(t_ns, {'pwm': pwm, 'motor_us': 0, 'motor_pct': m or 0, 'flags': 0,
                              'seq': self.count, 'board_t_ms': 0, 'quality': 'legacy', 'valid': valid})]

    def _poll_usb(self):
        try:
            samples = [self._fed.get(timeout=0.2)]
        except queue.Empty:
            return []
        while not self._fed.empty():
            samples.append(self._fed.get_nowait())
        return samples

    def _poll(self):
        if self.protocol == 'usb':
            return self._poll_usb()
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(self._next - now, 0.2))
        self._next = max(self._next + self.period, time.monotonic() - self.period)
        return self._poll_frame() if self.protocol == 'frame' else self._poll_legacy()

    def status(self):
        st = super(VehicleSensor, self).status()
        st.update(protocol=self.protocol, stale=self.stale, invalid=self.invalid)
        if self.latest is not None:
            d = self.latest.data
            st['pwm'] = d['pwm']
            st['motor_pct'] = d['motor_pct']
        return st
