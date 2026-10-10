"""車両データ: JetFormer/FaBoJetRacer コントローラ基板(XIAO ESP32-S3, I2C 0x28)

プロポ受信機の CH1(ステアリング)/ CH2(スロットル)/ CH3(モード切替)のパルス幅と、
ブラシレスモーターのホールセンサーから求めた回転数を読む。

snapshot (ファーム leracer_ctrl v2、既定)
    レジスタ 0x20 から28バイトのスナップショットを読む(基板は読まれた瞬間の値を詰めて返す)。
    struct "<BBHIHHHBBiiHBB":
      magic 0xA5 | version 1 | seq u16 | t_us u32 | ch1 u16 | ch2 u16 | ch3 u16 | flags u8 | hall_state u8
      | erpm i32 | hall_count i32 | hall_errors u16 | reserved u8 | CRC-8(poly 0x07, init 0, byte 0..26)
    flags: bit0-2 = CH1-3 受信中(100ms以内にパルスあり), bit3 = SELECT(1=AI走行側), bit4 = モーター回転中
    erpm: 電気角rpm(符号=回転方向)。モーター軸rpm = erpm / 極対数

legacy (量産ファーム v1)
    レジスタ毎に write→read。v1は古い値がFIFOに残りチャネルが混ざるため、学習には使えない。
"""

import struct
import time

from smbus2 import SMBus, i2c_msg

from .. import clock
from .base import Sample, SensorThread

SNAP_REG = 0x20
SNAP_LEN = 28
SNAP_MAGIC = 0xA5
SNAP_VERSION = 1
SNAP = struct.Struct('<BBHIHHHBBiiHBB')

FLAG_SELECT = 0x08
FLAG_MOVING = 0x10

PLAUSIBLE = {
    'pwm': lambda v: v == 0 or 500 <= v <= 2600,
    'pct': lambda v: 0 <= v <= 100,
}


def crc8(data):
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def decode_snapshot(data):
    """28バイト → dict。形式違い・CRC不一致は ValueError"""
    data = bytes(data)
    if len(data) != SNAP_LEN:
        raise ValueError('長さ %d != %d' % (len(data), SNAP_LEN))
    (magic, ver, seq, t_us, ch1, ch2, ch3, flags, hall_state,
     erpm, hall_count, hall_errors, _rsv, crc) = SNAP.unpack(data)
    if magic != SNAP_MAGIC or ver != SNAP_VERSION:
        raise ValueError('magic/version 不一致 0x%02x/%d' % (magic, ver))
    if crc8(data[:-1]) != crc:
        raise ValueError('CRC 不一致')
    return {
        'pwm': [ch1, ch2, ch3],                         # [steering, throttle, mode] µs
        'valid': [bool(flags & (1 << k)) for k in range(3)],
        'select': bool(flags & FLAG_SELECT),            # True = AI走行側(人の操縦データではない)
        'moving': bool(flags & FLAG_MOVING),
        'hall_state': hall_state, 'erpm': erpm, 'hall_count': hall_count, 'hall_errors': hall_errors,
        'seq': seq, 'board_t_us': t_us, 'quality': 'snapshot',
    }


def decode_legacy(raw, kind):
    """v1: 位置がずれた繰り返しint32列から、妥当な値を復元する。復元できなければNone"""
    candidates = []
    for p in range(max(len(raw) - 3, 0)):
        word = raw[p:p + 4]
        for rot in range(4):
            v = struct.unpack('<i', word[rot:] + word[:rot])[0]
            if PLAUSIBLE[kind](v):
                candidates.append(v)
    if not candidates:
        return None
    nonzero = [v for v in candidates if v != 0]
    return max(set(nonzero), key=nonzero.count) if nonzero else 0


class VehicleSensor(SensorThread):
    name = 'VEHICLE'
    # 速く走るとモーターのノイズでI2Cの読み取りが時々失敗する(2026-10-10のAI走行で8回、毎回0.5秒途切れて
    # 走行が中立で止まった)。1回の失敗はその周期(10ms)を飛ばすだけにし、続いたときだけ開き直す
    MAX_FAIL_STREAK = 30         # 約0.3秒続けて読めなければ開き直す
    error_backoff = 0.05
    # 基板が再起動したとき(2026-10-10 ai_run_008 で12回。強く加速したときの電圧低下とみられる)、基板は約0.07秒で
    # 戻るのに、開き直しを2秒待っていたため走行が2.4秒止まった。すぐ開き直す
    reconnect_interval = 0.25

    def __init__(self, cfg):
        super(VehicleSensor, self).__init__()
        self.cfg = cfg
        self.bus = None
        self.addr = cfg['address']
        self.period = 1.0 / cfg['rate_hz']
        self.protocol = cfg['protocol']
        self._next = 0.0
        self._last_seq = None
        self.read_errors = 0       # CRC不一致・形式違い(読み直し前の回数)
        self.bus_errors = 0        # I2Cの通信エラー(ノイズなど)
        self.skipped_polls = 0     # 読み直しても読めず、その周期を飛ばした回数
        self.fail_streak = 0
        self.last_read_error = ''
        self.stale = 0
        self.board_resets = 0      # 基板の再起動(基板の時計が巻き戻った回数)。電源・バッテリーの電圧低下の目安
        self.last_board_reset = None
        self._last_board_t = None

    def _open(self):
        self.bus = SMBus(self.cfg['bus'])
        if self.protocol == 'snapshot':
            last = None
            for _ in range(5):     # 書き込み直後などに1バイトずれて届くことがあるので読み直す
                try:
                    decode_snapshot(self.bus.read_i2c_block_data(self.addr, SNAP_REG, SNAP_LEN))
                    break
                except ValueError as e:
                    last = e
            else:
                raise RuntimeError('0x%02x がスナップショット形式で応答しません(ファームv2を書き込んでください): %s'
                                   % (self.addr, last))
        else:
            self.bus.i2c_rdwr(i2c_msg.read(self.addr, 1))
        self._next = time.monotonic()

    def _close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def _poll_snapshot(self):
        last = None
        for _ in range(3):
            try:
                data = self.bus.read_i2c_block_data(self.addr, SNAP_REG, SNAP_LEN)
                t_ns = clock.now_ns()
                values = decode_snapshot(data)
                break
            except ValueError as e:              # CRC不一致など
                self.read_errors += 1
                last = e
            except OSError as e:                 # I2Cの通信エラー(ノイズ・NACK・タイムアウト)
                self.bus_errors += 1
                last = e
        else:
            self.fail_streak += 1
            self.skipped_polls += 1
            self.last_read_error = '%s: %s' % (type(last).__name__, last)
            if self.fail_streak >= self.MAX_FAIL_STREAK:
                raise RuntimeError('車両データを約%.1f秒続けて読めません: %s' % (self.fail_streak * self.period, last))
            return []                            # この周期だけ飛ばす
        self.fail_streak = 0
        if values['seq'] == self._last_seq:
            self.stale += 1
            return []
        self._last_seq = values['seq']
        bt = values['board_t_us']
        if self._last_board_t is not None and bt + 1_000_000 < self._last_board_t and self._last_board_t < 4_000_000_000:
            self.board_resets += 1                     # 時計が戻った(32bit µs の一周 約71分 は除く)
            self.last_board_reset = time.time()
        self._last_board_t = bt
        return [Sample(t_ns, values)]

    def _poll_legacy(self):
        pwm, valid = [], []
        for reg in (0x00, 0x01, 0x02):
            self.bus.i2c_rdwr(i2c_msg.write(self.addr, [reg]))
            msg = i2c_msg.read(self.addr, 8)
            self.bus.i2c_rdwr(msg)
            v = decode_legacy(bytes(msg), 'pwm')
            pwm.append(v or 0)
            valid.append(bool(v))
        return [Sample(clock.now_ns(), {
            'pwm': pwm, 'valid': valid, 'select': False, 'moving': False, 'hall_state': 0, 'erpm': 0,
            'hall_count': 0, 'hall_errors': 0, 'seq': self.count, 'board_t_us': 0, 'quality': 'legacy'})]

    def _poll(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(self._next - now, 0.2))
        self._next = max(self._next + self.period, time.monotonic() - self.period)
        return self._poll_snapshot() if self.protocol == 'snapshot' else self._poll_legacy()

    def status(self):
        st = super(VehicleSensor, self).status()
        st.update(protocol=self.protocol, read_errors=self.read_errors, bus_errors=self.bus_errors,
                  skipped_polls=self.skipped_polls, last_read_error=self.last_read_error, stale=self.stale,
                  board_resets=self.board_resets,
                  board_reset_ago_s=round(time.time() - self.last_board_reset, 1) if self.last_board_reset else None)
        if self.latest is not None:
            d = self.latest.data
            st.update(pwm=d['pwm'], erpm=d['erpm'], select=d['select'], valid=d['valid'])
        return st
