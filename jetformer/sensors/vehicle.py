"""車両データ: JetFormer/FaBoJetRacer コントローラ基板(XIAO ESP32-S3, I2C 0x28)

プロポ受信機の CH1(ステアリング)/ CH2(スロットル)/ CH3(モード切替)のパルス幅と、
ブラシレスモーターのホールセンサーから求めた回転数を読む。

snapshot (ファーム leracer_ctrl v2、既定)
    レジスタ 0x20 から28バイトのスナップショットを読む(基板は読まれた瞬間の値を詰めて返す)。
    struct "<BBHIHHHBBiiHBB":
      magic 0xA5 | version 1 | seq u16 | t_us u32 | ch1 u16 | ch2 u16 | ch3 u16 | flags u8 | hall_state u8
      | erpm i32 | hall_count i32 | hall_errors u16 | vbat_code u8(もと reserved) | CRC-8(poly 0x07, init 0, byte 0..26)
    flags: bit0-2 = CH1-3 受信中(100ms以内にパルスあり), bit3 = SELECT(1=AI走行側), bit4 = モーター回転中,
           bit5 = バイト 26 に電池の電圧あり(2026-10-25 の追加。古いファームは 0)
    erpm: 電気角rpm(符号=回転方向)。モーター軸rpm = erpm / 極対数
    vbat_code(バイト 26): 電池の電圧 = code × 0.05 V(1〜254 = 0.05〜12.70 V、255 = 12.75 V 以上、0 = 無し)。
      flags bit5 = 1 かつ code > 0 のときだけ使う。version は 1 のまま(上げると古い Jetson が読めない)。
      ファームの仕様: ~/jetformer_reports/speed_control_20261025/firmware_vbat_spec.md
    解読した dict の vbat_v [V] は較正の係数(data/vbat_calib.json の scale、無ければ 1.0)を掛けた値。電圧が無いときは
    NaN(走行ループは .get('vbat_v', nan) で数として読む)・vbat_ok False。画面・JSON に出す status() では None。
    **6.0〜9.0 V(2S)の外の値は「おかしい」とみて使わない**(vbat_ok False・vbat_v NaN。分圧の外れ・ADC の張り付きで
    走行のゲインを上げない。2026-10-25 fix.md)。そのときは vbat_bad True・vbat_raw_v に値(画面の「電圧の値がおかしい」)。

legacy (量産ファーム v1)
    レジスタ毎に write→read。v1は古い値がFIFOに残りチャネルが混ざるため、学習には使えない。
"""

import collections
import json
import math
import os
import struct
import threading
import time

from smbus2 import SMBus, i2c_msg

from .. import clock, config
from .base import Sample, SensorThread

SNAP_REG = 0x20
SNAP_LEN = 28
SNAP_MAGIC = 0xA5
SNAP_VERSION = 1
SNAP = struct.Struct('<BBHIHHHBBiiHBB')

FLAG_SELECT = 0x08
FLAG_MOVING = 0x10
FLAG_VBAT = 0x20                 # 1 = バイト 26 に電池の電圧(vbat_code)が入っている(2026-10-25)
VBAT_LSB_V = 0.05                # vbat_code 1 あたりの電圧 [V]
VBAT_CALIB_PATH = config.DATA_DIR / 'vbat_calib.json'      # 任意: {"scale": 1.0}(電圧計の値 ÷ 基板の値)
VBAT_SCALE_RANGE = (0.85, 1.15)  # 較正の係数として受け付ける範囲(分圧の抵抗 1% が 2 本で最大 約 ±2%。外れたら配線か入力の誤り)
VBAT_RANGE_V = (6.0, 9.0)        # 電圧として使う範囲 [V](2S。外れた値 = 分圧の外れ・張り付き。説明欄の V_before と同じ範囲)
NAN = float('nan')

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


def vbat_raw_from_code(flags, code, scale=1.0):
    """flags と予備のバイト → 電圧 [V](較正の後。範囲は見ない)。bit5 = 1 かつ code > 0 のときだけ、ほかは NaN"""
    if flags & FLAG_VBAT and code > 0:
        return code * VBAT_LSB_V * scale
    return NAN


def vbat_in_range(v):
    """電圧として使える値か(数で、VBAT_RANGE_V の中)"""
    lo, hi = VBAT_RANGE_V
    return v is not None and math.isfinite(v) and lo <= v <= hi


def vbat_from_code(flags, code, scale=1.0):
    """flags と予備のバイト → (電圧 [V] または NaN, 使えるか)。bit5 = 1 かつ code > 0 で、6.0〜9.0 V の中のときだけ電圧"""
    v = vbat_raw_from_code(flags, code, scale)
    if vbat_in_range(v):
        return v, True
    return NAN, False


def decode_snapshot(data, vbat_scale=1.0):
    """28バイト → dict。形式違い・CRC不一致は ValueError。vbat_scale = 電圧の較正の係数"""
    data = bytes(data)
    if len(data) != SNAP_LEN:
        raise ValueError('長さ %d != %d' % (len(data), SNAP_LEN))
    (magic, ver, seq, t_us, ch1, ch2, ch3, flags, hall_state,
     erpm, hall_count, hall_errors, vbat_code, crc) = SNAP.unpack(data)
    if magic != SNAP_MAGIC or ver != SNAP_VERSION:
        raise ValueError('magic/version 不一致 0x%02x/%d' % (magic, ver))
    if crc8(data[:-1]) != crc:
        raise ValueError('CRC 不一致')
    vbat_v, vbat_ok = vbat_from_code(flags, vbat_code, vbat_scale)
    vbat_raw = vbat_raw_from_code(flags, vbat_code, vbat_scale)
    return {
        'pwm': [ch1, ch2, ch3],                         # [steering, throttle, mode] µs
        'valid': [bool(flags & (1 << k)) for k in range(3)],
        'select': bool(flags & FLAG_SELECT),            # True = AI走行側(人の操縦データではない)
        'moving': bool(flags & FLAG_MOVING),
        'hall_state': hall_state, 'erpm': erpm, 'hall_count': hall_count, 'hall_errors': hall_errors,
        'seq': seq, 'board_t_us': t_us, 'quality': 'snapshot',
        'vbat_v': vbat_v, 'vbat_ok': vbat_ok,           # 電池の電圧 [V](較正の後。無い・6.0〜9.0 V の外なら NaN・False)
        'vbat_raw_v': vbat_raw,                         # 範囲を見ない値(画面の「電圧の値がおかしい」用。無ければ NaN)
        'vbat_bad': bool(math.isfinite(vbat_raw) and not vbat_ok),   # 基板は電圧を送ったが 6.0〜9.0 V の外
    }


def load_vbat_scale(path=None):
    """較正の係数 → (係数, 説明)。ファイルが無い・読めない・範囲の外なら 1.0(説明に理由)"""
    path = path or VBAT_CALIB_PATH
    try:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
    except FileNotFoundError:
        return 1.0, ''
    except (OSError, ValueError) as e:
        return 1.0, '較正のファイルを読めません(1.0 を使います): %s' % e
    try:
        x = float(doc.get('scale'))
    except (AttributeError, TypeError, ValueError):
        return 1.0, '較正のファイルに scale がありません(1.0 を使います)'
    lo, hi = VBAT_SCALE_RANGE
    if not (math.isfinite(x) and lo <= x <= hi):
        return 1.0, '較正の係数 %r が範囲 %.2f〜%.2f の外です(1.0 を使います)' % (doc.get('scale'), lo, hi)
    return x, ''


def save_vbat_scale(scale, meter_v=None, board_v=None, path=None):
    """較正の係数を書く(画面の「係数を保存」)。範囲の外は ValueError"""
    path = path or VBAT_CALIB_PATH
    x = float(scale)
    lo, hi = VBAT_SCALE_RANGE
    if not (math.isfinite(x) and lo <= x <= hi):
        raise ValueError('係数 %.3f は範囲 %.2f〜%.2f の外です(電圧計の値・配線を確かめてください)' % (x, lo, hi))
    doc = {'scale': round(x, 5), 'meter_v': meter_v, 'board_v': board_v,
           'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'note': '電圧計の値 ÷ 基板の値(止まっているとき)'}
    os.makedirs(os.path.dirname(str(path)) or '.', exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, str(path))
    return doc


def vbat_level(v, warn_v=7.90, bad_v=7.70):
    """電圧の色分け(qualify_plan 3.7): 'ok'(7.90 V 以上)・'warn'(7.70〜7.90 V)・'bad'(7.70 V 未満)・None(無し)"""
    if v is None or not math.isfinite(v):
        return None
    return 'ok' if v >= warn_v else ('warn' if v >= bad_v else 'bad')


class VbatTracker(object):
    """車両のサンプルの電圧と eRPM を時刻つきで覚える(走行の前後の電圧・走行中の最小。ハードウェアには触れない)。

    add(t, v, erpm): t = CLOCK_MONOTONIC [秒]、v = 電圧 [V](無しは NaN)。購読するときは on_sample を渡す。
    - still_median(now, span_s, min_n): 直近 span_s 秒が「止まっている」(|eRPM| の中央 < 50)で、使える電圧が min_n 以上なら、
      その中央値。そうでなければ None(理由は still_check)
    - bin_min(t0, t1): t0〜t1 の 1 秒ごとの中央値の最小(走行中の Vb_min)
    """
    STILL_ERPM = 50.0

    def __init__(self, keep_s=30.0, bin_s=1.0, keep_bins=1800):
        self.keep_s, self.bin_s = float(keep_s), float(bin_s)
        self.raw = collections.deque()           # (t, v, erpm)
        self.bins = collections.deque(maxlen=int(keep_bins))   # (始めの時刻, 中央値, 数)
        self._bin_t0, self._bin_vals = None, []
        self.seen = False                        # 電圧の値が一度でも来たか(基板が対応しているか)
        self.bad_seen = False                    # 6.0〜9.0 V の外の値が一度でも来たか(基板は送っている・値がおかしい)
        self.bad_last = None                     # 最後のおかしい値 (t, V)
        self.lock = threading.Lock()

    def on_sample(self, channel, sample):
        d = sample.data
        v = d.get('vbat_v', NAN) if d.get('vbat_ok') else NAN
        if d.get('vbat_bad'):
            self.note_bad(sample.t_ns / 1e9, d.get('vbat_raw_v', NAN))
        self.add(sample.t_ns / 1e9, v, d.get('erpm', 0))

    def note_bad(self, t, raw):
        """6.0〜9.0 V の外の値が来た(使わない。画面に「電圧の値がおかしい」を出すために覚えるだけ)"""
        with self.lock:
            self.bad_seen = True
            self.bad_last = (float(t), float(raw) if raw is not None else NAN)

    def bad_latest(self, now, max_age_s=2.0):
        """直近 max_age_s 秒のおかしい値 [V](無ければ None)"""
        with self.lock:
            b = self.bad_last
        if b is None or now - b[0] > max_age_s:
            return None
        return b[1]

    def add(self, t, v, erpm):
        t = float(t)
        v = float(v) if v is not None else NAN
        if not vbat_in_range(v):                 # 範囲の外は無いのと同じ(解読も外す。古い解読・偽物でも守る)
            v = NAN
        with self.lock:
            self.raw.append((t, v, float(erpm or 0)))
            while self.raw and self.raw[0][0] < t - self.keep_s:
                self.raw.popleft()
            if math.isfinite(v):
                self.seen = True
                if self._bin_t0 is None or t >= self._bin_t0 + self.bin_s or t < self._bin_t0:
                    self._flush()
                    self._bin_t0 = t
                self._bin_vals.append(v)

    def _flush(self):
        if self._bin_vals:
            vals = sorted(self._bin_vals)
            self.bins.append((self._bin_t0, _median_sorted(vals), len(vals)))
        self._bin_t0, self._bin_vals = None, []

    def _recent(self, t_from):
        """t_from 以後のサンプル(古い順)。右から見て止める(走行中に画面が 0.25 秒ごとに呼ぶので、30 秒ぶんを毎回見ない)"""
        out = []
        with self.lock:
            for r in reversed(self.raw):
                if r[0] < t_from:
                    break
                out.append(r)
        out.reverse()
        return out

    def latest(self, now, max_age_s=2.0, span_s=0.5):
        """最新の電圧(最後の値から span_s 秒の使える値の中央値)と、その古さ [秒]。無ければ (None, 古さ または None)"""
        rows = [(t, v) for t, v, _ in self._recent(now - max_age_s - span_s) if math.isfinite(v)]
        if not rows or now - rows[-1][0] > max_age_s:
            if rows or not self.seen:
                return None, (round(now - rows[-1][0], 1) if rows else None)
            with self.lock:                              # 古い値しか無い: 最後の値の古さだけ
                tl = next((t for t, v, _ in reversed(self.raw) if math.isfinite(v)), None)
            return None, (round(now - tl, 1) if tl is not None else None)
        tl = rows[-1][0]
        vals = sorted(v for t, v in rows if t >= tl - span_s)
        return _median_sorted(vals), round(max(now - tl, 0.0), 2)

    def still_check(self, now, span_s=2.0, min_n=20, fresh_s=0.5, q=0.5):
        """(中央値 [V] または None, 理由)。直近 span_s 秒がずっと記録にあり・止まっていて(|eRPM| の q 分位 < 50。
        走行の開始は q = 0.5 = 中央値、終わりは 0.95 = ほぼずっと)・使える値が min_n 以上"""
        rows = self._recent(now - span_s)
        if not rows or now - rows[-1][0] > fresh_s:
            return None, '車両の値が来ていません'
        if rows[0][0] > now - span_s * 0.9:
            return None, '車両の値が %.1f 秒ぶんありません' % span_s
        erpm = sorted(abs(r[2]) for r in rows)
        e_q = erpm[min(int(q * len(erpm)), len(erpm) - 1)] if q != 0.5 else _median_sorted(erpm)
        if e_q >= self.STILL_ERPM:
            return None, '車が止まっていません(|eRPM| の%s %.0f)' % ('中央' if q == 0.5 else ' %d%% 点' % round(q * 100), e_q)
        vals = sorted(r[1] for r in rows if math.isfinite(r[1]))
        if len(vals) < min_n:
            return None, ('基板の電圧がありません' if not vals else '基板の電圧の値が足りません(%d < %d)' % (len(vals), min_n))
        return _median_sorted(vals), ''

    def still_median(self, now, span_s=2.0, min_n=20, q=0.5):
        return self.still_check(now, span_s, min_n, q=q)[0]

    def bin_min(self, t0, t1, min_n=10):
        """t0〜t1 [秒] に始まった 1 秒ごとの中央値の最小(数が min_n 以上の秒だけ)。無ければ None"""
        with self.lock:
            bins = list(self.bins)
            if self._bin_vals:
                vals = sorted(self._bin_vals)
                bins.append((self._bin_t0, _median_sorted(vals), len(vals)))
        xs = [m for b0, m, n in bins if t0 <= b0 < t1 and n >= min_n]
        return min(xs) if xs else None


def _median_sorted(vals):
    n = len(vals)
    if not n:
        return None
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


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
        # 電池の電圧(2026-10-25。ファームが flags bit5 と予備のバイトで送るとき)
        self.vbat_scale, self.vbat_calib_note = load_vbat_scale(cfg.get('vbat_calib_path'))
        self.vbat_seen = False     # 電圧の値が一度でも来たか(ファームが対応しているか)
        self.vbat_last = None      # 最後に来た電圧 [V](較正の後)
        self.vbat_last_t = None    # その時刻(time.monotonic)
        self.vbat_bad_last = None  # 最後に来た 6.0〜9.0 V の外の値 [V](使わない。画面に出すだけ)
        self.vbat_bad_last_t = None

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
                values = decode_snapshot(data, self.vbat_scale)
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
        if values['vbat_ok']:
            self.vbat_seen = True
            self.vbat_last, self.vbat_last_t = values['vbat_v'], time.monotonic()
        elif values.get('vbat_bad'):
            self.vbat_bad_last, self.vbat_bad_last_t = values['vbat_raw_v'], time.monotonic()
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
            'hall_count': 0, 'hall_errors': 0, 'seq': self.count, 'board_t_us': 0, 'quality': 'legacy',
            'vbat_v': NAN, 'vbat_ok': False, 'vbat_raw_v': NAN, 'vbat_bad': False})]

    def _poll(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(self._next - now, 0.2))
        self._next = max(self._next + self.period, time.monotonic() - self.period)
        return self._poll_snapshot() if self.protocol == 'snapshot' else self._poll_legacy()

    def set_vbat_scale(self, scale):
        """較正の係数を替える(次のサンプルから)"""
        self.vbat_scale, self.vbat_calib_note = float(scale), ''

    def status(self):
        st = super(VehicleSensor, self).status()
        st.update(protocol=self.protocol, read_errors=self.read_errors, bus_errors=self.bus_errors,
                  skipped_polls=self.skipped_polls, last_read_error=self.last_read_error, stale=self.stale,
                  board_resets=self.board_resets,
                  board_reset_ago_s=round(time.time() - self.last_board_reset, 1) if self.last_board_reset else None)
        # 電圧(JSON に出すので、無いときは NaN ではなく None)。vbat_ok = 最新のサンプルに電圧があったか
        age = time.monotonic() - self.vbat_last_t if self.vbat_last_t is not None else None
        st.update(vbat_v=round(self.vbat_last, 3) if self.vbat_last is not None else None,
                  vbat_ok=bool(self.latest is not None and self.latest.data.get('vbat_ok')),
                  vbat_age_s=round(age, 2) if age is not None else None, vbat_supported=self.vbat_seen,
                  vbat_scale=self.vbat_scale, vbat_calib_note=self.vbat_calib_note)
        # 6.0〜9.0 V の外の値(使っていない)。直近 2 秒に来ていれば、その値
        bad_age = time.monotonic() - self.vbat_bad_last_t if self.vbat_bad_last_t is not None else None
        st.update(vbat_bad_v=round(self.vbat_bad_last, 3) if bad_age is not None and bad_age <= 2.0 else None,
                  vbat_range_v=list(VBAT_RANGE_V))
        if self.latest is not None:
            d = self.latest.data
            st.update(pwm=d['pwm'], erpm=d['erpm'], select=d['select'], valid=d['valid'])
        return st
