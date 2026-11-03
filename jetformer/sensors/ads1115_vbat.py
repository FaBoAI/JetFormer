"""電池の電圧の基板 BatteryTapADC v1.0(TI ADS1115・I2C 7・0x48)を読む(2026-10-31)

ユーザーの了承(10/31 18:57): (1) 電圧の記録 = 1 秒に約 50 回読み、ファイルに残す(tools/as5601_logger.py と同じ考え方:
CSV・30 分で新しいファイル・1 秒ごとに fsync・ディスクの空きが 2 GiB 未満なら書かずに待つ)(2) 管理画面で使う(走行の画面の電圧・
説明欄の V_before / V_after・収集のシーン・較正)。速さの補正(speed_auto)に使うことは、まだ了承が無い(この部品は値を出すだけ)。

基板(ユーザーの資料 BatteryTapADC_order.zip の配線): ADS1115 の 4 番 AIN0 = 電池を 33 kΩ / 10 kΩ で分けた電圧(× 4.3。1% の抵抗)
→ 1 kΩ と 100 nF を通る。AIN1〜AIN3 は GND。ADDR = GND で 0x48。VDD = 3.3 V(Jetson から)。ALERT/RDY は使わない。

設定(データシート SBAS444 の 9.6.3 Config Register):
  CONFIG_RUN = 0x4283 = OS 0 | MUX 100(AIN0 対 GND)| PGA 001(±4.096 V・1 LSB = 125 µV)| MODE 0(連続変換)
                        | DR 100(128 SPS)| COMP_MODE 0 | COMP_POL 0 | COMP_LAT 0 | COMP_QUE 11(比較器なし・ALERT/RDY は高インピーダンス)
  電源を入れた直後は 0x8583(MUX 000・PGA 010 ±2.048 V・1 回変換で休む)。比べるときは OS(bit15)を除く(読むと「変換中か」の印)
  連続変換なので、1 回の読みは変換レジスタ 0x00 の 2 バイトだけ(I2C の取引 1 つ・約 0.5 ms)。128 SPS = 7.8 ms ごとに新しい値に
  なるので、20 ms ごと(50 Hz)に読めばいつも 7.8 ms 以内の値。1 回変換(0xC383)だと、読むたびに書く・待つ・確かめるの 3 つ以上が要る
値: 変換レジスタは 2 の補数の 16 ビット(上位のバイトが先)。AIN0 [V] = raw × 125 µV。電池 [V](較正の前)= AIN0 × 4.3。
  0x7FFF・0x8000 は測れる幅の端(張り付き。sat)。較正の係数(data/vbat_calib.json の scale)は vehicle.py が掛ける(ここでは掛けない)
読む順(読み手のスレッド VBAT-ADS1115):
  - 始め・失敗のあと・CONFIG_CHECK_S(1 秒)ごと・変換の値がちょうど 0(電源が入り直すと 0x0000)のときに設定レジスタを読み、
    違っていたら(既定 0x8583 に戻った・ほかの道具が書いた)CONFIG_RUN を書き直して読み返す。書いたあと SETTLE_S は変換の値を使わない。
    0 での確かめは ZERO_CHECK_S(0.2 秒)に 1 回まで(電池を外すと AIN0 ≈ 0 V で 0 が続き、読むたびに取引が 2 倍になるため)
  - I2C の失敗(OSError 110 タイムアウト・121 応答なしなど)ではすぐ読み直さない: 0.1・0.2・0.4・… 秒(5 秒まで)あけてから。
    5 回続いたら SMBus を開き直す(バスには何も送らない)。バスが止まっても、詰めて読み直さない(10/31 18:15〜18:50 にバス 7 が
    「I2C transfer timed out」で止まり続けた)
  - 触るのは 0x48 だけ(アドレスは変えられない)。書くのは設定レジスタ 0x01 だけ
記録(記録のスレッド VBAT-LOG。読み手とは列でつなぐ = fsync で読みが止まらない): <out>/<YYYYMMDD-HHMMSS>.csv と同じ名前の .json。
  CSV の欄 FIELDS: epoch_us(UNIX µs)・mono_us(CLOCK_MONOTONIC µs = 収集・走行の記録の時計)・raw・ain0_v・board_v(× 4.3、
  較正の前 = 画面の較正の「基板の値」)・scale(その時の較正の係数)・vbat_v(board_v × scale)・ok(1 = 6.0〜9.0 V の中で張り付きでない)・cfg(その回に
  読んだ設定レジスタ。読まなかった回は空)・err(続いた失敗の数。読めた回は 0)・note(設定を書き直した・失敗の種類・sat)。
  読めなかった回は raw などが空。30 分(ROTATE_S)ごとに新しいファイル・1 秒(FSYNC_S)ごとに flush と fsync・空きが 2 GiB 未満
  なら書かずに待つ(60 秒ごとに見直す。その間の行は捨てて数える)。書けない(ENOSPC・EROFS・EIO など)も 60 秒待つ。消す決まりは無い
始めと終わり: vehicle.VehicleSensor の start・stop と同じ = 読むのも記録するのもセンサーが動いている間だけ(画面の「センサー開始」・
  収集・走行・点検で始まり、「センサー停止」と jetformer-web の終わりの処理 hub.stop() で止まる。起動しただけでは読まない)。
  start・stop は鍵で 1 つずつ。読み手のスレッドが動いていれば start は何もしない(何度呼んでも 1 組)。前の stop の待ちが時間切れで
  記録のスレッドだけが残っていたら(SD カードの fsync の待ちなど)、新しい列・新しいファイルで始める(残った方は自分の行を書いて閉じる)。
  環境変数 JETFORMER_VBAT_ADS1115=0(off・false・no)で使わない(make_reader が None を返す。前と同じ動き)
"""

import collections
import csv
import errno
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime

BUS = 7                          # I2C の 7 番(40 ピンの 3・5 番。PCA9685・制御基板・AS5601・表示板と同じ線)
ADDR = 0x48                      # ADS1115 の ADDR = GND(変えられない)
REG_CONV, REG_CONFIG = 0x00, 0x01
CONFIG_DEFAULT = 0x8583          # 電源を入れた直後(データシートの既定。10/31 18:54 に本物で読んだ値)
MUX_AIN0_GND, PGA_4V096, MODE_CONT, DR_128, COMP_OFF = 0b100, 0b001, 0, 0b100, 0b11
CONFIG_MASK = 0x7FFF             # 比べるときは OS(bit15)を除く
FSR_V = 4.096                    # PGA 001 = ±4.096 V
LSB_V = FSR_V / 32768            # 125 µV
DIVIDER = (33.0 + 10.0) / 10.0   # 分圧 R1 33 kΩ / R2 10 kΩ = × 4.3(基板の印字「DIV 1/4.3 (33k/10k)」)
DR_SPS = 128
RATE_HZ = 50.0                   # 1 秒に読む回数
CONFIG_CHECK_S = 1.0             # 設定レジスタを確かめる間
ZERO_CHECK_S = 0.2               # 変換の値がちょうど 0 のときの設定の確かめは、この間に 1 回まで(電池を外して 0 が続くとき)
SETTLE_S = 0.02                  # 設定を書いてから最初の変換が終わるまで(1/128 秒 = 7.8 ms・時計のずれ ±10%)
BACKOFF_FIRST_S = 0.1            # I2C の失敗のあとの間(続けて失敗したら 2 倍ずつ・BACKOFF_MAX_S まで)
BACKOFF_MAX_S = 5.0
REOPEN_EVERY = 5                 # 続けてこの回数失敗したら SMBus を開き直す
WARN_EVERY_S = 300.0             # 続く失敗の知らせ(journal)は 5 分に 1 回まで
ROTATE_S = 1800.0                # 30 分で新しいファイル
FSYNC_S = 1.0                    # 1 秒ごとに flush と fsync
MIN_FREE = 2 * 1024 ** 3         # 空きがこれより少なければ書かずに待つ(収集の記録と同じディスク)
FREE_CHECK_S = 10.0              # 書いている間も空きを見る間
DISK_WAIT_S = 60.0               # 空きが足りない・書けないときに待つ間
QUEUE_MAX = 3000                 # 読み手 → 記録の列(60 秒ぶん。あふれたら捨てて数える)
LEFTOVER_JOIN_S = 0.2            # start のとき、前の stop の待ちを越えて残った記録のスレッドを待つ長さ(越えたら新しいファイルで始める)
RANGE_V = (6.0, 9.0)             # ok の欄の範囲(vehicle.VBAT_RANGE_V と同じ。vehicle.py が渡す)
ENV = 'JETFORMER_VBAT_ADS1115'
FIELDS = ['epoch_us', 'mono_us', 'raw', 'ain0_v', 'board_v', 'scale', 'vbat_v', 'ok', 'cfg', 'err', 'note']
FIELD_NOTES = {
    'epoch_us': 'UNIX 時刻 µs(壁の時計)',
    'mono_us': 'CLOCK_MONOTONIC µs(収集・走行の記録と同じ時計。同じ boot_id の中だけ並べられる)',
    'raw': '変換レジスタ(2 の補数 −32768〜32767。32767・−32768 は端に張り付いた印。読めなかった行は空)',
    'ain0_v': 'AIN0 の電圧 [V] = raw × 125 µV',
    'board_v': '電池の電圧 [V](較正の前)= ain0_v × 4.3(画面の較正の「基板の値」)',
    'scale': 'その時の較正の係数(data/vbat_calib.json。無ければ 1.0)',
    'vbat_v': '電池の電圧 [V] = board_v × scale(画面・説明欄・収集が使う値)',
    'ok': '1 = 6.0〜9.0 V の中で張り付きでない(使える値)',
    'cfg': 'その回に読んだ設定レジスタ(16 進。読まなかった回は空)',
    'err': '続いた失敗の数(読めた行は 0)',
    'note': 'config・reconfig(設定を書いた・書き直した)・失敗の種類と errno・sat(張り付き)',
}

MUX_NAMES = {0b000: 'AIN0-AIN1', 0b001: 'AIN0-AIN3', 0b010: 'AIN1-AIN3', 0b011: 'AIN2-AIN3',
             0b100: 'AIN0-GND', 0b101: 'AIN1-GND', 0b110: 'AIN2-GND', 0b111: 'AIN3-GND'}
PGA_V = {0b000: 6.144, 0b001: 4.096, 0b010: 2.048, 0b011: 1.024, 0b100: 0.512, 0b101: 0.256, 0b110: 0.256, 0b111: 0.256}
DR_NAMES = {0b000: 8, 0b001: 16, 0b010: 32, 0b011: 64, 0b100: 128, 0b101: 250, 0b110: 475, 0b111: 860}


def config_word(os_bit=0, mux=MUX_AIN0_GND, pga=PGA_4V096, mode=MODE_CONT, dr=DR_128, comp_mode=0, comp_pol=0,
                comp_lat=0, comp_que=COMP_OFF):
    """設定レジスタの 16 ビット(データシートのビットの並び)"""
    return ((os_bit & 1) << 15 | (mux & 7) << 12 | (pga & 7) << 9 | (mode & 1) << 8 | (dr & 7) << 5
            | (comp_mode & 1) << 4 | (comp_pol & 1) << 3 | (comp_lat & 1) << 2 | (comp_que & 3))


CONFIG_RUN = config_word()       # 0x4283(連続変換・AIN0 対 GND・±4.096 V・128 SPS・比較器なし)


def describe_config(word):
    """設定レジスタ → 読める形(README・状態の表示・試験)"""
    w = int(word) & 0xFFFF
    return {'os': w >> 15 & 1, 'mux': MUX_NAMES[w >> 12 & 7], 'pga_v': PGA_V[w >> 9 & 7],
            'mode': 'single' if w >> 8 & 1 else 'continuous', 'dr_sps': DR_NAMES[w >> 5 & 7],
            'comp_mode': w >> 4 & 1, 'comp_pol': w >> 3 & 1, 'comp_lat': w >> 2 & 1,
            'comp_que': 'disabled' if (w & 3) == 3 else 'assert after %d' % ((w & 3) + 1 if (w & 3) < 2 else 4)}


def config_matches(word):
    """読んだ設定レジスタが CONFIG_RUN と同じか(OS を除く)"""
    return word is not None and (int(word) & CONFIG_MASK) == (CONFIG_RUN & CONFIG_MASK)


def to_int16(word):
    """変換レジスタの 16 ビット → 2 の補数の数(−32768〜32767)"""
    w = int(word) & 0xFFFF
    return w - 0x10000 if w & 0x8000 else w


def raw_to_ain0(raw):
    """raw → AIN0 の電圧 [V](±4.096 V の 1 LSB = 125 µV)"""
    return int(raw) * LSB_V


def raw_to_vbat(raw):
    """raw → 電池の電圧 [V](分圧 × 4.3。較正の係数を掛ける前)"""
    return raw_to_ain0(raw) * DIVIDER


def is_saturated(raw):
    """測れる幅の端(0x7FFF・0x8000)。電池の電圧ではありえない(4.096 V × 4.3 = 17.6 V)= 配線・PGA の外れ"""
    return int(raw) >= 0x7FFF or int(raw) <= -0x8000


def backoff_s(fail_streak):
    """続けて fail_streak 回失敗したあとに、次に読むまであける間 [秒](0.1・0.2・0.4・…・5.0)"""
    k = max(int(fail_streak), 1) - 1
    return min(BACKOFF_MAX_S, BACKOFF_FIRST_S * (2 ** min(k, 16)))


def enabled_by_env(environ=None):
    """JETFORMER_VBAT_ADS1115 が 0・off・false・no なら False(既定は入)"""
    v = (os.environ if environ is None else environ).get(ENV, '1')
    return str(v).strip().lower() not in ('0', 'off', 'false', 'no')


def free_bytes(path):
    """path(無ければ、ある親)のディスクの空き [バイト]"""
    p = os.path.abspath(str(path))
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def boot_id():
    try:
        with open('/proc/sys/kernel/random/boot_id') as fh:
            return fh.read().strip()
    except OSError:
        return None


def _free(path):
    return free_bytes(path)          # 呼んだときの free_bytes(試験が差しかえられる)


def _fsync(fd):
    return os.fsync(fd)


def _smbus(bus_no):
    """本物の SMBus(読み手が始めてから最初に読むときだけ作る。import・作るだけではバスを開かない)"""
    from smbus2 import SMBus
    return SMBus(bus_no)


def _say(msg):
    print(msg, flush=True)


class CsvLog(object):
    """CSV の記録(呼んだスレッドだけが書く)。30 分ごとに新しいファイル・1 秒ごとに flush と fsync・空きが 2 GiB 未満なら書かずに待つ。
    write・tick・close は例外を出さない(書けないときは閉じて DISK_WAIT_S 待つ。その間の行は捨てて数える)"""

    def __init__(self, out_dir, meta=None, rotate_s=ROTATE_S, fsync_s=FSYNC_S, min_free=MIN_FREE,
                 free_check_s=FREE_CHECK_S, disk_wait_s=DISK_WAIT_S, clock=time, free_bytes=None,
                 fsync=None, printer=None):
        self.out_dir = str(out_dir)
        self.meta = dict(meta or {})
        self.rotate_s, self.fsync_s, self.min_free = float(rotate_s), float(fsync_s), int(min_free)
        self.free_check_s, self.disk_wait_s = float(free_check_s), float(disk_wait_s)
        self.clock, self.free_bytes, self.fsync = clock, free_bytes or _free, fsync or _fsync
        self.printer = printer or _say
        self.fh = self.w = self.path = None
        self.t_open = self.t_sync = self.t_free = 0.0
        self.pending = False               # fsync していない行がある
        self.wait_until = None             # 空きが足りない・書けない: この時刻まで書かない
        self.waiting_disk = False
        self.files = []                    # 書いたファイル(書いた順)
        self.rows = self.dropped = self.syncs = 0
        self.last_error = ''
        self.last_free = None

    def _name(self):
        name = datetime.fromtimestamp(self.clock.time()).strftime('%Y%m%d-%H%M%S')
        base, k = name, 2
        while os.path.exists(os.path.join(self.out_dir, name + '.csv')):      # 同じ秒に区切ったときに上書きしない
            name = '%s-%d' % (base, k)
            k += 1
        return name

    def _open(self, now):
        os.makedirs(self.out_dir, exist_ok=True)
        name = self._name()
        meta = dict(self.meta, name=name, start_epoch_us=int(self.clock.time() * 1e6), boot_id=boot_id(),
                    rt_minus_mono_us=int((self.clock.time() - self.clock.monotonic()) * 1e6), fields=FIELDS,
                    rotate_s=self.rotate_s, fsync_s=self.fsync_s, min_free_bytes=self.min_free)
        with open(os.path.join(self.out_dir, name + '.json'), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
            fh.flush()
            self.fsync(fh.fileno())
        path = os.path.join(self.out_dir, name + '.csv')
        self.fh = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.writer(self.fh, lineterminator='\n')
        self.w.writerow(FIELDS)
        self.path = path
        self.t_open, self.t_sync, self.t_free = now, now + self.fsync_s, now + self.free_check_s
        self.pending = True
        self.files.append(path)

    def _sync(self, now):
        self.fh.flush()
        self.fsync(self.fh.fileno())
        self.syncs += 1
        self.pending = False
        self.t_sync = now + self.fsync_s

    def _close(self):
        fh, self.fh, self.w = self.fh, None, None
        if fh is None:
            return
        try:
            fh.flush()
            self.fsync(fh.fileno())
            self.syncs += 1
        finally:
            self.pending = False
            fh.close()

    def _wait(self, now, why):
        if not self.waiting_disk:
            self.printer('電池の電圧の記録を止めて待ちます(%s)。%.0f 秒ごとに見直します' % (why, self.disk_wait_s))
        self.waiting_disk = True
        self.wait_until = now + self.disk_wait_s

    def _enough_space(self, now):
        try:
            self.last_free = int(self.free_bytes(self.out_dir))
        except OSError as e:
            self.last_error = '空きを読めません: %s' % e
            self._wait(now, self.last_error)
            return False
        if self.last_free < self.min_free:
            self._wait(now, 'ディスクの空き %.2f GiB < %.0f GiB' % (self.last_free / 1024 ** 3, self.min_free / 1024 ** 3))
            return False
        return True

    def write(self, row):
        """1 行書く(row = FIELDS の順の値)。書いたら True・書かずに捨てたら False"""
        now = self.clock.monotonic()
        if self.wait_until is not None and now < self.wait_until:
            self.dropped += 1
            return False
        try:
            if self.fh is not None and now - self.t_open >= self.rotate_s:          # 30 分: 新しいファイル
                self._close()
            if self.fh is not None and now >= self.t_free:                           # 書いている間も空きを見る
                self.t_free = now + self.free_check_s
                if not self._enough_space(now):
                    self._close()
                    self.dropped += 1
                    return False
            if self.fh is None:
                if not self._enough_space(now):
                    self.dropped += 1
                    return False
                self._open(now)
                if self.waiting_disk:
                    self.printer('電池の電圧の記録を続けます: %s' % self.path)
                self.waiting_disk, self.wait_until = False, None
            self.w.writerow(row)
            self.rows += 1
            self.pending = True
            if now >= self.t_sync:
                self._sync(now)
            return True
        except Exception as e:                    # 書けない(ENOSPC・EROFS・EIO・EDQUOT など): 閉じて待つ(読み手は止めない)
            self.last_error = '%s: %s' % (type(e).__name__, e)
            try:
                self._close()
            except Exception:
                self.fh = self.w = None
            self._wait(now, '書けません: %s' % self.last_error)
            self.dropped += 1
            return False

    def tick(self):
        """行が来なくても 1 秒ごとに fsync する(読みが失敗して間があいているとき)"""
        now = self.clock.monotonic()
        if self.fh is not None and self.pending and now >= self.t_sync:
            try:
                self._sync(now)
            except Exception as e:
                self.last_error = '%s: %s' % (type(e).__name__, e)
                try:
                    self._close()
                except Exception:
                    self.fh = self.w = None
                self._wait(now, '書けません: %s' % self.last_error)

    def close(self):
        try:
            self._close()
        except Exception as e:
            self.last_error = '%s: %s' % (type(e).__name__, e)
            self.fh = self.w = None

    def status(self):
        return {'enabled': True, 'dir': self.out_dir, 'file': self.path if self.fh is not None else None,
                'files': len(self.files), 'rows': self.rows, 'dropped': self.dropped, 'syncs': self.syncs,
                'waiting_disk': self.waiting_disk, 'last_error': self.last_error,
                'free_gib': None if self.last_free is None else round(self.last_free / 1024 ** 3, 2)}


class Ads1115Vbat(object):
    """ADS1115(0x48)を読む人(読み手と記録の 2 つのスレッド)。作るだけではバスを開かない・スレッドも始めない。

    fresh(max_age_s) → 最新の値 {'t'(time.monotonic の秒), 'raw', 'ain0_v', 'board_v'(× 4.3・係数の前), 'sat'} か None。
    step(now) は 1 回ぶん(スレッドと試験が呼ぶ)。次に読む時刻は next_t。bus_factory・clock・free_bytes・fsync・printer は試験のため
    (既定は smbus2.SMBus・time・本物のディスク・os.fsync・print)"""

    def __init__(self, out_dir=None, bus_no=BUS, hz=RATE_HZ, log=True, bus_factory=None, scale_fn=None, range_v=RANGE_V,
                 clock=time, free_bytes=None, fsync=None, printer=None, **log_kw):
        self.bus_no = int(bus_no)
        self.period = 1.0 / float(hz)
        self.bus_factory = bus_factory
        self.scale_fn = scale_fn
        self.range_v = tuple(range_v)
        self.clock = clock
        self.printer = printer or _say
        self.log = None
        if log and out_dir is not None:
            meta = {'addr': '0x%02x' % ADDR, 'bus': self.bus_no, 'hz': float(hz), 'config': '0x%04X' % CONFIG_RUN,
                    'config_meaning': describe_config(CONFIG_RUN), 'lsb_v': LSB_V, 'divider': DIVIDER,
                    'range_v': list(self.range_v), 'field_notes': FIELD_NOTES,
                    'note': '電池の電圧の基板 BatteryTapADC(ADS1115)を jetformer-web が読んだ記録(jetformer/sensors/ads1115_vbat.py)。'
                            'board_v = raw × 125 µV × 4.3(較正の前)・vbat_v = board_v × scale(画面・説明欄・収集が使う値)。'
                            '走りの記録と合わせるときは mono_us(CLOCK_MONOTONIC = 収集の時計。同じ boot_id の中だけ)'}
            self.log = CsvLog(out_dir, meta=meta, clock=clock, free_bytes=free_bytes, fsync=fsync, printer=self.printer,
                              **log_kw)
        self._queue = queue.Queue(maxsize=QUEUE_MAX)
        self._life = threading.Lock()      # start・stop を 1 つずつ(走行のスレッドと API のスレッドが同時に呼んでも 1 組だけ)
        self._stop = threading.Event()     # 合図は start のたびに新しく作る(残った記録のスレッドが次の回の印を見ないように)
        self._reader_done = threading.Event()
        self._reader = self._writer = None
        self._threads = []
        self.stale_writers = 0             # 前の stop の待ちを越えて残った記録のスレッドを置いて、新しいファイルで始めた回数
        self.bus = None
        self.configured = False            # 設定レジスタが CONFIG_RUN だと確かめた(失敗・開き直しで False)
        self.configured_once = False       # 一度でも設定できた(このあとの食い違い = 電源が入り直した・リセット)
        self.next_t = 0.0                  # 次に読む時刻
        self.next_check = 0.0              # 次に設定レジスタを確かめる時刻
        self.next_zero_check = -1e18       # 変換の値が 0 のときの設定の確かめを、次にしてよい時刻(ZERO_CHECK_S に 1 回まで)
        self.settle_until = -1e18          # 設定を書いたあと、この時刻まで変換の値を使わない
        self.fail_streak = 0
        self.backoff = 0.0
        self._latest = None
        self.reads = self.errors = self.reconfigs = self.config_writes = self.saturated = self.opens = 0
        self.dropped_queue = 0
        self.last_error = ''
        self.last_cfg = None
        self.t_warn = -1e18
        self._times = collections.deque(maxlen=100)

    # ------------------------------------------------------------ 値
    def fresh(self, max_age_s):
        a = self._latest
        if a is None:
            return None
        age = self.clock.monotonic() - a['t']
        return a if -0.05 <= age <= max_age_s else None

    def _scale(self):
        try:
            x = float(self.scale_fn()) if self.scale_fn is not None else 1.0
        except Exception:
            return 1.0
        return x if x == x else 1.0

    def rate_hz(self, now=None):
        times = list(self._times)
        now = self.clock.monotonic() if now is None else now
        if len(times) < 2 or now - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / max(times[-1] - times[0], 1e-6)

    # ------------------------------------------------------------ I2C(0x48 だけ)
    def _read16(self, reg):
        d = self.bus.read_i2c_block_data(ADDR, reg, 2)
        if d is None or len(d) != 2:
            raise OSError(errno.EIO, '0x%02x のレジスタ 0x%02x から 2 バイト読めません: %r' % (ADDR, reg, d))
        return (int(d[0]) & 0xFF) << 8 | (int(d[1]) & 0xFF)

    def _write16(self, reg, word):
        self.bus.write_i2c_block_data(ADDR, reg, [(word >> 8) & 0xFF, word & 0xFF])

    def _close_bus(self):
        bus, self.bus = self.bus, None
        self.configured = False
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass

    def _check_config(self, now):
        """設定レジスタを読み、CONFIG_RUN でなければ書き直して読み返す → (cfg の欄, note の欄)"""
        cfg = self._read16(REG_CONFIG)
        self.last_cfg = cfg
        self.next_check = now + CONFIG_CHECK_S
        if config_matches(cfg):
            self.configured = self.configured_once = True
            return '%04X' % cfg, ''
        again = self.configured_once
        self._write16(REG_CONFIG, CONFIG_RUN)
        self.config_writes += 1
        back = self._read16(REG_CONFIG)
        self.last_cfg = back
        if not config_matches(back):
            raise OSError(errno.EIO, '設定レジスタに 0x%04X を書いたのに 0x%04X が読めました' % (CONFIG_RUN, back))
        self.configured = self.configured_once = True
        self.settle_until = now + SETTLE_S
        if again:
            self.reconfigs += 1
            self.printer('電池の電圧の基板(0x48)の設定が 0x%04X になっていました(電源が入り直した・リセット・ほかの道具が書いた)。'
                         '0x%04X を書き直しました(%d 回目)' % (cfg, CONFIG_RUN, self.reconfigs))
        return '%04X' % back, ('reconfig %04X->%04X' % (cfg, CONFIG_RUN) if again else 'config %04X->%04X' % (cfg, CONFIG_RUN))

    def step(self, now=None):
        """1 回ぶん: (開く)→(設定を確かめる)→ 変換レジスタを読む。失敗したら間をあける(next_t)。例外は出さない(OSError)"""
        now = self.clock.monotonic() if now is None else now
        cfg_txt, note = '', ''
        try:
            if self.bus is None:
                self.bus = (self.bus_factory or _smbus)(self.bus_no)
                self.opens += 1
                self.configured = False
            if not self.configured or now >= self.next_check:
                cfg_txt, note = self._check_config(now)
            if now < self.settle_until:                     # 設定を書いた直後: 最初の変換が終わるまで読まない
                self._ok(now)
                self.next_t = self.settle_until
                self._row(now, None, cfg_txt, note)
                return
            raw = to_int16(self._read16(REG_CONV))
            if raw == 0 and not cfg_txt and now >= self.next_zero_check:
                # 電源が入り直した直後の変換レジスタは 0x0000: 設定を確かめてから使う。0 が続く(電池を外した = AIN0 ≈ 0 V)
                # ときは ZERO_CHECK_S に 1 回まで(読むたびに確かめると ADS1115 の取引が約 2 倍になる)
                self.next_zero_check = now + ZERO_CHECK_S
                cfg_txt, note = self._check_config(now)
                if now < self.settle_until:
                    self._ok(now)
                    self.next_t = self.settle_until
                    self._row(now, None, cfg_txt, note)
                    return
        except OSError as e:
            self._failed(now, e)
            return
        t = self.clock.monotonic()
        sat = is_saturated(raw)
        self.reads += 1
        if sat:
            self.saturated += 1
            note = (note + ' sat').strip()
        self._latest = {'t': t, 'raw': raw, 'ain0_v': raw_to_ain0(raw), 'board_v': raw_to_vbat(raw), 'sat': sat}
        self._times.append(t)
        self._ok(now)
        self.next_t += self.period
        if self.next_t <= now:                              # 遅れた(GIL・I2C の待ち): 追いつこうとせず、今から数え直す
            self.next_t = now + self.period
        self._row(t, raw, cfg_txt, note)

    def _ok(self, now):
        if self.fail_streak:
            self.printer('電池の電圧の基板(0x48)を読めるようになりました(続けて %d 回失敗したあと)' % self.fail_streak)
        self.fail_streak = 0
        self.backoff = 0.0

    def _failed(self, now, e):
        self.fail_streak += 1
        self.errors += 1
        self.last_error = '%s: %s' % (type(e).__name__, e)
        self.configured = False              # 戻ったら、まず設定レジスタを確かめる(基板の電源が入り直したかもしれない)
        self.backoff = backoff_s(self.fail_streak)
        self.next_t = now + self.backoff     # すぐ読み直さない
        if self.fail_streak % REOPEN_EVERY == 0:
            self._close_bus()                # 次に読むときに開き直す(バスには何も送らない)
        code = getattr(e, 'errno', None)
        self._row(now, None, '', '%s%s' % (type(e).__name__, '' if code is None else ' %s' % code), err=self.fail_streak)
        if self.fail_streak == 1 or now - self.t_warn >= WARN_EVERY_S:
            self.t_warn = now
            self.printer('電池の電圧の基板(I2C %d・0x%02x)を読めません(%s)。%.1f 秒あけて読み直します(続けて %d 回目)'
                         % (self.bus_no, ADDR, self.last_error, self.backoff, self.fail_streak))

    def _row(self, t, raw, cfg_txt='', note='', err=0):
        if self.log is None:
            return
        if raw is None:
            vals = ['', '', '', '', '', '']
        else:
            vraw = raw_to_vbat(raw)
            scale = self._scale()
            v = vraw * scale
            ok = (not is_saturated(raw)) and self.range_v[0] <= v <= self.range_v[1]
            vals = [raw, '%.6f' % raw_to_ain0(raw), '%.4f' % vraw, '%.5f' % scale, '%.4f' % v, 1 if ok else 0]
        row = [int(self.clock.time() * 1e6), int(t * 1e6)] + vals + [cfg_txt, err, note]
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped_queue += 1

    def drain(self):
        """列に溜まった行を全部書く(記録のスレッド・試験)。書いた数"""
        n = 0
        while True:
            try:
                row = self._queue.get_nowait()
            except queue.Empty:
                return n
            if self.log is not None:
                self.log.write(row)
            n += 1

    # ------------------------------------------------------------ スレッド
    @property
    def running(self):
        """読み手のスレッドが動いているか(記録のスレッドは log_running)"""
        return self._reader is not None and self._reader.is_alive()

    @property
    def log_running(self):
        return self._writer is not None and self._writer.is_alive()

    @staticmethod
    def _new_log(like):
        """like と同じ場所・同じ決まりの新しい記録(残った記録のスレッドとファイルを分ける。名前は始めた時刻)"""
        return CsvLog(like.out_dir, meta=like.meta, rotate_s=like.rotate_s, fsync_s=like.fsync_s, min_free=like.min_free,
                      free_check_s=like.free_check_s, disk_wait_s=like.disk_wait_s, clock=like.clock,
                      free_bytes=like.free_bytes, fsync=like.fsync, printer=like.printer)

    def start(self):
        """始める → 始めたら True。読み手のスレッドが動いていれば何もしない(False。何度呼んでも 1 組)。start・stop は鍵で
        1 つずつ(走行のスレッドと API のスレッドが同時に hub.start を呼んでも 1 組だけ)。前の stop の待ちを越えて記録のスレッド
        だけが残っていたら(SD カードの fsync の待ちなど)、LEFTOVER_JOIN_S 待ち、まだ残っていれば新しい列・新しい記録
        (新しいファイル)で始める。残った方は自分の列の行を書き終えて自分のファイルを閉じる(同じファイルに 2 つで書かない)"""
        with self._life:
            if self.running:
                return False
            old = self._writer
            if old is not None and old.is_alive():
                old.join(LEFTOVER_JOIN_S)
                if old.is_alive():
                    self.stale_writers += 1
                    self._queue = queue.Queue(maxsize=QUEUE_MAX)
                    self.log = self._new_log(self.log)
                    self.printer('電池の電圧の記録のスレッドが、前に止めたときから残っています(ディスクの待ちなど)。'
                                 '新しいファイルで記録を始めます(残った方は前のファイルを書き終えて閉じます。%d 回目)'
                                 % self.stale_writers)
            stop_ev, done_ev = threading.Event(), threading.Event()
            self._stop, self._reader_done = stop_ev, done_ev
            self.configured = False
            self.next_t = self.clock.monotonic()
            reader = threading.Thread(target=self._run, args=(stop_ev, done_ev), name='VBAT-ADS1115', daemon=True)
            writer = None
            if self.log is not None:
                writer = threading.Thread(target=self._write_loop, args=(self._queue, self.log, done_ev), name='VBAT-LOG',
                                          daemon=True)
            self._reader, self._writer = reader, writer
            self._threads = [th for th in (reader, writer) if th is not None]
            for th in self._threads:
                th.start()
        self.printer('電池の電圧の基板(ADS1115・I2C %d・0x%02x)を読みます: 連続変換 0x%04X(AIN0 対 GND・±4.096 V・128 SPS)・'
                     '1 秒に約 %.0f 回・記録 %s' % (self.bus_no, ADDR, CONFIG_RUN, 1.0 / self.period,
                                                   self.log.out_dir if self.log is not None else 'なし'))
        return True

    def request_stop(self):
        """止める合図だけ(待たない。車両の stop が先に呼ぶ)"""
        self._stop.set()

    def stop(self, timeout=3.0):
        """止める(読み手が終わってから、記録が残りの行を書いて閉じる)。timeout は 1 つのスレッドを待つ長さ"""
        with self._life:
            self._stop.set()
            for th in list(self._threads):
                th.join(timeout)
            self._threads = [th for th in self._threads if th.is_alive()]

    def _run(self, stop_ev=None, done_ev=None):
        """読み手のスレッド: next_t まで待って step。stop_ev で抜けて SMBus を閉じ、done_ev を立てる(記録のスレッドへの印)"""
        stop_ev = self._stop if stop_ev is None else stop_ev
        done_ev = self._reader_done if done_ev is None else done_ev
        try:
            while not stop_ev.is_set():
                now = self.clock.monotonic()
                if now < self.next_t:
                    stop_ev.wait(min(self.next_t - now, 0.5))
                    continue
                try:
                    self.step(now)
                except Exception as e:             # 思わぬ失敗(壊れた値など)でもスレッドは止めない。間をあけて読み直す
                    self._failed(now, e)
        finally:
            self._close_bus()
            done_ev.set()

    def _write_loop(self, q=None, log=None, done_ev=None):
        """記録のスレッド: 列の行を書く。行が来ない 0.2 秒ごとに tick(行が来なくても 1 秒ごとに fsync)。読み手が終わった印
        (done_ev)のあと、列が空になってから閉じて終わる(列に残った行は全部書く。読み手が間をあけている間は終わらない)"""
        q = self._queue if q is None else q
        log = self.log if log is None else log
        done_ev = self._reader_done if done_ev is None else done_ev
        try:
            while True:
                try:
                    row = q.get(timeout=0.2)
                except queue.Empty:
                    if done_ev.is_set():
                        break
                    log.tick()
                    continue
                log.write(row)
        finally:
            log.close()

    def status(self):
        """画面・GET /api/status 用(JSON にするので NaN を出さない)"""
        a = self._latest
        now = self.clock.monotonic()
        return {'enabled': True, 'running': self.running, 'log_running': self.log_running,
                'stale_writers': self.stale_writers, 'bus': self.bus_no, 'addr': '0x%02x' % ADDR,
                'config': '0x%04X' % CONFIG_RUN, 'config_read': None if self.last_cfg is None else '0x%04X' % self.last_cfg,
                'configured': bool(self.configured), 'hz': round(1.0 / self.period, 1), 'rate_hz': round(self.rate_hz(now), 1),
                'reads': self.reads, 'errors': self.errors, 'fail_streak': self.fail_streak, 'backoff_s': round(self.backoff, 2),
                'reconfigs': self.reconfigs, 'config_writes': self.config_writes, 'saturated': self.saturated,
                'opens': self.opens, 'last_error': self.last_error, 'dropped_queue': self.dropped_queue,
                'raw': a['raw'] if a else None, 'ain0_v': round(a['ain0_v'], 6) if a else None,
                'board_v': round(a['board_v'], 4) if a else None, 'sat': bool(a['sat']) if a else None,
                'age_s': round(now - a['t'], 3) if a else None, 'divider': DIVIDER, 'lsb_v': LSB_V,
                'log': self.log.status() if self.log is not None else {'enabled': False}}


def make_reader(out_dir, scale_fn=None, range_v=RANGE_V, environ=None, **kw):
    """読み手を作る(まだ始めない・バスも開かない)。環境変数 JETFORMER_VBAT_ADS1115 が 0・off・false・no なら None"""
    if not enabled_by_env(environ):
        return None
    return Ads1115Vbat(out_dir=out_dir, scale_fn=scale_fn, range_v=range_v, **kw)
