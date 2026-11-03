"""電池の電圧の基板 BatteryTapADC v1.0(TI ADS1115・I2C 7・0x48)を jetformer-web に入れた直し(2026-10-31)の試験。
jetformer/sensors/ads1115_vbat.py(読み手・記録)と jetformer/sensors/vehicle.py のつなぎ(VehicleSensor が vbat_* を埋める)。

ユーザーの了承(10/31 18:57): (1) 電圧の記録(約 50 Hz・CSV・30 分で新しいファイル・1 秒ごとに fsync・空きが 2 GiB 未満なら書かずに
待つ)(2) 管理画面で使う(走行の画面の電圧・説明欄の V_before / V_after の自動・収集のシーン・較正)。(3) 速さの補正(speed_auto)に
使うのは了承が無い → speed_auto は切のまま(既定・設定を変えない)。切なら走りが変わらないことをここで見る。

本物の I2C・装置・ディスクの空きには触れない:
  - バスは偽物 FakeAds(データシート SBAS444 の決まりの一部を真似る: 設定レジスタ・連続変換・1 回変換・電源を入れた直後の 0x8583 と
    変換レジスタ 0x0000)。全部の読み書き(アドレス・レジスタ・中身・時刻)を記録し、0x48 以外・設定レジスタ以外への書き込み・
    知らない関数は bad に残して例外 → 試験の中で bad が空かを見る
  - smbus2.SMBus・vehicle.SMBus・ads1115_vbat._smbus・ads1115_vbat.free_bytes は、どの試験でも呼ばれたら残す偽物にする(autouse。
    終わりに何も呼ばれていないかを見る。空きの偽物は 0 = 書かない側を返す)。記録を回す試験の空き・fsync はいつも偽物で、どの繰り返しにも
    上限がある(10/31 の as5601_logger の試験の OOM の二の舞にしない)
  - 時計は偽物(スレッドの試験だけ本物の時計・短い上限つき)。サーバーは import するだけ(tests/test_vbat_desc.py と同じ。PCA9685・
    センサーは起動しない。走行の開始を呼ぶ試験は、本物の Driver.start が偽物に替わっていることを先に確かめる)。走行ループは
    tests/test_speed_ctrl.py の偽物(スレッドなし・決まった時刻・PCA9685 は偽物)
  - スレッドを始める試験が試験の外にスレッドを残しても本物の I2C に届かないように、4 つの守りを重ねる(10/31 確かめ直しの指摘 V1・
    確かめの指摘 VR1-S1・VR1-N1。下の「本物の I2C に届かないための見張り」): (1) start は try の中・finally で必ず止める (2) 偽物は
    インスタンスに入れる(車両 = no_board(s)・読み手 = bus_factory)(3) 後片付けで、その試験が始めたスレッド(名前は問わない)が
    残っていないかを見る (4) このファイルの jetformer・test_* の import より前に 1 度だけ入れる監査の見張り(sys.addaudithook)が、
    入れたあとは、どのスレッドの /dev/i2c の open・I2C の ioctl も止めて数える

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider tests/test_vbat_ads1115.py
"""
import ast
import atexit
import csv
import errno
import inspect
import json
import math
import os
import stat
import sys
import threading
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# ================================================================ 本物の I2C に届かないための見張り(10/31 確かめ直しの指摘 V1・確かめの指摘 VR1-S1・VR1-N1)
# 10/31 20:57 ごろ、前の直す係のわざと壊す確かめ R4(VehicleSensor.start が読み手の start の例外を囲まない形)の回で、start が try の
# 外にあったため車両のスレッドが試験の外に残り、試験の終わりに monkeypatch が vs.SMBus を本物の smbus2.SMBus に戻したあと、本物の
# /dev/i2c-7 を開いて制御基板 0x28 を約 2 秒読んだ可能性が高い(読むだけ。書き込みは無い)。もう起きないように 4 つを重ねる
# (どれか 1 つでは足りない: monkeypatch が戻したあとも、残ったスレッドが本物に届かないこと):
#   (1) スレッドを始める試験は、start・hub.start を try の中に入れ、finally で必ず止める
#   (2) スレッドを始める試験の偽物は、モジュールの名前ではなくインスタンスに入れる(車両 = no_board(s) で s._open・読み手 =
#       bus_factory)。試験のあとに戻されないので、スレッドが残っても本物に届かない
#   (3) 後片付け(guard_threads_and_i2c)で、その試験が始めたスレッドが残っていないことを見る。名前は問わない(VR1-N1: 前は名前が
#       VEHICLE・VBAT-… の物だけを見ていたので、名前の違うスレッドが残っても見つからなかった)。少し待っても残っていれば止める合図を
#       出して、試験を落とす。monkeypatch が偽物を戻す前に見る
#   (4) この下の監査の見張り。このファイルの jetformer・test_* の import より前に 1 度だけ sys.addaudithook で入れる(VR1-S1: 前は
#       import のあとに入れていたので、server の import の間 = SensorHub() が車両・読み手を作る間の /dev/i2c の open を止められなかった。
#       並べ替えの道具で import をこの上へ動かさないこと。試験 test_i2c_guard_is_installed_before_the_jetformer_and_test_imports が見る)。
#       入れたあとは、どのスレッドでも・いつでも(import の間・試験の間・試験と試験の間・このファイルの試験のあと。名前・試験の間かでは
#       分けない = VR1-N1)、次の 2 つをシステムコールの前に PermissionError で止めて数を残す:
#         - 'open' の知らせ(open()・os.open が出す。smbus2 は os.open('/dev/i2c-N', os.O_RDWR) で開く)のパスが /dev/i2c で始まる
#           (/dev/./i2c-7 など形の違う書き方・相対の i2c-N・/dev/char/89:N や /proc/self/fd/N などの別の名前も)
#         - 'fcntl.ioctl' の知らせ(smbus2 は 1 回の取引ごとに I2C_SMBUS・I2C_RDWR、相手を決めるのに I2C_SLAVE などを呼ぶ)が I2C の
#           要求で、fd が i2c-dev(文字の装置 89)か、開いていない・分からない fd(見張りより前に開いた fd での取引も止まる)
#       後片付けで、止めた数が 0 でなければ落とす(残ったスレッドか作るときの処理が本物を開こうとした印)。監査のフックはプロセスに
#       残って外せないので 1 度だけ入れ、'open'・'fcntl.ioctl' 以外の知らせではすぐ返す。分けたあとの思わぬ失敗は止める側(通さない)。
#       ほかの試験ファイルへの影響: このファイルを集めた(import した)あとは、同じプロセスのどの試験も /dev/i2c を開けない・I2C の
#       ioctl を呼べない。試験一式に、それをする試験は無い(README 20 章。grep)= 結果は変わらない。このファイルより前に集める試験
#       ファイルの import の間は守れない(それには conftest が要る。今回の 3 ファイルの外)
SENSOR_THREAD_NAMES = ('VEHICLE', 'VBAT-ADS1115', 'VBAT-LOG')   # 車両(= VehicleSensor.name。import のあとで確かめる)・読み手・記録の
                                                                 # スレッドの名前(README 14 章。見張りは名前では分けない)
LEFTOVER_WAIT_S = 2.0            # 後片付けで、スレッドが終わるのを待つ長さ(ふつうは finally の stop で終わっていて待たない)
I2C_MAJOR = 89                   # Linux の i2c-dev の装置の番号(/dev/i2c-N は文字の装置 89:N)
I2C_IOCTLS = {0x0703: 'I2C_SLAVE', 0x0705: 'I2C_FUNCS', 0x0706: 'I2C_SLAVE_FORCE', 0x0707: 'I2C_RDWR', 0x0708: 'I2C_PEC',
              0x0720: 'I2C_SMBUS'}                               # uapi/linux/i2c-dev.h の番号(smbus2 0.6.1 の台本と同じ)
ALIAS_DIRS = ('/dev/', '/proc/')                                 # /dev/i2c の別の名前がありうる所(/dev/char/89:7・/proc/self/fd/N)
GUARD_EVENTS = frozenset(('open', 'fcntl.ioctl'))                # 見張る監査の知らせ(ほかはすぐ返す)


class I2cGuard(object):
    """監査の見張りの中身(フック _audit が呼ぶ。試験・外の見張りからも呼べる)。止めるときの記録のほかは I/O をしない
    (パスが /dev・/proc の下のときだけ、別の名前かを realpath・stat で見る。装置は開かない)"""
    SELFTEST = '/dev/i2c-jf-selftest'    # 見張りの試験に使う名前(装置は無い。sys.audit で 'open' の知らせを出すだけで、開かない)
    BAD_FD = 1 << 20                     # 見張りの試験に使う fd の番号(開いていない番号。ioctl の知らせを出すだけ)

    def __init__(self):
        self.installed = False           # sys.addaudithook を入れた
        self.armed = None                # 今の段: このファイルの import の間 = IMPORT_MARK・試験の間 = その試験の nodeid・ほか None
                                         # (どの段で止めたかを残すだけ。止めるかどうかには使わない)
        self.blocked = []                # 止めた物: (time.monotonic(), スレッドの名前, 何を, その時の段, 'open' か 'ioctl')
        self.lost = 0                    # 止めたが記録できなかった数(記録の思わぬ失敗。それでも止める)
        self.seen = 0                    # take() で返した数
        self.watch, self.watch_hits = None, 0     # 見張りの試験: このパスの 'open' の知らせがフックに届いた数
        self.lock = threading.Lock()

    @staticmethod
    def i2c_path(path, realpath=os.path.realpath, stat_fn=os.stat):
        """open のパス → /dev/i2c(i2c-dev の装置)なら止める物の文字列・ほかは None。思わぬ例外は出さない(パスでない物 = 数の
        ファイル記述子・None・パスにできない物は None。i2c の名前・/dev・/proc の下の名前で、分ける途中の思わぬ失敗は止める側)"""
        if path is None or isinstance(path, int):
            return None
        try:
            p = os.fsdecode(path)
        except Exception:
            return None                  # パスでない物(open そのものが失敗する)
        rel = not p.startswith('/')
        try:
            q = os.path.normpath(os.path.join(os.getcwd(), p) if rel else p)
        except Exception:
            return p if 'i2c' in p else None      # 形を確かめられない i2c の名前は止める側
        if 'i2c' not in q and not q.startswith(ALIAS_DIRS):
            return None                  # ほとんどの open はここで返る(遅くしない)
        try:
            if q.startswith('/dev/i2c') or (rel and os.path.basename(q).startswith('i2c-')):
                return q                 # 相対の i2c-N は、dir_fd で /dev を指しているかもしれないので止める側
            if q.startswith(ALIAS_DIRS):                         # 別の名前(/dev/char/89:7・/proc/self/fd/N など)
                r = realpath(q)
                if str(r).startswith('/dev/i2c'):
                    return '%s -> %s' % (q, r)
                try:
                    st = stat_fn(q)
                except OSError:
                    return None          # 無い・見られない(open も失敗する)
                if stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == I2C_MAJOR:
                    return '%s(文字の装置 %d:%d)' % (q, I2C_MAJOR, os.minor(st.st_rdev))
            return None
        except Exception:
            return q                     # 分ける途中の思わぬ失敗は止める側(ここに来るのは i2c・/dev・/proc の名前だけ)

    @staticmethod
    def i2c_ioctl(fd, request, fstat=os.fstat):
        """fcntl.ioctl → I2C の要求で、fd が i2c-dev(文字の装置 89)・開いていない・分からないなら止める物の文字列。ほかの要求・
        ほかの fd(ふつうのファイル・端末など)は None。思わぬ例外は出さない"""
        try:
            name = I2C_IOCTLS.get(int(request))
        except Exception:
            return None                  # 数でない要求(ioctl そのものが失敗する)
        if name is None:
            return None                  # I2C の要求でない(ほとんどの ioctl はここで返る)
        try:
            st = fstat(fd)
        except Exception:
            return '%s(fd %s: 開いていない・分からない)' % (name, fd)
        try:
            if stat.S_ISCHR(st.st_mode) and os.major(st.st_rdev) == I2C_MAJOR:
                return '%s(fd %s: i2c-%d)' % (name, fd, os.minor(st.st_rdev))
            return None
        except Exception:
            return '%s(fd %s: 分からない)' % (name, fd)

    def would_block(self, thread_name=None):
        """止めるか: 入れたあとは、どのスレッドでも・いつでも止める(スレッドの名前・試験の間かでは分けない。VR1-N1)"""
        return bool(self.installed)

    def check(self, path):
        """'open' の知らせ: /dev/i2c なら止める(数を残して PermissionError。システムコールの前に止まる)。外の見張りからも呼ばれる"""
        try:
            if self.watch is not None and path == self.watch:
                self.watch_hits += 1
        except Exception:
            pass
        what = self.i2c_path(path)       # 思わぬ例外を出さない(止める・通すを中で決める)
        if what is not None:
            self._block('open', what)

    def check_ioctl(self, fd, request):
        """'fcntl.ioctl' の知らせ: I2C の要求で、fd が i2c-dev か分からなければ止める(数を残して PermissionError)"""
        what = self.i2c_ioctl(fd, request)
        if what is not None:
            self._block('ioctl', what)

    def _block(self, kind, what):
        """数を残して PermissionError。記録が思わぬ失敗をしても止める(通す側に倒れない)"""
        try:
            name = threading.current_thread().name
            with self.lock:
                self.blocked.append((time.monotonic(), name, what, self.armed, kind))
        except Exception:
            self.lost += 1
        raise PermissionError(errno.EACCES, '試験の見張り: %s の %s を止めました' % (what, kind), str(what))

    def take(self):
        """まだ見ていない「止めた物」(後片付け・見張りの試験が見る)"""
        with self.lock:
            new, self.seen = self.blocked[self.seen:], len(self.blocked)
        return new


def _audit(event, args):
    if event not in GUARD_EVENTS:        # 'open'・'fcntl.ioctl' 以外の知らせはすぐ返す(遅くしない)
        return
    if event == 'open':
        I2C_GUARD.check(args[0])         # 止めるときは PermissionError(分ける前の思わぬ失敗では、ほかの open を壊さない)
    else:
        I2C_GUARD.check_ioctl(args[0], args[1])


def _report_at_exit():
    """このファイルの試験のあと(試験一式のほかの試験ファイル・プロセスの終わりの処理)に止めた物を、プロセスの終わりに知らせる
    (そのときはもう試験を落とせないので、知らせだけ。止めることはどの時点でもしている)"""
    try:
        left = I2C_GUARD.take()
        if left or I2C_GUARD.lost:
            sys.stderr.write('[tests/test_vbat_ads1115.py の見張り] このファイルの試験のあとに /dev/i2c の open・I2C の ioctl を %d 回'
                             '止めました(記録できなかった物 %d): %r\n' % (len(left), I2C_GUARD.lost, left[:5]))
    except Exception:
        pass


I2C_GUARD = getattr(sys, '_jf_test_i2c_guard', None)       # 2 度目の import でも同じ見張り(フックは 1 つだけ)
if I2C_GUARD is None:
    I2C_GUARD = I2cGuard()
    sys.addaudithook(_audit)             # 外せない・プロセスに残る(だから 1 度だけ)。この下の jetformer・test_* の import より前
    I2C_GUARD.installed = True
    sys._jf_test_i2c_guard = I2C_GUARD   # わざと壊す確かめの外の見張り(scratchpad の i2c_guard_plugin)も、この判断を先に呼ぶ
    atexit.register(_report_at_exit)

# ---------------------------------------------------------------- 見張りを入れてから import する(server の import は SensorHub() で
# 車両のセンサーと電圧の読み手を作る。作るときにバスを開く・スレッドを始める形に壊れても、上の見張りが止める。VR1-S1)
IMPORT_MARK = '<import of tests/test_vbat_ads1115.py>'
_ARMED_OUTSIDE = I2C_GUARD.armed
I2C_GUARD.armed = _ARMED_OUTSIDE or IMPORT_MARK      # 止めた物に「import の間」と残す(最初の試験の始めで落とす)
try:
    from jetformer import clock as jclock                                                     # noqa: E402
    from jetformer import config                                                              # noqa: E402
    from jetformer import drive, speed_auto                                                   # noqa: E402
    from jetformer.recorder import Recorder                                                   # noqa: E402
    from jetformer.sensors import SensorHub                                                   # noqa: E402
    from jetformer.sensors import ads1115_vbat as A                                           # noqa: E402
    from jetformer.sensors import vehicle as vs                                               # noqa: E402
    from jetformer.web import bench                                                           # noqa: E402
    from jetformer.web import server as srv                                                   # noqa: E402

    from test_drive_caps import same                                                          # noqa: E402
    from test_speed_ctrl import PARAMS as SC_PARAMS                                           # noqa: E402
    from test_speed_ctrl import build as sc_build                                             # noqa: E402
    from test_speed_ctrl import run as sc_run                                                 # noqa: E402
    from test_vbat_decode import _Bus, snap                                                   # noqa: E402
    from test_vbat_desc import DESC, _log, _scene_desc, _start, fake_start, parse, tracker    # noqa: E402,F401  (fixture)
    from test_vbat_record import _ai, _scene                                                  # noqa: E402
finally:
    I2C_GUARD.armed = _ARMED_OUTSIDE

assert vs.VehicleSensor.name == SENSOR_THREAD_NAMES[0], '車両のスレッドの名前が README・試験と違う: %r' % vs.VehicleSensor.name

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
FSR = {0: 6.144, 1: 4.096, 2: 2.048, 3: 1.024, 4: 0.512, 5: 0.256, 6: 0.256, 7: 0.256}     # データシートの表(PGA → ±V)
SPS = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128, 5: 250, 6: 475, 7: 860}                         # DR → 1 秒の変換の数
AIN0_NOW = 14708 * 125e-6        # 10/31 18:54 に本物で読んだ値(raw 14708 = AIN0 1.8385 V = 電池 7.906 V。較正の前)


def live_threads():
    """生きているスレッド(主のスレッドと、C から入った仮のスレッド threading._DummyThread を除く。名前は問わない)"""
    main, dummy = threading.main_thread(), getattr(threading, '_DummyThread', ())
    return [th for th in threading.enumerate() if th is not main and th.is_alive() and not isinstance(th, dummy)]


def benign(th):
    """試験の外に残ってよいスレッド: 収集の Recorder が作るたびに 1 つできる JPEG の書き手(Recorder._io_worker。列を待つだけで I2C に
    触れない・止める道が無い。test_collected_scene_keeps_the_board_voltage が作る)。名前ではなく、動かしている関数で見る"""
    f = getattr(getattr(th, '_target', None), '__func__', None)
    return f is not None and f is Recorder._io_worker


def leftover_threads(before, wait_s=LEFTOVER_WAIT_S):
    """before に無かったスレッド(名前は問わない。VR1-N1。benign の物を除く)のうち、wait_s 待っても終わらない物(0.01 秒ごとに見る)"""
    end = time.monotonic() + wait_s
    while True:
        left = [th for th in live_threads() if th not in before and not benign(th)]
        if not left or time.monotonic() >= end:
            return left
        time.sleep(0.01)


def signal_stop(th):
    """残ったスレッドに止める合図だけ出す(待たない)。車両 = SensorThread の _stop・読み手 = 止める合図と終わった印・
    記録 = 読み手が終わった印(スレッドの target と args。CPython のスレッドが動いている間だけある)。出した合図の数"""
    owner = getattr(getattr(th, '_target', None), '__self__', None)
    n = 0
    for ev in [getattr(owner, '_stop', None)] + list(getattr(th, '_args', None) or ()):
        if isinstance(ev, threading.Event):
            ev.set()
            n += 1
    return n


@pytest.fixture(autouse=True)
def no_real_hw(monkeypatch):
    """本物の SMBus を作る・本物のディスクの空きを読むと、ここに残る(試験の終わりに空かを見る)"""
    hit = {'smbus': [], 'free': []}

    def smbus(*a, **k):
        hit['smbus'].append(a)
        raise AssertionError('本物の SMBus を作ろうとした %r' % (a,))

    def free(path):
        hit['free'].append(str(path))
        return 0                                    # 書かない側
    import smbus2
    monkeypatch.setattr(smbus2, 'SMBus', smbus)
    monkeypatch.setattr(vs, 'SMBus', smbus)
    monkeypatch.setattr(A, '_smbus', smbus)
    monkeypatch.setattr(A, 'free_bytes', free)
    yield hit
    assert hit == {'smbus': [], 'free': []}


@pytest.fixture(autouse=True)
def guard_threads_and_i2c(request, monkeypatch, no_real_hw):
    """V1 の守り (3)(4): 後片付けで、その試験が始めたスレッド(名前は問わない)が残っていないこと・見張りが止めた物が 0 であることを
    見る。monkeypatch・no_real_hw のあとに始まり先に終わるので、偽物が戻される前に、残ったスレッドへ止める合図を出せる"""
    early = I2C_GUARD.take()             # この試験の前(このファイルの import の間・前の試験の後片付けのあと)に止めた物
    if early:
        pytest.fail('この試験の前(このファイルの import の間か、前の試験のあと)に、見張りが /dev/i2c の open・I2C の ioctl を %d 回'
                    '止めました(残ったスレッドか、作るときの処理が本物を開こうとした印): %r' % (len(early), early[:5]))
    before = set(live_threads())         # 前から残っている物は数えない(残した試験で落とした)
    lost0 = I2C_GUARD.lost
    I2C_GUARD.armed = request.node.nodeid
    yield I2C_GUARD
    left = leftover_threads(before)
    for th in left:
        signal_stop(th)
    still = leftover_threads(before, 1.0) if left else []
    I2C_GUARD.armed = None
    blocked = I2C_GUARD.take()
    msgs = []
    if left:
        msgs.append('試験のあとにスレッドが残っていました: %s(止める合図を出しました。まだ残っている物: %s)'
                    % (sorted(th.name for th in left), sorted(th.name for th in still)))
    if blocked:
        msgs.append('見張りが /dev/i2c の open・I2C の ioctl を %d 回止めました: %r' % (len(blocked), blocked[:5]))
    if I2C_GUARD.lost != lost0:
        msgs.append('見張りが止めたが記録できなかった物が %d あります' % (I2C_GUARD.lost - lost0))
    if msgs:
        pytest.fail(' / '.join(msgs))


@pytest.fixture(autouse=True, scope='module')
def guard_after_the_last_test():
    """V1 の守り (3)(4) のこのファイルの終わり: 最後の試験のあとに残ったスレッド(このファイルの試験の前には無かった物)・止めた物も見る
    (このあと、試験一式のほかの試験ファイルの間に止めた物は、プロセスの終わりに知らせる = _report_at_exit)"""
    before = set(live_threads())
    yield
    left = leftover_threads(before, 0.5)
    for th in left:
        signal_stop(th)
    blocked = I2C_GUARD.take()
    if left or blocked:
        pytest.fail('このファイルの試験のあと: 残ったスレッド %s・見張りが止めた物 %d 回 %r'
                    % (sorted(th.name for th in left), len(blocked), blocked[:5]))


# ================================================================ 偽物
class Clock(object):
    """偽の時計(monotonic・time・sleep)"""

    def __init__(self, m=1000.0, wall=1793440000.0):
        self.m, self.wall0 = float(m), float(wall)

    def monotonic(self):
        return self.m

    def time(self):
        return self.wall0 + self.m

    def sleep(self, dt):
        self.m += dt


class FakeAds(object):
    """ADS1115 の偽物(データシート SBAS444 の決まりの一部)。
    - 設定レジスタ 0x01・変換レジスタ 0x00(16 ビット・上位のバイトが先)。電源を入れた直後・power_cycle() のあとは
      設定 0x8583・変換 0x0000
    - MODE 0(連続変換)を書くと、1/DR 秒たってから変換レジスタが今の値になり続ける(MUX AIN0 対 GND・AIN0 対 AIN1 は ain0 V、
      AIN1〜AIN3 は GND = 0 V。PGA の幅で 2 の補数の 16 ビット・端で張り付く)
    - MODE 1 で OS 1(1 回変換)は、その時の値で 1 回だけ(休んでいる間は新しくならない)
    - 読んだ設定レジスタの OS: 連続変換中は os_in_cont(既定 0)・休んでいるときは 1
    - 0x48 以外・レジスタ 0x00/0x01 の 2 バイト以外の読み・設定レジスタ 0x01 の 2 バイト以外の書き込み・知らない関数は bad に残して
      AssertionError。fail(op, reg) が例外を返したら、それを出す(I2C の失敗)"""

    def __init__(self, clock, ain0=AIN0_NOW, os_in_cont=0, ignore_writes=False):
        self.clock, self.ain0, self.os_in_cont, self.ignore_writes = clock, ain0, os_in_cont, ignore_writes
        self.calls, self.bad = [], []
        self.fail = None
        self.closed = 0
        self.power_cycle()

    def power_cycle(self):
        self.cfg = 0x8583
        self.conv = 0
        self.t_cont = None

    def _code(self):
        mux = self.cfg >> 12 & 7
        v = self.ain0 if mux in (0b000, 0b100) else 0.0
        c = int(round(v / FSR[self.cfg >> 9 & 7] * 32768))
        return max(-32768, min(32767, c)) & 0xFFFF

    def _update(self):
        if (not self.cfg >> 8 & 1) and self.t_cont is not None and \
                self.clock.monotonic() - self.t_cont >= 1.0 / SPS[self.cfg >> 5 & 7]:
            self.conv = self._code()

    def _check(self, op, addr, reg, n):
        ok = addr == 0x48 and n == 2 and ((op == 'read' and reg in (0, 1)) or (op == 'write' and reg == 1))
        if not ok:
            self.bad.append((op, addr, reg, n))
            raise AssertionError('ADS1115 の偽物: してはいけない %s addr=0x%02x reg=0x%02x n=%s' % (op, addr, reg, n))
        if self.fail is not None:
            e = self.fail(op, reg)
            if e is not None:
                raise e

    def read_i2c_block_data(self, addr, reg, n):
        self.calls.append(('read', addr, reg, n, self.clock.monotonic()))
        self._check('read', addr, reg, n)
        self._update()
        if reg == 0:
            w = self.conv
        else:
            cont = not (self.cfg >> 8 & 1)
            w = (self.cfg & 0x7FFF) | ((self.os_in_cont if cont else 1) << 15)
        return [w >> 8 & 0xFF, w & 0xFF]

    def write_i2c_block_data(self, addr, reg, data):
        data = list(data)
        self.calls.append(('write', addr, reg, data, self.clock.monotonic()))
        self._check('write', addr, reg, len(data))
        if self.ignore_writes:
            return
        w = (data[0] & 0xFF) << 8 | (data[1] & 0xFF)
        self.cfg = w & 0x7FFF
        if not (w >> 8 & 1):
            self.t_cont = self.clock.monotonic()
        else:
            self.t_cont = None
            if w >> 15:
                self.conv = self._code()

    def close(self):
        self.closed += 1

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)

        def f(*a, **k):
            self.bad.append((name,) + a)
            raise AssertionError('ADS1115 の偽物に無い関数 %s' % name)
        return f


class Factory(object):
    """bus_factory の偽物(SMBus(7) の代わりに偽物を返す。作った番号を残す)"""

    def __init__(self, dev):
        self.dev, self.made = dev, []

    def __call__(self, bus_no):
        self.made.append(bus_no)
        return self.dev


def make(tmp_path=None, log=False, scale=1.0, dev=None, clk=None, free=None, **kw):
    clk = clk or Clock()
    dev = dev or FakeAds(clk)
    fac = Factory(dev)
    syncs, msgs = [], []
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat') if log else None, log=log, bus_factory=fac,
                      scale_fn=scale if callable(scale) else (lambda: scale), clock=clk,
                      free_bytes=free or (lambda p: 50 * GIB), fsync=syncs.append, printer=msgs.append, **kw)
    return types.SimpleNamespace(r=r, dev=dev, clk=clk, fac=fac, syncs=syncs, msgs=msgs)


def run_for(r, clk, seconds, drain=False, max_steps=200000):
    """読み手の next_t どおりに step を呼ぶ(スレッドなし。回数の上限つき)。drain = 1 回ごとに記録を書く"""
    end = clk.m + seconds
    n = 0
    while n < max_steps:
        t = max(clk.m, r.next_t)
        if t > end + 1e-9:
            break
        clk.m = t
        r.step(t)
        if drain:
            r.drain()
        n += 1
    clk.m = max(clk.m, end)
    return n


def _csv(path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ================================================================ 設定の値・raw → 電圧(データシートどおり)
def test_config_words_follow_the_datasheet():
    assert A.ADDR == 0x48 and A.BUS == 7 and A.REG_CONV == 0x00 and A.REG_CONFIG == 0x01
    assert A.CONFIG_RUN == 0x4283 and A.CONFIG_DEFAULT == 0x8583
    # データシートのビットの並び(OS 15・MUX 14-12・PGA 11-9・MODE 8・DR 7-5・COMP_MODE 4・POL 3・LAT 2・QUE 1-0)を自前で
    assert (0 << 15 | 0b100 << 12 | 0b001 << 9 | 0 << 8 | 0b100 << 5 | 0 << 4 | 0 << 3 | 0 << 2 | 0b11) == 0x4283
    assert A.describe_config(0x4283) == {'os': 0, 'mux': 'AIN0-GND', 'pga_v': 4.096, 'mode': 'continuous', 'dr_sps': 128,
                                         'comp_mode': 0, 'comp_pol': 0, 'comp_lat': 0, 'comp_que': 'disabled'}
    d = A.describe_config(0x8583)
    assert (d['os'], d['mux'], d['pga_v'], d['mode'], d['dr_sps'], d['comp_que']) == (1, 'AIN0-AIN1', 2.048, 'single', 128,
                                                                                         'disabled')
    assert A.config_word(os_bit=1, mode=1) == 0xC383          # main session が 10/31 18:54 に本物で使った 1 回変換の値
    assert A.config_matches(0x4283) and A.config_matches(0xC283)              # OS は比べない
    assert not A.config_matches(0x8583) and not A.config_matches(0xC383) and not A.config_matches(None)
    assert A.LSB_V == pytest.approx(125e-6, rel=0, abs=1e-15) and A.DIVIDER == pytest.approx(4.3, rel=0, abs=1e-12)
    assert A.RATE_HZ == 50.0 and A.DR_SPS == 128 and A.ENV == 'JETFORMER_VBAT_ADS1115'
    assert A.ROTATE_S == 1800.0 and A.FSYNC_S == 1.0 and A.MIN_FREE == 2 * GIB
    assert vs.VBAT_ADS_DIR == config.DATA_DIR / 'vbat' and vs.VBAT_ADS_FRESH_S == 0.25     # 記録の場所・使う新しさ


@pytest.mark.parametrize('word,raw', [(0x3974, 14708), (0x0000, 0), (0x0001, 1), (0xFFFF, -1), (0xFFFE, -2),
                                      (0x8001, -32767), (0x8000, -32768), (0x7FFF, 32767), (0x7FFE, 32766)])
def test_twos_complement(word, raw):
    assert A.to_int16(word) == raw


def test_raw_to_volts_negative_and_saturation():
    assert A.raw_to_ain0(14708) == pytest.approx(1.8385, abs=1e-9)            # 10/31 18:54 の本物の値
    assert A.raw_to_vbat(14708) == pytest.approx(7.90555, abs=1e-6)           # × 4.3(較正の前)
    assert A.raw_to_ain0(-1) == pytest.approx(-125e-6) and A.raw_to_vbat(-1) == pytest.approx(-0.0005375)
    assert A.raw_to_vbat(A.to_int16(0xFFFF)) < 0                             # 2 の補数(65535 × 125 µV × 4.3 = 35 V にしない)
    assert A.is_saturated(32767) and A.is_saturated(-32768) and not A.is_saturated(32766) and not A.is_saturated(-32767)
    assert A.raw_to_vbat(32767) == pytest.approx(17.61, abs=0.01)             # 張り付き = 電池ではありえない
    for v in (8.4, 7.0, 6.0):
        assert A.raw_to_vbat(int(round(v / 4.3 / 125e-6))) == pytest.approx(v, abs=0.0006)    # 1 LSB × 4.3 = 0.54 mV


def test_backoff_schedule():
    assert [A.backoff_s(k) for k in range(1, 10)] == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5.0, 5.0, 5.0])
    assert A.backoff_s(10 ** 6) == 5.0 and A.backoff_s(0) == pytest.approx(0.1)


def test_module_touches_only_0x48_and_writes_only_the_config_register():
    """台本の中の I2C の呼び出し: 読むのは read_i2c_block_data・書くのは write_i2c_block_data だけで、アドレスはいつも ADDR(0x48)。
    書く関数 _write16 を呼ぶのは 1 か所(設定レジスタに CONFIG_RUN)だけ。read_word_data(SMBus の語は下位のバイトが先)は使わない"""
    tree = ast.parse((ROOT / 'jetformer' / 'sensors' / 'ads1115_vbat.py').read_text(encoding='utf-8'))
    i2c = {'read_i2c_block_data', 'write_i2c_block_data', 'read_byte_data', 'write_byte_data', 'read_word_data',
           'write_word_data', 'i2c_rdwr', 'read_byte', 'write_byte', 'write_quick', 'process_call', 'block_process_call',
           'read_block_data', 'write_block_data'}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in i2c]
    assert sorted(c.func.attr for c in calls) == ['read_i2c_block_data', 'write_i2c_block_data']
    assert all(isinstance(c.args[0], ast.Name) and c.args[0].id == 'ADDR' for c in calls)
    w16 = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == '_write16']
    assert len(w16) == 1 and [a.id for a in w16[0].args] == ['REG_CONFIG', 'CONFIG_RUN']


# ================================================================ 読み手(偽の ADS1115・偽の時計・スレッドなし)
def test_start_writes_continuous_config_and_touches_only_0x48():
    k = make()
    k.r.step(k.clk.m)
    assert k.fac.made == [7]                                                  # I2C の 7 番だけ
    assert [c[:4] for c in k.dev.calls] == [('read', 0x48, 0x01, 2), ('write', 0x48, 0x01, [0x42, 0x83]), ('read', 0x48, 0x01, 2)]
    assert k.r.fresh(1.0) is None and k.r.next_t == pytest.approx(1000.0 + A.SETTLE_S)     # 書いた直後の変換は使わない
    run_for(k.r, k.clk, 2.0)
    a = k.r.fresh(0.25)
    assert a['raw'] == 14708 and a['board_v'] == pytest.approx(7.906, abs=0.001) and a['sat'] is False
    conv = [c for c in k.dev.calls if c[0] == 'read' and c[2] == 0]
    assert 99 <= len(conv) <= 101 and k.r.reads == len(conv)                  # 2 秒 × 50 回
    gaps = np.diff([c[4] for c in conv])
    assert gaps.min() == pytest.approx(0.02, abs=1e-6) and gaps.max() == pytest.approx(0.02, abs=1e-6)
    assert len([c for c in k.dev.calls if c[0] == 'write']) == 1               # 設定は始めの 1 回だけ(書き続けない)
    assert 3 <= len([c for c in k.dev.calls if c[0] == 'read' and c[2] == 1]) <= 5    # 始めの 2 回 + 1 秒ごと
    assert {c[1] for c in k.dev.calls} == {0x48} and k.dev.bad == []
    assert k.r.errors == 0 and k.r.config_writes == 1 and k.r.reconfigs == 0 and k.dev.cfg == 0x4283
    assert k.r.status()['config_read'] == '0x4283' and k.r.status()['configured'] is True


def test_values_follow_the_battery_in_continuous_mode():
    """連続変換: 電池の電圧が変わると次の読みから新しい値(1 回変換のままなら変換レジスタが止まる)"""
    k = make()
    run_for(k.r, k.clk, 0.5)
    for v in (7.40, 8.40, 6.95):
        k.dev.ain0 = v / 4.3
        run_for(k.r, k.clk, 0.1)
        assert k.r.fresh(0.25)['board_v'] == pytest.approx(v, abs=0.001), v
    assert (k.dev.cfg >> 8 & 1) == 0 and k.dev.bad == []


def test_negative_and_saturated_values_are_flagged():
    k = make()
    run_for(k.r, k.clk, 0.2)
    k.dev.ain0 = -0.0002                                                      # 電池を外した(わずかに負)
    run_for(k.r, k.clk, 0.1)
    a = k.r.fresh(0.25)
    assert a['raw'] == -2 and a['board_v'] < 0 and a['sat'] is False
    k.dev.ain0 = 5.0                                                          # 4.096 V より上 = 端に張り付く
    run_for(k.r, k.clk, 0.1)
    a = k.r.fresh(0.25)
    assert a['raw'] == 32767 and a['sat'] is True and k.r.saturated >= 1


def test_power_cycle_back_to_default_is_reconfigured_and_zero_is_not_published():
    """基板の電源が入り直して既定(設定 0x8583・変換レジスタ 0x0000)に戻った: 次の読みの 0 で設定を確かめ、0x4283 を書き直す。
    0 V(電源が入り直した印)は電池の値として出さない"""
    k = make()
    run_for(k.r, k.clk, 0.5)
    k.dev.power_cycle()
    published = []
    for _ in range(50):
        t = max(k.clk.m, k.r.next_t)
        k.clk.m = t
        k.r.step(t)
        a = k.r.fresh(0.25)
        if a is not None and a['t'] == t:
            published.append(a['raw'])
    assert k.r.reconfigs == 1 and k.r.config_writes == 2 and k.dev.cfg == 0x4283
    assert published and min(published) == 14708 == max(published)
    assert [w[3] for w in k.dev.calls if w[0] == 'write'] == [[0x42, 0x83], [0x42, 0x83]]
    assert any('書き直しました' in m for m in k.msgs) and k.dev.bad == []


def test_config_changed_by_another_tool_is_fixed_within_a_second():
    """ほかの道具が 1 回変換(0xC383 → 読むと 0x4383 系)を書いた・変換レジスタは古い値のまま: 1 秒ごとの確かめで書き直す"""
    k = make()
    run_for(k.r, k.clk, 0.3)
    k.dev.cfg, k.dev.t_cont = 0x4383, None                                    # 1 回変換で休んでいる
    run_for(k.r, k.clk, 1.05)
    assert k.r.config_writes == 2 and k.r.reconfigs == 1 and (k.dev.cfg >> 8 & 1) == 0
    k.dev.ain0 = 7.5 / 4.3
    run_for(k.r, k.clk, 0.1)
    assert k.r.fresh(0.25)['board_v'] == pytest.approx(7.5, abs=0.001)


@pytest.mark.parametrize('os_bit', [0, 1])
def test_os_bit_is_not_compared(os_bit):
    """連続変換中に読んだ OS が 0 でも 1 でも、書き直し続けない(比べるのは OS を除いた 15 ビット)"""
    clk = Clock()
    k = make(dev=FakeAds(clk, os_in_cont=os_bit), clk=clk)
    run_for(k.r, k.clk, 3.0)
    assert k.r.config_writes == 1 and k.r.reconfigs == 0 and k.r.reads >= 145


def test_config_that_does_not_stick_is_a_failure_with_backoff():
    """書いた設定が読み返せない(書き込みが届かない): 失敗として間をあける。値は出さない・詰めて書き直さない"""
    clk = Clock()
    k = make(dev=FakeAds(clk, ignore_writes=True), clk=clk)
    run_for(k.r, k.clk, 10.0)
    assert k.r.reads == 0 and k.r.fresh(100.0) is None
    assert 5 <= k.r.errors <= 8 and '0x4283' in k.r.last_error and 'OSError' in k.r.last_error
    assert len([c for c in k.dev.calls if c[0] == 'write']) == k.r.errors     # 1 回の試しに 1 回だけ書く


@pytest.mark.parametrize('code', [errno.ETIMEDOUT, errno.EREMOTEIO])
def test_stuck_bus_is_not_hammered(code):
    """バスが止まった形(OSError 110 タイムアウト・121 応答なしが続く): すぐ読み直さない。0.1・0.2・0.4・…・5.0 秒あける
    (60 秒で 20 回より少ない。50 Hz のままなら 3000 回)。5 回ごとに SMBus を開き直す(バスには何も送らない)。値は古くなる"""
    k = make()
    run_for(k.r, k.clk, 0.5)
    assert k.r.fresh(vs.VBAT_ADS_FRESH_S) is not None
    k.dev.fail = lambda op, reg: OSError(code, os.strerror(code))
    n0 = len(k.dev.calls)
    run_for(k.r, k.clk, 60.0)
    tries = [c[4] for c in k.dev.calls[n0:]]
    assert 12 <= len(tries) <= 20, len(tries)
    gaps = list(np.diff(tries))
    assert gaps == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 3.2] + [5.0] * (len(gaps) - 6), abs=1e-6)
    assert k.r.fresh(vs.VBAT_ADS_FRESH_S) is None                              # 車両は使わない(古い)
    assert k.r.errors == len(tries) == k.r.fail_streak and k.r.backoff == 5.0
    assert len(k.fac.made) == 1 + (len(tries) - 1) // 5 and k.dev.closed == len(tries) // 5
    assert sum('読めません' in m for m in k.msgs) == 1                          # journal の知らせは 5 分に 1 回まで
    st = k.r.status()
    assert st['fail_streak'] == len(tries) and str(code) in st['last_error']
    json.dumps(st, allow_nan=False)


def test_recovery_checks_the_config_first_then_reads():
    k = make()
    run_for(k.r, k.clk, 0.3)
    k.dev.fail = lambda op, reg: OSError(errno.ETIMEDOUT, 'Connection timed out')
    run_for(k.r, k.clk, 2.0)
    k.dev.fail = None
    n0 = len(k.dev.calls)
    run_for(k.r, k.clk, 2.0)                                                  # 次の読みは 1.6 秒あけたあと
    assert k.dev.calls[n0][:3] == ('read', 0x48, 0x01)                         # 戻ったら、まず設定レジスタ
    assert k.r.fail_streak == 0 and k.r.fresh(0.25)['raw'] == 14708
    assert any('読めるようになりました' in m for m in k.msgs)
    assert len([c for c in k.dev.calls if c[0] == 'write']) == 1                # 設定はそのまま(書き直さない)


def test_one_failure_waits_a_little_and_the_value_stays_fresh_for_the_vehicle():
    """1 回だけの失敗(モーターのノイズ): 0.1 秒あけて読み直す。車両が使う古さ 0.25 秒は越えない"""
    k = make()
    run_for(k.r, k.clk, 0.3)
    left = [1]

    def fail(op, reg):
        if left[0] > 0:
            left[0] -= 1
            return OSError(errno.EREMOTEIO, 'Remote I/O error')
    k.dev.fail = fail
    worst = 0.0
    for _ in range(30):
        t = max(k.clk.m, k.r.next_t)
        worst = max(worst, t - k.r._latest['t'])
        k.clk.m = t
        k.r.step(t)
    assert k.r.errors == 1 and k.r.fail_streak == 0
    assert 0.1 < worst <= vs.VBAT_ADS_FRESH_S


def test_fresh_ages_out():
    k = make()
    run_for(k.r, k.clk, 0.2)
    assert k.r.fresh(0.25) is not None
    k.clk.m += 0.26
    assert k.r.fresh(0.25) is None and k.r.fresh(1.0) is not None


def test_queue_overflow_drops_rows_and_keeps_reading(tmp_path):
    k = make(tmp_path, log=True)
    run_for(k.r, k.clk, 62.0)                                                 # 記録が書かない(列があふれる)
    assert k.r.dropped_queue > 0 and k.r.reads > 3000 and k.r.errors == 0


# ================================================================ CSV の記録(30 分・fsync・空き 2 GiB。空き・fsync は偽物)
def test_csv_rows_fields_and_fsync_every_second(tmp_path):
    k = make(tmp_path, log=True, scale=1.0125)
    run_for(k.r, k.clk, 3.0, drain=True)
    k.r.log.close()
    files = sorted((tmp_path / 'vbat').glob('*.csv'))
    assert len(files) == 1
    rows = _csv(files[0])
    assert list(rows[0].keys()) == A.FIELDS
    assert rows[0]['raw'] == '' and rows[0]['cfg'] == '4283' and rows[0]['note'] == 'config 8583->4283'   # 設定を書いた行
    data = [r for r in rows if r['raw']]
    assert 148 <= len(data) <= 151
    r = data[10]
    assert (r['raw'], r['ain0_v'], r['board_v'], r['scale'], r['ok'], r['err']) == ('14708', '1.838500', '7.9055', '1.01250',
                                                                                       '1', '0')
    assert float(r['vbat_v']) == pytest.approx(7.90555 * 1.0125, abs=1e-4)
    assert abs(int(r['epoch_us']) - int(r['mono_us']) - int(k.clk.wall0 * 1e6)) <= 2     # 壁の時計 − 単調な時計
    assert sum(1 for x in rows if x['cfg']) >= 3                               # 1 秒ごとに設定レジスタを読んだ行
    meta = json.loads(files[0].with_suffix('.json').read_text(encoding='utf-8'))
    assert meta['addr'] == '0x48' and meta['bus'] == 7 and meta['config'] == '0x4283' and meta['fields'] == A.FIELDS
    assert meta['divider'] == 4.3 and meta['config_meaning']['mode'] == 'continuous' and meta['rotate_s'] == 1800.0
    assert list(meta['field_notes']) == A.FIELDS                                # 欄の意味も残す
    assert 'rt_minus_mono_us' in meta and 'boot_id' in meta
    assert 4 <= len(k.syncs) <= 7                                              # 始めの json・1 秒ごと(3 秒)・閉じるとき


def test_csv_new_file_every_30_minutes(tmp_path):
    clk = Clock()
    syncs = []
    log = A.CsvLog(tmp_path, clock=clk, free_bytes=lambda p: 50 * GIB, fsync=syncs.append, printer=lambda m: None)
    for i in range(3660):                                                     # 61 分(1 秒に 1 行)
        assert log.write([i] * len(A.FIELDS)) is True
        clk.m += 1.0
    log.close()
    files = sorted(tmp_path.glob('*.csv'))
    assert len(files) == 3 == len(log.files) and [len(_csv(f)) for f in files] == [1800, 1800, 60]
    assert [f.stem for f in files] == [datetime.fromtimestamp(clk.wall0 + 1000.0 + s).strftime('%Y%m%d-%H%M%S')
                                       for s in (0, 1800, 3600)]
    assert all(f.with_suffix('.json').exists() for f in files) and log.syncs >= 3600


def test_csv_same_second_does_not_overwrite(tmp_path):
    clk = Clock()
    for _ in range(2):
        log = A.CsvLog(tmp_path, clock=clk, free_bytes=lambda p: 50 * GIB, fsync=lambda fd: None, printer=lambda m: None)
        log.write([0] * 11)
        log.close()
    names = sorted(f.name for f in tmp_path.glob('*.csv'))
    assert len(names) == 2 and sum(n.endswith('-2.csv') for n in names) == 1


def test_csv_waits_while_free_space_is_below_2gib(tmp_path):
    clk = Clock()
    free = {'v': int(1.5 * GIB)}
    msgs = []
    out = tmp_path / 'vbat'
    log = A.CsvLog(out, clock=clk, free_bytes=lambda p: free['v'], fsync=lambda fd: None, printer=msgs.append)
    for i in range(50 * 30):                                                  # 30 秒(50 Hz): 書かない
        assert log.write([i] * 11) is False
        clk.m += 0.02
    assert not out.exists() and log.dropped == 1500 and log.rows == 0 and log.waiting_disk is True
    assert len(msgs) == 1 and '待ちます' in msgs[0] and '2 GiB' in msgs[0]
    free['v'] = 3 * GIB                                                       # 空いた: 60 秒の見直しで書き始める
    wrote = 0
    for i in range(50 * 40):
        wrote += bool(log.write([i] * 11))
        clk.m += 0.02
    assert 400 <= wrote <= 600 and len(list(out.glob('*.csv'))) == 1 and any('続けます' in m for m in msgs)
    free['v'] = GIB                                                           # 書いている間に減った: 閉じて待つ(10 秒ごとに見る)
    n0 = log.rows
    for i in range(50 * 15):
        log.write([i] * 11)
        clk.m += 0.02
    assert log.fh is None and log.waiting_disk is True and log.rows - n0 <= 50 * 10 + 1


def test_csv_write_errors_never_raise_and_wait(tmp_path):
    clk = Clock()
    blocker = tmp_path / 'not_a_dir'
    blocker.write_text('x')
    msgs = []
    log = A.CsvLog(blocker / 'vbat', clock=clk, free_bytes=lambda p: 50 * GIB, fsync=lambda fd: None, printer=msgs.append)
    assert log.write([1] * 11) is False and log.last_error and log.waiting_disk
    clk.m += 30.0
    assert log.write([1] * 11) is False and log.dropped == 2                  # 60 秒は試さない
    log.tick()
    log.close()

    def no_space(fd):
        raise OSError(errno.ENOSPC, 'No space left on device')
    log2 = A.CsvLog(tmp_path / 'v2', clock=clk, free_bytes=lambda p: 50 * GIB, fsync=no_space, printer=msgs.append)
    assert log2.write([1] * 11) is False and 'No space' in log2.last_error
    log2.tick()
    log2.close()

    def broken_free(p):
        raise OSError(errno.EIO, 'statvfs failed')
    log3 = A.CsvLog(tmp_path / 'v3', clock=clk, free_bytes=broken_free, fsync=lambda fd: None, printer=msgs.append)
    assert log3.write([1] * 11) is False and log3.waiting_disk and not (tmp_path / 'v3').exists()


def test_tick_fsyncs_rows_even_when_no_new_rows_come(tmp_path):
    clk = Clock()
    syncs = []
    log = A.CsvLog(tmp_path, clock=clk, free_bytes=lambda p: 50 * GIB, fsync=syncs.append, printer=lambda m: None)
    log.write([1] * 11)
    n = len(syncs)
    log.tick()
    assert len(syncs) == n                                                    # まだ 1 秒たっていない
    clk.m += 1.0
    log.tick()
    assert len(syncs) == n + 1
    log.tick()
    assert len(syncs) == n + 1                                                # 新しい行が無ければ fsync しない
    log.close()


def test_csv_ok_column_follows_the_range_and_saturation(tmp_path):
    """CSV の ok は画面・説明欄と同じ決まり(6.0〜9.0 V × 係数・張り付きでない)。範囲の外も行は残す(raw・board_v はそのまま)"""
    k = make(tmp_path, log=True, scale=1.0)
    run_for(k.r, k.clk, 0.2, drain=True)
    for v in (5.5, 9.3):
        k.dev.ain0 = v / 4.3
        run_for(k.r, k.clk, 0.1, drain=True)
    k.dev.ain0 = 5.0                                                          # 端に張り付く
    run_for(k.r, k.clk, 0.1, drain=True)
    k.r.log.close()
    rows = [r for r in _csv(next((tmp_path / 'vbat').glob('*.csv'))) if r['raw']]
    by = {}
    for r in rows:
        by.setdefault(round(float(r['board_v']), 1), set()).add(r['ok'])
    assert by[7.9] == {'1'} and by[5.5] == {'0'} and by[9.3] == {'0'} and by[17.6] == {'0'}
    assert any(r['note'] == 'sat' for r in rows if r['raw'] == '32767')


def test_failure_rows_are_logged_with_errno(tmp_path):
    k = make(tmp_path, log=True)
    run_for(k.r, k.clk, 0.3, drain=True)
    k.dev.fail = lambda op, reg: OSError(errno.ETIMEDOUT, 'Connection timed out')
    run_for(k.r, k.clk, 1.0, drain=True)
    k.r.log.close()
    rows = _csv(next((tmp_path / 'vbat').glob('*.csv')))
    bad = [r for r in rows if r['err'] not in ('', '0')]
    assert [r['err'] for r in bad] == ['1', '2', '3', '4'] and all(r['raw'] == '' and r['note'] == 'TimeoutError 110' for r in bad)


# ================================================================ VehicleSensor のつなぎ(vbat_* を埋める)
class FakeReader(object):
    """vehicle が使う読み手の形(fresh・start・request_stop・stop・status)だけ。値と古さを決める。I2C には触れない"""

    def __init__(self, v=7.906, age=0.01, sat=False):
        self.v, self.age, self.sat = v, age, sat
        self.asked, self.calls = [], []
        self.stop_timeouts = []          # 車両の stop が渡した待ちの長さ(待ち 0 で呼ばないか。V2)

    def fresh(self, max_age_s):
        self.asked.append(max_age_s)
        if self.v is None or not self.age <= max_age_s:
            return None
        return {'t': 0.0, 'raw': int(round(self.v / 4.3 / 125e-6)), 'ain0_v': self.v / 4.3, 'board_v': self.v,
                'sat': self.sat}

    def start(self):
        self.calls.append('start')

    def request_stop(self):
        self.calls.append('request_stop')

    def stop(self, timeout=3.0):
        self.calls.append('stop')
        self.stop_timeouts.append(timeout)

    def status(self):
        return {'enabled': True, 'fake': True}


def sensor(tmp_path, reader, scale=None):
    path = tmp_path / 'vbat_calib.json'
    if scale is not None:
        path.write_text(json.dumps({'scale': scale}))
    return vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot', 'vbat_calib_path': path,
                             'vbat_ads1115': reader})


def no_board(s):
    """車両のセンサーの制御基板(0x28)を「つながっていない」偽物にする(スレッドを始める試験)。偽物はインスタンスの _open に入れる
    (モジュールの vs.SMBus ではなく)= 試験のあとに monkeypatch が vs.SMBus を本物に戻しても、残ったスレッドは本物を開かない(V1 の (2))"""
    s.open_calls = 0

    def _open():
        s.open_calls += 1
        raise OSError(errno.ENOENT, '偽物: 制御基板は無い')
    s._open = _open
    return s


def polls(s, frames):
    s.bus = _Bus(frames)
    return [x for _ in frames for x in s._poll_snapshot()]


def test_vehicle_uses_the_fresh_board_value_times_the_scale_once(tmp_path):
    rd = FakeReader(7.906)
    s = sensor(tmp_path, rd, scale=1.0125)
    out = polls(s, [snap(seq=i) for i in range(1, 4)])                        # ESP32 は電圧を送らない(flags bit5 = 0)
    d = out[-1].data
    assert d['vbat_ok'] is True and d['vbat_src'] == 'ads1115' and d['vbat_bad'] is False
    assert d['vbat_v'] == pytest.approx(7.906 * 1.0125, abs=1e-9) and d['vbat_raw_v'] == pytest.approx(7.906 * 1.0125, abs=1e-9)
    assert rd.asked == [vs.VBAT_ADS_FRESH_S] * 3 and vs.VBAT_ADS_FRESH_S == 0.25
    s._publish(out[-1])
    st = s.status()
    assert st['vbat_supported'] is True and st['vbat_ok'] is True and st['vbat_src'] == 'ads1115'
    assert st['vbat_v'] == pytest.approx(round(7.906 * 1.0125, 3)) and st['vbat_scale'] == pytest.approx(1.0125)
    assert st['ads1115'] == {'enabled': True, 'fake': True} and st['vbat_ads_errors'] == 0
    json.dumps(st, allow_nan=False)
    s.set_vbat_scale(1.0)                                                     # 画面の較正で替えた → 次のサンプルから
    assert polls(s, [snap(seq=9)])[0].data['vbat_v'] == pytest.approx(7.906, abs=1e-9)


def test_vehicle_does_not_use_a_stale_board_value(tmp_path):
    rd = FakeReader(7.906, age=0.30)
    s = sensor(tmp_path, rd)
    d = polls(s, [snap(seq=1)])[0].data
    assert math.isnan(d['vbat_v']) and d['vbat_ok'] is False and d['vbat_bad'] is False and d['vbat_src'] is None
    rd.age = 0.24
    d = polls(s, [snap(seq=2)])[0].data
    assert d['vbat_ok'] is True and d['vbat_src'] == 'ads1115'
    rd.v = None                                                               # 一度も読めていない
    d = polls(s, [snap(seq=3)])[0].data
    assert d['vbat_src'] is None and math.isnan(d['vbat_v']) and s.status()['vbat_src'] is None


def test_vehicle_order_board_first_then_esp32(tmp_path):
    rd = FakeReader(7.906)
    s = sensor(tmp_path, rd)
    d = polls(s, [snap(seq=1, code=168, flags=0x27)])[0].data               # ESP32 も 8.40 V を送る
    assert d['vbat_src'] == 'ads1115' and d['vbat_v'] == pytest.approx(7.906)
    rd.age = 1.0                                                              # 基板の値が古い → ESP32 の値(前と同じ)
    d = polls(s, [snap(seq=2, code=168, flags=0x27)])[0].data
    assert d['vbat_src'] == 'esp32' and d['vbat_v'] == pytest.approx(8.40)
    d = polls(s, [snap(seq=3, code=255, flags=0x27)])[0].data               # ESP32 の張り付き 12.75 V は今まで通り使わない
    assert d['vbat_src'] == 'esp32' and d['vbat_ok'] is False and d['vbat_bad'] is True
    rd.age = 0.0
    d = polls(s, [snap(seq=4, code=255, flags=0x27)])[0].data               # 基板の値が新しければ基板
    assert d['vbat_src'] == 'ads1115' and d['vbat_ok'] is True and d['vbat_bad'] is False


@pytest.mark.parametrize('v,sat', [(5.95, False), (9.05, False), (17.61, True), (-0.0005, False), (8.0, True)])
def test_vehicle_out_of_range_or_saturated_board_value_is_not_used(tmp_path, v, sat):
    rd = FakeReader(v, sat=sat)
    s = sensor(tmp_path, rd)
    out = polls(s, [snap(seq=1), snap(seq=2)])
    d = out[-1].data
    assert d['vbat_ok'] is False and math.isnan(d['vbat_v']) and d['vbat_bad'] is True and d['vbat_raw_v'] == pytest.approx(v)
    s._publish(out[-1])
    st = s.status()
    assert st['vbat_v'] is None and st['vbat_supported'] is False and st['vbat_bad_v'] == pytest.approx(v, abs=1e-3)
    json.dumps(st, allow_nan=False)
    tr = vs.VbatTracker()
    for x in out:
        tr.on_sample('VEHICLE', x)
    assert tr.seen is False and tr.bad_seen is True                           # 画面は「電圧の値がおかしい」


def test_vehicle_with_the_real_reader_and_a_fake_board(tmp_path):
    """本物の読み手(偽の ADS1115・偽の時計)→ 車両のサンプル = 読み手の値 × 係数。読み手が読まなくなると 0.25 秒で使わない"""
    k = make()
    s = sensor(tmp_path, k.r, scale=1.01)
    run_for(k.r, k.clk, 0.5)
    d = polls(s, [snap(seq=1)])[0].data
    assert d['vbat_src'] == 'ads1115' and d['vbat_v'] == pytest.approx(A.raw_to_vbat(14708) * 1.01, abs=1e-9)
    k.clk.m += 0.3
    d = polls(s, [snap(seq=2)])[0].data
    assert d['vbat_src'] is None and d['vbat_ok'] is False
    st = s.status()
    assert st['ads1115']['addr'] == '0x48' and st['ads1115']['raw'] == 14708 and st['ads1115']['config_read'] == '0x4283'
    assert st['ads1115']['board_v'] == pytest.approx(7.9056, abs=1e-4) and st['ads1115']['log'] == {'enabled': False}
    assert st['ads1115']['rate_hz'] == pytest.approx(50.0, abs=1.0)           # README 14 章の「rate_hz 約 50」(V4)
    json.dumps(st, allow_nan=False)


def test_vehicle_survives_a_broken_reader(tmp_path):
    """読み手が思わぬ例外を出しても、車両のサンプルは出る(ESP32 の値のまま・数えて残す)"""
    class Broken(FakeReader):
        def fresh(self, max_age_s):
            raise ValueError('壊れた読み手')

        def status(self):
            raise RuntimeError('壊れた status')
    s = sensor(tmp_path, Broken())
    out = polls(s, [snap(seq=1, code=166, flags=0x27)])
    assert len(out) == 1 and out[0].data['vbat_src'] == 'esp32' and out[0].data['vbat_v'] == pytest.approx(8.30)
    st = s.status()
    assert st['vbat_ads_errors'] == 1 and 'ValueError' in st['vbat_ads_error'] and 'RuntimeError' in st['ads1115']['error']
    json.dumps(st, allow_nan=False)


@pytest.mark.parametrize('val,on', [(None, True), ('1', True), ('yes', True), ('0', False), ('off', False), ('False', False),
                                    ('no', False), (' 0 ', False)])
def test_env_switch(monkeypatch, tmp_path, val, on):
    """JETFORMER_VBAT_ADS1115 = 0・off・false・no で使わない(前と同じ動き)。既定は入(作るだけ: バスを開かない・何も書かない)"""
    if val is None:
        monkeypatch.delenv(A.ENV, raising=False)
    else:
        monkeypatch.setenv(A.ENV, val)
    monkeypatch.setattr(vs, 'VBAT_ADS_DIR', tmp_path / 'vbat')
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot',
                          'vbat_calib_path': tmp_path / 'none.json'})
    d = polls(s, [snap(seq=1)])[0].data
    assert d['vbat_src'] is None and math.isnan(d['vbat_v'])                 # 読み手を始めていない・ESP32 は送らない
    st = s.status()
    json.dumps(st, allow_nan=False)
    if on:
        assert isinstance(s.ads, A.Ads1115Vbat) and s.ads.bus is None and not s.ads.running
        assert s.ads.bus_no == 7 and s.ads.log.out_dir == str(tmp_path / 'vbat') and s.ads.range_v == vs.VBAT_RANGE_V
        assert not (tmp_path / 'vbat').exists() and st['ads1115']['enabled'] is True and st['ads1115']['running'] is False
        s.set_vbat_scale(1.02)
        assert s.ads._scale() == 1.02                                         # 記録の scale の欄は車両の係数
    else:
        assert s.ads is None and st['ads1115'] == {'enabled': False, 'why': 'JETFORMER_VBAT_ADS1115=0(使わない)'}
    assert A.make_reader(tmp_path, environ={A.ENV: '0'}) is None and A.make_reader(tmp_path, environ={}).bus is None
    off = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot',
                            'vbat_calib_path': tmp_path / 'none.json', 'vbat_ads1115': False})
    assert off.ads is None


def test_vehicle_built_reader_logs_with_the_vehicle_scale(tmp_path, monkeypatch):
    """VehicleSensor が作った読み手(make_reader のつなぎ)の記録: 場所 data/vbat・scale の欄は車両の係数・vbat_v は画面と同じ値"""
    monkeypatch.delenv(A.ENV, raising=False)
    monkeypatch.setattr(vs, 'VBAT_ADS_DIR', tmp_path / 'vbat')
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot',
                          'vbat_calib_path': tmp_path / 'none.json'})
    clk = Clock()
    dev = FakeAds(clk)
    r = s.ads
    r.bus_factory, r.clock, r.printer = Factory(dev), clk, (lambda m: None)
    r.log.clock, r.log.free_bytes, r.log.fsync, r.log.printer = clk, (lambda p: 50 * GIB), (lambda fd: None), (lambda m: None)
    s.set_vbat_scale(1.0125)
    run_for(r, clk, 0.5, drain=True)
    r.log.close()
    rows = [x for x in _csv(next((tmp_path / 'vbat').glob('*.csv'))) if x['raw']]
    assert rows and all(x['scale'] == '1.01250' for x in rows)
    d = polls(s, [snap(seq=1)])[0].data
    assert d['vbat_v'] == pytest.approx(float(rows[-1]['vbat_v']), abs=1e-4) and dev.bad == []


# ================================================================ 画面・説明欄・較正・収集(サーバーは import するだけ)
def _ns_clock(monkeypatch, t0):
    box = {'ns': int(t0 * 1e9)}
    monkeypatch.setattr(jclock, 'now_ns', lambda: box['ns'])
    return box


def start_with_the_fake_driver(desc=DESC):
    """走行の開始(_start → srv.api_drive_start)は、本物の Driver.start が偽物(fake_start)に替わっているときだけ呼ぶ。
    fake_start が外れる・効かなくなると、本物の走行のスレッドが本物の SensorHub(IMU 0x68・車両 0x28・カメラ・ToF)と PCA9685(0x40)を
    動かす(10/31 確かめの指摘 VR1-N1)。そうなっていたら、呼ぶ前に落とす"""
    now = srv.driver.start
    assert getattr(now, '__func__', now) is not drive.Driver.start, '本物の Driver.start のまま(fake_start が効いていない)'
    return _start(desc)


def feed_vehicle(s, tr, box, t0, t1, erpm=0, hz=100):
    """本物の _poll_snapshot(ESP32 は電圧を送らない)で t0〜t1 秒の車両のサンプルを作り、追跡に入れる"""
    out = []
    for i in range(int(round((t1 - t0) * hz))):
        box['ns'] = int(round((t0 + i / hz) * 1e9))
        s._test_seq = getattr(s, '_test_seq', 0) + 1
        s.bus = _Bus([snap(seq=s._test_seq, erpm=erpm)])
        for x in s._poll_snapshot():
            tr.on_sample('VEHICLE', x)
            out.append(x)
    return out


def test_screen_shows_the_board_voltage_and_v_before_is_filled(tmp_path, tracker, fake_start, monkeypatch):
    s = sensor(tmp_path, FakeReader(7.906))
    monkeypatch.setattr(srv, 'hub', {'VEHICLE': s})
    box = _ns_clock(monkeypatch, 97.0)
    assert srv.vbat_view()['supported'] is False                               # まだ何も来ていない(前と同じ)
    feed_vehicle(s, tracker, box, 97.0, 100.0)
    v = srv.vbat_view()
    assert v['supported'] is True and v['v'] == 7.91 and v['level'] == 'ok' and v['broken_v'] is None and v['scale'] == 1.0
    json.dumps(v, allow_nan=False)
    r = start_with_the_fake_driver()                                          # 本物の Driver.start を呼ばない(VR1-N1)
    d = parse(fake_start[-1]['description'])
    assert d['V_before'] == '7.91' and d['V_src'] == 'board' and r['v_before'] == 7.91 and d['memo'] == 'R1 で 2 回'


def test_screen_shows_a_saturated_board_value_as_broken_and_never_writes_it(tmp_path, tracker, fake_start, monkeypatch):
    s = sensor(tmp_path, FakeReader(17.61, sat=True))
    monkeypatch.setattr(srv, 'hub', {'VEHICLE': s})
    box = _ns_clock(monkeypatch, 97.0)
    feed_vehicle(s, tracker, box, 97.0, 100.0)
    v = srv.vbat_view()
    assert v['supported'] is True and v['v'] is None and v['broken_v'] == pytest.approx(17.61)
    start_with_the_fake_driver()
    assert fake_start[-1]['description'].count('V_before=__') == 1 and srv.VBAT_RUN['v_before'] is None


def test_v_after_and_v_min_from_the_board(tmp_path, tracker, monkeypatch):
    s = sensor(tmp_path, FakeReader(7.80))
    box = _ns_clock(monkeypatch, 100.0)
    log = _log(tmp_path, monkeypatch, bench.vbat_before_description(DESC, 8.10))
    feed_vehicle(s, tracker, box, 100.0, 130.0, erpm=-6000)                   # 走っている(30 秒)
    s.ads.v = 7.85
    feed_vehicle(s, tracker, box, 130.0, 132.0, erpm=0)                       # 止まった
    tracker.clk[0] = 132.0
    srv.VBAT_RUN.update(id=5)
    v_after, v_min, n = srv.vbat_after_run(log, 100.0, 130.0, run_id=5, sleep=lambda dt: None)
    assert v_after == pytest.approx(7.85) and v_min == pytest.approx(7.80) and n == 2
    for d in map(parse, _scene_desc(tmp_path, log)):
        assert d['V_before'] == '8.10' and d['V_after'] == '7.85' and d['Vb_min'] == '7.80'
    assert srv.VBAT_RUN['state'] == 'done' and srv.VBAT_RUN['v_after'] == 7.85


def test_calibration_with_the_board_value(tmp_path, tracker, monkeypatch):
    """画面の較正(POST /api/vbat/calib): 係数 = 電圧計 ÷ 基板の値(係数の前)。保存のあとは 基板 × 係数 = 電圧計(1 回だけ掛ける)。
    もう一度較正しても同じ係数(二重に掛けない)"""
    path = tmp_path / 'cal' / 'vbat_calib.json'
    monkeypatch.setattr(vs, 'VBAT_CALIB_PATH', path)
    s = sensor(tmp_path, FakeReader(7.906))
    monkeypatch.setattr(srv, 'hub', {'VEHICLE': s})
    box = _ns_clock(monkeypatch, 97.0)
    feed_vehicle(s, tracker, box, 97.0, 100.0)
    r = srv.api_vbat_calib({'meter_v': 7.95})
    assert r['scale'] == pytest.approx(7.95 / 7.906, abs=1e-5) and r['board_v'] == 7.91
    assert json.loads(path.read_text())['scale'] == r['scale'] and s.vbat_scale == r['scale']
    d = feed_vehicle(s, tracker, box, 100.0, 100.05)[-1].data
    assert d['vbat_v'] == pytest.approx(7.906 * r['scale'], abs=1e-9) and abs(d['vbat_v'] - 7.95) < 0.001
    feed_vehicle(s, tracker, box, 100.05, 103.0)
    tracker.clk[0] = 103.0
    r2 = srv.api_vbat_calib({'meter_v': 7.95})
    assert r2['scale'] == pytest.approx(r['scale'], abs=2e-4) and r2['board_v'] == 7.91
    srv.api_vbat_calib({'reset': True})
    assert s.vbat_scale == 1.0


def test_collected_scene_keeps_the_board_voltage(tmp_path):
    s = sensor(tmp_path, FakeReader(7.906), scale=1.0125)
    rows = [x.data for x in polls(s, [snap(seq=i) for i in range(1, 6)])]
    p = _scene(tmp_path, '20261103-100000-0001', rows, [_ai(i) for i in range(5)])
    veh = np.load(p / 'VEHICLE.npz')
    assert veh['vbat_ok'].all() and np.allclose(veh['vbat_v'], 7.906 * 1.0125, atol=1e-5)
    sc = json.loads((p / 'scene.json').read_text())
    assert sc['stats']['vbat'] == {'first': 8.005, 'last': 8.005, 'min': 8.005, 'median': 8.005, 'n': 5}


# ================================================================ スレッドの始めと終わり(本物の時計・短い上限)
def test_reader_threads_start_and_stop_without_leftovers(tmp_path):
    dev = FakeAds(time)
    fac = Factory(dev)
    syncs = []
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=fac, clock=time, free_bytes=lambda p: 50 * GIB,
                      fsync=syncs.append, printer=lambda m: None)
    try:
        r.start()                                                             # try の中(V1 の (1))
        assert wait_until(lambda: r.reads >= 10)
        r.start()                                                             # 2 回目の start は何もしない
        assert sum(th.is_alive() for th in r._threads) == 2
    finally:
        t0 = time.monotonic()
        r.stop(timeout=3.0)
    assert time.monotonic() - t0 < 2.0 and not r.running
    assert not any(th.is_alive() and th.name in ('VBAT-ADS1115', 'VBAT-LOG') for th in threading.enumerate())
    assert dev.closed >= 1 and dev.bad == [] and fac.made == [7] and r.log.fh is None and syncs
    files = list((tmp_path / 'vbat').glob('*.csv'))
    assert len(files) == 1 and len([x for x in _csv(files[0]) if x['raw']]) >= 10
    try:
        r.start()                                                             # もう一度始めて止める(新しいファイル)
        assert wait_until(lambda: r.reads >= 20)
    finally:
        r.stop()
    assert not r.running and len(list((tmp_path / 'vbat').glob('*.csv'))) == 2


def test_reader_thread_survives_unexpected_errors():
    class Garbage(FakeAds):
        def read_i2c_block_data(self, addr, reg, n):
            super().read_i2c_block_data(addr, reg, n)
            return ['x', 'y']                                                 # 壊れた値(int にできない)
    dev = Garbage(time)
    r = A.Ads1115Vbat(out_dir=None, log=False, bus_factory=Factory(dev), clock=time, printer=lambda m: None)
    try:
        r.start()
        assert wait_until(lambda: r.errors >= 2) and r.running                # 止まらない・間をあけて読み直す
    finally:
        r.stop()
    assert not r.running and r.reads == 0 and 'ValueError' in r.last_error and r.errors <= 6


def test_vehicle_start_stop_runs_and_stops_the_reader(tmp_path):
    """VehicleSensor.start で読み手も始まり、stop(SensorHub.stop = jetformer-web の終わりの処理)で読み手のスレッドも残らない。
    stop から戻ったときには、記録のスレッドも列の行を書き終えて CSV を閉じている(車両の stop が読み手の stop を待ち 0 で呼ばない。V2)。
    車両のスレッドの名前は README 14 章の名前(SENSOR_THREAD_NAMES)と同じ(見張りは名前では分けない)"""
    dev = FakeAds(time)
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=Factory(dev), clock=time, free_bytes=lambda p: 50 * GIB,
                      fsync=lambda fd: None, printer=lambda m: None)
    s = no_board(sensor(tmp_path, r))                                         # 制御基板の偽物はインスタンスに(V1 の (2))
    hub = SensorHub.__new__(SensorHub)
    hub.sensors, hub.channels = {'VEHICLE': s}, ('VEHICLE',)
    try:
        hub.start()                                                           # try の中(V1 の (1))
        assert wait_until(lambda: r.reads >= 5) and r.running and r.log_running and s.running
        assert s._thread.name == vs.VehicleSensor.name == 'VEHICLE' and s._thread.name in SENSOR_THREAD_NAMES
        assert sorted(th.name for th in r._threads) == ['VBAT-ADS1115', 'VBAT-LOG']
        hub.start()                                                           # 何度 start しても 1 組
        assert sum(th.is_alive() for th in r._threads) == 2
    finally:
        hub.stop()
    assert not r.running and not r.log_running and r.log.fh is None           # stop から戻ったときには閉じている(V2)
    assert not s.running and not s._thread.is_alive() and dev.bad == []
    assert s.errors >= 1 and '制御基板' in s.last_error and s.open_calls >= 1  # 制御基板の偽物は開けない(I2C に触れない)
    rows = _csv(next((tmp_path / 'vbat').glob('*.csv')))
    assert len([x for x in rows if x['raw']]) >= 5 and len(rows) == r.log.rows


def test_server_shutdown_stops_the_sensors():
    """jetformer-web の終わりの処理(lifespan の yield のあと)が hub.stop() を呼ぶ = 読み手も止まる(server.py は変えていない)"""
    src = inspect.getsource(inspect.unwrap(srv.lifespan))
    assert 'yield' in src and src.index('hub.stop()') > src.index('yield')
    assert isinstance(srv.hub['VEHICLE'], vs.VehicleSensor)


# ================================================================ speed_auto は切のまま(走りは変わらない)
def _with_board(sc_, v=7.906):
    """走行ループの偽物の車両の値に、本物の _fill_vbat で基板の電圧を入れる"""
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot',
                          'vbat_calib_path': '/nonexistent/vbat_calib.json', 'vbat_ads1115': FakeReader(v)})
    veh = []
    for t, d in sc_['veh']:
        d = dict(d)
        s._fill_vbat(d)
        veh.append((t, d))
    return dict(sc_, veh=veh)


def test_defaults_keep_speed_auto_off():
    assert drive.DEFAULT_PARAMS['speed_auto'] == 0 and drive.DEFAULT_PARAMS['auto_vref_v'] == 8.35
    assert drive.DEFAULT_PARAMS['speed_ceiling'] == 0 and 'speed_auto' in drive.RESET_ON_BOOT      # 起動のたびに 0(切)
    p = speed_auto.AutoParams()
    assert (p.vboard_lo, p.vboard_hi) == vs.VBAT_RANGE_V == A.RANGE_V


@pytest.mark.parametrize('extra', [{}, {'speed_ceiling': 1}, {'throttle_gain': 0.97, 'start_kick': 1}])
def test_speed_auto_off_drive_output_is_unchanged_by_the_board_voltage(monkeypatch, tmp_path, extra):
    """speed_auto が 0(切。drive_settings.json の今の値)なら、基板の電圧が来ても PCA9685 への出力・速度ゲインは同じ。
    変わるのは影の欄(sa_src が 0 なし → 1 基板 など)だけ"""
    sc0 = sc_build()
    sc1 = _with_board(sc0)
    assert all(d['vbat_src'] == 'ads1115' and d['vbat_ok'] for _, d in sc1['veh'])
    params = dict(SC_PARAMS, speed_auto=0, auto_vref_v=8.35, **extra)
    _, pca0, rec0 = sc_run(drive, monkeypatch, tmp_path, params, sc0)
    _, pca1, rec1 = sc_run(drive, monkeypatch, tmp_path, params, sc1)
    assert len(pca0.pairs) > 900 and pca0.pairs == pca1.pairs and pca0.single == pca1.single
    r0, r1 = [r for _, r in rec0.rows], [r for _, r in rec1.rows]
    assert len(r0) == len(r1) and all(same(a['out'], b['out']) and same(a['gains'], b['gains']) for a, b in zip(r0, r1))
    assert all(r['sa_applied'] is False and r['sa_on'] is False for r in r1)
    src0, src1 = {r['sa_src'] for r in r0 if r['sa_src'] >= 0}, {r['sa_src'] for r in r1 if r['sa_src'] >= 0}   # -1 = 中立の行
    assert src0 == {speed_auto.SRC_NONE} and speed_auto.SRC_BOARD in src1


def test_speed_auto_on_would_raise_the_gain_with_the_board_voltage(monkeypatch, tmp_path):
    """(了承なし・入れていない。README の「入れたら何が変わるか」)speed_auto 1・天井 1 なら、基板の 7.906 V で
    m_ff = 8.35 / 7.906 = 1.056 → 前進の速度ゲインが上がる(出力が変わる)"""
    sc0 = sc_build()
    sc1 = _with_board(sc0)
    params = dict(SC_PARAMS, speed_ceiling=1, speed_auto=1, auto_vref_v=8.35)
    _, pca0, rec0 = sc_run(drive, monkeypatch, tmp_path, params, sc0)
    _, pca1, rec1 = sc_run(drive, monkeypatch, tmp_path, params, sc1)
    assert pca0.pairs != pca1.pairs
    m_ff = [r['sa_m_ff'] for _, r in rec1.rows if np.isfinite(r['sa_m_ff'])]
    assert max(m_ff) == pytest.approx(8.35 / 7.906, abs=1e-3) and speed_auto.ff_of_voltage(7.906) == pytest.approx(8.35 / 7.906)
    assert {r['sa_src'] for _, r in rec0.rows if r['sa_src'] >= 0} == {speed_auto.SRC_NONE}


# ================================================================ 粗探しの指摘の直し(10/31 直す係): 記録のスレッド・範囲と係数・始めと終わり
# S2: 作る係の 68 個では見つからなかった壊し方(R1〜R12)を見つける試験。N1(start の鍵・残った記録のスレッド)・N2(0 での確かめの
# 回数)・N4(読み手が無い理由)。本物のスレッドを使う試験は、偽物のバス・偽物の空き・偽物の fsync と短い上限(待つのは最長 4 秒)
def _writer(r, done):
    th = threading.Thread(target=r._write_loop, args=(r._queue, r.log, done), name='VBAT-LOG-test', daemon=True)
    th.start()
    return th


def test_log_thread_fsyncs_through_gaps_and_does_not_end_while_the_reader_waits(tmp_path):
    """記録のスレッド(_write_loop): 行が来ない間も 1 秒ごとに fsync する(tick)。0.2 秒より長く行が来なくても(読み手が I2C の
    失敗で間をあけている)、読み手が終わった印が無ければ終わらない。印のあとは閉じて終わる(R2・R3)"""
    k = make(tmp_path, log=True)
    done = threading.Event()
    th = None
    try:
        th = _writer(k.r, done)                                               # try の中(V1 の (1))
        k.r._row(k.clk.m, 14708)
        assert wait_until(lambda: k.r.log.rows == 1)
        n = len(k.syncs)
        k.clk.m += 1.5                                                        # 偽の時計で 1.5 秒、行が来ない(失敗の間)
        assert wait_until(lambda: len(k.syncs) > n, 2.0), '行が来ない間に fsync しない'
        assert k.r.log.pending is False and k.r.log.fh is not None            # 書き切った(閉じたのではない)
        time.sleep(0.6)                                                       # 列の待ち 0.2 秒の 3 倍、行が来ない
        assert th.is_alive(), '読み手が間をあけている間に記録のスレッドが終わった'
        k.r._row(k.clk.m, 14709)
        assert wait_until(lambda: k.r.log.rows == 2)
    finally:
        done.set()
        if th is not None:
            th.join(3.0)
    assert not th.is_alive() and k.r.log.fh is None and k.r.log.rows == 2


def test_log_thread_writes_every_queued_row_before_it_closes(tmp_path):
    """読み手が終わった印のあとも、列に残った行を全部書いてから閉じる(止めるときの最後の行を捨てない。R9)"""
    k = make(tmp_path, log=True)
    for i in range(25):
        k.r._row(k.clk.m + 0.02 * i, 14708 + i)
    done = threading.Event()
    done.set()                                                                # 読み手はもう終わった
    try:
        th = _writer(k.r, done)
        th.join(3.0)
    finally:
        done.set()                                                            # 止める印(V1 の (1)。この試験では始めから立っている)
    assert not th.is_alive() and k.r.log.fh is None
    rows = _csv(next((tmp_path / 'vbat').glob('*.csv')))
    assert [int(x['raw']) for x in rows] == list(range(14708, 14733)) and k.r.log.rows == 25 and k.r._queue.empty()


def test_csv_keeps_going_after_i2c_failures_and_after_a_restart(tmp_path):
    """本物のスレッド・偽物のバス: I2C の失敗で 0.4 秒行が来なくても記録のスレッドは生きていて、戻ったあとの行も CSV に書く。
    止めたら CSV の行の数 = 読めた回 + 失敗の回 + 設定を書いた回(捨てた行なし)。止めて始め直しても同じ(2 つ目のファイル。R3・R9・R11)"""
    dev = FakeAds(time)
    syncs = []
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=Factory(dev), clock=time, free_bytes=lambda p: 50 * GIB,
                      fsync=lambda fd: syncs.append(time.monotonic()), printer=lambda m: None)

    def fail_for(seconds):
        end = time.monotonic() + seconds
        dev.fail = lambda op, reg: OSError(errno.ETIMEDOUT, 'Connection timed out') if time.monotonic() < end else None

    for run in (1, 2):
        try:
            r.start()                                                         # try の中(V1 の (1))
            reads0 = r.reads
            assert wait_until(lambda: r.reads >= reads0 + 10)
            e0 = r.errors
            fail_for(0.6)                     # 失敗は 0・0.1・0.3 秒 → 次の試しは 0.7 秒(0.4 秒、列に行が来ない)
            assert wait_until(lambda: r.errors >= e0 + 3 and r.fail_streak == 0 and r.reads >= reads0 + 20, 4.0)
            rows0 = r.log.rows
            assert wait_until(lambda: r.log.rows >= rows0 + 10, 3.0), '%d 回目: 戻ったあとの行が CSV に増えない' % run
            assert r.log_running and r.running
        finally:
            r.stop()
        assert not r.running and not r.log_running
        files = sorted((tmp_path / 'vbat').glob('*.csv'))
        assert len(files) == run
        n = sum(len(_csv(f)) for f in files)
        assert r.dropped_queue == 0 and r.log.dropped == 0
        assert n == r.log.rows == r.reads + r.errors + r.config_writes, (run, n, r.reads, r.errors, r.config_writes)
    assert r.config_writes == 1 and r.errors >= 6 and dev.bad == [] and len(syncs) >= 4


@pytest.mark.parametrize('board,used', [(8.90, False), (5.95, True)])
def test_the_range_is_checked_on_the_scaled_value_in_samples_and_csv(tmp_path, board, used):
    """6.0〜9.0 V は係数を掛けたあとの値(画面・説明欄と同じ値)で見る。係数 1.02 なら 基板 8.90 V → 9.08 V は使わない(vbat_bad)、
    基板 5.95 V → 6.07 V は使う。車両のサンプル(偽の読み手・本物の読み手)と CSV の ok の両方(R1・R6)"""
    scale = 1.02
    s = sensor(tmp_path, FakeReader(board), scale=scale)
    k = make(tmp_path, log=True, scale=scale)
    k.dev.ain0 = board / 4.3
    run_for(k.r, k.clk, 0.3, drain=True)
    s_real = sensor(tmp_path, k.r, scale=scale)
    for sens in (s, s_real):
        d = polls(sens, [snap(seq=1)])[0].data
        assert d['vbat_src'] == 'ads1115' and d['vbat_ok'] is used and d['vbat_bad'] is (not used)
        assert d['vbat_raw_v'] == pytest.approx(board * scale, abs=2e-3)
        assert (d['vbat_v'] == pytest.approx(board * scale, abs=2e-3)) if used else math.isnan(d['vbat_v'])
    k.r.log.close()
    rows = [x for x in _csv(next((tmp_path / 'vbat').glob('*.csv'))) if x['raw']]
    assert len(rows) >= 10 and {x['ok'] for x in rows} == {'1' if used else '0'}
    assert all(abs(float(x['board_v']) - board) < 1e-3 and abs(float(x['vbat_v']) - board * scale) < 2e-3 for x in rows)


def test_vehicle_start_survives_a_reader_that_cannot_start_and_stop_waits_for_the_reader(tmp_path):
    """読み手の start が例外でも、車両の start は例外を出さず、車両のスレッドは動く(失敗は数えて残す。R4)。車両の stop は
    読み手に先に合図し、車両のスレッドのあとで読み手の stop(終わるのを待つ)まで呼ぶ(合図だけで終わらない。R12)。読み手の stop に
    渡す待ちは車両の stop と同じ 3 秒(待ち 0 で呼ばない。V2)。start は try の中・制御基板の偽物はインスタンスに(R4 の壊し方でも、
    車両のスレッドを試験の外に残さない・本物の SMBus に届かない。V1)"""
    class BadStart(FakeReader):
        def start(self):
            self.calls.append('start')
            raise RuntimeError('偽物: 読み手を始められない')
    for rd, bad in ((BadStart(), True), (FakeReader(), False)):
        s = no_board(sensor(tmp_path, rd))                                    # 制御基板(0x28)の偽物: つながっていない形
        try:
            s.start()                                                         # try の中(V1 の (1))
            assert s.running and wait_until(lambda: s.errors >= 1)          # 車両のスレッドは動いている(制御基板を開こうとする)
            assert s._thread.name in SENSOR_THREAD_NAMES and s.open_calls >= 1
            assert s.vbat_ads_errors == (1 if bad else 0) and ('RuntimeError' in s.vbat_ads_error) is bad
        finally:
            s.stop()
        assert not s.running and rd.calls == ['start', 'request_stop', 'stop']
        assert rd.stop_timeouts == [3.0]                                      # 読み手が終わるのを 3 秒まで待つ(V2)
        json.dumps(s.status(), allow_nan=False)


def test_reader_threads_are_daemons_and_sleep_between_reads(tmp_path):
    """スレッドは daemon(どこかで固まってもプロセスの終わりを止めない。R7)。読む間は待つ(回り続けない。R5):
    本物の時計の 0.5 秒で、読みは 50 Hz を越えず、時計を見る回数も読みの数に比例する分だけ"""
    class CountingClock(object):
        def __init__(self):
            self.n = 0

        def monotonic(self):
            self.n += 1
            return time.monotonic()

        def time(self):
            return time.time()
    clk = CountingClock()
    dev = FakeAds(clk)
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=Factory(dev), clock=clk, free_bytes=lambda p: 50 * GIB,
                      fsync=lambda fd: None, printer=lambda m: None)
    try:
        r.start()                                                             # try の中(V1 の (1))
        assert sorted(th.name for th in r._threads) == ['VBAT-ADS1115', 'VBAT-LOG'] and all(th.daemon for th in r._threads)
        assert all(th.name in SENSOR_THREAD_NAMES for th in r._threads)       # README 14 章の名前と同じ
        assert wait_until(lambda: r.reads >= 5)
        n0, reads0, t0 = clk.n, r.reads, time.monotonic()
        time.sleep(0.5)
        calls, reads, dt = clk.n - n0, r.reads - reads0, time.monotonic() - t0
    finally:
        r.stop()
    assert 1 <= reads <= 50 * dt + 3, (reads, dt)
    assert calls <= 20 * (reads + 5), (calls, reads)                          # 待たずに回ると 0.5 秒で何万回


def test_start_after_a_stop_that_timed_out_on_the_disk_starts_reading_with_a_new_file(tmp_path):
    """前の stop の待ちを越えて記録のスレッドが残った(fsync が SD カードで止まった形): 次の start でも読み手は始まり、新しい列・新しい
    ファイルに書く(同じファイルに 2 つで書かない。N1)。ディスクが戻ると、残った方は自分の列の行だけを前のファイルに書き終えて閉じる
    = 新しい読み手が動いている間に終わり、前のファイルの mono_us はどれも新しいファイルの最初の行より前(列が同じだと、新しい行を
    取り合って終わらず、前のファイルに新しい時刻の行が入る。V3)"""
    dev = FakeAds(time)
    gate, state, msgs = threading.Event(), {'block': False}, []

    def fsync(fd):
        if state['block']:
            state['block'] = False
            gate.wait(5.0)                                                    # ディスクが止まった形(5 秒まで)
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=Factory(dev), clock=time, free_bytes=lambda p: 50 * GIB,
                      fsync=fsync, printer=msgs.append)
    old_log = old_writer = None
    try:
        r.start()                                                             # try の中(V1 の (1))
        old_log, old_writer = r.log, r._writer
        assert wait_until(lambda: r.reads >= 10)
        state['block'] = True                                                 # 次の fsync(1 秒ごと)で止まる
        assert wait_until(lambda: not state['block'], 3.0)
        r.stop(timeout=0.3)
        assert not r.running and old_writer.is_alive()                        # 読み手は終わった・記録のスレッドは残った
        assert r.start() is True                                              # それでも読み手を始める(前は何もしなかった)
        assert r.running and r.stale_writers == 1 and r.log is not old_log and r._writer is not old_writer
        reads0 = r.reads
        assert wait_until(lambda: r.reads >= reads0 + 10 and r.log.rows >= 10)
        gate.set()                                                            # ディスクが戻った(新しい読み手は動いたまま)
        assert wait_until(lambda: not old_writer.is_alive(), 2.0), '残った記録のスレッドが終わらない(新しい読み手の行を取り続ける)'
        assert r.running and r.log_running and old_log.fh is None             # 新しい読み手・記録は動いたまま
        rows1 = r.log.rows
        assert wait_until(lambda: r.log.rows >= rows1 + 5)                    # 新しい記録は続く
        assert any('新しいファイル' in m for m in msgs) and r.status()['stale_writers'] == 1
    finally:
        gate.set()
        r.stop()
        if old_writer is not None:
            old_writer.join(3.0)
    assert not old_writer.is_alive() and not r.running and not r.log_running
    assert old_log.fh is None and r.log.fh is None and old_log.path != r.log.path
    files = sorted((tmp_path / 'vbat').glob('*.csv'))
    assert len(files) == 2 and sum(len(_csv(f)) for f in files) == old_log.rows + r.log.rows
    assert old_log.rows >= 10 and r.log.rows >= 10 and dev.bad == []
    old_mono = [int(x['mono_us']) for x in _csv(old_log.path)]
    new_mono = [int(x['mono_us']) for x in _csv(r.log.path)]
    assert max(old_mono) < min(new_mono), '前のファイルに、新しい読み手の行が入った(列が同じ)'


def test_two_threads_starting_at_once_make_one_reader(tmp_path):
    """走行のスレッドと API のスレッドが同時に start しても、読み手・記録は 1 組(start・stop は鍵で 1 つずつ。N1)"""
    class SlowClock(object):
        """start の中で読む時計を遅くして、2 つの start が重なる窓を広げる(鍵が無ければ 2 組できる)"""
        def monotonic(self):
            if threading.current_thread().name.startswith('starter'):
                time.sleep(0.05)
            return time.monotonic()

        def time(self):
            return time.time()
    clk = SlowClock()
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=Factory(FakeAds(clk)), clock=clk,
                      free_bytes=lambda p: 50 * GIB, fsync=lambda fd: None, printer=lambda m: None)
    go, res = threading.Barrier(2), []

    def starter():
        go.wait(2.0)
        res.append(r.start())
    ths = [threading.Thread(target=starter, name='starter-%d' % i, daemon=True) for i in range(2)]
    try:
        for th in ths:
            th.start()
        for th in ths:
            th.join(3.0)
        alive = sorted(th.name for th in threading.enumerate() if th.is_alive() and th.name in ('VBAT-ADS1115', 'VBAT-LOG'))
        assert sorted(res) == [False, True] and alive == ['VBAT-ADS1115', 'VBAT-LOG']
    finally:
        r.stop()
    assert not any(th.is_alive() and th.name in ('VBAT-ADS1115', 'VBAT-LOG') for th in threading.enumerate())


def test_zero_values_check_the_config_at_most_every_0_2_s():
    """電池を外した(AIN0 ≈ 0 V で変換の値が 0 が続く): 0 での設定の確かめは 0.2 秒に 1 回まで(読むたびに確かめて ADS1115 の取引を
    約 2 倍にしない)。50 Hz の読みは続き、そのあとの電源の入り直し(設定 0x8583)も 0.5 秒以内に見つけて書き直す(N2)"""
    k = make()
    run_for(k.r, k.clk, 0.5)
    k.dev.ain0 = 0.0
    n0 = len(k.dev.calls)
    run_for(k.r, k.clk, 2.0)
    calls = k.dev.calls[n0:]
    conv = [c for c in calls if c[0] == 'read' and c[2] == 0]
    cfg = [c for c in calls if c[0] == 'read' and c[2] == 1]
    assert 99 <= len(conv) <= 101                                             # 50 Hz のまま
    assert len(cfg) <= 2.0 / A.ZERO_CHECK_S + 2.0 / A.CONFIG_CHECK_S + 2, len(cfg)   # 0.2 秒に 1 回 + 1 秒ごと(前は 100 回)
    assert k.r.fresh(0.25)['raw'] == 0 and k.r.config_writes == 1 and k.r.errors == 0 and A.ZERO_CHECK_S == 0.2
    k.dev.power_cycle()
    run_for(k.r, k.clk, 0.5)
    assert k.r.reconfigs == 1 and k.dev.cfg == 0x4283 and k.dev.bad == []


def test_status_rate_hz_counts_the_reads_per_second():
    """status の rate_hz(README 14 章で「約 50」を見る数)は、読めた回の時刻から数える: 50 Hz で読めば約 50・読みが 1 秒より長く
    止まると 0(読みの時刻を残さないと、いつも 0 になる。V4)"""
    k = make()
    run_for(k.r, k.clk, 2.0)
    st = k.r.status()
    assert st['rate_hz'] == pytest.approx(50.0, abs=0.5) and st['hz'] == 50.0 and st['reads'] >= 99
    k.dev.fail = lambda op, reg: OSError(errno.ETIMEDOUT, 'Connection timed out')
    run_for(k.r, k.clk, 1.5)                                                  # 読めない 1.5 秒
    assert k.r.status()['rate_hz'] == 0.0
    k.dev.fail = None
    run_for(k.r, k.clk, 3.0)                                                  # 戻る(1.6 秒あけたあと読み直す)
    assert k.r.status()['rate_hz'] == pytest.approx(50.0, abs=0.5)


def test_board_already_in_continuous_mode_is_not_written_and_the_first_row_has_an_empty_note(tmp_path):
    """基板がもう 0x4283(前の回の連続変換のまま = 同じ電源のまま jetformer-web を起動し直した・センサーを止めて始め直した)なら、
    設定を書かない: その回の CSV の最初の行は raw に値・cfg 4283・note 空(正しい形。README 7・14 章。V5)"""
    clk = Clock()
    dev = FakeAds(clk)
    dev.cfg, dev.t_cont = A.CONFIG_RUN & 0x7FFF, clk.m - 1.0                  # 前の回の設定のまま(変換も動いている)
    k = make(tmp_path, log=True, dev=dev, clk=clk)
    run_for(k.r, k.clk, 1.0, drain=True)
    k.r.log.close()
    rows = _csv(next((tmp_path / 'vbat').glob('*.csv')))
    assert (rows[0]['raw'], rows[0]['cfg'], rows[0]['note'], rows[0]['err']) == ('14708', '4283', '', '0')
    assert [c for c in dev.calls if c[0] == 'write'] == [] and k.r.config_writes == 0 and k.r.reconfigs == 0
    assert all(x['note'] == '' and x['raw'] for x in rows) and dev.bad == []


def test_status_says_why_there_is_no_reader(tmp_path, monkeypatch):
    """環境変数は入なのに読み手を作れなかった(make_reader の例外): status の ads1115.why に本当の理由を出す(「使わない(設定)」に
    しない。N4)。車両の値はそのまま"""
    monkeypatch.delenv(A.ENV, raising=False)

    def broken(*a, **k):
        raise PermissionError(errno.EACCES, '偽物: data/vbat を作れない')
    monkeypatch.setattr(A, 'make_reader', broken)
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot',
                          'vbat_calib_path': tmp_path / 'none.json'})
    st = s.status()
    assert s.ads is None and st['ads1115']['enabled'] is False and 'PermissionError' in st['ads1115']['why']
    assert '作れません' in st['ads1115']['why'] and 'PermissionError' in st['vbat_ads_error']
    assert polls(s, [snap(seq=1, code=166, flags=0x27)])[0].data['vbat_src'] == 'esp32'
    json.dumps(st, allow_nan=False)


# ================================================================ V1 の見張りそのもの(10/31 直す係の 2 回目・3 回目): 本物の /dev/i2c は開かない
def _fake_chr(major, minor):
    """os.stat・os.fstat の偽物の答え(文字の装置 major:minor)。本物の装置は見ない"""
    return types.SimpleNamespace(st_mode=stat.S_IFCHR | 0o600, st_rdev=os.makedev(major, minor))


def test_i2c_guard_is_installed_before_the_jetformer_and_test_imports(request):
    """(4) 見張りは、このファイルの jetformer・test_* の import より前に入る(VR1-S1。server の import の間 = SensorHub() が車両の
    センサーと電圧の読み手を作る間に、作るときにバスを開く・スレッドを始める形に壊れても止める)。台本の並びを AST で見る(並べ替えの
    道具などで import を見張りの上へ動かすと落ちる)。フックは 1 つだけ・車両のスレッドの名前は README と同じ"""
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))

    def hook(n):
        return isinstance(n, ast.Call) and ast.unparse(n.func) == 'sys.addaudithook'

    def ours(n):
        if not isinstance(n, (ast.Import, ast.ImportFrom)):
            return False
        mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or '']
        return any(m.split('.')[0] == 'jetformer' or m.startswith('test_') for m in mods)

    def runs_at_import(stmt, pred):
        """モジュールの直下の文 stmt を import で実行すると pred の節を通るか(関数・クラス・lambda の中は数えない)"""
        todo = [stmt]
        while todo:
            n = todo.pop()
            if pred(n):
                return True
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                todo.extend(ast.iter_child_nodes(n))
        return False
    assert len([n for n in ast.walk(tree) if hook(n)]) == 1                          # フックを入れる所は 1 つだけ
    hook_at = [i for i, st in enumerate(tree.body) if runs_at_import(st, hook)]
    imports_at = [i for i, st in enumerate(tree.body) if runs_at_import(st, ours)]
    assert len(hook_at) == 1 and imports_at and hook_at[0] < imports_at[0], (hook_at, imports_at[:3])   # 見張りが先
    assert sum(1 for n in ast.walk(tree) if ours(n)) >= 15
    g = I2C_GUARD
    assert g.installed and sys._jf_test_i2c_guard is g and g.armed == request.node.nodeid and IMPORT_MARK.startswith('<import')
    assert vs.VehicleSensor.name == SENSOR_THREAD_NAMES[0] == 'VEHICLE'


def test_i2c_guard_blocks_dev_i2c_opens_in_any_thread_at_any_time(tmp_path, request):
    """(4) 'open' の見張り: os.open と open() の 'open' の知らせがフックに届く(smbus2 0.6.1 は os.open('/dev/i2c-N', O_RDWR) で開く =
    台本を読む)。入れたあとは、試験の間でも外(試験と試験の間・このファイルの試験のあと)でも、どの名前のスレッドでも止める(VR1-N1:
    前は試験の外では名前が VEHICLE・VBAT-… のスレッドだけ)。パスの見分け(bytes・Path・形の違う書き方・相対の i2c-N・/dev/char/89:N・
    /proc/self/fd/N などの別の名前。別の名前は realpath・stat を偽物にして形だけ見る)。'open' 以外の知らせは止めない。
    本物の /dev/i2c は開かない(sys.audit で知らせを出すだけ・装置の無い名前 SELFTEST)"""
    g = I2C_GUARD
    assert g.installed and g.armed == request.node.nodeid and g.would_block('MainThread') and g.would_block('ADS-READER')
    import smbus2                                                            # noqa: F401(smbus2.SMBus はこの試験の間は偽物)
    src = inspect.getsource(sys.modules['smbus2.smbus2'].SMBus.open)          # 本物の開き方の台本を読むだけ(作らない)
    assert 'self.fd = os.open(filepath, os.O_RDWR)' in src and '"/dev/i2c-{}".format(bus)' in src   # 見張る形で開く(0.6.1)
    probe = str(tmp_path / 'probe.txt')
    g.watch, n0 = probe, g.watch_hits
    try:
        os.close(os.open(probe, os.O_WRONLY | os.O_CREAT, 0o600))            # os.open → 'open' の知らせ → このフック
        with open(probe, encoding='utf-8'):                                  # open() も
            pass
    finally:
        g.watch = None
    assert g.watch_hits == n0 + 2
    with pytest.raises(PermissionError):                                     # 試験の間(主のスレッドでも)
        sys.audit('open', g.SELFTEST, None, os.O_RDWR)
    got = {}

    def opener():
        try:
            sys.audit('open', g.SELFTEST, None, os.O_RDWR)
            got[threading.current_thread().name] = 'open'
        except PermissionError:
            got[threading.current_thread().name] = 'blocked'
    names = SENSOR_THREAD_NAMES + ('VBAT-LOG-test', 'ADS-READER', 'IMU', 'drive', 'Thread-9', 'AnyIO worker thread')
    armed, g.armed = g.armed, None                                           # 試験の外の形(試験と試験の間・このファイルの試験のあと)
    try:
        for name in names:
            th = threading.Thread(target=opener, name=name, daemon=True)
            th.start()
            th.join(2.0)
        try:
            sys.audit('open', g.SELFTEST, None, os.O_RDWR)                   # 主のスレッドも(試験の外の形)
            got['main'] = 'open'
        except PermissionError:
            got['main'] = 'blocked'
    finally:
        g.armed = armed
    assert got == dict({n: 'blocked' for n in names}, main='blocked')
    sys.audit('jetformer.selftest.not_open', g.SELFTEST)                     # 'open' 以外の知らせは止めない
    assert [I2cGuard.i2c_path(p) for p in ('/dev/i2c-7', b'/dev/i2c-7', Path('/dev/./i2c-7'), '/tmp/../dev/i2c-7', '/dev/i2c', 7,
                                           None, '/tmp/x.csv', '/sys/bus/i2c/devices', 'i2c_guard_plugin.py', '/dev/null')] == \
        ['/dev/i2c-7', '/dev/i2c-7', '/dev/i2c-7', '/dev/i2c-7', '/dev/i2c', None, None, None, None, None, None]
    assert I2cGuard.i2c_path('i2c-7') == os.path.join(os.getcwd(), 'i2c-7')  # 相対の i2c-N(dir_fd で /dev を指すかも)は止める側

    def keep(p):
        return p

    def no_such(p):
        raise FileNotFoundError(errno.ENOENT, 'no such', p)

    def boom(p):
        raise RuntimeError('思わぬ失敗')
    assert I2cGuard.i2c_path('/dev/char/89:7', realpath=lambda p: '/dev/i2c-7') == '/dev/char/89:7 -> /dev/i2c-7'
    assert I2cGuard.i2c_path('/proc/self/fd/9', realpath=lambda p: '/dev/i2c-7') == '/proc/self/fd/9 -> /dev/i2c-7'
    assert I2cGuard.i2c_path('/dev/odd', realpath=keep, stat_fn=lambda p: _fake_chr(89, 3)) == '/dev/odd(文字の装置 89:3)'
    assert I2cGuard.i2c_path('/dev/odd', realpath=keep, stat_fn=lambda p: _fake_chr(1, 3)) is None    # /dev/null と同じ番号
    assert I2cGuard.i2c_path('/dev/odd', realpath=keep, stat_fn=no_such) is None                      # 無い(open も失敗する)
    assert I2cGuard.i2c_path('/dev/odd', realpath=boom) == '/dev/odd'          # 分ける途中の思わぬ失敗は止める側
    assert I2cGuard.i2c_path('/tmp/x.csv', realpath=boom, stat_fn=boom) is None   # /dev・/proc・i2c でない名前は見ない(遅くしない)
    took = g.take()                                                          # 止めた分を受け取る(後片付けで落とさない)
    assert [(t[1], t[2], t[4]) for t in took] == \
        [('MainThread', g.SELFTEST, 'open')] + [(n, g.SELFTEST, 'open') for n in names] + [('MainThread', g.SELFTEST, 'open')]
    assert took[0][3] == armed and all(t[3] is None for t in took[1:])


def test_i2c_guard_blocks_i2c_ioctls_and_fails_closed(tmp_path):
    """(4) 'fcntl.ioctl' の見張り: smbus2 は 1 回の取引ごとに ioctl(I2C_SMBUS)を呼ぶ = 見張りより前に開いた fd での取引も止まる。
    I2C の要求(番号は smbus2 の台本と同じ)で、fd が i2c-dev(文字の装置 89)・開いていない・分からないなら止める。ふつうのファイルの
    fd・I2C でない要求は止めない。記録が思わぬ失敗をしても止める(通す側に倒れない。VR1-N1)。'open'・'fcntl.ioctl' 以外の知らせは
    すぐ返す(関数の最初の文)。本物の装置は開かない(開いていない fd 番号 BAD_FD・偽物の fstat)"""
    g = I2C_GUARD
    import smbus2                                                            # noqa: F401
    mod = sys.modules['smbus2.smbus2']
    assert {getattr(mod, n) for n in ('I2C_SLAVE', 'I2C_SLAVE_FORCE', 'I2C_FUNCS', 'I2C_RDWR', 'I2C_SMBUS', 'I2C_PEC')} == set(I2C_IOCTLS)
    assert 'ioctl(self.fd, I2C_SMBUS, msg)' in inspect.getsource(mod.SMBus.read_i2c_block_data)   # 取引ごとに I2C_SMBUS
    with pytest.raises(PermissionError):
        sys.audit('fcntl.ioctl', g.BAD_FD, 0x0720, 0)                       # 開いていない fd の I2C_SMBUS → 止める
    with open(tmp_path / 'plain.txt', 'w', encoding='utf-8') as fh:
        assert I2cGuard.i2c_ioctl(fh.fileno(), 0x0720) is None              # ふつうのファイル(i2c-dev でない)は止めない
        sys.audit('fcntl.ioctl', fh.fileno(), 0x0703, 0x48)
    sys.audit('fcntl.ioctl', g.BAD_FD, 0x5401, 0)                           # I2C でない要求(TCGETS)は止めない
    assert I2cGuard.i2c_ioctl(5, 0x0703, fstat=lambda fd: _fake_chr(89, 7)) == 'I2C_SLAVE(fd 5: i2c-7)'
    assert I2cGuard.i2c_ioctl(5, 0x0707, fstat=lambda fd: _fake_chr(4, 1)) is None          # 端末など
    assert I2cGuard.i2c_ioctl(5, 'x') is None and I2cGuard.i2c_ioctl(5, 0x0720, fstat=lambda fd: None).startswith('I2C_SMBUS')
    lock, lost0 = g.lock, g.lost
    g.lock = None                                                            # 記録が思わぬ失敗をする形(with None: で例外)
    try:
        with pytest.raises(PermissionError):                                 # 見張りの判断を直に呼ぶ(外の見張りが先に止めても、
            g.check(g.SELFTEST)                                              # この見張りが通したかが分かる)
        with pytest.raises(PermissionError):
            g.check_ioctl(g.BAD_FD, 0x0707)
        with pytest.raises(PermissionError):                                 # フックを通しても
            sys.audit('open', g.SELFTEST, None, os.O_RDWR)
    finally:
        g.lock = lock
    assert g.lost == lost0 + 3                                               # 記録できなくても止めた
    g.lost = lost0                                                           # この試験の分(後片付けで落とさない)
    took = g.take()
    assert [(t[2].split('(')[0], t[4]) for t in took] == [('I2C_SMBUS', 'ioctl')]
    body = ast.parse(inspect.getsource(_audit)).body[0].body
    assert ast.unparse(body[0]) == 'if event not in GUARD_EVENTS:\n    return' and GUARD_EVENTS == {'open', 'fcntl.ioctl'}


def test_leftover_threads_of_any_name_are_found_and_signalled(tmp_path):
    """(3) 後片付けの中身: 試験が止め忘れたスレッドを、名前を問わず見つけ(少し待っても終わらない物。VR1-N1: 前は名前が VEHICLE・
    VBAT-… の物だけ)、止める合図だけで終わらせる(後片付けが、monkeypatch が偽物を戻す前にする)。この試験は自分でも止める(finally)"""
    before = set(live_threads())
    r = A.Ads1115Vbat(out_dir=str(tmp_path / 'vbat'), bus_factory=Factory(FakeAds(time)), clock=time,
                      free_bytes=lambda p: 50 * GIB, fsync=lambda fd: None, printer=lambda m: None)
    s = no_board(sensor(tmp_path, FakeReader()))
    ev = threading.Event()
    odd = [threading.Thread(target=lambda e: e.wait(5.0), args=(ev,), name=n, daemon=True) for n in ('ADS-READER', 'rec-io')]
    rec = Recorder(None)                                                     # 収集の JPEG の書き手(止める道が無い = benign)
    assert benign(rec._io_thread) and rec._io_thread.name == 'rec-io' and not any(benign(th) for th in odd)   # 名前では見ない
    try:
        r.start()
        s.start()
        for th in odd:                                                       # 名前の違うスレッド・benign と同じ名前のスレッド
            th.start()
        left = leftover_threads(before, 0.2)                                 # 止め忘れた形(0.2 秒待っても動いている)
        assert sorted(th.name for th in left) == sorted(SENSOR_THREAD_NAMES + ('ADS-READER', 'rec-io'))
        assert rec._io_thread.is_alive() and rec._io_thread not in left
        assert all(signal_stop(th) >= 1 for th in left)
        assert leftover_threads(before, 2.0) == [], '止める合図で終わらない'
        assert not r.running and not r.log_running and not s.running and r.log.fh is None and not any(th.is_alive() for th in odd)
    finally:
        ev.set()
        r.stop()
        s.stop()
