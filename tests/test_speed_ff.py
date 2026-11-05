"""電圧の前送り(speed_ff。2026-11-01)の試験。~/jetformer_reports/bench_20261030/speed_ff/README.md。

ユーザー(11/1)「V2で電圧で最高速を調整がいい」「23秒台を安定して出せるようにすることが目標」。11/1 13:1x の決め D-1 =「自動の補正を
作る」(電圧の前送りだけのスイッチを作って 11/3 に使う。間に合わなければ紙の表で走る)。中身(issues_1103.md 5.1 と main session の決め):
  - drive の新しい設定 speed_ff(0/1・既定 0・起動のたびに 0 = RESET_ON_BOOT)と基準の電圧 ff_vref_v(既定 8.10 V・7.60〜8.80 V の外は
    断る・起動しても残す)。走行中は ON にも OFF にもできない・基準の電圧も変えられない。速さの天井・自動合わせと一緒には ON にできない。
  - 走行開始(▶)のとき、server.py の vbat_start_description が決めた止まった基板の電圧 V0(説明欄の V_before と同じ値)で 1 回だけ
    m = clip(ff_vref_v ÷ V0, 0.85, 1.06) を決め、走行の間は変えない。ON なのに V0 が無ければ走行を始めない(409)。下の切りは 11/1 の
    粗探しの指摘 S2 で 0.94 → 0.85(遅くする向きを止めない)。
  - 掛けるのは AI のフレーム(CH3 が AI 側で dry run でない)の速度ゲインだけ(apply_gains の前進側)。OFF のとき PCA9685 に書く値・
    記録の行・出来事は直す前とビット単位で同じ。
  - 記録: log.json の drive に speed_ff・ff_vref_v・ff_v0・ff_m・ff_gain、ON の走行の AI_CMD に ff_on・ff_m・ff_applied。
  - 画面の版 10 → 11(古い画面の ▶ は 409)。走行タブにスイッチと基準の電圧の欄・見込み・▶ の確認の文の行・走行中の行。
直す前の写し tests/fixtures/speed_ff_1101_before/(drive.py・server.py・app.js・index.html・recorder.py = 11/1 の本物の版 = 10/31 USB を
外して走る のあと。sha256 は SHA256SUMS と tests/speed_ff_1101_back.py の BEFORE_SHA)。前の直しの見張りの試験に足した段は
tests/speed_ff_1101_back.py(この試験が値を照らす)。

実機・GPU・torch・サーバーの起動は使わない(センサー・PCA9685・TensorRT・時計は偽物。走行ループの偽物は tests/test_tether_free.py の
run_tf と同じ形。server は import するだけ。画面は node で本物の app.js を読む)。本物の I2C に届かないように、tests/test_vbat_ads1115.py の
監査の見張り(sys.addaudithook。/dev/i2c の open と I2C の ioctl を止めて数える)を jetformer の import より前に入れ、試験ごとに
止めた数が 0 か・本物の SMBus を作ろうとしていないかを見る。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big --mem 3000M --no-drive-wait -- nice -n 10 \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider tests/test_speed_ff.py
"""
import ast
import collections
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import threading
import time as _time
import types
from pathlib import Path

import pytest

# 本物の I2C に届かないための見張り(tests/test_vbat_ads1115.py の監査の見張り。この import が sys.addaudithook を入れてから jetformer を
# import する)。このファイルの jetformer の import より前に置く(並べ替えない。test_guard_is_installed_first が見る)
import test_vbat_ads1115 as I2C                                                  # noqa: E402,I100

import numpy as np                                                               # noqa: E402
from fastapi import HTTPException                                                # noqa: E402

from jetformer import clock, config, drive                                       # noqa: E402
from jetformer import recorder as rec_mod                                        # noqa: E402
from jetformer.sensors import vehicle as vs                                      # noqa: E402
from jetformer.web import bench as bench_web                                     # noqa: E402
from jetformer.web import server as srv                                          # noqa: E402

import speed_ff_1101_back as FB                                                  # noqa: E402
from test_drive_caps import FakePCA, FakeRecorder, SeqRunner, fake_normalize                # noqa: E402
from test_drive_caps import Feeder as CapsFeeder                                 # noqa: E402
from test_launch_drive import FLOOR_MM, N, PARAMS, REF, T0, launch_scene         # noqa: E402
from test_launch_drive import Feeder as LaunchFeeder                             # noqa: E402
from test_tether_free import _changed as _ast_changed                           # noqa: E402
from test_tether_free import _code_lines, _edit_stmts, _fn                        # noqa: E402
from test_tether_free import (FakeHub, OsProxy, RunRecorder, Unplug, _extract, _run_dir, cruise, fake_runtime,  # noqa: E402
                              same_ev, same_rows, scene_of)
from test_vbat_record import _ai as _ai_row                                      # noqa: E402
from test_vbat_record import _scene as _rec_scene                                # noqa: E402
from test_vbat_record import _veh as _veh_row                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
DRIVE_PY = ROOT / 'jetformer' / 'drive.py'
SERVER_PY = ROOT / 'jetformer' / 'web' / 'server.py'
# 2026-11-03 録画のロックの直しのあとは、その直す前の写し(= この直しの版)で照らす。あとの直しの中身は tests/test_rec_imu_1103.py が見る
RECORDER_PY = ROOT / 'tests' / 'fixtures' / 'rec_imu_1103_before' / 'recorder.py'
APP = STATIC / 'app.js'
INDEX = STATIC / 'index.html'
BEFORE = FB.before_dir(ROOT)
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix8.js'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
START_KW = {'record', 'description', 'tags', 'dry_run', 'params'}               # OFF の走行で server が driver.start に渡す名前(前と同じ)
FF_KW = START_KW | {'ff_v0', 'ff_v0_note'}                                       # ON の走行(電圧を足す)
CONFIRM_ON = '電圧の前送り: ON(電圧 8.42 V・基準 8.10 V・× 0.962 → ゲイン 0.866)'      # 依頼の文の例(0.90 × 8.10 ÷ 8.42)
M_RANGE = (0.85, 1.06)     # 倍率の切り(決めた値。11/1 粗探しの指摘 S2 で下を 0.94 → 0.85)。試験はこの 1 か所から読む


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_BEFORE_MOD = {}


def before_drive():
    """直す前の drive.py(写し。SHA256SUMS・BEFORE_SHA を照らしてから)をモジュールにする。今のファイルの sha256 には頼らない
    (わざと壊した drive.py でも、比べる相手は直す前の写しのまま)"""
    if 'd' not in _BEFORE_MOD:
        FB.check_before(ROOT)
        src = (BEFORE / 'drive.py').read_text(encoding='utf-8')
        assert 'speed_ff' not in src and 'ff_vref_v' not in src                 # 写しは直す前
        spec = importlib.util.spec_from_loader('jetformer._drive_before_speed_ff', loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'jetformer'
        exec(compile(src, str(BEFORE / 'drive.py'), 'exec'), mod.__dict__)
        _BEFORE_MOD['d'] = mod
    return _BEFORE_MOD['d']


# ---------------------------------------------------------------- 本物の I2C に届かないための見張り(試験ごと)
@pytest.fixture(autouse=True)
def no_real_i2c(monkeypatch):
    """試験の前・後に、監査の見張りが止めた物が 0 か(このファイルの試験が /dev/i2c を開こうとしていない)。本物の SMBus を作ると落ちる"""
    early = I2C.I2C_GUARD.take()
    assert not early, '試験の前に /dev/i2c の open・I2C の ioctl を止めていました: %r' % early[:5]
    hit = []

    def smbus(*a, **k):
        hit.append(a)
        raise AssertionError('本物の SMBus を作ろうとした %r' % (a,))
    import smbus2
    monkeypatch.setattr(smbus2, 'SMBus', smbus)
    monkeypatch.setattr(vs, 'SMBus', smbus)
    monkeypatch.setattr(I2C.A, '_smbus', smbus)
    I2C.I2C_GUARD.armed = 'tests/test_speed_ff.py'
    yield
    I2C.I2C_GUARD.armed = None
    blocked = I2C.I2C_GUARD.take()
    assert not blocked and not hit, '本物の I2C に届こうとした: %r %r' % (blocked[:5], hit[:5])


def test_guard_is_installed_first():
    """監査の見張りは入っている・このファイルの jetformer の import より前(import の並び)。見張りの試験の名前は止める"""
    assert I2C.I2C_GUARD.installed is True
    src = Path(__file__).read_text(encoding='utf-8')
    assert src.index('import test_vbat_ads1115 as I2C') < src.index('from jetformer import clock, config, drive')
    assert I2C.I2cGuard.i2c_path('/dev/i2c-7') is not None and I2C.I2cGuard.i2c_path('/tmp/x') is None


# ---------------------------------------------------------------- 決まり(値・式)
def test_settings_defaults_and_limits():
    """speed_ff は既定 0・0/1・起動のたびに 0。ff_vref_v は既定 8.10 V・7.60〜8.80 V・起動しても残す(RESET_ON_BOOT に入れない)。
    m の上下 0.85・1.06(M_RANGE。下は 11/1 の粗探しの指摘 S2 で 0.94 → 0.85)、V0 の範囲 6.0〜9.0 V(= 基板の電圧の範囲
    vehicle.VBAT_RANGE_V)。基準の電圧の欄の刻み 0.05 V。auto_vref_v(自動合わせ用)は変えていない。画面の版 11(サーバーと画面)"""
    assert drive.DEFAULT_PARAMS['speed_ff'] == 0 and drive.DEFAULT_PARAMS['ff_vref_v'] == 8.10
    assert drive.PARAM_LIMITS['speed_ff'] == (0, 1) and drive.PARAM_LIMITS['ff_vref_v'] == (7.6, 8.8)
    assert 'speed_ff' in drive.RESET_ON_BOOT and 'ff_vref_v' not in drive.RESET_ON_BOOT
    assert drive.FF_M_RANGE == M_RANGE == (0.85, 1.06) and drive.FF_V0_RANGE == tuple(vs.VBAT_RANGE_V) == (6.0, 9.0)
    assert drive.FF_VREF_STEP == 0.05 and drive.SPEED_FF_VERSION == '2026-11-01'
    assert drive.DEFAULT_PARAMS['auto_vref_v'] == 8.35 and drive.PARAM_LIMITS['auto_vref_v'] == (7.0, 9.0)
    assert 'auto_vref_v' not in drive.RESET_ON_BOOT
    before = before_drive()
    assert 'speed_ff' not in before.DEFAULT_PARAMS
    assert {k: v for k, v in drive.DEFAULT_PARAMS.items() if k not in ('speed_ff', 'ff_vref_v')} == before.DEFAULT_PARAMS
    assert {k: v for k, v in drive.PARAM_LIMITS.items() if k not in ('speed_ff', 'ff_vref_v')} == before.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == before.RESET_ON_BOOT + ('speed_ff',)
    assert srv.DRIVE_CLIENT_VERSION == 11
    js = APP.read_text(encoding='utf-8')
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['11'] and re.findall(r'const body = \{ client: (\d+),', js) == ['11']


@pytest.mark.parametrize('vref,v0', [(8.10, 8.42), (8.10, 8.10), (8.10, 7.90), (7.60, 8.40), (8.80, 8.31), (8.10, 7.6415),
                                     (8.10, 8.6170), (8.10, 6.0), (8.10, 9.0), (7.6, 7.6), (8.8, 8.8), (7.80, 8.40), (7.80, 8.57),
                                     (7.60, 8.94), (7.60, 9.0)])
def test_multiplier_formula(vref, v0):
    """m = clip(基準 ÷ V0, 0.85, 1.06)。端の値の V0 も使う。結果は Python の float の割り算そのまま(丸めない)"""
    m = drive.ff_multiplier(vref, v0)
    assert m == min(max(vref / v0, M_RANGE[0]), M_RANGE[1]) and isinstance(m, float)
    assert M_RANGE[0] <= m <= M_RANGE[1]


def test_multiplier_examples():
    """例: 8.10 ÷ 8.42 = 0.962(依頼の文の例)。上は 1.06(低い電圧)で切る。下の 0.85 は、受け付ける基準の電圧(7.60 V から)と
    8.94 V までの電圧では当たらない(11/1 の粗探しの指摘 S2: 下の切りが 0.94 だと、新しいパックで列を左へ動かしても遅くならなかった)"""
    assert round(drive.ff_multiplier(8.10, 8.42), 3) == 0.962 and round(0.90 * drive.ff_multiplier(8.10, 8.42), 3) == 0.866
    assert drive.ff_multiplier(8.10, 7.0) == 1.06 and drive.ff_multiplier(8.10, 8.10 / 1.06) == pytest.approx(1.06)
    # 指摘 S2 の例(計算の例): 基準 8.00 の本のあと 2 列 左で 7.80・V0 8.40 V → ねらいの 7.80 ÷ 8.40 = 0.929(前は 0.94 で止まった)
    assert drive.ff_multiplier(7.80, 8.40) == 7.80 / 8.40 and drive.ff_multiplier(7.80, 8.40) < 0.93 < drive.ff_multiplier(8.00, 8.40)
    # 係 A1 の例: 列 7.8・8.57 V → 0.90 × 7.8 ÷ 8.57 = 0.819(紙の表の 0.82 と同じ。前は 0.90 × 0.94 = 0.846)
    assert round(0.90 * drive.ff_multiplier(7.80, 8.57), 3) == 0.819
    # 紙の表(prep_1103/z_data/z2_gain_table.py)の全部のマス(列 7.8〜8.6・画面の電圧 7.90〜8.59 V)は、× のマス(ゲイン 0.95 超え)の
    # ほか切りに当たらない(= 紙の表と同じ式)
    for col in (7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6):
        for v in (7.90, 7.95, 8.10, 8.25, 8.40, 8.55, 8.59):
            if 0.90 * col / v <= 0.95:
                assert drive.ff_multiplier(col, v) == col / v, (col, v)
    assert drive.ff_multiplier(7.60, 8.94) == 7.60 / 8.94 and drive.ff_multiplier(7.60, 9.0) == 0.85     # 下は受け付ける端の組だけ


@pytest.mark.parametrize('vref,v0', [(8.10, 5.99), (8.10, 9.01), (8.10, float('nan')), (8.10, float('inf')), (8.10, None),
                                     (8.10, 'x'), (7.59, 8.0), (8.81, 8.0), (float('nan'), 8.0), (None, 8.0), (8.10, -8.0)])
def test_multiplier_refuses_bad_voltage(vref, v0):
    """V0 が 6.0〜9.0 V の外・数でない、基準が 7.60〜8.80 V の外・数でない → ValueError(黙って 1.00 にしない)"""
    with pytest.raises(ValueError):
        drive.ff_multiplier(vref, v0)


# ---------------------------------------------------------------- 設定(起動・set_params の決まり)
def _driver(monkeypatch, tmp_path, saved=None):
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    if saved is not None:
        (tmp_path / 'drive_settings.json').write_text(json.dumps(saved), encoding='utf-8')
    runs = tmp_path / 'runs'
    if not (runs / 'm1').exists():
        _run_dir(runs, 'm1')
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    return drive.Driver(hub=None, recorder=FakeRecorder())


class _Alive(object):
    def is_alive(self):
        return True

    def join(self, *a):
        pass


def test_boot_turns_off_and_keeps_vref(monkeypatch, tmp_path):
    """起動のたびに speed_ff は 0(設定のファイルに 1 があっても)。ff_vref_v は残す。古い設定のファイル(欄が無い)は既定"""
    d = _driver(monkeypatch, tmp_path, {'params': {'speed_ff': 1, 'ff_vref_v': 8.25, 'throttle_gain': 0.9}})
    assert d.params['speed_ff'] == 0 and d.params['ff_vref_v'] == 8.25 and d.params['throttle_gain'] == 0.9
    d2 = _driver(monkeypatch, tmp_path, {'params': {'throttle_gain': 0.9}})
    assert d2.params['speed_ff'] == 0 and d2.params['ff_vref_v'] == 8.10
    d2.set_params({'speed_ff': 1, 'ff_vref_v': 8.3})
    saved = json.loads((tmp_path / 'drive_settings.json').read_text(encoding='utf-8'))['params']
    assert saved['speed_ff'] == 1 and saved['ff_vref_v'] == 8.3                  # ファイルには書く(起動で戻す)
    d3 = drive.Driver(hub=None, recorder=FakeRecorder())
    assert d3.params['speed_ff'] == 0 and d3.params['ff_vref_v'] == 8.3
    assert d3.status()['params']['speed_ff'] == 0 and d3.status()['ff_spec']['vref_v'] == 8.3


def test_set_params_when_stopped(monkeypatch, tmp_path):
    """止まっているとき: ON・OFF・基準の電圧(範囲の中)を変えられる。範囲の外・数でない値は断る(400 = ValueError)・値は変えない。
    0.5 以上は ON(ほかのスイッチと同じ読み方)"""
    d = _driver(monkeypatch, tmp_path)
    assert d.set_params({'speed_ff': 1})['speed_ff'] == 1
    assert d.set_params({'speed_ff': 0})['speed_ff'] == 0
    assert d.set_params({'speed_ff': 0.7})['speed_ff'] == 1 and d.set_params({'speed_ff': '0'})['speed_ff'] == 0
    for v in (7.6, 8.8, 8.05, '8.2', 8.123):
        assert d.set_params({'ff_vref_v': v})['ff_vref_v'] == float(v)
    for bad in (7.59, 8.81, 9.0, 0.0, 81.0, -8.1):
        with pytest.raises(ValueError, match='基準の電圧'):
            d.set_params({'ff_vref_v': bad})
        assert d.params['ff_vref_v'] == 8.123
    for bad in ('x', float('nan'), float('inf')):
        with pytest.raises(ValueError):
            d.set_params({'ff_vref_v': bad})
    assert d.params['ff_vref_v'] == 8.123 and d.set_params({})['ff_vref_v'] == 8.123


def test_set_params_conflicts_with_ceiling_and_auto(monkeypatch, tmp_path):
    """速さの天井・自動合わせと一緒には ON にできない(どちらの順でも断る・値は変えない)。片方を OFF にすれば ON にできる"""
    d = _driver(monkeypatch, tmp_path)
    d.set_params({'speed_ff': 1})
    for bad in ({'speed_ceiling': 1}, {'speed_ceiling': 1, 'speed_auto': 1}):
        with pytest.raises(ValueError, match='電圧の前送りが ON のあいだは'):
            d.set_params(bad)
        assert d.params['speed_ceiling'] == 0 and d.params['speed_auto'] == 0 and d.params['speed_ff'] == 1
    d.set_params({'speed_ff': 0})
    d.set_params({'speed_ceiling': 1})
    with pytest.raises(ValueError, match='速さの天井・速さのゲインの自動合わせが ON のあいだは'):
        d.set_params({'speed_ff': 1})
    d.set_params({'speed_auto': 1})
    with pytest.raises(ValueError, match='ON のあいだは、電圧の前送りを ON にできません'):
        d.set_params({'speed_ff': 1})
    assert d.params['speed_ff'] == 0
    with pytest.raises(ValueError):
        d.set_params({'speed_ff': 1, 'speed_ceiling': 1, 'speed_auto': 1})
    assert d.set_params({'speed_ceiling': 0, 'speed_ff': 1})['speed_ff'] == 1          # 天井を OFF(自動合わせも OFF)にして ON
    assert d.params['speed_auto'] == 0


def test_set_params_when_running(monkeypatch, tmp_path):
    """走行中: ON にも OFF にもできない・基準の電圧も変えられない(同じ値を送るのはよい)。ほかの設定(速度ゲインなど)は今まで通り"""
    d = _driver(monkeypatch, tmp_path)
    for on in (0, 1):
        d.thread = None
        d.set_params({'speed_ff': on, 'ff_vref_v': 8.1})
        d.thread = _Alive()
        with pytest.raises(ValueError, match='走行中は電圧の前送りを ON にも OFF にもできません'):
            d.set_params({'speed_ff': 1 - on})
        with pytest.raises(ValueError, match='走行中は電圧の前送りの基準の電圧を変えられません'):
            d.set_params({'ff_vref_v': 8.15})
        assert d.params['speed_ff'] == on and d.params['ff_vref_v'] == 8.1
        p = d.set_params({'speed_ff': on, 'ff_vref_v': 8.1, 'throttle_gain': 0.91})
        assert p['throttle_gain'] == 0.91 and p['speed_ff'] == on
        assert d.set_params({'throttle_gain': 0.90})['throttle_gain'] == 0.90
    d.thread = None


# ---------------------------------------------------------------- 走行開始(V0 と m)
def test_start_decides_m_once_and_needs_v0(monkeypatch, tmp_path):
    """ON: V0 から m を決めて st['ff'] に入れる(実のゲイン = 走行開始の throttle_gain × m)。V0 が無い・範囲の外・天井と一緒 → 始めない
    (スレッドを作らない・設定も設定のファイルも変えない)。OFF は V0 を見ない(無くても始める)・st['ff'] は {'on': False}"""
    d = _driver(monkeypatch, tmp_path)
    d._run = lambda *a: None
    d.set_params({'speed_ff': 1, 'ff_vref_v': 8.10, 'throttle_gain': 0.90})
    before = json.loads((tmp_path / 'drive_settings.json').read_text(encoding='utf-8'))
    for v0, note, want in ((None, '車が止まっていません(|eRPM| の中央 300)', '車が止まっていません'), (None, '', 'V0)がありません。'),
                           (5.5, '', '6.0〜9.0 V の外'), (9.4, '', '6.0〜9.0 V の外')):
        with pytest.raises(RuntimeError, match=re.escape(want)) as e:
            d.start('m1', 4, ff_v0=v0, ff_v0_note=note)
        assert '走行は始めていません' in str(e.value) and d.thread is None
    with pytest.raises(ValueError, match='電圧の前送りが ON のあいだは'):
        d.start('m1', 4, params={'speed_ceiling': 1}, ff_v0=8.42)
    assert d.thread is None and d.params['speed_ceiling'] == 0
    with pytest.raises(RuntimeError, match='V0'):                                  # body の値も入れない(断った走行の設定を残さない)
        d.start('m1', 4, params={'throttle_gain': 0.95, 'steer_gain': 1.2}, ff_v0=None)
    assert d.thread is None and d.params['throttle_gain'] == 0.90 and d.params['steer_gain'] == 1.0
    assert json.loads((tmp_path / 'drive_settings.json').read_text(encoding='utf-8')) == before
    d.start('m1', 4, ff_v0=8.42)
    d.thread.join(5)
    ff = d.st['ff']
    m = 8.10 / 8.42
    assert ff == {'on': True, 'version': '2026-11-01', 'vref_v': 8.10, 'v0': 8.42, 'm': m, 'throttle_gain': 0.90, 'gain': 0.90 * m}
    assert d.status()['ff'] == ff and d.status()['ff_spec']['m_range'] == list(M_RANGE)
    assert d.status()['ff_spec']['title'] == '電圧の前送り(倍率 = 基準の電圧 ÷ 走る前の電圧・0.85〜1.06 倍)'
    d.set_params({'speed_ff': 0})
    d.start('m1', 4)
    d.thread.join(5)
    assert d.st['ff'] == {'on': False}
    d.start('m1', 4, ff_v0=8.42, ff_v0_note='x')                                   # OFF なら V0 は使わない
    d.thread.join(5)
    assert d.st['ff'] == {'on': False}
    d.start('m1', 4, params={'speed_ff': 1}, ff_v0=7.5)                             # body の params で ON(低い電圧 → 上の切り)
    d.thread.join(5)
    assert d.st['ff']['on'] is True and d.st['ff']['m'] == 1.06 and d.params['speed_ff'] == 1


def test_start_signature_adds_only_two_keywords():
    """start の引数は直す前と同じ + ff_v0・ff_v0_note(既定 None・'' = 渡さない呼び方は前と同じ)"""
    old = inspect.signature(before_drive().Driver.start)
    new = inspect.signature(drive.Driver.start)
    assert list(new.parameters)[:len(old.parameters)] == list(old.parameters)
    assert [(k, v.default) for k, v in new.parameters.items() if k not in old.parameters] == [('ff_v0', None), ('ff_v0_note', '')]
    for k, v in old.parameters.items():
        assert new.parameters[k] == v, k


def test_bad_saved_vref_refuses_on_start(monkeypatch, tmp_path):
    """設定のファイルを手で書き換えて基準の電圧が範囲の外(9.5)でも、ON の走行は黙って端の値にしない(始めない)"""
    d = _driver(monkeypatch, tmp_path, {'params': {'ff_vref_v': 9.5}})
    d._run = lambda *a: None
    d.params['speed_ff'] = 1
    with pytest.raises(RuntimeError, match='基準の電圧 9.5 V'):
        d.start('m1', 4, ff_v0=8.0)
    assert d.thread is None


# ---------------------------------------------------------------- 走行ループ(偽物。tests/test_tether_free.py の run_tf と同じ形)
def run_ff(mod, monkeypatch, tmp_path, frames, imu, params, v0=None, dry_run=False, base='launch', hook=None):
    """mod(今の drive か直す前の写し)で、本物の Driver.start で走行を始め(params を先に入れる。v0 は今の drive だけ)、start が作った
    st で _loop を決まった時刻で回す → (driver, pca, recorder, unplug)。画面はつながったまま(生存確認はフレームごと)"""
    box = {'ns': T0, 'm': 100.0}
    monkeypatch.setattr(clock, 'now_ns', lambda: box['ns'])
    monkeypatch.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                           time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
    monkeypatch.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    runs = tmp_path / 'runs'
    if not (runs / 'm1').exists():
        _run_dir(runs, 'm1')
    monkeypatch.setattr(mod, 'RUNS_DIR', runs)
    fp = tmp_path / 'floor_ref.npy'
    if base == 'launch':
        np.save(fp, np.full((42, 54), float(FLOOR_MM), np.float32))
    monkeypatch.setattr(mod, 'FLOOR_REF_PATH', fp if base == 'launch' else tmp_path / 'no_floor_ref.npy')
    rec = FakeRecorder()
    d = mod.Driver(hub=None, recorder=rec)
    d._run = lambda *a: None
    d.params = dict(params)
    d.start('m1', 4, dry_run=dry_run, **({'ff_v0': v0} if v0 is not None else {}))
    d.thread.join(5)
    d.params = dict(params)
    pca = FakePCA()
    out = mod._Output(pca, config.PCA9685, REF, lambda: d.params)
    bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
    cam_slot = [None]
    feeder = (LaunchFeeder if base == 'launch' else CapsFeeder)(d, frames, imu, bufs, cam_slot, box)
    unplug = Unplug(feeder, d, box, lambda m: True, None, hook)
    runner = SeqRunner([f['pred'] for f in frames])
    d._loop(d.st, runner, fake_normalize, pca, out, REF, 6, 0.0, cam_slot, unplug, *bufs, dry_run)
    d._stop.clear()
    return d, pca, rec, unplug


def scene(name):
    """場面の名前 → (frames, imu, base, params)。holds = AI・人側・電波切れ・車両の値が古い・ToF ガード・止まっていてブレーキ。
    caps = 区間の速度上限の試験の場面(人側・後退側・弱い前進・CH3 が届かない)。launch = 発進の前進 ON(人側と AI 側の行き来)"""
    if name == 'launch':
        frames, imu = launch_scene()
        return frames, imu, 'launch', dict(PARAMS, launch_boost=1)
    frames, imu, base = scene_of(name)
    return frames, imu, base, dict(PARAMS)


def assert_same_run(a, b):
    """2 つの走行ループの PCA9685 に書いた値・記録の行・出来事・止めた理由・処理したフレームがビット単位で同じ"""
    (da, pa, ra, ua), (db, pb, rb, ub) = a, b
    assert pa.pairs == pb.pairs and pa.single == pb.single
    same_rows(ra.rows, rb.rows)
    assert len(ra.events) == len(rb.events) and all(ka == kb and same_ev(xa, xb) for (ka, xa), (kb, xb) in zip(ra.events, rb.events))
    assert da.st.get('stop_reason', '') == db.st.get('stop_reason', '') and ua.ms == ub.ms


@pytest.mark.parametrize('name', ['cruise', 'holds', 'caps', 'launch'])
@pytest.mark.parametrize('gain', [1.0, 0.9])
def test_off_is_bit_identical_to_before(monkeypatch, tmp_path, name, gain):
    """OFF(既定)の走行は、直す前の写しとビット単位で同じ: PCA9685 に書いた値(組・1 つずつ)・記録の行(欄の名前と値。ff_* の欄は無い)・
    出来事・止めた理由・処理したフレーム。基準の電圧の値・走る前の電圧を渡しても(OFF なら使わない)同じ"""
    frames, imu, base, params = scene(name)
    params = dict(params, throttle_gain=gain)
    ref = run_ff(before_drive(), monkeypatch, tmp_path, frames, imu, params, base=base)
    for p, v0 in ((dict(params, speed_ff=0), None), (dict(params, speed_ff=0, ff_vref_v=8.8), 7.0)):
        cur = run_ff(drive, monkeypatch, tmp_path, frames, imu, p, v0=v0, base=base)
        assert cur[0].st['ff'] == {'on': False}
        assert_same_run(cur, ref)
        assert not any(k.startswith('ff_') for _, r in cur[2].rows for k in r)
        assert all('ff' not in r for r in [cur[0].st['now']] if r)
    assert len(ref[1].pairs) == len(ref[2].rows) > 100


def _ai_frame(r):
    return bool(r['select']) and not r['dry_run'] and not r['hold']


@pytest.mark.parametrize('name', ['cruise', 'holds', 'caps', 'launch'])
@pytest.mark.parametrize('vref,v0', [(8.10, 8.42), (8.10, 7.20), (7.60, 9.0)])
def test_on_changes_only_the_ai_frames(monkeypatch, tmp_path, name, vref, v0):
    """ON(走行開始の V0 で m を決める)の走行は、AI のフレーム(CH3 が AI 側・中立止めでない)では「速度ゲインが throttle_gain × m の
    OFF の走行」と、ほかのフレーム(人側・中立止め)では「OFF の走行」と、PCA9685 に書いた値がビット単位で同じ。= 掛かるのは AI の
    前進の指令だけ(ゲインはもとから前進側だけ = apply_gains。ブレーキ・後退・リカバリー・後退の受け付け・発進の補助と発進の前進の下限・
    中立は今のまま)。舵はどのフレームも同じ。m は走行のあいだ同じ値(8.42 V = 0.962・7.20 V = 上の切り 1.06・基準 7.60 で 9.0 V = 下の
    切り 0.85)。記録の gains[1] は AI 側(dry run でない)の行なら中立止めの行も throttle_gain × m(11/1 粗探しの指摘 S1)・人側は
    throttle_gain のまま"""
    frames, imu, base, params = scene(name)
    tg = 0.9
    m = drive.ff_multiplier(vref, v0)
    assert m in (8.10 / 8.42, M_RANGE[1], M_RANGE[0])
    on = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(params, throttle_gain=tg, speed_ff=1, ff_vref_v=vref), v0=v0, base=base)
    off = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(params, throttle_gain=tg), base=base)
    offm = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(params, throttle_gain=tg * m), base=base)
    rows = [r for _, r in on[2].rows]
    assert len(on[1].pairs) == len(rows) == len(off[1].pairs) == len(offm[1].pairs) > 100
    assert [tuple(map(float, r['out'])) for r in rows] == [tuple(map(float, x)) for x in on[1].pairs]     # 行 i = 書いた i 番目
    n_ai = n_diff = 0
    for i, r in enumerate(rows):
        want = offm[1].pairs[i] if _ai_frame(r) else off[1].pairs[i]
        assert on[1].pairs[i] == want, (i, r['select'], r['hold'], r['assist'])
        assert on[1].pairs[i][0] == off[1].pairs[i][0]                          # 舵は同じ
        assert r['ff_on'] is True and r['ff_m'] == m and r['ff_applied'] is _ai_frame(r)
        assert r['gains'][1] == (tg * m if (r['select'] and not r['dry_run']) else tg)       # AI 側の中立止めの行も × m(指摘 S1)
        assert r['sa_applied'] is False                                          # 自動合わせの欄には数えない(自動合わせは OFF)
        n_ai += _ai_frame(r)
        n_diff += on[1].pairs[i] != off[1].pairs[i]
    assert on[1].single == off[1].single
    assert n_ai > 50 and (n_diff > 20 or m == 1.0)
    assert on[0].st['ff']['m'] == m and on[0].st['ff']['v0'] == v0
    if name in ('holds', 'caps', 'launch'):
        assert any(not r['select'] for r in rows)                                       # 人側のフレームがある場面
    if name in ('holds', 'caps'):
        assert any(r['hold'] for r in rows)                                             # 中立止めのフレームがある場面
    if name == 'launch':
        assert any(r['assist'] == drive.assist_mod.ASSIST_LAUNCH for r in rows)          # 発進の前進のフレームがある場面


def test_on_ai_forward_is_the_scaled_gain(monkeypatch, tmp_path):
    """2 m/s で AI が前進(中立 − 120 µs)し続ける場面: 書いたスロットルは 中立 + (throttle_gain × m) × (−120)(式そのまま)"""
    frames, imu = cruise(6.0)
    tg, v0 = 0.9, 8.42
    m = 8.10 / v0
    d, pca, rec, _ = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(PARAMS, throttle_gain=tg, speed_ff=1, ff_vref_v=8.10),
                            v0=v0)
    ai = [(x, r) for x, (_, r) in zip(pca.pairs, rec.rows) if _ai_frame(r)]
    assert len(ai) > 100
    for x, r in ai:
        assert x[1] == float(np.clip(REF[1] + (tg * m) * (r['raw'][1] - REF[1]), config.DRIVE['min_us'], config.DRIVE['max_us']))
        assert r['raw'][1] == N - 120.0
    assert d.st['now']['ff'] == {'on': True, 'm': round(m, 4), 'v0': 8.42, 'vref_v': 8.1, 'gain': round(tg * m, 4), 'applied': True}


def test_on_dry_run_writes_neutral(monkeypatch, tmp_path):
    """dry run(モーターに出力しない)+ ON: PCA9685 に書くのは中立だけで、OFF の dry run とビット単位で同じ。m は決めて記録する
    (ff_on・ff_m)が、掛けたフレームは無い(ff_applied False・gains は throttle_gain のまま)"""
    frames, imu, base, params = scene('holds')
    on = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(params, throttle_gain=0.9, speed_ff=1), v0=8.42, dry_run=True)
    off = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(params, throttle_gain=0.9), dry_run=True)
    assert on[1].pairs == off[1].pairs and on[1].single == off[1].single
    assert set(on[1].pairs) == {(REF[0], REF[1])}
    rows = [r for _, r in on[2].rows]
    assert rows and all(r['ff_on'] and r['ff_m'] == 8.10 / 8.42 and not r['ff_applied'] and r['gains'][1] == 0.9 for r in rows)


def test_m_stays_and_gain_buttons_still_work(monkeypatch, tmp_path):
    """走行の途中で速度ゲインを変えると(±0.01 のボタン = set_params)、そのあとの AI のフレームは 新しいゲイン × 同じ m。
    途中で speed_ff・基準の電圧を変えようとすると断る(m は変わらない)。車両のサンプルの電圧が変わっても m は同じ"""
    frames, imu = cruise(6.0)
    for i, f in enumerate(frames):                                              # 走っている間に電圧が下がる(負荷)
        f['veh'] = dict(f['veh'], vbat_v=8.40 - 0.002 * i, vbat_ok=True)
    errs = []

    def hook(i, d):
        if i == 60:
            d.thread = _Alive()                                                 # 走行中(set_params の決まり)
            d.set_params({'throttle_gain': 0.95})
        if i == 61:
            for bad in ({'speed_ff': 0}, {'ff_vref_v': 8.3}):
                try:
                    d.set_params(bad)
                except ValueError as e:
                    errs.append(str(e))
    d, pca, rec, _ = run_ff(drive, monkeypatch, tmp_path, frames, imu, dict(PARAMS, throttle_gain=0.9, speed_ff=1, ff_vref_v=8.10),
                            v0=8.42, hook=hook)
    d.thread = None
    m = 8.10 / 8.42
    t61 = frames[61]['cam_t']           # フレーム 60 の前に変えた(走行ループはそのフレームの設定を先に読むので、効くのは 61 から)
    early = {r['gains'][1] for t, r in rec.rows if _ai_frame(r) and t < t61}
    late = {r['gains'][1] for t, r in rec.rows if _ai_frame(r) and t >= t61}
    assert early == {0.9 * m} and late == {0.95 * m}
    assert {r['ff_m'] for _, r in rec.rows} == {m} and len(errs) == 2 and d.params['speed_ff'] == 1


# ---------------------------------------------------------------- 記録(log.json・AI_CMD)
def test_log_json_drive_fields(monkeypatch, tmp_path):
    """収集した走行の log.json の drive: ON は speed_ff 1・ff_vref_v・ff_v0・ff_m・ff_gain(走行開始の throttle_gain × m)。
    OFF は speed_ff 0・ff_vref_v(設定の値)・ほかは None。params にも speed_ff・ff_vref_v(設定の値)。tether_free などは前と同じ"""
    order = []
    fake_runtime(monkeypatch, order)
    monkeypatch.setattr(drive, 'os', OsProxy(lambda: order.append('sync')))
    d = _driver(monkeypatch, tmp_path)

    def quick(st, *a):
        st.update(phase='driving', announced=True, stats={'frames': 1})
        st['stop_reason'] = '停止ボタン'
    d._loop = quick
    for on, v0 in ((1, 8.42), (0, None)):
        rec = RunRecorder(order)
        d.recorder = rec
        d.hub = FakeHub()
        d.set_params({'speed_ff': on, 'ff_vref_v': 8.10, 'throttle_gain': 0.9})
        d.start('m1', 4, record=True, description='x', **({'ff_v0': v0} if v0 else {}))
        d.thread.join(10)
        dr = rec.drive
        if on:
            m = 8.10 / 8.42
            assert (dr['speed_ff'], dr['ff_vref_v'], dr['ff_v0'], dr['ff_m'], dr['ff_gain']) == (1, 8.10, 8.42, m, 0.9 * m)
        else:
            assert (dr['speed_ff'], dr['ff_vref_v'], dr['ff_v0'], dr['ff_m'], dr['ff_gain']) == (0, 8.10, None, None, None)
        assert dr['params']['speed_ff'] == on and dr['params']['ff_vref_v'] == 8.10 and dr['tether_free'] is True
        json.dumps(dr, allow_nan=False)


def test_recorder_saves_ff_columns(tmp_path):
    """AI_CMD.npz の ff_on・ff_m・ff_applied(ON の走行の行の値)。欄の無い行(OFF の走行・古い行)は False・NaN・False。型は bool・float32"""
    assert rec_mod.FF_FIELDS == (('ff_on', bool, False), ('ff_m', np.float32, np.nan), ('ff_applied', bool, False))
    m = 8.10 / 8.42
    rows = [dict(_ai_row(i), ff_on=True, ff_m=m, ff_applied=i != 2) for i in range(5)]
    p = _rec_scene(tmp_path, '20261103-110000-0001', [_veh_row(i, 8.3) for i in range(5)], rows)
    ai = np.load(p / 'AI_CMD.npz')
    assert ai['ff_on'].dtype == bool and ai['ff_m'].dtype == np.float32 and ai['ff_applied'].dtype == bool
    assert ai['ff_on'].tolist() == [True] * 5 and ai['ff_applied'].tolist() == [True, True, False, True, True]
    assert np.allclose(ai['ff_m'], np.float32(m)) and ai['gains'][0].tolist() == pytest.approx([1.0, 0.9])
    p2 = _rec_scene(tmp_path, '20261103-110100-0001', [_veh_row(i, 8.3) for i in range(5)], [_ai_row(i) for i in range(5)])
    ai2 = np.load(p2 / 'AI_CMD.npz')
    assert ai2['ff_on'].tolist() == [False] * 5 and np.isnan(ai2['ff_m']).all() and ai2['ff_applied'].tolist() == [False] * 5
    for k in ai2.files:                                                         # 今までの欄は同じ形
        if not k.startswith('ff_'):
            assert k in ai.files


# ---------------------------------------------------------------- サーバー(走行開始・状態)
@pytest.fixture
def server_env(monkeypatch, tmp_path):
    """server の走行の API を、本物の Driver(走行のスレッドの中身は回さない)と偽の電圧の列で呼ぶ"""
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    runs = tmp_path / 'runs'
    _run_dir(runs, 'm1')
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    gate = threading.Event()
    d._run = lambda *a: gate.wait(5)
    calls = []
    real = d.start

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    d.start = spy
    tr = vs.VbatTracker()
    monkeypatch.setattr(srv, 'driver', d)
    monkeypatch.setattr(srv, 'vbat', tr)
    monkeypatch.setattr(srv, 'VBAT_RUN', {})
    now = {'t': 100.0}
    monkeypatch.setattr(srv, '_now_s', lambda: now['t'])
    monkeypatch.setattr(bench_web, 'AI_NO_PATH', tmp_path / 'ai_run_no.json')
    monkeypatch.setattr(bench_web, 'BENCH_PATH', tmp_path / 'no_bench.json')
    monkeypatch.setattr(srv.trainer.__class__, 'running', property(lambda self: False))
    yield d, calls, gate, tr, now
    gate.set()
    if d.thread is not None:
        d.thread.join(5)


def _body(d, **kw):
    b = {'client': srv.DRIVE_CLIENT_VERSION, 'boot_at': d.boot_at, 'run': 'm1', 'mode_id': 4, 'description': 'AI走行 m1 V_before=__',
         'dry_run': False, 'record': False, 'params': {}}
    b.update(kw)
    return b


def _feed(tr, t0, t1, v, erpm=0.0):
    for i in range(int(round((t1 - t0) * 100))):
        tr.add(t0 + i / 100.0, v, erpm)


def _finish(d, gate):
    gate.set()
    d.thread.join(5)
    gate.clear()


def test_server_passes_v0_when_on(server_env):
    """ON の走行: server が説明欄の V_before と同じ値(2 秒・止まっている・20 個以上の中央値を 0.01 V に丸めた値)を driver.start に
    渡し、その V0 で m を決める。返事の形は前と同じ。dry run も同じ。body の params の speed_ff でも ON(今の設定より先)"""
    d, calls, gate, tr, now = server_env
    d.set_params({'speed_ff': 1, 'ff_vref_v': 8.10, 'throttle_gain': 0.9})
    _feed(tr, 97.0, 100.0, 8.4249)
    for dry in (False, True):
        r = srv.api_drive_start(_body(d, dry_run=dry))
        assert set(r) == {'ok', 'run', 'token', 'ai_no', 'plan_diffs', 'v_before'} and r['v_before'] == 8.42
        assert set(calls[-1]) == FF_KW and calls[-1]['ff_v0'] == 8.42 and calls[-1]['ff_v0_note'] == ''
        assert 'V_before=8.42' in calls[-1]['description'] and d.st['ff']['v0'] == 8.42 and d.st['ff']['m'] == 8.10 / 8.42
        st = srv.api_drive_status(token=r['token'])
        assert st['ff']['on'] is True and st['ff']['gain'] == 0.9 * (8.10 / 8.42) and st['ff_spec']['vref_v'] == 8.10
        _finish(d, gate)
    d.set_params({'speed_ff': 0})
    r = srv.api_drive_start(_body(d, params={'speed_ff': 1}))
    assert set(calls[-1]) == FF_KW and d.st['ff']['on'] is True
    _finish(d, gate)


@pytest.mark.parametrize('case', ['empty', 'moving', 'bad_range', 'short'])
def test_server_refuses_without_v0(server_env, case):
    """ON で止まっているときの基板の電圧が無い(電圧が来ていない・車が動いている・範囲の外・2 秒ぶん無い)→ 409 で走行を始めない
    (理由の文を出す。driver のスレッドを作らない・AI の番号を進めない・VBAT_RUN を変えない)。黙って 1.00 にしない"""
    d, calls, gate, tr, now = server_env
    d.set_params({'speed_ff': 1})
    if case == 'moving':
        _feed(tr, 97.0, 100.0, 8.42, erpm=900.0)
    elif case == 'bad_range':
        _feed(tr, 97.0, 100.0, 9.6)
    elif case == 'short':
        _feed(tr, 99.5, 100.0, 8.42)
    with pytest.raises(HTTPException) as e:
        srv.api_drive_start(_body(d))
    assert e.value.status_code == 409 and '電圧の前送りが ON ですが' in e.value.detail and '走行は始めていません' in e.value.detail
    want = {'empty': '車両の値が来ていません', 'moving': '車が止まっていません', 'bad_range': '基板の電圧がありません',
            'short': '2.0 秒ぶんありません'}[case]
    assert want in e.value.detail, e.value.detail
    assert d.thread is None and not (Path(bench_web.AI_NO_PATH)).exists() and srv.VBAT_RUN == {}
    assert set(calls[-1]) == FF_KW and calls[-1]['ff_v0'] is None


def test_server_off_passes_the_same_arguments(server_env, monkeypatch):
    """OFF の走行は前と同じ引数(電圧を渡さない)。電圧の前送りを知らない偽物の start(ほかの試験の偽物と同じ形)でも動く。
    電圧が無くても始める(前と同じ)"""
    d, calls, gate, tr, now = server_env
    r = srv.api_drive_start(_body(d))
    assert r['ok'] is True and set(calls[-1]) == START_KW and d.st['ff'] == {'on': False}
    _finish(d, gate)
    seen = []

    def old_start(run, mode_id, record=False, description='', tags=None, dry_run=False, params=None):
        seen.append(run)
        return run, 'tok'
    monkeypatch.setattr(d, 'start', old_start)
    assert srv.api_drive_start(_body(d))['ok'] is True and seen == ['m1']
    assert srv._ff_requested({'params': {'speed_ff': 0}}) is False and srv._ff_requested({'params': {'speed_ff': '1'}}) is True
    assert srv._ff_requested({'params': {'speed_ff': 'x'}}) is False and srv._ff_requested({}) is False


def test_server_refuses_old_page(server_env):
    """版 11 より古い画面(版 10 = スイッチも確認の文の行も無い画面)からは走行を始めない(409。driver.start を呼ばない)"""
    d, calls, gate, tr, now = server_env
    for client in (10, 9, 0, None, 'x'):
        with pytest.raises(HTTPException) as e:
            srv.api_drive_start(_body(d, client=client))
        assert e.value.status_code == 409 and '管理画面が古いまま' in e.value.detail and '版 11 から' in e.value.detail, client
    assert calls == [] and d.thread is None


def test_server_params_api_rules(server_env):
    """POST /api/drive/params: 範囲の外の基準の電圧・天井と一緒・走行中の切り替えは 400(値は変えない)"""
    d, calls, gate, tr, now = server_env
    assert srv.api_drive_params({'speed_ff': 1, 'ff_vref_v': 8.2})['speed_ff'] == 1
    for bad in ({'ff_vref_v': 9.1}, {'speed_ceiling': 1}):
        with pytest.raises(HTTPException) as e:
            srv.api_drive_params(bad)
        assert e.value.status_code == 400
    _feed(tr, 97.0, 100.0, 8.2)
    srv.api_drive_start(_body(d))
    for bad in ({'speed_ff': 0}, {'ff_vref_v': 8.25}):
        with pytest.raises(HTTPException) as e:
            srv.api_drive_params(bad)
        assert e.value.status_code == 400 and '走行中は電圧の前送り' in e.value.detail
    assert d.params['speed_ff'] == 1 and d.params['ff_vref_v'] == 8.2 and d.st['ff']['m'] == 1.0
    _finish(d, gate)


# ---------------------------------------------------------------- 画面(node で本物の app.js)
def _node(code):
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def _fns(*names):
    js = APP.read_text(encoding='utf-8')
    return ''.join(_extract(js, n) for n in names)


@needs_node
def test_js_multiplier_matches_the_server():
    """画面の見込みの倍率(ffMult)はサーバー(drive.ff_multiplier)と同じ式・同じ切り・同じ範囲(使えない電圧は null)"""
    cases = [(8.10, 8.42), (8.10, 7.0), (8.10, 8.95), (7.6, 8.0), (8.8, 8.0), (8.10, 6.0), (8.10, 9.0), (8.10, 5.99), (8.10, 9.01),
             (7.59, 8.0), (8.81, 8.0), (8.10, None), (7.80, 8.40), (7.80, 8.57), (7.60, 9.0), (7.60, 8.94)]
    code = (_fns('FF_FALLBACK', 'ffMult')
            + 'process.stdout.write(JSON.stringify(%s.map(([r, v]) => ffMult(r, v, FF_FALLBACK))));' % json.dumps(cases))
    got = _node(code)
    for (r, v), g in zip(cases, got):
        try:
            want = drive.ff_multiplier(r, v)
        except ValueError:
            want = None
        assert g == want, (r, v, g, want)
    js = APP.read_text(encoding='utf-8')
    assert 'const FF_FALLBACK = { mlo: 0.85, mhi: 1.06, vlo: 6.0, vhi: 9.0, rlo: 7.6, rhi: 8.8 };' in js     # drive の値と同じ
    assert (drive.FF_M_RANGE, drive.FF_V0_RANGE, drive.PARAM_LIMITS['ff_vref_v']) == (M_RANGE, (6.0, 9.0), (7.6, 8.8))


@needs_node
def test_js_start_line_texts():
    """▶ の確認の文の行: ON は「電圧の前送り: ON(電圧 8.42 V・基準 8.10 V・× 0.962 → ゲイン 0.866)」(今の画面の電圧・
    サーバーの基準の電圧・速度ゲインのスライダー)。電圧が出ていないときは「走行を始めません」の行。OFF・知らないサーバーは行なし(OFF の
    確認の文は直す前とまったく同じ。前の直しの見張りの試験が文を決め打ちしている)。切りの注の数はサーバーの ff_spec の値(指摘 S2)"""
    stub = ("const $ = (s) => ({ value: '0.90' });\nconst fmtN = (x, d = 2) => (x === undefined || x === null || !Number.isFinite(+x) ? '-' "
            ": (+x).toFixed(d));\n")
    code = stub + _fns('FF_FALLBACK', 'ffSupported', 'ffSpec', 'ffMult', 'ffPreview', 'ffText', 'ffClipNote', 'ffStartLine')
    sts = [{'params': {'speed_ff': 1, 'ff_vref_v': 8.10}, 'vbat': {'v': 8.42}},
           {'params': {'speed_ff': 1, 'ff_vref_v': 8.10}, 'vbat': {'v': None}},
           {'params': {'speed_ff': 0, 'ff_vref_v': 8.10}, 'vbat': {'v': 8.42}},
           {'params': {'ff_vref_v': 8.10}, 'vbat': {'v': 8.42}},
           {'params': {'speed_ff': 1, 'ff_vref_v': 8.10}, 'vbat': {'v': 9.4}},
           {'params': {'speed_ff': 1, 'ff_vref_v': 8.10}, 'vbat': {'v': 7.2},
            'ff_spec': {'m_range': [0.85, 1.06], 'v0_range': [6.0, 9.0], 'vref_range': [7.6, 8.8]}},
           {'params': {'speed_ff': 1, 'ff_vref_v': 7.80}, 'vbat': {'v': 8.57}},
           {'params': {'speed_ff': 1, 'ff_vref_v': 7.60}, 'vbat': {'v': 9.0}},
           {'params': {'speed_ff': 1, 'ff_vref_v': 7.80}, 'vbat': {'v': 8.57},
            'ff_spec': {'m_range': [0.94, 1.05], 'v0_range': [6.0, 9.0], 'vref_range': [7.6, 8.8]}}]
    code += 'process.stdout.write(JSON.stringify(%s.map((s) => ffStartLine(s))));' % json.dumps(sts)
    got = _node(code)
    assert got[0] == '\n' + CONFIRM_ON
    assert got[1].startswith('\n電圧の前送り: ON(基準 8.10 V。今は電池の電圧が出ていません。') and '走行を始めません' in got[1]
    assert got[2] == '' and got[3] == '' and '電池の電圧が出ていません' in got[4]
    assert got[5] == ('\n電圧の前送り: ON(電圧 7.20 V・基準 8.10 V・× 1.060 → ゲイン 0.954)\n'
                      '  倍率が上の切り 1.06 に当たっています(基準 ÷ 電圧 = 1.125。これより速くはしません)。'
                      '電圧がふだんより低いときは、パックと基板を確かめてください')
    assert got[6] == '\n電圧の前送り: ON(電圧 8.57 V・基準 7.80 V・× 0.910 → ゲイン 0.819)'      # 列 7.8・満充電に近い = 紙の表と同じ(指摘 S2)
    assert got[7] == ('\n電圧の前送り: ON(電圧 9.00 V・基準 7.60 V・× 0.850 → ゲイン 0.765)\n'          # 受け付ける端の組だけ下の切り
                      '  倍率が下の切り 0.85 に当たっています(基準 ÷ 電圧 = 0.844)。紙の表より速い向きです')
    assert got[8] == ('\n電圧の前送り: ON(電圧 8.57 V・基準 7.80 V・× 0.940 → ゲイン 0.846)\n'          # 切りの数はサーバーの値
                      '  倍率が下の切り 0.94 に当たっています(基準 ÷ 電圧 = 0.910)。紙の表より速い向きです')


def sim_ff_js(tmp_path):
    """tests/fixtures/app_sim_fix8.js の写し(もとは変えない)に足す: 偽のサーバーの status に入力の server_extra(vbat・ff_spec・
    tether_free など)・POST /api/drive/params に電圧の前送りの決まり(drive._ff_rules と同じに断る)・操作 ['fire', sel, 事]"""
    src = SIM.read_text(encoding='utf-8')
    a1 = '  if (bootAt !== null) st.boot_at = bootAt;\n'
    a2 = "  if (KNOWN.has('rec_hold_keep') && on(p.rec_hold_keep) && !on(p.rec_back_guard)) return '中立止めのあとも残すは、後ろの見張りが ON のときだけ';\n"
    a3 = "  else if (op === 'set') { $(a).value = String(b); out.push({ page: pg.name, op }); }\n"
    for a in (a1, a2, a3):
        assert src.count(a) == 1, a
    ff_rules = (a2 + "  if (KNOWN.has('speed_ff')) {\n"
                "    if (body && body.ff_vref_v !== undefined && !(+body.ff_vref_v >= 7.6 - 1e-9 && +body.ff_vref_v <= 8.8 + 1e-9)) return '電圧の前送りの基準の電圧は受け付けません';\n"
                "    if (isRunning && (on(p.speed_ff) !== on(prev.speed_ff) || +p.ff_vref_v !== +prev.ff_vref_v)) return '走行中は電圧の前送りを ON にも OFF にもできません';\n"
                "    if (on(p.speed_ff) && (on(p.speed_ceiling) || on(p.speed_auto))) return '電圧の前送りと速さの天井・自動合わせは一緒に使えません';\n"
                "  }\n")
    src = src.replace(a1, a1 + '  if (inp.server_extra) Object.assign(st, JSON.parse(JSON.stringify(inp.server_extra)));\n')
    src = src.replace(a2, ff_rules)
    src = src.replace(a3, a3 + "  else if (op === 'fire') { pg.confirms.length = 0; await $(a).fire(b); await flush(); "
                               "out.push({ page: pg.name, op, sel: a, confirm: pg.confirms[0] || '' }); }\n")
    p = tmp_path / 'app_sim_ff.js'
    p.write_text(src, encoding='utf-8')
    return p


def sim(tmp_path, steps, server_params=None, extra=None, app=None, name='in.json'):
    """本物の app.js(app = 直す前の写しのときは、その index.html)を node で読む(偽のサーバー・DOM)"""
    c = bench_web.load_candidates()
    inp = tmp_path / name
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=0.9, lookahead=3, start_kick=1)
    sp.update(server_params or {})
    d = {'bench': c, 'server_params': sp, 'steps': steps, 'server_client_version': 11, 'confirm': True, 'boot_at': 1000.0,
         'models': [c['baseline']['run']], 'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True,
         'pages': ['A'], 'server_extra': extra}
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    html = BEFORE / 'index.html' if app is not None else INDEX
    r = subprocess.run([NODE, str(sim_ff_js(tmp_path)), str(app or APP), str(html), str(inp)],
                       capture_output=True, text=True, timeout=180, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


PRE = [['tab', 'drive'], ['tick'], ['tick']]
VB = {'vbat': {'supported': True, 'v': 8.42, 'age_s': 0.1, 'level': 'ok', 'warn_v': 7.9, 'bad_v': 7.7, 'broken_v': None,
               'range_v': [6.0, 9.0], 'scale': 1.0, 'run': None},
      'ff_spec': drive.ff_spec(), 'tether_free': True, 'tether_free_max_s': 1800.0}


def _starts(res):
    return [p for x in res['steps'] if x['op'] == 'posted' for p in x['value'] if p[1] == '/api/drive/start']


def _params_posts(res):
    return [p[2] for x in res['steps'] if x['op'] == 'posted' for p in x['value'] if p[1] == '/api/drive/params']


@needs_node
def test_page_confirm_line_and_same_body(tmp_path):
    """サーバーが ON(基準 8.10 V)・画面の電圧 8.42 V・速度ゲイン 0.90: ▶ の確認の文に「電圧の前送り: ON(…× 0.962 → ゲイン 0.866)」の
    行が 1 つ(速さの行のあと)。ほかの行は直す前の app.js と同じ。送った body は版(11)のほか直す前と同じ(speed_ff・ff_vref_v は
    送らない = サーバーの値)。OFF のサーバーでは確認の文が直す前とまったく同じ"""
    steps = PRE + [['start'], ['posted']]
    new = sim(tmp_path, steps, {'speed_ff': 1, 'ff_vref_v': 8.10}, VB)
    old = sim(tmp_path, steps, {'speed_ff': 1, 'ff_vref_v': 8.10}, VB, app=BEFORE / 'app.js', name='o.json')
    (sn,), (so,) = [x for x in new['steps'] if x['op'] == 'start'], [x for x in old['steps'] if x['op'] == 'start']
    assert sn['status'] == 200 and so['status'] == 409                      # 直す前の画面(版 10)は今のサーバーが断る
    lines_n, lines_o = sn['confirm'].split('\n'), so['confirm'].split('\n')
    assert lines_n.count(CONFIRM_ON) == 1 and CONFIRM_ON not in lines_o
    assert [x for x in lines_n if x != CONFIRM_ON] == lines_o
    (pn,), (po,) = _starts(new), _starts(old)
    assert pn[2]['client'] == 11 and po[2]['client'] == 10
    assert {k: v for k, v in pn[2].items() if k != 'client'} == {k: v for k, v in po[2].items() if k != 'client'}
    assert 'speed_ff' not in pn[2]['params'] and 'ff_vref_v' not in pn[2]['params']
    off_n = sim(tmp_path, steps, {'speed_ff': 0}, VB, name='n2.json')
    off_o = sim(tmp_path, steps, {'speed_ff': 0}, VB, app=BEFORE / 'app.js', name='o2.json')
    (a,), (b,) = [x for x in off_n['steps'] if x['op'] == 'start'], [x for x in off_o['steps'] if x['op'] == 'start']
    assert a['confirm'] == b['confirm'] and '電圧の前送り' not in a['confirm']


@needs_node
def test_page_switch_and_vref_post_at_once(tmp_path):
    """スイッチを ON(確認の文あり)→ すぐ {speed_ff: 1} を送る。OFF → {speed_ff: 0}(確認なし)。基準の電圧の欄 → {ff_vref_v: 値}
    (0.01 に丸める)。範囲の外の値は送らずに戻す。天井が ON のあいだは ON にできない(送らない)。走行中はどちらも送らない"""
    steps = PRE + [['check', '#dr-speed-ff', True], ['posted'], ['el', '#dr-ff-note', 'html'], ['el', '#dr-ff-preview', 'html'],
                   ['check', '#dr-speed-ff', False], ['posted'],
                   ['set', '#dr-ff-vref', '8.25'], ['fire', '#dr-ff-vref', 'change'], ['posted'],
                   ['set', '#dr-ff-vref', '9.30'], ['fire', '#dr-ff-vref', 'change'], ['posted'], ['el', '#dr-ff-vref', 'value'],
                   ['api', {'speed_ceiling': 1}], ['tick'], ['check', '#dr-speed-ff', True], ['posted'], ['api', {'speed_ceiling': 0}],
                   ['tick'], ['set_running', True], ['tick'], ['check', '#dr-speed-ff', True], ['posted'],
                   ['set', '#dr-ff-vref', '8.30'], ['fire', '#dr-ff-vref', 'change'], ['posted'], ['toasts']]
    res = sim(tmp_path, steps, {'speed_ff': 0, 'ff_vref_v': 8.10}, VB)
    posts = [x['value'] for x in res['steps'] if x['op'] == 'posted']
    sent = [[p[2] for p in v if p[1] == '/api/drive/params' and p[0] != 'curl'] for v in posts]     # 画面が送った物(curl は除く)
    assert sent[0] == [{'speed_ff': 1}] and sent[1] == [{'speed_ff': 0}] and sent[2] == [{'ff_vref_v': 8.25}]
    assert sent[3] == [] and sent[4] == [] and sent[5] == [] and sent[6] == []
    checks = [x for x in res['steps'] if x['op'] == 'check']
    assert checks[0]['confirm'].startswith('電圧の前送りを ON にします') and checks[1]['confirm'] == ''
    assert 'ON: AI の前進の速度ゲイン' in [x for x in res['steps'] if x['op'] == 'el'][0]['value']
    assert '見込み: 電圧 8.42 V・基準 8.10 V・× 0.962 → ゲイン 0.866' in [x for x in res['steps'] if x['op'] == 'el'][1]['value']
    assert [x for x in res['steps'] if x['op'] == 'el'][2]['value'] == '8.25'
    toasts = [x for x in res['steps'] if x['op'] == 'toasts'][-1]['value']
    assert any('7.60〜8.80 V で入れてください' in t for t in toasts)
    assert any('速さの天井・自動合わせが ON のあいだは' in t for t in toasts)
    assert any('走行中は電圧の前送りを ON にも OFF にもできません' in t for t in toasts)
    assert any('走行中は電圧の前送りの基準の電圧を変えられません' in t for t in toasts)


@needs_node
def test_page_refuses_start_while_sending(tmp_path):
    """スイッチの送信の返事を待っている間の ▶ は断る(返事の前の表示で確認の文を作らない)"""
    js = APP.read_text(encoding='utf-8')
    i = js.index("if (drNotSynced()) return;\n  // 電圧の前送りのスイッチ")
    assert js.index("if (ffBusy.sw || ffBusy.vref) { toast('電圧の前送りの設定を送っています。") > i
    assert js.count('ffBusy.sw = true;') == 1 and js.count('ffBusy.sw = false;') == 1 and js.count('ffBusy.vref = true;') == 1


@needs_node
def test_page_live_row(tmp_path):
    """走行中・走り終わりの表示の行(st.ff = この走行の値): 走る前の電圧・基準・倍率・実のゲイン(今の速度ゲイン × m)"""
    stub = ("const fmtN = (x, d = 2) => (x === undefined || x === null || !Number.isFinite(+x) ? '-' : (+x).toFixed(d));\n"
            "const esc = (s) => String(s);\n")
    code = stub + _fns('FF_FALLBACK', 'ffSpec', 'ffClipNote', 'ffLiveText')
    m = 8.10 / 8.42
    st = {'running': True, 'params': {'throttle_gain': 0.9}, 'ff': {'on': True, 'v0': 8.42, 'vref_v': 8.1, 'm': m}}
    now = {'ff': {'on': True, 'm': round(m, 4), 'gain': round(0.9 * m, 4), 'applied': False}}
    code += 'process.stdout.write(JSON.stringify([ffLiveText(%s, %s), ffLiveText(%s, null)]));' % (
        json.dumps(st), json.dumps(now), json.dumps(dict(st, running=False)))
    a, b = _node(code)
    assert '走る前の電圧 8.42 V ・ 基準 8.10 V ・ 倍率 × 0.962 ・ 実のゲイン <b>0.866</b>(今の速度ゲイン 0.90 × 0.962)' in a
    assert 'このフレームは掛けていません' in a and '(最後の走行)' in b and 'このフレームは' not in b and '切り' not in a + b


def test_static_panel():
    """走行タブの速度ゲインのすぐ下に、スイッチ・基準の電圧の欄(7.6〜8.8・0.05 刻み・初め 8.10)・見込み・説明。▶ で送る物の
    readDriveParams は変えていない(スイッチは押したときに送る)。起動し直しの知らせに speed_ff(既定 0)"""
    html = INDEX.read_text(encoding='utf-8')
    assert html.count('id="dr-speed-ff"') == 1 and html.count('id="dr-ff-vref"') == 1
    assert '<input type="number" id="dr-ff-vref" min="7.6" max="8.8" step="0.05" value="8.10" class="num wide" autocomplete="off">' in html
    # 基準の電圧の欄も、スイッチの行と同じく注を次の行へ折り返し、欄はほかの数の欄と同じ短い形(num wide)で「基準の電圧 [8.10] V」を 1 行にする
    # (注と同じ行に詰めると「基準の電圧」が 1 字ずつ縦に並んだ。11/1 動画の確かめ N2)
    assert '<label class="inline check-line">基準の電圧 <input type="number" id="dr-ff-vref"' in html
    # 倍率の幅の数は index.html に書かない(画面がサーバーの ff_spec で書く。11/1 粗探しの指摘 S2)
    assert '<span id="dr-ff-spec" class="muted small">(倍率 = 基準の電圧 ÷ 走る前の電圧)</span>' in html
    panel = html[html.index('id="dr-ff-panel"'):html.index('id="dr-nopost-note"')]
    assert not any(x in panel for x in ('0.94', '0.85', '0.90', '1.06', '〜1.0', '倍)'))
    assert html.index('ゲインは走行中に動かしても、すぐ反映されます。') < html.index('id="dr-ff-panel"') < html.index('id="dr-nopost-note"')
    assert '電圧で最高速をそろえる(電圧の前送り)' in html
    js = APP.read_text(encoding='utf-8')
    rd = _extract(js, 'readDriveParams')
    assert 'speed_ff' not in rd and 'ff_vref' not in rd
    assert rd == _extract((BEFORE / 'app.js').read_text(encoding='utf-8'), 'readDriveParams')
    assert 'speed_ff: 0 };' in js and "'zone_caps', 'speed_ff'];" in js


# ---------------------------------------------------------------- 11/1 粗探しの指摘 S3: 画面の変えた所(足した行まで照らす)
# app.js(説明の行・行の終わりの注を除く)の、直す前の写しからの違い: PAGE_SMALL = 前からある行のあいだで消した・足した行の組(▶ の断り・
# 確認の文の行・版・起動し直しの既定・表示の同期・走行中の行・走り終わりの注・枠の描き方・基準の値の控え・名前)。PAGE_BLOCK = 1 か所に
# まとめて足した電圧の前送りの部品の、宣言ごとの sha256 の頭 16 文字(どこかを書き替えれば、どの宣言かが分かって落ちる)。
# INDEX_ADDED_SHA = index.html に足した枠の行の sha256。値を変えるのは、ここに書いた理由のある直しのときだけ(README 13 章)
# ---- 決め打ちの値(ここから。c1/tools/new_tests_page.py が今のファイルから書く)
PAGE_SMALL = [(['const DRIVE_CLIENT = 10;'], ['const DRIVE_CLIENT = 11;']),
 ([], ["  if (ffBusy.sw || ffBusy.vref) { toast('電圧の前送りの設定を送っています。少し待ってから押してください', true); return; }"]),
 (['  const speedLine = speedStartLines(state.drive.st) + launchStartLine(state.drive.st);'],
  ['  const speedLine = speedStartLines(state.drive.st) + launchStartLine(state.drive.st) + ffStartLine(state.drive.st);']),
 (['  const body = { client: 10, build: PAGE_BUILD, boot_at: state.drive.bootAt ?? null, run: m.name, mode_id: '
   "+$('#dr-mode').value, dry_run: dry,"],
  ['  const body = { client: 11, build: PAGE_BUILD, boot_at: state.drive.bootAt ?? null, run: m.name, mode_id: '
   "+$('#dr-mode').value, dry_run: dry,"]),
 (['  zone_caps: 0, tcp_mode: 0, speed_scale: 1.0 };',
   "const BOOT_FLAGS = ['rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'speed_ceiling', 'speed_auto', 'launch_boost', "
   "'zone_caps'];"],
  ['  zone_caps: 0, tcp_mode: 0, speed_scale: 1.0, speed_ff: 0 };',
   "const BOOT_FLAGS = ['rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'speed_ceiling', 'speed_auto', 'launch_boost', "
   "'zone_caps', 'speed_ff'];"]),
 ([],
  ['    for (const k of FF_KEYS) if ((drAckAt[k] || 0) > since && prev.params[k] !== undefined) st.params[k] = prev.params[k];']),
 ([], ["    if (st.ff && st.ff.on) rows.push(['電圧の前送り', ffLiveText(st, now)]);"]),
 ([], ["    if (!st.running && st.auto_end && ffAutoEndNote(st)) rows.push(['自動合わせの目安の注', ffAutoEndNote(st)]);"]),
 ([], ['  renderFfSwitch(st);']),
 (['            rec_hold_keep: 0, launch_no_kick: 0 },'],
  ['            rec_hold_keep: 0, launch_no_kick: 0, speed_ff: 0, ff_vref_v: 8.1 },']),
 (["                      launch_boost: '発進の前進', launch_us: '発進の前進の強さ', launch_mps: '発進の前進の速さ', launch_max_s: '発進の前進の時間' };"],
  ["                      launch_boost: '発進の前進', launch_us: '発進の前進の強さ', launch_mps: '発進の前進の速さ', launch_max_s: '発進の前進の時間',",
   "                      speed_ff: '電圧の前送り', ff_vref_v: '電圧の前送りの基準の電圧' };"])]
PAGE_BLOCK = {"const FF_KEYS = ['speed_ff', 'ff_vref_v'];": 'c91d8a47a583138c',
 'const ffSupported = (st) => !!(st && st.params &': 'ef8a6360562db847',
 'const FF_FALLBACK = { mlo: 0.85, mhi: 1.06, vlo:': '9422b8b00a68f645',
 'const ffBusy = { sw: false, vref: false };': '52fbe0347c5d46c2',
 'function ffSpec(st) {': 'e605c094a431ea2c',
 'function ffMult(vref, v0, sp) {': '8488a9212d7de4b8',
 'function ffPreview(st) {': 'aebdc2a56c10274b',
 'const ffText = (x) => `電圧 ${x.v.toFixed(2)} V・基準': '08aee8346444b0f5',
 'function ffClipNote(vref, v, sp) {': '77053d0121bb2a14',
 'function ffStartLine(st) {': 'f3b1bf76655a762b',
 'function ffConfirmText(st) {': 'af6fb6672e91014a',
 'function ffAutoEndNote(st) {': 'f1f620034bcc4559',
 'function ffLiveText(st, now) {': '7322feccbf98d27f',
 'function renderFfSwitch(st) {': '33831d4fc96b8d84',
 "if ($('#dr-speed-ff')) $('#dr-speed-ff').addEven": '4e5be115a7b6874d',
 "if ($('#dr-ff-vref')) $('#dr-ff-vref').addEventL": 'ccc751ceea8cfaca',
 "$('#dr-throttle-gain').addEventListener('input',": '6878318487202919'}
INDEX_ADDED_SHA = 'b9ff008870d3321a7a75161f1ce5adc77e7b560e04d9c2ff267e1edc7c80969f'
# ---- 決め打ちの値(ここまで)


def _decl_groups(lines):
    """足した行を、1 文字目が空白でない行ごとの宣言のかたまりに分ける(閉じかっこの行は前のかたまり)"""
    out = []
    for x in lines:
        if not out or (x and not x[0].isspace() and x not in ('}', '});', '};')):
            out.append([x])
        else:
            out[-1].append(x)
    return out


def test_page_changed_only_these_lines():
    """11/1 粗探しの指摘 S3: app.js の直す前の写しからの違いは、前からある行のあいだの決めた組(PAGE_SMALL)と、1 か所(発進の前進の
    スイッチの次)にまとめて足した電圧の前送りの部品(PAGE_BLOCK の宣言ごとの sha256)だけ。index.html は速度ゲインの説明の行の次に枠を
    1 つ足しただけ(INDEX_ADDED_SHA)。前の試験は消した行だけを見ていた(足した行は、どこに何行あっても通った)"""
    old = _code_lines((BEFORE / 'app.js').read_text(encoding='utf-8'))
    new = _code_lines(APP.read_text(encoding='utf-8'))
    ops = [(o, old[o[1]:o[2]], new[o[3]:o[4]]) for o in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
           if o[0] != 'equal']
    big = [x for x in ops if len(x[2]) > 40]
    assert len(big) == 1, [len(x[2]) for x in ops]
    (tag, a, _, _, _), removed, block = big[0]
    assert tag == 'insert' and removed == []
    assert old[a - 2:a] == ['  if (state.drive.st) renderLaunchSwitch(state.drive.st);', '});']
    assert [(r, x) for _, r, x in ops if len(x) <= 40] == PAGE_SMALL
    groups = _decl_groups(block)
    got = {re.sub(r'\s+', ' ', g[0])[:48]: hashlib.sha256('\n'.join(g).encode()).hexdigest()[:16] for g in groups}
    assert len(groups) == len(got) == len(PAGE_BLOCK)
    assert {k: v for k, v in got.items() if PAGE_BLOCK.get(k) != v} == {} and set(got) == set(PAGE_BLOCK)
    ho = _code_lines((BEFORE / 'index.html').read_text(encoding='utf-8'))
    hn = _code_lines(INDEX.read_text(encoding='utf-8'))
    hops = [o for o in difflib.SequenceMatcher(None, ho, hn, autojunk=False).get_opcodes() if o[0] != 'equal']
    assert len(hops) == 1 and hops[0][0] == 'insert'
    assert ho[hops[0][1] - 1].strip() == '<div class="muted small">ゲインは走行中に動かしても、すぐ反映されます。</div>'
    added = hn[hops[0][3]:hops[0][4]]
    assert len(added) == 12 and hashlib.sha256('\n'.join(added).encode()).hexdigest() == INDEX_ADDED_SHA


@needs_node
def test_page_spec_text_follows_the_server(tmp_path):
    """11/1 粗探しの指摘 S2: スイッチの横の倍率の幅「(倍率 = 基準の電圧 ÷ 走る前の電圧・0.85〜1.06 倍)」は、サーバーの status の
    ff_spec の値で書く(index.html に数を決め打ちしない = drive.FF_M_RANGE を変えれば画面も変わる)。スイッチの確認の文も同じ値"""
    steps = PRE + [['el', '#dr-ff-spec', 'text'], ['check', '#dr-speed-ff', True]]
    for name, extra, want in (('s1.json', VB, '0.85〜1.06'),
                              ('s2.json', dict(VB, ff_spec=dict(drive.ff_spec(), m_range=[0.90, 1.05])), '0.90〜1.05')):
        res = sim(tmp_path, steps, {'speed_ff': 0, 'ff_vref_v': 8.10}, extra, name=name)
        (e,) = [x for x in res['steps'] if x['op'] == 'el']
        (c,) = [x for x in res['steps'] if x['op'] == 'check']
        assert e['value'] == '(倍率 = 基準の電圧 ÷ 走る前の電圧・%s 倍)' % want, e
        assert '÷ 走る前の電圧・%s 倍)。' % want in c['confirm'], c['confirm']
    assert '(倍率 = 基準の電圧 ÷ 走る前の電圧・%.2f〜%.2f 倍)' % M_RANGE == '(倍率 = 基準の電圧 ÷ 走る前の電圧・0.85〜1.06 倍)'


@needs_node
def test_running_disables_switch_and_vref(tmp_path):
    """走行中(推論の確認中も)は、スイッチと基準の電圧の欄が disabled(灰色で押せない)。止まれば戻る。README 7.1 の 7 の確かめは
    これを見る(11/1 粗探しの指摘 S1-desk: 「走行中に押してトーストを見る」は本物の画面では押せないのでできない)"""
    steps = PRE + [['el', '#dr-speed-ff', 'disabled'], ['el', '#dr-ff-vref', 'disabled'], ['set_running', True], ['tick'],
                   ['el', '#dr-speed-ff', 'disabled'], ['el', '#dr-ff-vref', 'disabled'], ['el', '#dr-speed-ff', 'checked'],
                   ['set_running', False], ['tick'], ['el', '#dr-speed-ff', 'disabled'], ['el', '#dr-ff-vref', 'disabled']]
    res = sim(tmp_path, steps, {'speed_ff': 1, 'ff_vref_v': 8.10}, VB)
    assert [x['value'] for x in res['steps'] if x['op'] == 'el'] == [False, False, True, True, True, False, False]


@needs_node
def test_reboot_turns_off_without_the_on_line(tmp_path):
    """サーバーが起き直して OFF に戻った(boot_at が変わり speed_ff 0)あと: 枠に起き直しの注・スイッチは OFF・▶ の確認の文に
    「電圧の前送り: ON」の行が無い(= OFF。守りは 1 枚の毎本の手順の「行を読み上げる・無ければキャンセルして ON にし直す」。11/1
    粗探しの指摘 N4・N5。「OFF」の行を出す形は、前の直しの見張りの試験 64 件が決め打ちする OFF の確認の文を変えるので入れなかった)"""
    steps = PRE + [['reboot', 2000.0, {'speed_ff': 0}], ['tick'], ['tick'], ['el', '#dr-ff-note', 'html'],
                   ['el', '#dr-speed-ff', 'checked'], ['start'], ['tick'], ['start']]
    res = sim(tmp_path, steps, {'speed_ff': 1, 'ff_vref_v': 8.10}, VB)
    note, checked = [x['value'] for x in res['steps'] if x['op'] == 'el']
    assert '起動し直したので、電圧の前送りは OFF に戻っています' in note and checked is False
    confirms = [x.get('confirm') or '' for x in res['steps'] if x['op'] == 'start']
    assert len(confirms) == 2 and any(confirms)
    for c in confirms:
        assert '電圧の前送り' not in c, c


@needs_node
def test_auto_end_note_for_ff_runs():
    """11/1 粗探しの指摘 N3: 電圧の前送りが ON の本の「自動合わせ(走り終わり)」の目安(使わないなら …)は × m のあとの実のゲイン
    から作られる。その下に注の行を出す(速度ゲインの欄の値ではない・÷ m)。OFF の本・まだ走っていない画面は注なし"""
    stub = "const fmtN = (x, d = 2) => (x === undefined || x === null || !Number.isFinite(+x) ? '-' : (+x).toFixed(d));\n"
    m = 8.10 / 8.42
    sts = [{'ff': {'on': True, 'm': m}}, {'ff': {'on': False}}, {}, None, {'ff': {'on': True, 'm': None}}]
    got = _node(stub + _fns('ffAutoEndNote') + 'process.stdout.write(JSON.stringify(%s.map((s) => ffAutoEndNote(s))));' % json.dumps(sts))
    assert '倍率 × 0.962 を掛けたあとの実のゲイン' in got[0] and '速度ゲインの欄の値ではありません(欄の値にするなら ÷ 0.962)' in got[0]
    assert got[1:] == ['', '', '', '']
    js = APP.read_text(encoding='utf-8')
    assert js.count("if (!st.running && st.auto_end && ffAutoEndNote(st)) rows.push(['自動合わせの目安の注', ffAutoEndNote(st)]);") == 1


# ---------------------------------------------------------------- 11/1 確かめ直しの指摘 S1-page-text-vs-form-B: 画面の文と 1 枚の形 B
# 当日の 1 枚(prep_1103/run_1103_v2.md 4.6)の形 B = 基準の電圧は一日 8.10 V のまま触らず、列は速度ゲインの ± 0.01 で動かす(main の決め
# 11/1 18:30)。画面の文はこの字のとおり(係 F1)。前の文「列を動かすときは基準の電圧を動かします」は形 B と逆で、紙どおりゲインも動かすと
# 1 列のつもりが約 2 列になった(計算の例。README 13 章)
FORM_B_CONFIRM = '基準の電圧は一日同じ値のまま触らず、列は速度ゲインの ± 0.01 で動かします(当日の 1 枚のとおり)。'
FORM_B_PANEL = 'ON の日は紙の表でゲインを引かず、基準の電圧は一日同じ値のまま、列は速度ゲインの ± 0.01 で動かします(当日の 1 枚のとおり)。'
FORM_B_VREF = '(この電圧ならゲインそのまま。一日同じ値のまま使います。7.60〜8.80 V・0.05 V 刻み。起動しても残ります)'
FORM_A_WORDS = ('基準の電圧を動か', '紙の表の「列」', '列 = 基準', '列を動かすときは基準', '1 つの値のまま')     # 形 A(基準の電圧で列を動かす)の言い方


def _panel_text(html):
    """index.html の電圧の前送りの枠(#dr-ff-panel から #dr-nopost-note の前まで)の字(タグを除き、行の頭と終わりの空白を除いてつなぐ)"""
    panel = html[html.index('<div class="zone-panel ff-panel" id="dr-ff-panel">'):html.index('<div id="dr-nopost-note"')]
    return ''.join(x.strip() for x in re.sub(r'<[^>]*>', '', panel).split('\n'))


@needs_node
def test_page_texts_say_form_b(tmp_path):
    """11/1 確かめ直しの指摘 S1-page-text-vs-form-B: スイッチを ON にするときの確認の文(本物の app.js が confirm に渡す文)と、走行タブの
    枠の注意書き・基準の電圧の欄の注は、1 枚の形 B(基準の電圧は一日同じ値のまま・列は速度ゲインの ± 0.01)を言い、「基準の電圧を動かす」
    と読める文(形 A の言い方)を含まない。drive.py の説明の文と注も同じ(「紙の表の『列』」「基準の電圧を左へ動かし」を書かない)"""
    res = sim(tmp_path, PRE + [['check', '#dr-speed-ff', True]], {'speed_ff': 0, 'ff_vref_v': 8.10}, VB)
    (c,) = [x for x in res['steps'] if x['op'] == 'check']
    conf = c['confirm']
    assert conf.startswith('電圧の前送りを ON にします(倍率 = 基準の電圧 8.10 V ÷ 走る前の電圧・0.85〜1.06 倍)。'), conf
    assert conf.endswith('ON の日は紙の表でゲインを引かないでください(二重に直します)。' + FORM_B_CONFIRM
                         + '走行中は ON にも OFF にもできません。'), conf
    html, js = INDEX.read_text(encoding='utf-8'), APP.read_text(encoding='utf-8')
    panel = _panel_text(html)
    assert panel.count(FORM_B_PANEL) == 1 and panel.count(FORM_B_VREF) == 1, panel
    assert panel.endswith('速さの天井・自動合わせとは一緒に使えません。' + FORM_B_PANEL), panel
    assert js.count(FORM_B_CONFIRM) == 1 and html.count(FORM_B_PANEL) == 1 and html.count(FORM_B_VREF) == 1
    for w in FORM_A_WORDS:
        assert w not in conf and w not in panel and w not in js and w not in html, w
    src = DRIVE_PY.read_text(encoding='utf-8')
    assert not any(w in src for w in ('紙の表の「列」', '基準の電圧を左へ動か', '基準の電圧を動か'))
    assert src.count('列は速度ゲインの ± 0.01 で動かす') == 1 and src.count('1 枚の形 B では一日 8.10 V のまま') == 1


# ---------------------------------------------------------------- 直す前の写し・変えた所・見張りの段
def test_before_copies_are_the_real_of_11_01():
    """直す前の写しは SHA256SUMS・BEFORE_SHA と同じで、10/31 USB を外して走る の版(前の直しの試験の LATER_TETHER_1031 と同じ値)"""
    FB.check_before(ROOT)
    import test_ui_fix6
    import test_fix3_r3_2_tools
    later = dict(test_ui_fix6.LATER_TETHER_1031)
    assert later['jetformer/web/server.py'] == FB.BEFORE_SHA['server.py'] and later['jetformer/web/static/app.js'] == FB.BEFORE_SHA['app.js']
    t = test_fix3_r3_2_tools.LATER_TETHER_1031
    assert t['jetformer/drive.py'] == FB.BEFORE_SHA['drive.py'] and t['jetformer/web/static/index.html'] == FB.BEFORE_SHA['index.html']


def test_recorder_copy_is_the_speed_ff_version():
    """RECORDER_PY は固定の写し。中身がこの直しの版のままであること"""
    assert sha(RECORDER_PY) == FB.LATER_SPEED_FF_1101['jetformer/recorder.py']


def test_later_is_this_fix():
    """LATER_SPEED_FF_1101 は今のファイル(この直しの版)。このあと誰かが直せば落ちて知らせる(見張りの段も合わせる)"""
    for rel, h in FB.LATER_SPEED_FF_1101.items():
        assert sha(ROOT / rel) in (h, FB.ALSO_LATER.get(rel)), rel      # 2026-11-03 録画のロックの直しは tests/test_rec_imu_1103.py が見る
        assert FB.back(ROOT / rel) == BEFORE / Path(rel).name
    bm = FB.back_module(drive)
    assert bm is not drive and Path(bm.__file__) == BEFORE / 'drive.py' and bm.DEFAULT_PARAMS == before_drive().DEFAULT_PARAMS
    assert FB.back(ROOT / 'jetformer' / 'config.py') == ROOT / 'jetformer' / 'config.py'
    assert FB.back(BEFORE / 'drive.py') == BEFORE / 'drive.py'


def test_back_does_not_pass_a_broken_copy(tmp_path):
    """見張りの段は、写しの中身・SHA256SUMS が 1 つでも違えば通さない(AssertionError)。今のファイルが違えばそのまま"""
    root = tmp_path / 'jf'
    for rel in FB.LATER_SPEED_FF_1101:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
    shutil.copytree(BEFORE, FB.before_dir(root))
    assert FB.back(root / 'jetformer' / 'drive.py') == FB.before_dir(root) / 'drive.py'
    with open(FB.before_dir(root) / 'app.js', 'a', encoding='utf-8') as f:
        f.write('\n')
    with pytest.raises(AssertionError):
        FB.back(root / 'jetformer' / 'drive.py')
    shutil.copy2(BEFORE / 'app.js', FB.before_dir(root) / 'app.js')
    (FB.before_dir(root) / 'SHA256SUMS').write_text('0' * 64 + '  drive.py\n', encoding='utf-8')
    with pytest.raises(AssertionError):
        FB.back(root / 'jetformer' / 'drive.py')
    with open(root / 'jetformer' / 'drive.py', 'a', encoding='utf-8') as f:
        f.write('\n')
    assert FB.back(root / 'jetformer' / 'drive.py') == root / 'jetformer' / 'drive.py'


def _removed(before, after):
    a = Path(before).read_text(encoding='utf-8').splitlines()
    b = Path(after).read_text(encoding='utf-8').splitlines()
    return [x[1:] for x in difflib.unified_diff(a, b, n=0, lineterm='') if x.startswith('-') and not x.startswith('---')]


def test_changed_only_these_lines():
    """直す前の写しから消した・書き替えた行は、決めた行だけ(ほかは足しただけ)。index.html は足しただけ。config.py・style.css は
    変えていない"""
    assert _removed(BEFORE / 'drive.py', DRIVE_PY) == [
        "                 'rec_hold_keep', 'launch_no_kick')     # 2026-10-26 fix3 safety1-1・1-2",
        "                  'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}",
        "                  'launch_boost', 'rec_back_guard', 'rec_hold_keep', 'launch_no_kick'):     # 古い設定ファイル・古い画面には無い項目",
        "        for k in SPEED_CTRL_VALUES + LAUNCH_VALUES:    # 古い設定ファイルには無い",
        "    def start(self, run, mode_id, record=False, description='', tags=None, dry_run=False, params=None):",
        '        """走行を始める。戻り値 (run, token)。token は生存確認(heartbeat)に使う"""',
        "                       'tether_free': bool(config.DRIVE.get('tether_free', False))}",
        "                              'tether_free': bool(st.get('tether_free'))}    # 「USB を外して走る」の ON/OFF(2026-10-31)",
        "                    'gains': (p['steer_gain'], p['throttle_gain']), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],",
        "                    'launch_on': False, 'launch_end': int(lb_end)})",
        "                **sauto.record(int(p.get('speed_auto', 0) or 0) == 1, g_used, p['throttle_gain']),",
        "                'launch_on': bool(launch_on), 'launch_end': int(launch_end)})"]
    assert _removed(BEFORE / 'server.py', SERVER_PY) == [
        'DRIVE_CLIENT_VERSION = 10', "                                      dry_run=dry, params=body.get('params') or {})"]
    assert _removed(BEFORE / 'recorder.py', RECORDER_PY) == [
        '                     # 実際に掛けた速さのゲインはいつも gains[:,1]',
        '                     **{k: _ai_column(a, k, t, d) for k, t, d in CEIL_FIELDS + SA_FIELDS + LAUNCH_FIELDS})']
    assert _removed(BEFORE / 'index.html', INDEX) == []
    assert _removed(BEFORE / 'app.js', APP) == [
        'const DRIVE_CLIENT = 10;',
        '  const speedLine = speedStartLines(state.drive.st) + launchStartLine(state.drive.st);',
        '  // 文の最後の行を status の tether_free で決める。送る物は同じ)。古い画面からは走行を始めない)。build: この画面の app.js の更新時刻',
        "  const body = { client: 10, build: PAGE_BUILD, boot_at: state.drive.bootAt ?? null, run: m.name, mode_id: +$('#dr-mode').value, dry_run: dry,",
        '  zone_caps: 0, tcp_mode: 0, speed_scale: 1.0 };',
        "const BOOT_FLAGS = ['rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'speed_ceiling', 'speed_auto', 'launch_boost', 'zone_caps'];",
        '            rec_hold_keep: 0, launch_no_kick: 0 },',
        "                      launch_boost: '発進の前進', launch_us: '発進の前進の強さ', launch_mps: '発進の前進の速さ', launch_max_s: '発進の前進の時間' };"]
    import speed_ff_1101_back
    assert sha(ROOT / 'jetformer' / 'config.py') == '004ced0bc85a4ab27bf332e42b26f83dd1e73941ec30f49aa62231b217364c59'
    assert sha(STATIC / 'style.css') == 'b1e900656e2463c3c2292fba885bb415298a286aa131db25f481f5ecbce1924a'
    assert speed_ff_1101_back.ROOT == ROOT


def test_ff_used_only_where_written():
    """drive.py で電圧の前送りが出力に効くのは _loop の 1 か所(ff_applied のときの g_used)だけ。_loop は st['ff'] だけを読み、
    params の speed_ff は読まない(走行の途中で変わらない)。server.py で driver.start に足すのは ff_kw だけ"""
    src = DRIVE_PY.read_text(encoding='utf-8')
    loop = src[src.index('    def _loop('):src.index('    # ------------------------------------------------------------ 状態')]
    assert loop.count("g_used = p['throttle_gain'] * ff_m") == 1 and loop.count('ff_applied = ff_on and ai_frame and not auto_on') == 1
    assert "p.get('speed_ff'" not in loop and "p['speed_ff']" not in loop and "st.get('ff')" in loop
    s = SERVER_PY.read_text(encoding='utf-8')
    assert s.count('**ff_kw') == 1 and s.count(' if _ff_requested(body) else {}') == 1


# ---------------------------------------------------------------- 11/1 粗探しの指摘 S3: 変えた所(足した文まで AST で照らす)
def _mod_tree(path):
    return ast.parse(Path(path).read_text(encoding='utf-8'))


def _top_assign(path, name):
    for node in _mod_tree(path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) == name:
            return node
    raise AssertionError('%s に %s がありません' % (path, name))


def _drop_items(node, pred):
    """node の中の dict の表記から pred(key, value) が真の項目を除く(** の項目は key が None)。除いた数"""
    n = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            keep = [(k, v) for k, v in zip(sub.keys, sub.values) if not pred(k, v)]
            n += len(sub.keys) - len(keep)
            sub.keys[:] = [k for k, _ in keep]
            sub.values[:] = [v for _, v in keep]
    return n


class _IfExpToElse(ast.NodeTransformer):
    """「a if 条件 else b」のうち、文そのもの(ast.unparse)が srcs のどれかの物を b に替える(替えた文を seen に順に残す)"""

    def __init__(self, srcs):
        self.srcs, self.seen = set(srcs), []

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if ast.unparse(node) in self.srcs:
            self.seen.append(ast.unparse(node))
            return node.orelse
        return node


# _loop に足した文・式(ast.unparse の形)。中身まで決める(足した文を書き替えても落ちる)
LOOP_ADDED_STMTS = ["ffs = st.get('ff') or {}", "ff_on = bool(ffs.get('on'))", "ff_m = float(ffs['m']) if ff_on else 1.0",
                    'ff_applied = ff_on and ai_frame and (not auto_on)', "if ff_applied:\n    g_used = p['throttle_gain'] * ff_m",
                    "if ff_on:\n    st['now']['ff'] = {'on': True, 'm': _num(ff_m, 4), 'v0': _num(ffs.get('v0')), "
                    "'vref_v': _num(ffs.get('vref_v')), 'gain': _num(p['throttle_gain'] * ff_m, 4), 'applied': bool(ff_applied)}"]
LOOP_ADDED_IFEXP = ["p['throttle_gain'] * ff_m if ff_on and veh_select and (not dry_run) else p['throttle_gain']",     # 中立止めの行(S1)
                    "p['throttle_gain'] if ff_applied else g_used"]                                                     # 自動合わせの欄
LOOP_ADDED_ITEMS = ["{'ff_on': True, 'ff_m': ff_m, 'ff_applied': False} if ff_on else {}",                              # 中立止めの行
                    "{'ff_on': True, 'ff_m': ff_m, 'ff_applied': bool(ff_applied)} if ff_on else {}"]                   # ふつうの行


DRIVE_CHANGED = ['def:Driver._clean_params', 'def:Driver._ff_rules', 'def:Driver._ff_start', 'def:Driver._loop', 'def:Driver._run',
                 'def:Driver.set_params', 'def:Driver.start', 'def:Driver.status', 'def:ff_log', 'def:ff_multiplier', 'def:ff_spec',
                 'doc', 'set:DEFAULT_PARAMS', 'set:Driver.FF_NAME', 'set:FF_M_RANGE', 'set:FF_V0_RANGE', 'set:FF_VREF_STEP',
                 'set:PARAM_LIMITS', 'set:RESET_ON_BOOT', 'set:SPEED_FF_VERSION']


def test_python_changed_only_these_statements():
    """11/1 粗探しの指摘 S3: drive.py・server.py・recorder.py で、直す前の写しから変わった一番外側の名前は決めた物だけ(README 2.6 の表)。
    前からある関数・表は、この直しが足した文・項目を(数を決めて)取り除くと、直す前と AST が同じ(足した行がどこにあっても見つける。
    test_changed_only_these_lines は消した行だけを見る)。足した名前(ff_multiplier・FF_M_RANGE など)の中身は、ほかの試験が値と動きで見る"""
    old, now = BEFORE / 'drive.py', DRIVE_PY
    assert _ast_changed(old, now) == DRIVE_CHANGED
    d_old, d_new = ast.get_docstring(_mod_tree(old), clean=False), ast.get_docstring(_mod_tree(now), clean=False)
    assert d_new.startswith(d_old + '\n電圧の前送り(speed_ff。2026-11-01') and d_new.count('電圧の前送り(speed_ff。') == 1
    for name in ('PARAM_LIMITS', 'DEFAULT_PARAMS'):
        node = _top_assign(now, name)
        assert _drop_items(node, lambda k, v: isinstance(k, ast.Constant) and k.value in ('speed_ff', 'ff_vref_v')) == 2, name
        assert ast.dump(node) == ast.dump(_top_assign(old, name)), name
    node = _top_assign(now, 'RESET_ON_BOOT')
    assert [ast.unparse(e) for e in node.value.elts].count("'speed_ff'") == 1 and ast.unparse(node.value.elts[-1]) == "'speed_ff'"
    del node.value.elts[-1]
    assert ast.dump(node) == ast.dump(_top_assign(old, 'RESET_ON_BOOT'))
    # Driver._clean_params: フラグの tuple に 'speed_ff'・数の項目に ('ff_vref_v',)
    fn, n = _fn(now, 'Driver._clean_params'), 0
    for sub in ast.walk(fn):
        if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Tuple) and ast.unparse(sub.iter.elts[-1]) == "'speed_ff'":
            del sub.iter.elts[-1]
            n += 1
        elif isinstance(sub, ast.For) and ast.unparse(sub.iter) == "SPEED_CTRL_VALUES + LAUNCH_VALUES + ('ff_vref_v',)":
            sub.iter = sub.iter.left
            n += 1
    assert n == 2 and ast.dump(fn) == ast.dump(_fn(old, 'Driver._clean_params'))
    # Driver.set_params: 決まりを 1 つ足しただけ
    fn = _fn(now, 'Driver.set_params')
    assert _edit_stmts(fn, lambda s: ast.unparse(s) == 'p = self._ff_rules(p, self.running, values)') == 1
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver.set_params'))
    # Driver.start: 引数 2 つ(既定 None・'')・説明の文・文 2 つ・st の 1 項目
    fn, fo = _fn(now, 'Driver.start'), _fn(old, 'Driver.start')
    assert [a.arg for a in fn.args.args][-2:] == ['ff_v0', 'ff_v0_note'] and [ast.unparse(x) for x in fn.args.defaults][-2:] == ['None', "''"]
    del fn.args.args[-2:]
    del fn.args.defaults[-2:]
    assert ast.get_docstring(fn).startswith(ast.get_docstring(fo) + '。\nff_v0 = ')
    fn.body[0] = fo.body[0]
    assert _edit_stmts(fn, lambda s: ast.unparse(s) in ('new_params = self._ff_rules(new_params, False, params or {})',
                                                        'ff = self._ff_start(new_params, ff_v0, ff_v0_note)')) == 2
    assert _drop_items(fn, lambda k, v: isinstance(k, ast.Constant) and k.value == 'ff' and ast.unparse(v) == 'ff') == 1
    assert ast.dump(fn) == ast.dump(fo)
    # Driver._run: log.json の drive に **ff_log(...) の 1 項目
    fn = _fn(now, 'Driver._run')
    assert _drop_items(fn, lambda k, v: k is None and ast.unparse(v) == "ff_log(st.get('ff'), self.params)") == 1
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver._run'))
    # Driver._loop: m を読む 3 文・掛ける 2 文(ff_applied と if)・画面の値の if・中立止めの行のゲインと自動合わせの欄の 2 つの
    # 「… if … else 前の値」・記録の行の ff の欄 2 つ。これを取り除くと直す前と同じ(足した文・分かれがどこにあっても落ちる)
    fn, seen = _fn(now, 'Driver._loop'), []
    assert _edit_stmts(fn, lambda s: ast.unparse(s) in LOOP_ADDED_STMTS and not seen.append(ast.unparse(s))) == 6
    assert sorted(seen) == sorted(LOOP_ADDED_STMTS)
    t = _IfExpToElse(LOOP_ADDED_IFEXP)
    t.visit(fn)
    assert sorted(t.seen) == sorted(LOOP_ADDED_IFEXP) and len(t.seen) == 2
    assert _drop_items(fn, lambda k, v: k is None and ast.unparse(v) in LOOP_ADDED_ITEMS) == 2
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver._loop'))
    # Driver.status: 1 行
    fn = _fn(now, 'Driver.status')
    assert _edit_stmts(fn, lambda s: ast.unparse(s) == "st['ff_spec'] = ff_spec(self.params)") == 1
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver.status'))
    # server.py: 版の数・_ff_requested(新しい)・api_drive_start(ff_kw の 1 文と **ff_kw)
    so, sn = BEFORE / 'server.py', SERVER_PY
    assert _ast_changed(so, sn) == ['def:_ff_requested', 'def:api_drive_start', 'set:DRIVE_CLIENT_VERSION']
    fn, n = _fn(sn, 'api_drive_start'), 0
    assert _edit_stmts(fn, lambda s: ast.unparse(s) == "ff_kw = {'ff_v0': vrun.get('v_before'), 'ff_v0_note': "
                       "vrun.get('v_before_note') or ''} if _ff_requested(body) else {}") == 1
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call):
            ks = [k for k in sub.keywords if k.arg is None and ast.unparse(k.value) == 'ff_kw']
            n += len(ks)
            sub.keywords[:] = [k for k in sub.keywords if k not in ks]
    assert n == 1 and ast.dump(fn) == ast.dump(_fn(so, 'api_drive_start'))
    # recorder.py: FF_FIELDS(新しい)と AI_CMD の欄の並びに + FF_FIELDS
    ro, rn = BEFORE / 'recorder.py', RECORDER_PY
    assert _ast_changed(ro, rn) == ['def:Recorder._save_scene', 'set:FF_FIELDS']
    fn, n = _fn(rn, 'Recorder._save_scene'), 0
    for sub in ast.walk(fn):
        if isinstance(sub, ast.comprehension) and ast.unparse(sub.iter) == 'CEIL_FIELDS + SA_FIELDS + LAUNCH_FIELDS + FF_FIELDS':
            sub.iter = sub.iter.left
            n += 1
    assert n == 1 and ast.dump(fn) == ast.dump(_fn(ro, 'Recorder._save_scene'))


# ---------------------------------------------------------------- 11/1 粗探しの指摘 S1: 中立止めの行のゲインと夜の道具
def test_hold_rows_keep_the_ai_gain_for_night_tools(monkeypatch, tmp_path):
    """ON の走行の AI 側の中立止めの行(プロポの電波切れ・車両の値が古い など)も gains[1] は throttle_gain × m(中立なので掛けては
    いない = ff_applied False)。夜の道具 tools/standing_3lap.py の midrun_changes は、ON の本を「途中で速度ゲインを変えた」と読まない
    (OFF の本と同じ [])。前は中立止めの行だけ throttle_gain のままで、「速さ 0.866→0.9」と出て 3 周目が終わる発進を数えなかった"""
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    import standing_3lap as s3
    frames, imu, base, params = scene('holds')
    tg, m = 0.9, 8.10 / 8.42
    res = {}
    for name, p, v0 in (('on', dict(params, throttle_gain=tg, speed_ff=1, ff_vref_v=8.10), 8.42), ('off', dict(params, throttle_gain=tg), None)):
        d, pca, rec, _ = run_ff(drive, monkeypatch, tmp_path, frames, imu, p, v0=v0, base=base)
        rows = [r for _, r in rec.rows]
        lab = {'t_s': np.array([(t - T0) / 1e9 for t, _ in rec.rows]), 'gains': np.array([r['gains'] for r in rows], np.float32),
               'trim': np.array([r['trim'] for r in rows], np.float32)}
        ai = np.array([bool(r['select']) and not r['dry_run'] for r in rows])
        raw = tmp_path / ('raw_' + name)
        raw.mkdir()
        res[name] = s3.midrun_changes({'drive': {'params': dict(p)}}, {}, lab, ai, [], str(raw), 'nolog', None)
        holds = [r for r in rows if r['hold'] and r['select'] and not r['dry_run']]
        assert len(holds) >= 3 and any(not r['select'] for r in rows)            # AI 側の中立止めと人側の行がある場面
        if name == 'on':
            assert all(r['gains'][1] == tg * m and r['ff_applied'] is False and r['ff_m'] == m for r in holds)
            assert {r['gains'][1] for r in rows if r['select'] and not r['dry_run']} == {tg * m}
            assert {r['gains'][1] for r in rows if not r['select']} == {tg}
        else:
            assert {r['gains'][1] for r in rows} == {tg}
    assert res == {'on': [], 'off': []}


# ---------------------------------------------------------------- 文書の版の台本(docs_ban9.py の 2026-11-01 の段)
DOCS_TOOL = ROOT / 'experiments' / '20261026_fix' / 'fix6' / 'ui' / 'docs_ban9.py'
_DSEQ = [0]


def _load_docs(path=DOCS_TOOL):
    _DSEQ[0] += 1
    spec = importlib.util.spec_from_file_location('docs_ban9_speed_ff_%d' % _DSEQ[0], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _docs_tree(tmp_path):
    """偽の ~/jetformer: app.js・server.py はこの直しの版(写しの今のファイル)、tests/fixtures の 2 つの直す前の写し(この直しのと 10/31 の)"""
    jf = tmp_path / 'jf'
    for rel in ('jetformer/web/static/app.js', 'jetformer/web/server.py'):
        (jf / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, jf / rel)
    shutil.copytree(BEFORE, FB.before_dir(jf))
    shutil.copytree(ROOT / 'tests' / 'fixtures' / 'tether_1031_before', jf / 'tests' / 'fixtures' / 'tether_1031_before')
    m = _load_docs()
    m.JF, m.APP, m.SERVER = jf, jf / 'jetformer/web/static/app.js', jf / 'jetformer/web/server.py'
    m.TETHER_1031_BEFORE = jf / 'tests' / 'fixtures' / 'tether_1031_before'
    m.SPEED_FF_1101_BEFORE = FB.before_dir(jf)
    return m, jf


def test_docs_tool_reads_the_version_through_both_copies(tmp_path):
    """docs_ban9.py: 今の app.js・server.py が両方ともこの直しの版(版 11)なら、この直しの直す前の写し(版 10)を「今の 2 つ」にして
    10/31 の段をたどり、10/31 の直す前の写し(版 9)から (9, 9, 9) を読む(文書は 10/26〜10/28 の記録で書き換えない = 前と同じ答え)。
    決めた値は tests/speed_ff_1101_back.py と同じ・写しの中身を確かめる"""
    m, jf = _docs_tree(tmp_path)
    assert m.LATER_SPEED_FF_1101 == {'app.js': FB.LATER_SPEED_FF_1101['jetformer/web/static/app.js'],
                                     'server.py': FB.LATER_SPEED_FF_1101['jetformer/web/server.py']}
    assert m.SPEED_FF_1101_BEFORE_SHA == {'app.js': FB.BEFORE_SHA['app.js'], 'server.py': FB.BEFORE_SHA['server.py']}
    assert m.version_files() == (m.TETHER_1031_BEFORE / 'app.js', m.TETHER_1031_BEFORE / 'server.py')
    assert m.versions() == (9, 9, 9)
    real = _load_docs()
    assert real.SPEED_FF_1101_BEFORE == Path('/home/jetson/jetformer/tests/fixtures/speed_ff_1101_before')
    # 1 バイトでも違えば、今まで通り本物から読む(版 11 = 文書の試験は落ちて知らせる)
    with open(m.APP, 'a', encoding='utf-8') as f:
        f.write('\n')
    assert m.version_files() == (m.APP, m.SERVER) and m.versions() == (11, 11, 11)


@pytest.mark.parametrize('brk', ['app_byte', 'sums_line', 'no_sums', 'no_dir'])
def test_docs_tool_refuses_a_broken_copy(tmp_path, brk):
    """この直しの版のとき、直す前の写しが SHA256SUMS・決めた値とそろわなければ(無いときも)黙って切りかえず RuntimeError"""
    m, jf = _docs_tree(tmp_path)
    d = FB.before_dir(jf)
    if brk == 'app_byte':
        with open(d / 'app.js', 'a', encoding='utf-8') as f:
            f.write('\n')
    elif brk == 'sums_line':
        (d / 'SHA256SUMS').write_text((d / 'SHA256SUMS').read_text().replace(FB.BEFORE_SHA['server.py'], '0' * 64))
    elif brk == 'no_sums':
        (d / 'SHA256SUMS').unlink()
    else:
        shutil.rmtree(d)
    with pytest.raises(RuntimeError, match='直す前の写し'):
        m.versions()


def test_docs_tool_changed_only_the_new_stage():
    """docs_ban9.py で消した・書き替えた行は version_files の始めの 3 行だけ(ほかは足しただけ)。直す前の写し(10/31 の docsver の版)の
    ほかの関数・EDITS・HISTORY は AST で同じ"""
    import ast
    assert _removed(BEFORE / 'docs_ban9.py', DOCS_TOOL) == [
        '    どちらか 1 つでも違えば、今まで通り本物 (APP, SERVER)"""',
        "    if _sha(APP) != LATER_TETHER_1031['app.js'] or _sha(SERVER) != LATER_TETHER_1031['server.py']:",
        '        return APP, SERVER']

    def top(p):
        out = {}
        for n in ast.parse(Path(p).read_text(encoding='utf-8')).body:
            if isinstance(n, ast.Assign):
                k = 'set:' + ','.join(t.id for t in n.targets if isinstance(t, ast.Name))
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                k = 'def:' + n.name
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                k = 'import:' + ','.join(a.name for a in n.names)
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                k = 'doc'
            else:
                k = 'other:' + ast.dump(n)
            assert k not in out, k
            out[k] = ast.dump(n)
        return out
    o, n = top(BEFORE / 'docs_ban9.py'), top(DOCS_TOOL)
    assert sorted(k for k in o if o[k] != n.get(k)) == ['def:version_files', 'doc'] and set(o) <= set(n)
    assert sorted(set(n) - set(o)) == ['def:_before_pair', 'set:LATER_SPEED_FF_1101', 'set:SPEED_FF_1101_BEFORE',
                                       'set:SPEED_FF_1101_BEFORE_SHA']
