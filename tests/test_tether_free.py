"""「USB を外して走る」(tether_free。いつも ON)と、止めたときの保存(os.sync)の試験(2026-10-31)。

ユーザー(10/31 08:08)「予選会場では無線が使えなくなる可能性があるので、USBで接続し、192.168.55.1:8080でブラウザで接続し、
AIモデルを選んで走行開始ボタンを押し、その後USBを外して、実車で走行する感じになります。走行が終わったら、USBで繋いで、AIを停止します。」
ユーザーの決め(10/31 10:2x)「USB を外して走るは、基本常にOnです。選択ボタンも入りません」。
前は、走行を始めた画面からの生存確認(GET /api/drive/status?token=…)が config.DRIVE['heartbeat_s'](5 秒)途絶えると走行ループが
止まったので、USB を抜いた約 5 秒後に AI が止まった。直し:
  - config.DRIVE['tether_free'] = True(既定)。Driver.start が走行を始めるときに読んで st['tether_free'] に入れ、その走行の途中では
    変えない。画面にスイッチは無く、走行開始の body の tether_free は読まない(来ても無視)。
  - True の走行は、画面との通信が切れても止めない。念のための上限 config.DRIVE['tether_free_max_s'](1800 秒 = 30 分): 最後の生存確認
    (一度も来ていなければ走行開始)からそれだけ途絶えたら止める。CH3・センサーの見張り・プロポの電波切れの中立・ほかの止め方
    (POST /api/drive/stop は token が要らない)は全部そのまま。status と収集の log.json(drive.tether_free)に残す。
    送信機の電源断(プロポの電波切れ)は今まで通り一時的な中立で、走行は終わらない(電源を入れて CH3 が AI 側なら AI に戻る。
    USB を抜いたまま最長 30 分この形が続くので、README に「送信機の電源を入れる前に CH3 を人側に」。2026-10-31 粗探しの指摘)。
  - config を False にすると前の決まり(heartbeat_s で止める。直す前の写しとビット単位で同じ)。
  - リポジトリの README.md(AI走行の章)を今の決まりに合わせた(画面との通信が切れても止めない・30 分・False なら 5 秒・
    送信機の電源断は一時的な中立・収集中の 2 秒ごとの書き込み。数は config の値と照らす)。
  - 画面の版 9 → 10。版 10 の画面は ▶ の確認の文の最後の行・▶ の上の 1 行・走行中の印を status の tether_free(と tether_free_max_s)で
    決める(false・知らないサーバーは今までの文)。送る物は版のほかは前と同じ。版 9 の画面(「通信断が5秒続くと止まります」と出す)からは
    走行を始めない。
  - 収集した走行の後始末で、recorder.stop のあとに os.sync() を 1 回(10/30 は電源断で最後の約 39 秒の記録が消えた)。
    収集している間の定期的な書き込み(recorder の rec-sync・config.RECORD_SYNC_S。ユーザー 10/31 10:3x「データは定期的に書き込みが
    いいですね」)の試験は tests/test_record_sync.py。
直す前の写し tests/fixtures/tether_1031_before/(drive.py・server.py・app.js・index.html・config.py・recorder.py と、版の 1 行を合わせた
tests/test_vbat_ui.py。git の 29aa281 = 10/29 fix9 のあとの版。sha256 は SHA256SUMS)。

実機・GPU・torch・サーバーの起動は使わない(センサー・PCA9685・TensorRT・時計は偽物。走行ループの偽物は tests/test_launch_drive.py・
test_drive_caps.py と同じ。走行は本物の Driver.start で始め(走行のスレッドの中身 _run は回さない)、start が作った st で本物の _loop を
回す。走行の後始末の試験は torch・PCA9685・TensorRT の模型を sys.modules に入れ、os.sync は drive の os を sync だけ偽物にした入れ物に
替える。server は import するだけ = tests/test_vbat_desc.py と同じ。画面は node で本物の app.js を読む = tests/fixtures/app_sim_fix8.js の
写しの偽のサーバーの status に tether_free などを足して使う。もとの台本は変えない)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q tests/test_tether_free.py
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
import sys
import threading
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from jetformer import assist as A
from jetformer import clock, config, drive, sound
from jetformer.sensors import vehicle as vs
from jetformer.web import bench as bench_web
from jetformer.web import server as srv

import speed_ff_1101_back as FB      # 2026-11-01 電圧の前送り(speed_ff)の段: drive.py などがその直しの版なら直す前の写し(= この直しの版)で照らす

from test_drive_caps import FakePCA, FakeRecorder, SeqRunner, fake_normalize, same, scenario
from test_drive_caps import Feeder as CapsFeeder
from test_launch_drive import C, FLOOR_MM, N, PARAMS, REF, T0, build, launch_scene, tof_near
from test_launch_drive import Feeder as LaunchFeeder

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'tether_1031_before'
STATIC = ROOT / 'jetformer' / 'web' / 'static'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix8.js'
NODE = shutil.which('node')
# 2026-11-01 電圧の前送り(speed_ff)が画面の版を 10 → 11 にした(server.py・app.js)。今のファイルがその直しの版なら 11(この直しの中身と
# 版 11 は tests/test_speed_ff.py が見る)。この試験の「直す前の写しから変えた所」の比べは、その直しの直す前の写し(FB.back = この直しの版)で
UI_VER = 11 if FB.later_rel(ROOT / 'jetformer' / 'web' / 'server.py') else 10
UI_VER_JS = 11 if FB.later_rel(STATIC / 'app.js') else 10
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
# 直す前の写しの sha256(drive.py・app.js は前の直しの試験の LATER_FIX9_PRIORITY と同じ値。recorder.py は git の 29aa281 と同じ)
BEFORE_SHA = {'drive.py': 'ecfb6656cae624950d5de221dfa71a5ac31b84ba82dacfbea2136832eae92315',
              'server.py': 'ad7a7d05f7a9167c2992d78d77e39f05f49153cc3b13d8651b166efb4a073b94',
              'app.js': '6b9e160c43c870dc1f4c2a5e2c23e2ad7513fe7e925d76a12cb69d5854966233',
              'index.html': 'ef81bb6017d63522bf4b870f0d156ce38673f5af3fd4249a31bcfbe019d191ec',
              'config.py': '1736a59382af50a767aff1b0090bb35d8e33f9c2a5f12c8082ca6995bda6159b',
              'test_vbat_ui.py': 'e022452ea861ce875dcbeeca0903f2d75201d2525bcafc14f86238b1ba8ab396',   # 版の 1 行を変えた試験(test_ui_fix8_r2 が照らす)
              'recorder.py': '34e644e572e7e721c3342f2b98472c403362014271c470ac026985e7c91e4d0f'}      # 定期的な書き込み(tests/test_record_sync.py)
OLD_LAST = 'この画面は開いたままにしてください(閉じる・スリープ・通信断が5秒続くと止まります)。'
NEW_LAST = ('USB を外しても AI は止まりません(画面との通信が 30 分途絶えたら止まります)。止めるときは、CH3 を人側にしてから、'
            'USB をつなぎ直して ■ 停止を押してください。')
NOTE_ON = ('USB を外して走れます(画面との通信が切れても AI は止まりません・30 分途絶えたら止まります)。止めるときは CH3 を人側 → '
           'USB をつなぎ直して ■ 停止')
MARK = 'USB を外しても止まりません(画面との通信が 30 分途絶えたら止まります)'
HB_TEXT = '走行を始めた画面との通信が 5 秒途絶えたため'
HB_NEVER = '走行を始めた画面からの応答が一度もありませんでした。画面を再読み込みしてから開始してください'
TF_TEXT = '「USB を外して走る」の上限: 走行を始めた画面との通信が 30 分途絶えたため'
START_KW = {'record', 'description', 'tags', 'dry_run', 'params'}      # server が driver.start に渡す名前(直す前と同じ)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_BEFORE = {}


def load_before():
    """直す前の drive.py(写し)をモジュールにする(sha256 を照らしてから)"""
    if _BEFORE:
        return _BEFORE['d']
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sums == BEFORE_SHA
    for name, h in sums.items():
        assert sha(FIX / name) == h, name
    src = (FIX / 'drive.py').read_text(encoding='utf-8')
    assert 'tether' not in src                                  # 写しは直す前
    spec = importlib.util.spec_from_loader('jetformer._drive_before_tether', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile(src, str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    _BEFORE['d'] = mod
    return mod


def _run_dir(runs, name):
    (runs / name).mkdir(parents=True)
    (runs / name / drive.ENGINE_FILE).write_bytes(b'engine')
    (runs / name / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'trained_modes': [4, 5],
                                                               'steer_center_us': REF[0], 'throttle_neutral_us': REF[1]}))


class OsProxy(object):
    """モジュールの os の代わり: sync だけ偽物、ほかは本物の os(そのモジュールの os.sync だけを偽物にする。本物の os を書き換えないので、
    ほかのスレッド(収集の rec-sync など)の os.sync は数えない)"""

    def __init__(self, sync):
        self.sync = sync

    def __getattr__(self, name):
        return getattr(os, name)


# ---------------------------------------------------------------- 走行ループ(USB を抜いた・つなぎ直した)
class Unplug(object):
    """cam_evt の代わり。base(test_launch_drive・test_drive_caps の Feeder)はフレームごとに生存確認を入れるので、それを取り消し、
    connected(単調時計の秒)が True の間だけ本物の Driver.heartbeat を start の token で呼ぶ(画面の GET /api/drive/status?token= が
    届く = USB がつながっている)。jump(フレームの番号)= そのフレームで単調時計だけを余分に進める秒(30 分の上限を短いフレームの列で
    試す。センサーの時刻は 30 fps のまま)。hook(フレームの番号, driver) はフレームの前(停止ボタン・config の書き換えなど)。
    ms = 処理したフレームの単調時計"""

    def __init__(self, base, d, box, connected, jump=None, hook=None):
        self.base, self.d, self.box, self.connected, self.jump, self.hook = base, d, box, connected, jump, hook
        self.extra, self.ms, self.hb_ms = 0.0, [], []

    def wait(self, timeout):
        i = self.base.i
        if self.hook is not None and i < len(self.base.frames):
            self.hook(i, self.d)
        hb_t = self.d.heartbeat_t
        ok = self.base.wait(timeout)
        if ok:
            self.extra += float(self.jump(i)) if self.jump is not None else 0.0
            self.box['m'] += self.extra
            self.d.heartbeat_t = hb_t
            if self.connected(self.box['m']):
                assert self.d.heartbeat(self.d.hb_token) is True
                self.hb_ms.append(self.box['m'])
            self.ms.append(self.box['m'])
        return ok

    def clear(self):
        pass

    def set(self):
        pass


def run_tf(mod, monkeypatch, tmp_path, frames, imu, connected=lambda m: True, jump=None, hook=None, base='launch',
           params=None, dry_run=False):
    """mod(今の drive か直す前の写し)で、本物の Driver.start で走行を始め(走行のスレッドの中身 _run は回さない)、start が作った
    st で _loop を決まった時刻で回す → (driver, pca, recorder, unplug)。start は heartbeat_t を始めた時刻(単調時計 100.0)・hb_count 0
    にし、今の drive は st['tether_free'] を config.DRIVE['tether_free'] から入れる(直す前の写しの st には無い)"""
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
    if base == 'launch':                    # ToF ガードの床の基準(test_launch_drive の場面。test_drive_caps の場面は基準なし)
        np.save(fp, np.full((42, 54), float(FLOOR_MM), np.float32))
    monkeypatch.setattr(mod, 'FLOOR_REF_PATH', fp if base == 'launch' else tmp_path / 'no_floor_ref.npy')
    rec = FakeRecorder()
    d = mod.Driver(hub=None, recorder=rec)
    d._run = lambda *a: None                                                # 走行のスレッドの中身は回さない(st を下の _loop に渡す)
    d.start('m1', 4, dry_run=dry_run)
    d.thread.join(5)
    assert d.heartbeat_t == 100.0 and d.hb_count == 0 and d.st['phase'] == 'starting' and d.hb_token
    d.params = dict(params or PARAMS)
    pca = FakePCA()
    out = mod._Output(pca, config.PCA9685, REF, lambda: d.params)
    bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
    cam_slot = [None]
    feeder = (LaunchFeeder if base == 'launch' else CapsFeeder)(d, frames, imu, bufs, cam_slot, box)
    unplug = Unplug(feeder, d, box, connected, jump, hook)
    runner = SeqRunner([f['pred'] for f in frames])
    d._loop(d.st, runner, fake_normalize, pca, out, REF, 6, 0.0, cam_slot, unplug, *bufs, dry_run)
    d._stop.clear()
    return d, pca, rec, unplug


def cruise(dur):
    """AI 側で 2 m/s で走り続ける場面(モデルは中立 − 120 µs の前進)"""
    return build(dur, sel=True, v=2.0, pred=(C, N - 120.0))


def holds_scene():
    """抜いたあと(0.5 秒)に、中立止めと補助が全部ある場面(test_launch_drive.build。ToF ガードの床の基準あり。ほかは AI 側で 2 m/s・
    モデルは中立 − 120 µs): 2.0〜2.5 秒 CH3 を人側 / 3.0〜3.2 秒 AI 側のまま CH3 が届かない(プロポの電波切れ)/ 4.0〜4.4 秒 車両の
    値が来ない(古い)/ 5.0〜5.3 秒 前 0.4 m に物(ToF ガード)/ 6.0〜7.0 秒 止まっていてモデルはブレーキ(後退の受け付け = 中立を挟む)"""
    return build(8.0, sel=lambda s: not (2.0 <= s < 2.5), valid3=lambda s: not (3.0 <= s < 3.2),
                 veh_skip=lambda s: 4.0 <= s < 4.4, tof=lambda s: tof_near(400) if 5.0 <= s < 5.3 else None,
                 v=lambda s: 0.0 if 6.0 <= s < 7.0 else 2.0,
                 pred=lambda s: (C, N + 100.0) if 6.0 <= s < 7.0 else (C, N - 120.0))


def scene_of(name):
    """場面の名前 → (frames, imu, base)"""
    if name == 'cruise':
        return cruise(8.0) + ('launch',)
    if name == 'holds':
        return holds_scene() + ('launch',)
    return scenario() + ('caps',)


def same_rows(ra, rb):
    """記録の行(時刻・欄の名前・値。infer_ms は時間の計測なので除く)がビット単位で同じ"""
    assert [t for t, _ in ra] == [t for t, _ in rb]
    for (t, a), (_, b) in zip(ra, rb):
        assert set(a) == set(b), t
        for k in a:
            assert k == 'infer_ms' or same(a[k], b[k]), (t, k, a[k], b[k])


def same_ev(a, b):
    return set(a) == set(b) and all(same(a[k], b[k]) if not isinstance(a[k], dict) else a[k] == b[k] for k in a)


def assert_same_run(a, b):
    """2 つの走行ループ(run_tf の戻り値)の PCA9685 に書いた値・記録の行・出来事・止めた理由・処理したフレームが同じ"""
    (da, pa, ra, ua), (db, pb, rb, ub) = a, b
    assert pa.pairs == pb.pairs and pa.single == pb.single
    same_rows(ra.rows, rb.rows)
    assert len(ra.events) == len(rb.events) and all(ka == kb and same_ev(xa, xb) for (ka, xa), (kb, xb) in zip(ra.events, rb.events))
    assert da.st.get('stop_reason', '') == db.st.get('stop_reason', '') and ua.ms == ub.ms


def test_config_and_texts():
    """既定は ON(ユーザーの決め「基本常にOn」)。上限は 1800 秒(30 分)。生存確認の 5 秒は変えない(False のときだけ使う)。上限の理由の文に
    「USB を外して走る」と 30 分。音声の案内は「画面との通信が途絶えたため、AI走行を止めました」(注意の音)"""
    assert config.DRIVE['tether_free'] is True
    assert config.DRIVE['heartbeat_s'] == 5.0 and config.DRIVE['tether_free_max_s'] == 1800
    assert drive.tether_free_stop_text(1800) == TF_TEXT and drive.tether_free_stop_text(1800.0) == TF_TEXT
    assert drive.tether_free_stop_text(600) == TF_TEXT.replace('30 分', '10 分')
    assert drive.tether_free_stop_text(90) == '「USB を外して走る」の上限: 走行を始めた画面との通信が 90 秒途絶えたため'
    assert sound.phrase('drive_stop', reason=TF_TEXT) == ('画面との通信が途絶えたため、AI走行を止めました', 'warn')
    assert sound.phrase('drive_stop', reason=TF_TEXT) == sound.phrase('drive_stop', reason=HB_TEXT)    # 前と同じ案内


def test_start_reads_config_per_run(monkeypatch, tmp_path):
    """走行を始めるとき(Driver.start)に config.DRIVE['tether_free'] を読んで st に入れる(True・False)。start の引数は直す前と同じ
    (走行ごとの引数は無い)。params には入らない(知らない項目は読み飛ばす)。config に無ければ False(前の決まり)"""
    assert inspect.signature(FB.back_module(drive).Driver.start) == inspect.signature(load_before().Driver.start)   # 2026-11-01 の段
    assert 'tether_free' not in drive.DEFAULT_PARAMS and 'tether_free' not in drive.PARAM_LIMITS
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    runs = tmp_path / 'runs'
    _run_dir(runs, 'm1')
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    d._run = lambda *a: None
    seen = []
    for v in (True, False, 1, 0, None):
        monkeypatch.setitem(config.DRIVE, 'tether_free', v)
        d.start('m1', 4)
        d.thread.join(5)
        seen.append(d.st['tether_free'])
    monkeypatch.delitem(config.DRIVE, 'tether_free')
    d.start('m1', 4, dry_run=True)
    d.thread.join(5)
    seen.append(d.st['tether_free'])
    assert seen == [True, False, True, False, False, False]
    d.set_params({'tether_free': 1, 'steer_gain': 1.1})
    assert 'tether_free' not in d.params and d.params['steer_gain'] == 1.1


@pytest.mark.parametrize('scene', ['cruise', 'launch'])
def test_default_on_keeps_driving_after_unplug_like_always_connected(monkeypatch, tmp_path, scene):
    """既定(config ON)の走行は、USB を抜いても(そのあと生存確認が来ない)heartbeat_s(5 秒)で止めず、最後のフレームまで走る。
    PCA9685 に書いた値・記録の行・出来事は、直す前の写しで画面がずっとつながっていた走行とビット単位で同じ(止めないだけ)。
    同じ場面を config OFF で流すと、前と同じに 5 秒で止まる(直す前の写しで同じく抜いた走行とビット単位で同じ)"""
    assert config.DRIVE['tether_free'] is True
    if scene == 'cruise':                   # 20 秒・1 秒で抜く
        (frames, imu), params, cut = cruise(20.0), PARAMS, 101.0
    else:                                   # 発進の前進(launch_boost)・人側と AI 側の行き来・止まっている間のブレーキ(6 秒)・0.5 秒で抜く
        (frames, imu), params, cut = launch_scene(), dict(PARAMS, launch_boost=1), 100.5
    unplug = lambda m: m <= cut                                             # noqa: E731
    on = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=unplug, params=params)
    ref = run_tf(load_before(), monkeypatch, tmp_path, frames, imu, connected=lambda m: True, params=params)
    d, pca, rec, u = on
    assert d.st['tether_free'] is True and d.st.get('stop_reason', '') == '' and len(u.ms) == len(frames)
    assert u.hb_ms and max(u.hb_ms) <= cut and u.ms[-1] - max(u.hb_ms) > (18.0 if scene == 'cruise' else 5.0)
    assert_same_run(on, ref)
    if scene == 'cruise':
        ai = [r for _, r in rec.rows if not r['hold']]
        assert ai and ai[-1]['out'][1] < N - 50                             # 最後まで AI の前進を出している(中立で止めていない)
    else:
        assert d.st['stats']['launch_boosts'] > 0                           # 抜いたあとの発進の前進も同じ
    monkeypatch.setitem(config.DRIVE, 'tether_free', False)
    off = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=unplug, params=params)
    old = run_tf(load_before(), monkeypatch, tmp_path, frames, imu, connected=unplug, params=params)
    assert off[0].st['tether_free'] is False and off[0].st['stop_reason'] == HB_TEXT and len(off[3].ms) < len(frames)
    assert_same_run(off, old)


@pytest.mark.parametrize('scene', ['holds', 'caps'])
def test_default_on_keeps_safety_holds_unchanged(monkeypatch, tmp_path, scene):
    """既定(ON)でも中立止め(CH3 を人側・AI 側のまま CH3 が届かない = プロポの電波切れ・車両の値が古い・ToF ガード・後退の受け付け)は
    今まで通り: 0.5 秒で USB を抜いた走行と、直す前の写しで画面がずっとつながっていた走行で、書いた値・記録の行・出来事がビット単位で
    同じ。どれも抜いたあとにあり、走行は最後まで続く(holds = 上の holds_scene。caps = 区間の速度上限の試験の場面 tests/test_drive_caps
    の scenario: 人側 200〜214・CH3 が届かない 230〜232 のフレーム)"""
    frames, imu, base = scene_of(scene)
    on = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 100.5, base=base)
    ref = run_tf(load_before(), monkeypatch, tmp_path, frames, imu, connected=lambda m: True, base=base)
    assert_same_run(on, ref)
    d, _, rec, u = on
    assert len(u.ms) == len(frames) and d.st.get('stop_reason', '') == '' and max(u.hb_ms) <= 100.5
    k_of = {f['cam_t']: k for k, f in enumerate(frames)}
    rows = [(k_of[t], (t - T0) / 1e9, r) for t, r in rec.rows]
    if scene == 'holds':
        win = lambda a, b: [r for _, s, r in rows if a <= s < b]                                  # noqa: E731
        assert win(2.05, 2.45) and all(not r['select'] for r in win(2.05, 2.45))                 # 人側(CH3)
        assert not any(r['hold'] for r in win(2.6, 2.95))
        assert any(r['hold'] and r['select'] for r in win(3.0, 3.25))                            # AI 側のまま CH3 が届かない
        assert any(r['hold'] for r in win(4.1, 4.45))                                            # 車両の値が古い
        assert any(r.get('guard', 0) > 0 for r in win(5.0, 5.4))                                 # ToF ガード
        assert any(r.get('assist') == A.ASSIST_ESC for r in win(6.0, 7.1))                       # 後退の受け付け(中立を挟む)
        assert not any(r['hold'] for r in win(7.3, 8.0))                                         # そのあとは今まで通り走る
    else:
        by_k = {k: r for k, _, r in rows}
        assert all(not by_k[k]['select'] for k in range(201, 214))                               # 人側(CH3)
        assert all(by_k[k]['hold'] and by_k[k]['select'] for k in range(230, 233))               # AI 側のまま CH3 が届かない
    assert d.st['stats']['holds'] > 0


@pytest.mark.parametrize('case', ['tx_off', 'failsafe_human'])
def test_default_on_tx_off_is_a_temporary_neutral(monkeypatch, tmp_path, case):
    """既定(ON)で USB を抜いたあとの送信機の電源断は一時的な中立(か人側)で、走行は終わらない。送信機の電源を入れて CH3 が AI 側なら、
    そのまま AI の出力に戻る(覚えておく印は無い = 直す前と同じ。前は USB を抜くと 5 秒で走行が終わったので、この形はほとんど起きな
    かった。README の「送信機の電源を入れる前に CH3 を人側に」の元)。tx_off = 受信機のパルスが止まる設定(基板は SELECT を AI 側の
    まま保つ → 2.0〜5.0 秒は「AI 側のまま CH3 が届かない」の中立)。failsafe_human = 受信機のフェイルセーフで CH3 が人側(2.0〜5.0 秒は
    人側で止まっている・発進の前進 ON)→ 電源を入れると人側 → AI 側の切り替わりと同じで、その最初のフレームから発進の前進。
    どちらも、直す前の写しで画面がずっとつながっていた走行とビット単位で同じ。車のスピーカーの案内(走行ループの出来事): tx_off は
    「プロポの信号が届きません。停止します」だけで、電源を入れて AI に戻るときは何も言わない。failsafe_human は「人の操縦に戻りました」
    →「AIが操縦します」"""
    if case == 'tx_off':
        frames, imu = build(8.0, sel=True, v=2.0, valid3=lambda s: not (2.0 <= s < 5.0), pred=(C, N - 120.0))
        params = PARAMS
    else:
        frames, imu = build(8.0, sel=lambda s: not (2.0 <= s < 5.0), v=lambda s: 0.0 if 2.0 <= s < 5.2 else 2.0,
                            pred=(C, N - 120.0))
        params = dict(PARAMS, launch_boost=1)
    evs = []
    emit = drive.Driver._emit
    monkeypatch.setattr(drive.Driver, '_emit', lambda self, ev, **kw: (evs.append((ev, kw)), emit(self, ev, **kw)))
    on = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 100.5, params=params)
    ref = run_tf(load_before(), monkeypatch, tmp_path, frames, imu, connected=lambda m: True, params=params)
    assert_same_run(on, ref)
    said = [sound.phrase(ev, **kw)[0] for ev, kw in evs]
    assert said == (['AI走行を開始します', 'プロポの信号が届きません。停止します'] if case == 'tx_off' else
                    ['AI走行を開始します', '人の操縦に戻りました', 'AIが操縦します'])
    d, _, rec, u = on
    assert d.st['tether_free'] is True and d.st.get('stop_reason', '') == '' and len(u.ms) == len(frames)
    assert max(u.hb_ms) <= 100.5
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    gone = [r for s, r in rows if 2.05 <= s < 4.95]                         # 送信機の電源が切れている間
    back = [r for s, r in rows if s >= 5.0]                                 # 電源を入れたあと(CH3 は AI 側)
    if case == 'tx_off':
        assert gone and all(r['hold'] and r['select'] and tuple(r['out']) == REF for r in gone)     # 中立
    else:
        assert gone and all(not r['select'] for r in gone)                                         # 人側
        assert back[0]['assist'] == A.ASSIST_LAUNCH and back[0]['out'][1] == N - 250.0              # 最初のフレームから
    assert back and all(r['select'] and not r['hold'] and r['out'][1] < N - 50 for r in back)       # AI の前進に戻る


@pytest.mark.parametrize('scene,conn', [('cruise', 'unplug'), ('cruise', 'never'), ('holds', 'unplug'), ('caps', 'always')])
def test_off_is_bit_identical_to_before(monkeypatch, tmp_path, scene, conn):
    """config を False にすると前の決まり: 最後の生存確認(一度も来ていなければ走行開始)から heartbeat_s(5 秒)を過ぎた最初の確かめで
    止まる(早くも遅くもない)。止めた理由の文・PCA9685 に書いた値・記録の行・出来事・処理したフレームは、直す前の写しとビット単位で同じ。
    つながったままなら最後まで走る(それも同じ)"""
    monkeypatch.setitem(config.DRIVE, 'tether_free', False)
    frames, imu, base = scene_of(scene)
    cut = {'unplug': lambda m: m <= 101.0, 'never': lambda m: False, 'always': lambda m: True}[conn]
    new = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=cut, base=base)
    old = run_tf(load_before(), monkeypatch, tmp_path, frames, imu, connected=cut, base=base)
    assert_same_run(new, old)
    d, _, _, u = new
    assert d.st['tether_free'] is False and 'tether_free' not in old[0].st and d.hb_count == len(u.hb_ms)
    if conn == 'always':
        assert d.st.get('stop_reason', '') == '' and len(u.ms) == len(frames)
        return
    last = u.hb_ms[-1] if u.hb_ms else 100.0
    assert d.st['stop_reason'] == (HB_NEVER if conn == 'never' else HB_TEXT)
    assert u.ms[-1] - last > 5.0 and u.ms[-2] - last <= 5.0                 # 5 秒を過ぎた最初の確かめ(早くも遅くもない)
    assert len(u.ms) < len(frames)


def test_on_stops_at_max_after_last_heartbeat(monkeypatch, tmp_path):
    """既定(ON): 最後の生存確認から tether_free_max_s(1800 秒)を過ぎた最初の確かめで止まる(早くも遅くもない)。理由の文に
    「USB を外して走る」と 30 分。抜いたあと 1 フレームごとに単調時計を 3 秒進める(センサーは 30 fps のまま)"""
    frames, imu = cruise(24.0)
    d, pca, rec, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 101.0,
                            jump=lambda i: 0.0 if i < 30 else 3.0)
    last = max(u.hb_ms)
    assert d.st['stop_reason'] == TF_TEXT
    assert u.ms[-1] - last > 1800.0 and u.ms[-2] - last <= 1800.0
    assert len(u.ms) < len(frames)
    assert [r for _, r in rec.rows if not r['hold']][-1]['out'][1] < N - 50     # 止まる直前まで AI が走っていた


def test_on_max_follows_the_config(monkeypatch, tmp_path):
    """上限は config.DRIVE['tether_free_max_s'](走行を始めたときの値)。60 秒にすると 60 秒を過ぎた最初の確かめで止まり、理由の文は
    「1 分」。status の上限も同じ値(画面の文の分)"""
    monkeypatch.setitem(config.DRIVE, 'tether_free_max_s', 60.0)
    frames, imu = cruise(24.0)
    d, _, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 101.0,
                        jump=lambda i: 0.0 if i < 30 else 1.0)
    last = max(u.hb_ms)
    assert d.st['stop_reason'] == TF_TEXT.replace('30 分', '1 分')
    assert u.ms[-1] - last > 60.0 and u.ms[-2] - last <= 60.0
    assert d.status()['tether_free_max_s'] == 60.0


def test_on_counts_from_start_when_no_heartbeat(monkeypatch, tmp_path):
    """既定(ON)で生存確認が一度も来ない(▶ の返事を受け取る前に USB を抜いた): 走行開始(start が heartbeat_t を始めた時刻にする)から
    1800 秒で止まる"""
    frames, imu = cruise(24.0)
    d, _, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: False, jump=lambda i: 3.0)
    assert d.hb_count == 0 and not u.hb_ms and d.st['stop_reason'] == TF_TEXT
    assert u.ms[-1] - 100.0 > 1800.0 and u.ms[-2] - 100.0 <= 1800.0


def test_on_reconnect_restarts_the_window(monkeypatch, tmp_path):
    """既定(ON)で途中に USB をつなぎ直した(生存確認がまた来た)ら、30 分はそこから数え直す(最初に抜いたときからではない)"""
    frames, imu = cruise(32.0)
    conn = lambda m: m <= 101.0 or 1000.0 <= m <= 1010.0                    # noqa: E731
    d, _, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=conn, jump=lambda i: 0.0 if i < 30 else 3.0)
    last = max(u.hb_ms)
    assert 1000.0 <= last <= 1010.0 and d.st['stop_reason'] == TF_TEXT
    assert u.ms[-1] - last > 1800.0 and u.ms[-2] - last <= 1800.0 and u.ms[-1] > 101.0 + 1800.0 + 800.0


def test_on_dry_run_same(monkeypatch, tmp_path):
    """dry run(モーターに出力しない。机の上で試す)でも同じ: 抜いても止まらず(出力は中立のまま)、上限で止まる"""
    frames, imu = cruise(10.0)
    d, pca, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 101.0, dry_run=True)
    assert d.st['dry_run'] is True and d.st['tether_free'] is True
    assert d.st.get('stop_reason', '') == '' and len(u.ms) == len(frames)
    assert pca.pairs and set(pca.pairs) == {REF}                           # 出力はずっと中立
    frames, imu = cruise(24.0)
    d, _, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 101.0,
                        jump=lambda i: 0.0 if i < 30 else 3.0, dry_run=True)
    assert d.st['stop_reason'] == TF_TEXT


@pytest.mark.parametrize('at_start', [True, False])
def test_config_change_mid_run_does_not_change_the_run(monkeypatch, tmp_path, at_start):
    """走行の途中で config を書き換えても、その走行は始めたときの値のまま(ON で始めた走行は止まらない・OFF で始めた走行は 5 秒で
    止まる)。status もその走行の値"""
    monkeypatch.setitem(config.DRIVE, 'tether_free', at_start)
    got = []

    def flip(i, d):
        if i == 15:                                                         # 0.5 秒(まだつながっている)
            monkeypatch.setitem(config.DRIVE, 'tether_free', not at_start)
        if i == 60:
            got.append(d.status()['tether_free'])
    frames, imu = cruise(12.0)
    d, _, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 101.0, hook=flip)
    assert got == [at_start] and d.st['tether_free'] is at_start and config.DRIVE['tether_free'] is (not at_start)
    if at_start:
        assert d.st.get('stop_reason', '') == '' and len(u.ms) == len(frames)
    else:
        assert d.st['stop_reason'] == HB_TEXT and len(u.ms) < len(frames)


class Alive(object):
    def is_alive(self):
        return True


def test_stop_without_token_stops_the_run(monkeypatch, tmp_path):
    """既定(ON)の走行も、token なしの停止(POST /api/drive/stop の body が空 = USB をつなぎ直した画面・別の画面の ■)で止まる。
    走行ループは押した次の確かめで抜ける(理由は「停止ボタン」)"""
    frames, imu = cruise(20.0)
    got = []

    def press(i, d):
        if i == 400:                                                        # USB を抜いて約 12 秒後
            monkeypatch.setattr(srv, 'driver', d)
            d.thread = Alive()                                              # 走行のスレッドが生きている(この試験はスレッドを使わない)
            got.append(srv.api_drive_stop({}))
    d, _, _, u = run_tf(drive, monkeypatch, tmp_path, frames, imu, connected=lambda m: m <= 101.0, hook=press)
    assert got == [{'ok': True}] and d.st['stop_reason'] == '停止ボタン' and len(u.ms) == 401
    assert max(u.hb_ms) <= 101.0 and u.ms[-1] - max(u.hb_ms) > 5.0          # 抜いたまま(5 秒を過ぎても走っていた)


# ---------------------------------------------------------------- サーバー(走行開始・状態)
@pytest.fixture
def server_env(monkeypatch, tmp_path):
    """server の走行の API を、偽の driver(本物の Driver。走行のスレッドは回さない・音声の通知先なし)で呼ぶ"""
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    runs = tmp_path / 'runs'
    _run_dir(runs, 'm1')
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    gate = threading.Event()
    d._run = lambda *a: gate.wait(5)                                        # 走行のスレッドの中身は回さない(止めるまで生きている)
    calls = []
    real = d.start

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    d.start = spy
    monkeypatch.setattr(srv, 'driver', d)
    monkeypatch.setattr(srv, 'vbat', vs.VbatTracker())                     # 電圧の列は空(説明欄に書かない)
    monkeypatch.setattr(srv, 'VBAT_RUN', {})
    monkeypatch.setattr(bench_web, 'AI_NO_PATH', tmp_path / 'ai_run_no.json')
    monkeypatch.setattr(bench_web, 'BENCH_PATH', tmp_path / 'no_bench.json')
    monkeypatch.setattr(srv.trainer.__class__, 'running', property(lambda self: False))
    yield d, calls, gate
    gate.set()
    if d.thread is not None:
        d.thread.join(5)


def _body(d, **kw):
    b = {'client': srv.DRIVE_CLIENT_VERSION, 'boot_at': d.boot_at, 'run': 'm1', 'mode_id': 4, 'description': 'AI走行 m1',
         'dry_run': False, 'record': False, 'params': {}}
    b.update(kw)
    return b


def _finish(d, gate):
    gate.set()
    d.thread.join(5)
    gate.clear()


def test_server_ignores_body_and_status_shows_the_run(server_env, monkeypatch):
    """走行開始の body の tether_free は読まない(true・false・文字・1 でも、無くても同じ): driver.start は直す前と同じ引数で呼ばれ、
    その走行は config の値(既定 ON)。status(token 付きの生存確認でも、無い表示用でも)に tether_free と上限 1800。走り終わっても
    status は最後の走行の値。config を False にすると、body に true があっても OFF。dry run でも同じ"""
    d, calls, gate = server_env
    assert srv.DRIVE_CLIENT_VERSION == UI_VER                                      # 2026-11-01 電圧の前送りの版なら 11
    for extra in ({}, {'tether_free': True}, {'tether_free': False}, {'tether_free': 'true'}, {'tether_free': 1},
                  {'dry_run': True}, {'dry_run': True, 'tether_free': False}):
        r = srv.api_drive_start(_body(d, **extra))
        assert set(r) == {'ok', 'run', 'token', 'ai_no', 'plan_diffs', 'v_before'} and r['ok'] is True, extra   # 返事も前と同じ
        assert set(calls[-1]) == START_KW and d.st['tether_free'] is True, extra
        st = srv.api_drive_status(token=r['token'])
        assert st['running'] is True and st['controller'] is True and st['tether_free'] is True and st['tether_free_max_s'] == 1800
        assert srv.api_drive_status(token='')['tether_free'] is True
        _finish(d, gate)
        assert srv.api_drive_status(token='')['tether_free'] is True and srv.api_drive_status(token='')['running'] is False
    assert 'tether_free' not in json.loads(drive.SETTINGS_PATH.read_text(encoding='utf-8'))['params']
    monkeypatch.setitem(config.DRIVE, 'tether_free', False)
    r = srv.api_drive_start(_body(d, tether_free=True))
    assert d.st['tether_free'] is False and srv.api_drive_status(token=r['token'])['tether_free'] is False
    _finish(d, gate)


def test_status_at_boot_is_the_config(monkeypatch, tmp_path):
    """起動したばかり(まだ走っていない)の status は config の値(= 次の走行の値)。上限の値も出す(画面の「30 分」)"""
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    st = drive.Driver(hub=None, recorder=FakeRecorder()).status()
    assert st['tether_free'] is True and st['tether_free_max_s'] == 1800 and st['running'] is False
    monkeypatch.setitem(config.DRIVE, 'tether_free', False)
    assert drive.Driver(hub=None, recorder=FakeRecorder()).status()['tether_free'] is False


def test_server_start_unchanged_for_old_fakes(server_env, monkeypatch):
    """server の走行開始は直す前とまったく同じ(AST)。tether_free を知らない偽物の start(ほかの試験の偽物と同じ形)でも動く"""
    d, _, gate = server_env
    fn = lambda src: next(ast.dump(n) for n in ast.parse(src).body if getattr(n, 'name', '') == 'api_drive_start')   # noqa: E731
    assert fn(FB.back(ROOT / 'jetformer' / 'web' / 'server.py').read_text(encoding='utf-8')) == fn((FIX / 'server.py').read_text(encoding='utf-8'))
    seen = []

    def old_start(run, mode_id, record=False, description='', tags=None, dry_run=False, params=None):
        seen.append(run)
        return run, 'tok'
    monkeypatch.setattr(d, 'start', old_start)
    assert srv.api_drive_start(_body(d, tether_free=True))['ok'] is True and seen == ['m1']


def test_server_refuses_old_page(server_env):
    """版 10 より古い画面(版 9 = 「通信断が5秒続くと止まります」と出す画面・キャッシュ)からは走行を始めない(409。driver.start を
    呼ばない)"""
    d, calls, gate = server_env
    for client in (9, 8, 0, None, 'x'):
        body = _body(d)
        body['client'] = client
        with pytest.raises(HTTPException) as e:
            srv.api_drive_start(body)
        assert e.value.status_code == 409 and '管理画面が古いまま' in e.value.detail and '版 %d から' % UI_VER in e.value.detail, client
    assert calls == [] and d.thread is None


# ---------------------------------------------------------------- 走行の後始末(os.sync)
class _Sensor(object):
    def __init__(self):
        self.subs = []

    def subscribe(self, cb):
        self.subs.append(cb)

    def unsubscribe(self, cb):
        self.subs.remove(cb)


class FakeHub(object):
    def __init__(self):
        self.s = {ch: _Sensor() for ch in ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE')}

    def start(self):
        pass

    def __getitem__(self, ch):
        return self.s[ch]


class RunRecorder(object):
    """収集の偽物。order に start・stop の順を残す。start_fail = すでに人が収集している"""
    recording = False

    def __init__(self, order, start_fail=False):
        self.order, self.start_fail, self.drive = order, start_fail, None

    def start(self, description='', tags=None, mode_id=None, drive=None):
        if self.start_fail:
            raise RuntimeError('すでに収集中です')
        self.drive, self.recording = drive, True
        self.order.append('rec_start')
        return 'LOG1'

    def stop(self, only_log=None):
        assert only_log == 'LOG1'
        self.order.append('rec_stop')
        self.recording = False
        return True

    def add_event(self, kind, data):
        return True

    def add_ai(self, t_ns, data):
        pass


def fake_runtime(monkeypatch, order):
    """走行の後始末の試験: torch・PCA9685・TensorRT・入力の正規化の模型を sys.modules に入れる(本物は読まない・触れない)"""
    torch = types.ModuleType('torch')
    torch.zeros = lambda *a, **k: None
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: order.append('gpu_free'))
    monkeypatch.setitem(sys.modules, 'torch', torch)

    class PCA(object):
        def __init__(self, cfg, scale):
            pass

        def set_pair_us(self, sc_, s, tc, t):
            order.append(('pca', float(s), float(t)))

        def set_us(self, ch, us):
            order.append(('pca1', ch, float(us)))

        def set_osc_scale(self, scale):
            pass

        def close(self):
            order.append('pca_close')
    act = types.ModuleType('jetformer.actuator')
    act.PCA9685 = PCA
    monkeypatch.setitem(sys.modules, 'jetformer.actuator', act)
    td = types.ModuleType('jetformer.train_data')
    td.normalize_inputs = fake_normalize
    monkeypatch.setitem(sys.modules, 'jetformer.train_data', td)

    class Runner(object):
        def __init__(self, path):
            self.inputs, self.outputs = ['image', 'tof', 'imu', 'state', 'mode'], ['actions']
            self.shapes = {'image': (1, 3, 224, 224), 'tof': (1, 2, 42, 54), 'imu': (1, 10, 6), 'state': (1, 3), 'mode': (1, 6),
                           'actions': (1, 10, 2)}

        def run_numpy(self, **kw):
            return {'actions': np.zeros((1, 10, 2), np.float32)}
    trt = types.ModuleType('jetformer.trt_runtime')
    trt.TRTRunner = Runner
    monkeypatch.setitem(sys.modules, 'jetformer.trt_runtime', trt)


def run_full(monkeypatch, tmp_path, record=True, sync_fail=False, start_fail=False, rec=None, loop=None, on_sync=None):
    """Driver.start → _run(本物)を模型で回す。走行ループは偽物(loop。既定はすぐ止める)→ (driver, order, recorder)。
    drive の os.sync は偽物(order に 'sync'。on_sync() も呼ぶ)。rec = 収集を作る関数 rec(order)(既定は偽物の RunRecorder)"""
    order = []
    fake_runtime(monkeypatch, order)

    def fake_sync():
        order.append('sync')
        if on_sync is not None:
            on_sync()
        if sync_fail:
            raise OSError(5, 'Input/output error')
    monkeypatch.setattr(drive, 'os', OsProxy(fake_sync))
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    runs = tmp_path / 'runs'
    if not (runs / 'm1').exists():
        _run_dir(runs, 'm1')
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    rec = rec(order) if rec is not None else RunRecorder(order, start_fail=start_fail)
    d = drive.Driver(hub=FakeHub() if not hasattr(rec, 'hub') else rec.hub, recorder=rec)
    d.listeners.append(lambda ev, **kw: order.append(('emit', ev)))

    def quick(st, *a):
        order.append('loop')
        st.update(phase='driving', announced=True, stats={'frames': 1})
        st['stop_reason'] = '停止ボタン'
    d._loop = loop or quick
    d.start('m1', 4, record=record, description='x')
    d.thread.join(10)
    assert not d.running
    return d, order, rec


def test_sync_once_after_recorder_stop(monkeypatch, tmp_path):
    """収集した走行: 後始末で、中立(いちばん先)→ PCA9685 を閉じる → recorder.stop → os.sync を 1 回 → 止まった知らせ →
    GPU を返す。画面の文は今まで通り「収集したシーンを保存しました」"""
    d, order, rec = run_full(monkeypatch, tmp_path)
    names = [x if isinstance(x, str) else x[0] for x in order]
    assert names.count('sync') == 1
    i_loop, i_stop, i_sync = names.index('loop'), names.index('rec_stop'), names.index('sync')
    after = names[i_loop + 1:]
    assert after[0] == 'pca1' and after[1] == 'pca1' and order[i_loop + 1][1] == config.PCA9685['throttle_ch']   # 中立がいちばん先
    assert names.index('pca_close') < i_stop < i_sync < order.index(('emit', 'drive_stop')) < names.index('gpu_free')
    assert d.st['message'] == '停止しました(停止ボタン)・収集したシーンを保存しました' and d.st['phase'] == 'idle'
    assert d.st['error'] is None and d.st.get('finished')


def test_sync_failure_does_not_stop_cleanup(monkeypatch, tmp_path):
    """os.sync が失敗しても(例外)、後始末は続く: 止まった知らせ・GPU を返す・終わりの時刻。エラーにはしない"""
    d, order, _ = run_full(monkeypatch, tmp_path, sync_fail=True)
    names = [x if isinstance(x, str) else x[0] for x in order]
    assert names.count('sync') == 1 and ('emit', 'drive_stop') in order and 'gpu_free' in names
    assert names.index('sync') < order.index(('emit', 'drive_stop')) < names.index('gpu_free')
    assert d.st.get('finished') and d.st['phase'] == 'idle' and d.st['error'] is None
    assert d.st['message'].endswith('・収集したシーンを保存しました')


@pytest.mark.parametrize('record,start_fail', [(False, False), (True, True)])
def test_no_sync_without_own_recording(monkeypatch, tmp_path, record, start_fail):
    """収集をしていない走行(収集なし・人が始めた収集の途中 = この走行は収集を止めない)では os.sync を呼ばない"""
    d, order, _ = run_full(monkeypatch, tmp_path, record=record, start_fail=start_fail)
    assert 'sync' not in order and 'rec_stop' not in order
    assert ('emit', 'drive_stop') in order and d.st.get('finished')


def test_tether_free_recorded_in_log(monkeypatch, tmp_path):
    """記録: 既定(ON)の走行は log.json の drive.tether_free = True、config を False にした走行は False。status も同じ"""
    d, _, rec = run_full(monkeypatch, tmp_path)
    assert rec.drive['tether_free'] is True and d.st['tether_free'] is True and d.status()['tether_free'] is True
    monkeypatch.setitem(config.DRIVE, 'tether_free', False)
    d2, _, rec2 = run_full(monkeypatch, tmp_path)
    assert rec2.drive['tether_free'] is False and d2.st['tether_free'] is False and d2.status()['tether_free'] is False


# ---------------------------------------------------------------- 画面(app.js・index.html)
def _src(p):
    return Path(p).read_text(encoding='utf-8')


def test_static_no_switch_and_version():
    """画面に「USB を外して走る」のスイッチは無い(チェックの欄・送る物なし)。▶ のすぐ上に 1 行(#dr-tether-note。初めは今までの文 =
    サーバーを読むまで)。画面の版 10(サーバーと同じ)。走行開始を呼ぶのは ▶ だけ"""
    html = _src(STATIC / 'index.html')
    assert 'dr-tether-free' not in html and 'type="checkbox" id="dr-tether' not in html
    line = '<div id="dr-tether-note" class="muted small">%s</div>' % OLD_LAST
    assert html.count('id="dr-tether-note"') == 1 and line in html
    assert html.index(line) < html.index('id="btn-dr-start"') and html.index('id="dr-dry"') < html.index(line)
    nxt = html[html.index(line) + len(line):].lstrip()
    assert nxt.startswith('<div class="row buttons"><button id="btn-dr-start"')             # ▶ のすぐ上
    js = _src(STATIC / 'app.js')
    assert 'dr-tether-free' not in js and 'body.tether_free' not in js and not re.search(r'\btether_free\s*:', js)
    jsv = _src(FB.back(STATIC / 'app.js'))                     # 版の数は、2026-11-01 電圧の前送りの版なら直す前の写し(= この直しの版)で
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', jsv, re.M) == ['10'] and re.findall(r'const body = \{ client: (\d+),', jsv) == ['10']
    assert js.count("'/api/drive/start'") == 1
    assert re.findall(r'^DRIVE_CLIENT_VERSION = (\d+)$', _src(FB.back(ROOT / 'jetformer' / 'web' / 'server.py')), re.M) == ['10']
    r = subprocess.run([NODE, '--check', str(STATIC / 'app.js')], capture_output=True, text=True) if NODE else None
    assert r is None or r.returncode == 0, r.stderr


def _extract(js, name):
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError('app.js に %s がありません' % name)


@needs_node
def test_js_texts_follow_the_status():
    """文はサーバーの status で決まる: tether_free が true(JSON の true だけ)で上限 tether_free_max_s があるときだけ新しい文
    (確認の文の最後の行・▶ の上の 1 行。分はサーバーの値から)。false・知らないサーバー(上限が無い)・文字の "true" は今までの文"""
    js = _src(STATIC / 'app.js')
    code = ''.join(_extract(js, n) for n in ('tetherOn', 'tetherMaxText', 'HB_LINE', 'startLastLine', 'tetherNoteText'))
    cases = [{'tether_free': True, 'tether_free_max_s': 1800}, {'tether_free': True, 'tether_free_max_s': 1800.0},
             {'tether_free': True, 'tether_free_max_s': 600}, {'tether_free': True, 'tether_free_max_s': 90},
             {'tether_free': False, 'tether_free_max_s': 1800}, {'tether_free': True}, {'tether_free': True, 'tether_free_max_s': None},
             {'tether_free': True, 'tether_free_max_s': 0}, {'tether_free': 'true', 'tether_free_max_s': 1800}, {}, None]
    code += 'process.stdout.write(JSON.stringify(%s.map((s) => [tetherOn(s), startLastLine(s), tetherNoteText(s)])));' % json.dumps(cases)
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert [g[0] for g in got] == [True, True, True, True] + [False] * 7
    assert got[0][1:] == [NEW_LAST, NOTE_ON] and got[1][1:] == [NEW_LAST, NOTE_ON]
    assert got[2][1:] == [NEW_LAST.replace('30 分', '10 分'), NOTE_ON.replace('30 分', '10 分')]
    assert got[3][1:] == [NEW_LAST.replace('30 分', '90 秒'), NOTE_ON.replace('30 分', '90 秒')]
    assert all(g[1:] == [OLD_LAST, OLD_LAST] for g in got[4:])


def sim_js(tmp_path):
    """tests/fixtures/app_sim_fix8.js の写し(もとは変えない)。偽のサーバーの status に入力の server_tether(tether_free・
    tether_free_max_s など。無ければ足さない = 知らないサーバー)を足す"""
    src = SIM.read_text(encoding='utf-8')
    anchor = '  if (bootAt !== null) st.boot_at = bootAt;\n'
    assert src.count(anchor) == 1
    p = tmp_path / 'app_sim_tether.js'
    p.write_text(src.replace(anchor, anchor + '  if (inp.server_tether) Object.assign(st, inp.server_tether);\n'), encoding='utf-8')
    return p


def sim(tmp_path, steps, tether=None, app=None, name='in.json'):
    """本物の app.js を node で読む(偽のサーバー・DOM)。app = 直す前の写しのときは、その index.html で読む"""
    c = bench_web.load_candidates()
    inp = tmp_path / name
    d = {'bench': c, 'server_params': dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1),
         'steps': steps, 'server_client_version': 9, 'confirm': True, 'boot_at': 1000.0, 'models': [c['baseline']['run']],
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True, 'pages': ['A'], 'server_tether': tether}
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    html = FIX / 'index.html' if app is not None else STATIC / 'index.html'
    r = subprocess.run([NODE, str(sim_js(tmp_path)), str(app or STATIC / 'app.js'), str(html), str(inp)],
                       capture_output=True, text=True, timeout=180, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


ON = {'tether_free': True, 'tether_free_max_s': 1800.0}                    # 新しいサーバー(既定)の status
PRE = [['tab', 'drive'], ['tick'], ['tick']]
LOOK = [['el', '#dr-tether-note', 'text'], ['el', '#dr-message', 'html']]


def _starts(res):
    return [p for x in res['steps'] if x['op'] == 'posted' for p in x['value'] if p[1] == '/api/drive/start']


def _els(res):
    return [x['value'] for x in res['steps'] if x['op'] == 'el']


def _no_client(b):
    return {k: v for k, v in b.items() if k != 'client'}


@needs_node
def test_page_on_server_texts_and_same_sending(tmp_path):
    """新しいサーバー(status の tether_free = true・上限 1800): ▶ の確認の文の最後の行が新しい文で、そのほかの行は直す前の app.js と
    同じ。送った body は版のほか直す前と同じ(tether_free は無い)。トーストも同じ。▶ の上の 1 行は「USB を外して走れます…」。
    走行中の状態の欄に短い印、止まると消える"""
    steps = PRE + LOOK + [['start'], ['posted'], ['tick']] + LOOK + [['stop'], ['tick']] + LOOK + [['posted']]
    new, old = sim(tmp_path, steps, ON), sim(tmp_path, steps, ON, app=FIX / 'app.js', name='o.json')
    (sn,), (so,) = [x for x in new['steps'] if x['op'] == 'start'], [x for x in old['steps'] if x['op'] == 'start']
    assert sn['status'] == so['status'] == 200
    assert sn['confirm'].split('\n')[-1] == NEW_LAST and OLD_LAST not in sn['confirm'] and so['confirm'].split('\n')[-1] == OLD_LAST
    assert sn['confirm'].rsplit('\n', 1)[0] == so['confirm'].rsplit('\n', 1)[0]
    (pn,), (po,) = _starts(new), _starts(old)
    assert pn[2]['client'] == UI_VER_JS and po[2]['client'] == 9 and 'tether_free' not in pn[2] and _no_client(pn[2]) == _no_client(po[2])
    assert new['toasts'] == old['toasts']
    note0, msg0, note1, msg1, note2, msg2 = _els(new)
    assert note0 == note1 == note2 == NOTE_ON
    assert MARK not in msg0 and MARK in msg1 and MARK not in msg2
    stops = [p for x in new['steps'] if x['op'] == 'posted' for p in x['value'] if p[1] == '/api/drive/stop']
    assert len(stops) == 1 and stops[0][2] == {'reason': '停止ボタン'}


@needs_node
@pytest.mark.parametrize('tether', [None, {'tether_free': False, 'tether_free_max_s': 1800.0}, {'tether_free': True},
                                    {'tether_free': 'true', 'tether_free_max_s': 1800.0}])
def test_page_off_or_unknown_server_is_same_as_before(tmp_path, tether):
    """知らないサーバー(tether_free・上限が無い = 古いまま)・config を False にしたサーバー・上限の無い・文字の "true": 確認の文・送った
    body(版のほか)・トーストは直す前の app.js とまったく同じ(最後の行は今までの「5秒続くと止まります」)。▶ の上の 1 行も今までの文・
    走行中の印は出ない"""
    steps = PRE + LOOK + [['start'], ['posted'], ['tick']] + LOOK
    new, old = sim(tmp_path, steps, tether), sim(tmp_path, steps, tether, app=FIX / 'app.js', name='o.json')
    (sn,), (so,) = [x for x in new['steps'] if x['op'] == 'start'], [x for x in old['steps'] if x['op'] == 'start']
    assert sn['confirm'] == so['confirm'] and sn['confirm'].split('\n')[-1] == OLD_LAST and sn['status'] == 200
    (pn,), (po,) = _starts(new), _starts(old)
    assert pn[2]['client'] == UI_VER_JS and po[2]['client'] == 9 and _no_client(pn[2]) == _no_client(po[2])
    assert new['toasts'] == old['toasts']
    note0, msg0, note1, msg1 = _els(new)
    assert note0 == note1 == OLD_LAST and 'USB' not in msg0 + msg1


@needs_node
def test_page_dry_run_on_server(tmp_path):
    """dry run(モーターに出力しない)+ 新しいサーバー: 確認の文は出さない(dry run は今まで通り)。送った body は版のほか直す前と同じ・
    トーストも同じ(「推論の確認を開始しました(出力は中立)」)。▶ の上の 1 行と走行中の印で読む"""
    steps = PRE + [['check', '#dr-dry', True], ['start'], ['posted'], ['tick']] + LOOK
    new, old = sim(tmp_path, steps, ON), sim(tmp_path, steps, ON, app=FIX / 'app.js', name='o.json')
    (sn,) = [x for x in new['steps'] if x['op'] == 'start']
    assert sn['confirm'] == '' and sn['status'] == 200
    (pn,), (po,) = _starts(new), _starts(old)
    assert pn[2]['dry_run'] is True and _no_client(pn[2]) == _no_client(po[2]) and 'tether_free' not in pn[2]
    assert new['toasts'] == old['toasts'] and any(t == '推論の確認を開始しました(出力は中立)' for t in new['toasts']['A'])
    note, msg = _els(new)
    assert note == NOTE_ON and MARK in msg


# ---------------------------------------------------------------- 直す前の写しから変えた所
def _tree(path):
    return ast.parse(Path(path).read_text(encoding='utf-8'))


def _defs(path):
    """モジュールの一番外側の文(クラスはメソッドとクラスの変数ごと)を名前 → AST の文字にする。名前の無い文は中身で数える
    (前に関数を足しても、位置がずれただけで変わったことにならない)"""
    out = {}
    for i, node in enumerate(_tree(path).body):
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out['def:%s.%s' % (node.name, m.name)] = ast.dump(m)
                elif isinstance(m, ast.Assign) and len(m.targets) == 1 and isinstance(m.targets[0], ast.Name):
                    out['set:%s.%s' % (node.name, m.targets[0].id)] = ast.dump(m)
                else:
                    out['#%s.%s' % (node.name, hashlib.sha1(ast.dump(m).encode()).hexdigest()[:12])] = ast.dump(m)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out['def:' + node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out['set:' + node.targets[0].id] = ast.dump(node)
        elif i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, 'value', None), ast.Constant):
            out['doc'] = ast.dump(node)
        else:
            out['#' + hashlib.sha1(ast.dump(node).encode()).hexdigest()[:12]] = ast.dump(node)
    return out


def _changed(before, after):
    a, b = _defs(before), _defs(after)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _fn(path, qual):
    """'Class.method' か 'func' の関数の AST"""
    cls, _, name = qual.rpartition('.')
    for node in _tree(path).body:
        if not cls and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if cls and isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == name:
                    return m
    raise AssertionError('%s に %s がありません' % (path, qual))


def _edit_stmts(fn, pred, repl=None):
    """fn の中の文の並び(どの深さでも)で pred(文) が真の文を、repl(文) の文の列に替える(repl が無ければ除く)。替えた数"""
    n = 0
    for node in list(ast.walk(fn)):
        for field in ('body', 'orelse', 'finalbody'):
            seq = getattr(node, field, None)
            if not isinstance(seq, list):
                continue
            out = []
            for s in seq:
                if isinstance(s, ast.stmt) and pred(s):
                    n += 1
                    out.extend(repl(s) if repl else [])
                else:
                    out.append(s)
            seq[:] = out
    return n


def _drop_dict_key(fn, key):
    """fn の中の dict の表記から key の項目を除く。除いた数"""
    n = 0
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Dict):
            for i, k in enumerate(list(sub.keys)):
                if isinstance(k, ast.Constant) and k.value == key:
                    del sub.keys[i]
                    del sub.values[i]
                    n += 1
                    break
    return n


def _src_of(s):
    return ast.unparse(s)


def _no_doc(path, name):
    """関数 name の AST(説明の文を除く)"""
    fn = _fn(path, name)
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]
    return ast.dump(fn)


def test_python_changed_only_these():
    """drive.py: 変えたのは説明の文・Driver.start・_run・_loop・status、足したのは tether_free_stop_text だけ。変えた所を除けば直す前と
    AST が同じ: start は st の 1 項目(tether_free)、_run は記録の 1 項目と os.sync の try 1 つ、status は 2 行、_loop は 2 行と、
    生存確認の if の前に足した分かれ(OFF の分かれ = 前の if そのまま)。server.py: 説明の文(api_drive_status)と DRIVE_CLIENT_VERSION
    (9 → 10)だけ(走行開始は同じ)。config.py: DRIVE に tether_free(True)と tether_free_max_s(1800)・RECORD_SYNC_S(2.0。
    定期的な書き込み)を足しただけ(ほかの値は同じ)"""
    load_before()
    now, old = FB.back(ROOT / 'jetformer' / 'drive.py'), FIX / 'drive.py'       # 2026-11-01 電圧の前送りの版なら直す前の写し(= この直しの版)
    assert _changed(old, now) == ['def:Driver._loop', 'def:Driver._run', 'def:Driver.start', 'def:Driver.status',
                                  'def:tether_free_stop_text', 'doc']
    assert inspect.getdoc(FB.back_module(drive).Driver.start) == inspect.getdoc(load_before().Driver.start)
    fn = _fn(now, 'Driver.start')
    assert _drop_dict_key(fn, 'tether_free') == 1 and ast.dump(fn) == ast.dump(_fn(old, 'Driver.start'))
    fn = _fn(now, 'Driver._run')
    assert _drop_dict_key(fn, 'tether_free') == 1
    assert _edit_stmts(fn, lambda s: isinstance(s, ast.Try) and _src_of(s.body[0]) == 'os.sync()') == 1
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver._run'))
    fn = _fn(now, 'Driver.status')
    assert _edit_stmts(fn, lambda s: isinstance(s, ast.Assign) and _src_of(s.targets[0]) in
                       ("st['tether_free']", "st['tether_free_max_s']")) == 2
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver.status'))
    fn = _fn(now, 'Driver._loop')
    assert _edit_stmts(fn, lambda s: isinstance(s, ast.Assign) and _src_of(s.targets[0]) in ('tether_free', 'tf_max')) == 2
    assert _edit_stmts(fn, lambda s: isinstance(s, ast.If) and _src_of(s.test) == 'tether_free', lambda s: s.orelse) == 1
    assert ast.dump(fn) == ast.dump(_fn(old, 'Driver._loop'))
    srv_now, srv_old = FB.back(ROOT / 'jetformer' / 'web' / 'server.py'), FIX / 'server.py'
    assert _changed(srv_old, srv_now) == ['def:api_drive_status', 'set:DRIVE_CLIENT_VERSION']
    assert _no_doc(srv_old, 'api_drive_status') == _no_doc(srv_now, 'api_drive_status')
    ns = {'__file__': str(ROOT / 'jetformer' / 'config.py'), '__name__': 'jetformer._config_before_tether'}
    exec(compile((FIX / 'config.py').read_text(encoding='utf-8'), 'config_before.py', 'exec'), ns)
    assert {k: v for k, v in config.DRIVE.items() if k not in ('tether_free', 'tether_free_max_s')} == ns['DRIVE']
    assert set(config.DRIVE) - set(ns['DRIVE']) == {'tether_free', 'tether_free_max_s'}
    assert {k: getattr(config, k) for k in ('CAMERA', 'TOF', 'IMU', 'VEHICLE', 'PCA9685', 'CALIBRATION', 'SCENE_SECONDS')} == \
        {k: ns[k] for k in ('CAMERA', 'TOF', 'IMU', 'VEHICLE', 'PCA9685', 'CALIBRATION', 'SCENE_SECONDS')}
    assert _changed(FIX / 'config.py', ROOT / 'jetformer' / 'config.py') == ['set:DRIVE', 'set:RECORD_SYNC_S']
    assert 'RECORD_SYNC_S' not in ns


# app.js で足した行(説明の行を除く。行の終わりの注も除く)。上から、版の数・▶ の確認の文と body の版の 3 行(前の行を替えた)、
# 文を作る関数(▶ の上の 1 行・確認の文の最後の行)、走行中の印、▶ の上の 1 行の表示
APP_ADDED = [
    'const DRIVE_CLIENT = 10;',
    "const tetherOn = (st) => !!(st && st.tether_free === true && Number.isFinite(+st.tether_free_max_s) && +st.tether_free_max_s > 0);",
    'function tetherMaxText(st) {',
    '  const s = +st.tether_free_max_s;',
    '  return s >= 60 && s % 60 === 0 ? `${s / 60} 分` : `${Math.round(s)} 秒`;',
    '}',
    "const HB_LINE = 'この画面は開いたままにしてください(閉じる・スリープ・通信断が5秒続くと止まります)。';",
    'function startLastLine(st) {',
    '  return tetherOn(st) ? `USB を外しても AI は止まりません(画面との通信が ${tetherMaxText(st)}途絶えたら止まります)。`',
    "    + '止めるときは、CH3 を人側にしてから、USB をつなぎ直して ■ 停止を押してください。' : HB_LINE;",
    '}',
    'function tetherNoteText(st) {',
    '  return tetherOn(st) ? `USB を外して走れます(画面との通信が切れても AI は止まりません・${tetherMaxText(st)}途絶えたら止まります)。`',
    "    + '止めるときは CH3 を人側 → USB をつなぎ直して ■ 停止' : HB_LINE;",
    '}',
    "    + (st.running && tetherOn(st) ? `<div class=\"muted\">USB を外しても止まりません(画面との通信が ${esc(tetherMaxText(st))}途絶えたら止まります)</div>` : '')",
    "  const tn = $('#dr-tether-note');",
    '  if (tn) tn.textContent = tetherNoteText(st);',
]


def _code_lines(text):
    """説明だけの行を除き、行の終わりの注(2 つ以上の空白のあとの //)を除いた行"""
    out = []
    for x in text.split('\n'):
        if x.strip().startswith('//'):
            continue
        out.append(re.sub(r'\s{2,}// .*$', '', x).rstrip())
    return out


def test_page_changed_only_these():
    """index.html は ▶ のすぐ上に 1 行を足しただけ。app.js(説明の行を除く)で前の行を替えたのは版の数・▶ の確認の文の最後の行・body の
    版の 3 行だけ、足したのは上の APP_ADDED の行(文を作る関数・走行中の印・▶ の上の 1 行の表示)だけ。足した行に送る物・問い合わせ・
    タイマー・確認・トーストは無い"""
    a = _src(FIX / 'index.html')
    b = _src(FB.back(STATIC / 'index.html'))                         # 2026-11-01 電圧の前送りの版なら直す前の写し(= この直しの版)
    line = '          <div id="dr-tether-note" class="muted small">%s</div>\n' % OLD_LAST
    assert b.count(line) == 1 and b.replace(line, '') == a
    old, new = _code_lines(_src(FIX / 'app.js')), _code_lines(_src(FB.back(STATIC / 'app.js')))
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    removed = [x for o in sm.get_opcodes() if o[0] in ('replace', 'delete') for x in old[o[1]:o[2]]]
    added = [x for o in sm.get_opcodes() if o[0] in ('replace', 'insert') for x in new[o[3]:o[4]]]
    confirm_old = [x for x in old if x.startswith('  if (!dry && !confirm(`AI走行を開始します。')]
    body_old = [x for x in old if x.startswith('  const body = { client: 9, build: PAGE_BUILD,')]
    assert len(confirm_old) == 1 and len(body_old) == 1
    assert collections.Counter(removed) == collections.Counter(['const DRIVE_CLIENT = 9;'] + confirm_old + body_old), removed
    tail = '\\nこの画面は開いたままにしてください(閉じる・スリープ・通信断が5秒続くと止まります)。`)) return;'
    assert confirm_old[0].endswith(tail)
    confirm_new = confirm_old[0][:-len(tail)] + '\\n${startLastLine(state.drive.st)}`)) return;'
    body_new = body_old[0].replace('client: 9,', 'client: 10,', 1)
    assert collections.Counter(added) == collections.Counter(APP_ADDED + [confirm_new, body_new]), added
    assert not any(re.search(r'setTimeout|setInterval|fetch\(|api\(|confirm\(|toast\(|body\.|\.checked|sessionStorage', x)
                   for x in APP_ADDED)


def test_repo_readme_matches_the_rules():
    """リポジトリの README.md(AI走行の章)が今の決まりと同じ: 「走行を始めた画面は開いたままにする」「画面が5秒途絶えたら止める」を
    既定として書かない。既定 ON(config の tether_free)・上限 30 分(tether_free_max_s)・False なら 5 秒(heartbeat_s)・USB を抜いて
    いる間は CH3 を人側 → USB をつなぎ直して停止・プロポの信号が届かないときの中立は一時的(走行は終わらない。音声の表も)・収集中の
    2 秒ごとの書き込み(RECORD_SYNC_S)。数は config の値から作る(2026-10-31 粗探しの指摘: 画面の文は直したが README が前のままだった)"""
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    a = text.index('## AI走行(管理画面の「走行」タブ)')
    ch = text[a:text.index('\n## ', a + 1)]
    assert '走行を始めた画面は開いたままにする' not in ch and '走行を始めた画面が5秒途絶えた' not in ch
    mins, hb, every = ('%d分' % (config.DRIVE['tether_free_max_s'] // 60), '%g秒' % config.DRIVE['heartbeat_s'],
                       '%g秒ごと' % config.RECORD_SYNC_S)
    assert (config.DRIVE['tether_free'], mins, hb, every) == (True, '30分', '5秒', '2秒ごと')
    for s in ("既定(`config.DRIVE['tether_free'] = True`。", '画面との通信が切れてもAIは止まりません',
              'その画面からの問い合わせが%s(`tether_free_max_s`)' % mins, '問い合わせが%s途絶えたら止める' % hb,
              '先にCH3を人側にしてから、USBをつなぎ直して停止ボタン',
              '走行を始めた画面との通信が%s途絶えた(`tether_free` が `False` なら%s)' % (mins, hb),
              '中立は一時的で、**走行は終わりません**(信号が戻ると、CH3がAI側ならAIがまた走らせる。そのとき案内の声は無い)',
              '送信機の電源を入れる前にCH3を人側に',
              '収集している間は%sに、書いたファイルをディスクへ書き切る(`config.RECORD_SYNC_S`。0で止める。' % every):
        assert ch.count(s) == 1, s
    assert text.count('「プロポの信号が届きません。停止します」(AI側のまま送信機の信号が途絶えたとき。中立にするだけで、走行は'
                      '終わっていない)') == 1
