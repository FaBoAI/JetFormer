"""収集中の定期的な書き込み(recorder の rec-sync。2026-10-31)の試験。

ユーザー(10/31 10:3x)「データは定期的に書き込みがいいですね」。収集している間だけ、書いたファイルを config.RECORD_SYNC_S 秒
(既定 2.0。0 なら止める)ごとに os.sync() でディスクへ書き切る、別のスレッド(rec-sync・daemon)。収集を始めるとき(start)に動かし、
止めるとき(stop)に止めて待つ(長くても Recorder.SYNC_JOIN_S = 1 秒)。os.sync の失敗は数えて続ける(収集・走行を止めない)。
走行のループ・カメラの書き込み(rec-io)・シーンの書き出し(rec-final)を待たせない(os.sync はこのスレッドだけで呼ぶ・lock を持たない)。
止めたときの os.sync(drive.py の後始末。最後のシーンをすぐ書き切る)はそのまま(tests/test_tether_free.py と下の最後の試験)。

実機・カメラ・センサーは使わない(ハブは偽物で、値は _on_sample で直に入れる。os.sync は偽物 = recorder の os を、sync だけ偽物にした
入れ物に替える。数えるのは、その試験の Recorder の rec-sync のスレッドからの呼び出しだけ)。直す前の写しは
tests/fixtures/tether_1031_before/recorder.py(git の 29aa281)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q tests/test_record_sync.py
"""
import ast
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jetformer import config, recorder

import speed_ff_1101_back as FB      # 2026-11-01 電圧の前送り(speed_ff)の段: recorder.py がその直しの版なら直す前の写し(= この直しの版)で照らす
from jetformer.sensors.base import Sample

from test_tether_free import BEFORE_SHA, FIX, FakeHub, OsProxy, _changed, _edit_stmts, _fn, run_full, sha

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / 'jetformer' / 'recorder.py'
WAIT_S = 5.0            # 「そのうち起きる」を待つ長さ(重い処理と同時に流しても足りるように)


class Hub(object):
    """recorder だけの試験のハブ(test_drive.FakeHub と同じ。センサーは使わない)"""

    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


class SyncSpy(object):
    """偽の os.sync。呼んだスレッドと時刻(単調時計)を残す。block = release まで戻らない(遅い os.sync)・sleep_s = その秒だけ
    かかる・fail = OSError を出す"""

    def __init__(self, fail=False, block=False, sleep_s=0.0):
        self.fail, self.block, self.sleep_s = fail, block, sleep_s
        self.calls, self.done = [], []
        self.release = threading.Event()

    def __call__(self):
        th = threading.current_thread()
        self.calls.append((th, time.monotonic()))
        if self.block:
            self.release.wait(30)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        self.done.append((th, time.monotonic()))
        if self.fail:
            raise OSError(5, 'Input/output error')

    def by(self, th):
        """th(その試験の Recorder の rec-sync)から呼んだ時刻(ほかの Recorder のスレッドの呼び出しは数えない)"""
        return [t for x, t in self.calls if x is th]

    def done_by(self, th):
        return [t for x, t in self.done if x is th]


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """rig(間隔, **偽の os.sync の形) → (Recorder, 偽の os.sync)。シーンは 1 秒・記録は tmp。落ちた試験でもスレッドを残さない"""
    made = []

    def make(interval, **kw):
        monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'raw')
        monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
        monkeypatch.setattr(config, 'RECORD_SYNC_S', interval)
        spy = SyncSpy(**kw)
        monkeypatch.setattr(recorder, 'os', OsProxy(spy))
        rec = recorder.Recorder(Hub())
        rec._warm_up = lambda: None
        made.append((rec, spy))
        return rec, spy
    yield make
    for rec, spy in made:
        spy.release.set()
        if rec.recording:
            rec.stop()


def cam(rec, i):
    """カメラの絵を 1 枚(収集の始まりから i 枚目 = 33 ms ごと)"""
    rec._on_sample('CAM_FRONT', Sample(rec.log['start_ns'] + i * 33_000_000,
                                       {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))


def ai_row():
    """AI の指令の行(drive の add_ai と同じ欄。値は何でもよい)"""
    return {'pred': np.zeros((10, 2), np.float32), 'raw': [1500.0, 1500.0], 'out': [1500.0, 1500.0], 'gains': [1.0, 1.0],
            'trim': 0.0, 'lookahead': 1, 'mode_id': 4, 'hold': False, 'select': True, 'latency_ms': 20.0, 'infer_ms': 5.0,
            'dry_run': True}


def until(cond, timeout=WAIT_S):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


def scene_dir(log):
    return config.RAW_DIR / log / ('%s-0001' % log)


def jpgs(log):
    return len(list((scene_dir(log) / 'CAM_FRONT').glob('*.jpg')))


def new_threads(before, name):
    return [t for t in threading.enumerate() if t.name == name and t not in before]


def test_config_default():
    """既定は 2.0 秒(config.RECORD_SYNC_S。環境変数 JETFORMER_RECORD_SYNC_S で変えられる)。stop が待つのは長くても 1 秒"""
    if 'JETFORMER_RECORD_SYNC_S' not in os.environ:
        assert config.RECORD_SYNC_S == 2.0
    assert recorder.Recorder.SYNC_JOIN_S == 1.0


def test_syncs_periodically_only_while_recording(rig):
    """収集している間だけ、間隔ごとに os.sync を呼ぶ(rec-sync・daemon)。始める前・止めたあとは呼ばない(止めたらスレッドも終わって
    いる)。収集ごとに動かす(2 回目の収集でもまた呼び、止めたら止まる)。収集は今まで通り(シーンを保存する)"""
    rec, spy = rig(0.05)
    before = set(threading.enumerate())
    time.sleep(0.25)
    assert rec._sync_thread is None and not new_threads(before, 'rec-sync')
    for k in range(2):
        t0 = time.monotonic()
        log = rec.start(description='t%d' % k)
        th = rec._sync_thread
        assert th is not None and th.name == 'rec-sync' and th.daemon and th.is_alive()
        for i in range(10):
            cam(rec, i)
        assert until(lambda: len(spy.by(th)) >= 5)
        assert rec.stop(only_log=log) is True
        t1 = time.monotonic()
        n = len(spy.by(th))
        assert not th.is_alive() and rec._sync_thread is None and rec._sync_stop is None
        time.sleep(0.3)                                                     # 6 回分待っても
        calls = spy.by(th)
        assert len(calls) == n and len(spy.done_by(th)) == n               # 止めたあとは呼ばない(途中の os.sync も無い)
        assert all(t0 <= t <= t1 for t in calls)
        assert min(np.diff([t0] + calls)) >= 0.04                           # 間隔ごと(続けて呼ばない)
        assert (scene_dir(log) / 'scene.json').exists() and jpgs(log) == 10
        time.sleep(1.1)                                                     # 次の収集は別の名前(秒の時刻)
    assert rec.sync_count == len([x for x, _ in spy.calls if x.name == 'rec-sync' and x not in before]) and rec.sync_errors == 0


def test_sync_ends_when_recording_ends_without_the_stop_mark(rig):
    """止める印(stop)が来なくても、収集が止まっていたら次は呼ばずに抜ける(rec-final と同じ。スレッドを残さない)"""
    rec, spy = rig(0.05)
    log = rec.start()
    th = rec._sync_thread
    assert until(lambda: len(spy.by(th)) >= 2)
    with rec.lock:
        rec.recording = False                                               # stop を通らずに止まった形
    th.join(WAIT_S)
    n = len(spy.by(th))
    time.sleep(0.2)
    assert not th.is_alive() and len(spy.by(th)) == n
    with rec.lock:
        rec.recording = True
    assert rec.stop(only_log=log) is True                                   # 後始末(印を立てるだけ。スレッドはもう無い)


@pytest.mark.parametrize('interval', [0, 0.0, -1.0])
def test_zero_never_syncs(rig, interval):
    """0 以下なら止める: rec-sync を動かさず、os.sync を呼ばない。収集で動かすスレッドは前と同じ(rec-final だけ)。収集は今まで通り"""
    rec, spy = rig(interval)
    before = set(threading.enumerate())
    log = rec.start()
    assert rec._sync_thread is None and rec._sync_stop is None
    assert [t.name for t in threading.enumerate() if t not in before] == ['rec-final']
    for i in range(10):
        cam(rec, i)
    time.sleep(0.3)
    assert rec.stop(only_log=log) is True and (scene_dir(log) / 'scene.json').exists() and jpgs(log) == 10
    assert not [x for x, _ in spy.calls if x not in before] and rec.sync_count == 0 and rec.sync_errors == 0


def test_sync_failure_is_counted_and_recording_goes_on(rig):
    """os.sync が失敗しても(OSError)数えて続ける(次の間隔でまた呼ぶ)。収集は止めない: 絵を受け付けて書き、止めればシーンを保存する。
    収集の失敗(画面の ⚠ = last_error)にはしない"""
    rec, spy = rig(0.05, fail=True)
    log = rec.start()
    th = rec._sync_thread
    assert until(lambda: len(spy.by(th)) >= 3)
    assert th.is_alive() and rec.recording
    for i in range(10):
        cam(rec, i)
    assert until(lambda: jpgs(log) == 10)
    assert rec.sync_errors >= 3 and rec.sync_count == 0 and rec.sync_last_error == 'OSError: [Errno 5] Input/output error'
    assert rec.last_error == '' and rec.status()['last_error'] == ''
    assert rec.stop(only_log=log) is True and (scene_dir(log) / 'scene.json').exists()
    n = len(spy.by(th))
    time.sleep(0.2)
    assert not th.is_alive() and len(spy.by(th)) == n and rec.sync_errors == n


def test_stop_does_not_wait_long_for_a_slow_sync(rig):
    """os.sync が戻ってこない(SD カードが遅い)ときも、stop が待つのは長くても SYNC_JOIN_S(1 秒)。保存は os.sync を待たずに終わる。
    os.sync が戻ったら、rec-sync は次を呼ばずに抜ける"""
    rec, spy = rig(0.05, block=True)
    log = rec.start()
    th = rec._sync_thread
    assert until(lambda: spy.by(th))                                        # rec-sync が os.sync の中で止まっている
    for i in range(10):
        cam(rec, i)
    t = time.monotonic()
    assert rec.stop(only_log=log) is True
    dt = time.monotonic() - t
    j = recorder.Recorder.SYNC_JOIN_S
    assert j * 0.9 <= dt < j + 2.0, dt                                      # 待つのは SYNC_JOIN_S まで(+ 保存の時間)
    assert th.is_alive() and not rec.recording and (scene_dir(log) / 'scene.json').exists()
    spy.release.set()
    th.join(WAIT_S)
    assert not th.is_alive() and len(spy.by(th)) == 1


def test_stop_waits_for_a_short_sync_in_progress(rig):
    """書き切っている途中(0.4 秒かかる os.sync)に止めたら、それが終わってから stop が戻る(止めたあとに os.sync が残らない)"""
    rec, spy = rig(0.05, sleep_s=0.4)
    log = rec.start()
    th = rec._sync_thread
    assert until(lambda: spy.by(th))
    assert rec.stop(only_log=log) is True
    assert not th.is_alive() and len(spy.done_by(th)) == len(spy.by(th)) >= 1


def test_slow_sync_does_not_block_writes(rig):
    """os.sync が戻ってこない間も、センサーの受信・AI の指令・出来事の記録(走行のループが呼ぶ。lock を待たない)、カメラの絵の書き込み
    (rec-io)、シーンの保存(rec-final。1 秒のシーンの終わりの 1 秒あと)は止まらない(os.sync は rec-sync だけで呼び、lock を持たない)"""
    rec, spy = rig(0.05, block=True)
    log = rec.start()
    th = rec._sync_thread
    assert until(lambda: spy.by(th))
    t = time.monotonic()
    for i in range(20):
        cam(rec, i)
        rec.add_ai(rec.log['start_ns'] + i * 33_000_000 + 1, ai_row())
    assert rec.add_event('test', {'a': 1}) is True
    assert time.monotonic() - t < 0.5
    assert until(lambda: jpgs(log) == 20)                                   # rec-io は書き続ける
    assert until(lambda: (scene_dir(log) / 'scene.json').exists(), timeout=10.0)     # rec-final はシーンを保存する
    assert len(spy.by(th)) == 1 and th.is_alive() and not spy.release.is_set()      # その間ずっと os.sync の中
    assert (scene_dir(log) / 'AI_CMD.npz').exists()
    ev = [json.loads(x) for x in (config.RAW_DIR / log / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [(e['kind'], e['a']) for e in ev] == [('test', 1)]
    spy.release.set()
    assert rec.stop(only_log=log) is True


def test_drive_final_sync_after_recorder_stop(monkeypatch, tmp_path):
    """AI 走行の収集(本物の Recorder・偽のハブ): 走っている間は rec-sync が間隔ごとに書き切り、止めると後始末で recorder.stop
    (rec-sync を止めて待つ)→ drive の os.sync を 1 回(最後のシーンをすぐ書き切る。そのまま)→ 止まった知らせ → GPU を返す。
    drive の os.sync のときに rec-sync はもう終わっていて、そのあとは呼ばない。log.json の drive.tether_free は True(既定)"""
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'raw')
    monkeypatch.setattr(config, 'RECORD_SYNC_S', 0.05)
    spy = SyncSpy()
    monkeypatch.setattr(recorder, 'os', OsProxy(spy))
    box = {}

    def make_rec(order):
        r = recorder.Recorder(FakeHub())
        r._warm_up = lambda: None
        r.listeners.append(lambda ev, **kw: order.append(('rec', ev)))
        box.update(order=order, rec=r)
        return r

    def loop(st, *a):
        box['order'].append('loop')
        th = box['th'] = box['rec']._sync_thread
        box['ran'] = th is not None and until(lambda: len(spy.by(th)) >= 3)       # 走っている間は書き切る
        st.update(phase='driving', announced=True, stats={'frames': 1})
        st['stop_reason'] = '停止ボタン'

    def at_sync():
        th = box['th']
        box['at_sync'] = (th.is_alive(), len(spy.by(th)), box['rec'].recording)
    d, order, rec = run_full(monkeypatch, tmp_path, rec=make_rec, loop=loop, on_sync=at_sync)
    try:
        names = [x if isinstance(x, str) else (x[0] if x[0] != 'rec' else 'rec_' + x[1]) for x in order]
        assert box['ran'] is True and names.count('sync') == 1
        i = names.index
        assert i('rec_record_start') < i('loop') < i('pca_close') < i('rec_record_stop') < i('sync') < \
            order.index(('emit', 'drive_stop')) < i('gpu_free')
        alive, n, recording = box['at_sync']
        assert alive is False and recording is False and n >= 3 and len(spy.by(box['th'])) == n
        assert d.st['message'] == '停止しました(停止ボタン)・収集したシーンを保存しました' and d.st['error'] is None
        logs = sorted(p for p in (tmp_path / 'raw').iterdir() if p.is_dir())
        assert len(logs) == 1 and json.loads((logs[0] / 'log.json').read_text(encoding='utf-8'))['drive']['tether_free'] is True
    finally:
        if rec.recording:
            rec.stop()


def test_recorder_changed_only_these():
    """recorder.py で変えたのは説明の文・Recorder.__init__(数と印の 5 行)・start(1 行)・stop(2 か所)、足したのは SYNC_JOIN_S と
    _sync_begin・_sync_loop・_sync_end だけ。変えた所を除けば直す前と AST が同じ(rec-io・rec-final・受信・保存の関数は同じ)。
    os.sync を呼ぶのは recorder では _sync_loop の 1 か所だけ、drive では Driver._run の後始末の 1 か所だけ"""
    old = FIX / 'recorder.py'
    assert sha(old) == BEFORE_SHA['recorder.py']
    rec = FB.back(REC)              # 2026-11-01 電圧の前送り(AI_CMD の ff_* の欄)の版なら直す前の写し(= この直しの版)
    assert _changed(old, rec) == ['def:Recorder.__init__', 'def:Recorder._sync_begin', 'def:Recorder._sync_end',
                                  'def:Recorder._sync_loop', 'def:Recorder.start', 'def:Recorder.stop', 'doc',
                                  'set:Recorder.SYNC_JOIN_S']
    fn = _fn(rec, 'Recorder.__init__')
    assert [ast.unparse(s) for s in fn.body[-5:]] == ['self._sync_stop = None', 'self._sync_thread = None', 'self.sync_count = 0',
                                                      'self.sync_errors = 0', "self.sync_last_error = ''"]
    fn.body = fn.body[:-5]
    assert ast.dump(fn) == ast.dump(_fn(old, 'Recorder.__init__'))
    fn = _fn(rec, 'Recorder.start')
    assert _edit_stmts(fn, lambda s: ast.unparse(s) == 'self._sync_begin()') == 1
    assert ast.dump(fn) == ast.dump(_fn(old, 'Recorder.start'))
    fn = _fn(rec, 'Recorder.stop')
    join = 'if sync_th is not None:\n    sync_th.join(self.SYNC_JOIN_S)'
    assert ast.unparse(fn.body[-2]) == join and ast.unparse(fn.body[-1]) == 'return True'      # 待つのは保存と知らせのあと
    assert _edit_stmts(fn, lambda s: ast.unparse(s) in ('sync_th = self._sync_end()', join)) == 2
    assert ast.dump(fn) == ast.dump(_fn(old, 'Recorder.stop'))

    def syncs(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'os.sync']
    assert len(syncs(ast.parse(REC.read_text(encoding='utf-8')))) == 1 and len(syncs(_fn(REC, 'Recorder._sync_loop'))) == 1
    dpy = ROOT / 'jetformer' / 'drive.py'
    assert len(syncs(ast.parse(dpy.read_text(encoding='utf-8')))) == 1 and len(syncs(_fn(dpy, 'Driver._run'))) == 1
