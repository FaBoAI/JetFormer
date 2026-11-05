"""2026-11-03 の実車の解析で見つけた 2 つの直し(~/jetformer_reports/bench_20261103_day/analysis/report_1103.md 4 章)。

1. 録画のロック: シーンの変わり目の mkdir を self.lock の中でしていたので、ディスクが止まると全部のセンサーと走行のループが
   待つ形だった(本 24 の 4.36・10.69 秒の穴)。今は、走行のループの add_ai はロックを待たず、フォルダは rec-io と保存で作る。
2. IMU の壊れた行: 読みの途中から全部 1 の行(終わりの温度が 0xFFFF)を捨てる。
"""
import ast
import builtins
import glob
import hashlib
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jetformer import config, recorder
from jetformer.sensors import imu as im
from jetformer.sensors.base import Sample

import speed_ff_1101_back as FB

ROOT = Path(__file__).resolve().parents[1]

class Hub(object):
    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


@pytest.fixture
def rec(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'raw')
    monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
    monkeypatch.setattr(config, 'RECORD_SYNC_S', 0)
    r = recorder.Recorder(Hub())
    r._warm_up = lambda: None
    yield r
    if r.recording:
        r.stop()


def cam(r, t_s, seq=0):
    r._on_sample('CAM_FRONT', Sample(r.log['start_ns'] + int(t_s * 1e9), {'image': np.zeros((8, 8, 3), np.uint8), 'seq': seq}))


def imu_sample(r, t_s):
    r._on_sample('IMU', Sample(r.log['start_ns'] + int(t_s * 1e9), {'accel': np.zeros(3, np.float32), 'gyro': np.zeros(3, np.float32),
                                                                     'mag': np.zeros(3, np.float32), 'temp': 30.0}))


def ai_row(v=1500.0):
    return {'pred': np.zeros((10, 2), np.float32), 'raw': [v, v], 'out': [v, v], 'gains': [1.0, 1.0], 'trim': 0.0, 'lookahead': 1,
            'mode_id': 4, 'hold': False, 'select': True, 'latency_ms': 20.0, 'infer_ms': 5.0, 'dry_run': False}


def add_ai(r, t_s, v=1500.0):
    r.add_ai(r.log['start_ns'] + int(t_s * 1e9), ai_row(v))


def saved_ai(log_dir):
    out = []
    for f in sorted(glob.glob(str(log_dir / '*' / 'AI_CMD.npz'))):
        with np.load(f) as z:
            out.extend(z['raw'][:, 0].tolist())
    return out


def log_path(r):
    return Path(r.log['path'])


def in_thread(fn, timeout=5.0):
    """走行のループの役: 別のスレッドで呼び、終わるまで待つ。待たされる形に戻っても、試験は止まらずに落ちる"""
    err = []

    def run():
        try:
            fn()
        except BaseException as e:      # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    assert not th.is_alive(), '待たされています'
    assert not err, err


def wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# ------------------------------------------------------------------ 見張り: 直す前の写しと、変えた所
BEFORE = ROOT / 'tests' / 'fixtures' / 'rec_imu_1103_before'
REC_PY = ROOT / 'jetformer' / 'recorder.py'
IMU_PY = ROOT / 'jetformer' / 'sensors' / 'imu.py'
BEFORE_SHA = {'recorder.py': '01497034c9288e35c56b5e7dfc3378be87519a8d2a6bf29566d9aed6944d10f0',    # = tests/speed_ff_1101_back.py の LATER
              'imu.py': 'f45f8c0a3913e5f81f9bf2df78f2c96c66a9e88048ca603f740e7a2d9dcf94a4'}
LATER_SHA = {'recorder.py': 'b5d77cea4c363b6f9302977b4591402ae7502c42c30435d90ef42fe4c15a4783', 'imu.py': '073b8675748ce8d7cdbabf04a1f6711e879849387f6fe43518908c38c6f4ff4b'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fns(path):
    out = {}

    def walk(node, prefix=''):
        for x in node.body:
            if isinstance(x, (ast.FunctionDef, ast.ClassDef)):
                out[prefix + x.name] = ast.dump(x) if isinstance(x, ast.FunctionDef) else None
                if isinstance(x, ast.ClassDef):
                    walk(x, prefix + x.name + '.')
    walk(ast.parse(Path(path).read_text(encoding='utf-8')))
    return {k: v for k, v in out.items() if v is not None}


def test_before_copies():
    lines = (BEFORE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    assert dict(reversed(x.split(None, 1)) for x in lines if x.strip()) == BEFORE_SHA
    for name, h in BEFORE_SHA.items():
        assert sha(BEFORE / name) == h


def test_later_is_this_fix():
    """今のファイルはこの直しの版。このあと誰かが直せば落ちて知らせる(値を直すときは下の 2 つの試験も見る)"""
    assert sha(REC_PY) == LATER_SHA['recorder.py'] == FB.ALSO_LATER['jetformer/recorder.py']
    assert sha(IMU_PY) == LATER_SHA['imu.py']


def test_recorder_changed_only_these_functions():
    old, now = _fns(BEFORE / 'recorder.py'), _fns(REC_PY)
    assert sorted(k for k in now if k not in old) == ['Recorder._drain_ai', 'Recorder._event_worker']
    assert [k for k in old if k not in now] == []
    assert sorted(k for k in old if old[k] != now[k]) == [
        'Recorder.__init__', 'Recorder._finalize', 'Recorder._finalize_loop', 'Recorder._io_worker', 'Recorder._on_sample',
        'Recorder._save_scene', 'Recorder.add_ai', 'Recorder.add_event', 'Recorder.start', 'Recorder.status', 'Recorder.stop',
        'Recorder.update_drive', '_SceneBuffer.__init__']


def test_imu_changed_only_these_functions():
    old, now = _fns(BEFORE / 'imu.py'), _fns(IMU_PY)
    assert set(old) == set(now)
    assert sorted(k for k in old if old[k] != now[k]) == ['ImuSensor.__init__', 'ImuSensor._poll', 'ImuSensor.status']


def test_no_disk_call_under_the_lock_in_the_hot_paths():
    """センサーと走行のループが通る所(_on_sample・add_ai・_drain_ai・_scene_for・_SceneBuffer)にディスクの呼び出しが無い"""
    tree = ast.parse(REC_PY.read_text(encoding='utf-8'))
    names = {'_on_sample', 'add_ai', '_drain_ai', '_scene_for'}
    checked = []
    for cls in (x for x in tree.body if isinstance(x, ast.ClassDef) and x.name in ('Recorder', '_SceneBuffer')):
        for fn in (x for x in cls.body if isinstance(x, ast.FunctionDef)):
            if fn.name in names or (cls.name == '_SceneBuffer' and fn.name == '__init__'):
                checked.append(fn.name)
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call):
                        text = ast.unparse(n.func)
                        assert not any(w in text for w in ('mkdir', 'open', 'write', 'save', 'rmtree', 'imwrite', 'sync')), (fn.name, text)
    assert sorted(checked) == ['__init__', '_drain_ai', '_on_sample', '_scene_for', 'add_ai']


# ------------------------------------------------------------------ 1. 録画のロック
def test_add_ai_does_not_wait_for_the_lock(rec):
    """ロックを誰かが持っていても add_ai はすぐ戻り、指令は失われない(空いた人があとで移す)"""
    rec.start()
    d = log_path(rec)
    cam(rec, 0.1)
    with rec.lock:
        in_thread(lambda: (add_ai(rec, 0.1, 1501.0), add_ai(rec, 0.2, 1502.0)))
        assert rec.ai_deferred == 2 and len(rec._ai_pending) == 2
    imu_sample(rec, 0.25)                     # 次のセンサーの値で移る
    assert not rec._ai_pending
    assert [x[1]['raw'][0] for x in rec.scenes[0].ai] == [1501.0, 1502.0]
    rec.stop()
    assert saved_ai(d) == [1501.0, 1502.0]


def test_add_ai_moves_right_away_when_the_lock_is_free(rec):
    rec.start()
    cam(rec, 0.1)
    add_ai(rec, 0.1, 1501.0)
    assert not rec._ai_pending and rec.ai_deferred == 0
    assert [x[1]['raw'][0] for x in rec.scenes[0].ai] == [1501.0]


def test_new_scene_does_not_touch_the_disk_under_the_lock(rec, monkeypatch):
    """シーンの変わり目で mkdir が止まっても、センサーも走行のループも待たない(本 24 の形)。絵は rec-io が作ってから書く"""
    rec.start()
    d = log_path(rec)
    release = threading.Event()
    real = Path.mkdir
    called = []

    def slow_mkdir(self, *a, **k):
        called.append((threading.current_thread().name, self))
        release.wait(10)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, 'mkdir', slow_mkdir)
    try:
        def one_frame():
            cam(rec, 0.1, 1)                  # 新しいシーン(ここで前は mkdir が走った)
            imu_sample(rec, 0.1)
            add_ai(rec, 0.1, 1501.0)
            add_ai(rec, 1.1, 1502.0)          # 走行のループが次のシーンを作る形
            cam(rec, 1.1, 2)                  # 絵の無いシーンは保存しない決まりなので、2 つ目にも絵を入れる

        in_thread(one_frame)
        assert rec.ai_deferred == 0
        assert wait_until(lambda: called)
        assert [n for n, _ in called] == ['rec-io'], called    # rec-io だけが mkdir を呼んで待っている
    finally:
        release.set()
    monkeypatch.setattr(Path, 'mkdir', real)
    rec.stop()
    assert saved_ai(d) == [1501.0, 1502.0]
    assert glob.glob(str(d / '*-0001' / 'CAM_FRONT' / '*.jpg'))
    assert glob.glob(str(d / '*-0002' / 'CAM_FRONT' / '*.jpg'))      # 2 つ目のシーンのフォルダも rec-io が作る


def test_scene_folder_is_made_on_save_even_if_the_images_could_not_be_written(rec):
    rec.start()
    d = log_path(rec)
    with rec.lock:
        buf = rec._scene_for(rec.log['start_ns'] + 100_000_000)
        buf.cam.append((rec.log['start_ns'] + 100_000_000, 1, 'CAM_FRONT/1.jpg'))      # rec-io が書けなかった形(行はあるが絵もフォルダも無い)
        assert not buf.path.exists()
    rec.stop()
    assert (d / buf.name / 'scene.json').exists()


def test_full_queue_leaves_no_row_without_a_picture(rec):
    """rec-io の待ち行列が満杯のときは行も残さない(絵の無い行があると dataset の写しが FileNotFoundError で止まる)"""
    rec.start()
    d = log_path(rec)
    full = queue.Queue(maxsize=1)
    full.put(None)
    real, rec._io = rec._io, full
    try:
        cam(rec, 0.1, 1)
    finally:
        rec._io = real
    assert rec.dropped_writes == 1 and rec.scenes[0].cam == []
    cam(rec, 0.2, 2)
    rec.stop()
    scene = d / rec.saved[0]
    with np.load(scene / 'CAM_FRONT.npz') as z:
        assert z['seq'].tolist() == [2]
        assert all((scene / str(f)).exists() for f in z['files'])


def test_empty_scene_leaves_no_folder(rec):
    rec.start()
    d = log_path(rec)
    imu_sample(rec, 0.1)                      # 絵の無いシーンは残さない(前と同じ)
    rec.stop()
    assert not glob.glob(str(d / '*-0001'))


def test_stop_moves_what_is_left(rec):
    rec.start()
    d = log_path(rec)
    cam(rec, 0.1)
    with rec.lock:
        in_thread(lambda: add_ai(rec, 0.2, 1503.0))
        assert len(rec._ai_pending) == 1
    rec.stop()
    assert saved_ai(d) == [1503.0]


def test_scene_save_takes_what_is_left(rec):
    """シーンを保存する瞬間に置いてあった指令も、そのシーンに入る"""
    rec.start()
    d = log_path(rec)
    cam(rec, 0.1)
    rec._ai_pending.append((rec.log['start_ns'] + 200_000_000, ai_row(1505.0)))
    rec._finalize(0, partial=False)
    assert saved_ai(d) == [1505.0]
    rec.stop()


def test_rec_final_moves_what_is_left_when_no_sensor_comes(rec):
    """センサーの値が来なくても、rec-final が移す(待ち行列にたまり続けない)"""
    rec.start()
    with rec.lock:
        in_thread(lambda: add_ai(rec, 0.05, 1504.0))
        assert len(rec._ai_pending) == 1
    assert wait_until(lambda: not rec._ai_pending)


def test_move_after_stop_does_not_raise(rec):
    """止めたあと(log が無い)に 1 行だけ間に合わなかった形でも、走行のループへ例外を返さない"""
    rec.start()
    t = rec.log['start_ns']
    rec.stop()
    rec._ai_pending.append((t, ai_row()))
    with rec.lock:
        rec._drain_ai()
    assert not rec._ai_pending


def test_next_start_does_not_take_old_commands(rec):
    rec.start()
    rec.stop()
    rec._ai_pending.append((0, ai_row()))     # 止める瞬間に間に合わなかった 1 行
    rec.start()
    assert not rec._ai_pending


def test_add_ai_after_stop_is_ignored(rec):
    rec.start()
    t = rec.log['start_ns']
    rec.stop()
    rec.add_ai(t, ai_row())
    assert not rec._ai_pending


def test_order_and_count_survive_contention(rec, monkeypatch):
    """ロックを取り合っても、AI の指令の順番と数は変わらない(行の数は AI_PENDING_MAX より少なく。シーンは閉じない長さ)"""
    monkeypatch.setattr(config, 'SCENE_SECONDS', 60.0)
    rec.start()
    d = log_path(rec)
    cam(rec, 0.1)
    stop = threading.Event()
    held = threading.Event()

    def holder():                             # ロックを持ったまま少し待つ人(シーンの変わり目などの役)
        while not stop.is_set():
            with rec.lock:
                held.set()
                time.sleep(0.001)
            time.sleep(0.0002)

    def sensors():
        i = 0
        while not stop.is_set():
            imu_sample(rec, 0.1 + (i % 50) * 0.01)
            i += 1
            time.sleep(0.001)

    ths = [threading.Thread(target=f, daemon=True) for f in (holder, sensors)]
    for th in ths:
        th.start()
    assert held.wait(5)
    want = [1500.0 + i for i in range(500)]
    for i, v in enumerate(want):
        add_ai(rec, 0.2 + i * 1e-4, v)
        if i % 50 == 49:
            time.sleep(0.002)
    stop.set()
    for th in ths:
        th.join(2)
    assert rec.ai_deferred > 0                # 本当にぶつかった
    rec.stop()
    assert saved_ai(d) == want


def test_oldest_is_dropped_when_the_lock_never_comes(rec, monkeypatch):
    """ロックが空かないまま上限を超えたら、古い方から捨てる(メモリを増やし続けない)"""
    monkeypatch.setattr(rec, '_ai_pending', recorder.collections.deque(maxlen=3))
    rec.start()
    d = log_path(rec)
    cam(rec, 0.1)
    with rec.lock:
        in_thread(lambda: [add_ai(rec, 0.2, 1500.0 + i) for i in range(5)])
        assert len(rec._ai_pending) == 3
    rec.stop()
    assert saved_ai(d) == [1502.0, 1503.0, 1504.0]


def test_pending_is_bounded(rec):
    assert recorder.Recorder.AI_PENDING_MAX >= 30 * 30         # 30 Hz で 30 秒より長く持てる
    assert rec._ai_pending.maxlen == recorder.Recorder.AI_PENDING_MAX


def test_add_event_does_not_wait_for_a_stuck_disk(rec, monkeypatch):
    """走行のループからも呼ぶので、ディスクが止まっても EVENT_WAIT_S で戻る。止めるときに書き切る"""
    rec.start()
    d = log_path(rec)
    release = threading.Event()
    real_open = builtins.open

    def slow_open(path, *a, **k):
        if str(path).endswith('events.jsonl'):
            release.wait(10)
        return real_open(path, *a, **k)

    monkeypatch.setattr(recorder, 'open', slow_open, raising=False)
    try:
        took = []

        def two():
            assert rec.add_event('recovery_guard', {'a': 1}) is True
            t0 = time.monotonic()
            assert rec.add_event('recovery_guard', {'a': 2}) is True
            took.append(time.monotonic() - t0)

        in_thread(two)                        # release はまだ: ディスクを待っていない
        assert took[0] >= rec.EVENT_WAIT_S * 0.9     # ただし EVENT_WAIT_S までは待つ
        assert not rec.lock.locked()          # 書いている間もロックは空いている(センサーを待たせない)
    finally:
        release.set()
    rec.stop()
    lines = (d / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    assert [('"a": 1' in lines[0]), ('"a": 2' in lines[1])] == [True, True]


def test_add_event_is_written_by_rec_event(rec):
    rec.start()
    d = log_path(rec)
    assert rec.add_event('params', {'x': 1}) is True
    rec._events.join()
    assert '"kind": "params"' in (d / 'events.jsonl').read_text(encoding='utf-8')


def test_rec_event_says_when_it_has_written(rec):
    """書き終わりを知らせる(知らせないと、ふつうのディスクでも毎回 EVENT_WAIT_S 待つ)"""
    rec.start()
    done = threading.Event()
    rec._events.put((log_path(rec) / 'x.jsonl', 'line', done))
    assert done.wait(5)


def test_status_reads_the_mode_name_outside_the_lock(rec, monkeypatch):
    """ジャンルの名前はファイルから読むので、ロックの外で(ロックを持つと、ディスクが止まったときセンサーが待つ)"""
    rec.start()
    seen = []
    real = recorder.modes.name_of

    def spy(mode_id):
        seen.append(rec.lock.locked())
        return real(mode_id)

    monkeypatch.setattr(recorder.modes, 'name_of', spy)
    st = rec.status()
    assert seen == [False] and st['mode_name'] == real(st['mode_id']) and st['recording'] is True
    with rec.lock:
        rec.runs_mode = True
    st = rec.status()
    assert seen == [False, False] and st['mode_name'] == real(st['mode_id']) and st['runs_mode'] is True


def test_add_event_when_not_recording(rec):
    assert rec.add_event('params', {'x': 1}) is False


def test_add_event_during_stop_is_refused(rec):
    rec.start()
    rec.recording = False                     # stop が保存を待っている間の形(log は残っている)
    assert rec.add_event('params', {'x': 1}) is False
    rec.recording = True


def test_add_event_write_error_does_not_stop_recording(rec, monkeypatch):
    rec.start()

    def bad_open(path, *a, **k):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(recorder, 'open', bad_open, raising=False)
    assert rec.add_event('params', {'x': 1}) is True
    rec._events.join()
    assert '出来事の保存失敗' in rec.last_error
    monkeypatch.delattr(recorder, 'open', raising=False)
    assert rec.recording


def test_update_drive_writes_outside_the_lock(rec, monkeypatch):
    rec.start(drive={'run': 'x'})
    name = rec.log['name']
    seen = []
    real = recorder.write_json

    def spy(path, obj):
        seen.append(rec.lock.locked())
        return real(path, obj)

    monkeypatch.setattr(recorder, 'write_json', spy)
    assert rec.update_drive(name, params_end={'a': 1}) is True
    assert seen == [False]


# ------------------------------------------------------------------ 2. IMU の壊れた行
# 2026-10-30・11-03 の記録にあった本物の行(accel 3・gyro 3・温度 の生の値)
BROKEN = [[-2294, -715, 4531, 133, 398, 4095, -1],       # gz の下の桁から全部 1
          [-1571, -196, 2759, 968, -968, 32767, -1],
          [-2060, 453, 5206, 523, -1879, -1, -1],
          [-1179, -683, 573, 2047, -1, -1, -1],
          [-3321, -704, 7679, -1, -1, -1, -1],
          [-2272, 8191, -1, -1, -1, -1, -1],
          [-2745, -1, -1, -1, -1, -1, -1],
          [32767, -1, -1, -1, -1, -1, -1],               # No.25 の「R1 8.02G」
          [-1, -1, -1, -1, -1, -1, -1]]
REAL = [[-94, -70, 4030, -1, -1, -1, 7568],              # 止まっている車: gyro が 3 軸とも -1 でも温度が正しければ本物
        [-8689, -32768, -1214, 3520, -6660, 6336, 6576],  # 本物の強い当たり(加速度の満点)
        [0, 0, 4096, 0, 0, 0, 6000]]


class FakeMsg(object):
    def __init__(self, data=b''):
        self.data = data

    def __bytes__(self):
        return self.data


class FakeBus(object):
    def __init__(self, rows):
        self.rows = list(rows)

    def write_byte_data(self, addr, reg, val):
        pass

    def i2c_rdwr(self, msg_w, msg_r):
        msg_r.data = struct.pack('>hhhhhhh', *self.rows.pop(0))


class FakeI2cMsg(object):
    @staticmethod
    def write(addr, data):
        return FakeMsg()

    @staticmethod
    def read(addr, n):
        return FakeMsg()


@pytest.fixture
def sensor(monkeypatch):
    monkeypatch.setattr(im, 'i2c_msg', FakeI2cMsg)
    s = im.ImuSensor(dict(config.IMU, rate_hz=100000))
    s.a_scale, s.g_scale = 4096.0, 32.8
    s._next = 0.0
    return s


@pytest.mark.parametrize('row', BROKEN)
def test_broken_read_is_dropped(sensor, row):
    sensor.bus = FakeBus([row])
    assert sensor._poll() == []
    assert sensor.bad_reads == 1 and sensor.fail_streak == 1 and sensor.bus_errors == 0


@pytest.mark.parametrize('row', REAL)
def test_real_read_is_kept(sensor, row):
    sensor.bus = FakeBus([row])
    got = sensor._poll()
    assert len(got) == 1 and sensor.bad_reads == 0 and sensor.fail_streak == 0
    assert got[0].data['accel'][0] == pytest.approx(row[0] / 4096.0 * im.G)
    assert got[0].data['gyro'][2] == pytest.approx(row[5] / 32.8)


def test_good_read_resets_the_streak(sensor):
    sensor.bus = FakeBus([BROKEN[0], BROKEN[1], REAL[0], BROKEN[2]])
    assert [len(sensor._poll()) for _ in range(4)] == [0, 0, 1, 0]
    assert sensor.bad_reads == 3 and sensor.fail_streak == 1


def test_broken_reads_in_a_row_reopen(sensor):
    """壊れた行が続くのは読めないのと同じなので、開き直す(前は壊れた値を出し続けた)"""
    n = sensor.MAX_FAIL_STREAK
    sensor.bus = FakeBus([BROKEN[0]] * n)
    for _ in range(n - 1):
        assert sensor._poll() == []
    with pytest.raises(RuntimeError):
        sensor._poll()


def test_broken_read_during_discard_is_counted_once(sensor):
    sensor._discard = 2
    sensor.bus = FakeBus([REAL[0], BROKEN[0], REAL[0], REAL[0]])
    assert [len(sensor._poll()) for _ in range(4)] == [0, 0, 0, 1]
    assert sensor.bad_reads == 1 and sensor._discard == 0


def test_status_has_the_count(sensor):
    sensor.bad_reads = 3
    assert sensor.status()['bad_reads'] == 3


def test_the_rule_on_the_real_record():
    """本物の記録: No.25 の R1 8.02G の行は捨てる行で、そのほかの強い加速度の行は捨てない"""
    s = sorted(glob.glob(str(ROOT / 'data' / 'raw' / '20261103-185433' / '*' / 'IMU.npz')))
    if not s:
        pytest.skip('11/3 の記録がありません')
    rows, broken = 0, 0
    strong_kept = 0
    for f in s:
        with np.load(f) as z:
            a, t = z['accel'].copy(), z['temp'].copy()
        raw_t = np.rint((t - 21.0) * 333.87 + 21.0).astype(int)
        hg = np.hypot(a[:, 0], a[:, 1]) / im.G
        drop = raw_t == -1
        rows += len(t)
        broken += int(drop.sum())
        strong_kept += int(((hg >= 4.0) & ~drop).sum())
        assert not ((hg >= 7.9) & (np.rint(a[:, 0] / im.G * 4096).astype(int) == 32767) & ~drop).any()
    assert rows > 0 and 0 < broken < rows * 0.01
    assert strong_kept > 0
