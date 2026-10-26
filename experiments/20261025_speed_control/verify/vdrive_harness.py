"""反証 V-drive(2026-10-25): 走行ループ(drive._loop)を偽物のハードウェアで回す独立の道具。
段 drive の試験(tests/test_speed_ctrl*.py)とは別に作った。違うところ:
  - 実車の記録(10/24)の ToF の深さ・IMU の 3 軸の加速度(az も)・車両の値をそのまま入れる。床の基準(data/tof_floor_ref.npy)
    を読むので ToF ガードが動く(段の試験は床の基準なし = ガードなし)
  - モデルの偽物は「撮像の時刻 → 記録の model_pred」で引く(中立で止めたフレームがずれても予測がずれない)
  - 車両の値に電圧(vbat_v・vbat_ok)を足せる。センサーの途切れ(時刻の範囲で消す)を足せる
  - 閉ループ: 書いたスロットルで速さが変わる偽物の車(course_sim)
車・PCA9685・I2C・カメラ・ToF には触れない。torch は読まない。
"""
import collections
import contextlib
import glob
import importlib.util
import json
import os
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np

from jetformer import clock, config, drive
from jetformer import safety_caps as sc
from jetformer.sensors.base import Sample

ROOT = Path('/home/jetson/jetformer')
RAW = ROOT / 'data' / 'raw'
FIX = ROOT / 'tests' / 'fixtures' / 'speed_ctrl_before'
NS = 1_000_000_000


def load_before():
    spec = importlib.util.spec_from_loader('jetformer._vdrive_before', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    assert 'speed_ceiling' not in mod.PARAM_LIMITS
    return mod


# ---------------------------------------------------------------- 偽物
class PCA(object):
    def __init__(self):
        self.pairs, self.single = [], []
        self.last = None

    def set_pair_us(self, sc_, s, tc, t):
        self.pairs.append((s, t))
        self.last = (s, t)

    def set_us(self, ch, us):
        self.single.append((ch, us))

    def set_osc_scale(self, scale):
        pass


class Rec(object):
    recording = False

    def __init__(self):
        self.rows, self.events = [], []

    def add_ai(self, t_ns, data):
        self.rows.append((t_ns, data))

    def add_event(self, kind, data):
        self.events.append((kind, data))
        return True


def fake_normalize(image, tof_raw, imu, vehicle, use_command):
    return (np.zeros((3, 4, 4), np.float32), np.zeros((1, 2, 2), np.float32), np.asarray(imu, np.float32),
            (np.asarray(vehicle, np.float32) - 1500.0) / 500.0)


class TimedRunner(object):
    """撮像の時刻 → 予測 [10,2] µs。pred_of(cam_t, feeder) を呼ぶ"""

    def __init__(self, pred_of, feeder):
        self.pred_of, self.feeder = pred_of, feeder
        self.n = 0

    def run_numpy(self, **kw):
        self.n += 1
        p = np.asarray(self.pred_of(self.feeder.cam_t, self.feeder), np.float32)
        return {'actions': ((p - drive.PWM_CENTER) / drive.PWM_SCALE)[None].astype(np.float32)}


class Feeder(object):
    """cam_evt の代わり。wait() のたびに次の撮像までのセンサー値をバッファへ。open loop(記録)と閉ループ(sim)の両方"""

    def __init__(self, d, src, bufs, cam_slot, box, imu_lead_ns=3_000_000):
        self.d, self.src = d, src
        self.tof_buf, self.imu_buf, self.veh_buf = bufs
        self.cam_slot, self.box = cam_slot, box
        self.i = 0
        self.cam_t = None
        self.imu_lead = imu_lead_ns

    def wait(self, timeout):
        cam_t = self.src.next_cam(self.i)
        if cam_t is None:
            self.d._stop.set()
            return False
        self.cam_t = cam_t
        for t, x in self.src.imu_upto(cam_t + self.imu_lead):
            self.imu_buf.append((t, x))
        for t, x in self.src.veh_upto(cam_t - 1_000_000):
            self.veh_buf.append((t, x))
        for t, x in self.src.tof_upto(cam_t):
            self.tof_buf.append((t, x))
        self.cam_slot[0] = Sample(cam_t, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': self.i})
        self.box['ns'] = cam_t + 8_000_000
        self.box['m'] = (cam_t - self.src.t0) / 1e9 + 100.0
        self.d.heartbeat_t = self.box['m']
        self.i += 1
        return True

    def clear(self):
        pass

    def set(self):
        pass


@contextlib.contextmanager
def patched(mod, box, floor_ref):
    saved = (clock.now_ns, mod.time, mod.SETTINGS_PATH, mod.FLOOR_REF_PATH)
    tmp = tempfile.mkdtemp(prefix='vdrive_')
    clock.now_ns = lambda: box['ns']
    mod.time = types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter, time=lambda: 1.0,
                                     sleep=lambda s: None, strftime=_time.strftime)
    mod.SETTINGS_PATH = Path(tmp) / 'drive_settings.json'
    mod.FLOOR_REF_PATH = Path(floor_ref) if floor_ref else Path(tmp) / 'none.npy'
    try:
        yield
    finally:
        clock.now_ns, mod.time, mod.SETTINGS_PATH, mod.FLOOR_REF_PATH = saved


def run_loop(mod, params, src, pred_of, dry_run=False, description=None, floor_ref=str(ROOT / 'data' / 'tof_floor_ref.npy'),
             ref=(1441.0, 1496.0), raw_params=False):
    """mod の _loop を回す → dict(pca, rec, st, d)。raw_params = True なら決まり(_speed_ctrl_rules)を通さずに params を入れる"""
    box = {'ns': src.t0, 'm': 100.0}
    with patched(mod, box, floor_ref):
        rec = Rec()
        d = mod.Driver(hub=None, recorder=rec)
        d.params = dict(mod.DEFAULT_PARAMS, **{k: v for k, v in params.items() if k in mod.DEFAULT_PARAMS})
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 5}
        if description is not None:
            d.st['description'] = description
        pca = PCA()
        out = mod._Output(pca, config.PCA9685, ref, lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        feeder = Feeder(d, src, bufs, cam_slot, box)
        src.attach(pca, ref)
        runner = TimedRunner(pred_of, feeder)
        d._loop(d.st, runner, fake_normalize, pca, out, ref, 6, 0.0, cam_slot, feeder, *bufs, dry_run)
        d._stop.clear()
    return {'pca': pca, 'rec': rec, 'st': d.st, 'd': d}


# ---------------------------------------------------------------- 記録(open loop)
class LogSource(object):
    def __init__(self, log, vbat=None, drop=(), shift_to=1_800_000_000 * NS):
        cat = collections.defaultdict(list)
        keys = {'AI_CMD': ('timestamps_us', 'model_pred', 'select', 'hold', 'dry_run', 'out'),
                'VEHICLE': ('timestamps_us', 'erpm', 'pwm', 'valid', 'select', 'moving', 'hall_state', 'hall_count',
                            'hall_errors', 'seq', 'board_t_us'),
                'IMU': ('timestamps_us', 'accel', 'gyro'), 'TOF_FRONT': ('timestamps_us', 'depth')}
        for dd in sorted(glob.glob(str(RAW / log / (log + '-*')))):
            parts = {}
            try:
                for n, ks in keys.items():
                    with np.load(os.path.join(dd, n + '.npz')) as z:
                        parts[n] = {k: z[k] for k in ks}
            except Exception:       # noqa: BLE001  壊れたチャンクは使わない
                continue
            for n, ks in keys.items():
                for k in ks:
                    cat[(n, k)].append(parts[n][k])
        r = {k: np.concatenate(v) for k, v in cat.items()}
        for n in keys:
            o = np.argsort(r[(n, 'timestamps_us')], kind='stable')
            for k in keys[n]:
                r[(n, k)] = r[(n, k)][o]
        self.r = r
        meta = json.load(open(RAW / log / 'log.json'))
        self.meta = meta
        at = r[('AI_CMD', 'timestamps_us')].astype(np.int64) * 1000
        self.shift = shift_to - int(at[0]) + 150_000_000
        self.t0 = shift_to
        self.cam = at + self.shift
        self.pred = r[('AI_CMD', 'model_pred')].astype(np.float64)

        self._drop = drop
        it = r[('IMU', 'timestamps_us')].astype(np.int64) * 1000 + self.shift
        self.imu = [(int(it[k]), np.concatenate([r[('IMU', 'accel')][k], np.deg2rad(r[('IMU', 'gyro')][k])]))
                    for k in range(len(it)) if self._keep('IMU', it[k])]
        vt = r[('VEHICLE', 'timestamps_us')].astype(np.int64) * 1000 + self.shift
        self.veh = []
        for k in range(len(vt)):
            if not self._keep('VEHICLE', vt[k]):
                continue
            dct = {'pwm': [int(x) for x in r[('VEHICLE', 'pwm')][k]], 'valid': [bool(x) for x in r[('VEHICLE', 'valid')][k]],
                   'select': bool(r[('VEHICLE', 'select')][k]), 'moving': bool(r[('VEHICLE', 'moving')][k]),
                   'erpm': int(r[('VEHICLE', 'erpm')][k]), 'hall_state': int(r[('VEHICLE', 'hall_state')][k]),
                   'hall_count': int(r[('VEHICLE', 'hall_count')][k]), 'hall_errors': int(r[('VEHICLE', 'hall_errors')][k]),
                   'seq': int(r[('VEHICLE', 'seq')][k]), 'board_t_us': int(r[('VEHICLE', 'board_t_us')][k]),
                   'quality': 'snapshot'}
            if vbat is not None:
                v = vbat((vt[k] - self.t0) / 1e9, k)
                dct['vbat_ok'] = v is not None and np.isfinite(v)
                dct['vbat_v'] = float(v) if v is not None else float('nan')
            self.veh.append((int(vt[k]), dct))
        tt = r[('TOF_FRONT', 'timestamps_us')].astype(np.int64) * 1000 + self.shift
        self.tof = [(int(tt[k]), r[('TOF_FRONT', 'depth')][k]) for k in range(len(tt)) if self._keep('TOF', tt[k])]
        self.ki = self.kv = self.kt = 0

    def _keep(self, ch, t):
        s = (t - self.t0) / 1e9
        return not any(a <= s < b and (c is None or c == ch) for c, a, b in self._drop)

    def attach(self, pca, ref):
        self.ki = self.kv = self.kt = 0

    def next_cam(self, i):
        return int(self.cam[i]) if i < len(self.cam) else None

    def _upto(self, lst, attr, t):
        k = getattr(self, attr)
        out = []
        while k < len(lst) and lst[k][0] <= t:
            out.append(lst[k])
            k += 1
        setattr(self, attr, k)
        return out

    def imu_upto(self, t):
        return self._upto(self.imu, 'ki', t)

    def veh_upto(self, t):
        return [(t_, dict(x)) for t_, x in self._upto(self.veh, 'kv', t)]

    def tof_upto(self, t):
        return self._upto(self.tof, 'kt', t)

    def pred_of(self, cam_t, feeder):
        i = int(np.searchsorted(self.cam, cam_t))
        p = self.pred[min(i, len(self.pred) - 1)]
        if not np.all(np.isfinite(p)):
            p = np.tile([[1441.0, 1496.0]], (10, 1))
        return p


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b
