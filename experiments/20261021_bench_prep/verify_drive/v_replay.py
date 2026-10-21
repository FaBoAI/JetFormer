#!/usr/bin/env python3
"""V-drive(確認・反証する側, 2026-10-21): 10/10 の実車の記録のセンサーの流れを、そのまま Driver._loop に入れて回す。

変更前の木(git archive HEAD を取り出したもの)と今の木で、同じ入力・同じ時刻・同じ乱れを入れ、結果を pickle に書く。
比べるのは v_compare.py。車・PCA9685・I2C・カメラ・ToF・GPU・torch は使わない(全部偽物。推論は記録の pred を返すだけ)。

    python -I v_replay.py --tree <木> --log 20261010-173445 --variant plain --zone 0 --out x.pkl

variant:
  plain    乱れなし(基準の値: ステア 1.0・速度 1.0・先読み 3・ToF ガード 1・リカバリー ON)
  perturb  乱れあり(車両・IMU の途切れ、IMU の時刻の逆戻り・重複、カメラの途切れ、処理の遅れ、推論の NaN、
           PCA9685 の書き込みの失敗、CH3 が届かない、同じ seq、走行中のパラメータの変更、ジャンルの変更)
  dry      推論の確認(dry_run)
  toggle   zone_caps を走行の途中で 0 → 1 → 0(今の木だけ意味がある)
"""
import argparse
import collections
import hashlib
import pickle
import sys
import time as _time
import types
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--tree', required=True)
ap.add_argument('--log', required=True)
ap.add_argument('--variant', default='plain')
ap.add_argument('--zone', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--max-frames', type=int, default=0)
ap.add_argument('--zone-fail-at', type=int, default=-1, help='今の木: zone_frame がこの回で例外')
ap.add_argument('--imu-fail-at', type=int, default=-1, help='今の木: zone_imu がこの回で例外')
ap.add_argument('--pca-fail', default='', help='このフレームの番号(カンマ区切り)で PCA9685 の書き込みを 1 回失敗させる')
args = ap.parse_args()

TREE = Path(args.tree).resolve()
sys.path.insert(0, str(TREE))
from jetformer import clock, config, drive  # noqa: E402
from jetformer.sensors.base import Sample  # noqa: E402

assert Path(drive.__file__).resolve().parent.parent == TREE, drive.__file__
NEW = hasattr(drive, '_ZoneCaps')
RAW = Path('/home/jetson/jetformer/data/raw')
FLOOR = Path('/home/jetson/jetformer/data/tof_floor_ref.npy')
REF = (1441.0, 1496.0)                 # モデル B のステアの中心・スロットルの中立
BASE = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'steer_trim_us': 0.0, 'lookahead': 3, 'osc_percent': 0.0,
        'esc_reverse': 1, 'recovery': 1, 'rev_us': 300.0, 'rev_s': 0.4, 'tof_guard': 1, 'speed_cap': 0.0,
        'speed_scale': 1.0, 'tcp_mode': 0, 'ensemble_j': 0, 'ensemble_m': 0.5}


# ---------------------------------------------------------------- 記録を読む
def load(log):
    root = RAW / log
    parts = collections.defaultdict(list)
    for s in sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')):
        for n in ('IMU', 'VEHICLE', 'AI_CMD', 'TOF_FRONT'):
            p = s / (n + '.npz')
            if p.exists():
                z = np.load(p)
                if len(z['timestamps_us']):
                    parts[n].append({k: z[k] for k in z.files if z[k].ndim >= 1})
    out = {}
    for n, ps in parts.items():
        keys = set.intersection(*[set(p) for p in ps])
        out[n] = {k: np.concatenate([p[k] for p in ps]) for k in keys}
    return out


L = load(args.log)
I, V, A, T = L['IMU'], L['VEHICLE'], L['AI_CMD'], L['TOF_FRONT']
o = np.argsort(I['timestamps_us'], kind='stable')
imu = [(int(t) * 1000, np.concatenate([a, np.deg2rad(g)]))
       for t, a, g in zip(I['timestamps_us'][o], I['accel'][o], I['gyro'][o])]
o = np.argsort(V['timestamps_us'], kind='stable')
veh = [(int(V['timestamps_us'][i]) * 1000,
        {'pwm': [int(x) for x in V['pwm'][i]], 'valid': [bool(x) for x in V['valid'][i]], 'select': bool(V['select'][i]),
         'moving': bool(V['moving'][i]), 'erpm': int(V['erpm'][i])}) for i in o]
o = np.argsort(T['timestamps_us'], kind='stable')
tof = [(int(T['timestamps_us'][i]) * 1000, T['depth'][i]) for i in o]
o = np.argsort(A['timestamps_us'], kind='stable')
cam_t = [int(t) * 1000 for t in A['timestamps_us'][o]]
preds = A['pred'][o].astype(np.float32)
if args.max_frames:
    cam_t, preds = cam_t[:args.max_frames], preds[:args.max_frames]
# 同じ時刻のフレームは捨てる(seq を増やすため)
keep = [0] + [i for i in range(1, len(cam_t)) if cam_t[i] > cam_t[i - 1]]
cam_t = [cam_t[i] for i in keep]
preds = preds[keep]
N = len(cam_t)

# ---------------------------------------------------------------- 乱れの計画(両方の木で同じ。乱数の種は記録の名前)
rng = np.random.default_rng(int(hashlib.sha256(args.log.encode()).hexdigest()[:8], 16))
plan = {'veh_drop': set(), 'imu_drop': set(), 'cam_timeout': set(), 'late': set(), 'nan': set(), 'pca_fail': set(),
        'ch3_lost': set(), 'dup_seq': set(), 'imu_swap': set(), 'imu_dup': set(), 'params': {}, 'mode': {}}
if args.variant == 'perturb':
    i = 0
    while i < N:
        r = rng.random()
        if r < 0.006:
            plan['veh_drop'].update(range(i, i + int(rng.integers(3, 12))))       # 0.1〜0.4 秒の途切れ
        elif r < 0.010:
            plan['imu_drop'].update(range(i, i + int(rng.integers(2, 9))))
        elif r < 0.014:
            plan['cam_timeout'].add(i)
        elif r < 0.017:
            plan['late'].add(i)
        elif r < 0.019:
            plan['nan'].add(i)
        elif r < 0.021:
            plan['pca_fail'].add(i)
        elif r < 0.023:
            plan['ch3_lost'].update(range(i, i + int(rng.integers(2, 6))))
        elif r < 0.026:
            plan['dup_seq'].add(i)
        elif r < 0.036:
            plan['imu_swap'].add(i)
        elif r < 0.040:
            plan['imu_dup'].add(i)
        i += 1
    changes = [{'steer_gain': 1.5, 'throttle_gain': 0.5}, {'lookahead': 1}, {'tof_guard': 2}, {'tof_guard': 3},
               {'recovery': 0}, {'esc_reverse': 0}, {'recovery': 1, 'esc_reverse': 1}, {'steer_trim_us': 12.0},
               {'throttle_gain': 1.2, 'lookahead': 5}, {'ensemble_j': 2}, {'tof_guard': 1, 'ensemble_j': 0}]
    for k, ch in enumerate(changes):
        plan['params'][int((k + 1) * N / (len(changes) + 1))] = ch
    plan['mode'][int(N * 0.37)] = 4
    plan['mode'][int(N * 0.55)] = 5
if args.pca_fail:
    plan['pca_fail'].update(int(x) for x in args.pca_fail.split(','))
if args.variant == 'toggle':
    plan['params'][int(N * 0.25)] = {'zone_caps': 1}
    plan['params'][int(N * 0.60)] = {'zone_caps': 0}


# ---------------------------------------------------------------- 偽物
class Runner(object):
    def __init__(self, box):
        self.box = box
        self.calls = 0

    def run_numpy(self, **kw):
        self.calls += 1
        i = self.box['i']
        p = preds[i].copy()
        if i in plan['nan']:
            p[3, 1] = np.nan
        return {'actions': ((p - drive.PWM_CENTER) / drive.PWM_SCALE)[None].astype(np.float32)}


class FakePCA(object):
    def __init__(self, box):
        self.box = box
        self.pairs, self.single = [], []
        self.failed = set()

    def set_pair_us(self, sc_, s, tc, t):
        i = self.box['i']
        if i in plan['pca_fail'] and i not in self.failed:
            self.failed.add(i)
            raise OSError('試験の I2C の失敗')
        self.pairs.append((s, t))

    def set_us(self, ch, us):
        self.single.append((ch, us))

    def set_osc_scale(self, scale):
        pass


class FakeRecorder(object):
    recording = False

    def __init__(self):
        self.rows, self.events = [], []

    def add_ai(self, t_ns, data):
        self.rows.append((t_ns, data))

    def add_event(self, kind, data):
        self.events.append((kind, data))
        return True

    def set_mode(self, m):
        pass


def fake_normalize(image, tof_raw, imu_, vehicle, use_command):
    return (np.zeros((3, 4, 4), np.float32), np.zeros((1, 2, 2), np.float32), np.asarray(imu_, np.float32),
            (np.asarray(vehicle, np.float32) - 1500.0) / 500.0)


T0 = min(imu[0][0], veh[0][0], tof[0][0], cam_t[0])
box = {'ns': T0, 'm': 100.0, 'i': 0, 'zt': 0.0}
clock.now_ns = lambda: box['ns']
drive.time = types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter, time=lambda: 1.0,
                                   sleep=lambda s: None, strftime=_time.strftime)
tmpdir = Path(args.out).parent
drive.SETTINGS_PATH = tmpdir / ('settings_%s_%s_%s.json' % (args.log, args.variant, 'new' if NEW else 'old'))
drive.FLOOR_REF_PATH = FLOOR

# 今の木: 区間の速度上限のつなぎの時間を測る(frame・hold・record・view の合計をフレームごとに)・例外を入れる
zone_frame_ms = []
if NEW:
    from jetformer import safety_caps as sc
    for name in ('frame', 'hold', 'record', 'view'):
        orig = getattr(drive._ZoneCaps, name)

        def wrap(self, *a, _o=orig, **kw):
            t0 = _time.perf_counter()
            try:
                return _o(self, *a, **kw)
            finally:
                box['zt'] += (_time.perf_counter() - t0) * 1000.0
        setattr(drive._ZoneCaps, name, wrap)
    if args.zone_fail_at >= 0:
        real_zf = sc.zone_frame
        n_zf = [0]

        def zf(*a, **kw):
            n_zf[0] += 1
            if n_zf[0] == args.zone_fail_at:
                raise ZeroDivisionError('試験の例外(zone_frame)')
            return real_zf(*a, **kw)
        sc.zone_frame = zf
    if args.imu_fail_at >= 0:
        real_zi = sc.zone_imu
        n_zi = [0]

        def zi(*a, **kw):
            n_zi[0] += 1
            if n_zi[0] == args.imu_fail_at:
                raise IndexError('試験の例外(zone_imu)')
            return real_zi(*a, **kw)
        sc.zone_imu = zi


class Feeder(object):
    """cam_evt の代わり。wait() のたびに次の出来事(カメラのフレーム / 0.1 秒の途切れ)まで、センサーの値をバッファへ"""

    def __init__(self, d, bufs, cam_slot):
        self.d = d
        self.tof_buf, self.imu_buf, self.veh_buf = bufs
        self.cam_slot = cam_slot
        self.i = 0
        self.ki = self.kv = self.kt = 0
        self.t = T0
        self.pending_timeout = None
        self.seq = 0
        self.loop_ms = []
        self.t_ret = None
        self.last_kind = None

    def _push(self, upto_imu, upto_veh, upto_tof, frame_i):
        while self.ki < len(imu) and imu[self.ki][0] <= upto_imu:
            t, x = imu[self.ki]
            self.ki += 1
            if frame_i in plan['imu_drop']:
                continue
            if frame_i in plan['imu_swap'] and self.ki < len(imu) and imu[self.ki][0] <= upto_imu:
                t2, x2 = imu[self.ki]
                self.ki += 1
                self.imu_buf.append((t2, x2))             # 時刻が逆戻りする順に届く
            self.imu_buf.append((t, x))
            if frame_i in plan['imu_dup']:
                self.imu_buf.append((t, x))               # 同じサンプルが 2 回
        while self.kv < len(veh) and veh[self.kv][0] <= upto_veh:
            t, dd = veh[self.kv]
            self.kv += 1
            if frame_i in plan['veh_drop']:
                continue
            dd = dict(dd)
            if frame_i in plan['ch3_lost']:
                dd['valid'] = [dd['valid'][0], dd['valid'][1], False]
            self.veh_buf.append((t, dd))
        while self.kt < len(tof) and tof[self.kt][0] <= upto_tof:
            self.tof_buf.append(tof[self.kt])
            self.kt += 1

    def _clock(self, ns):
        box['ns'] = ns
        box['m'] = (ns - T0) / 1e9 + 100.0
        self.d.heartbeat_t = box['m']

    def wait(self, timeout):
        now = _time.perf_counter()
        if self.t_ret is not None:
            self.loop_ms.append(((now - self.t_ret) * 1000.0, box['zt'], self.i, self.last_kind))
        box['zt'] = 0.0
        if self.i >= N:
            self.d._stop.set()
            return False
        nxt = cam_t[self.i]
        # 前の出来事から 0.15 秒より空いていたら 0.1 秒ごとの途切れ(カメラが来ない)を入れる(IMU を位相へ入れ続ける)
        if nxt - self.t > 150_000_000 or (self.i in plan['cam_timeout'] and self.pending_timeout != self.i):
            if self.i in plan['cam_timeout'] and nxt - self.t <= 150_000_000:
                self.pending_timeout = self.i
                t_to = max(self.t + 1_000_000, nxt - 1_000_000)        # 次のフレームの直前に 1 回だけ途切れる
            else:
                t_to = self.t + 100_000_000
            self._push(t_to, t_to, t_to, -1)
            self.t = t_to
            self._clock(t_to + 1_000_000)
            self.last_kind = 'timeout'
            self.t_ret = _time.perf_counter()
            return False
        i = self.i
        box['i'] = i
        for k, ch in plan['params'].items():
            if k == i:
                self.d.set_params(ch)
        if i in plan['mode']:
            self.d.st['mode_id'] = plan['mode'][i]
        self._push(nxt + 2_000_000, nxt + 1_000_000, nxt, i)
        if i not in plan['dup_seq']:
            self.seq += 1
        self.cam_slot[0] = Sample(nxt, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': self.seq})
        self._clock(nxt + (200_000_000 if i in plan['late'] else 8_000_000))
        self.t = nxt
        self.i += 1
        self.last_kind = 'frame'
        self.t_ret = _time.perf_counter()
        return True

    def clear(self):
        pass

    def set(self):
        pass


rec = FakeRecorder()
d = drive.Driver(hub=None, recorder=rec)
d.params = dict(drive.DEFAULT_PARAMS)
d.params.update(BASE)
if NEW:
    d.params['zone_caps'] = int(args.zone)
d.hb_token = 'tok'
d.hb_count = 1
d.st = {'phase': 'driving', 'mode_id': 5, 'run': 'v_replay', 'model': {'command_input': False}}
pca = FakePCA(box)
out = drive._Output(pca, config.PCA9685, REF, lambda: d.params)
bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
cam_slot = [None]
feeder = Feeder(d, bufs, cam_slot)
runner = Runner(box)
t_start = _time.perf_counter()
err = None
try:
    d._loop(d.st, runner, fake_normalize, pca, out, REF, 6, 0.0, cam_slot, feeder, *bufs, args.variant == 'dry')
except Exception as e:      # noqa: BLE001
    import traceback
    traceback.print_exc()
    err = '%s: %s' % (type(e).__name__, e)
wall = _time.perf_counter() - t_start
res = {'tree': str(TREE), 'new': NEW, 'log': args.log, 'variant': args.variant, 'zone': args.zone, 'N': N,
       'pairs': pca.pairs, 'single': pca.single, 'rows': rec.rows, 'events': rec.events, 'err': err,
       'st': {k: v for k, v in d.st.items() if k not in ('model',)}, 'params_end': dict(d.params),
       'runner_calls': runner.calls, 'loop_ms': feeder.loop_ms, 'wall_s': wall,
       'plan_sizes': {k: len(v) for k, v in plan.items()}}
with open(args.out, 'wb') as f:
    pickle.dump(res, f)
print('%s %s %s zone=%d N=%d rows=%d pairs=%d err=%s wall=%.1fs' % ('new' if NEW else 'old', args.log, args.variant,
                                                                    args.zone, N, len(rec.rows), len(pca.pairs), err, wall))
