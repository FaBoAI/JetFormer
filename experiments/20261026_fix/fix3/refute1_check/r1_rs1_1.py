"""指摘 r1-rs1-1 の確かめ(反証の係。直さない・読むだけ・回すだけ)。
本物の drive._loop を偽物のセンサーで回す。カメラが来ない間は、本物の cam_evt.wait(0.1) と同じく
「前のフレームから 0.1 秒たったら False(= hold('カメラの映像が来ていません'))」にする(指摘の台本は 33 ms ごとに False)。
PCA9685 に書いた値を時刻つきで取り、後ろに当たったあとに後退側(中立 + 40 µs より上)を出した秒を数える。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/refute1_check/r1_rs1_1.py
"""
import collections
import json
import sys
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import clock, config, drive          # noqa: E402
from jetformer.sensors.base import Sample           # noqa: E402
import test_launch_drive as TL                      # noqa: E402
from test_drive_caps import FakeRecorder, SeqRunner, fake_normalize   # noqa: E402
from test_launch_drive import C, N, PARAMS, T0, build   # noqa: E402

TMP = ROOT / 'experiments' / '20261026_fix' / 'fix3' / 'tmp_refute1_check'
OUT = Path(__file__).resolve().parent / 'out'


class TimedPCA(object):
    def __init__(self, box):
        self.box, self.w = box, []

    def set_pair_us(self, sc_, s, tc, t):
        self.w.append(((self.box['ns'] - T0) / 1e9, float(s), float(t)))

    def set_us(self, ch, us):
        pass

    def set_osc_scale(self, scale):
        pass


class RealWaitFeeder(TL.Feeder):
    """drop のフレームはカメラが来ない(センサーは来る・時計は進む)。wait は前のフレームから 0.1 秒で False"""

    def _feed(self, f):
        cam_t = f['cam_t']
        while self.k < len(self.imu) and self.imu[self.k][0] <= cam_t + 2_000_000:
            t, g, ax = self.imu[self.k]
            self.imu_buf.append((t, np.array([ax, 0.0, 9.8, 0.0, 0.0, np.deg2rad(g)], np.float32)))
            self.k += 1
        if not f['veh_skip']:
            self.veh_buf.append((cam_t - 3_000_000, dict(f['veh'])))
        self.tof_buf.append((cam_t - 5_000_000, f['tof']))

    def wait(self, timeout):
        t_start = self.box['ns']
        while True:
            if self.i >= len(self.frames):
                self.d._stop.set()
                return False
            f = self.frames[self.i]
            if not f.get('drop'):
                return super().wait(timeout)
            if f['cam_t'] - t_start >= int(timeout * 1e9):      # 0.1 秒の待ちが切れた(このフレームの時刻で)
                self.box['ns'] = t_start + int(timeout * 1e9)
                self.box['m'] = (self.box['ns'] - T0) / 1e9 + 100.0
                self.d.heartbeat_t = self.box['m']
                return False
            self._feed(f)
            self.box['ns'] = f['cam_t']
            self.box['m'] = (f['cam_t'] - T0) / 1e9 + 100.0
            self.d.heartbeat_t = self.box['m']
            self.i += 1


def run(params, frames, imu):
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(drive, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                        time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        np.save(fp, np.full((42, 54), float(TL.FLOOR_MM), np.float32))
        mp.setattr(drive, 'FLOOR_REF_PATH', fp)
        rec = FakeRecorder()
        d = drive.Driver(hub=None, recorder=rec)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = TimedPCA(box)
        out = drive._Output(pca, config.PCA9685, TL.REF, lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        feeder = RealWaitFeeder(d, frames, imu, bufs, cam_slot, box)
        runner = SeqRunner([f['pred'] for f in frames if not f.get('drop')])
        d._loop(d.st, runner, fake_normalize, pca, out, TL.REF, 6, 0.0, cam_slot, feeder, *bufs, False)
        d._stop.clear()
        return pca, rec


def scene(hit_at, drop, after_v=0.0, veh_hold=None):
    """tests/test_rec_hold_keep.py の recovery_scene と同じ形: 3 m/s → 2.0 秒で止まる → リカバリー。2.95 秒から後ろへ 1 m/s。
    hit_at に後ろの衝撃(IMU 横 6G・0.02 秒)。そのあとは after_v(0 = 止まったまま。負 = 後ろへ空転などで eRPM が後ろ向き)"""
    def v(s):
        if s < 2.0:
            return 3.0
        if 2.95 <= s < hit_at:
            return -1.0
        if s >= hit_at:
            return after_v
        return 0.0
    frames, imu = build(5.5, sel=True, v=v, pred=lambda s: (C, N - 150.0) if s < 1.8 else (C, N + 20.0),
                        ax=lambda s: 60.0 if hit_at <= s < hit_at + 0.02 else 0.0,
                        veh_skip=(lambda s: veh_hold[0] <= s < veh_hold[1]) if veh_hold else False)
    for f in frames:
        f['drop'] = bool(drop and drop[0] <= f['s'] < drop[1])
    return frames, imu


def rear_push(pca, rec, hit_at):
    """当たったあと PCA9685 に後退側(> 中立 + 40)を書いていた秒(書いた値は次に書くまで続く)"""
    w = [x for x in pca.w]
    tot, first, last = 0.0, None, None
    for (t0, s0, th0), (t1, _, _) in zip(w, w[1:]):
        if th0 > N + 40.0 and t1 > hit_at:
            a = max(t0, hit_at)
            tot += t1 - a
            first = a if first is None else first
            last = t1
    ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    us = sorted({round(th, 1) for t, s, th in w if t >= hit_at and th > N + 40.0})
    neutral_writes = sum(1 for t, s, th in w if t >= hit_at and abs(th - N) < 1.0)
    return {'push_s': round(tot, 3), 'first': None if first is None else round(first, 3),
            'last': None if last is None else round(last, 3), 'us': us, 'guard_events': ev,
            'neutral_writes_after_hit': neutral_writes}


def main():
    res = {}
    hit = 3.25
    combos = ((0, 0), (0, 1), (1, 0), (1, 1))
    # 1. 指摘の場面: 当たる 0.02 秒前からカメラが来ない(長さを変える)
    for drop_len in (0.0, 0.08, 0.1, 0.134, 0.167, 0.25, 0.5):
        drop = (hit - 0.02, hit - 0.02 + drop_len) if drop_len else None
        for keep, guard in combos:
            frames, imu = scene(hit, drop)
            pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
            res['camdrop=%.3f keep=%d guard=%d' % (drop_len, keep, guard)] = rear_push(pca, rec, hit)
    # 2. カメラの止まりの始まりをずらす(当たりより前・あと)
    for start in (-0.10, -0.05, 0.0, 0.02, 0.05):
        drop = (hit + start, hit + start + 0.2)
        frames, imu = scene(hit, drop)
        pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1), frames, imu)
        res['shift start=%+.2f len=0.2 keep=1 guard=1' % start] = rear_push(pca, rec, hit)
    # 3. 当たったあと車輪が後ろへ空転(eRPM が後ろ向き = 下がっていると数える)なら、早く惰性に移るか
    for av in (-0.3, -0.5):
        frames, imu = scene(hit, (hit - 0.02, hit + 0.23), after_v=av)
        pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1), frames, imu)
        res['camdrop=0.25 after_v=%.1f keep=1 guard=1' % av] = rear_push(pca, rec, hit)
    # 4. 車両の値が来ない中立止め(指摘の A2)
    for keep, guard in combos:
        frames, imu = scene(3.30, None, veh_hold=(3.10, 3.50))
        pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
        res['vehstale 3.10-3.50 hit=3.30 keep=%d guard=%d' % (keep, guard)] = rear_push(pca, rec, 3.30)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'r1_rs1_1.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in res.items():
        print(k, v)


if __name__ == '__main__':
    main()
