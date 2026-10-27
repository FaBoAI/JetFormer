"""反証 1 回目・safety(直す 3 回目のあと、2026-10-26 夜): 本物の drive._loop を偽物のセンサーで回して、
今日の補助(発進の前進・StartKick・リカバリー・後ろの見張り・rec_hold_keep・launch_no_kick)の組み合わせで
「まちがった出力」が出る場面を探す。装置は開かない。直さない(読むだけ・回すだけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/refute1_safety/probe_refute1.py

場面:
  A  後退の途中に後ろへ当たり、その衝撃と同じころにカメラが 0.25 秒止まる(衝撃の CRC。中立止め)。
     rec_hold_keep = 1(+ rec_back_guard = 1)は残したリカバリーが当たりを見落として後ろの壁を押し続けるか
  A2 同じ場面を「車両の値が来ない」中立止めで(反証 fix3 の試験 rec_back_hold の中立止めを長くしたもの)
  B  launch_boost = 1・launch_no_kick = 1 で始めた区間で、リカバリー中止のあとに launch_no_kick を 1 → 0(走行中にできる)。
     StartKick が中止の中立を上書きして前へ出すか
  B2 start_kick を 0 → 1(走行中に画面のチェックでできる)で同じ
  C  走っている車が長い中立止め(カメラ 1.2 秒)で止まったあと、モデルが弱い(動き出せない)。rec_hold_keep = 1 は
     衝突なしでリカバリー(後退)を始めるか
"""
import collections
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A          # noqa: E402
from jetformer import clock, config, drive  # noqa: E402
from jetformer.sensors.base import Sample  # noqa: E402
import test_launch_drive as TL             # noqa: E402
from test_drive_caps import FakePCA, FakeRecorder, SeqRunner, fake_normalize   # noqa: E402
from test_launch_drive import C, N, PARAMS, T0, build, tof_near   # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix3' / 'tmp_refute1_safety'


class CamDropFeeder(TL.Feeder):
    """フレームの cam_drop が True なら、センサーの値は入れ・時計は進め、カメラは来ない(wait が False = 0.1 秒の待ちが切れた)"""

    def wait(self, timeout):
        if self.i >= len(self.frames):
            self.d._stop.set()
            return False
        f = self.frames[self.i]
        if not f.get('cam_drop'):
            return super().wait(timeout)
        if self.hook is not None:
            self.hook(self.i, self.d)
        cam_t = f['cam_t']
        while self.k < len(self.imu) and self.imu[self.k][0] <= cam_t + 2_000_000:
            t, g, ax = self.imu[self.k]
            self.imu_buf.append((t, np.array([ax, 0.0, 9.8, 0.0, 0.0, np.deg2rad(g)], np.float32)))
            self.k += 1
        if not f['veh_skip']:
            self.veh_buf.append((cam_t - 3_000_000, dict(f['veh'])))
        self.tof_buf.append((cam_t - 5_000_000, f['tof']))
        self.box['ns'] = cam_t + 8_000_000
        self.box['m'] = (cam_t - T0) / 1e9 + 100.0
        self.d.heartbeat_t = self.box['m']
        self.i += 1
        return False


def run(params, frames, imu, floor=True, hook=None):
    """tests/test_launch_drive.run と同じ(カメラが来ないフレームを足しただけ)→ (driver, pca, rec)"""
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        import time as _time
        mp.setattr(drive, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                        time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        if floor:
            np.save(fp, np.full((42, 54), float(TL.FLOOR_MM), np.float32))
        mp.setattr(drive, 'FLOOR_REF_PATH', fp if floor else tmp_path / 'no_floor_ref.npy')
        rec = FakeRecorder()
        d = drive.Driver(hub=None, recorder=rec)
        d.params = dict(params)
        d.hb_token = 'tok'
        d.hb_count = 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = FakePCA()
        out = drive._Output(pca, config.PCA9685, TL.REF, lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        feeder = CamDropFeeder(d, frames, imu, bufs, cam_slot, box, hook)
        runner = SeqRunner([f['pred'] for f in frames if not f.get('cam_drop')])
        d._loop(d.st, runner, fake_normalize, pca, out, TL.REF, 6, 0.0, cam_slot, feeder, *bufs, False)
        d._stop.clear()
        return d, pca, rec


def tl(frames, rec):
    s_of = {f['cam_t']: f['s'] for f in frames}
    return [(round(s_of[t], 3), int(r['assist']), bool(r['hold']), round(float(r['out'][1]), 1), bool(r['select']))
            for t, r in rec.rows]


def mark(frames, key, a, b):
    for f in frames:
        if a <= f['s'] < b:
            f[key] = True
    return frames


# ---------------------------------------------------------------- A: 後ろの当たりとカメラの止まりが重なる
def scene_a(hit_at=3.25, drop=(3.22, 3.47), veh_hold=None):
    """recovery_scene(tests/test_rec_hold_keep.py)と同じ: 3 m/s → 2.0 秒で止まる → リカバリー。2.95 秒から後ろへ 1 m/s、
    hit_at に後ろの衝撃(横 6G)で止まる(そのあと 0)。drop の間はカメラが来ない(中立止め)。veh_hold は車両の値が来ない区切り"""
    def v(s):
        if s < 2.0:
            return 3.0
        if 2.95 <= s < hit_at:
            return -1.0
        return 0.0
    frames, imu = build(5.5, sel=True, v=v, pred=lambda s: (C, N - 150.0) if s < 1.8 else (C, N + 20.0),
                        ax=lambda s: 60.0 if hit_at <= s < hit_at + 0.02 else 0.0,
                        veh_skip=(lambda s: veh_hold[0] <= s < veh_hold[1]) if veh_hold else False)
    if drop:
        mark(frames, 'cam_drop', *drop)
    return frames, imu


def summarize_rear(frames, rec, pca, hit_at):
    rows = tl(frames, rec)
    rev_after = [x for x in rows if x[0] >= hit_at and not x[2] and x[3] > N + 40.0]
    guard_ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    return {'reverse_frames_after_rear_hit': len(rev_after),
            'reverse_s_after_rear_hit': round(len(rev_after) / 30.0, 2),
            'first_last': [rev_after[0][0], rev_after[-1][0]] if rev_after else None,
            'reverse_us': sorted({x[3] for x in rev_after}),
            'guard_events': guard_ev,
            'holds_rows': sum(1 for x in rows if x[2]),
            'pca_neutral_writes_in_drop': None}


def probe_a():
    res = {}
    hit = 3.25
    for name, kw in (('A_camdrop', dict(hit_at=hit, drop=(3.22, 3.47))),
                     ('A2_vehstale', dict(hit_at=3.30, drop=None, veh_hold=(3.10, 3.50)))):
        h = kw['hit_at']
        for keep, guard in ((0, 0), (0, 1), (1, 1)):
            frames, imu = scene_a(**kw)
            d, pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
            r = summarize_rear(frames, rec, pca, h)
            res['%s keep=%d guard=%d' % (name, keep, guard)] = r
        # 同じ場面で当たりのフレームが見えるとき(中立止めなし)の基準
        frames, imu = scene_a(hit_at=h, drop=None)
        d, pca, rec = run(dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1), frames, imu)
        res['%s no_hold keep=1 guard=1' % name] = summarize_rear(frames, rec, pca, h)
    return res


# ---------------------------------------------------------------- B: 中止のあとの StartKick
def scene_b(dur=24.0, thr=N - 30.0):
    """人側 1 秒 → AI 側(走っている = 発進の前進は moving で始めない)。1.0〜2.0 秒は 2 m/s、2.0 秒に横から当たって止まったまま
    (前の ToF には写らない: DE の後輪・横の当たり)。モデルは弱い前進 thr"""
    return build(dur, sel=lambda s: s >= 1.0, v=lambda s: 2.0 if 1.0 <= s < 2.0 else 0.0, pred=(C, thr),
                 ax=lambda s: 45.0 if 2.0 <= s < 2.02 else 0.0)


def probe_b():
    res = {}
    base_frames, base_imu = scene_b()
    for name, p0, change in (
            ('B_launch_no_kick_1to0', dict(PARAMS, launch_boost=1, launch_no_kick=1, start_kick=1), {'launch_no_kick': 0}),
            ('B2_start_kick_0to1', dict(PARAMS, launch_boost=0, start_kick=0), {'start_kick': 1})):
        for keep in (0, 1):
            p = dict(p0, recovery=1, rec_back_guard=keep, rec_hold_keep=keep)
            frames, imu = scene_b()
            _, _, rec0 = run(p, frames, imu)
            rows0 = tl(frames, rec0)
            g0 = next((s for s, k, h, o, sel in rows0 if k == A.ASSIST_GIVEUP), None)
            if g0 is None:
                res['%s keep=%d' % (name, keep)] = {'giveup': None}
                continue
            at = g0 + 1.0
            idx = next(i for i, f in enumerate(frames) if f['s'] >= at)

            def hook(i, d, idx=idx, change=change):
                if i == idx:
                    d.set_params(change)
            frames, imu = scene_b()
            d, pca, rec = run(p, frames, imu, hook=hook)
            rows = tl(frames, rec)
            after = [x for x in rows if x[0] >= at and not x[2]]
            kick = [x for x in after if x[1] == A.ASSIST_START]
            res['%s keep=%d' % (name, keep)] = {
                'giveup_at_s': g0, 'changed_at_s': round(frames[idx]['s'], 3), 'change': change,
                'kick_frames_after_change': len(kick), 'kick_s': round(len(kick) / 30.0, 2),
                'kick_out_us': sorted({x[3] for x in kick}),
                'first_kick': kick[0] if kick else None,
                'giveup_rows_before_change': sum(1 for x in rows if g0 <= x[0] < at and x[1] == A.ASSIST_GIVEUP),
                'giveup_rows_after_kick': sum(1 for x in after if x[1] == A.ASSIST_GIVEUP),
                'kinds_after_change_first_10': [x[1] for x in after[:10]],
                'without_change_kick_after': sum(1 for x in rows0 if x[0] >= at and x[1] == A.ASSIST_START),
                'without_change_kind_after': sorted({x[1] for x in rows0 if x[0] >= at and not x[2]})}
    return res


# ---------------------------------------------------------------- C: 長い中立止めで止まった車
def scene_c(drop=(3.0, 4.2), thr_after=N - 50.0, dur=8.0):
    """人側 1 秒 → AI 側。3 m/s で走る(armed)。drop の間カメラが来ない(中立止め)。その間に惰性で 0 へ(3.0〜4.0 秒)。
    そのあとは止まったまま(モデルは弱い前進 thr_after で動き出せない)。衝突なし・前は空き"""
    def v(s):
        if s < 1.0:
            return 0.0
        if s < 3.0:
            return 3.0
        if s < 4.0:
            return 3.0 * (4.0 - s)
        return 0.0
    frames, imu = build(dur, sel=lambda s: s >= 1.0, v=v, pred=lambda s: (C, N - 200.0) if s < 3.0 else (C, thr_after))
    if drop:
        mark(frames, 'cam_drop', *drop)
    return frames, imu


def probe_c():
    res = {}
    for drop in ((3.0, 4.2), (3.0, 3.5)):
        for keep in (0, 1):
            frames, imu = scene_c(drop=drop)
            p = dict(PARAMS, recovery=1, start_kick=1, launch_boost=0, rec_back_guard=1, rec_hold_keep=keep)
            d, pca, rec = run(p, frames, imu)
            rows = tl(frames, rec)
            after = [x for x in rows if x[0] >= drop[1] and not x[2]]
            rev = [x for x in after if x[3] > N + 40.0]
            recs = [x for x in after if x[1] == A.ASSIST_RECOVERY]
            res['C drop=%s keep=%d' % (list(drop), keep)] = {
                'recovery_frames': len(recs), 'first_recovery_s': recs[0][0] if recs else None,
                'reverse_side_frames': len(rev), 'reverse_side_s': round(len(rev) / 30.0, 2),
                'reverse_us': sorted({x[3] for x in rev}),
                'escape_forward_frames': sum(1 for x in recs if x[3] < N - 100.0),
                'impacts': 0}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {'A': probe_a(), 'B': probe_b(), 'C': probe_c()}
    (OUT / 'probe_refute1.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
