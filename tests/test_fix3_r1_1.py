"""直す 1 回目・1 組目(2026-10-26 fix3。反証 1 回目の指摘 r1-rs1-1・r1-rs1-2)の試験。走行ループ・補助の側。

  r1-rs1-1: rec_hold_keep = 1・rec_back_guard = 1 のとき、リカバリーの後退の途中に 0.1 秒より長い中立止めをはさんで後ろへ当たると、
    Recovery が衝撃を見落とし、明けたあと 'reverse' が後ろの物を最大 約 0.8 秒押した(中立止めの間は step が呼ばれず、impact_g は
    処理したフレームの直近 0.15 秒の IMU だけ)。直し: 残した中立止めのあとの最初のフレームで、その間の IMU の水平 G の最大を
    hold_g として Recovery に渡し、後退側・下がる惰性の段で 2.5G 以上ならすぐ前進(出来事 rear_impact_hold)。
  r1-rs1-2: 走行中に launch_no_kick を 1 → 0(または start_kick を 0 → 1)にすると、StartKick の 2 秒の窓が AI の区間の途中で
    新しく開き、リカバリーの中止・後退の途中を上書きして前進 80 µs を 2 秒出した。直し: launch_no_kick で「出さない」になった
    区間は、途中で 0 にしても出さない(CH3 を人側にするまで)。start_kick は走行中に 0 → 1 にできない(set_params が断る)。

OFF(rec_hold_keep = 0・launch_no_kick が走行中ずっと同じ・start_kick を走行中に ON にしない)は、直す前の写し
tests/fixtures/fix3_r1_before/(drive.py・assist.py。sha256 は SHA256SUMS)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。
実機・GPU・torch は使わない(偽物は tests/test_launch_drive.py・test_drive_caps.py と同じ。カメラが来ない間は本物の
cam_evt.wait(0.1) と同じく、前のフレームから 0.1 秒で False を返す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix3_r1_1.py
"""
import collections
import hashlib
import importlib.util
import random
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A
from jetformer import clock, config, drive
from jetformer.sensors.base import Sample

from test_drive_caps import FakeRecorder, SeqRunner, assert_rows_equal_before, fake_normalize
from test_launch_drive import C, N, PARAMS, REF, T0, FLOOR_MM, Feeder, build, run
from test_rec_hold_keep import fuzz_scene

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix3_r1_before'
DT = 1 / 30.0
HIT = 3.25


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sums():
    return dict(reversed(line.split()) for line in (FIX / 'SHA256SUMS').read_text().splitlines())


_BEFORE = {}


def load_before():
    """直す前の assist.py・drive.py(写し。2026-10-26 21:50。fix3 の safety・ui の直しのあと)をモジュールにする"""
    if _BEFORE:
        return _BEFORE['a'], _BEFORE['d']
    s = sums()
    for name in ('drive.py', 'assist.py'):
        assert sha(FIX / name) == s[name], name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix3r1', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix3r1', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    assert 'hold_g' not in amod.Recovery.step.__code__.co_varnames            # 写しは直す前
    _BEFORE.update(a=amod, d=mod)
    return amod, mod


# ---------------------------------------------------------------- カメラが来ない場面の走行ループ
class TimedPCA(object):
    """PCA9685 の偽物。書いた値を時刻(場面の始まりからの秒)つきで残す"""

    def __init__(self, box):
        self.box, self.w = box, []

    def set_pair_us(self, sc_, s, tc, t):
        self.w.append((round((self.box['ns'] - T0) / 1e9, 6), float(s), float(t)))

    def set_us(self, ch, us):
        self.w.append((round((self.box['ns'] - T0) / 1e9, 6), 'single', float(us)))

    def set_osc_scale(self, scale):
        pass


class DropFeeder(Feeder):
    """f['drop'] のフレームはカメラが来ない(IMU・車両の値・ToF は来る・時計は進む)。wait は、前のフレームから timeout 秒で
    False(本物の cam_evt.wait(0.1) と同じ = hold('カメラの映像が来ていません'))"""

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
            if f['cam_t'] - t_start >= int(timeout * 1e9):          # 0.1 秒の待ちが切れた(このフレームの時刻で)
                self.box['ns'] = t_start + int(timeout * 1e9)
                self.box['m'] = (self.box['ns'] - T0) / 1e9 + 100.0
                self.d.heartbeat_t = self.box['m']
                return False
            self._feed(f)
            self.box['ns'] = f['cam_t']
            self.box['m'] = (f['cam_t'] - T0) / 1e9 + 100.0
            self.d.heartbeat_t = self.box['m']
            self.i += 1


def run_wait(mod, monkeypatch, tmp_path, params, frames, imu, hook=None):
    """mod(drive か直す前の写し)の _loop を、カメラが来ない区間つきで回す → (driver, pca, recorder)"""
    box = {'ns': T0, 'm': 100.0}
    monkeypatch.setattr(clock, 'now_ns', lambda: box['ns'])
    monkeypatch.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                           time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
    monkeypatch.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    fp = tmp_path / 'floor_ref.npy'
    np.save(fp, np.full((42, 54), float(FLOOR_MM), np.float32))
    monkeypatch.setattr(mod, 'FLOOR_REF_PATH', fp)
    rec = FakeRecorder()
    d = mod.Driver(hub=None, recorder=rec)
    d.params = dict(params)
    d.hb_token, d.hb_count = 'tok', 1
    d.st = {'phase': 'driving', 'mode_id': 4}
    pca = TimedPCA(box)
    out = mod._Output(pca, config.PCA9685, REF, lambda: d.params)
    bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
    cam_slot = [None]
    feeder = DropFeeder(d, frames, imu, bufs, cam_slot, box, hook)
    runner = SeqRunner([f['pred'] for f in frames if not f.get('drop')])
    d._loop(d.st, runner, fake_normalize, pca, out, REF, 6, 0.0, cam_slot, feeder, *bufs, False)
    d._stop.clear()
    return d, pca, rec


def rear_scene(hit_at=HIT, drop=None, veh_hold=None, after_v=0.0):
    """反証の場面(tests/test_rec_hold_keep.py の recovery_scene と同じ形): 3 m/s → 2.0 秒で止まる → リカバリー。2.95 秒から後ろへ
    1 m/s(見張りの back_early)。hit_at に後ろの衝撃(IMU 横 6G・0.02 秒)。そのあとは after_v。drop = (始まり, 終わり) はカメラが
    来ない。veh_hold = (始まり, 終わり) は車両の値が来ない。
    モデルは 1.8〜2.0 秒ブレーキ側(中立 + 200)、そのあと中立の帯。止まる前のブレーキ側は 2026-10-27 fix7 safety s4-a で足した
    (ESC が後退を待っていた 10/23 の形。'brake' の段で後ろへ動くのはこの形だけ。tests/test_recovery_back_guard.recovery_scene の説明)"""
    def v(s):
        if s < 2.0:
            return 3.0
        if 2.95 <= s < hit_at:
            return -1.0
        return after_v if s >= hit_at else 0.0
    frames, imu = build(5.5, sel=True, v=v, pred=lambda s: (C, N - 150.0) if s < 1.8 else ((C, N + 200.0) if s < 2.0 else (C, N + 20.0)),
                        ax=lambda s: 60.0 if hit_at <= s < hit_at + 0.02 else 0.0,
                        veh_skip=(lambda s: veh_hold[0] <= s < veh_hold[1]) if veh_hold else False)
    for f in frames:
        f['drop'] = bool(drop and drop[0] <= f['s'] < drop[1])
    return frames, imu


def rear_push(pca, rec, hit_at, until=99.0):
    """当たったあと(until 秒まで): PCA9685 に後退側(中立 + 40 µs より上)を新しく書いたフレームの数と、後退側が出ていた秒(前に
    書いた値は次に書くまで続く。中立止めの中立を書くまでの前の値も数える)"""
    w = [x for x in pca.w if x[1] != 'single' and x[0] < until]
    tot = 0.0
    for (t0, _, th0), (t1, _, _) in zip(w, w[1:]):
        if th0 > N + 40.0 and t1 > hit_at:
            tot += t1 - max(t0, hit_at)
    return {'frames': sum(1 for t, s, th in w if t >= hit_at and th > N + 40.0), 'push_s': round(tot, 3),
            'events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}


def same_out(a, b):
    (_, pca0, rec0), (_, pca1, rec1) = a, b
    assert pca0.w == pca1.w
    assert_rows_equal_before(rec0.rows, rec1.rows)
    assert rec0.events == rec1.events


# ================================================================ 0. 写し・既定値
def test_before_copies_and_defaults_unchanged():
    s = sums()
    assert s['drive.py'].startswith('ecb1ff62d04e') and s['assist.py'].startswith('acbf702ad197')     # safety.md 1 章の「後」
    assert s['app.js'].startswith('07c93783ef40') and s['index.html'].startswith('664edbde6c3c')      # ui.md 9 章の「後」
    for name in ('app.js', 'index.html'):
        assert sha(FIX / name) == s[name], name
    amod, before = load_before()
    assert drive.DEFAULT_PARAMS == before.DEFAULT_PARAMS                 # 既定値は 1 つも変えていない(新しい項目も無い)
    assert drive.PARAM_LIMITS == before.PARAM_LIMITS and drive.RESET_ON_BOOT == before.RESET_ON_BOOT
    assert drive.START_KICK_TTC_S == before.START_KICK_TTC_S
    for k in ('IMPACT_G', 'BACK_RECENT_S', 'MAX_REV_TOTAL_S', 'ESCAPE_US', 'ESCAPE_S'):
        assert getattr(A.Recovery, k) == getattr(amod.Recovery, k)
    assert (A.StartKick.KICK_US, A.StartKick.MAX_S, A.StartKick.MOVE_ERPM) == (80.0, 2.0, -800)


# ================================================================ 1. r1-rs1-1 中立止めの間の後ろの衝撃
@pytest.mark.parametrize('drop_len', [0.134, 0.167, 0.25, 0.5])
def test_rs1_1_camera_gap_rear_hit_no_reverse_after(monkeypatch, tmp_path, drop_len):
    """指摘の場面 A: 当たる 0.02 秒前からカメラが drop_len 秒来ない。keep = 1・guard = 1 で、当たったあとに後退側を書いたフレームは 0。
    出ていた秒は中立止めの中立を書くまでの前の値だけ(0.1 秒の待ちが切れるまで。避けられない)。直す前は 0.4 秒より長く押した"""
    _, before = load_before()
    frames, imu = rear_scene(drop=(HIT - 0.02, HIT - 0.02 + drop_len))
    p = dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1)
    now = rear_push(*run_wait(drive, monkeypatch, tmp_path, p, frames, imu)[1:], HIT)
    old = rear_push(*run_wait(before, monkeypatch, tmp_path, p, frames, imu)[1:], HIT)
    assert old['frames'] > 10 and old['push_s'] > 0.4 and 'rear_impact' not in old['events']      # 指摘の再現(直す前)
    assert now['frames'] == 0 and now['push_s'] <= 0.11
    assert 'rear_impact_hold' in now['events']


@pytest.mark.parametrize('start', [-0.10, -0.05, 0.0, 0.02, 0.05])
def test_rs1_1_camera_gap_shifted(monkeypatch, tmp_path, start):
    """カメラが 0.2 秒来ない区間の始まりをずらす(当たりの −0.10〜+0.05 秒)。どれも当たったあとの後退側は 0 フレーム"""
    frames, imu = rear_scene(drop=(HIT + start, HIT + start + 0.2))
    p = dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1)
    r = rear_push(*run_wait(drive, monkeypatch, tmp_path, p, frames, imu)[1:], HIT)
    assert r['frames'] == 0 and r['push_s'] <= 0.11
    assert {'rear_impact', 'rear_impact_hold'} & set(r['events'])


def test_rs1_1_vehicle_stale_hold(monkeypatch, tmp_path):
    """指摘の場面 A2: 車両の値が来ない中立止め(3.10〜3.50 秒)の間に当たる(3.30 秒)。直す前は 0.63 秒押した"""
    _, before = load_before()
    frames, imu = rear_scene(hit_at=3.30, veh_hold=(3.10, 3.50))
    p = dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1)
    now = rear_push(*run_wait(drive, monkeypatch, tmp_path, p, frames, imu)[1:], 3.30)
    old = rear_push(*run_wait(before, monkeypatch, tmp_path, p, frames, imu)[1:], 3.30)
    assert old['frames'] > 10 and old['push_s'] > 0.5
    assert now['frames'] == 0 and now['push_s'] == 0.0 and 'rear_impact_hold' in now['events']


@pytest.mark.parametrize('keep,guard', [(0, 0), (0, 1), (1, 0)])
@pytest.mark.parametrize('drop_len,veh_hold', [(0.0, None), (0.134, None), (0.25, None), (0.5, None), (0.0, (3.10, 3.50))])
def test_rs1_1_off_bit_identical(monkeypatch, tmp_path, keep, guard, drop_len, veh_hold):
    """rec_hold_keep が効かない組(keep = 0、または見張り OFF)は、直す前の写しとビット単位で同じ"""
    _, before = load_before()
    hit = 3.30 if veh_hold else HIT
    frames, imu = rear_scene(hit_at=hit, drop=(HIT - 0.02, HIT - 0.02 + drop_len) if drop_len else None, veh_hold=veh_hold)
    p = dict(PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep)
    same_out(run_wait(drive, monkeypatch, tmp_path, p, frames, imu), run_wait(before, monkeypatch, tmp_path, p, frames, imu))


@pytest.mark.parametrize('drop_len', [0.0, 0.08, 0.1])
def test_rs1_1_short_or_no_gap_unchanged_with_keep(monkeypatch, tmp_path, drop_len):
    """0.1 秒以下の止まり・止まりなし(見張りが前から見える場面)は、keep = 1・guard = 1 でも直す前と同じ(rear_impact のまま)"""
    _, before = load_before()
    frames, imu = rear_scene(drop=(HIT - 0.02, HIT - 0.02 + drop_len) if drop_len else None)
    p = dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=1)
    a = run_wait(drive, monkeypatch, tmp_path, p, frames, imu)
    same_out(a, run_wait(before, monkeypatch, tmp_path, p, frames, imu))
    assert 'rear_impact' in rear_push(*a[1:], HIT)['events']


def _fuzz(seed):
    """でたらめな場面(反証 2 回目の fuzz_loop の場面)に、カメラが来ない区間・車両の値が来ない区間と、その中の衝撃を足す"""
    rng = random.Random(1000 + seed)
    frames, imu = fuzz_scene(seed)
    dur = frames[-1]['s']
    wins = []
    for _ in range(rng.randint(1, 5)):
        a = rng.uniform(0.6, dur - 0.7)
        wins.append((a, a + rng.uniform(0.12, 0.5), rng.random() < 0.5))
    for f in frames:
        f['drop'] = any(a <= f['s'] < b for a, b, cam in wins if cam)
        if any(a <= f['s'] < b for a, b, cam in wins if not cam):
            f['veh_skip'] = True
    hits = [rng.uniform(a, b) for a, b, _ in wins if rng.random() < 0.7]
    imu = [(t, g, 50.0 if any(h <= (t - T0) / 1e9 < h + 0.02 for h in hits) else ax) for t, g, ax in imu]
    return frames, imu


def _rec_fuzz(seed):
    """リカバリーの途中を狙ったでたらめな場面: 走る → 止まる → リカバリー → 後ろへ(見張りの back_early)→ 後ろの衝撃。
    当たりの近くに中立止め(カメラか車両の値が来ない。0.12〜0.5 秒)を 1 つ、ほかにも 0〜2 か所。当たったあとは止まるか少し後ろへ"""
    rng = random.Random(5000 + seed)
    t_stop = rng.uniform(1.6, 2.4)
    back_from = t_stop + rng.uniform(0.85, 1.3)
    hit = back_from + rng.uniform(0.05, 0.5)
    after_v = rng.choice([0.0, 0.0, -0.3])
    thr_after = rng.choice([N + 20.0, N, N - 30.0])

    def v(s):
        if s < t_stop:
            return 3.0
        if back_from <= s < hit:
            return -1.0
        return after_v if s >= hit else 0.0
    frames, imu = build(6.0, sel=True, v=v, pred=lambda s: (C, N - 150.0) if s < t_stop - 0.2 else (C, thr_after),
                        ax=lambda s: 60.0 if hit <= s < hit + 0.02 else 0.0)
    a = hit - rng.uniform(0.0, 0.1)
    wins = [(a, a + rng.uniform(0.12, 0.5), rng.random() < 0.6)]
    for _ in range(rng.randint(0, 2)):
        b = rng.uniform(0.6, 5.0)
        wins.append((b, b + rng.uniform(0.12, 0.4), rng.random() < 0.5))
    for f in frames:
        f['drop'] = any(x <= f['s'] < y for x, y, cam in wins if cam)
        if any(x <= f['s'] < y for x, y, cam in wins if not cam):
            f['veh_skip'] = True
    return frames, imu


def _fuzz_params(rng, keep, guard):
    return dict(PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep, start_kick=rng.choice([0, 1]),
                launch_boost=rng.choice([0, 1]), launch_no_kick=rng.choice([0, 1]), tof_guard=rng.choice([1, 2, 3]),
                esc_reverse=rng.choice([0, 1]), throttle_gain=rng.choice([0.8, 1.0, 1.2]))


FUZZ = [('loop', s) for s in range(6)] + [('rec', s) for s in range(10)]


def _without_hold_since(step):
    """Recovery.step から hold_since・hold_back_s を外す(直す 2 回目の前の形。2026-10-27 fix3 r2-r2-rs-1・rs-2)"""
    def wrapped(self, *a, **kw):
        kw.pop('hold_since', None)
        kw.pop('hold_back_s', None)
        return step(self, *a, **kw)
    return wrapped


def _scene(kind, seed):
    return _fuzz(seed) if kind == 'loop' else _rec_fuzz(seed)


def _without_esc_rev(step):
    """Recovery.step から esc_rev を外す(2026-10-27 fix7 safety s4-a の前の形。drive の EscMayReverse が見る「ESC が後退で
    走らせていたかもしれないか」を渡さない = 分からない)。でたらめな開いたループの場面は、書いた値と関係なく車を後ろへ動かす
    (前進のあとの 'brake' でも後ろへ動く。このESCでは起きない)ので、r1-rs1-1 の直しだけを前と同じに確かめるときに使う。
    s4-a は tests/test_rec_fix7_safety.py が見る"""
    def wrapped(self, *a, **kw):
        kw.pop('esc_rev', None)
        return step(self, *a, **kw)
    return wrapped


@pytest.mark.parametrize('kind,seed', FUZZ)
def test_rs1_1_fuzz_off_identical_and_on_changes_only_at_rear_impact_hold(monkeypatch, tmp_path, kind, seed):
    """でたらめな場面 16 本(反証 2 回目の fuzz_loop の場面 6・リカバリーの途中を狙った場面 10): keep = 0・見張り OFF は直す前と
    全く同じ。keep = 1・guard = 1 は、rear_impact_hold が出るまでは直す前と同じで、出なければ最後まで同じ(直しが効くのは、残した
    中立止めの間に 2.5G 以上が来たときだけ)。出たら、そのフレームは前進。
    今の版は fix7 safety s4-a の esc_rev を外して流す(_without_esc_rev。開いたループの場面は書いた値と関係なく後ろへ動かすため)"""
    _, before = load_before()
    rng = random.Random(seed)
    frames, imu = _scene(kind, seed)
    for keep, guard in ((0, 1), (1, 0), (0, 0)):
        p = _fuzz_params(rng, keep, guard)
        with monkeypatch.context() as m:       # fix7 s4-a の esc_rev は外して照らす(_without_esc_rev)
            m.setattr(A.Recovery, 'step', _without_esc_rev(A.Recovery.step))
            now = run_wait(drive, monkeypatch, tmp_path, p, frames, imu)
        same_out(now, run_wait(before, monkeypatch, tmp_path, p, frames, imu))
    p = _fuzz_params(rng, 1, 1)
    # 2026-10-27 fix3 直す 2 回目 r2-r2-rs-1・rs-2: 残した中立止めのあとの hold_since・hold_back_s の直し(段の時間を止める・下がった
    # 時間・長い中立止めは中止)は tests/test_fix3_r2_1.py で確かめる。ここでは hold_since を渡さない形にして、r1-rs1-1 の直し
    # (rear_impact_hold)だけが前と違うことを、前と同じに確かめる
    with monkeypatch.context() as m:
        m.setattr(A.Recovery, 'step', _without_hold_since(_without_esc_rev(A.Recovery.step)))
        pn_rn = run_wait(drive, monkeypatch, tmp_path, p, frames, imu)
    (_, pn, rn), (_, po, ro) = pn_rn, run_wait(before, monkeypatch, tmp_path, p, frames, imu)
    hits = [i for i, e in enumerate(rn.events) if e[0] == 'recovery_guard' and e[1]['what'] == 'rear_impact_hold']
    if not hits:
        assert pn.w == po.w and rn.events == ro.events
        assert_rows_equal_before(ro.rows, rn.rows)
        return
    t_us = rn.events[hits[0]][1]['cam_epoch_us']
    first = next(i for i, (t, r) in enumerate(rn.rows) if int(clock.to_epoch_us(t)) == t_us)
    assert_rows_equal_before(ro.rows[:first], rn.rows[:first])                  # それまでは同じ
    assert rn.rows[first][1]['assist'] == A.ASSIST_RECOVERY and rn.rows[first][1]['out'][1] < N      # 当たったら前へ


def test_rs1_1_fuzz_covers_rear_impact_hold(monkeypatch, tmp_path):
    """上のでたらめな場面のどれかで、直しの道(rear_impact_hold)を通っている(試験が空振りしていない)。今の版は上と同じく
    fix7 safety s4-a の esc_rev を外して流す"""
    _, before = load_before()
    n_scenes, pushes, old_pushes = 0, [], []
    for kind, seed in FUZZ:
        if kind != 'rec':
            continue
        rng = random.Random(seed)
        frames, imu = _scene(kind, seed)
        for keep, guard in ((0, 1), (1, 0), (0, 0)):
            _fuzz_params(rng, keep, guard)
        p = _fuzz_params(rng, 1, 1)
        with monkeypatch.context() as m:       # fix7 s4-a の esc_rev は外す(_without_esc_rev。開いたループの場面のため)
            m.setattr(A.Recovery, 'step', _without_esc_rev(A.Recovery.step))
            _, pca, rec = run_wait(drive, monkeypatch, tmp_path, p, frames, imu)
        ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
        if 'rear_impact_hold' in ev:
            n_scenes += 1
        hit = next(f['s'] for f in frames if any(abs(a - 60.0) < 1e-9 for t, g, a in imu
                                                     if f['cam_t'] - 34_000_000 < t <= f['cam_t']))
        # 当たってから 1 秒の内(前進のあとの見つけ直しは、前進 0.5 秒・間 1.0 秒・止まって 0.8 秒のあと = 2.3 秒より後。
        # 開いたループの場面は前進しても動かないので、そのあとの次のリカバリーは数えない)
        pushes.append(rear_push(pca, rec, hit, until=hit + 1.0)['frames'])
        _, pca0, rec0 = run_wait(before, monkeypatch, tmp_path, p, frames, imu)
        old_pushes.append(rear_push(pca0, rec0, hit, until=hit + 1.0)['frames'])
    assert n_scenes >= 3                                                       # 直しの道を通っている
    assert sum(1 for x in old_pushes if x > 5) >= 2, old_pushes                # 直す前は後ろを押す場面がある(指摘の再現)
    assert all(x == 0 for x in pushes), pushes                                 # どの場面も、当たったあとに後退側を書かない


# ---------------------------------------------------------------- Recovery の形で
def _to_reverse(r, t=0.0):
    """Recovery を 'reverse' まで進める(止まって見つける → ブレーキ → 待つ → 中立 → 後退)。戻り値 次の時刻"""
    for _ in range(40):
        r.step((C, N - 150.0), -5000, t); t += DT       # 前へ走る(armed)
    while r.phase != 'reverse':
        r.step((C, N + 20.0), 0, t); t += DT
        assert t < 10.0
    return t


def test_rs1_1_recovery_hold_g_only_in_backing_phases():
    r = A.Recovery((C, N))
    t = _to_reverse(r)
    o, k = r.step((C, N + 20.0), 0, t, hold_g=2.4)          # 2.5G 未満は今まで通り(後退のまま)
    assert o[1] > N + 40 and r.phase == 'reverse'
    o, k = r.step((C, N + 20.0), 0, t + DT, hold_g=3.0)
    assert o[1] < N and r.phase == 'escape' and r.guard_event[0] == 'rear_impact_hold'
    # 中立止めの前に手順が始まっていなかった(このフレームで見つけた)ときは見ない(前の物に当たった衝撃かもしれない)
    r2, t2 = A.Recovery((C, N)), 0.0
    for _ in range(40):
        r2.step((C, N - 150.0), -5000, t2); t2 += DT
    while True:
        o, k = r2.step((C, N + 20.0), 0, t2, hold_g=5.0); t2 += DT
        if r2.phase is not None:
            break
    assert r2.phase == 'brake' and o[1] > N and r2.guard_event is None
    # 中止・前進('escape')は変わらない
    r3 = A.Recovery((C, N))
    r3.phase = 'giveup'
    assert r3.step((C, N - 100.0), 0, 0.0, hold_g=9.0) == ((C, N), A.ASSIST_GIVEUP)
    r4 = A.Recovery((C, N))
    t4 = _to_reverse(r4)
    r4._go('escape', t4)
    o, k = r4.step((C, N), 0, t4 + DT, hold_g=9.0)
    assert r4.phase == 'escape' and r4.guard_event is None and o[1] == N - A.Recovery.ESCAPE_US


# ================================================================ 2. r1-rs1-2 StartKick の窓が区間の途中で開かない
def _assist_fuzz(mod, seed, n=3000, nk_mode=0):
    """でたらめな入力(人側・中立止め・止まる・走る・衝撃・前が近い)を Assist に通す。nk_mode: 0 = launch_no_kick ずっと 0、
    1 = ずっと 1。発進の前進は ON/OFF を混ぜる(launch=None か LaunchIn)"""
    rng = random.Random(seed)
    a = mod.Assist((C, N))
    t, out = 0.0, []
    sel, erpm, thr, kick_on, lb_on, guard = False, 0.0, N, True, rng.random() < 0.5, rng.random() < 0.5
    for i in range(n):
        if rng.random() < 0.02:
            sel = not sel
        if rng.random() < 0.08:
            erpm = rng.choice([0.0, 0.0, -300.0, -900.0, -3000.0, 500.0, 1500.0])
        if rng.random() < 0.05:
            thr = rng.choice([N, N - 30.0, N - 80.0, N - 150.0, N + 20.0, N + 100.0])
        if rng.random() < 0.03:
            a.launch_hold(); a.reset(keep_recovery=rng.random() < 0.5); t += DT
            out.append(('hold',))
            continue
        g = rng.choice([0.0] * 30 + [3.0, 6.0])
        blocked = rng.random() < 0.1
        ttc = rng.choice([0.5, 2.0, 5.0, 9.0])
        inp = mod.LaunchIn(v=-erpm * 0.0008, a=0.0, gz=0.0, steer_rel=0.0, ttc=ttc, ttc5=(ttc,) * 5, tof_ok=rng.random() < 0.9,
                           veh_ok=True, impact_g=g) if lb_on else None
        o, k = a.step((C, thr), erpm, sel, t, impact_g=g, kick_on=kick_on, blocked=blocked, launch=inp, rec_guard=guard,
                      launch_no_kick=bool(nk_mode))
        out.append((round(o[0], 6), round(o[1], 6), k, a.launch_on, a.launch_end,
                    None if a.boost.event is None else repr(a.boost.event), a.rec.guard_event))
        a.rec.guard_event = None
        t += DT
    return out


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('nk_mode', [0, 1])
def test_rs1_2_assist_same_as_before_when_launch_no_kick_constant(seed, nk_mode):
    """launch_no_kick が走行の間ずっと同じ(0 でも 1 でも)なら、Assist の出力・種類・出来事は直す前の写しと同じ(hold_g は 0)"""
    amod, _ = load_before()
    assert _assist_fuzz(A, seed, nk_mode=nk_mode) == _assist_fuzz(amod, seed, nk_mode=nk_mode)


def _giveup_then(mod, change_at_s, nk0=True, kick_on0=True, kick_on1=True, nk1=False, dur=16.0):
    """人側 1 秒 → AI 側。2 m/s → 2.0 秒に 3G で当たって止まる・モデルは中立 − 30。発進の前進 ON(止まっていない = moving で
    出さない)。change_at_s 秒から launch_no_kick を nk1・start_kick を kick_on1 にする。戻り値 [(秒, 出す値, 種類)]"""
    a = mod.Assist((C, N))
    out = []
    for i in range(int(dur / DT)):
        s = i * DT
        sel = s >= 1.0
        erpm = -2500.0 if s < 2.0 else 0.0
        g = 3.0 if 1.99 <= s < 2.03 else 0.0
        late = s >= change_at_s
        inp = mod.LaunchIn(v=-erpm * 0.0008, a=0.0, gz=0.0, steer_rel=0.0, ttc=9.0, ttc5=(9.0,) * 5, tof_ok=True, veh_ok=True,
                           impact_g=g)
        o, k = a.step((C, N - 30.0), erpm, sel, 100.0 + s, impact_g=g, kick_on=kick_on1 if late else kick_on0, launch=inp,
                      launch_no_kick=nk1 if late else nk0)
        out.append((round(s, 3), round(o[1], 3), k))
    return out


def test_rs1_2_assist_no_kick_window_after_launch_no_kick_off():
    """指摘の場面 B: 中止のあと launch_no_kick を 1 → 0。直す前は ASSIST_START を 2 秒出した。今は出ない(変えなかったときと同じ)"""
    amod, _ = load_before()
    base = _giveup_then(A, 99.0)
    g0 = next(s for s, o, k in base if k == A.ASSIST_GIVEUP)
    old = _giveup_then(amod, g0 + 1.0)
    new = _giveup_then(A, g0 + 1.0)
    assert sum(1 for s, o, k in old if k == amod.ASSIST_START) >= 55              # 指摘の再現(約 2 秒)
    assert sum(1 for s, o, k in new if k == A.ASSIST_START) == 0
    assert new == base
    # 後退の途中に 1 → 0 にしても、変えなかったときと同じ(直す前は 1796 → 1416 に変わった)
    rv = [s for s, o, k in base if k == A.ASSIST_RECOVERY and o > N + 100]
    old_r = _giveup_then(amod, rv[2])
    assert any(k == amod.ASSIST_START for s, o, k in old_r if s >= rv[2])
    assert _giveup_then(A, rv[2]) == base


def test_rs1_2_assist_next_section_still_kicks():
    """「出さない」は CH3 を人側にするまで。次の AI の区間で発進の前進が OFF(launch=None)なら、今まで通り StartKick が出る"""
    a = A.Assist((C, N))
    near = A.LaunchIn(v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=2.0, ttc5=(2.0,) * 5, tof_ok=True, veh_ok=True)
    t = 0.0
    a.step((C, N), 0.0, False, t)
    kinds = []
    for i in range(20):
        t += DT
        kinds.append(a.step((C, N), 0.0, True, t, kick_on=True, launch=near, launch_no_kick=True)[1])
    assert A.ASSIST_START not in kinds and a._no_kick
    for i in range(10):
        t += DT
        kinds.append(a.step((C, N), 0.0, True, t, kick_on=True, launch=near, launch_no_kick=False)[1])
    assert A.ASSIST_START not in kinds                          # 区間の途中で 0 にしても出さない
    t += DT
    a.step((C, N), 0.0, False, t)                               # CH3 を人側
    assert not a._no_kick
    t += DT
    assert a.step((C, N), 0.0, True, t, kick_on=True, launch=None, launch_no_kick=False)[1] == A.ASSIST_START


def test_rs1_2_set_params_start_kick_rules(monkeypatch, tmp_path):
    """走行中は start_kick を 0 → 1 にできない(400 の文)。1 → 0・止まっているときの 0 → 1・走行開始の 1 は今まで通り"""
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    flag = {'on': True}
    monkeypatch.setattr(drive.Driver, 'running', property(lambda self: flag['on']))
    rec = FakeRecorder()
    d = drive.Driver(hub=None, recorder=rec)
    d.st = {'run': 'x'}
    d.params = dict(d.params, start_kick=0)
    with pytest.raises(ValueError, match='発進の補助'):
        d.set_params({'start_kick': 1})
    assert d.params['start_kick'] == 0
    assert d.set_params({'start_kick': 0, 'steer_gain': 1.1})['steer_gain'] == 1.1       # 0 のまま・ほかの項目は通る
    d.params = dict(d.params, start_kick=1)
    assert d.set_params({'start_kick': 0})['start_kick'] == 0                             # 1 → 0 はできる
    assert d.set_params({'start_kick': 0, 'recovery': 1})['start_kick'] == 0
    flag['on'] = False
    assert d.set_params({'start_kick': 1})['start_kick'] == 1                             # 止まっていれば入れられる
    _, before = load_before()
    monkeypatch.setattr(before, 'SETTINGS_PATH', tmp_path / 'b.json')
    monkeypatch.setattr(before.Driver, 'running', property(lambda self: True))
    db = before.Driver(hub=None, recorder=FakeRecorder())
    db.st, db.params = {'run': 'x'}, dict(db.params, start_kick=0)
    assert db.set_params({'start_kick': 1})['start_kick'] == 1                            # 直す前は走行中に通った(指摘)


def _loop_change(mod, monkeypatch, tmp_path, params, change, at_s):
    """本物の走行ループ。人側 1 秒 → AI 側(2 m/s)→ 2.0 秒に 3G で当たって止まる・モデルは中立 − 30。at_s 秒に走行中の
    set_params(change)。戻り値 (行 [(秒, 出す値, 種類)], 受け付けたか)"""
    frames, imu = build(16.0, sel=lambda s: s >= 1.0, v=lambda s: 2.0 if s < 2.0 else 0.0,
                        ax=lambda s: 30.0 if 1.99 <= s < 2.03 else 0.0, pred=(C, N - 30.0))
    flag, res = {'on': False}, {}

    def hook(i, d):
        if at_s is not None and 'done' not in res and frames[i]['s'] >= at_s:
            flag['on'] = True
            try:
                d.set_params(change)
                res['ok'] = True
            except ValueError:
                res['ok'] = False
            flag['on'] = False
            res['done'] = True
    monkeypatch.setattr(mod.Driver, 'running', property(lambda self: flag['on']))
    _, _, rec = run(mod, monkeypatch, tmp_path, params, frames, imu, hook=hook)
    s_of = {f['cam_t']: f['s'] for f in frames}
    return [(round(s_of[t], 3), float(r['out'][1]), int(r['assist'])) for t, r in rec.rows if not r['hold']], res.get('ok')


def test_rs1_2_loop_launch_no_kick_off_mid_run(monkeypatch, tmp_path):
    """本物の走行ループで指摘の場面 B・B2: 中止の 1 秒あと・後退の途中に launch_no_kick を 1 → 0。受け付けるが、StartKick は出ず、
    変えなかった走行と同じ。直す前の写しは 61 フレーム(約 2 秒)出した"""
    _, before = load_before()
    P = dict(PARAMS, launch_boost=1, launch_no_kick=1, start_kick=1, recovery=1, tof_guard=1)
    base, _ = _loop_change(drive, monkeypatch, tmp_path, P, None, None)
    g0 = next(s for s, o, k in base if k == A.ASSIST_GIVEUP)
    rv = [s for s, o, k in base if k == A.ASSIST_RECOVERY and o > N + 100]
    for at in (g0 + 1.0, rv[2]):
        new, ok = _loop_change(drive, monkeypatch, tmp_path, P, {'launch_no_kick': 0}, at)
        old, ok_old = _loop_change(before, monkeypatch, tmp_path, P, {'launch_no_kick': 0}, at)
        assert ok and ok_old
        assert sum(1 for s, o, k in old if k == A.ASSIST_START) >= 55
        assert new == base


def test_rs1_2_loop_start_kick_on_mid_run_refused(monkeypatch, tmp_path):
    """本物の走行ループで指摘の場面(launch_boost = 0・start_kick = 0 → 中止の 1 秒あと {'start_kick': 1})。今は断る → StartKick は
    出ず、変えなかった走行と同じ。直す前の写しは受け付けて 61 フレーム出した"""
    _, before = load_before()
    P = dict(PARAMS, launch_boost=0, launch_no_kick=0, start_kick=0, recovery=1, tof_guard=1)
    base, _ = _loop_change(drive, monkeypatch, tmp_path, P, None, None)
    g0 = next(s for s, o, k in base if k == A.ASSIST_GIVEUP)
    new, ok = _loop_change(drive, monkeypatch, tmp_path, P, {'start_kick': 1}, g0 + 1.0)
    old, ok_old = _loop_change(before, monkeypatch, tmp_path, P, {'start_kick': 1}, g0 + 1.0)
    assert ok is False and new == base
    assert ok_old is True and sum(1 for s, o, k in old if k == A.ASSIST_START) >= 55


@pytest.mark.parametrize('lb,nk,sk', [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 1, 0), (0, 0, 0)])
def test_rs1_2_loop_constant_params_bit_identical(monkeypatch, tmp_path, lb, nk, sk):
    """走行中に変えなければ、本物の走行ループの PCA9685 に書く値・記録の行・出来事は直す前の写しと同じ"""
    _, before = load_before()
    P = dict(PARAMS, launch_boost=lb, launch_no_kick=nk, start_kick=sk, recovery=1, tof_guard=1)
    frames, imu = build(16.0, sel=lambda s: s >= 1.0 and not (8.0 <= s < 8.5), v=lambda s: 2.0 if s < 2.0 else 0.0,
                        ax=lambda s: 30.0 if 1.99 <= s < 2.03 else 0.0, pred=(C, N - 30.0),
                        veh_skip=lambda s: 5.0 <= s < 5.3 or 12.0 <= s < 12.2)
    a = run(drive, monkeypatch, tmp_path, P, frames, imu)
    b = run(before, monkeypatch, tmp_path, P, frames, imu)
    assert a[1].pairs == b[1].pairs and a[1].single == b[1].single
    assert_rows_equal_before(b[2].rows, a[2].rows)
    assert a[2].events == b[2].events
