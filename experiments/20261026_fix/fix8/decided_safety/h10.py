"""fix8 係 safety-decided: approvals 3「途中で OFF にした日」の ⑤ の読み方 (a) の前提を確かめる共通の部品。
前提 =「後ろの見張り(rec_back_guard)・残す(rec_hold_keep)が変えるのは、リカバリー(当たった・止まった)と中立止めのあとの
扱いだけで、当たらず中立止めも無い発進は同じ値」。

本物の drive._loop(jetformer/ の今の版)を、fix3/refute2_safety/closed_loop.py と fix7/safety_r2/h9.py(= 反証の
review_safety_r2/h8.py の写し)と同じ「閉じたループ」(偽物の ESC と車。PCA9685 に書いた値で車が動き、その結果が次の
eRPM・IMU・ToF になる)で回す。h9 から変えたところ:
  - 一時の場所は fix8/tmp_safety_decided(終わったら消す)。写し(直す前の版)は使わない(今の版の ON と OFF を並べるだけ)
  - 車(TrackCar): h9.Car の ESC の模型(scene['esc'] = (中立で下がれる秒, 後退側から下がり始める秒))に、電圧の倍率
    scene['volt'](前進の速さと加速に掛ける)を足した。前の壁(物理の当たり)は scene['xf'](既定はずっと先 = 当たらない)
  - コースの代わり(1 次元): scene['corners'] = 曲がり始める位置 [m] の並び。ToF は「まだ曲がっていない次の角の壁」
    (角の位置 + scene['wall_gap'])までの距離を見る(曲がったら壁は前から消える = 当たらない)。発進の前の距離 = 最初の角
  - IMU: 舵と速さからヨー(gz)と横の加速を出す(発進の前進の ③・速さの天井の左・周の数えが動く)。縦は closed_loop と同じ
  - 車両の値は 100 Hz(本物と同じ。scene['veh_dt'])。eRPM の跳ね scene['glitch'] = [(時刻, eRPM)]
    (そのサンプルの eRPM だけ差し替える。車は動かない)・車両の値の止まり scene['veh_skip']・ToF の欠け scene['tof_skip']・
    カメラの止まり scene['drops']・止まらない衝撃 scene['spikes'] = [(時刻, G)](段差の形)
  - モデル(scene['model'] = f(t, car)): コースの角の手前で減速・角で舵、発進の始めは弱い前進から(make_model)
装置に触れない・GPU を使わない・API を呼ばない。
"""
import collections
import math
import os
import shutil
import sys
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
FIXD = ROOT / 'experiments' / '20261026_fix'
TMP = FIXD / 'fix8' / 'tmp_safety_decided' / 'h' / os.environ.get('DS_TAG', 'main')
for p in (ROOT / 'tests', ROOT, FIXD / 'fix3' / 'refute2_safety'):
    sys.path.insert(0, str(p))

from jetformer import assist as A          # noqa: E402
from jetformer import clock, config, drive  # noqa: E402
from jetformer.sensors.base import Sample   # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize   # noqa: E402
import test_launch_drive as TLD            # noqa: E402
import closed_loop as CL                   # noqa: E402

C, N, PARAMS, T0, NS = TLD.C, TLD.N, TLD.PARAMS, TLD.T0, TLD.NS
DT = CL.DT
KV = CL.KV


class TrackCar(CL.Car):
    """h9.Car(= closed_loop.Car + scene['esc'])と同じ ESC の模型。足したもの: scene['volt'](前進の速さ・加速の倍率)、
    位置 x は「コースに沿った距離」。前の壁 scene['xf'] に当たったら止まる(closed_loop と同じ)"""

    def __init__(self, s):
        super().__init__(s)
        self.steer = C
        self.yaw_dps = 0.0

    def step(self, t, thr):
        s = self.s
        esc = s.get('esc')
        volt = float(s.get('volt', 1.0))
        neu, rdelay = esc if esc is not None else (0.1, 0.08)
        tol = 1e-9 if esc is not None else 0.0
        fwd, rev = thr < N - 20.0, thr > N + 40.0
        v0 = self.v
        if fwd:
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            if self.v < 0:
                self.v = min(0.0, self.v + 8.0 * DT)
            else:
                vt = min((N - thr) / 80.0 * 1.2, 6.0) * volt
                self.v += float(np.clip((vt - self.v) / 0.3, -6.0 * volt, 6.0 * volt)) * DT
        elif rev:
            self.neu_t = 0.0
            if self.rev_on:
                a = (thr - N) / 300.0 * 5.0
                self.v = max(self.v - a * DT, -(thr - N) / 300.0 * 3.0)
            elif self.ready and self.v <= 0.075:
                self.rev_wait += DT
                if self.rev_wait >= rdelay - tol:
                    self.rev_on = True
            else:
                self.braked = True
                self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= neu - tol:
                self.ready, self.braked = True, False
            if self.rev_on:
                self.rev_on = False
                self.ready = True
            self.v = max(0.0, self.v - 1.5 * DT) if self.v > 0 else min(0.0, self.v + 1.5 * DT)
        nx = self.x + self.v * DT
        prev_con = self.contact
        self.contact = None
        if nx >= s['xf']:
            if self.v > 0.05 and prev_con != 'front':
                g = min(2.0 + 2.5 * self.v, 9.0) * s.get('front_soft', 1.0)
                self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                self.spike = [g, g]
            nx, self.v = s['xf'], 0.0
            self.contact = 'front'
        if s.get('xr') is not None:
            if not self.rear_active and nx > s['xr'] + 0.1:
                self.rear_active = True
            if self.rear_active and nx <= s['xr']:
                if self.v < -0.05 and prev_con != 'rear':
                    g = min(2.0 + 2.5 * -self.v, 9.0) * s.get('rear_soft', 1.0)
                    self.hits.append((round(t, 3), 'rear', round(self.v, 2), round(g, 2)))
                    self.spike = [g, g]
                nx, self.v = s['xr'], 0.0
                self.contact = 'rear'
        self.x = nx
        acc = (self.v - v0) / DT
        g = self.spike.pop(0) if self.spike else 0.0
        return acc, g


def in_any(t, spans):
    return any(a <= t < b for a, b in spans)


class TrackSim(CL.Sim):
    """closed_loop.Sim の時計・カメラ(wait)をそのまま使い、センサーの作り方(_advance)だけ変える"""

    def __init__(self, d, sc_, bufs, cam_slot, box, pca, hook=None):
        super().__init__(d, sc_, bufs, cam_slot, box, pca, hook)
        self.car = TrackCar(sc_)
        self.next_cam = sc_.get('cam0', 0.15)
        self.n_glitch = 0
        self.cam_timeouts = []          # カメラが 0.1 秒来なかった時刻(drive は中立止め hold('カメラの映像が来ていません'))

    def wait(self, timeout):
        r = super().wait(timeout)
        if not r and not self.d._stop.is_set():
            self.cam_timeouts.append(round(self.t, 3))
        return r

    def wall_ahead(self):
        """まだ曲がっていない次の角の壁までの距離 [m](無ければ None)"""
        x = self.car.x
        gap = float(self.s.get('wall_gap', 0.6))
        for c in self.s.get('corners', []):
            if x < c:
                return c + gap - x
        return None

    def _advance(self, t_end):
        sp = self.s.get('spikes') or []
        gl = self.s.get('glitch') or []
        vdt = float(self.s.get('veh_dt', 0.01))
        while self.t + DT <= t_end + 1e-9:
            self.t = round(self.t + DT, 6)
            for ts, g in sp:
                if abs(self.t - ts) < DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            acc, g = self.car.step(self.t, self.pca.thr)
            if len(self.car.hits) > self.n_hits:
                self.n_hits = len(self.car.hits)
                if self.hit_drops:
                    dl, ln = self.hit_drops
                    self.drops.append((self.t + dl, self.t + dl + ln))
            v = self.car.v
            yaw = float(np.clip((C - self.pca.steer) / 400.0 * 45.0 * max(v, 0.0), -250.0, 250.0))
            t_ns = T0 + int(round(self.t * NS))
            ay = acc + g * 9.80665
            lat = math.radians(yaw) * max(v, 0.0)
            self.imu_buf.append((t_ns, np.array([lat, ay, 9.8, 0.0, 0.0, math.radians(yaw)], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
                if not in_any(self.t, self.s.get('veh_skip', [])):
                    sl = self.sel(self.t)
                    e = int(round(-v / KV))
                    for ts, ge in gl:
                        if abs(self.t - ts) < DT / 2:
                            e = int(ge)
                            self.n_glitch += 1
                    self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                                'select': sl, 'moving': abs(v) > 0.01, 'erpm': e}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                if not in_any(self.t, self.s.get('tof_skip', [])):
                    dist = self.wall_ahead()
                    if dist is None:
                        dist = self.s['xf'] - self.car.x
                    self.tof_buf.append((t_ns, CL.tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else CL.TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, self.pca.thr, self.car.contact, self.car.rev_on,
                             self.sel(self.t), self.pca.steer))
        self.box['ns'] = T0 + int(round(self.t * NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']


def make_model(sc):
    """コースの代わりのモデル(f(t, car) → (舵, スロットル) µs)。AI 側にする前(t < ai_from)は中立(車は止まったまま)。
    sc: ai_from・vmax(直線の目標 m/s)・vcorner(角)・corners・m0(発進の最初の前進 µs。小さいと発進の補助が出る)・
    ramp_s(最初の前進から目標まで寄せる秒)・steer0(発進の向き µs、dir_s 秒)・brake0(AI 側にした最初の dir 秒にブレーキ側)・
    turn(角の舵の向き +1 / -1 の並び)"""
    ta = float(sc['ai_from'])
    corners = list(sc.get('corners', []))
    turns = list(sc.get('turns', [1] * len(corners)))
    vmax, vcor = float(sc.get('vmax', 4.0)), float(sc.get('vcorner', 2.6))
    m0, ramp = float(sc.get('m0', 60.0)), max(float(sc.get('ramp_s', 0.8)), 1e-3)
    steer0, dir_s = float(sc.get('steer0', 0.0)), float(sc.get('dir_s', 1.0))
    brake0 = float(sc.get('brake0', 0.0))
    volt_m = float(sc.get('volt_model', 1.0))       # モデルが見込む速さの倍率(電圧と合わないと速さがずれる)

    def f(t, car):
        if t < ta:
            return (C, N)
        el = t - ta
        x, v = car.x, car.v
        vt, st = vmax, C
        for c, tn in zip(corners, turns):
            if c - 2.5 <= x < c:
                vt = min(vt, vcor + (vmax - vcor) * max(0.0, (c - x - 1.0) / 1.5))
            if c <= x < c + 2.0:
                vt = min(vt, vcor)
                st = C - tn * 300.0
        if el < dir_s:
            st = C + steer0
        fwd = vt / (1.2 * volt_m) * 80.0 + 60.0 * (vt - v)
        fwd = float(np.clip(fwd, 0.0, 330.0))
        if el < ramp:
            fwd = min(fwd, m0 + (fwd - m0) * el / ramp) if fwd > m0 else fwd
        if el < brake0:
            return (st, N + 100.0)
        if v > vt + 0.6:
            return (st, N + 120.0)          # 角の手前のブレーキ(前へ走っている間はブレーキ)
        return (st, N - fwd)
    return f


def run(params, scene, collect_ph=False):
    """本物の drive._loop を閉じたループで回す → (sim, rec, pca, stats)"""
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(drive, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                        time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        np.save(fp, np.full((42, 54), float(CL.FLOOR_MM), np.float32))
        mp.setattr(drive, 'FLOOR_REF_PATH', fp)
        rec = FakeRecorder()
        d = drive.Driver(hub=None, recorder=rec)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = CL.TimedPCA(box)
        out = drive._Output(pca, config.PCA9685, (C, N), lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        sim = TrackSim(d, scene, bufs, cam_slot, box, pca)
        d._loop(d.st, CL.Runner(sim), fake_normalize, pca, out, (C, N), 6, 0.0, cam_slot, sim, *bufs, False)
        d._stop.clear()
        stats = dict(d.st.get('stats') or {})
    return sim, rec, pca, stats


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) == type(b) and a == b


def compare(ra, rb, t_to=None):
    """PCA9685 に書いた値(時刻・舵・スロットル)・記録の行(AI_CMD の全部の欄。infer_ms のほか)・出来事を、フレームごとに比べる
    (t_to 秒まで。None = 全部)。→ dict(same, pca_first, row_first, ev_first, n_pca, n_rows, n_ev)"""
    (sa, reca, pa, _), (sb, recb, pb, _) = ra, rb
    cut = (lambda t: True) if t_to is None else (lambda t: t <= t_to + 1e-9)
    wa = [w for w in pa.w if cut(w[0])]
    wb = [w for w in pb.w if cut(w[0])]
    out = {'n_pca': len(wa)}
    out['pca_first'] = None
    if wa != wb:
        i = next((k for k, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb)))
        out['pca_first'] = {'i': i, 't': (wa[i] if i < len(wa) else wb[i])[0], 'on': wa[i] if i < len(wa) else None,
                            'off': wb[i] if i < len(wb) else None}
    ra_ = [(t, r) for t, r in reca.rows if cut((t - T0) / 1e9)]
    rb_ = [(t, r) for t, r in recb.rows if cut((t - T0) / 1e9)]
    out['n_rows'] = len(ra_)
    out['row_first'] = None
    for k, ((t0, r0), (t1, r1)) in enumerate(zip(ra_, rb_)):
        if t0 != t1 or set(r0) != set(r1):
            out['row_first'] = {'i': k, 't': (t0 - T0) / 1e9, 'col': 'time_or_keys'}
            break
        bad = [c for c in r0 if c != 'infer_ms' and not _same(r0[c], r1[c])]
        if bad:
            out['row_first'] = {'i': k, 't': (t0 - T0) / 1e9, 'col': bad,
                                'on': {c: _short(r0[c]) for c in bad}, 'off': {c: _short(r1[c]) for c in bad}}
            break
    if out['row_first'] is None and len(ra_) != len(rb_):
        out['row_first'] = {'i': min(len(ra_), len(rb_)), 't': None, 'col': 'rows_len', 'on': len(ra_), 'off': len(rb_)}
    ea = [e for e in reca.events if _ev_t(e) is None or cut(_ev_t(e))]
    eb = [e for e in recb.events if _ev_t(e) is None or cut(_ev_t(e))]
    out['n_ev'] = len(ea)
    out['ev_first'] = None
    for k, (x, y) in enumerate(zip(ea, eb)):
        if x[0] != y[0] or not _same(dict(x[1]), dict(y[1])):
            out['ev_first'] = {'i': k, 'on': [x[0], _short(dict(x[1]))], 'off': [y[0], _short(dict(y[1]))]}
            break
    if out['ev_first'] is None and len(ea) != len(eb):
        k = min(len(ea), len(eb))
        out['ev_first'] = {'i': k, 'on': [list(map(_short, e)) for e in ea[k:k + 3]],
                           'off': [list(map(_short, e)) for e in eb[k:k + 3]]}
    out['same'] = out['pca_first'] is None and out['row_first'] is None and out['ev_first'] is None
    return out


def _ev_t(e):
    """出来事の時刻 [秒](cam_epoch_us があれば。無ければ None = いつでも比べる)"""
    try:
        us = e[1].get('cam_epoch_us')
    except AttributeError:
        return None
    if us is None:
        return None
    return float(us) / 1e6 - float(clock.to_epoch_us(T0)) / 1e6


def _short(x):
    if isinstance(x, np.ndarray):
        return [round(float(v), 2) for v in x.ravel()[:4]]
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _short(v) for k, v in list(x.items())[:8]}
    if isinstance(x, (list, tuple)):
        return [_short(v) for v in list(x)[:6]]
    return x


def launch_cols(res, ai_from, t_end, d_r1=7.0, x_lh=4.0):
    """⑤(diag_launch/collect.py)の列の代わり(閉じたループの車から): 起点まで to_r1(AI 側から d_r1 m に届くまで)・
    動く move_0p2 / move_0p4(0.2 / 0.4 m/s に届くまで)・LH の入口の速さ v_lh(x_lh m に届いたときの速さ)・
    空転の窓(move_0p2 から 0.6 秒)の前進の平均 f_slipwin(µs。模型は空転しないので比はいつも 1。違いは前進の出し方で見る)・
    最初の周の当たり hits・リカバリー・中止のフレーム・中立止め"""
    sim, rec, pca, stats = res
    log = [r for r in sim.log if ai_from <= r[0] <= t_end]

    def first(cond):
        return next((r[0] - ai_from for r in log if cond(r)), None)
    m2 = first(lambda r: r[2] >= 0.2)
    m4 = first(lambda r: r[2] >= 0.4)
    r1 = first(lambda r: r[1] >= d_r1)
    lh = next((r for r in log if r[1] >= x_lh), None)
    fs = None
    if m2 is not None:
        ws = [N - r[3] for r in log if m2 <= r[0] - ai_from < m2 + 0.6]
        fs = round(float(np.mean(ws)), 2) if ws else None
    rows = [r for t, r in rec.rows if ai_from <= (t - T0) / 1e9 <= t_end]
    kinds = collections.Counter(int(r['assist']) for r in rows if not r['hold'])
    return {'to_r1': None if r1 is None else round(r1, 3), 'move_0p2': None if m2 is None else round(m2, 3),
            'move_0p4': None if m4 is None else round(m4, 3), 'v_lh': None if lh is None else round(lh[2], 3),
            'f_slipwin': fs, 'hits': [h for h in sim.car.hits if ai_from <= h[0] <= t_end],
            'rec_frames': kinds.get(A.ASSIST_RECOVERY, 0), 'giveup_frames': kinds.get(A.ASSIST_GIVEUP, 0),
            'launch_frames': kinds.get(A.ASSIST_LAUNCH, 0), 'kick_frames': kinds.get(A.ASSIST_START, 0),
            'hold_rows': sum(1 for r in rows if r['hold']), 'holds': stats.get('holds', 0),
            'ai_holds': ai_holds(res, ai_from, t_end),
            'x_end': round(log[-1][1], 3) if log else None,
            'guards': [e[1].get('what') for e in rec.events if e[0] == 'recovery_guard']}


def ai_holds(res, ai_from, t_end):
    """AI 側の中立止めの時刻(カメラが来なかった・記録の行のある中立止めで CH3 が AI 側)。人側の中立止めは ON も OFF も手順を
    全部戻すので数えない"""
    sim, rec = res[0], res[1]
    hs = [t for t in sim.cam_timeouts if ai_from <= t <= t_end and sim.sel(t)]
    hs += [round((t - T0) / 1e9, 3) for t, r in rec.rows if r['hold'] and r['select'] and ai_from <= (t - T0) / 1e9 <= t_end]
    return sorted(hs)


def ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=1)


def OFF(p):
    return dict(p, rec_back_guard=0, rec_hold_keep=0)


def base_params(**kw):
    """11/3 の回 3 の形(押すと: 速さ 0.90・start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・rev_us 300・左 4.1)"""
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, lookahead=3, tof_guard=1, throttle_gain=0.9,
             rev_s=0.4, rev_us=300.0, launch_boost=0, launch_no_kick=0, speed_ceiling=0, speed_auto=0, zone_caps=0,
             ceil_left_mps=4.1, rec_back_guard=0, rec_hold_keep=0)
    p.update(kw)
    return p


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
    for q in (TMP.parent, TMP.parent.parent):
        try:
            q.rmdir()
        except OSError:
            pass
