#!/usr/bin/env python3
"""走行ログの軌跡をコース図に当てはめる(推測航法 + コース図 + カメラの目印)。後処理専用(未来のデータも使う)

夜間作業の当てはめ(jetformer_reports/overnight_20261011/course/src/fit_run.py)を、次の点で広げたもの。
- 人のログ: 多くのログはシーンとシーンの間に 17〜21 秒の欠け(録画していない時間)がある。欠けで区間を切り、
  区間ごとに「向きの変化 vs 走った距離」をコース図の基準線と照合して、最初の位置と向きを決める(全体探索)。
  R1(右ヘアピン)が無い短い区間でも当てはめられる。
- 目印を増やした: 白いスタートライン 3 本・芝の左端(入るとき/出るとき)・青マットの奥と手前の端・LED 矢印ゲート。
  目印の時刻は、画面の行の動きと走った距離から「画面の基準の行に届いた瞬間」を内挿して決める(1コマより細かく)。
- 補正は時間の節点(既定 3 秒ごと)の折れ線で、節点間の変化にはランダムウォークの事前分布をかける。
- 検証: 目印の種類を1つずつ隠して当てはめ、隠した目印での残差を見る(保留検証)。

使い方:
    python tools/course_fit.py 20261010-200916 [...]        # 当てはめて tracks_<log>.npz / .json を書く
    python tools/course_fit.py --human --ai --validate       # 人のログ 12 本と AI のログ全部 + 保留検証

出力(既定 jetformer_reports/overnight_20261011/course/): tracks_<log>.npz(カメラのコマごと:
t, t_us, x, z, heading, speed, progress_m, progress_cum_m, lap, lap_phase, section, cross_track_m, valid, segment, ...)
と tracks_<log>.json(区間ごとの当てはめの情報・目印の残差・衝突)。
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')          # 行列計算を1スレッドに(並列はログ単位で行う)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import course_map as cm  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW = os.path.join(REPO, 'data', 'raw')
OUT = '/home/jetson/jetformer_reports/overnight_20261011/course'
G = 9.80665
KH = 5.0e-3                       # ホールセンサー 1 カウント = 5.0 mm(0.1 × eRPM カウント/秒、0.5e-3 m/s/eRPM)
GYRO_BIAS_DEFAULT = -0.09         # 止まっている時間が無いログのジャイロ z のずれ [deg/s](他のログの値)
FS = 50.0                         # 当てはめの時間刻み [Hz](IMU 100 Hz を 2 つずつまとめる)
# LogData が読む欄(2026-10-26 fix3。これ以外の欄は読まないので、壊れていてもシーンを捨てない)
USED_FIELDS = {'IMU': ('timestamps_us', 'accel', 'gyro'),
               'VEHICLE': ('timestamps_us', 'pwm', 'erpm', 'select', 'hall_count', 'board_t_us'),
               'CAM_FRONT': ('timestamps_us',),
               'AI_CMD': ('timestamps_us', 'select', 'out')}

HUMAN_LOGS = ['20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707',
              '20261010-192141', '20261010-192654', '20261010-193130', '20261010-193638', '20261010-200916',
              '20261010-201626', '20261010-202633']
AI_LOGS = ['20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
           '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
           '20261010-204605']


# ---------------------------------------------------------------- 読み込み
class LogData(object):
    """1ログの IMU・車両・カメラ・AI_CMD(あれば)。時刻は秒で、0 = 最初のカメラ画像"""

    def __init__(self, log, raw=RAW):
        self.log = log
        root = os.path.join(raw, log)
        self.meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
        scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
        parts = {'IMU': [], 'VEHICLE': [], 'CAM_FRONT': [], 'AI_CMD': []}
        self.scenes = []
        ai_bad_us = []         # AI_CMD.npz が読めなかったシーンの時刻の範囲(その VEHICLE の最初と最後、µs)

        def load(p, name):     # 使う欄だけ中身まで読む(壊れた zip は読むときに Bad CRC で落ちるので try の中で読む。2026-10-26 fix2。
            # fix3: 前は欄を全部読み、使わない欄(AI_CMD の pred・IMU の mag など)が壊れただけでシーンを捨てた。
            # 無い欄は読まない(今まで通り、あとの cat で KeyError)
            with np.load(p, allow_pickle=True) as z:
                return {k: z[k] for k in USED_FIELDS[name] if k in z.files}
        for sc in scenes:
            try:
                d = {n: load(os.path.join(root, sc, n + '.npz'), n) for n in ('IMU', 'VEHICLE', 'CAM_FRONT')}
                for n in d:
                    d[n]['timestamps_us']
            except Exception as e:
                print('注意: %s/%s が読めません(%s)。このシーンを飛ばします' % (log, sc, type(e).__name__), file=sys.stderr)
                continue
            for n in d:
                parts[n].append(d[n])
            p = os.path.join(root, sc, 'AI_CMD.npz')
            if os.path.exists(p):
                try:
                    parts['AI_CMD'].append(load(p, 'AI_CMD'))
                except Exception as e:
                    print('注意: %s/%s/AI_CMD.npz が読めません(%s)。飛ばします(このシーンの AI 側かどうかは VEHICLE の select)'
                          % (log, sc, type(e).__name__), file=sys.stderr)
                    vts = np.asarray(d['VEHICLE']['timestamps_us'], np.int64)
                    if len(vts):
                        ai_bad_us.append((int(vts.min()), int(vts.max())))
            self.scenes.append(sc)

        def cat(name, key, dtype=None):
            a = np.concatenate([p[key] for p in parts[name]]) if parts[name] else np.zeros(0)
            return a.astype(dtype) if dtype else a
        self.cam_us = cat('CAM_FRONT', 'timestamps_us', np.int64)
        self.t0_us = int(self.cam_us[0])
        sec = lambda a: (np.asarray(a, np.int64) - self.t0_us) / 1e6   # noqa: E731
        self.ct = sec(self.cam_us)
        self.cam_scene = np.concatenate([np.full(len(p['timestamps_us']), k) for k, p in enumerate(parts['CAM_FRONT'])])
        self.it = sec(cat('IMU', 'timestamps_us', np.int64))
        self.acc = cat('IMU', 'accel', float)
        self.gyr = cat('IMU', 'gyro', float)
        self.vt = sec(cat('VEHICLE', 'timestamps_us', np.int64))
        self.pwm = cat('VEHICLE', 'pwm', float)
        self.erpm = cat('VEHICLE', 'erpm', float)
        self.vsel = cat('VEHICLE', 'select').astype(bool)
        self.hall = cat('VEHICLE', 'hall_count', np.int64)
        self.board = cat('VEHICLE', 'board_t_us', np.int64)
        self.has_ai = len(parts['AI_CMD']) > 0
        if self.has_ai:
            self.at = sec(cat('AI_CMD', 'timestamps_us', np.int64))
            self.a_select = cat('AI_CMD', 'select').astype(bool)
            self.a_out = cat('AI_CMD', 'out', float)
        else:
            self.at = np.zeros(0)
            self.a_select = np.zeros(0, bool)
        # AI_CMD が読めなかったシーンの範囲 [秒](2026-10-26 fix3。前はその間も前のシーンの最後の select を返した)
        self.ai_bad = [(sec(a), sec(b)) for a, b in ai_bad_us] if self.has_ai else []
        self.g = np.abs(np.linalg.norm(self.acc, axis=1) - G) / G      # 重力を除いた加速度の大きさ [g]
        self.hg = np.hypot(self.acc[:, 0], self.acc[:, 1]) / G         # 水平の加速度 [g](+x 右、+y 前)

    def ai_at(self, t):
        """時刻 t に AI 側(CH3)だったか。AI_CMD が無ければ VEHICLE の select。
        AI_CMD.npz が読めなかったシーンの間も VEHICLE の select(2026-10-26 fix3)"""
        if self.has_ai and len(self.at):
            i = np.clip(np.searchsorted(self.at, t, side='right') - 1, 0, len(self.at) - 1)
            out = self.a_select[i]
            if self.ai_bad:
                tt = np.asarray(t, float)
                m = np.zeros(np.shape(tt), bool)
                for lo, hi in self.ai_bad:
                    m |= (tt >= lo) & (tt <= hi)
                if m.any():
                    j = np.clip(np.searchsorted(self.vt, tt, side='right') - 1, 0, len(self.vt) - 1)
                    out = np.where(m, self.vsel[j], out)
            return out
        i = np.clip(np.searchsorted(self.vt, t, side='right') - 1, 0, len(self.vt) - 1)
        return self.vsel[i]

    def impacts(self, gthr=2.5):
        """衝突(水平 gthr 以上)。0.6 秒以内の連続は1件。前後 50 ms で 1 サンプルだけのもの(異常値)は除く"""
        ev = []
        for i in np.nonzero(self.hg > gthr)[0]:
            if ev and self.it[i] - ev[-1]['t'] <= 0.6:
                if self.hg[i] > ev[-1]['g']:
                    ev[-1].update(g=float(self.hg[i]), i=int(i), tp=float(self.it[i]))
                continue
            near = (np.abs(self.it - self.it[i]) <= 0.05) & (self.hg > 1.0)
            ev.append({'t': float(self.it[i]), 'tp': float(self.it[i]), 'g': float(self.hg[i]), 'i': int(i),
                       'suspect': bool(near.sum() <= 1)})
        return [e for e in ev if not e['suspect']]


# ---------------------------------------------------------------- 推測航法
def gyro_bias(d):
    """止まっている間(|eRPM| < 1 かつ |gz| < 5 deg/s)のジャイロ z の中央値"""
    er = np.interp(d.it, d.vt, d.erpm)
    gz = d.gyr[:, 2]
    st = (np.abs(er) < 1) & (np.abs(gz) < 5)
    if st.sum() > 50:
        return float(np.median(gz[st])), int(st.sum())
    return GYRO_BIAS_DEFAULT, int(st.sum())


def dead_reckoning(d):
    """向き ψ [deg](ジャイロの積分、IMU の時刻)と、走った距離 sv [m](ホールセンサー、VEHICLE の時刻。
    基板の再起動などで飛んだところは eRPM で埋める)"""
    bias, nst = gyro_bias(d)
    dt = np.clip(np.diff(d.it, prepend=d.it[0]), 0, 0.05)
    psi = np.cumsum((d.gyr[:, 2] - bias) * dt)
    dh = -np.diff(d.hall).astype(float)
    vdt = np.diff(d.vt)
    bad = (np.abs(dh) > 60) | (vdt <= 0) | (vdt > 0.1)
    dh[bad] = (0.1 * (-d.erpm[1:]) * np.clip(vdt, 0, 0.1))[bad]
    sv = np.r_[0, np.cumsum(dh)] * KH
    return psi, sv, {'gyro_bias_dps': bias, 'gyro_bias_still_samples': nst, 'hall_gaps_filled': int(bad.sum())}


def _runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False].astype(int)
    dd = np.diff(m)
    return list(zip(np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]))


def segments(d, psi, gap_s=0.25, bridge_s=2.0, carry_deg=20.0, min_dist=6.0):
    """当てはめる区間。IMU の bridge_s 秒以上の欠け(シーン間の録画していない時間)と、止まっている間に向きが
    carry_deg 以上変わったところ(手で持ち上げて置き直した)で切る。それより短い IMU の欠け(I2C の一時的な途切れ、
    約 0.9 秒)は、車両のデータがあればつないで、向きの変化を未知数(ramp)にする。
    戻り値: [(t0, t1)], 切った所 [(t0, t1, 理由)], つないだ IMU の欠け [(t0, t1)]"""
    cuts, bridges = [], []
    for i in np.nonzero(np.diff(d.it) > gap_s)[0]:
        a, b = float(d.it[i]), float(d.it[i + 1])
        vin = (d.vt > a) & (d.vt < b)
        if b - a < bridge_s and vin.sum() >= 0.8 * (b - a) * 100 - 5:
            bridges.append((a, b))
        else:
            cuts.append((a, b, 'gap'))
    er = np.interp(d.it, d.vt, d.erpm)
    for a, b in _runs(np.abs(er) < 100):
        if b - a < 5 or np.any(np.diff(d.it[a:b]) > gap_s):
            continue
        rot = np.abs(np.diff(psi[a:b])).sum()
        if rot > carry_deg:
            cuts.append((float(d.it[a]), float(d.it[b - 1]), 'carried'))
    cuts.sort()
    out = []
    t_start = float(d.it[0])
    for a, b, why in cuts:
        if a > t_start:
            out.append((t_start, a))
        t_start = max(t_start, b)
    if d.it[-1] > t_start:
        out.append((t_start, float(d.it[-1])))
    return out, cuts, bridges


# ---------------------------------------------------------------- 全体探索(区間の最初の位置と向き)
def register(psi_deg, s, cl, ds=0.10, ks=None, gs=(1.0, 1.02), s0_step=0.10):
    """向きの変化 vs 走った距離 を基準線の向き θ(s) と照合する。
    ψ(d)·g + ψ0 ≈ θ(s0 + k·d) となる (s0, k, g, ψ0) を全探索する(ψ0 は円周平均で閉じた形)。
    戻り値: dict(s0, k, g, psi0, score(0〜1、1 = 完全一致), score2(s0 が 3 m 以上離れた次点), n)"""
    ks = np.arange(0.86, 1.161, 0.02) if ks is None else np.asarray(ks)
    sa = np.r_[0, np.cumsum(np.abs(np.diff(s)))]
    fwd = np.r_[True, np.diff(s) > 0]
    if sa[-1] < 2.0:
        return None
    dgrid = np.arange(0, sa[-1], ds)
    ps = np.radians(np.interp(dgrid, sa, psi_deg))
    w = np.interp(dgrid, sa, fwd.astype(float)) > 0.5      # 前進している所だけ
    if w.sum() < 10:
        return None
    dgrid, ps = dgrid[w], ps[w]
    L = cl.L
    s0s = np.arange(0, L, s0_step)
    cth = np.cos(cl.h)
    sth = np.sin(cl.h)
    best = None
    table = []
    for g in gs:
        e = np.exp(-1j * g * ps)
        for k in ks:
            ss = (s0s[:, None] + k * dgrid[None, :]) % L
            c = np.interp(ss, cl.s, cth, period=L) + 1j * np.interp(ss, cl.s, sth, period=L)
            z = (c * e[None, :]).mean(1)
            sc = np.abs(z)
            table.append((g, k, sc, z))
            j = int(np.argmax(sc))
            if best is None or sc[j] > best['score']:
                best = {'s0': float(s0s[j]), 'k': float(k), 'g': float(g), 'psi0': float(np.angle(z[j])),
                        'score': float(sc[j])}
    # 次点(s0 が 3 m 以上離れたもの)
    sc2 = 0.0
    for g, k, sc, z in table:
        dd = np.abs(((s0s - best['s0']) + L / 2) % L - L / 2)
        m = dd > 3.0
        if m.any():
            sc2 = max(sc2, float(sc[m].max()))
    best['score2'] = sc2
    best['n'] = int(len(dgrid))
    best['dist'] = float(sa[-1])
    return best


def initial_pose(reg, psi_deg, s, cl):
    """全体探索の結果から、区間全体の向きのずれ ψ0 と位置のずれ (Bx, Bz)(定数)を作る"""
    h = reg['g'] * np.radians(psi_deg) + reg['psi0']
    hm = np.r_[h[0], 0.5 * (h[1:] + h[:-1])]
    ds = np.diff(s, prepend=s[0])
    X = np.cumsum(ds * np.cos(hm))
    Z = np.cumsum(ds * np.sin(hm))
    sa = np.r_[0, np.cumsum(np.abs(np.diff(s)))]
    tgt, _ = cl.at(reg['s0'] + reg['k'] * sa)
    return reg['psi0'], float(np.mean(tgt[:, 0] - X)), float(np.mean(tgt[:, 1] - Z))


# ---------------------------------------------------------------- 当てはめ
PARAMS = {
    'knot_dt': 3.0,            # 補正の節点の間隔 [s]
    'a_rw': 1.5,               # 向きの補正のランダムウォーク [deg/√s]
    'b_rw': 0.08,              # 位置の補正のランダムウォーク [m/√s]
    'g_sd': 0.01,              # ジャイロの倍率の事前分布の標準偏差(平均 g0)
    'sig_sd': 0.05,            # 距離の倍率の標準偏差(平均 1)
    'kv_sd': 0.02,             # 速さによる距離の倍率の変化 [1/(m/s)]
    'c_sd': 30.0,              # ジャイロが振り切れた衝突・IMU の欠けでの向きの段差 [deg]
    'cw': 0.3,                 # 基準線へ引き寄せる重み(横ずれ [m] に掛ける)
    'huber': 0.25,             # 基準線への引き寄せの Huber の幅 [m]
    'dz': 0.45, 'dzw': 1.0,    # 基準線から dz 以上離れた分への罰則
    'wall_w': 20.0, 'margin': 0.10, 'look': 0.5,   # 壁に margin より近づいた分への罰則
    'lm_sd': 0.05,             # 目印の残差の標準偏差 [m]
    'lm_huber': 0.15,          # 目印の Huber の幅 [m]
    'lm_reject': 0.5,          # これより残差が大きい目印は外れ値として使わない [m]
    'n_outer': 10, 'n_gn': 2,
    'virtual_wall': True,
    'out_w': 5.0,              # 外周の外に出た距離 [m] への罰則
}
# 仮の壁: 駐車区画の入口(外周の板が無い X 1.88〜5.60、Z=0)。手前の直線を +X へ走っている間だけ効かせる
# (区画の仕切りがあるので、直線を走りながら Z < 0 には入れない。駐車するときは -Z を向いているので効かない)
VIRTUAL_WALLS = np.array([[[1.88, 0.0], [5.60, 0.0]]])
# コースの外周(この外には出られない)。駐車区画を含む
OUTER = np.array([[0.20, 5.60], [9.55, 5.60], [10.30, 4.70], [10.30, 0.85], [9.85, 0.0], [5.60, 0.0], [5.60, -0.50],
                  [1.88, -0.50], [1.88, 0.0], [0.0, 0.0], [0.0, 4.65]])
OUTER_SEGS = np.stack([OUTER, np.roll(OUTER, -1, 0)], 1)


def inside_outer(P):
    from matplotlib.path import Path
    return Path(OUTER).contains_points(np.asarray(P, float))


# 目印の「画面の基準の行までの距離」D の事前分布(平均, 標準偏差)[m]。群ごとに区間で共通の未知数
D_GROUPS = ['floor', 'mat', 'gate']
D_PRIOR = {'floor': (0.30, 0.15), 'mat': (0.30, 0.20), 'gate': (1.50, 0.20)}
LM_GROUP = {'line': 'floor', 'turf_in': 'floor', 'turf_out': 'floor', 'mat_n': 'mat', 'mat_s': 'mat', 'gate': 'gate'}


class Fitter(object):
    """1区間の当てはめ。パラメータ x = [g, σ, kv, A_0..A_{K-1}, Bx_0.., Bz_0.., c_0..c_{J-1}, D_0..D_{G-1}]
    向き h(t) = g·ψ(t) + A(t) + Σ c_j·ramp_j(t)、位置 p(t) = σ ∫ (1 + kv·(v-2.5)) ds [cos, sin](h) + B(t)
    A, B: 節点の折れ線(節点間の差にランダムウォークの事前分布)。ramp_j: 振り切れ・欠けの区間で 0→1"""

    def __init__(self, t, psi_rad, s, knots, ramps, cl, walls, prm=None, g0=1.0):
        self.t = np.asarray(t, float)
        self.psi = np.asarray(psi_rad, float)
        self.ds = np.diff(s, prepend=s[0])
        v = np.gradient(s, self.t)
        self.vrel = np.convolve(v, np.ones(7) / 7, 'same') - 2.5
        self.T = np.asarray(knots, float)
        self.K = len(self.T)
        self.ramps = list(ramps)
        self.J = len(self.ramps)
        self.R = np.array([np.clip((self.t - a) / max(b - a, 1e-3), 0, 1) for a, b in self.ramps]) if self.J else np.zeros((0, len(t)))
        self.c0 = np.zeros(self.J)                       # 段差の事前分布の平均 [rad]
        self.csd = np.full(self.J, np.radians(dict(PARAMS, **(prm or {}))['c_sd']))
        self.G = len(D_GROUPS)
        self.cl = cl
        self.n_real = len(walls)
        self.walls = np.concatenate([walls, VIRTUAL_WALLS]) if self.prm_virtual(prm) else walls
        _, ts = cm.wall_segments(two_sided=True)
        self.two_sided = np.r_[ts, np.zeros(len(self.walls) - len(ts), bool)] if len(ts) == self.n_real else np.zeros(len(self.walls), bool)
        self.prm = dict(PARAMS, **(prm or {}))
        self.g0 = g0
        self.events = []
        self.reject = False                # True: 残差が lm_reject を超えた目印を外れ値として使わない
        self.D0 = np.array([D_PRIOR[k][0] for k in D_GROUPS])
        self.Dsd = np.array([D_PRIOR[k][1] for k in D_GROUPS])
        # 節点の補間の重み
        j = np.clip(np.searchsorted(self.T, self.t, side='right') - 1, 0, max(self.K - 2, 0))
        if self.K >= 2:
            w = np.clip((self.t - self.T[j]) / (self.T[j + 1] - self.T[j]), 0, 1)
        else:
            w = np.zeros(len(self.t))
        self.kj, self.kw = j, w
        self.ipt = np.arange(0, len(self.t), max(1, int(round(FS / 10))))     # 基準線・壁を見る点(10 Hz)
        self.ipt = self.ipt[np.abs(np.gradient(s, self.t))[self.ipt] > 0.3]    # 止まっている所は使わない

    @staticmethod
    def prm_virtual(prm):
        return dict(PARAMS, **(prm or {})).get('virtual_wall', True)

    # パラメータの並び
    def sl(self):
        K, J, G = self.K, self.J, self.G
        o = 3
        return {'A': slice(o, o + K), 'Bx': slice(o + K, o + 2 * K), 'Bz': slice(o + 2 * K, o + 3 * K),
                'c': slice(o + 3 * K, o + 3 * K + J), 'D': slice(o + 3 * K + J, o + 3 * K + J + G)}

    def nparams(self):
        return 3 + 3 * self.K + self.J + self.G

    def interp_knots(self, v):
        if self.K == 1:
            return np.full(len(self.t), v[0])
        return v[self.kj] * (1 - self.kw) + v[self.kj + 1] * self.kw

    def heading(self, x):
        S = self.sl()
        h = x[0] * self.psi + self.interp_knots(x[S['A']])
        if self.J:
            h = h + x[S['c']] @ self.R
        return h

    def pose(self, x):
        S = self.sl()
        h = self.heading(x)
        hm = np.r_[h[0], 0.5 * (h[1:] + h[:-1])]
        f = x[1] * (1.0 + x[2] * self.vrel) * self.ds
        X = np.cumsum(f * np.cos(hm)) + self.interp_knots(x[S['Bx']])
        Z = np.cumsum(f * np.sin(hm)) + self.interp_knots(x[S['Bz']])
        return X, Z, h

    def init(self, psi0, bx, bz, sigma=1.0):
        x = np.zeros(self.nparams())
        S = self.sl()
        x[0] = self.g0
        x[1] = sigma
        x[S['A']] = psi0
        x[S['Bx']] = bx
        x[S['Bz']] = bz
        x[S['D']] = self.D0
        x[S['c']] = self.c0
        return x

    def set_events(self, events):
        """目印: [{'t', 'axis', 'value', 'group'}]。時刻 t の車の位置 + D·(向きの単位ベクトル) の axis 成分 = value"""
        self.events = []
        for e in events:
            i = int(np.clip(np.searchsorted(self.t, e['t']) - 1, 0, len(self.t) - 2))
            w = float(np.clip((e['t'] - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-6), 0, 1))
            self.events.append(dict(e, i=i, w=w, gi=D_GROUPS.index(e['group'])))

    def event_residuals(self, x, X=None, Z=None, h=None):
        if X is None:
            X, Z, h = self.pose(x)
        D = x[self.sl()['D']]
        out = []
        for e in self.events:
            i, w = e['i'], e['w']
            xe = X[i] * (1 - w) + X[i + 1] * w
            ze = Z[i] * (1 - w) + Z[i + 1] * w
            he = h[i] * (1 - w) + h[i + 1] * w
            if e['axis'] == 'x':
                out.append(xe + D[e['gi']] * np.cos(he) - e['value'])
            else:
                out.append(ze + D[e['gi']] * np.sin(he) - e['value'])
        return np.array(out)

    # ---- 残差とヤコビアン(姿勢 → 残差は解析的、姿勢 → パラメータは B, D 以外を数値微分)
    def _corr(self, X, Z, h):
        """対応づけ: 基準線の点と、近い壁"""
        p = self.prm
        P = np.stack([X[self.ipt], Z[self.ipt]], 1)
        idx, off = self.cl.match(P, h[self.ipt])
        ok = idx >= 0
        a = np.abs(off)
        w = np.ones(len(idx))
        big = a > p['huber']
        w[big] = np.sqrt(p['huber'] / a[big])
        w[~ok] = 0
        self.c_idx = np.where(ok, idx, 0)
        self.c_w = w
        # 壁: 通路の側は、対応する基準線の点の側。ただし基準線と直角に近い壁(駐車区画の仕切りなど)は今いる側
        dd, k = cm.dist_to_segments(P, self.walls)
        sel = (dd < p['look']) & ok
        if self.n_real < len(self.walls):
            # 仮の壁(駐車区画の入口 Z=0): 手前の直線を +X へ走っている(向き ±60° 以内)ときだけ
            virt = k >= self.n_real
            sel &= ~virt | (np.abs(cm.wrap(h[self.ipt])) < np.radians(60))
        self.w_pts = np.nonzero(sel)[0]
        self.w_seg = k[sel]
        A = self.walls[self.w_seg, 0]
        B = self.walls[self.w_seg, 1]
        C = self.cl.xz[idx[sel]]
        cr = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
        side = np.sign(cr)
        AB = B - A
        u = AB / np.maximum(np.hypot(AB[:, 0], AB[:, 1]), 1e-9)[:, None]
        perp_wall = (np.abs((u * self.cl.t[idx[sel]]).sum(1)) < 0.5) | self.two_sided[self.w_seg]
        Pw = P[sel]
        cur = np.sign(AB[:, 0] * (Pw[:, 1] - A[:, 1]) - AB[:, 1] * (Pw[:, 0] - A[:, 0]))
        side[perp_wall] = cur[perp_wall]
        side[side == 0] = 1
        self.w_side = side
        # 外周の外にいる点(基準線に対応しない遠い点も含む)
        self.o_pts = np.nonzero(~inside_outer(P))[0]
        # 目印の Huber 重み
        self.e_w = np.ones(len(self.events))
        if self.events:
            r = np.abs(self.event_residuals(self.x, X, Z, h))
            big = r > p['lm_huber']
            self.e_w[big] = np.sqrt(p['lm_huber'] / r[big])
            if self.reject:
                self.e_w[r > p['lm_reject']] = 0.0
        return off, ok

    def _res_pose(self, X, Z, h, D):
        """姿勢の関数としての残差と、その姿勢への偏微分。戻り値: r, 行ごとの (点の番号, dX, dZ, dh) の組"""
        p = self.prm
        i = self.ipt
        n = self.cl.n[self.c_idx]
        c = self.cl.xz[self.c_idx]
        raw = (X[i] - c[:, 0]) * n[:, 0] + (Z[i] - c[:, 1]) * n[:, 1]
        r1 = raw * self.c_w * p['cw']
        d1 = (self.c_w * p['cw'])[:, None] * n
        exc = np.abs(raw) > p['dz']
        r2 = np.sign(raw) * np.maximum(0, np.abs(raw) - p['dz']) * self.c_w * p['dzw']
        d2 = (exc * self.c_w * p['dzw'])[:, None] * n
        rows = [(i, d1[:, 0], d1[:, 1], np.zeros(len(i))), (i, d2[:, 0], d2[:, 1], np.zeros(len(i)))]
        res = [r1, r2]
        if p['wall_w'] and len(self.w_pts):
            j = i[self.w_pts]
            P = np.stack([X[j], Z[j]], 1)
            A = self.walls[self.w_seg, 0]
            B = self.walls[self.w_seg, 1]
            AB = B - A
            L2 = np.maximum((AB * AB).sum(1), 1e-12)
            tt = ((P - A) * AB).sum(1) / L2
            inner = (tt > 0) & (tt < 1)
            nrm = np.stack([-AB[:, 1], AB[:, 0]], 1) / np.sqrt(L2)[:, None] * self.w_side[:, None]
            perp = ((P - A) * nrm).sum(1)
            Cp = A + np.clip(tt, 0, 1)[:, None] * AB
            dv = P - Cp
            dn = np.maximum(np.hypot(dv[:, 0], dv[:, 1]), 1e-9)
            dist = np.where(inner, perp, dn)
            grad = np.where(inner[:, None], nrm, dv / dn[:, None])
            act = (p['margin'] - dist) > 0
            res.append(p['wall_w'] * np.maximum(0, p['margin'] - dist))
            gw = -(p['wall_w'] * act)[:, None] * grad
            rows.append((j, gw[:, 0], gw[:, 1], np.zeros(len(j))))
        if p['out_w'] and len(getattr(self, 'o_pts', [])):
            j = i[self.o_pts]
            P = np.stack([X[j], Z[j]], 1)
            dd, k = cm.dist_to_segments(P, OUTER_SEGS)
            A = OUTER_SEGS[k, 0]
            B = OUTER_SEGS[k, 1]
            AB = B - A
            tt = np.clip(((P - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
            dv = P - (A + tt[:, None] * AB)
            dn = np.maximum(np.hypot(dv[:, 0], dv[:, 1]), 1e-9)
            res.append(p['out_w'] * dn)
            g = p['out_w'] * dv / dn[:, None]
            rows.append((j, g[:, 0], g[:, 1], np.zeros(len(j))))
        return res, rows

    def set_targets(self, xz, w=1.0):
        """全体探索の対応(基準線上の点)へ直接引き寄せる(最初の粗い当てはめ用)。xz: [len(ipt), 2]、NaN は使わない"""
        self.targets = None if xz is None else (np.asarray(xz, float), w)

    def residuals(self, x, full=False):
        X, Z, h = self.pose(x)
        S = self.sl()
        if getattr(self, 'targets', None) is not None:
            tg, w = self.targets
            i = self.ipt
            ok = np.isfinite(tg[:, 0])
            dx = np.where(ok, X[i] - tg[:, 0], 0.0)
            dz = np.where(ok, Z[i] - tg[:, 1], 0.0)
            a = np.hypot(dx, dz)
            hw = np.where(a > 0.5, np.sqrt(0.5 / np.maximum(a, 1e-9)), 1.0) * w
            res = [dx * hw, dz * hw] + self._prior(x)
            return np.concatenate(res)
        res, _ = self._res_pose(X, Z, h, x[S['D']])
        res = list(res)
        res += self._prior(x)
        if self.events:
            res.append(self.event_residuals(x, X, Z, h) * self.e_w / self.prm['lm_sd'])
        return np.concatenate(res)

    def _prior(self, x):
        p = self.prm
        S = self.sl()
        dT = np.maximum(np.diff(self.T), 1e-3)
        out = [np.atleast_1d((x[0] - self.g0) / p['g_sd']), np.atleast_1d((x[1] - 1.0) / p['sig_sd']),
               np.atleast_1d(x[2] / p['kv_sd']),
               np.diff(x[S['A']]) / (np.radians(p['a_rw']) * np.sqrt(dT)),
               np.diff(x[S['Bx']]) / (p['b_rw'] * np.sqrt(dT)), np.diff(x[S['Bz']]) / (p['b_rw'] * np.sqrt(dT)),
               (x[S['c']] - self.c0) / self.csd, (x[S['D']] - self.D0) / self.Dsd]
        return out

    def jacobian_numeric(self, x):
        """数値微分のヤコビアン(検算用)"""
        r0 = self.residuals(x)
        Jm = np.empty((len(r0), len(x)))
        for j in range(len(x)):
            e = 1e-6 if j >= 3 else 1e-7
            xx = x.copy()
            xx[j] += e
            Jm[:, j] = (self.residuals(xx) - r0) / e
        return r0, Jm

    def _hat(self):
        """節点の折れ線の重み [N, K](疎だが小さいので密に持つ)"""
        if getattr(self, '_U', None) is None:
            U = np.zeros((len(self.t), self.K))
            n = np.arange(len(self.t))
            if self.K == 1:
                U[:, 0] = 1
            else:
                U[n, self.kj] = 1 - self.kw
                U[n, self.kj + 1] += self.kw
            self._U = U
        return self._U

    def jacobian(self, x):
        """解析的なヤコビアン。姿勢 (X, Z, h) のパラメータ微分 → 残差の姿勢微分(連鎖律)"""
        if getattr(self, 'targets', None) is not None:
            return self.jacobian_numeric(x)
        p = self.prm
        S = self.sl()
        P = len(x)
        X, Z, h = self.pose(x)
        hm = np.r_[h[0], 0.5 * (h[1:] + h[:-1])]
        f0 = (1.0 + x[2] * self.vrel) * self.ds
        f = x[1] * f0
        gx = -f * np.sin(hm)          # dX/dhm の各項
        gz = f * np.cos(hm)
        U = self._hat()

        def mid(u):
            return np.concatenate([u[:1], 0.5 * (u[1:] + u[:-1])], 0)
        # 向きに効くパラメータ: g, A, c
        Hcols = [self.psi[:, None], U] + ([self.R.T] if self.J else [])
        Uh = np.concatenate(Hcols, 1)                     # [N, 1+K+J]  dh/dθ
        Um = mid(Uh)
        dXh = np.cumsum(gx[:, None] * Um, 0)
        dZh = np.cumsum(gz[:, None] * Um, 0)
        # 姿勢の微分 [N, P]
        JX = np.zeros((len(self.t), P))
        JZ = np.zeros((len(self.t), P))
        JH = np.zeros((len(self.t), P))
        hidx = np.r_[0, np.arange(S['A'].start, S['A'].stop), np.arange(S['c'].start, S['c'].stop)].astype(int)
        JX[:, hidx] = dXh
        JZ[:, hidx] = dZh
        JH[:, hidx] = Uh
        cx = np.cumsum(f0 * np.cos(hm))
        cz = np.cumsum(f0 * np.sin(hm))
        JX[:, 1] = cx
        JZ[:, 1] = cz
        JX[:, 2] = np.cumsum(x[1] * self.vrel * self.ds * np.cos(hm))
        JZ[:, 2] = np.cumsum(x[1] * self.vrel * self.ds * np.sin(hm))
        JX[:, S['Bx']] = U
        JZ[:, S['Bz']] = U
        # 残差
        res, rows = self._res_pose(X, Z, h, x[S['D']])
        R_ = []
        Jr = []
        for r, (idx, dX, dZ, dH) in zip(res, rows):
            R_.append(r)
            Jr.append(dX[:, None] * JX[idx] + dZ[:, None] * JZ[idx] + dH[:, None] * JH[idx])
        pri = self._prior(x)
        npri = sum(len(np.atleast_1d(v)) for v in pri)
        Jp = np.zeros((npri, P))
        dT = np.maximum(np.diff(self.T), 1e-3)
        o = 0
        Jp[o, 0] = 1 / p['g_sd']; o += 1
        Jp[o, 1] = 1 / p['sig_sd']; o += 1
        Jp[o, 2] = 1 / p['kv_sd']; o += 1
        for key, sd in (('A', np.radians(p['a_rw'])), ('Bx', p['b_rw']), ('Bz', p['b_rw'])):
            st = S[key].start
            for k in range(self.K - 1):
                Jp[o, st + k] = -1 / (sd * np.sqrt(dT[k]))
                Jp[o, st + k + 1] = 1 / (sd * np.sqrt(dT[k]))
                o += 1
        for j in range(self.J):
            Jp[o, S['c'].start + j] = 1 / self.csd[j]; o += 1
        for g_ in range(self.G):
            Jp[o, S['D'].start + g_] = 1 / self.Dsd[g_]; o += 1
        R_ += pri
        Jr.append(Jp)
        if self.events:
            D = x[S['D']]
            er = self.event_residuals(x, X, Z, h)
            sc = self.e_w / p['lm_sd']
            Je = np.zeros((len(self.events), P))
            for n_, e in enumerate(self.events):
                i, w = e['i'], e['w']
                he = h[i] * (1 - w) + h[i + 1] * w
                Dg = D[e['gi']]
                if e['axis'] == 'x':
                    jj = (1 - w) * (JX[i] - Dg * np.sin(he) * JH[i]) + w * (JX[i + 1] - Dg * np.sin(he) * JH[i + 1])
                    jj[S['D'].start + e['gi']] += np.cos(he)
                else:
                    jj = (1 - w) * (JZ[i] + Dg * np.cos(he) * JH[i]) + w * (JZ[i + 1] + Dg * np.cos(he) * JH[i + 1])
                    jj[S['D'].start + e['gi']] += np.sin(he)
                Je[n_] = jj * sc[n_]
            R_.append(er * sc)
            Jr.append(Je)
        return np.concatenate(R_), np.concatenate(Jr, 0)

    def fit(self, x, n_outer=None, n_gn=None, log=None):
        p = self.prm
        n_outer = n_outer or p['n_outer']
        n_gn = n_gn or p['n_gn']
        self.x = x
        for it in range(n_outer):
            X, Z, h = self.pose(x)
            if getattr(self, 'targets', None) is not None:
                off, ok = np.zeros(1), np.ones(1, bool)
            else:
                off, ok = self._corr(X, Z, h)
            lam = 1e-3
            for _ in range(n_gn):
                r0, Jm = self.jacobian(x)
                Hm = Jm.T @ Jm
                step = np.linalg.solve(Hm + lam * np.diag(np.diag(Hm) + 1e-6), -Jm.T @ r0)
                xn = x + step
                if (self.residuals(xn) ** 2).sum() < (r0 ** 2).sum():
                    x = xn
                    lam *= 0.3
                else:
                    lam *= 10
            self.x = x
            if log is not None:
                log.append({'iter': it, 'matched': float(ok.mean()), 'med_abs_off': float(np.nanmedian(np.abs(off[ok]))) if ok.any() else None,
                            'g': float(x[0]), 'sigma': float(x[1]), 'kv': float(x[2])})
        return x


STEER_YAW = (1.9, -0.45, 0.06)    # ヨーレート [rad/s] = v·(a·u + b·u³)、u = (ステア µs - 中心)/500、遅れ 0.06 s(人のログで合わせた値)


def steer_yaw(d, t0, t1, center=None):
    """t0〜t1 の向きの変化 [rad] を、ステアリングの操作値と速さ(eRPM)から予測する(IMU が欠けた間を埋める)。
    AI 側のときは AI_CMD の出力、人のときは VEHICLE のスティックの値を使う。0.9 秒で誤差の中央値 約 10°"""
    a, b, lag = STEER_YAW
    center = center or float((d.meta.get('drive') or {}).get('steer_center_us') or 1443)
    tt = np.arange(t0, t1, 0.01)
    if not len(tt):
        return 0.0
    st = np.interp(tt - lag, d.vt, d.pwm[:, 0])
    if d.has_ai and len(d.at):
        ai = d.ai_at(tt - lag)
        try:
            out = np.interp(tt - lag, d.at, d.a_out[:, 0])
            st = np.where(ai, out, st)
        except AttributeError:
            pass
    v = -np.interp(tt, d.vt, d.erpm) * 0.5e-3
    u = (st - center) / 500.0
    return float(np.sum(v * (a * u + b * u ** 3)) * 0.01)


# ---------------------------------------------------------------- 1区間
class Segment(object):
    """1区間の格子(FS Hz)・推測航法・当てはめ"""

    def __init__(self, d, psi, sv, a, b, bridges, cl, walls, prm=None, g0=1.0):
        self.d = d
        self.a, self.b = a, b
        self.t = np.arange(a, b, 1.0 / FS)
        self.psi_deg = np.interp(self.t, d.it, psi)
        s = np.interp(self.t, d.vt, sv)
        self.s = s - s[0]
        self.cl, self.walls = cl, walls
        self.prm = dict(PARAMS, **(prm or {}))
        # 向きの段差: ジャイロの振り切れ(±1000 deg/s)と、つないだ IMU の欠け
        m = (d.it >= a) & (d.it <= b)
        sat = np.abs(d.gyr[m, 2]) >= 990
        ramps = []
        for i0, i1 in _runs(sat):
            ta, tb = d.it[m][i0] - 0.02, d.it[m][i1 - 1] + 0.02
            if ramps and ta - ramps[-1][1] < 0.3:
                ramps[-1] = (ramps[-1][0], tb)
            else:
                ramps.append((ta, tb))
        self.n_sat = len(ramps)
        br = [(x0, x1) for x0, x1 in bridges if a < x0 < b]
        ramps += br
        self.ramps = ramps
        # 節点: knot_dt ごと + 両端
        n = max(1, int(np.ceil((b - a) / self.prm['knot_dt'])))
        self.knots = np.linspace(a, self.t[-1], n + 1)
        self.g0 = g0
        self.F = Fitter(self.t, np.radians(self.psi_deg), self.s, self.knots, ramps, cl, walls, self.prm, g0=g0)
        # つないだ IMU の欠けの向きの変化は、ステアリングと速さから予測した値を事前分布の平均にする
        for j, (x0, x1) in enumerate(br):
            self.F.c0[self.n_sat + j] = steer_yaw(d, x0, x1)
            self.F.csd[self.n_sat + j] = np.radians(25.0)
        self.x = None
        self.reg = None
        self.info = {}

    def register(self, win=20.0, step=10.0):
        """全体探索。長い区間(30 秒より長い)は 20 秒の窓(10 秒ずらし)ごとに探す(周ごとに距離の倍率が違うため)"""
        self.reg = register(self.psi_deg, self.s, self.cl)
        self.windows = []
        if self.reg is None:
            return None
        if self.t[-1] - self.t[0] <= 30.0:
            self.windows = [(0, len(self.t), self.reg)]
        else:
            starts = np.arange(self.t[0], self.t[-1] - win / 2, step)
            for ta in starts:
                i0 = int(np.searchsorted(self.t, ta))
                i1 = int(np.searchsorted(self.t, min(ta + win, self.t[-1] + 1)))
                r = register(self.psi_deg[i0:i1], self.s[i0:i1] - self.s[i0], self.cl)
                if r is not None and r['score'] > 0.85:
                    self.windows.append((i0, i1, r))
            if not self.windows:
                self.windows = [(0, len(self.t), self.reg)]
        return self.reg

    def init_from_reg(self):
        """全体探索の結果から初期値を作り、その対応(基準線上の点)へ引き寄せる粗い当てはめをする"""
        F = self.F
        sa = np.r_[0, np.cumsum(np.abs(np.diff(self.s)))]
        # 点ごとに、点を含む窓のうち一致度の高いものを使う
        best = np.full(len(self.t), -1)
        bsc = np.full(len(self.t), -1.0)
        for w, (i0, i1, r) in enumerate(self.windows):
            m = np.zeros(len(self.t), bool)
            m[i0:i1] = True
            up = m & (r['score'] > bsc)
            best[up] = w
            bsc[up] = r['score']
        # 窓のない端は近い窓で埋める
        idx = np.nonzero(best >= 0)[0]
        best = best[idx[np.clip(np.searchsorted(idx, np.arange(len(self.t))), 0, len(idx) - 1)]]
        tg = np.full((len(self.t), 2), np.nan)
        h0 = np.full(len(self.t), np.nan)
        for w, (i0, i1, r) in enumerate(self.windows):
            m = best == w
            if not m.any():
                continue
            tg[m], _ = self.cl.at(r['s0'] + r['k'] * (sa[m] - sa[i0]))
            h0[m] = r['g'] * np.radians(self.psi_deg[m] - self.psi_deg[i0]) + r['psi0'] + 0.0
        # 窓ごとの ψ0 は、その窓の最初を 0 にした ψ に対するもの → 全体の ψ に直してから節点の向きの補正 A に
        A = h0 - np.radians(self.psi_deg)
        A = np.unwrap(A)
        kidx = np.clip(np.searchsorted(self.t, F.T), 0, len(self.t) - 1)
        x = F.init(0.0, 0.0, 0.0)
        x[0] = 1.0
        S = F.sl()
        x[S['A']] = A[kidx]
        X, Z, _ = F.pose(x)
        U = F._hat()
        wsum = U.sum(0) + 1e-9
        x[S['Bx']] = (U * (tg[:, 0] - X)[:, None]).sum(0) / wsum
        x[S['Bz']] = (U * (tg[:, 1] - Z)[:, None]).sum(0) / wsum
        F.set_targets(tg[F.ipt], 0.3)
        x = F.fit(x, n_outer=3, n_gn=3)
        F.set_targets(None)
        return x

    def fit(self, x=None, events=None, log=None, reject=False, n_outer=None):
        if x is None:
            x = self.init_from_reg()
            self.F.prm = dict(self.prm, wall_w=0.0)       # 壁なしで寄せてから、壁を入れる
            x = self.F.fit(x, n_outer=4)
            self.F.prm = dict(self.prm)
        self.F.set_events(events or [])
        self.F.reject = reject
        self.x = self.F.fit(x, n_outer=n_outer, log=log)
        return self.x

    def pose(self, x=None):
        return self.F.pose(self.x if x is None else x)


def fit_log_stage1(log, prm=None, verbose=False):
    d = LogData(log)
    psi, sv, dri = dead_reckoning(d)
    segs, cuts, bridges = segments(d, psi)
    cl = cm.Centerline()
    walls = cm.wall_segments()
    out = []
    for a, b in segs:
        S = Segment(d, psi, sv, a, b, bridges, cl, walls, prm)
        dist = float(np.abs(np.diff(S.s)).sum())
        if dist < 6.0 or len(S.t) < 50:
            out.append((S, {'fitted': False, 'reason': 'short', 'dist': dist}))
            continue
        r = S.register()
        if r is None:
            out.append((S, {'fitted': False, 'reason': 'register', 'dist': dist}))
            continue
        t0 = time.time()
        lg = []
        S.fit(log=lg)
        out.append((S, {'fitted': True, 'dist': dist, 'reg': r, 'fit': lg[-1], 'sec': time.time() - t0}))
        if verbose:
            print(log, round(a, 1), round(b, 1), json.dumps(out[-1][1], default=float))
    return d, out


# ---------------------------------------------------------------- カメラの目印
def frame_pose(seg, x, ct):
    """区間の当てはめから、カメラのコマの時刻 ct での (X, Z, h, s)。区間の外は NaN"""
    X, Z, h = seg.F.pose(x)
    m = (ct >= seg.t[0]) & (ct <= seg.t[-1])
    out = [np.full(len(ct), np.nan) for _ in range(4)]
    for k, v in enumerate((X, Z, np.unwrap(h), seg.s)):
        out[k][m] = np.interp(ct[m], seg.t, v)
    return out


def _gate_mask(kind, X, Z, H, pad=0.3, dh=50):
    L = cm.LANDMARKS[kind]
    x0, x1, z0, z1 = L['region']
    return ((X > x0 - pad) & (X < x1 + pad) & (Z > z0 - pad) & (Z < z1 + pad) &
            (np.abs(cm.wrap(H - L['heading'])) < np.radians(dh)) & np.isfinite(X))


def detect_events(F, ct, poses, models, kinds=None):
    """全区間のカメラのコマの姿勢 poses = (X, Z, H, S, seg_id) から、目印を横切った瞬間を探す。
    戻り値: [{'kind', 't', 'axis', 'value', 'group', 'seg', 'n', 'spread', 'pred'}]"""
    import course_landmarks as clm
    X, Z, H, S, sid = poses
    kinds = kinds or clm.KINDS
    ev = []
    for kind in kinds:
        L = cm.LANDMARKS[kind]
        b = clm.boundary_rows(F, kind)
        mask = _gate_mask(kind, X, Z, H)
        model = models.get('gate' if kind == 'gate' else 'floor')
        for p in clm.find_passes(b, mask, kind):
            if len(set(sid[p])) != 1:
                continue
            sx, sd = clm.crossing(p, b, S, kind, model)
            # 走行距離 sx になった時刻(通過の前後のコマの中で)
            q = np.arange(max(p[0] - 2, 0), min(p[-1] + 3, len(S)))
            q = q[sid[q] == sid[p[0]]]
            ss, tt = S[q], ct[q]
            if np.any(np.diff(ss) <= 0):
                k = np.nonzero(np.r_[True, np.diff(ss) > 0])[0]
                ss, tt = ss[k], tt[k]
            if len(ss) < 2 or sx < ss[0] - 0.3 or sx > ss[-1] + 0.3:
                continue
            te = float(np.interp(sx, ss, tt, left=np.nan, right=np.nan))
            if not np.isfinite(te):
                v = (ss[-1] - ss[0]) / max(tt[-1] - tt[0], 1e-3)
                te = float(tt[-1] + (sx - ss[-1]) / max(v, 0.2)) if sx > ss[-1] else float(tt[0] + (sx - ss[0]) / max(v, 0.2))
            i = p[-1]
            dd = sx - S[i]
            pred = (X[i] + dd * np.cos(H[i]), Z[i] + dd * np.sin(H[i]))
            ev.append({'kind': kind, 't': te, 'axis': L['axis'], 'group': LM_GROUP[kind], 'seg': int(sid[i]),
                       'n': int(len(p)), 'spread': sd, 's': sx, 'pred': pred, 'frame': int(i)})
    ev.sort(key=lambda e: e['t'])
    _assign_values(ev)
    return [e for e in ev if e.get('value') is not None]


def _assign_values(ev, D_floor=0.25, D_gate=1.5):
    """目印の値(どの線か)を決める。スタートラインは同じ通過の中の間隔(1.80 m)も使って3本をまとめて決める"""
    lines = cm.LANDMARKS['line']['values']
    for e in ev:
        L = cm.LANDMARKS[e['kind']]
        if e['kind'] != 'line':
            D = D_gate if e['kind'] == 'gate' else D_floor
            ax = 0 if L['axis'] == 'x' else 1
            pc = e['pred'][ax] + D * (np.cos(L['heading']) if ax == 0 else np.sin(L['heading']))
            e['value'] = L['values'][0] if abs(pc - L['values'][0]) < 0.8 else None
    # スタートライン: 同じ区間で 2.5 秒以内に続くものを1回の通過にまとめる
    li = [e for e in ev if e['kind'] == 'line']
    groups = []
    for e in li:
        if groups and e['seg'] == groups[-1][-1]['seg'] and e['t'] - groups[-1][-1]['t'] < 2.5:
            groups[-1].append(e)
        else:
            groups.append([e])
    for g in groups:
        pc = np.array([e['pred'][0] + D_floor for e in g])
        ss = np.array([e['s'] for e in g])
        best = None
        n = len(g)
        # 連続する線の番号の割り当て(飛ばしもありうるので、走行距離の差から番号の差を決める)
        steps = np.r_[0, np.round(np.diff(ss) / 1.80).astype(int)]
        idx = np.cumsum(steps)
        if np.any(np.abs(np.diff(ss) - 1.80 * np.diff(idx)) > 0.45) or np.any(np.diff(idx) <= 0) and n > 1:
            idx = None
        if idx is not None:
            for j0 in range(-idx[-1], 3):
                lab = j0 + idx
                if lab.min() < 0 or lab.max() > 2:
                    continue
                cost = np.abs(pc - np.array(lines)[lab]).mean()
                if best is None or cost < best[0]:
                    best = (cost, lab)
        if best is not None and best[0] < 0.8:
            for e, l in zip(g, best[1]):
                e['value'] = lines[l]
        else:
            for e, p_ in zip(g, pc):
                k = int(np.argmin(np.abs(np.array(lines) - p_)))
                e['value'] = lines[k] if abs(lines[k] - p_) < 0.6 else None


# ---------------------------------------------------------------- 1ログの当てはめ
FOOTPRINT = (-0.08, 0.32, 0.10)   # 車体の範囲(推測航法の基準点 ≒ 後輪の軸から、後ろ 8 cm〜前 32 cm、左右 10 cm)


def footprint_wall_dist(x, z, h, walls, fp=FOOTPRINT):
    """車体の長方形と最も近い壁との距離 [m](触れている・めり込んでいれば 0)"""
    b, f, w = fp
    c, s = np.cos(h), np.sin(h)
    loc = np.array([[b, -w], [f, -w], [f, w], [b, w]])
    poly = np.stack([x + loc[:, 0] * c - loc[:, 1] * s, z + loc[:, 0] * s + loc[:, 1] * c], 1)
    edges = np.stack([poly, np.roll(poly, -1, 0)], 1)
    # 壁の端点 → 車体の辺、車体の頂点 → 壁、交差していれば 0
    if len(cm.crossings(np.vstack([poly, poly[:1]]), walls)):
        return 0.0
    d1, _ = cm.dist_to_segments(poly, walls)
    d2, _ = cm.dist_to_segments(walls.reshape(-1, 2), edges)
    return float(min(d1.min(), d2.min()))


def _segment_events(ev, seg_id, kinds_out=()):
    return [e for e in ev if e['seg'] == seg_id and e['kind'] not in kinds_out]


def fit_log(log, prm=None, validate=True, verbose=False, kinds=None):
    """1ログを当てはめる。戻り値: dict(d, segs[Segment], x, events, info, models, val)"""
    import course_landmarks as clm
    t_start = time.time()
    prm = dict(PARAMS, **(prm or {}))
    d = LogData(log)
    psi, sv, dri = dead_reckoning(d)
    segs, cuts, bridges = segments(d, psi)
    cl = cm.Centerline()
    walls = cm.wall_segments()
    Fc = clm.load(log) if os.path.exists(os.path.join(clm.CACHE, log + '.npz')) else clm.load_or_extract(log)
    ct = (Fc['timestamps_us'] - d.t0_us) / 1e6
    out = {'log': log, 'dr': dri, 'cuts': [list(c) for c in cuts], 'bridges': [list(b) for b in bridges]}
    S_list, info = [], []
    # 1段目: 全体探索 → 基準線と壁だけで当てはめ
    for k, (a, b) in enumerate(segs):
        S = Segment(d, psi, sv, a, b, bridges, cl, walls, prm)
        dist = float(np.abs(np.diff(S.s)).sum())
        inf = {'segment': k, 't0': a, 't1': b, 'dist_m': dist, 'fitted': False}
        if dist < 6.0 or len(S.t) < 100:
            inf['reason'] = '走行距離が短い (< 6 m)'
        else:
            r = S.register()
            if r is None:
                inf['reason'] = '全体探索ができない'
            else:
                inf['register'] = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}
                lg = []
                S.x1 = S.fit(log=lg)
                inf['stage1'] = lg[-1]
                inf['fitted'] = True
        S_list.append(S)
        info.append(inf)
    fitted = [k for k, i in enumerate(info) if i['fitted']]

    def poses(xs):
        X = np.full(len(ct), np.nan)
        Z, H, Sd = X.copy(), X.copy(), X.copy()
        sid = np.full(len(ct), -1)
        for k in fitted:
            x_, z_, h_, s_ = frame_pose(S_list[k], xs[k], ct)
            m = np.isfinite(x_)
            X[m], Z[m], H[m], Sd[m] = x_[m], z_[m], h_[m], s_[m]
            sid[m] = k
        return X, Z, H, Sd, sid
    # 行 → 距離のモデルをログごとに合わせる(スタートラインの通過が 8 回以上あれば)
    P1 = poses({k: S_list[k].x1 for k in fitted})
    models = {'floor': clm.FLOOR_MODEL, 'gate': clm.GATE_MODEL}
    mi = {}
    bl = clm.boundary_rows(Fc, 'line')
    ps = [p for p in clm.find_passes(bl, _gate_mask('line', *P1[:3]), 'line') if len(set(P1[4][p])) == 1]
    if len(ps) >= 8:
        r = clm.fit_row_model([(bl[p], P1[3][p]) for p in ps], 'line')
        if r and r['rms_m'] < 0.05:
            models['floor'] = (r['K'], r['vh'])
            mi['floor'] = r
    bg = clm.boundary_rows(Fc, 'gate')
    ps = [p for p in clm.find_passes(bg, _gate_mask('gate', *P1[:3]), 'gate') if len(set(P1[4][p])) == 1]
    if len(ps) >= 5:
        r = clm.fit_row_model([(bg[p], P1[3][p]) for p in ps], 'gate', grid=[models['floor'][1] + 0.0])
        if r and 10 < r['K'] < 400:
            models['gate'] = (r['K'], r['vh'])
            mi['gate'] = r
    out['row_models'] = {'floor': list(models['floor']), 'gate': list(models['gate']), 'fit': mi}
    kinds = kinds or clm.KINDS
    # 2段目: 目印を入れる
    ev = detect_events(Fc, ct, P1, models, kinds)
    for k in fitted:
        S = S_list[k]
        S.x2 = S.fit(x=S.x1.copy(), events=_segment_events(ev, k))
    # D をログで共通にする(区間ごとの D の中央値に固定)
    Dlog = {}
    for gi, gname in enumerate(D_GROUPS):
        vals = [S_list[k].x2[S_list[k].F.sl()['D']][gi] for k in fitted
                if any(e['group'] == gname for e in _segment_events(ev, k))]
        Dlog[gname] = float(np.median(vals)) if vals else D_PRIOR[gname][0]
    for k in fitted:
        S_list[k].F.D0 = np.array([Dlog[g] for g in D_GROUPS])
        S_list[k].F.Dsd = np.array([0.01 for g in D_GROUPS])
    # 3段目: 2段目の軌跡で目印を探し直して、外れ値を除いて当てはめる
    P2 = poses({k: S_list[k].x2 for k in fitted})
    ev = detect_events(Fc, ct, P2, models, kinds)
    for k in fitted:
        S = S_list[k]
        x0 = S.x2.copy()
        x0[S.F.sl()['D']] = S.F.D0
        lg = []
        S.x3 = S.fit(x=x0, events=_segment_events(ev, k), reject=True, log=lg)
        info[k]['stage3'] = lg[-1]
    # 目印の残差(当てはめに使ったもの)
    for k in fitted:
        S = S_list[k]
        S.F.set_events(_segment_events(ev, k))
        r = S.F.event_residuals(S.x3) if S.F.events else np.zeros(0)
        for e, rr in zip(S.F.events, r):
            e_ = next(x for x in ev if x is e or (x['t'] == e['t'] and x['kind'] == e['kind']))
            e_['res_fit'] = float(rr)
    out['D_log'] = Dlog
    # 保留検証: 目印の種類を1つずつ外して当てはめ直し、外した目印の残差を見る
    val = []
    if validate:
        for kind in kinds:
            held = [e for e in ev if e['kind'] == kind]
            if not held:
                continue
            for k in sorted(set(e['seg'] for e in held)):
                S = S_list[k]
                rest = _segment_events(ev, k, kinds_out=(kind,))
                x = S.fit(x=S.x1.copy(), events=rest, reject=True)
                hk = [e for e in held if e['seg'] == k]
                S.F.set_events(hk)
                r = S.F.event_residuals(x)
                for e, rr in zip(hk, r):
                    val.append({'kind': kind, 't': e['t'], 'seg': k, 'res': float(rr), 'res_fit': e.get('res_fit')})
        # 当てはめを最終に戻す
        for k in fitted:
            S_list[k].F.set_events(_segment_events(ev, k))
    out['segments'] = info
    out['events'] = [{kk: (float(v) if isinstance(v, (np.floating, float)) else v) for kk, v in e.items() if kk != 'pred'} for e in ev]
    out['validation'] = val
    out['sec'] = time.time() - t_start
    return {'d': d, 'segs': S_list, 'fitted': fitted, 'ct': ct, 'out': out, 'cl': cl, 'walls': walls, 'Fc': Fc,
            'psi': psi, 'sv': sv}
