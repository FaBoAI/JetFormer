#!/usr/bin/env python3
"""シミュレーター較正(2026-10-16)のデータの読み込み: 実車のログ → 100 Hz の記録と、再生の窓

protocol.json(sha256 61fc91cb…)の data 章のとおりに作る。読み取りだけで、data/raw・data/labels は変えない。
実車・PCA9685・I2C には触れない。

- 記録は inventory.analyse_log(設計の段の関数、sha256 は protocol に記録)の値に、次を足したもの。
    v5      = movavg(−erpm × 0.5e-3, 5)        … 速さの「正解」
    r       = gyro_z [rad/s] − ずれ               … ヨーレートの「正解」(左が正)。ずれ = 当てはめのシーンの静止(|erpm| < 50 かつ |gyro_z| < 5 dps)の中央値
    a_lat   = movavg(−accel_x, 10) − ずれ         … 横加速度(左が正)。IMU の x は右が正
    u_s・u_v = ブリッジ(pwm_to_action)と同じ換算の指令(シミュレーターの単位)
- 分け方: シーンの番号 k(label_events.load_log の scenes の順)で k % 3 == 1 が保留(validation)、それ以外が当てはめ(fit)。
  part='fit' では、保留のシーンの値は読み込んだ直後に NaN にして、集計には一切入れない(brake_spans で指定した
  ブレーキの出来事の区間だけは別に取り出す。protocol I7 が 16 回を固定しているため)。
- 窓: 1 秒・0.5 秒ずらし。窓とその前 0.5 秒(準備)がすべて clean・指令あり・同じ操作者・同じ側(fit/保留)のシーン。
  始まりの速さ v5 ≥ 0.5 m/s。遅れの計算のため、準備のさらに前 0.2 秒の指令も持つ(欠けていれば窓を使わない)。
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import inventory as inv  # noqa: E402
import label_events as le  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016')
LOC = inv.LOC                                   # ['unknown','R1','GM','DE','other','HS','LH']
SEC_NAME = {'unknown': 'unknown', 'R1': 'R1', 'GM': 'GM', 'DE': 'DE', 'other': 'SW', 'HS': 'HS', 'LH': 'LH'}
EXCLUDED_SCENES = {'20261010-163247-0021', '20261010-163247-0022'}
STEER_C = {0: 1442.0, 1: 1443.0}                # 人・AI の舵の中心
THR_N = 1498.0
MAX_STEER = 0.52
MAX_SPEED = 7.24
DT = 0.01                                       # 再生のフレーム(100 Hz)
N_HIST, N_WARM, N_WIN = 20, 50, 100             # 遅れの履歴 0.2 秒・準備 0.5 秒・窓 1 秒
STRIDE = 50


def movavg(x, n):
    if n <= 1 or len(x) < n:
        return np.asarray(x, float)
    return np.convolve(x, np.ones(n) / n, mode='same')


def pwm_to_us(steer, op):
    """舵の PWM → u_s [rad](ブリッジと同じ: 左 2000 − 中心、右 中心 − 1000 で ±1 に正規化して × 0.52)"""
    c = np.where(op == 1, STEER_C[1], STEER_C[0])
    span = np.where(steer >= c, 2000.0 - c, c - 1000.0)
    return np.clip((steer - c) / span, -1.0, 1.0) * MAX_STEER


def pwm_to_uv(thr):
    """スロットルの PWM → u_v [m/s](thr ≤ 1498: 前進 498 µs で 7.24 m/s。thr > 1498: −(thr−1498)/502 × 0.8 = ブレーキ/後退の要求)"""
    thr = np.asarray(thr, float)
    return np.where(thr <= THR_N, (THR_N - thr) / 498.0 * MAX_SPEED, -(thr - THR_N) / 502.0 * 0.8)


def list_logs():
    return sorted(os.path.basename(p) for p in glob.glob(str(JF / 'data' / 'raw' / '2026*')) if len(os.path.basename(p)) == 15)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_record(log, part='fit', brake_spans=()):
    """1 本のログの 100 Hz の記録。part = 'fit'(保留のシーンを NaN に)・'heldout'(当てはめのシーンを NaN に)・'all'"""
    res, r = inv.analyse_log(log)
    if r is None:
        return None
    L = le.load_log(log)
    V, I = L['VEHICLE'], L['IMU']
    vt_us = V['timestamps_us'].astype(np.int64)
    it_us = I['timestamps_us'].astype(np.int64)
    gz_dps = np.interp(vt_us, it_us, I['gyro'][:, 2].astype(float))
    ax = np.interp(vt_us, it_us, I['accel'][:, 0].astype(float))
    scene = V['scene'].astype(int)
    scene_names = L['scenes']
    held = (scene % 3) == 1
    excl = np.array([scene_names[k] in EXCLUDED_SCENES for k in scene])
    if part == 'fit':
        keep = ~held
    elif part == 'heldout':
        keep = held
    else:
        keep = np.ones(len(scene), bool)
    keep &= ~excl
    # ブレーキの出来事(protocol I7 で固定の 16 回)は、保留のシーンでも区間だけ取り出せるようにする
    brake_keep = np.zeros(len(scene), bool)
    for (t_s, dur) in brake_spans:
        brake_keep |= (r['t'] >= t_s - 0.5) & (r['t'] <= t_s + dur + 0.5)
    v_raw = r['v']
    # 静止のずれ(当てはめのシーンだけで決める)
    still = (np.abs(V['erpm']) < 50) & (np.abs(gz_dps) < 5) & ~held & ~excl
    if still.sum() < 50:
        still = (np.abs(V['erpm']) < 50) & (np.abs(gz_dps) < 5) & ~excl
    gz_bias = float(np.median(gz_dps[still])) if still.any() else 0.0
    alat_raw = movavg(-ax, 10)
    alat_bias = float(np.median(-ax[still])) if still.any() else 0.0
    rec = {
        'log': log, 'ai_log': bool(res['ai_log']), 'scene_names': scene_names,
        't': r['t'], 't_us': vt_us, 'scene': scene, 'held': held, 'op': r['op'].astype(int), 'loc': r['loc'].astype(int),
        'clean': r['clean'].copy(), 'steer': r['steer'].copy(), 'thr': r['thr'].copy(), 'off': r['off'].copy(),
        'v_raw': v_raw, 'v5': movavg(v_raw, 5), 'vs': r['vs'], 'acc': r['acc'],
        'r': (gz_dps - gz_bias) * np.pi / 180.0, 'a_lat': alat_raw - alat_bias,
        'gz_bias_dps': gz_bias, 'alat_bias': alat_bias, 'dt': r['dt'],
        'latency_ms': res.get('latency_ms'),
    }
    rec['u_s'] = pwm_to_us(rec['steer'], rec['op'])
    rec['u_v'] = pwm_to_uv(rec['thr'])
    # 側の外の値を消す(集計に入らないように)
    rec['brake_only'] = brake_keep & ~keep
    hide = ~keep & ~brake_keep
    for k in ('clean',):
        rec[k] = rec[k] & keep
    for k in ('steer', 'thr', 'off', 'v_raw', 'v5', 'vs', 'acc', 'r', 'a_lat', 'u_s', 'u_v'):
        a = rec[k].astype(float).copy()
        a[hide] = np.nan
        rec[k] = a
    rec['keep'] = keep
    return rec


def battery_g(rec):
    """駆動の効き g = (1498 − thr)/(前進 eRPM/1000) を出せる時刻(inventory の drive_efficiency と同じ条件)。g は NaN 以外が有効"""
    dt = rec['dt']
    n1 = int(round(1.0 / dt))
    thr = rec['thr']
    sd = np.full(len(thr), np.inf)
    # 1 秒の移動の標準偏差(NaN を含む区間は無効)
    c1 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(thr))])
    c2 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(thr) ** 2)])
    cn = np.concatenate([[0], np.cumsum(np.isnan(thr).astype(int))])
    for j in range(len(thr)):
        a = max(0, j - n1)
        n = j + 1 - a
        if cn[j + 1] - cn[a] > 0:
            continue
        m = (c1[j + 1] - c1[a]) / n
        sd[j] = np.sqrt(max(0.0, (c2[j + 1] - c2[a]) / n - m * m))
    m = rec['clean'] & (thr < THR_N - 60) & (sd < 10) & (np.abs(rec['off']) < 120) & (np.abs(rec['acc']) < 0.5) & (rec['vs'] > 1.0)
    g = np.full(len(thr), np.nan)
    g[m] = (THR_N - thr[m]) / (rec['vs'][m] * 2.0)
    return g


def surface_of_loc(loc_name):
    """窓の区間 → シミュレーターの路面(protocol fidelity.replay.surface)"""
    return {'R1': 'shortcut', 'GM': 'turf'}.get(loc_name, 'carpet')


def make_windows(rec, g=None, g_ref=None):
    """再生の窓。始まり t0 は 0.5 秒ごとのサンプル。値は t0 からの等間隔 10 ms の格子に写す(指令は直前の値の保持、
    速さ・ヨーなどは線形補間)。格子の並び: k = −70〜100(履歴 20・準備 50・窓 100・終わり 1 = 171 点)"""
    n_all = N_HIST + N_WARM + N_WIN + 1
    kgrid = np.arange(-(N_HIST + N_WARM), N_WIN + 1)
    T = len(rec['t'])
    t = rec['t']
    out = []
    clean = rec['clean']
    cmd_ok = np.isfinite(rec['u_s']) & np.isfinite(rec['u_v'])
    for j0 in range(N_HIST + N_WARM + 5, T - N_WIN - 10, STRIDE):
        t0 = t[j0]
        grid = t0 + kgrid * DT
        ja = int(np.searchsorted(t, grid[N_HIST], side='right')) - 1      # 準備の始まり
        jh = int(np.searchsorted(t, grid[0], side='right')) - 1           # 履歴の始まり
        je = int(np.searchsorted(t, grid[-1], side='left'))               # 終わり
        if jh < 0 or je >= T:
            continue
        if not clean[ja:je + 1].all():
            continue
        if not cmd_ok[jh:je + 1].all():
            continue
        o = rec['op'][j0]
        if (rec['op'][jh:je + 1] != o).any():
            continue
        if not rec['keep'][jh:je + 1].all():
            continue
        if not (rec['v5'][j0] >= 0.5):
            continue
        if np.any(np.diff(t[jh:je + 1]) > 0.05):
            continue
        zi = np.clip(np.searchsorted(t, grid + 1e-9, side='right') - 1, 0, T - 1)   # 直前の値の保持
        kb = 1.0
        n_g = 0
        if g is not None and g_ref is not None:
            mm = (t >= t0 - 5.0) & (t < t0) & np.isfinite(g) & (rec['op'] == o)
            n_g = int(mm.sum())
            if n_g >= 30:
                kb = float(np.clip(g_ref[o] / np.median(g[mm]), 0.5, 1.2))
        locn = SEC_NAME[LOC[int(rec['loc'][j0])]]
        w = {'log': rec['log'], 'scene': int(rec['scene'][j0]), 'scene_name': rec['scene_names'][int(rec['scene'][j0])],
             'op': int(o), 'sec': locn, 'surface': surface_of_loc(locn), 't0': float(t0), 'k_batt': kb, 'n_g': n_g,
             't': kgrid * DT}
        for k in ('u_s', 'u_v', 'thr', 'off'):
            w[k] = rec[k][zi]
        for k, src in (('v', 'v5'), ('r', 'r'), ('a_lat', 'a_lat')):
            w[k] = np.interp(grid, t[jh:je + 1], rec[src][jh:je + 1])
        out.append(w)
    return out


def stack_windows(ws):
    """窓のリスト → 配列の dict"""
    if not ws:
        return None
    keys_arr = ['t', 'u_s', 'u_v', 'v', 'r', 'a_lat', 'thr', 'off']
    W = {k: np.stack([w[k] for w in ws]).astype(float) for k in keys_arr}
    for k in ('log', 'scene', 'scene_name', 'op', 'sec', 'surface', 't0', 'k_batt', 'n_g'):
        W[k] = np.array([w[k] for w in ws])
    W['n'] = len(ws)
    return W
