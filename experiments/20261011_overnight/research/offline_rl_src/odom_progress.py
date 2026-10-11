"""報酬「コース上の進み」を、自己位置なしで作れるかの確認(読み取りのみ)。
ジャイロZ(deg/s, 左+)と eRPM(負=前進, 0.5e-3 m/s/eRPM)で各周の軌跡を積分し、
(1) 1周の閉じ誤差、(2) 周の長さのばらつき、(3) 同じログの周同士の横ずれ(最近傍距離)を測る。
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.laps import LapCounter
RAW = '/home/jetson/jetformer/data/raw'
K = 0.5e-3
out = {}
logs = sys.argv[1:] or ['20261010-200916', '20261010-201626', '20261010-202633', '20261010-152405', '20261010-152707', '20261010-172910', '20261010-173445', '20261010-204208']
for log in logs:
    T, GZ, E, S = [], [], [], []
    for sc in sorted(glob.glob(os.path.join(RAW, log, log + '-*'))):
        try:
            V = np.load(os.path.join(sc, 'VEHICLE.npz')); I = np.load(os.path.join(sc, 'IMU.npz'))
        except Exception:
            continue
        it = I['timestamps_us'] / 1e6; vt = V['timestamps_us'] / 1e6
        T.append(it); GZ.append(I['gyro'][:, 2]); E.append(np.interp(it, vt, V['erpm'].astype(float)))
        S.append(V['select'][np.searchsorted(vt, it).clip(0, len(vt) - 1)])
    t = np.concatenate(T); gz = np.concatenate(GZ).astype(float); e = np.concatenate(E); s = np.concatenate(S)
    lc = LapCounter(); ends = []
    for k in range(len(t)):
        lap = lc.add(t[k], gz[k], e[k], ai=bool(s[k]))
        if lap: ends.append((lap['t_end'], lap['time_s'], lap['ai_only'], lap['impacts'], lap['takeovers']))
    # 走行全体を積分(バイアスはLapCounterの推定値を使う)
    bias = lc.bias
    dt = np.diff(t, prepend=t[0]); dt[(dt <= 0) | (dt > 0.2)] = 0
    yaw = np.cumsum(np.deg2rad(gz - bias) * dt)
    v = -e * K
    x = np.cumsum(v * np.cos(yaw) * dt); y = np.cumsum(v * np.sin(yaw) * dt)
    laps = []
    for te, ts, ai, imp, tk in ends:
        a = np.searchsorted(t, te - ts); b = np.searchsorted(t, te)
        if b - a < 50: continue
        L = float(np.sum(np.abs(v[a:b]) * dt[a:b]))
        clo = float(np.hypot(x[b - 1] - x[a], y[b - 1] - y[a]))
        laps.append(dict(t0=t[a], t1=t[b - 1], time=ts, len=L, closure=clo, ai=bool(ai), imp=int(imp), tk=int(tk), a=int(a), b=int(b)))
    # 周同士の横ずれ: 各周を始点に平行移動+閉じ誤差を線形に配分し、最初のきれいな周を基準に最近傍距離
    def traj(l):
        a, b = l['a'], l['b']
        px, py = x[a:b] - x[a], y[a:b] - y[a]
        w = np.linspace(0, 1, b - a)
        px = px - w * px[-1]; py = py - w * py[-1]
        return np.stack([px, py], 1)
    clean = [l for l in laps if l['imp'] == 0 and l['tk'] == 0]
    dev = []
    if len(clean) >= 2:
        ref = traj(clean[0])[::5]
        for l in clean[1:]:
            P = traj(l)[::5]
            d = np.sqrt(((P[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
            dev.append((float(np.median(d)), float(np.percentile(d, 90))))
    r = dict(n_laps=len(laps), n_clean=len(clean),
             lap_len_med=float(np.median([l['len'] for l in laps])) if laps else None,
             lap_len_iqr=[float(np.percentile([l['len'] for l in laps], q)) for q in (25, 75)] if laps else None,
             closure_med=float(np.median([l['closure'] for l in laps])) if laps else None,
             closure_p90=float(np.percentile([l['closure'] for l in laps], 90)) if laps else None,
             closure_clean_med=float(np.median([l['closure'] for l in clean])) if clean else None,
             xtrack_med_of_med=float(np.median([d[0] for d in dev])) if dev else None,
             xtrack_med_of_p90=float(np.median([d[1] for d in dev])) if dev else None,
             lap_time_med=float(np.median([l['time'] for l in laps])) if laps else None)
    out[log] = r
    print(log, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'odom_progress.json'), 'w'), indent=1)
