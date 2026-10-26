#!/usr/bin/env python3
"""記録ごとの「まっすぐ進むステア」(読むだけ)。AI が走っている(速さ 1.5 m/s 超)フレームで、
曲率 k = ヨー ÷ 速さ [1/m] をステア(out − 中央、+ = 左)の 0.1 秒前の値に直線で当てはめ、k = 0 になるステア s0 を出す。
s0 > 0 = 左に切らないとまっすぐ進まない(車が右へ流れる)。左右の曲がりやすさ(傾き)も左と右で別に出す。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M python3 -I steer_offset.py [day ...] → steer_offset.json"""
import json, sys
from pathlib import Path
import numpy as np
HOME = Path('/home/jetson')
LAB = HOME / 'jetformer/data/labels'
RAW = HOME / 'jetformer/data/raw'
OUT = Path(__file__).resolve().parent
LAG = 3          # 0.1 秒(30 fps)
days = sys.argv[1:] or ['20261024']
rows = []
for day in days:
    for pj in sorted(LAB.glob('%s-*.json' % day)):
        lj = json.loads(pj.read_text(encoding='utf-8'))
        if not lj.get('ai_log'):
            continue
        log = pj.stem
        t0 = int(lj['t0_us'])
        center = float(lj['ref_us']['steer_center'])
        with np.load(LAB / (log + '.npz')) as z:
            L = {k: z[k] for k in ('t_s', 'out', 'ai', 'speed_mps', 'collision', 'loc_code')}
        it, gz = [], []
        for s in sorted(p for p in (RAW / log).iterdir() if p.is_dir()):
            try:
                with np.load(s / 'IMU.npz') as z:
                    it.append((z['timestamps_us'] - t0) / 1e6); gz.append(z['gyro'][:, 2])
            except Exception:
                pass
        if not it:
            continue
        it, gz = np.concatenate(it), np.concatenate(gz)
        o = np.argsort(it); it, gz = it[o], gz[o]
        g = np.convolve(np.interp(L['t_s'], it, gz), np.ones(5) / 5, 'same')
        st = L['out'][:, 0] - center
        stl = np.r_[np.full(LAG, np.nan), st[:-LAG]]
        v = L['speed_mps']
        # 衝突の前後 1 秒は除く
        bad = np.convolve(L['collision'].astype(float), np.ones(61), 'same') > 0
        m = L['ai'] & (v > 1.5) & np.isfinite(stl) & ~bad & (np.abs(stl) < 450)
        if m.sum() < 200:
            continue
        k = np.deg2rad(g[m]) / v[m]
        x = stl[m]
        A = np.c_[x, np.ones(len(x))]
        c, b = np.linalg.lstsq(A, k, rcond=None)[0]
        s0 = -b / c
        # 左右別の傾き(s0 を中心に)
        lm, rm = x > s0 + 50, x < s0 - 50
        cl = float(np.median(k[lm] / (x[lm] - s0))) if lm.sum() > 30 else None
        cr = float(np.median(k[rm] / (x[rm] - s0))) if rm.sum() > 30 else None
        # まっすぐの区間: |ヨー| < 10 deg/s のときのステアの中央値
        sm = m & (np.abs(g) < 10)
        rows.append({'log': log, 'description': lj.get('description', '')[:40], 'frames': int(m.sum()),
                     's0_us': round(float(s0), 1), 'slope_per_100us': round(float(c * 100), 4),
                     'slope_left': None if cl is None else round(cl * 100, 4), 'slope_right': None if cr is None else round(cr * 100, 4),
                     'steer_when_straight_us': round(float(np.median(st[sm])), 1) if sm.sum() > 30 else None,
                     'straight_frames': int(sm.sum())})
        r = rows[-1]
        print(log, r['description'][:12], 'n', r['frames'], 's0', r['s0_us'], 'slope', r['slope_per_100us'], 'L', r['slope_left'], 'R', r['slope_right'],
              'straight steer', r['steer_when_straight_us'], r['straight_frames'])
(OUT / 'steer_offset.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
