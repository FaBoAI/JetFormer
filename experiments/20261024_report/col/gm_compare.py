#!/usr/bin/env python3
"""GM(緑マットの左 U ターン)の中のステア・ヨー・速さを、記録ごとに(読むだけ)。
No.9(16:20)と 3 回目の同じ設定(B 0.90)で、同じステアに対する曲がり方(ヨー ÷ 速さ = 曲率)が変わったかを見る。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M python3 -I gm_compare.py → gm_compare.json"""
import json
from pathlib import Path
import numpy as np
HOME = Path('/home/jetson')
LAB = HOME / 'jetformer/data/labels'
RAW = HOME / 'jetformer/data/raw'
OUT = Path(__file__).resolve().parent
d = json.loads((OUT / 'col_data.json').read_text(encoding='utf-8'))
rows = []
for log, info in sorted(d['logs'].items()):
    lj = json.loads((LAB / (log + '.json')).read_text(encoding='utf-8'))
    t0 = int(lj['t0_us'])
    with np.load(LAB / (log + '.npz')) as z:
        L = {k: z[k] for k in ('t_s', 'loc_detail', 'loc_code', 'out', 'ai', 'speed_mps')}
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
    g = np.interp(L['t_s'], it, gz)
    steer = L['out'][:, 0] - info['center']
    for code, name in ((2, 'GM'), (1, 'R1'), (5, 'SW')):
        m = (L['loc_detail'] == code) if name == 'SW' else (L['loc_code'] == code)
        m = m & L['ai'] & (L['speed_mps'] > 1.5)
        if m.sum() < 30:
            continue
        v = L['speed_mps'][m]
        k = np.deg2rad(g[m]) / v                    # 曲率 [1/m](+ = 左)
        st = steer[m]
        sel = np.abs(st) > 100
        rows.append({'log': log, 'ai_no': info['ai_no'], 'session': info['session'], 'setting': info['setting'], 'loc': name,
                     'frames': int(m.sum()), 'steer_mean_us': round(float(st.mean()), 1), 'v_mean_mps': round(float(v.mean()), 2),
                     'yaw_mean_dps': round(float(g[m].mean()), 1),
                     'curv_per_100us': round(float(np.median(k[sel] / st[sel] * 100)), 4) if sel.sum() > 10 else None})
(OUT / 'gm_compare.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
for r in rows:
    print(r['ai_no'], r['session'], r['setting'], r['loc'], r['frames'], 'steer', r['steer_mean_us'], 'v', r['v_mean_mps'], 'yaw', r['yaw_mean_dps'], 'curv/100us', r['curv_per_100us'])
