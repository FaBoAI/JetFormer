"""fix8 係 tools 直す 8-1 回目(読むだけ): 本物の 10/23・10/24 の記録で、人側・スティック中立(|PWM − 中立| < 8 µs)・前へ 1 m/s より速い
区間(0.3 秒以上)の車輪の速さの傾き(中立の惰性の減速)を出す。作り物の「本物に近い速さ」(tests/test_fix8_tools_r1.sday の R)の −0.9 m/s² の元。
出力 ../out/coast_decel_real.txt

  PYTHONDONTWRITEBYTECODE=1 jf-heavy --lock none --mem 1500M -- python -B coast_decel_real.py
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer/tools')
import label_events as le
RAW = Path('/home/jetson/jetformer/data/raw')
res = []
for p in sorted(RAW.iterdir()):
    if not p.is_dir() or p.name[:8] not in ('20261023', '20261024'):
        continue
    try:
        L = le.load_log(p.name, RAW)
    except Exception:
        continue
    V = L['VEHICLE']
    if V is None or len(V['timestamps_us']) < 100:
        continue
    meta = L['meta'] or {}
    n = float((meta.get('drive') or {}).get('throttle_neutral_us') or 1496.0)
    t = (V['timestamps_us'].astype(np.int64) - int(V['timestamps_us'][0])) / 1e6
    v = -V['erpm'].astype(float) * 0.5e-3
    sel = V['select'].astype(bool)
    thr = V['pwm'][:, 1].astype(float)
    # human side, stick neutral (|d| < 8 us), moving forward > 1 m/s, for >= 0.3 s
    m = (~sel) & (np.abs(thr - n) < 8) & (v > 1.0)
    for i0, i1 in le.runs(m):
        if t[i1] - t[i0] < 0.3:
            continue
        a = np.polyfit(t[i0:i1+1], v[i0:i1+1], 1)[0]
        res.append((p.name, round(float(t[i0]),2), round(float(t[i1]-t[i0]),2), round(float(v[i0]),2), round(float(a),2)))
for r in res[:40]:
    print(r)
if res:
    a = np.array([r[4] for r in res]); d = np.array([r[2] for r in res])
    print('n', len(res), 'decel median', np.median(a), 'weighted', np.sum(a*d)/np.sum(d), 'p10/p90', np.percentile(a,10), np.percentile(a,90))
