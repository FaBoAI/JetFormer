"""反証 2 回目・safety: 10/23 20261023-205458-0002 のリカバリー(行 103〜129)を、今の assist.Assist に記録の値
(モデルの出力 model_out・eRPM・IMU の衝撃・時刻)で通し、記録の out と同じになるかを見る(今のコードでも同じことが起きるか)。
そのうえで、リカバリーの各段で車が実際にどう動いたか(eRPM)を並べる。読むだけ。装置は開かない"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from jetformer import assist as A   # noqa: E402

D = ROOT / 'data/raw/20261023-205458/20261023-205458-0002'
OUT = Path(__file__).resolve().parent / 'out' / 'replay_1023_recovery.json'
z = np.load(D / 'AI_CMD.npz'); v = np.load(D / 'VEHICLE.npz'); im = np.load(D / 'IMU.npz')
t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool); hold = z['hold'].astype(bool)
mo = z['model_out'].astype(float); out = z['out'].astype(float); kind = z['assist'].astype(int)
lat = z['latency_ms'].astype(float)
vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
p = json.loads((D.parent / 'log.json').read_text())['drive']['params']
ref = (1441.0, 1496.0)
a = A.Assist(ref)
rows, mism = [], 0
for i in range(len(t)):
    if hold[i]:
        a.launch_hold(); a.reset()
        continue
    now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
    jv = np.searchsorted(vt, t[i], side='right') - 1          # 撮像時刻までの最新の車両の値(走行ループと同じ)
    e = ve[jv] if jv >= 0 else 0.0
    m = (it > t[i] - 150_000) & (it <= now_us)
    ig = float(hg[m][-10:].max()) if m.any() else 0.0
    o, k = a.step((mo[i, 0], mo[i, 1]), e, bool(sel[i]), now_us / 1e6, esc_on=p['esc_reverse'] == 1,
                  rec_on=p['recovery'] == 1, impact_g=ig, rev_us=p['rev_us'], rev_s=p['rev_s'], kick_on=False)
    same = bool(abs(o[1] - out[i, 1]) < 0.6 and k == kind[i])
    mism += int(not same)
    if 95 <= i <= 131:
        rows.append({'row': i, 's': round((t[i] - t[103]) / 1e6, 3), 'phase': a.rec.phase, 'out_now': round(o[1]),
                     'out_rec': round(out[i, 1]), 'same': same, 'erpm': int(e), 'impact_g': round(ig, 2)})
back = [r for r in rows if r['erpm'] > 300 and r['out_now'] > 1536 and r['row'] >= 103]
res = {'scene': str(D.relative_to(ROOT)), 'params': {k: p[k] for k in ('recovery', 'esc_reverse', 'rev_us', 'rev_s', 'throttle_gain')},
       'rows_total': int(len(t)), 'mismatch_rows_all': mism, 'window': rows,
       'reverse_side_while_backing_frames': len(back),
       'reverse_side_while_backing_s': round(back[-1]['s'] - back[0]['s'] + 1 / 30, 2) if back else 0.0,
       'max_back_erpm': max(r['erpm'] for r in rows), 'max_back_mps': round(max(r['erpm'] for r in rows) * 0.5e-3, 2),
       'impact_behind_g': round(float(hg[(it > t[103]) & (it < t[103] + 1_000_000)].max()), 2)}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != 'window'}, ensure_ascii=False))
for r in rows:
    print(r)
