"""反証 safety2-1: 今の jetformer/assist.py(作業ツリー)に 10/23 205458-0002 の記録の入力を通し、記録の out・assist と比べる。
さらに、リカバリーの各段で eRPM(後ろへ動いたか)・backed_s を並べる。読むだけ。装置は開かない"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import assist as A   # noqa: E402
rec = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/20261023-205458/20261023-205458-0002'
D = ROOT / rec
z = np.load(D / 'AI_CMD.npz'); v = np.load(D / 'VEHICLE.npz'); im = np.load(D / 'IMU.npz')
L = json.loads((D.parent / 'log.json').read_text())
p = L['drive']['params']; ref = (float(L['drive']['steer_center_us']), float(L['drive']['throttle_neutral_us']))
t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool); hold = z['hold'].astype(bool)
dry = z['dry_run'].astype(bool)
mo = z['model_out'].astype(float); out = z['out'].astype(float); kind = z['assist'].astype(int); lat = z['latency_ms'].astype(float)
vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
a = A.Assist(ref)
mism = []
rows = []
for i in range(len(t)):
    if hold[i]:
        a.launch_hold(); a.reset(); continue
    now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
    jv = np.searchsorted(vt, now_us, side='right') - 1      # 処理の時刻までの最新の車両の値
    e = ve[jv] if jv >= 0 else 0.0
    ji = np.searchsorted(it, now_us, side='right')
    idx = np.arange(max(0, ji - 10), ji)
    idx = idx[it[idx] > t[i] - 150_000]
    ig = float(hg[idx].max()) if len(idx) else 0.0
    ph0 = a.rec.phase
    o, k = a.step((mo[i, 0], mo[i, 1]), e, bool(sel[i]) and not dry[i], now_us / 1e6, esc_on=p.get('esc_reverse', 1) == 1,
                  rec_on=p.get('recovery', 1) == 1, impact_g=ig, rev_us=float(p.get('rev_us', 300)),
                  rev_s=float(p.get('rev_s', 0.4)), kick_on=int(p.get('start_kick', 0) or 0) == 1)
    if abs(o[1] - out[i, 1]) >= 0.6 or abs(o[0] - out[i, 0]) >= 0.6 or k != kind[i]:
        mism.append(i)
    if k == A.ASSIST_RECOVERY or ph0 is not None:
        rows.append((i, a.rec.phase, round(o[1]), round(out[i, 1]), int(e), round(ig, 2), round(a.rec.backed_s, 3)))
print(json.dumps({'rec': rec, 'params': {k: p.get(k) for k in ('recovery', 'esc_reverse', 'rev_us', 'rev_s', 'start_kick', 'tof_guard')},
                  'rows': len(t), 'mismatch': len(mism), 'mismatch_rows': mism[:20], 'recoveries': a.recoveries}, ensure_ascii=False))
for r in rows:
    print(r)
