"""safety2-1 の確かめ: 記録を今の assist.Assist に通して(replay_1023_recovery.py と同じ入れ方)、リカバリーの前後の段・出力・eRPM・衝撃を並べる。
使い方: python3 -I replay_scene.py <場面のディレクトリ> <始まりの行> [前の行数] [後の行数]。読むだけ"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import assist as A   # noqa: E402

D = Path(sys.argv[1]); r0 = int(sys.argv[2]); pre = int(sys.argv[3]) if len(sys.argv) > 3 else 40; post = int(sys.argv[4]) if len(sys.argv) > 4 else 60
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
mism = 0
for i in range(len(t)):
    if hold[i]:
        a.launch_hold(); a.reset()
        continue
    now_us = t[i] + (lat[i] * 1000 if np.isfinite(lat[i]) else 0)
    jv = np.searchsorted(vt, t[i], side='right') - 1
    e = ve[jv] if jv >= 0 else 0.0
    m = (it > t[i] - 150_000) & (it <= now_us)
    ig = float(hg[m][-10:].max()) if m.any() else 0.0
    o, k = a.step((mo[i, 0], mo[i, 1]), e, bool(sel[i]), now_us / 1e6, esc_on=p['esc_reverse'] == 1,
                  rec_on=p['recovery'] == 1, impact_g=ig, rev_us=p['rev_us'], rev_s=p['rev_s'], kick_on=False)
    same = bool(abs(o[1] - out[i, 1]) < 0.6 and k == kind[i])
    mism += int(not same)
    if r0 - pre <= i <= r0 + post:
        print(i, round((t[i] - t[r0]) / 1e6, 3), a.rec.phase, 'model', round(mo[i, 1]), 'out', round(o[1]), 'rec', round(out[i, 1]),
              'same' if same else 'DIFF', 'erpm', int(e), 'g', round(ig, 2), 'backed', round(a.rec.backed_s, 2))
print('mismatch_all', mism, 'rows', len(t), 'params', {k: p.get(k) for k in ('recovery', 'esc_reverse', 'rev_us', 'rev_s', 'tof_guard', 'start_kick')})
