"""10/10 20261010-181504-0004 の記録を読むだけ。中立止めの行・衝撃・記録上のリカバリー。今の Assist に通して reset あり/なしを比べる"""
import json, sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A
d = '/home/jetson/jetformer/data/raw/20261010-181504/20261010-181504-0004/'
z = np.load(d + 'AI_CMD.npz'); v = np.load(d + 'VEHICLE.npz'); imu = np.load(d + 'IMU.npz')
t = z['timestamps_us'] / 1e6; tv = v['timestamps_us'] / 1e6; ti = imu['timestamps_us'] / 1e6
hg = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / 9.80665
holds = [int(i) for i in np.where(z['hold'])[0]]
rec_rows = [int(i) for i in np.where(z['assist'] == 2)[0]]
pk = int(np.argmax(hg))
print('keys', list(z.keys()))
print('holds', holds[:20], 'rec rows first', rec_rows[:3], 'n_rows', len(t))
print('peak G %.2f at t=%.3f (row near %d)' % (hg[pk], ti[pk], int(np.searchsorted(t, ti[pk]))))
for i in holds[:5]:
    print('hold row', i, 'dt from peak %.3f' % (t[i] - ti[pk]), 'select', bool(z['select'][i]))
ref = (1443.0, 1498.0)
def replay(reset, zero_noise=False):
    h = A.Assist(ref); first = None; outs = []
    for i in range(100, 200):
        if not z['select'][i]:
            return first, outs, i
        if z['hold'][i]:
            if reset:
                h.launch_hold(); h.reset()
            continue
        e = float(v['erpm'][np.searchsorted(tv, t[i], side='right') - 1])
        if zero_noise and i > holds[0] and abs(e) >= 1000:
            e = 0.0
        m = (ti > t[i] - 0.15) & (ti <= t[i] + 0.01)
        g = float(hg[m].max()) if m.any() else 0.0
        o, k = h.step(tuple(float(x) for x in z['model_out'][i]), e, True, float(t[i]), impact_g=g)
        if k == A.ASSIST_RECOVERY and first is None:
            first = i
        if 140 <= i <= 192: outs.append((i, int(k), round(float(o[1]))))
    return first, outs, None
for r in (True, False):
    for zn in (False, True):
        f, outs, end = replay(r, zn)
        print('reset' if r else 'noreset', 'zero_noise' if zn else 'raw', 'first_recovery_row', f, 'human_back_row', end)
i0 = holds[0]
print('erpm rows', [(i, int(v['erpm'][np.searchsorted(tv, t[i], side='right') - 1])) for i in range(i0 - 4, i0 + 30)])
print('model_out thr rows 143-176', sorted(set(int(round(x)) for x in z['model_out'][143:177, 1])))
