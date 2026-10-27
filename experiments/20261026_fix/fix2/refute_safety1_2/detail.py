"""反証 safety1-2(読むだけ): 1 つの AI の区間の始まりの前後を行ごとに出す(TTC・eRPM・出力・モデル・assist・衝撃)"""
import glob, json, os, sys
import numpy as np
root = '/home/jetson/jetformer/data/raw'
lg, t_ai, before, after = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])


def load(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    except Exception:
        return None


A, V, I = [], [], []
for sc in sorted(glob.glob(os.path.join(root, lg, '*', 'AI_CMD.npz'))):
    d = os.path.dirname(sc)
    a, v, im = load(sc), load(d + '/VEHICLE.npz'), load(d + '/IMU.npz')
    if a is None or v is None or im is None:
        continue
    A.append(a); V.append(v); I.append(im)
a = {k: np.concatenate([x[k] for x in A]) for k in ['timestamps_us', 'assist', 'select', 'hold', 'ttc', 'out', 'model_out', 'guard']}
vt = np.concatenate([x['timestamps_us'] for x in V]) / 1e6; erpm = np.concatenate([x['erpm'] for x in V]).astype(float)
it = np.concatenate([x['timestamps_us'] for x in I]) / 1e6; acc = np.concatenate([x['accel'] for x in I])
gh = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
t = a['timestamps_us'] / 1e6
o = np.argsort(vt); vt, erpm = vt[o], erpm[o]
for i in np.where((t >= t_ai - before) & (t <= t_ai + after))[0]:
    w = (it > t[i] - 0.034) & (it <= t[i])
    print('%6d %+6.2f sel=%d hold=%d assist=%d ttc=%6.2f erpm=%6.0f v=%5.2f out=(%4.0f,%4.0f) model=(%4.0f,%4.0f) g=%.2f' % (
        i, t[i] - t_ai, a['select'][i], a['hold'][i], a['assist'][i], a['ttc'][i], np.interp(t[i], vt, erpm),
        -np.interp(t[i], vt, erpm) * 0.5e-3, a['out'][i, 0], a['out'][i, 1], a['model_out'][i, 0], a['model_out'][i, 1],
        gh[w].max() if w.any() else float('nan')))
