"""正解の通過点・目標速度を TCPPost に入れたときの、人の操作との差(制御器と較正だけの誤差の下限)。CPU・numpy だけ"""
import json, os, sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.policies import tcp_labels as L, controller as C, tcp
ROOT = '/home/jetson/jetformer/data/datasets/clean_driving_20261011'
splits, _ = L.read_splits(ROOT)
lab = L.labels_by_scene(ROOT, keys=('waypoints', 'v_tgt', 'v_now', 'wp_valid'))
pp = tcp.postprocess_from(L.load_calibration(ROOT), {})
center, neutral = 1441.0, 1496.0

def chord_heading(wp, wp_dt, fps=30, steps=10, t_look=0.25, l_min=0.3):
    """spec の文字どおり: θ_k = その区間(弦)の向き"""
    orig = C._wrap
    wp = np.asarray(wp, np.float64).reshape(-1, 2); nw = len(wp)
    P = np.zeros((nw + 1, 2)); P[1:] = wp
    d = np.diff(P, axis=0); seg = np.hypot(d[:, 0], d[:, 1]); s = np.concatenate([[0.0], np.cumsum(seg)])
    good = seg >= 1e-3
    if not good.any(): return None, float(s[-1])
    idx = np.arange(nw)
    nxt = np.minimum.accumulate(np.where(good, idx, nw)[::-1])[::-1]; prv = np.maximum.accumulate(np.where(good, idx, -1))
    use = np.where(nxt < nw, nxt, prv); theta = np.arctan2(d[use, 1], d[use, 0])
    tau = np.arange(1, steps + 1) / fps; u = tau / wp_dt; j = np.minimum(np.floor(u).astype(int), nw - 1); frac = u - j
    Pk = P[j] + frac[:, None] * d[j]; sk = s[j] + frac * seg[j]; th = theta[j]
    vk = np.maximum(seg[j] / wp_dt, 0.3); Lk = np.maximum(l_min, t_look * vk); sg = sk + Lk
    m = np.clip(np.searchsorted(s, sg, side='right') - 1, 0, nw - 1); sm = seg[m]
    f = np.where(sm > 0, (sg - s[m]) / np.where(sm > 0, sm, 1.0), 0.0); G = P[m] + np.clip(f, 0, 1)[:, None] * d[m]
    beyond = sg > s[-1]
    if beyond.any(): G[beyond] = P[-1] + (sg[beyond] - s[-1])[:, None] * np.array([np.cos(theta[-1]), np.sin(theta[-1])])
    dx, dy = G[:, 0] - Pk[:, 0], G[:, 1] - Pk[:, 1]; c, sn = np.cos(th), np.sin(th)
    x, y = c * dx + sn * dy, -sn * dx + c * dy
    return 2.0 * y / np.maximum(x * x + y * y, 1e-6), float(s[-1])

def run(split, variant):
    if variant == 'chord':
        C.pursuit_curvatures, keep = chord_heading, C.pursuit_curvatures
    errs = {k: [] for k in ('hyb', 'traj')}
    for sc in splits[split]:
        with np.load(os.path.join(ROOT, 'aligned', sc + '.npz')) as z:
            fut, fv, vv, ts = z['future_actions'], z['future_valid'], z['vehicle_valid'], z['timestamps_us']
            imp = L.impact_mask_frames(ts, z['imu'])
        d = lab[sc]
        ok = d['wp_valid'] & fv & vv & ~imp & (np.arange(len(ts)) % 3 == 0)
        for i in np.nonzero(ok)[0]:
            # 正解の目標速度を出す logits(two-hot の log)
            v = float(d['v_tgt'][i]); bins = np.array(pp['speed_bins'])
            q = np.maximum(1 - np.abs(bins - v) / 0.5, 1e-9); lg = np.log(q / q.sum())
            out = {'actions': np.zeros((10, 2), np.float32), 'waypoints': d['waypoints'][i], 'speed_logits': lg}
            for mode, key in ((1, 'hyb'), (3, 'traj')):
                ctl = C.TCPPost(pp, (center, neutral))
                y, info = ctl.step(out, float(d['v_now'][i]), {'tcp_mode': mode}, int(ts[i]) * 1000)
                errs[key].append(np.abs(y - fut[i]))
    if variant == 'chord':
        C.pursuit_curvatures = keep
    e = np.array(errs['traj']); h = np.array(errs['hyb'])
    return {'n': len(e), 'steer_k1_k3_k10': [round(float(e[:, j, 0].mean()), 1) for j in (0, 2, 9)],
            'steer_mean': round(float(e[:, :, 0].mean()), 1),
            'throttle_k1_k3_k10': [round(float(h[:, j, 1].mean()), 1) for j in (0, 2, 9)],
            'throttle_mean': round(float(h[:, :, 1].mean()), 1)}

out = {}
for split in ('train', 'val'):
    for variant in ('vertex', 'chord'):
        out['%s/%s' % (split, variant)] = run(split, variant)
        print(split, variant, out['%s/%s' % (split, variant)], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'oracle.json'), 'w'), indent=1)
