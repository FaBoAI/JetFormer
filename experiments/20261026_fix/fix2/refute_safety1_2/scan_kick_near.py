"""反証 safety1-2(読むだけ): 本物の記録で、StartKick(assist = 4)が出た AI の区間ごとに、
発進の前進なら始めたか(直近 5 フレームの TTC の中央値 >= 3.3 秒)と、そのあと何が起きたか(動いたか・衝撃)を見る。
0 バイト・壊れた npz は飛ばす。data/raw は変えない"""
import glob, json, os, sys
import numpy as np
root = '/home/jetson/jetformer/data/raw'
KV = 0.5e-3
out_path = sys.argv[1]


def load(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    except Exception:
        return None


def cat(ds, keys):
    return {k: np.concatenate([d[k] for d in ds]) for k in keys}


rows = []
logs = sorted(set(os.path.dirname(os.path.dirname(p)) for p in glob.glob(root + '/202610[2-3]*/*/AI_CMD.npz')))
for lg in logs:
    scenes = sorted(glob.glob(lg + '/*/AI_CMD.npz'))
    A, V, I = [], [], []
    for sc in scenes:
        a = load(sc); v = load(os.path.join(os.path.dirname(sc), 'VEHICLE.npz')); im = load(os.path.join(os.path.dirname(sc), 'IMU.npz'))
        if a is None or v is None or im is None or 'assist' not in a or 'ttc' not in a:
            continue
        A.append(a); V.append(v); I.append(im)
    if not A:
        continue
    try:
        a = cat(A, ['timestamps_us', 'assist', 'select', 'hold', 'dry_run', 'ttc', 'out', 'model_out'])
        v = cat(V, ['timestamps_us', 'erpm'])
        im = cat(I, ['timestamps_us', 'accel'])
    except Exception:
        continue
    o = np.argsort(a['timestamps_us']); a = {k: x[o] for k, x in a.items()}
    t = a['timestamps_us'] / 1e6; vt = v['timestamps_us'] / 1e6; it = im['timestamps_us'] / 1e6
    vo = np.argsort(vt); vt = vt[vo]; verpm = v['erpm'][vo].astype(float)
    io = np.argsort(it); it = it[io]; acc = im['accel'][io]
    gh = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
    erpm = np.interp(t, vt, verpm)
    ai = a['select'].astype(bool) & ~a['dry_run'].astype(bool)
    n = len(t)
    i = 0
    while i < n:
        if not ai[i]:
            i += 1
            continue
        j = i
        while j < n and ai[j] and (j == i or t[j] - t[j - 1] < 0.5):
            j += 1
        seg = np.arange(i, j)
        k = seg[a['assist'][seg] == 4]
        if len(k):
            ttc5 = a['ttc'][max(0, i - 4):i + 1].astype(float)
            fin = np.isfinite(ttc5).all() and len(ttc5) == 5
            med = float(np.median(ttc5)) if fin else float('nan')
            boost = 'start' if (fin and med >= 3.3 and abs(erpm[i]) < 400) else ('near' if fin else 'ttc_n/tof')
            k0, k1 = int(k[0]), int(k[-1])
            tk0, tk1 = t[k0], t[k1]
            mv = np.where((t >= tk0) & (erpm <= -800))[0]
            t_move = float(t[mv[0]] - tk0) if len(mv) else None
            w_kick = (it >= tk0) & (it <= tk1 + 0.05)
            w_after = (it >= tk0) & (it <= tk1 + 1.5)
            ttc_k = a['ttc'][k].astype(float)
            rows.append({'log': os.path.basename(lg), 'sec_row': int(i), 't_ai': round(float(t[i]), 2),
                         'erpm_at_ai': round(float(erpm[i])), 'ttc5': [round(float(x), 2) for x in ttc5], 'ttc5_med': None if not fin else round(med, 2),
                         'boost_would': boost, 'kick_frames': int(len(k)), 'kick_s': round(float(tk1 - tk0), 2),
                         'kick_out_us': [round(float(np.min(a['out'][k, 1]))), round(float(np.max(a['out'][k, 1])))],
                         'model_us_at_kick': [round(float(np.min(a['model_out'][k, 1]))), round(float(np.max(a['model_out'][k, 1])))],
                         'ttc_during_kick_min': None if not np.isfinite(ttc_k).any() else round(float(np.nanmin(ttc_k)), 2),
                         't_move_0p4mps_s': None if t_move is None else round(t_move, 2),
                         'max_g_during_kick': round(float(gh[w_kick].max()), 2) if w_kick.any() else None,
                         'max_g_kick_to_1p5s_after': round(float(gh[w_after].max()), 2) if w_after.any() else None,
                         'seg_s': round(float(t[j - 1] - t[i]), 2)})
        i = j
sumy = {'sections_with_kick': len(rows),
        'by_boost_would': {b: sum(1 for r in rows if r['boost_would'] == b) for b in ('start', 'near', 'ttc_n/tof')},
        'kick_while_boost_would_skip': [r for r in rows if r['boost_would'] != 'start'],
        'kick_never_moved': [r for r in rows if r['t_move_0p4mps_s'] is None],
        'impact_ge_2g_within_kick+1.5s': [r for r in rows if (r['max_g_kick_to_1p5s_after'] or 0) >= 2.0]}
json.dump({'summary': sumy, 'rows': rows}, open(out_path, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in sumy.items()}, ensure_ascii=False))
