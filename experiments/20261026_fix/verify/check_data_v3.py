"""反証: データ v3 の表を自分で数える(v3_common を使わない)。numpy だけ。"""
import json, sys
from pathlib import Path
import numpy as np

R = Path('/home/jetson/jetformer')
DS = R / 'data/datasets/dataset_v2_20261025'
T3 = R / 'data/derived/dataset_v3_20261026'
D1 = R / 'experiments/20261026_fix/diag_data/out/d1_samples.npz'
PROT = json.loads(Path('/home/jetson/jetformer_reports/fix_20261026/protocol_v3.json').read_text())
PROT2 = json.loads(Path('/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2.json').read_text())

def seal_logs(p):
    s = p.get('seal', {})
    logs = s.get('logs') or [x.get('log') for x in s.get('per_log', [])]
    return [str(x) for x in logs]

seal3, seal2 = seal_logs(PROT), seal_logs(PROT2)
out = {'seal_v3': seal3, 'seal_v2_same': sorted(seal3) == sorted(seal2)}

def table(path, key):
    z = np.load(path)
    sc, fr, v = z['scene'].astype(str), z['frame'].astype(np.int64), z[key]
    t = {}
    for s in np.unique(sc):
        m = sc == s
        t[s] = (fr[m], v[m])
    return t

def impact(t_us, imu, g=2.5):
    t = t_us.astype(np.float64) / 1e6
    hg = np.hypot(imu[..., 0], imu[..., 1]).max(axis=1) / 9.80665
    m = np.zeros(len(t), bool)
    for ti in t[hg >= g]:
        m |= (t >= ti - 1.0) & (t <= ti + 2.0)
    return m

splits = json.loads((DS / 'splits.json').read_text())
cache = {}
def scene(s):
    if s not in cache:
        a = np.load(DS / 'aligned' / (s + '.npz'))
        cache[s] = {k: a[k] for k in a.files if k in ('future_valid', 'vehicle_valid', 'mode', 'timestamps_us', 'imu', 'vehicle', 'future_actions', 'erpm_valid', 'tof')}
    return cache[s]

def keys_for(drop_path, split='train'):
    dt = table(drop_path, 'drop')
    ks = set()
    for s in splits[split]:
        a = scene(s)
        if 'vehicle' not in a or 'tof' not in a or 'imu' not in a:
            continue
        n = len(a['future_valid'])
        ok = a['future_valid'] & a['vehicle_valid']
        mode = a['mode'] if 'mode' in a else np.zeros(n, np.int16)
        ok &= mode != 0
        d = np.zeros(n, bool)
        fr, v = dt[s]
        d[fr] = v.astype(bool)
        ok &= ~d
        ok &= ~impact(a['timestamps_us'], a['imu'].astype(np.float32)[:, :, :2])
        ok &= (np.arange(n) % 3) == 0
        ks |= {(s, int(i)) for i in np.nonzero(ok)[0]}
    return ks

D = np.load(D1)
dkeys = [(str(s), int(f)) for s, f in zip(D['scene'], D['frame'])]
B41 = D['group'] == 'B41(10/10 人)'
tk = (~B41) & np.isin(D['sub'], ('交代の直後3秒', '走り+交代の直後'))
lb_all = (D['v'] < 1.0) & (D['thr0'] > 1598)
lb = lb_all & ~tk
out['d1_n'] = len(dkeys)
out['tk_n'] = int(tk.sum())
out['lb_n'] = int(lb.sum())
out['lb_in_B41'] = int((lb & B41).sum())
out['lb_new_human_non_tk'] = int((lb & ~B41).sum())
out['lb_all_incl_tk'] = int(lb_all.sum())

v2h = keys_for(R / 'data/derived/dataset_v2_20261025/drop_v2h.npz')
out['v2h_train'] = len(v2h)
out['v2h_equals_d1'] = v2h == set(dkeys)
idx = {k: i for i, k in enumerate(dkeys)}
res = {}
for name, expect in (('drop_v3n.npz', ~tk & ~lb), ('drop_v3d.npz', ~lb), ('drop_v3b.npz', B41 & ~lb)):
    ks = keys_for(T3 / name)
    exp = {k for k, m in zip(dkeys, expect) if m}
    res[name] = {'train': len(ks), 'expected': len(exp), 'equal_expected': ks == exp,
                 'extra': len(ks - exp), 'missing': len(exp - ks),
                 'seal_scenes': sorted({s for s, _ in ks if any(s.startswith(l) for l in seal3)})}
    vk = keys_for(T3 / name, 'val')
    res[name]['val'] = len(vk)
    res[name]['val_seal'] = sorted({s for s, _ in vk if any(s.startswith(l) for l in seal3)})
out['tables'] = res
# 封は分け方(splits)のどこかにあるか
allsc = splits['train'] + splits['val'] + splits.get('test', [])
out['seal_scenes_in_any_split'] = sorted({s for s in allsc if any(s.startswith(l) for l in seal3)})

# 差し替え: 鍵が tk と同じか・値が有限か・ステアは変えないか
ov = np.load(T3 / 'override_v3d_tk_thr_B.npz')
okeys = {(str(s), int(f)) for s, f in zip(ov['scene'], ov['frame'])}
tkkeys = {k for k, m in zip(dkeys, tk) if m}
out['override'] = {'rows': int(len(ov['scene'])), 'keys_equal_tk': okeys == tkkeys,
                   'finite': int(np.isfinite(ov['future_throttle_us']).sum()), 'files': ov.files}
# 差し替えた値と元の正解: 直線(動いている)の前進
thr_B = {(str(s), int(f)): ov['future_throttle_us'][i] for i, (s, f) in enumerate(zip(ov['scene'], ov['frame']))}
modes_tk = {}
for (s, f) in tkkeys:
    m = int(scene(s)['mode'][f])
    modes_tk[m] = modes_tk.get(m, 0) + 1
out['override']['modes_of_tk_samples'] = modes_tk
modes_all = {}
for (s, f) in dkeys:
    m = int(scene(s)['mode'][f])
    modes_all[m] = modes_all.get(m, 0) + 1
out['modes_all_v2h_train'] = modes_all

# 正解の形: 動いていて直線(d1 の straight と v >= 1)の前進 1496 − 行 2
mv = D['straight'] & (D['v'] >= 1.0)
fwd = 1496.0 - D['thr2']
fwd_v3d = fwd.copy()
for i, k in enumerate(dkeys):
    if tk[i]:
        fwd_v3d[i] = 1496.0 - float(thr_B[k][2])
def stat(mask, f=fwd):
    m = mask & mv
    return {'n': int(m.sum()), 'fwd_straight': round(float(f[m].mean()), 1) if m.any() else None}
labels = {}
for nm, keep, f in (('v2h', np.ones(len(dkeys), bool), fwd), ('v3n', ~tk & ~lb, fwd), ('v3d', ~lb, fwd_v3d), ('v3b', B41 & ~lb, fwd)):
    labels[nm] = {'all': stat(keep, f)}
    for sess in np.unique(D['sess']):
        labels[nm][str(sess) + ('_B41' if False else '')] = stat(keep & (D['sess'] == sess) & ~B41, f)
    labels[nm]['B41'] = stat(keep & B41, f)
    # 低速でブレーキ側の割合(v < 1)
    low = keep & (D['v'] < 1.0)
    thr0 = D['thr0'].copy()
    if nm == 'v3d':
        for i, k in enumerate(dkeys):
            if tk[i]:
                thr0[i] = float(thr_B[k][0])
    labels[nm]['low_brake_frac'] = round(float((thr0[low] > 1598).mean()), 4)
    labels[nm]['moving_brake_frac'] = round(float((thr0[keep & (D['v'] >= 1.0)] > 1598).mean()), 4)
    # 10/23 の新しい人の数
    labels[nm]['n_new_1023'] = int((keep & ~B41 & (D['sess'] == '10/23')).sum())
out['labels'] = labels
print(json.dumps(out, ensure_ascii=False, indent=1))
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
