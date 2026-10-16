#!/usr/bin/env python3
"""誤差の内訳・ただで試せる手(アンサンブル・時間のアンサンブル)・単純な基準(CPU だけ、予測の再計算なし)

    cd ~/jetformer && python experiments/20261015_real_accuracy/ideas_ceiling/breakdown.py
入力: cache/heldout_cache.npz(build_cache.py)。出力: out/breakdown.json と標準出力の表
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOC = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
Z = np.load(os.path.join(HERE, 'cache', 'heldout_cache.npz'), allow_pickle=False)
names = [str(x) for x in Z['names']]
P = Z['preds'].astype(np.float64)
Y = np.nan_to_num(Z['y'].astype(np.float64))
clean = Z['clean']
idx = {n.split('/')[-1]: i for i, n in enumerate(names)}
A = ['jetformer_v0_s0', 'jetformer_v0_s1']


def L(pred, m=clean):
    return float(np.abs(pred - Y)[m].mean())


def per_k(pred, m=clean):
    d = np.abs(pred - Y)[m]
    return {'steer': [round(float(x), 1) for x in d[:, :, 0].mean(0)], 'throttle': [round(float(x), 1) for x in d[:, :, 1].mean(0)]}


res = {}
bests = [n for n in idx if not n.endswith('_last')]
res['models_best'] = {n: round(L(P[idx[n]]), 3) for n in bests}
pa = (P[idx[A[0]]] + P[idx[A[1]]]) / 2
res['A_seed_mean_of_L'] = round((L(P[idx[A[0]]]) + L(P[idx[A[1]]])) / 2, 3)
# ---- 単純な基準
cur = Z['cur'].astype(np.float64)
persist = np.repeat(cur[:, None, :], 10, axis=1)
res['persistence_cur_action'] = {'L1': round(L(persist), 2), 'per_k': per_k(persist)}
med = np.zeros_like(Y)
for g in (4, 5):
    m = clean & (Z['mode'] == g)
    med[Z['mode'] == g] = np.median(Y[m], axis=0)          # 後から見た値(甘い)
res['genre_median_constant(oracle)'] = round(L(med), 2)
res['A_s0_per_k'] = per_k(P[idx[A[0]]])
res['A_mean_pred_per_k'] = per_k(pa)
# ---- チャンネル別・ジャンル別・シーン別・場所別(A の 2 本の L の平均)
def by(groups, labels=None):
    out = {}
    for g in np.unique(groups[clean]):
        m = clean & (groups == g)
        key = labels[g] if labels else str(g)
        out[key] = {'n': int(m.sum()),
                    'L1': round(np.mean([L(P[idx[a]], m) for a in A]), 2),
                    'steer': round(np.mean([float(np.abs(P[idx[a]] - Y)[m][:, :, 0].mean()) for a in A]), 1),
                    'throttle': round(np.mean([float(np.abs(P[idx[a]] - Y)[m][:, :, 1].mean()) for a in A]), 1),
                    'share_of_total_err': round(float(np.mean([np.abs(P[idx[a]] - Y)[m].mean(axis=(1, 2)).sum() / np.abs(P[idx[a]] - Y)[clean].mean(axis=(1, 2)).sum() for a in A])), 3)}
    return out


res['by_genre'] = by(Z['mode'])
res['by_scene'] = by(Z['scene'])
res['by_loc'] = by(Z['f_loc_code'].astype(int), LOC)
# 行ごとの誤差の分布(重い尾か)
r = np.mean([np.abs(P[idx[a]] - Y).mean(axis=(1, 2)) for a in A], axis=0)[clean]
q = np.quantile(r, [.5, .75, .9, .95, .99])
srt = np.sort(r)[::-1]
res['row_error_quantiles'] = [round(float(x), 1) for x in q]
res['share_of_error_top10pct_rows'] = round(float(srt[:len(srt) // 10].sum() / srt.sum()), 3)
res['L1_without_top5pct_rows'] = round(float(np.sort(r)[:int(len(r) * .95)].mean()), 2)
# ---- アンサンブル(予測の平均。推論は本数ぶん重くなる)
def ens(ns, how='mean'):
    s = np.stack([P[idx[n]] for n in ns])
    return np.median(s, 0) if how == 'median' else s.mean(0)


sets = {
    'A_s0+A_s1': A,
    'A_s0+A_s1+B': A + ['B'],
    'v0_real_only_5 (A2,B,AFT2)': A + ['B', 'AFT_s0', 'AFT_s1'],
    'v0_family_all_11 (A2,B,AFT2,C2,CFT2,CDIV2)': A + ['B', 'AFT_s0', 'AFT_s1', 'C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'CDIV_s0', 'CDIV_s1'],
    'zoo_5_policies_x2 (10)': [p + s for p in ('jetformer_v0', 'msfusion', 'tcp', 'act', 'dinov2') for s in ('_s0', '_s1')],
    'all_best_19': bests,
    'A_s0 best+last': ['jetformer_v0_s0', 'jetformer_v0_s0_last'],
}
res['ensembles'] = {}
for k, ns in sets.items():
    ns = [n for n in ns if n in idx]
    e = ens(ns)
    single = np.mean([L(P[idx[n]]) for n in ns])
    res['ensembles'][k] = {'n': len(ns), 'L1_mean_ens': round(L(e), 3), 'L1_median_ens': round(L(ens(ns, 'median')), 3),
                           'mean_single': round(float(single), 3), 'gain_vs_mean_single': round(float(single - L(e)), 3)}
# 本数を増やしたときの外挿(v0 の 11 本から k 本を選ぶ)
pool = sets['v0_family_all_11 (A2,B,AFT2,C2,CFT2,CDIV2)']
rng = np.random.default_rng(0)
curve = {}
for k in (1, 2, 3, 4, 6, 8, 11):
    v = []
    for _ in range(30 if k < 11 else 1):
        ns = list(rng.choice(pool, k, replace=False))
        v.append(L(ens(ns)))
    curve[k] = round(float(np.mean(v)), 3)
res['ensemble_curve_v0_family'] = curve
ks = np.array(list(curve.keys()), float)
vs = np.array(list(curve.values()))
b, a = np.polyfit(1 / ks, vs, 1)
res['ensemble_infinite_extrapolation_L1'] = round(float(a), 2)
# ---- 時間のアンサンブル(ACT の temporal ensembling と同じ考え。前のフレームの予測のうち同じ時刻を指すものを平均)
scene, frame = Z['scene'], Z['frame']
key = {(s, int(i)): j for j, (s, i) in enumerate(zip(scene, frame)) if Z['part'][j] == 0}
prev = np.full((len(scene), 10), -1, int)       # prev[i, j] = 同じシーンの j フレーム前の行
for j in range(len(scene)):
    if Z['part'][j] != 0:
        continue
    for d in range(10):
        prev[j, d] = key.get((scene[j], int(frame[j]) - d), -1)


def temporal(pred, J=9, decay=0.0):
    """行 i の k 番目(i から k+1 フレーム先)= 平均_j pred[i-j][k+j](k+j ≤ 9)。重み exp(-decay*j)"""
    out = pred.copy()
    for k in range(10):
        num = np.zeros((len(pred), 2))
        den = np.zeros(len(pred))
        for d in range(0, min(J, 9 - k) + 1):
            r = prev[:, d]
            ok = r >= 0
            w = np.exp(-decay * d)
            num[ok] += w * pred[r[ok], k + d]
            den[ok] += w
        good = den > 0
        out[good, k] = num[good] / den[good, None]
    return out


res['temporal_ensemble'] = {}
for n in A + ['B']:
    p = P[idx[n]]
    row = {'base': round(L(p), 3)}
    for J, dec in ((2, 0), (4, 0), (9, 0), (9, 0.3), (9, 1.0)):
        t = temporal(p, J, dec)
        row['J%d_decay%.1f' % (J, dec)] = round(L(t), 3)
    t = temporal(p, 9, 0.3)
    row['k3_base'] = round(float(np.abs(p - Y)[clean][:, 2].mean()), 2)
    row['k3_J9_decay0.3'] = round(float(np.abs(t - Y)[clean][:, 2].mean()), 2)
    res['temporal_ensemble'][n] = row
# 時間のアンサンブル + 種のアンサンブル
t2 = temporal(pa, 9, 0.3)
res['temporal_plus_seed_ensemble_A2'] = round(L(t2), 3)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(res, open(os.path.join(HERE, 'out', 'breakdown.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
