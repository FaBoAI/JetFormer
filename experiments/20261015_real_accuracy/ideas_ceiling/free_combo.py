#!/usr/bin/env python3
"""ただで(学習なしで)試せる組み合わせの上限: 予測の平均(アンサンブル)+ 時間のアンサンブル、介入の集合での誤差、
問題のシーン 193638-0003 の中身(CPU だけ。予測の再計算なし)"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, 'cache', 'heldout_cache.npz'), allow_pickle=False)
names = [str(x).split('/')[-1] for x in Z['names']]
P = Z['preds'].astype(np.float64)
Y = np.nan_to_num(Z['y'].astype(np.float64))
clean = Z['clean']
idx = {n: i for i, n in enumerate(names)}
scene, frame, part = Z['scene'], Z['frame'], Z['part']
key = {(s, int(i)): j for j, (s, i) in enumerate(zip(scene, frame)) if part[j] == 0}
prev = np.full((len(scene), 10), -1, int)
for j in range(len(scene)):
    if part[j] == 0:
        for d in range(10):
            prev[j, d] = key.get((scene[j], int(frame[j]) - d), -1)


def temporal(pred, decay=1.0):
    out = pred.copy()
    for k in range(10):
        num = np.zeros((len(pred), 2))
        den = np.zeros(len(pred))
        for d in range(0, 10 - k):
            r = prev[:, d]
            ok = r >= 0
            w = np.exp(-decay * d)
            num[ok] += w * pred[r[ok], k + d]
            den[ok] += w
        g = den > 0
        out[g, k] = num[g] / den[g, None]
    return out


def L(p, m=clean):
    return round(float(np.abs(p - Y)[m].mean()), 3)


res = {}
A = ['jetformer_v0_s0', 'jetformer_v0_s1']
het = [p + s for p in ('jetformer_v0', 'msfusion', 'tcp', 'act', 'dinov2') for s in ('_s0', '_s1')]
res['decay_scan_A_s0'] = {str(dec): L(temporal(P[idx[A[0]]], dec)) for dec in (0.3, 0.5, 1.0, 1.5, 2.0)}
res['decay_scan_A_s1'] = {str(dec): L(temporal(P[idx[A[1]]], dec)) for dec in (0.3, 0.5, 1.0, 1.5, 2.0)}
res['A_s0_temporal1.0'] = L(temporal(P[idx[A[0]]], 1.0))
res['A_s1_temporal1.0'] = L(temporal(P[idx[A[1]]], 1.0))
eA = P[[idx[a] for a in A]].mean(0)
res['A2_ens'] = L(eA)
res['A2_ens_temporal1.0'] = L(temporal(eA, 1.0))
eh = P[[idx[a] for a in het]].mean(0)
res['het10_ens'] = L(eh)
res['het10_ens_temporal1.0'] = L(temporal(eh, 1.0))
# 1 本の構成どうしの組(推論 2 本): v0 + msfusion など
pairs = {}
for a in ('msfusion', 'tcp', 'act', 'dinov2'):
    v = []
    for s in ('_s0', '_s1'):
        v.append(L((P[idx['jetformer_v0' + s]] + P[idx[a + s]]) / 2))
    pairs['v0+' + a] = v
res['two_model_pairs_same_seed'] = pairs
# 介入の集合(part 1 = intervention_val の採点フレーム。人が AI から取り上げて直した後の操作)
iv = (part == 1) & Z['fv']
res['intervention_set_n'] = int(iv.sum())
res['intervention_L1'] = {n: L(P[idx[n]], iv) for n in A + ['B', 'AFT_s0', 'AFT_s1', 'C_s0', 'C_s1', 'CDIV_s0', 'CDIV_s1', 'msfusion_s0', 'msfusion_s1', 'tcp_s0', 'tcp_s1', 'act_s0', 'act_s1', 'dinov2_s0', 'dinov2_s1']}
res['intervention_L1']['A2_ens'] = L(eA, iv)
res['intervention_L1']['het10_ens'] = L(eh, iv)
# 193638-0003(主指標の誤差の 24%)
s = '20261010-193638-0003'
m = clean & (scene == s)
g4 = clean & (Z['mode'] == 4)
res['scene_193638_0003'] = {
    'n': int(m.sum()),
    'human_thr_k3_median': round(float(np.median(Y[m][:, 2, 1])), 1),
    'other_genre4_human_thr_k3_median': round(float(np.median(Y[g4 & ~m][:, 2, 1])), 1),
    'A_s0_thr_k3_median': round(float(np.median(P[idx[A[0]]][m][:, 2, 1])), 1),
    'human_speed_mps_median': round(float(np.median(-Z['f_erpm'][m] * 0.5e-3)), 2),
    'other_genre4_speed_mps_median': round(float(np.median(-Z['f_erpm'][g4 & ~m] * 0.5e-3)), 2),
    'A_s0_L1_without_this_scene': L(P[idx[A[0]]], clean & (scene != s)),
    'A_s1_L1_without_this_scene': L(P[idx[A[1]]], clean & (scene != s)),
}
# 場所別の差: ステアの符号の正解率(3 フレーム先、|人 − 中心| > 50 µs のときだけ)
ref = 1442.0
for n in A:
    p = P[idx[n]][clean][:, 2, 0] - ref
    t = Y[clean][:, 2, 0] - ref
    mm = np.abs(t) > 50
    res.setdefault('steer_sign_acc_k3_|human|>50', {})[n] = round(float((np.sign(p[mm]) == np.sign(t[mm])).mean()), 3)
json.dump(res, open(os.path.join(HERE, 'out', 'free_combo.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
