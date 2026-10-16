#!/usr/bin/env python3
"""案出しの小さな確かめ 5(CPU・保存済みの予測だけ・判定には使わない)。
後から組を選んだ偏りを見るため、主指標と別のフレーム = part 1(intervention_val の採点の 1,570 フレーム。AI から人に交代した後の修正)で、
1 本と平均(アンサンブル)を比べる。出力: probe_pred_cache4.json"""
import json, os, sys
import numpy as np
JF = '/home/jetson/jetformer'
sys.path.insert(0, JF); sys.path.insert(0, JF + '/tools')
import rl_evaluate as ev  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
Z = JF + '/data/zoo/'
def f(name, s, ck='best'):
    sfx = '' if ck == 'best' else '_last'
    if name in ('jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2'):
        p = Z + 'policy_zoo_20261012/pred/%s_s%d%s.npz' % (name, s, sfx)
    elif name == 'C':
        p = Z + 'sim_cotrain_20261013/pred/C_s%d%s.npz' % (s, sfx)
    else:
        p = Z + 'sim_diverse_train_20261015/pred/CDIV_s%d%s.npz' % (s, sfx)
    with np.load(p) as z:
        return z['pred'].astype(np.float64)
tab = ev.pred_table(); cols, _ = ev._val_arrays(tab)
m1 = (tab['part'] == 1) & cols['fv'] & np.isfinite(cols['y']).all(axis=(1, 2))
y = np.nan_to_num(cols['y'].astype(np.float64))
L = lambda p: round(float(np.abs(p - y).mean(axis=(1, 2))[m1].mean()), 3)
out = {'frames_part1': int(m1.sum())}
for ck in ('best', 'last'):
    r = {}
    for n in ('jetformer_v0', 'msfusion', 'dinov2', 'act', 'tcp', 'C', 'CDIV'):
        r[n] = [L(f(n, s, ck)) for s in (0, 1)]
    r['v0+v0'] = [L((f('jetformer_v0', 0, ck) + f('jetformer_v0', 1, ck)) / 2)]
    for k, names in (('v0+C', ['jetformer_v0', 'C']), ('v0+CDIV', ['jetformer_v0', 'CDIV']),
                     ('v0+msfusion', ['jetformer_v0', 'msfusion']), ('v0+dinov2', ['jetformer_v0', 'dinov2']),
                     ('v0+C+CDIV', ['jetformer_v0', 'C', 'CDIV']),
                     ('zoo5', ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2'])):
        r[k] = [L(np.mean([f(n, s, ck) for n in names], axis=0)) for s in (0, 1)]
    out[ck] = r
json.dump(out, open(os.path.join(HERE, 'probe_pred_cache4.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
