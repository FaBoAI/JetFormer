#!/usr/bin/env python3
"""参考: 決まりの文で 1 つに決まらない所を変えても、D4・D5a の答え(門の不合格)が変わらないかを見る(読むだけ)。
  1. 低速のブレーキの線: 決まりは 1598 µs(10/21 と同じ)。v2 の中立 1500 + 100 = 1600 にしたら
  2. 「同じ日の B の p99」: 13 回の中央値(desk)の代わりに、同じ仕事の B・その日の最小・最大を使ったら
"""
import json, sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_train_v2/verify_final')
import recount_final as rf
C = rf.Ctx()
out = {'brake_1600': {}, 'inference_alt': {}}
ARM, MSF = rf.ARM, rf.MSF
tr = {}
for run in ['%s-s%d' % (ARM, k) for k in range(4)] + ['%s-s%d' % (MSF, k) for k in range(2)]:
    pb = C.eval_pred(run)
    job, d, z, cur = C.trt_pred(run)
    tr[run] = (pb, z, d)
    def frac(pe, tp, thr):
        vh = C.v_h1; okh = C.prim & (vh < 1.0)
        va = -z['erpm'].astype(np.float64) * 0.5e-3; oka = (z['part'] == 0) & (va < 1.0)
        return float(((okh & (pe[:, 0, 1] > thr)).sum() + (oka & (tp[:, 0, 1] > thr)).sum()) / (okh.sum() + oka.sum()))
    out['brake_1600'][run] = {'1598': frac(pb, z['pred'].astype(np.float64), 1598.0), '1600': frac(pb, z['pred'].astype(np.float64), 1600.0)}
for k in range(2):
    la, ms = '%s-s%d' % (ARM, k), '%s-s%d' % (MSF, k)
    pe = (tr[la][0] + tr[ms][0]) / 2; tp = (tr[la][1]['pred'].astype(np.float64) + tr[ms][1]['pred'].astype(np.float64)) / 2
    z = tr[la][1]
    vh = C.v_h1; okh = C.prim & (vh < 1.0)
    va = -z['erpm'].astype(np.float64) * 0.5e-3; oka = (z['part'] == 0) & (va < 1.0)
    for thr in (1598.0, 1600.0):
        out['brake_1600'].setdefault('ens s%d' % k, {})['%d' % thr] = float(((okh & (pe[:, 0, 1] > thr)).sum() + (oka & (tp[:, 0, 1] > thr)).sum()) / (okh.sum() + oka.sum()))
med, jobs = rf.b_p99_same_day(C)
vals = [j[1] for j in jobs]
for run, (pb, z, d) in tr.items():
    p99 = float(np.percentile(z['ms_timed'].astype(np.float64), 99))
    bj = [j[1] for j in jobs if j[0] == d['job']][0]
    out['inference_alt'][run] = {'p99': p99, 'median13': p99 <= 1.3 * med, 'same_job_B': p99 <= 1.3 * bj,
                                 'day_min_B': p99 <= 1.3 * min(vals), 'day_max_B': p99 <= 1.3 * max(vals)}
out['B_day'] = {'median': med, 'min': min(vals), 'max': max(vals)}
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
