#!/usr/bin/env python3
"""V-models: SLOW・WGT が仮説の向きに動いているか(区間ごと)。numpy だけ。
保留 primary(人のジャンル・PyTorch FP32 の保存した予測)と、GPU の段の TensorRT の予測(ジャンル 5・保留、AI 走行)を区間で分ける。
区間は aligned の loc_code(label_events の場所。unknown R1 GM DE other HS LH)と、R1 の起点の手前 1 秒(vm_recount の r1pre)。"""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(sys.argv[1])
DESK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
DB = json.load(open(DESK / 'desk_bench.json'))
rows = {c['id']: c for c in DB['candidates']}
A = np.load(OUT / 'frames_all.npz', allow_pickle=False)
y, prim, r1pre, loc, genre = A['y'], A['primary'], A['r1pre'], A['loc'], A['genre']
v = -A['erpm'] * 0.5e-3
LOC = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
def pred(files):
    return np.mean([np.load(f)['pred'].astype(np.float64) for f in files], axis=0)
P = {c: pred(rows[c]['pred_sources']['best']['files']) for c in rows}
PL = {c: pred(rows[c]['pred_sources']['last']['files']) for c in rows}
seed_v0 = {k: np.load(r['file'])['pred'].astype(np.float64)
           for k, r in DB['seed_spread_ref']['jetformer_v0(B と同じ形・同じデータ)']['runs'].items()}
groups = {'R1_pre1s': prim & r1pre}
for k, n in enumerate(LOC):
    groups[n] = prim & (loc == k)
groups['all'] = prim
groups['weighted(R1_pre1s|R1|GM|DE)'] = prim & (r1pre | np.isin(loc, [1, 2, 3]))
groups['not_weighted'] = prim & ~(r1pre | np.isin(loc, [1, 2, 3]))
def stats(p, m):
    if not m.any():
        return None
    d = p[m] - y[m]
    return {'n': int(m.sum()), 'thr_diff_row2': float(d[:, 2, 1].mean()), 'thr_diff_all': float(d[:, :, 1].mean()),
            'steer_bias_row2': float(d[:, 2, 0].mean()), 'l1': float(np.abs(d).mean()), 'l1_steer': float(np.abs(d[:, :, 0]).mean()),
            'l1_thr': float(np.abs(d[:, :, 1]).mean()),
            'brake_row0_frac': float((p[m][:, 0, 1] > 1598).mean()), 'human_brake_row0_frac': float((y[m][:, 0, 1] > 1598).mean())}
res = {'fp32_human_genre': {}, 'seed_v0': {}, 'fp32_last': {}}
for c in ('B', 'SLOW', 'WGT', 'AUG', 'ENS', 'ACT'):
    res['fp32_human_genre'][c] = {g: stats(P[c], m) for g, m in groups.items()}
    res['fp32_last'][c] = {g: stats(PL[c], m) for g, m in groups.items()}
for k, p in seed_v0.items():
    res['seed_v0'][k] = {g: stats(p, m) for g, m in groups.items()}
# 種の幅(v0 の 5 本)と WGT・SLOW の差
spread = {}
for g in groups:
    for f in ('l1', 'l1_steer', 'l1_thr', 'thr_diff_row2'):
        vals = [res['seed_v0'][k][g][f] for k in seed_v0 if res['seed_v0'][k][g]]
        spread.setdefault(g, {})[f] = [round(min(vals), 3), round(max(vals), 3)]
res['seed_v0_range'] = spread
# GPU の TensorRT(ジャンル 5)
F = np.load(OUT / 'frames.npz', allow_pickle=False)
idx = F['idx']
gp = {}
for c in ('B', 'SLOW', 'WGT', 'AUG', 'ENS', 'ACT'):
    f = OUT / ('gpu_%s.npz' % c)
    if f.exists():
        gp[c] = np.load(f, allow_pickle=False)
if gp:
    res['trt_genre5'] = {}; res['trt_human_vs_fp32'] = {}; res['ai_closed_loop'] = {}
    for c, z in gp.items():
        P5 = np.full((len(y), 10, 2), np.nan); P5[idx] = z['ho_5']
        PH = np.full((len(y), 10, 2), np.nan); PH[idx] = z['ho_h']
        res['trt_genre5'][c] = {g: stats(P5, m) for g, m in groups.items()}
        res['trt_genre5'][c]['l1_interv_humangenre'] = float(np.abs(PH[A['interv']] - y[A['interv']]).mean())
        sel = prim | A['interv']
        d = np.abs(PH[sel] - P[c][sel])
        res['trt_human_vs_fp32'][c] = {'frames': int(sel.sum()), 'max_us': float(d.max()), 'p99_us': float(np.percentile(d, 99)),
                                       'p999_us': float(np.percentile(d, 99.9)), 'mean_us': float(d.mean()),
                                       'frames_over_2p5us': int((d.max(axis=(1, 2)) > 2.5).sum()),
                                       'l1_best_trt_humangenre': float(np.abs(PH[prim] - y[prim]).mean()),
                                       'r1_app_trt_humangenre': float(np.mean(PH[prim & r1pre][:, 2, 1] - y[prim & r1pre][:, 2, 1]))}
    # AI 走行(閉ループの状態)の区間別: 予測スロットル(行 2)の平均と B との差
    z0 = gp['B']
    ai_scene, ai_frame = z0['ai_scene'], z0['ai_frame']
    ai_loc = np.full(len(ai_scene), -1)
    cache = {}
    for k, (s, i) in enumerate(zip(ai_scene, ai_frame)):
        s = str(s)
        if s not in cache:
            cache = {s: np.load('/home/jetson/jetformer/data/datasets/ai_mix_20261011/aligned/%s.npz' % s)['loc_code']}
        ai_loc[k] = cache[s][int(i)]
    va = -z0['ai_erpm'] * 0.5e-3
    for c, z in gp.items():
        pr = z['ai_pred'].astype(np.float64)
        out = {}
        for k, n in enumerate(LOC):
            m = ai_loc == k
            if m.any():
                out[n] = {'n': int(m.sum()), 'thr_row2_mean': float(pr[m][:, 2, 1].mean()),
                          'thr_row2_minus_B': float((pr[m][:, 2, 1] - gp['B']['ai_pred'][m][:, 2, 1]).mean()),
                          'steer_row2_minus_B_abs': float(np.abs(pr[m][:, 2, 0] - gp['B']['ai_pred'][m][:, 2, 0]).mean()),
                          'brake_row0_at_speed_ge1': int(((pr[m][:, 0, 1] > 1598) & (va[m] >= 1.0)).sum()),
                          'brake_row0_lowspeed': int(((pr[m][:, 0, 1] > 1598) & (va[m] < 1.0)).sum()),
                          'lowspeed_n': int((va[m] < 1.0).sum())}
        lo = va < 1.0
        out['lowspeed_brake'] = [int((pr[lo][:, 0, 1] > 1598).sum()), int(lo.sum())]
        out['brake_at_speed_ge1'] = [int(((pr[:, 0, 1] > 1598) & (va >= 1.0)).sum()), int((va >= 1.0).sum())]
        out['faster_than_B_by_20us_frac'] = float(((pr[:, 2, 1] - gp['B']['ai_pred'][:, 2, 1]) < -20).mean())
        res['ai_closed_loop'][c] = out
json.dump(res, open(OUT / 'hyp.json', 'w'), ensure_ascii=False, indent=1)
def show(tag, d, keys=('R1_pre1s', 'R1', 'GM', 'DE', 'unknown', 'HS', 'LH', 'other', 'all')):
    for c, g in d.items():
        print(tag, c, ' '.join('%s:%+.1f/%.1f' % (k, g[k]['thr_diff_row2'], g[k]['l1_steer']) for k in keys if g.get(k)))
show('fp32', res['fp32_human_genre']); show('last', {k: res['fp32_last'][k] for k in ('SLOW', 'WGT', 'AUG')})
print('seed range', json.dumps(spread))
if gp:
    show('trt5', {c: res['trt_genre5'][c] for c in gp})
    print(json.dumps(res['trt_human_vs_fp32']))
    for c in gp:
        print('AI', c, json.dumps(res['ai_closed_loop'][c], ensure_ascii=False)[:900])
