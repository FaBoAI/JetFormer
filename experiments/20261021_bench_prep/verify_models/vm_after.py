#!/usr/bin/env python3
"""V-models: GPU の段の結果(TensorRT)を数える。numpy だけ。
  - TensorRT(人のジャンル)と、机上の保存した PyTorch FP32 の予測の差(保留 6,240 フレーム)
  - TensorRT の予測から机上の欄を数え直す(人のジャンル)・ジャンル 5(走る日)の R1 の手前
  - 10/10 の AI 走行: NaN・幅・走行が切る幅(1000〜2000)の外・desk の sanity の予測との差・低速のブレーキ
  - AI 走行の R1 の手前(labels の LH → R1 の切り替わり)での SLOW − B などのスロットル"""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(sys.argv[1])
DESK = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
LAB = Path('/home/jetson/jetformer/data/labels')
DB = json.load(open(DESK / 'desk_bench.json'))
rows = {c['id']: c for c in DB['candidates']}
A = np.load(OUT / 'frames_all.npz', allow_pickle=False)
F = np.load(OUT / 'frames.npz', allow_pickle=False)
idx = F['idx']
y = A['y']; prim = A['primary']; interv = A['interv']; r1pre = A['r1pre']
gj = json.load(open(OUT / 'gpu.json'))
ids = list(gj['info'])
res = {'timing': gj['timing'], 'fp32_vs_trt_ref_frames': gj.get('fp32_vs_trt'), 'check_io': {c: gj['info'][c]['check_io_ok'] for c in ids},
       'outputs': {c: gj['info'][c]['outputs'] for c in ids}, 'nonfinite_ai': gj['nonfinite'], 'post_err_ai': gj['post_err'],
       'per': {}}
def pred(files):
    return np.mean([np.load(f)['pred'].astype(np.float64) for f in files], axis=0)
# AI 走行の R1 の起点(labels の LH(7) → R1(1))
z0 = np.load(OUT / 'gpu_B.npz', allow_pickle=False)
ai_scene = z0['ai_scene']; ai_t = z0['ai_t_us'] / 1e6; ai_erpm = z0['ai_erpm']
ai_log = np.array([str(s)[:15] for s in ai_scene])
ai_r1pre = np.zeros(len(ai_t), bool); n_anch = 0
for lg in np.unique(ai_log):
    zl = np.load(LAB / (lg + '.npz'))
    det, ts = zl['loc_detail'], zl['t_us'] / 1e6
    st = np.nonzero((det[1:] == 1) & (det[:-1] == 7))[0] + 1
    m = ai_log == lg
    for ta in ts[st] + 0.05:
        w = m & (ai_t >= ta - 1.0) & (ai_t < ta)
        if w.any():
            ai_r1pre |= w; n_anch += 1
res['ai_r1pre'] = {'frames': int(ai_r1pre.sum()), 'anchors': n_anch}
v_ai = -ai_erpm * 0.5e-3
v_ho = -A['erpm'] * 0.5e-3
for c in ids:
    z = np.load(OUT / ('gpu_%s.npz' % c), allow_pickle=False)
    PH = np.full((len(y), 10, 2), np.nan); PH[idx] = z['ho_h']
    P5 = np.full((len(y), 10, 2), np.nan); P5[idx] = z['ho_5']
    Pf = pred(rows[c]['pred_sources']['best']['files'])
    sel = prim | interv
    d = np.abs(PH[sel] - Pf[sel])
    r = {}
    r['trt_vs_saved_fp32'] = {'frames': int(sel.sum()), 'max_us': round(float(d.max()), 3), 'p99_us': round(float(np.percentile(d, 99)), 3),
                              'p999_us': round(float(np.percentile(d, 99.9)), 3), 'mean_us': round(float(d.mean()), 4),
                              'frames_over_2p5us': int((d.max(axis=(1, 2)) > 2.5).sum()),
                              'row2_max_us': round(float(d[:, 2, :].max()), 3)}
    dd = np.abs(PH - np.nan_to_num(y))
    r['desk_fields_from_trt'] = {
        'l1_best_us': [rows[c]['l1_best_us'], round(float(dd[prim].mean()), 3)],
        'l1_steer_best_us': [rows[c]['l1_steer_best_us'], round(float(dd[prim][:, :, 0].mean()), 3)],
        'l1_interv_us': [rows[c]['l1_interv_us'], round(float(dd[interv].mean()), 3)],
        'steer_amp_ratio': [rows[c]['steer_amp_ratio'], round(float(np.std(PH[prim][:, :, 0]) / np.std(y[prim][:, :, 0])), 4)],
        'r1_app_thr_diff_us(my anchors)': [rows[c]['r1_app_thr_diff_us'], round(float(np.mean(PH[prim & r1pre][:, 2, 1] - y[prim & r1pre][:, 2, 1])), 3)]}
    r['genre5'] = {'r1_app_thr_diff_us': round(float(np.mean(P5[prim & r1pre][:, 2, 1] - y[prim & r1pre][:, 2, 1])), 3),
                   'r1_app_thr_diff_us_genre4_rows': round(float(np.mean(P5[prim & r1pre & (A['genre'] == 4)][:, 2, 1] - y[prim & r1pre & (A['genre'] == 4)][:, 2, 1])), 3),
                   'r1_app_thr_diff_us_genre5_rows': round(float(np.mean(P5[prim & r1pre & (A['genre'] == 5)][:, 2, 1] - y[prim & r1pre & (A['genre'] == 5)][:, 2, 1])), 3),
                   'thr_diff_all_row2': round(float(np.mean(P5[prim][:, 2, 1] - y[prim][:, 2, 1])), 3),
                   'steer_amp_ratio': round(float(np.std(P5[prim][:, :, 0]) / np.std(y[prim][:, :, 0])), 4),
                   'brake_lowspeed_heldout': [int((P5[prim & (v_ho < 1.0)][:, 0, 1] > 1598).sum()), int((prim & (v_ho < 1.0)).sum())]}
    # AI 走行
    pa, pm = z['ai_pred'].astype(np.float64), z['ai_model'].astype(np.float64)
    fin = np.isfinite(pa).all(axis=(1, 2)) & np.isfinite(pm).all(axis=(1, 2))
    sd = np.load(DESK / 'pred' / ('sanity_%s.npz' % c), allow_pickle=False)
    same_rows = bool(len(sd['t_us']) == len(ai_t) and np.array_equal(sd['t_us'], z['ai_t_us']))
    lo = v_ai < 1.0
    r['ai'] = {'frames': int(len(pa)), 'nonfinite_frames': int((~fin).sum()),
               'outside_500_2500': int(((pa < 500) | (pa > 2500)).sum() + ((pm < 500) | (pm > 2500)).sum()),
               'outside_drive_1000_2000_all_rows': int(((pa < 1000) | (pa > 2000)).sum()),
               'outside_drive_1000_2000_row2': int(((pa[:, 2, :] < 1000) | (pa[:, 2, :] > 2000)).sum()),
               'steer_row2_min_max': [round(float(pa[:, 2, 0].min()), 1), round(float(pa[:, 2, 0].max()), 1)],
               'thr_row2_min_max': [round(float(pa[:, 2, 1].min()), 1), round(float(pa[:, 2, 1].max()), 1)],
               'same_rows_as_desk_sanity': same_rows,
               'max_abs_diff_vs_desk_sanity_us': round(float(np.abs(pa - sd['pred']).max()), 4) if same_rows else None,
               'lowspeed_brake_ai': [int((pa[lo][:, 0, 1] > 1598).sum()), int(lo.sum())],
               'brake_at_speed_ge1_row0': [int(((pa[:, 0, 1] > 1598) & ~lo).sum()), int((~lo).sum())],
               'reverse_side_row2_lowspeed(>1598)': int(((pa[:, 2, 1] > 1598) & lo).sum()),
               'r1pre_thr_row2_mean': round(float(pa[ai_r1pre][:, 2, 1].mean()), 2)}
    hk, hn = r['genre5']['brake_lowspeed_heldout']
    hk_h = int((PH[prim & (v_ho < 1.0)][:, 0, 1] > 1598).sum())
    ak, an = r['ai']['lowspeed_brake_ai']
    r['brake_lowspeed_frac'] = {'desk': rows[c]['brake_lowspeed_frac'], 'mine_trt(heldout human genre + AI)': round((hk_h + ak) / (hn + an), 5),
                                'mine_trt(heldout genre5 + AI)': round((hk + ak) / (hn + an), 5)}
    res['per'][c] = r
zb = np.load(OUT / 'gpu_B.npz', allow_pickle=False)['ai_pred'].astype(np.float64)
for c in ids:
    pa = np.load(OUT / ('gpu_%s.npz' % c), allow_pickle=False)['ai_pred'].astype(np.float64)
    res['per'][c]['ai']['r1pre_thr_row2_minus_B'] = round(float((pa[ai_r1pre][:, 2, 1] - zb[ai_r1pre][:, 2, 1]).mean()), 2)
    res['per'][c]['ai']['all_thr_row2_minus_B'] = round(float((pa[:, 2, 1] - zb[:, 2, 1]).mean()), 2)
    res['per'][c]['ai']['not_r1pre_thr_row2_minus_B'] = round(float((pa[~ai_r1pre][:, 2, 1] - zb[~ai_r1pre][:, 2, 1]).mean()), 2)
json.dump(res, open(OUT / 'after.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({'timing': res['timing'], 'fp32': res['fp32_vs_trt_ref_frames'], 'ai_r1pre': res['ai_r1pre']}, ensure_ascii=False))
for c in ids:
    print(c, json.dumps(res['per'][c], ensure_ascii=False))
