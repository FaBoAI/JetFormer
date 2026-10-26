#!/usr/bin/env python3
"""確認 V-decide(2026-10-23): C1 の決定に効く机上の欄を、保存した予測から別のコードで数え直す(CPU・torch を読まない)。

desk_lib・desk_lib2・desk_eval・desk_eval2・recount2・decide_c1 は import しない。
  - 予測: 1 本ずつの pred の npz(バンドルではない)を読む。バンドル・10/21 のバンドルとも比べる
  - 人の正解: clean_driving_20261011 の aligned の future_actions を自分で読み直し、バンドルの y と比べる
  - 区間: tools/cap_replay.prepare(G1 の再生の道。make_bench_tables.phase_of_log とは別のコード)のジャイロの位相と、
    jetformer.safety_caps.combine_zone(走行のコード)を自分で合わせる。zones_heldout.npz と比べる
  - R1 の手前 1 秒: cap_replay.prepare の起点(label_events.course_anchors)から自分で作り、バンドルの r1_pre1s と比べる
  - 欄: l1_steer_best_us・r1_app_thr_diff_us・nonzone_thr_diff_us・brake_lowspeed_frac・steer_amp_ratio を自分の式で
  - A1 の門・A10 の決め方を自分で書いて当てる

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 \
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261022_prep/verify_decide/vd_recount.py
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
import cap_replay as CR  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402

R21 = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
R22 = Path('/home/jetson/jetformer_reports/prep_20261022/desk')
OUT = Path('/home/jetson/jetformer_reports/prep_20261022/verify_decide_raw')
KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ALIGNED = JF / 'data' / 'datasets' / 'clean_driving_20261011' / 'aligned'
RUNS = {'B': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'SLOW': 'bench_20261021-slow-s0',
        'SLOW1': 'bench_20261021-slow-s1', 'SLOW-SEC0': 'bench_20261022-slow-sec-s0',
        'SLOW-SEC1': 'bench_20261022-slow-sec-s1'}
PRED_FILES = {
    'B': (JF / 'data/zoo/policy_zoo_20261012/pred/B.npz', JF / 'data/zoo/policy_zoo_20261012/pred/B_last.npz'),
    'SLOW': (R21 / 'pred/bench_20261021-slow-s0.npz', R21 / 'pred/bench_20261021-slow-s0_last.npz'),
    'SLOW1': (R22 / 'pred/bench_20261021-slow-s1.npz', R22 / 'pred/bench_20261021-slow-s1_last.npz'),
    'SLOW-SEC0': (R22 / 'pred/bench_20261022-slow-sec-s0.npz', R22 / 'pred/bench_20261022-slow-sec-s0_last.npz'),
    'SLOW-SEC1': (R22 / 'pred/bench_20261022-slow-sec-s1.npz', R22 / 'pred/bench_20261022-slow-sec-s1_last.npz')}
SANITY = {'B': R21 / 'pred/sanity_B.npz', 'SLOW': R21 / 'pred/sanity_SLOW.npz', 'SLOW1': R22 / 'pred/sanity_SLOW1.npz',
          'SLOW-SEC0': R22 / 'pred/sanity_SLOW-SEC0.npz', 'SLOW-SEC1': R22 / 'pred/sanity_SLOW-SEC1.npz'}
POOL = ('SLOW', 'SLOW1', 'SLOW-SEC0', 'SLOW-SEC1')      # A10 の 1 の集合
NEUTRAL, BRAKE_MARGIN, V_LOW = 1498.0, 100.0, 1.0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def r1(x):
    """小数第 1 位に四捨五入(round-half-even を避け、十進の切り上げ)"""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def main():
    t0 = time.time()
    res = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'), 'checks': {}}
    C = res['checks']
    b22 = np.load(R22 / 'pred/desk_inputs_20261022.npz')
    b21 = np.load(R21 / 'pred/desk_inputs.npz')
    # ---- 1. バンドル 10/22 と 10/21 の共通の欄が同じか
    for k in ('key', 'part', 'scene', 'frame', 'y', 'erpm', 't_us', 'primary', 'interv', 'r1_pre1s', 'r1_anchor_id'):
        C['bundle22_vs_21_' + k] = bool(np.array_equal(b22[k], b21[k]))
    for cid in ('B', 'SLOW'):
        for ck in ('best', 'last'):
            C['bundle22_vs_21_pred_%s_%s' % (cid, ck)] = bool(np.array_equal(b22['pred_%s_%s' % (cid, ck)],
                                                                             b21['pred_%s_%s' % (cid, ck)]))
    scene, frame = b22['scene'], b22['frame']
    prim = b22['primary'].astype(bool)
    y = b22['y'].astype(np.float64)
    n = len(scene)
    # ---- 2. 予測は 1 本ずつのファイルから(バンドルと同じかも見る)
    P = {}
    for cid, (fb, fl) in PRED_FILES.items():
        for ck, f in (('best', fb), ('last', fl)):
            with np.load(f) as z:
                assert str(z['key']) == KEY, (f, str(z['key']))
                assert str(z['run']) == RUNS[cid], (f, str(z['run']))
                assert str(z['ckpt']) == ck, (f, str(z['ckpt']))
                assert z['pred'].shape == (n, 10, 2)
                P[(cid, ck)] = z['pred'].astype(np.float64)
                C['pred_file_vs_bundle_%s_%s' % (cid, ck)] = bool(np.array_equal(z['pred'], b22['pred_%s_%s' % (cid, ck)]))
                C['pred_file_finite_%s_%s' % (cid, ck)] = bool(np.isfinite(z['pred']).all())
    # ---- 3. 人の正解を aligned から読み直す
    ycheck, vcheck = 0, 0
    yaln = np.full((n, 10, 2), np.nan)
    fv = np.zeros(n, bool)
    erpm_aln = np.full(n, np.nan)
    tus_aln = np.zeros(n, np.int64)
    for s in sorted(set(scene[prim])):
        with np.load(ALIGNED / (s + '.npz')) as z:
            m = scene == s
            idx = frame[m]
            yaln[m] = z['future_actions'][idx]
            fv[m] = z['future_valid'][idx].astype(bool)
            erpm_aln[m] = z['vehicle'][idx, 3]
            tus_aln[m] = z['timestamps_us'][idx]
    C['y_vs_aligned_primary_maxdiff_us'] = float(np.abs(yaln[prim] - y[prim]).max())
    C['erpm_vs_aligned_primary_same'] = bool(np.array_equal(erpm_aln[prim].astype(np.float32), b22['erpm'][prim]))
    C['t_us_vs_aligned_primary_same'] = bool(np.array_equal(tus_aln[prim], b22['t_us'][prim]))
    C['primary_all_future_valid'] = bool(fv[prim].all())
    C['primary_frames'] = int(prim.sum())
    C['primary_scenes'] = sorted(set(scene[prim]))
    # ---- 4. 区間を別の道(cap_replay.prepare)で作り直す。primary のシーンのログだけ
    logs = sorted({s.rsplit('-', 1)[0] for s in set(scene[prim])})
    myzone = np.array(['?'] * n, dtype=object)
    my_r1pre = np.zeros(n, bool)
    my_anchor = np.full(n, -1, np.int64)
    n_anchor_logs = {}
    anchor_seq = 0
    for lg in logs:
        d = CR.prepare(lg)
        t0us = d['info']['t0_us']
        it = d['it']
        zg = d['ver'][CR.VER]['zone_g']
        tr = d['ver'][CR.VER]['turning']
        anchors = np.asarray(d['anchors'], np.float64)
        n_anchor_logs[lg] = int(len(anchors))
        m = np.array([s.rsplit('-', 1)[0] == lg for s in scene]) & prim
        idx = np.flatnonzero(m)
        tf = (b22['t_us'][idx].astype(np.int64) - t0us) / 1e6
        ki = np.searchsorted(it, tf, side='right') - 1           # 撮像時刻までの最後の IMU のサンプル
        ok = (ki >= 0)
        kic = np.clip(ki, 0, len(it) - 1)
        fresh = ok & (np.abs(tf - it[kic]) < 0.1)
        v = -b22['erpm'][idx].astype(np.float64) * 0.5e-3
        sig_b = (y[idx, :, 0] <= 1150.0).any(axis=1) & fv[idx]
        for j, g in enumerate(idx):
            zgj = zg[kic[j]] if fresh[j] else 'unknown'
            trj = bool(tr[kic[j]]) and bool(fresh[j])
            myzone[g] = sc.combine_zone(zgj, trj, bool(sig_b[j]), v=float(v[j]), pred_min_speed=sc.PRED_MIN_SPEED)
        for a in anchors:
            w = (tf >= a - 1.0) & (tf < a)
            if w.any():
                my_r1pre[idx[w]] = True
                my_anchor[idx[w]] = anchor_seq
                anchor_seq += 1
    zh = np.load(R22 / 'pred/zones_heldout.npz')
    zmap = {(str(s), int(f)): str(z) for s, f, z in zip(zh['scene'], zh['frame'], zh['zone'])}
    zfile = np.array([zmap.get((str(scene[g]), int(frame[g])), '?') for g in range(n)], dtype=object)
    C['zones_file_vs_bundle_primary_same'] = bool(all(zfile[g] == b22['zone'][g] for g in np.flatnonzero(prim)))
    mism = [g for g in np.flatnonzero(prim) if myzone[g] != b22['zone'][g]]
    C['zone_mine_vs_bundle_primary_mismatch'] = len(mism)
    C['zone_mismatch_examples'] = [(str(scene[g]), int(frame[g]), str(myzone[g]), str(b22['zone'][g])) for g in mism[:10]]
    C['zone_mine_counts'] = {z: int(sum(myzone[g] == z for g in np.flatnonzero(prim))) for z in sc.ZONES}
    my_nonzone = prim & np.array([z in ('none', 'unknown') for z in myzone])
    C['nonzone_mine_frames'] = int(my_nonzone.sum())
    C['nonzone_mine_vs_bundle_same'] = bool(np.array_equal(my_nonzone, b22['nonzone'].astype(bool) & prim))
    C['nonzone_bundle_outside_primary'] = int((b22['nonzone'].astype(bool) & ~prim).sum())
    my_r1pre &= prim
    C['r1pre_mine_frames'] = int(my_r1pre.sum())
    C['r1pre_mine_anchors'] = int(len(set(my_anchor[my_r1pre])))
    C['r1pre_mine_vs_bundle_same'] = bool(np.array_equal(my_r1pre, b22['r1_pre1s'].astype(bool)))
    C['r1pre_mine_xor_bundle'] = int((my_r1pre ^ b22['r1_pre1s'].astype(bool)).sum())
    C['anchors_per_log'] = n_anchor_logs
    # ---- 5. 欄
    vprim = -b22['erpm'].astype(np.float64) * 0.5e-3
    low_h = prim & (vprim < V_LOW)
    C['lowspeed_heldout_frames'] = int(low_h.sum())
    yp = y[prim]
    F = {}
    sanity_ref = None
    for cid in RUNS:
        pb = P[(cid, 'best')]
        pl = P[(cid, 'last')]
        f = {}
        f['l1_best_us'] = float(np.abs(pb[prim] - yp).mean())
        f['l1_steer_best_us'] = float(np.abs(pb[prim][:, :, 0] - yp[:, :, 0]).mean())
        f['l1_thr_best_us'] = float(np.abs(pb[prim][:, :, 1] - yp[:, :, 1]).mean())
        f['steer_amp_ratio'] = float(np.std(pb[prim][:, :, 0]) / np.std(yp[:, :, 0]))
        rr = b22['r1_pre1s'].astype(bool)
        f['r1_app_thr_diff_us'] = float((pb[rr, 2, 1] - y[rr, 2, 1]).mean())
        # 起点ごとの平均の平均(参考: 重み付けを変えても境を越えないか)
        aid = b22['r1_anchor_id'][rr]
        per = [float((pb[rr][aid == a, 2, 1] - y[rr][aid == a, 2, 1]).mean()) for a in sorted(set(aid))]
        f['ref_r1_app_anchor_mean_us'] = float(np.mean(per))
        f['ref_r1_app_mine_anchors_us'] = float((pb[my_r1pre, 2, 1] - y[my_r1pre, 2, 1]).mean())
        nz = my_nonzone
        f['nonzone_thr_diff_us'] = float((pb[nz, 0, 1] - y[nz, 0, 1]).mean())
        f['ref_nonzone_row2_us'] = float((pb[nz, 2, 1] - y[nz, 2, 1]).mean())
        f['ref_nonzone_allrows_us'] = float((pb[nz, :, 1] - y[nz, :, 1]).mean())
        f['ref_nonzone_last_us'] = float((pl[nz, 0, 1] - y[nz, 0, 1]).mean())
        f['ref_r1_app_last_us'] = float((pl[rr, 2, 1] - y[rr, 2, 1]).mean())
        f['ref_nonzone_only_none_us'] = float((pb[prim & (myzone == 'none'), 0, 1] - y[prim & (myzone == 'none'), 0, 1]).mean())
        f['ref_nonzone_only_unknown_us'] = float((pb[prim & (myzone == 'unknown'), 0, 1]
                                                  - y[prim & (myzone == 'unknown'), 0, 1]).mean())
        # 低速のブレーキ: 保留(FP32 best)+ 10/10 の AI 走行(TensorRT、desk の sanity の保存)
        bh = int((low_h & (pb[:, 0, 1] > NEUTRAL + BRAKE_MARGIN)).sum())
        with np.load(SANITY[cid]) as z:
            assert str(z['run']) == RUNS[cid]
            mp = z['model_pred'].astype(np.float64)
            va = -np.asarray(z['erpm'], np.float64) * 0.5e-3
            keyrows = (tuple(z['scene']), tuple(int(x) for x in z['frame']))
            if sanity_ref is None:
                sanity_ref = keyrows
            f['ref_sanity_rows_same_as_B'] = bool(keyrows == sanity_ref)
            f['ref_sanity_finite'] = bool(np.isfinite(mp).all() and np.isfinite(z['pred']).all())
            f['ref_sanity_min_max_us'] = [float(mp.min()), float(mp.max())]
            f['ref_sanity_in_500_2500'] = bool(((mp >= 500) & (mp <= 2500)).all())
        la = va < V_LOW
        ba = int((la & (mp[:, 0, 1] > NEUTRAL + BRAKE_MARGIN)).sum())
        f['brake_lowspeed'] = {'heldout': [bh, int(low_h.sum())], 'ai_1010': [ba, int(la.sum())]}
        f['brake_lowspeed_frac'] = (bh + ba) / float(low_h.sum() + la.sum())
        F[cid] = f
    # ---- 6. desk_bench_20261022.json と比べる
    D = json.load(open(R22 / 'desk_bench_20261022.json'))
    Dc = {c['id']: c for c in D['candidates']}
    cmp_ = {}
    for cid in RUNS:
        for k in ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'steer_amp_ratio', 'r1_app_thr_diff_us',
                  'nonzone_thr_diff_us', 'brake_lowspeed_frac'):
            mine, theirs = F[cid][k], Dc[cid][k]
            tol = 0.00051 if k == 'brake_lowspeed_frac' else (0.00006 if k == 'steer_amp_ratio' else 0.0006)
            cmp_['%s.%s' % (cid, k)] = {'mine': round(mine, 5), 'desk': theirs, 'ok': abs(mine - theirs) <= tol}
        cmp_['%s.brake_counts' % cid] = {'mine': F[cid]['brake_lowspeed'],
                                         'desk': {'heldout': Dc[cid]['brake_lowspeed']['heldout'],
                                                  'ai_1010': Dc[cid]['brake_lowspeed']['ai_1010']},
                                         'ok': F[cid]['brake_lowspeed'] == {'heldout': Dc[cid]['brake_lowspeed']['heldout'],
                                                                            'ai_1010': Dc[cid]['brake_lowspeed']['ai_1010']}}
    res['compare_desk'] = cmp_
    res['compare_desk_all_ok'] = all(v['ok'] for v in cmp_.values())
    # ---- 7. A1 の門(protocol gates の 1〜4 + A1 の 5 番目の置き換え + brake_lowspeed_frac)
    gates = {}
    lB = F['B']['l1_steer_best_us']
    for cid in POOL:
        c = Dc[cid]
        why = []
        if c.get('engine_ok') is not True:
            why.append('engine_ok')
        if c.get('sanity_ok') is not True:
            why.append('sanity_ok')
        p99 = c.get('trt_p99_ms')
        if p99 is None or not (p99 <= 10.0):
            why.append('trt_p99_ms')
        if not (0.80 <= F[cid]['steer_amp_ratio'] <= 1.25):
            why.append('steer_amp_ratio')
        if not (F[cid]['l1_steer_best_us'] - lB < 10.0):
            why.append('l1_steer_best_us')
        if not (F[cid]['brake_lowspeed_frac'] <= 0.01):
            why.append('brake_lowspeed_frac')
        assert '-slow-' in RUNS[cid]          # 家族 label_r1(run の名前)→ A1 の門
        gates[cid] = {'pass': not why, 'fail': why, 'l1_steer_minus_B': F[cid]['l1_steer_best_us'] - lB,
                      'desk_gates_pass': c.get('gates_pass')}
    res['gates'] = gates
    # ---- 8. A10 の決め方
    tgt = [cid for cid in POOL if gates[cid]['pass'] and F[cid]['r1_app_thr_diff_us'] >= 20.0]
    if not tgt:
        pick = 'SLOW'
    else:
        tgt.sort(key=lambda c: (r1(F[c]['nonzone_thr_diff_us']), -r1(F[c]['r1_app_thr_diff_us']), c))
        pick = tgt[0]
    res['a10'] = {'targets_in_order': [(c, r1(F[c]['nonzone_thr_diff_us']), r1(F[c]['r1_app_thr_diff_us'])) for c in tgt],
                  'pick': pick, 'pick_run': RUNS[pick], 'desk_pick': D['a10_c1']['pick']}
    # 参考: 決め方の読み替えで答えが変わるか(判断には使わない)
    alt = {}
    for name, key in (('nonzone_row2', 'ref_nonzone_row2_us'), ('nonzone_allrows', 'ref_nonzone_allrows_us'),
                      ('nonzone_last_ckpt', 'ref_nonzone_last_us')):
        t2 = sorted(tgt, key=lambda c: (r1(F[c][key]), -r1(F[c]['r1_app_thr_diff_us']), c))
        alt[name] = [(c, round(F[c][key], 3)) for c in t2]
    t3 = [cid for cid in POOL if gates[cid]['pass'] and F[cid]['ref_r1_app_anchor_mean_us'] >= 20.0]
    alt['r1_app_by_anchor_mean_targets'] = [(c, round(F[c]['ref_r1_app_anchor_mean_us'], 3)) for c in POOL]
    alt['r1_app_by_anchor_mean_pick'] = sorted(t3, key=lambda c: (r1(F[c]['nonzone_thr_diff_us']), c))[:1]
    t4 = [cid for cid in POOL if gates[cid]['pass'] and F[cid]['ref_r1_app_last_us'] >= 20.0]
    alt['last_ckpt_r1_app'] = [(c, round(F[c]['ref_r1_app_last_us'], 3)) for c in POOL]
    alt['last_ckpt_targets'] = t4
    # 境までの余裕(門と A10 の 1)
    res['margins'] = {c: {'r1_app_minus_20': round(F[c]['r1_app_thr_diff_us'] - 20.0, 3),
                          'brake_frac_vs_0.01': round(0.01 - F[c]['brake_lowspeed_frac'], 5),
                          'l1_steer_margin_to_10': round(10.0 - gates[c]['l1_steer_minus_B'], 3)} for c in POOL}
    # brake 門の境: SLOW-SEC1 があと何フレームで落ちるか
    nb = F['SLOW-SEC1']['brake_lowspeed']
    den = nb['heldout'][1] + nb['ai_1010'][1]
    res['margins']['SLOW-SEC1_brake_frames_to_fail'] = int(math.floor(0.01 * den)) + 1 - (nb['heldout'][0] + nb['ai_1010'][0])
    res['alt_readings'] = alt
    res['fields'] = F
    res['inputs_sha256'] = {'desk_bench_20261022.json': sha(R22 / 'desk_bench_20261022.json'),
                            'desk_inputs_20261022.npz': sha(R22 / 'pred/desk_inputs_20261022.npz'),
                            'zones_heldout.npz': sha(R22 / 'pred/zones_heldout.npz')}
    res['seconds'] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'vd_recount.json', 'w') as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps({'checks': C, 'compare_desk_all_ok': res['compare_desk_all_ok'],
                      'bad': {k: v for k, v in cmp_.items() if not v['ok']}, 'gates': gates, 'a10': res['a10'],
                      'alt': alt, 'margins': res['margins'], 'seconds': res['seconds']},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
