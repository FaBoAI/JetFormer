#!/usr/bin/env python3
"""real_accuracy_20261015 の見直し(review): 見出しの数字を、保存した予測から「自分のコード」で数え直す(CPU だけ)。

評価の道具(eval/realacc_eval.py・realacc_lib.py)・rl_evaluate・policy_zoo は読み込まない。読むのは
  - 予測: data/zoo/policy_zoo_20261012/pred(A・B・10/12 の msfusion)と data/zoo/real_accuracy_20261015/pred(今回)
  - 正解・フラグ: データセットの aligned/*.npz(人のきれいな 11 シーンは clean_driving の aligned から取り、ai_mix の aligned と一致を確かめる)
だけで、予測の表(10,985 行)・主指標の集合・判定の規則・GAIN・TE・SIZE の式を protocol.json の文から書き直した。

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \\
        python experiments/20261015_real_accuracy/review/recount.py

出力: /home/jetson/jetformer_reports/real_accuracy_20261015/review/recount.json(evaluation.json との照合つき)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data' / 'datasets'
REPORT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015')
OUT = REPORT / 'review' / 'recount.json'
OLD_PRED = REPO / 'data' / 'zoo' / 'policy_zoo_20261012' / 'pred'
NEW_PRED = REPO / 'data' / 'zoo' / 'real_accuracy_20261015' / 'pred'
BATTERY = '20261010-193638-0003'
NEUTRAL = 1498.0
G = 9.80665
KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'

# ---------------------------------------------------------------- 1. 予測の表を作り直す(順番・key)


def splits(ds):
    return json.loads((DS / ds / 'splits.json').read_text())


def aligned(ds, scene, keys):
    with np.load(DS / ds / 'aligned' / (scene + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in keys if k in z.files}


rows = []
for part, ds in ((0, 'ai_mix_20261011'), (1, 'intervention_val_20261011')):
    for s in splits(ds)['val']:
        a = aligned(ds, s, ['vehicle_valid', 'future_valid'])
        ok = a['vehicle_valid'] & (a['future_valid'] if part == 1 else True)
        rows += [(part, s, int(i)) for i in np.nonzero(ok)[0]]
key = hashlib.sha1('|'.join('%d:%s:%d' % r for r in rows).encode()).hexdigest()
part = np.array([r[0] for r in rows])
scene = np.array([r[1] for r in rows])
frame = np.array([r[2] for r in rows])
N = len(rows)

# ---------------------------------------------------------------- 2. 正解・フラグ(自分で)


def my_impact(t_us, imu, g_thr=2.5, before=1.0, after=2.0):
    t = t_us.astype(np.float64) / 1e6
    hg = np.hypot(imu[:, :, 0], imu[:, :, 1]).max(axis=1) / G
    m = np.zeros(len(t), bool)
    for ti in t[hg >= g_thr]:
        m |= (t >= ti - before) & (t <= ti + after)
    return m


clean_val = splits('clean_driving_20261011')['val']
clean_train = splits('clean_driving_20261011')['train']
y = np.full((N, 10, 2), np.nan)
fv = np.zeros(N, bool)
imp = np.zeros(N, bool)
mode = np.zeros(N, int)
label_check = {}
scene_cache = {}            # GAIN の g_t 用(シーン全体)
for p, ds in ((0, None), (1, 'intervention_val_20261011')):
    for s in np.unique(scene[part == p]):
        idx = np.nonzero((part == p) & (scene == s))[0]
        if p == 0 and s not in clean_val:
            continue        # AI のシーン(主指標に入らない)
        src = 'clean_driving_20261011' if p == 0 else ds
        a = aligned(src, s, ['timestamps_us', 'imu', 'vehicle', 'vehicle_valid', 'future_actions', 'future_valid', 'mode'])
        fr = frame[idx]
        y[idx] = a['future_actions'][fr]
        fv[idx] = a['future_valid'][fr] & a['vehicle_valid'][fr]
        im = my_impact(a['timestamps_us'], a['imu'])
        imp[idx] = im[fr]
        mode[idx] = a['mode'][fr]
        scene_cache[(p, s)] = a
        if p == 0:          # ai_mix の aligned と同じか
            b = aligned('ai_mix_20261011', s, ['timestamps_us', 'future_actions', 'vehicle', 'mode'])
            label_check[s] = bool(np.array_equal(a['timestamps_us'], b['timestamps_us'])
                                  and np.array_equal(np.nan_to_num(a['future_actions']), np.nan_to_num(b['future_actions']))
                                  and np.array_equal(a['vehicle'], b['vehicle']) and np.array_equal(a['mode'], b['mode']))

yfin = np.isfinite(y).all(axis=(1, 2))
primary = (part == 0) & np.isin(scene, clean_val) & fv & ~imp
sets = {
    'primary': primary,
    'S1_no_battery_scene': primary & (scene != BATTERY),
    'S3_intervention_part1': (part == 1) & fv & yfin,
    'battery_scene_only': primary & (scene == BATTERY),
    'genre4': primary & (mode == 4),
    'genre5': primary & (mode == 5),
}
Y = np.nan_to_num(y)

# ---------------------------------------------------------------- 3. 予測


def load(name, ck):
    old = {'A_s0': 'jetformer_v0_s0', 'A_s1': 'jetformer_v0_s1', 'B': 'B', 'msf_s0': 'msfusion_s0', 'msf_s1': 'msfusion_s1'}
    if name in old:
        f = OLD_PRED / ('%s%s.npz' % (old[name], '' if ck == 'best' else '_last'))
    else:
        f = NEW_PRED / ('realacc_20261015-%s%s.npz' % (name, '' if ck == 'best' else '_last'))
    with np.load(f, allow_pickle=False) as z:
        assert str(z['key']) == KEY, (f, str(z['key']))
        assert str(z['ckpt']) == ck, (f, str(z['ckpt']))
        p = z['pred'].astype(np.float64)
    assert p.shape == (N, 10, 2) and np.isfinite(p).all(), f
    return p


NAMES = ['A_s0', 'A_s1', 'B', 'msf_s0', 'msf_s1', 'v0-s2', 'v0-s3', 'msf-s2', 'msf-s3', 'rot-s0', 'rot-s1',
         'sizeQ-s0', 'sizeQ-s1', 'sizeH-s0', 'sizeH-s1', 'sizeT-s0', 'sizeT-s1', 'gainl-s0', 'sizeHeq-s0']
P = {ck: {n: load(n, ck) for n in NAMES} for ck in ('best', 'last')}


def metrics(p):
    d = np.abs(p - Y)
    r = d.mean(axis=(1, 2))
    out = {k: float(r[m].mean()) for k, m in sets.items()}
    out['S2_k3_steer'] = float(d[primary][:, 2, 0].mean())
    out['k3_throttle'] = float(d[primary][:, 2, 1].mean())
    return out


# ---------------------------------------------------------------- 4. GAIN(protocol の文から)


def g_series(a):
    thr = a['vehicle'][:, 1].astype(np.float64)
    fwd = -a['vehicle'][:, 3].astype(np.float64)
    ok = a['vehicle_valid'] & (fwd > 2000.0) & ~my_impact(a['timestamps_us'], a['imu'])
    g = np.full(len(thr), np.nan)
    g[ok] = (NEUTRAL - thr[ok]) / fwd[ok] * 1000.0
    return g


def running_median(g, t_us, W=5.0, min_n=30):
    t = t_us.astype(np.float64) / 1e6
    out = np.full(len(g), np.nan)
    for i in range(len(g)):
        w = g[(t >= t[i] - W) & (t <= t[i])]
        w = w[np.isfinite(w)]
        if len(w) >= min_n:
            out[i] = np.median(w)
    return out


# g_ref を学習の 41 シーンから計算し直す
gs = {4: [], 5: []}
for s in clean_train:
    a = aligned('clean_driving_20261011', s, ['timestamps_us', 'imu', 'vehicle', 'vehicle_valid', 'mode'])
    g = g_series(a)
    for k in (4, 5):
        gs[k].append(g[(a['mode'] == k) & np.isfinite(g)])
g_ref_recomputed = {k: float(np.median(np.concatenate(v))) for k, v in gs.items()}
G_REF = {4: 36.41, 5: 36.86}       # 固定値


def gain_scale(W=5.0, lo=0.8, hi=2.0):
    gt = np.full(N, np.nan)
    for (p, s), a in scene_cache.items():
        idx = np.nonzero((part == p) & (scene == s))[0]
        gt[idx] = running_median(g_series(a), a['timestamps_us'], W)[frame[idx]]
    ref = np.array([G_REF.get(int(m), np.nan) for m in mode])
    sc = np.clip(gt / ref, lo, hi)
    return np.where(np.isfinite(sc), sc, 1.0)


def apply_gain(p, sc):
    q = p.copy()
    d = NEUTRAL - q[:, :, 1]
    q[:, :, 1] = np.where(d > 0, NEUTRAL - d * sc[:, None], q[:, :, 1])
    return q


SC5 = gain_scale(5.0)
SC13 = gain_scale(5.0, 0.8, 1.3)      # 参考: 実車のカードの上限 1.3
SC10 = gain_scale(10.0)               # 参考: 窓 10 秒(後から見た値)

# ---------------------------------------------------------------- 5. TE(j = 2・一様)
pos = {(p, s, f): k for k, (p, s, f) in enumerate(rows)}
prev = np.array([[pos.get((p, s, f - j), -1) for j in (0, 1, 2)] for (p, s, f) in rows])


def te(p):
    num = np.zeros_like(p)
    den = np.zeros((N, 10, 1))
    for j in range(3):
        r = prev[:, j]
        ok = r >= 0
        num[ok, :10 - j] += p[r[ok], j:]
        den[ok, :10 - j] += 1.0
    return num / den


# ---------------------------------------------------------------- 6. 変種と指標

V = {}
for ck in ('best', 'last'):
    p = P[ck]
    v = dict(p)
    v['P2'] = (p['v0-s2'] + p['msf-s2']) / 2
    v['P3'] = (p['v0-s3'] + p['msf-s3']) / 2
    v['SAME_A'] = (p['A_s0'] + p['A_s1']) / 2
    v['SAME_new'] = (p['v0-s2'] + p['v0-s3']) / 2
    v['PH0'] = (p['A_s0'] + p['msf_s0']) / 2
    v['PH1'] = (p['A_s1'] + p['msf_s1']) / 2
    for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'gainl-s0', 'P2', 'P3'):
        v['GAIN(%s)' % n] = apply_gain(v[n], SC5)
    for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3'):
        v['TE(%s)' % n] = te(v[n])
        v['GAIN13(%s)' % n] = apply_gain(v[n], SC13)
    for n in ('A_s0', 'A_s1'):
        v['GAIN10(%s)' % n] = apply_gain(v[n], SC10)
    V[ck] = v

M = {ck: {n: metrics(p) for n, p in V[ck].items()} for ck in ('best', 'last')}

# ---------------------------------------------------------------- 7. 判定(protocol の文: 10/13 の規則)
S_A = {ck: max(M[ck][n]['primary'] for n in ('A_s0', 'A_s1', 'B')) - min(M[ck][n]['primary'] for n in ('A_s0', 'A_s1', 'B'))
       for ck in ('best', 'last')}
SEC_FLOOR = {}
for m in ('S1_no_battery_scene', 'S2_k3_steer', 'S3_intervention_part1', 'battery_scene_only', 'genre4', 'genre5', 'k3_throttle'):
    SEC_FLOOR[m] = {ck: max(M[ck][n][m] for n in ('A_s0', 'A_s1', 'B')) - min(M[ck][n][m] for n in ('A_s0', 'A_s1', 'B'))
                    for ck in ('best', 'last')}


def judge(arm, ref, floor):
    arm, ref = np.array(arm), np.array(ref)
    s = max(ref.max() - ref.min(), floor)
    thr = 2 * s
    gain = ref.mean() - arm.mean()
    if arm.max() < ref.min() and gain > thr:
        c = 'better'
    elif arm.min() > ref.max() and -gain > thr:
        c = 'worse'
    else:
        c = 'equivalent'
    return {'class': c, 'gain_us': round(float(gain), 4), 'thr_us': round(float(thr), 4),
            'arm': [round(float(x), 4) for x in arm], 'ref': [round(float(x), 4) for x in ref]}


def question(arm, ref, metric='primary', floor=None):
    out = {}
    for ck in ('best', 'last'):
        fl = (S_A[ck] if metric == 'primary' else SEC_FLOOR[metric][ck]) if floor is None else floor[ck]
        out[ck] = judge([M[ck][a][metric] for a in arm], [M[ck][r][metric] for r in ref], fl)
    b, l = out['best']['class'], out['last']['class']
    out['final'] = ('better' if l == 'better' else 'equivalent') if b == 'better' else b
    return out


Q = {
    'ENS_vs_A': (['P2', 'P3'], ['A_s0', 'A_s1']),
    'SAME_vs_A': (['SAME_A', 'SAME_new'], ['A_s0', 'A_s1']),
    'ENS_vs_SAME': (['P2', 'P3'], ['SAME_A', 'SAME_new']),
    'FRESH_A': (['v0-s2', 'v0-s3'], ['A_s0', 'A_s1']),
    'FRESH_MSF': (['msf-s2', 'msf-s3'], ['A_s0', 'A_s1']),
    'ROT_vs_A': (['rot-s0', 'rot-s1'], ['A_s0', 'A_s1']),
    'GAIN_i': (['GAIN(A_s0)', 'GAIN(A_s1)'], ['A_s0', 'A_s1']),
    'GAIN_ii': (['GAIN(v0-s2)', 'GAIN(v0-s3)'], ['v0-s2', 'v0-s3']),
    'TE_i': (['TE(A_s0)', 'TE(A_s1)'], ['A_s0', 'A_s1']),
    'TE_ii': (['TE(v0-s2)', 'TE(v0-s3)'], ['v0-s2', 'v0-s3']),
    'ref_ENS_GAIN_vs_A': (['GAIN(P2)', 'GAIN(P3)'], ['A_s0', 'A_s1']),
    'ref_posthoc_pairs_vs_A': (['PH0', 'PH1'], ['A_s0', 'A_s1']),
    'ref_B_vs_A': (['B'], ['A_s0', 'A_s1']),
    'ref_GAIN13_i': (['GAIN13(A_s0)', 'GAIN13(A_s1)'], ['A_s0', 'A_s1']),
    'ref_GAIN13_ii': (['GAIN13(v0-s2)', 'GAIN13(v0-s3)'], ['v0-s2', 'v0-s3']),
    'ref_GAIN10_i': (['GAIN10(A_s0)', 'GAIN10(A_s1)'], ['A_s0', 'A_s1']),
}
res_q = {}
for qid, (arm, ref) in Q.items():
    r = {'primary': question(arm, ref)}
    for m in ('S1_no_battery_scene', 'S2_k3_steer', 'S3_intervention_part1', 'battery_scene_only', 'genre4', 'genre5', 'k3_throttle'):
        r[m] = question(arm, ref, m)
    res_q[qid] = r

# 感度(結論の規則 3): s_A' = {A_s0, A_s1, v0-s2, v0-s3, B} の範囲
SAP = {ck: max(M[ck][n]['primary'] for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B'))
       - min(M[ck][n]['primary'] for n in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B')) for ck in ('best', 'last')}
sens = {qid: question(arm, ref, 'primary', SAP)['final'] for qid, (arm, ref) in Q.items() if not qid.startswith('ref_')}

# GAIN の G1〜G3・電池のシーンの割合
gain_check = {
    'G1': res_q['GAIN_i']['primary']['final'] == 'better' and res_q['GAIN_ii']['primary']['final'] == 'better',
    'G2': all(res_q[q]['S1_no_battery_scene'][ck]['class'] != 'worse' for q in ('GAIN_i', 'GAIN_ii') for ck in ('best', 'last')),
    'G3': all(res_q[q]['S3_intervention_part1'][ck]['class'] != 'worse' for q in ('GAIN_i', 'GAIN_ii') for ck in ('best', 'last')),
}
nb, npri = sets['battery_scene_only'].sum(), primary.sum()
share = {}
for q, pair in (('GAIN_i', ('A_s0', 'A_s1')), ('GAIN_ii', ('v0-s2', 'v0-s3'))):
    tot = np.mean([M['best'][n]['primary'] - M['best']['GAIN(%s)' % n]['primary'] for n in pair])
    bat = np.mean([(M['best'][n]['battery_scene_only'] - M['best']['GAIN(%s)' % n]['battery_scene_only']) * nb / npri for n in pair])
    share[q] = {'gain_us': round(float(tot), 4), 'from_battery_us': round(float(bat), 4), 'share': round(float(bat / tot), 4)}
te_check = {'T1': res_q['TE_i']['primary']['final'] == 'better' and res_q['TE_ii']['primary']['final'] == 'better',
            'T3': all(res_q[q]['S3_intervention_part1'][ck]['class'] != 'worse' for q in ('TE_i', 'TE_ii') for ck in ('best', 'last'))}

# ---------------------------------------------------------------- 8. SIZE
NS = {'Q': 1682, 'H': 3046, 'T': 4682, 'full': 5912}
size = {}
for ck in ('best', 'last'):
    for met in ('primary', 'S1_no_battery_scene'):
        Lv = {k: float(np.mean([M[ck]['size%s-s%d' % (k, s)][met] for s in (0, 1)])) for k in ('Q', 'H', 'T')}
        Lv['full'] = float(np.mean([M[ck]['A_s%d' % s][met] for s in (0, 1)]))
        s1 = (Lv['Q'] - Lv['H']) / np.log2(NS['H'] / NS['Q'])
        s2 = (Lv['H'] - Lv['full']) / np.log2(NS['full'] / NS['H'])
        gpow = 0.0 if s1 <= 0 else s2 * float(np.clip(s2 / s1, 0, 1))
        if Lv['H'] - Lv['full'] <= 0.626 or s2 < 1.0:
            rd = 'low_value'
        elif gpow >= 2.0:
            rd = 'high_value'
        else:
            rd = 'middle'
        size['%s/%s' % (met, ck)] = {'L': {k: round(v, 4) for k, v in Lv.items()}, 's1': round(s1, 4), 's2': round(s2, 4),
                                     'G2_pow': round(gpow, 4), 'reading': rd,
                                     's2_T': round((Lv['T'] - Lv['full']) / np.log2(NS['full'] / NS['T']), 4)}
# シーン単位のブートストラップ(主指標・best。フレーム数で重み)
sc_list = sorted(set(scene[primary]))
d_best = {n: np.abs(V['best'][n] - Y).mean(axis=(1, 2)) for n in ('A_s0', 'A_s1', 'sizeQ-s0', 'sizeQ-s1', 'sizeH-s0', 'sizeH-s1')}
sums = {n: np.array([d[primary & (scene == s)].sum() for s in sc_list]) for n, d in d_best.items()}
cnt = np.array([(primary & (scene == s)).sum() for s in sc_list], float)
rng = np.random.default_rng(0)
ib = rng.integers(0, len(sc_list), size=(2000, len(sc_list)))
den = cnt[ib].sum(axis=1)
Lb = {n: v[ib].sum(axis=1) / den for n, v in sums.items()}
LQb = (Lb['sizeQ-s0'] + Lb['sizeQ-s1']) / 2
LHb = (Lb['sizeH-s0'] + Lb['sizeH-s1']) / 2
LFb = (Lb['A_s0'] + Lb['A_s1']) / 2
s1b = (LQb - LHb) / np.log2(NS['H'] / NS['Q'])
s2b = (LHb - LFb) / np.log2(NS['full'] / NS['H'])
gpb = np.where(s1b <= 0, 0.0, s2b * np.clip(s2b / np.where(s1b == 0, 1, s1b), 0, 1))
size['bootstrap_primary_best'] = {'LH_minus_Lfull_ci95': [round(float(x), 3) for x in np.percentile(LHb - LFb, [2.5, 97.5])],
                                  'G2_pow_ci95': [round(float(x), 3) for x in np.percentile(gpb, [2.5, 97.5])]}

# SIZE-EQ(種 0 だけの参考)
sizeeq = {}
for ck in ('best', 'last'):
    LH30, LH59, LF = M[ck]['sizeH-s0']['primary'], M[ck]['sizeHeq-s0']['primary'], M[ck]['A_s0']['primary']
    l2 = np.log2(NS['full'] / NS['H'])
    sizeeq[ck] = {'LH30': round(LH30, 4), 'LH59': round(LH59, 4), 'Lfull_A_s0': round(LF, 4),
                  's2': round((LH30 - LF) / l2, 4), 's2_eq': round((LH59 - LF) / l2, 4),
                  'closed_share': round((LH30 - LH59) / (LH30 - LF), 4)}

# GAINL(種 0 だけの参考)
gainl = {ck: {m: round(M[ck]['GAIN(gainl-s0)'][m] - M[ck]['GAIN(A_s0)'][m], 4)
              for m in ('primary', 'S1_no_battery_scene', 'S3_intervention_part1')} for ck in ('best', 'last')}

# ---------------------------------------------------------------- 9. シーン別(ENS と A)
per_scene = {}
for s in sc_list:
    m = primary & (scene == s)
    row = {n: float(np.abs(V['best'][n] - Y).mean(axis=(1, 2))[m].mean()) for n in ('A_s0', 'A_s1', 'P2', 'P3', 'GAIN(A_s0)')}
    row['ENS_minus_A'] = (row['P2'] + row['P3']) / 2 - (row['A_s0'] + row['A_s1']) / 2
    row['frames'] = int(m.sum())
    per_scene[s] = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
ens_scene_better = sum(1 for s in sc_list if per_scene[s]['ENS_minus_A'] < 0)
gscale_scene = {s: round(float(SC5[primary & (scene == s)].mean()), 3) for s in sc_list}

# 予測の食い違い
disagree = {'A_s0|A_s1': float(np.abs(V['best']['A_s0'] - V['best']['A_s1'])[primary].mean()),
            'v0-s2|msf-s2': float(np.abs(V['best']['v0-s2'] - V['best']['msf-s2'])[primary].mean())}

# ---------------------------------------------------------------- 10. 漏れの確かめ
size_sub = json.loads((REPO / 'experiments/20261015_real_accuracy/size/size_subsets.json').read_text())
prot = json.loads((REPORT / 'protocol.json').read_text())
subs = prot['arms']['SIZE']['subsets']
iv_logs = {s[:15] for s in splits('intervention_val_20261011')['val']}
leak = {
    'clean_train_and_val_overlap': sorted(set(clean_train) & set(clean_val)),
    'size_subsets_outside_train': {k: sorted(set(v['scenes']) - set(clean_train)) for k, v in subs.items()},
    'size_nested_Q_in_H_in_T': set(subs['Q']['scenes']) <= set(subs['H']['scenes']) <= set(subs['T']['scenes']),
    'intervention_logs_in_clean_train_logs': sorted(iv_logs & {s[:15] for s in clean_train}),
    'val_logs_shared_with_train_logs': sorted({s[:15] for s in clean_val} & {s[:15] for s in clean_train}),
    'primary_scenes': sorted(set(scene[primary])),
    'primary_scenes_equal_clean_val': sorted(set(scene[primary])) == sorted(clean_val),
}

# ---------------------------------------------------------------- 11. evaluation.json と照合
ev = json.loads((REPORT / 'evaluation.json').read_text())
mism = []
for n in ('A_s0', 'A_s1', 'B'):
    for ck in ('best', 'last'):
        for m, val in M[ck][n].items():
            e = ev['per_variant'][n][ck].get(m)
            if e is not None and abs(e - val) > 0.0015:
                mism.append(('per_variant', n, ck, m, e, round(val, 4)))
for qid in Q:
    REFMAP = {'ref_ENS_GAIN_vs_A': 'ENS_GAIN_vs_A', 'ref_posthoc_pairs_vs_A': 'PH_vs_A', 'ref_B_vs_A': 'MODEL_B_vs_A',
              'ref_GAIN13_i': 'GAIN13_i', 'ref_GAIN13_ii': 'GAIN13_ii', 'ref_GAIN10_i': 'GAIN10_i'}
    src = ev['questions'].get(qid) or ev['reference_questions'].get(REFMAP.get(qid, ''))
    if src is None or 'primary' not in src:
        continue
    for ck in ('best', 'last'):
        e, me = src['primary'][ck], res_q[qid]['primary'][ck]
        if e['class'] != me['class'] or abs(e['gain_us'] - me['gain_us']) > 0.002:
            mism.append(('question', qid, ck, e['class'], e['gain_us'], me['class'], me['gain_us']))
    if src.get('final_class') and src['final_class'] != res_q[qid]['primary']['final']:
        mism.append(('final', qid, src['final_class'], res_q[qid]['primary']['final']))
    for m in ('S1_no_battery_scene', 'S2_k3_steer', 'S3_intervention_part1', 'battery_scene_only', 'genre4', 'genre5', 'k3_throttle'):
        es = (src.get('secondary') or {}).get(m)
        if not es:
            continue
        for ck in ('best', 'last'):
            if ck in es and es[ck]['class'] != res_q[qid][m][ck]['class']:
                mism.append(('secondary', qid, m, ck, es[ck]['class'], res_q[qid][m][ck]['class']))
for qid, c in sens.items():
    e = ev['sensitivity_rule3']['rows'].get(qid, {}).get('final_class')
    if e is not None and e != c:
        mism.append(('sensitivity', qid, e, c))
for k, v in size.items():
    if k.startswith('bootstrap'):
        continue
    e = ev['size']['by_metric'].get(k)
    if e and (abs(e['slopes']['s2'] - v['s2']) > 0.002 or abs(e['slopes']['G2_pow'] - v['G2_pow']) > 0.002 or e['reading'] != v['reading']):
        mism.append(('size', k, e['slopes'], v))

out = {
    'pred_table_key': key, 'key_matches': key == KEY, 'rows': N,
    'frames': {k: int(m.sum()) for k, m in sets.items()},
    'clean_vs_ai_mix_aligned_equal': label_check,
    'impact_frames_in_human_val': int(((part == 0) & np.isin(scene, clean_val) & fv & imp).sum()),
    's_A': {k: round(v, 4) for k, v in S_A.items()}, 's_A_prime': {k: round(v, 4) for k, v in SAP.items()},
    'per_variant_primary': {ck: {n: round(M[ck][n]['primary'], 4) for n in V[ck]} for ck in ('best', 'last')},
    'per_variant_all': {ck: {n: {m: round(x, 4) for m, x in M[ck][n].items()} for n in V[ck]} for ck in ('best', 'last')},
    'questions': res_q, 'sensitivity_rule3': sens, 'gain_check': gain_check, 'gain_battery_share': share, 'te_check': te_check,
    'g_ref_recomputed': {k: round(v, 3) for k, v in g_ref_recomputed.items()},
    'gain_scale_mean_by_scene': gscale_scene,
    'size': size, 'size_eq_seed0': sizeeq, 'gainl_seed0_post_diff': gainl,
    'per_scene_best': per_scene, 'ens_better_scenes': ens_scene_better, 'prediction_disagreement_us': disagree,
    'leakage': leak, 'mismatches_vs_evaluation_json': mism,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps({k: out[k] for k in ('pred_table_key', 'key_matches', 'frames', 's_A', 's_A_prime', 'gain_check', 'te_check',
                                       'gain_battery_share', 'sensitivity_rule3', 'ens_better_scenes', 'g_ref_recomputed',
                                       'size_eq_seed0', 'gainl_seed0_post_diff', 'leakage', 'impact_frames_in_human_val')},
                 ensure_ascii=False, indent=1, default=str))
print('最終の判定:', {q: r['primary']['final'] for q, r in res_q.items()})
print('evaluation.json と合わないもの:', len(mism))
for x in mism:
    print('  ', x)
print('clean と ai_mix の aligned が同じ:', all(label_check.values()), len(label_check))
