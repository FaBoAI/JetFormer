#!/usr/bin/env python3
"""評価のレビュー(結論を崩せないかの確かめ)。CPU だけ・学習も GPU の予測のやり直しもしない。

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice -n 10 \
        jf-heavy --lock none --mem 2500M -- python -u /home/jetson/jetformer_reports/sim_cotrain_20261013/eval/review/review_check.py --cpu-predict

確かめること(結果は同じフォルダの review_check.json と標準出力。途中の段は review_parts/*.json に書き、済んだ段は飛ばす):
  0. ファイル: protocol.json が固定した sha256(混ぜたデータセットの splits・ドロップの表・10/12 の zoo のファイル)が今も同じか
  1. 保留データの漏れ: 主指標のシーン・フレーム(時刻・画像のファイル名)が C の学習に無いか、検証のシーンが A と同じか
  1b. シミュレーターの 60 シーンが元のデータセットと同じ中身か(実機のフレームが混ざっていないか)
  1c. 混ぜたデータセットの実機のシーンが clean_driving_20261011 と同じ中身か
  2. ドロップの表: drop がシミュレーターのフレームとちょうど同じか、C+FT の学習・検証のサンプルが A と同じか、微調整の初期値が C・A の best.pth か
  3. 主指標などの再計算: 評価の道具(sim_cotrain_eval.evaluate_all)を使わず、保存された予測と aligned から計算し直し、evaluation.json と比べる
  4. 後から見た内訳(判定には使わない): シーンごと、ジャンル × 場所ごとの差(速さの癖が場所で隠れていないか)
  5. 微調整で重みが動いたか
  6. (--cpu-predict)保存された予測が今の重みから出たものか(CPU・24 フレーム)
"""

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

REPO = '/home/jetson/jetformer'
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/tools')
os.chdir(REPO)
import rl_evaluate as ev  # noqa: E402
import sim_cotrain as sc  # noqa: E402
from jetformer.train_data import JetFormerDataset, impact_mask, load_frame_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, 'review_parts')
REPORT = '/home/jetson/jetformer_reports/sim_cotrain_20261013'
AM = 'data/datasets/ai_mix_20261011'
REAL = 'data/datasets/clean_driving_20261011'
MIX = 'data/datasets/simreal_mix_20261013'
SIM = 'data/datasets/jetformer-simulator-id8-20min_20261012-150956_b55616cf'
DROP = 'data/derived/simreal_mix_20261013/drop_sim_frames.npz'
Z = 'data/zoo/sim_cotrain_20261013/'
Z0 = 'data/zoo/policy_zoo_20261012/'
RUNS = {('A', 0): 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
        ('A', 1): 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
        ('B', 0): 'clean_driving_20261011-nocmd-s3-20261011-024626',
        ('C', 0): 'simreal_mix_20261013-cotrain-s0', ('C', 1): 'simreal_mix_20261013-cotrain-s1',
        ('C+FT', 0): 'simreal_mix_20261013-cotrain-ft-s0', ('C+FT', 1): 'simreal_mix_20261013-cotrain-ft-s1',
        ('A+FT', 0): 'clean_driving_20261011-jetformer_v0-s0-ft-real-lr1e-4-e12',
        ('A+FT', 1): 'clean_driving_20261011-jetformer_v0-s1-ft-real-lr1e-4-e12'}
PRED = {('A', 0): Z0 + 'pred/jetformer_v0_s0', ('A', 1): Z0 + 'pred/jetformer_v0_s1', ('B', 0): Z0 + 'pred/B',
        ('C', 0): Z + 'pred/C_s0', ('C', 1): Z + 'pred/C_s1', ('C+FT', 0): Z + 'pred/CFT_s0', ('C+FT', 1): Z + 'pred/CFT_s1',
        ('A+FT', 0): Z + 'pred/AFT_s0', ('A+FT', 1): Z + 'pred/AFT_s1'}
EVNAME = {('A', 0): 'A_s0', ('A', 1): 'A_s1', ('B', 0): 'B', ('C', 0): 'C_s0', ('C', 1): 'C_s1',
          ('C+FT', 0): 'CFT_s0', ('C+FT', 1): 'CFT_s1', ('A+FT', 0): 'AFT_s0', ('A+FT', 1): 'AFT_s1'}
PERT = ('img_down3', 'img_up3', 'bright_0.8', 'bright_1.2', 'tof_down1')


def splits(root):
    with open(root + '/splits.json') as f:
        return json.load(f)


def aligned(root, s, keys=None):
    with np.load('%s/aligned/%s.npz' % (root, s)) as z:
        return {k: z[k] for k in (keys or z.files)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def sha1_12(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:12]


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def part(name, fn, *a):
    """段ごとに review_parts/<name>.json に書く(電源断のあとは済んだ段を飛ばす)"""
    p = os.path.join(PARTS, name + '.json')
    if os.path.exists(p) and os.path.getsize(p) > 2:
        with open(p, encoding='utf-8') as f:
            out = json.load(f)
        print('[済み] %s' % name)
    else:
        out = fn(*a)
        os.makedirs(PARTS, exist_ok=True)
        with open(p + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(p + '.tmp', p)
    print('%s: %s' % (name, json.dumps(out, ensure_ascii=False)[:3000]))
    sys.stdout.flush()
    return out


# ---------------------------------------------------------------- 0
def check_files():
    proto = json.load(open(REPORT + '/protocol.json', encoding='utf-8'))
    out = {'protocol_sha256': sha256(REPORT + '/protocol.json')}
    for k, v in proto['files'].items():
        out[k] = sha256(v['path']) == v['sha256']
    return out


# ---------------------------------------------------------------- 1
def check_leak():
    sr, sm, sa = splits(REAL), splits(MIX), splits(AM)
    held_json = ev._heldout()
    hc = list(held_json['human_clean_scenes'])
    held = [s for s in sa['val'] if s in set(sr['val'])]
    ts_h, cf_h = set(), set()
    same = True
    for s in held:
        a, b = aligned(AM, s), aligned(REAL, s)
        ts_h |= set(a['timestamps_us'].tolist())
        cf_h |= set(map(str, a['cam_files']))
        same &= all(np.array_equal(a[k], b[k]) for k in ('timestamps_us', 'future_actions', 'future_valid', 'vehicle',
                                                          'vehicle_valid', 'mode', 'imu', 'tof'))
    ts_t, cf_t, n_t = set(), set(), 0
    for s in sm['train']:
        a = aligned(MIX, s, ['timestamps_us', 'cam_files'])
        ts_t |= set(a['timestamps_us'].tolist())
        cf_t |= set(os.path.basename(str(x)) for x in a['cam_files'])
        n_t += len(a['timestamps_us'])
    cf_hb = {os.path.basename(x) for x in cf_h}
    real_tr = [s for s in sm['train'] if not s.startswith('ds-')]
    logs_val = {s[:15] for s in sr['val']}
    logs_tr = {s[:15] for s in real_tr}
    return {
        'heldout_json_human_clean_equals_clean_val': sorted(hc) == sorted(sr['val']),
        'mix_val_equals_A_val_in_order': sm['val'] == sr['val'],
        'mix_train_in_val': sorted(set(sm['train']) & set(sm['val'])),
        'mix_train_in_heldout_scenes': sorted(set(sm['train']) & set(hc)),
        'real_train_same_as_A': sorted(real_tr) == sorted(sr['train']), 'n_real_train': len(real_tr),
        'n_sim_train': len(sm['train']) - len(real_tr), 'sim_names_all_ds_prefix': all(
            s.startswith('ds-jetformer-simulator') for s in sm['train'] if s not in sr['train']),
        'held_scenes': len(held), 'held_frames_all': len(ts_h), 'ai_mix_held_equals_clean_val_arrays': bool(same),
        'C_train_frames_all': n_t, 'timestamp_overlap_heldout_vs_C_train': len(ts_h & ts_t),
        'cam_basename_overlap_heldout_vs_C_train': len(cf_hb & cf_t),
        'val_recording_logs': len(logs_val), 'val_logs_also_in_train_logs': sorted(logs_val & logs_tr),
        'note_logs': '同じ録画(ログ)の別のシーンが学習と検証に分かれているのは A・C・C+FT・A+FT で同じ(腕の差にはならない)'}


def check_sim_content():
    ss, sm = splits(SIM), splits(MIX)
    first = {}
    for s in ss['train'] + ss['val']:
        first[int(aligned(SIM, s, ['timestamps_us'])['timestamps_us'][0])] = s
    bad, imgs, ref, real_t = 0, 0, [], []
    rng = np.random.default_rng(0)
    keys = ('timestamps_us', 'future_actions', 'future_valid', 'vehicle', 'vehicle_valid', 'imu', 'tof', 'mode')
    tmin = min(int(aligned(REAL, s, ['timestamps_us'])['timestamps_us'].min()) for s in splits(REAL)['train'] + splits(REAL)['val'])
    tmax = max(int(aligned(REAL, s, ['timestamps_us'])['timestamps_us'].max()) for s in splits(REAL)['train'] + splits(REAL)['val'])
    modes = set()
    sims = [x for x in sm['train'] if x.startswith('ds-')]
    for s in sims:
        a = aligned(MIX, s)
        on = first[int(a['timestamps_us'][0])]
        b = aligned(SIM, on)
        if on in set(ss['val']):
            ref.append(s[-4:])
        modes |= set(np.unique(a['mode']).tolist())
        real_t.append(int(((a['timestamps_us'] >= tmin) & (a['timestamps_us'] <= tmax)).sum()))
        for k in keys:
            bad += int(not np.array_equal(a[k], b[k]))
        for i in rng.choice(len(a['cam_files']), 2, replace=False):
            bad += int(md5(os.path.join(MIX, str(a['cam_files'][i]))) != md5(os.path.join(SIM, str(b['cam_files'][i]))))
            imgs += 1
    return {'scenes': len(sims), 'array_or_image_mismatches': bad, 'images_compared': imgs,
            'orig_val_as_mix_suffix': sorted(ref), 'sim_modes': sorted(modes),
            'sim_frames_inside_real_time_range': int(sum(real_t))}


def check_real_content():
    sr, sm = splits(REAL), splits(MIX)
    keys = ('timestamps_us', 'future_actions', 'future_valid', 'vehicle', 'vehicle_valid', 'imu', 'tof', 'mode')
    bad, imgs = [], 0
    rng = np.random.default_rng(2)
    for s in sr['train'] + sr['val']:
        a, b = aligned(MIX, s), aligned(REAL, s)
        for k in keys:
            if not np.array_equal(a[k], b[k]):
                bad.append('%s:%s' % (s, k))
        for i in rng.choice(len(a['cam_files']), 1, replace=False):
            if md5(os.path.join(MIX, str(a['cam_files'][i]))) != md5(os.path.join(REAL, str(b['cam_files'][i]))):
                bad.append('%s:img%d' % (s, i))
            imgs += 1
    return {'real_scenes': len(sr['train']) + len(sr['val']), 'mismatches': bad, 'images_compared': imgs}


# ---------------------------------------------------------------- 2
def check_drop():
    z = np.load(DROP)
    scn, frm, drop = z['scene'].astype(str), z['frame'], z['drop']
    is_sim = np.char.startswith(scn, 'ds-jetformer-simulator')
    sm = splits(MIX)
    sim_frames = sum(len(aligned(MIX, s, ['timestamps_us'])['timestamps_us']) for s in sm['train'] if s.startswith('ds-'))
    kw = dict(stride=3, use_command=False, mask_impacts=True, impact_g=2.5)

    def names(root, split, **extra):
        ds = JetFormerDataset(root, split, **kw, **extra)
        sp = splits(root)[split]
        assert len(ds.scenes) == len(sp)
        out = ([(sp[k], i) for k, i in ds.index], ds.dropped, ds.mode_counts())
        del ds
        return out
    nft, dropped, mft = names(MIX, 'train', drop=load_frame_table(DROP, 'drop'))
    na, _, ma = names(REAL, 'train')
    nc, _, mc = names(MIX, 'train')
    vft, _, _ = names(MIX, 'val', drop=load_frame_table(DROP, 'drop'))
    vc, _, _ = names(MIX, 'val')
    va, _, _ = names(REAL, 'val')
    inits = {}
    for arm, src in (('C+FT', 'C'), ('A+FT', 'A')):
        for s in (0, 1):
            r = json.load(open('data/runs/%s/result.json' % RUNS[(arm, s)]))['rl_options']
            inits['%s_s%d' % (arm, s)] = {'init': r['init'], 'init_is_%s_s%d_best' % (src, s):
                                          r['init'] == 'data/runs/%s/best.pth' % RUNS[(src, s)],
                                          'sha1_recorded': r['sha1'].get('init'),
                                          'sha1_now': sha1_12('data/runs/%s/best.pth' % RUNS[(src, s)]),
                                          'drop_mask': r['drop_mask'], 'dropped_train': r['dropped_frames']['train']}
            inits['%s_s%d' % (arm, s)]['sha1_match'] = inits['%s_s%d' % (arm, s)]['sha1_recorded'] == \
                inits['%s_s%d' % (arm, s)]['sha1_now']
    hp = {}
    for k, run in RUNS.items():
        if k[0] == 'B':
            continue
        j = json.load(open('data/runs/%s/result.json' % run))
        hp['%s_s%d' % k] = {kk: j.get(kk) for kk in ('epochs_done', 'lr', 'batch', 'stride', 'seed', 'train_samples',
                                                      'val_samples', 'best_epoch', 'command_input', 'impact_g')}
        hp['%s_s%d' % k]['dataset'] = j.get('dataset')
    return {'rows': int(len(scn)), 'drop_true': int(drop.sum()), 'sim_rows': int(is_sim.sum()), 'sim_frames_in_mix': int(sim_frames),
            'drop_equals_sim_rows': bool((drop == is_sim).all()),
            'sim_rows_cover_all_frames': int(len(set(zip(scn[is_sim].tolist(), frm[is_sim].tolist())))) == sim_frames,
            'FT_train_same_as_A_incl_order': nft == na, 'FT_sim_samples': sum(s.startswith('ds-') for s, _ in nft),
            'FT_train': len(nft), 'FT_dropped_before_stride': dropped, 'C_train': len(nc),
            'C_sim_samples': sum(s.startswith('ds-') for s, _ in nc),
            'C_real_part_same_as_A': [x for x in nc if not x[0].startswith('ds-')] == na,
            'mode_counts': {'A': ma, 'C': mc, 'FT': mft},
            'val_same_C_FT_A': vft == va and vc == va, 'val_samples': len(va),
            'optimizer_steps_per_run': {'A': (len(na) // 64 + (len(na) % 64 > 0)) * 30,
                                        'C': (len(nc) // 64 + (len(nc) % 64 > 0)) * 30,
                                        'FT': (len(nft) // 64 + (len(nft) % 64 > 0)) * 12},
            'real_samples_seen': {'A': len(na) * 30, 'C': (len(nc) - sum(s.startswith('ds-') for s, _ in nc)) * 30,
                                  'C+FT': len(na) * 42, 'A+FT': len(na) * 42},
            'inits': inits, 'hyper': hp}


# ---------------------------------------------------------------- 3
def primary_frames():
    tab = ev.pred_table()
    cv = set(splits(REAL)['val'])
    n = len(tab['part'])
    y = np.full((n, 10, 2), np.nan)
    clean, mode, loc = np.zeros(n, bool), np.zeros(n, int), np.zeros(n, int)
    cache = {}
    for i in np.nonzero((tab['part'] == 0) & np.isin(tab['scene'], list(cv)))[0]:
        s = tab['scene'][i]
        if s not in cache:
            a = aligned(AM, s, ['timestamps_us', 'imu', 'future_actions', 'future_valid', 'vehicle_valid', 'mode', 'loc_code'])
            a['_imp'] = impact_mask(a['timestamps_us'], a['imu'], 2.5)
            cache[s] = a
        a, f = cache[s], tab['frame'][i]
        y[i], mode[i], loc[i] = a['future_actions'][f], a['mode'][f], a['loc_code'][f]
        clean[i] = bool(a['future_valid'][f] and a['vehicle_valid'][f] and not a['_imp'][f])
    return tab, y, clean, mode, loc


def load_preds(tab, clean, mode):
    P = {}
    for k, base in PRED.items():
        for ck in ('best', 'last'):
            z = np.load(base + ('' if ck == 'best' else '_last') + '.npz')
            assert str(z['key']) == tab['key'] and str(z['ckpt']) == ck and str(z['run']) == RUNS[k], (k, ck)
            assert (z['genre'][clean] == mode[clean]).all() and not z['substituted'][clean].any(), (k, ck)
            P[k + (ck,)] = z['pred'].astype(np.float64)
    return P


def recompute(tab, y, clean, mode, P):
    evj = json.load(open(REPORT + '/evaluation.json', encoding='utf-8'))
    L = {k: float(np.abs(p[clean] - y[clean]).mean()) for k, p in P.items()}
    sA = {ck: max(L[(a, s, ck)] for a, s in (('A', 0), ('A', 1), ('B', 0))) -
          min(L[(a, s, ck)] for a, s in (('A', 0), ('A', 1), ('B', 0))) for ck in ('best', 'last')}

    def vals(arm, ck):
        return [L[(arm, s, ck)] for s in (0, 1)]
    verdicts = {}
    for arm, ref in (('C', 'A'), ('C+FT', 'A'), ('A+FT', 'A'), ('C+FT', 'C'), ('C+FT', 'A+FT')):
        floor = {'best': 0.313, 'last': 0.661}                  # protocol.json で固定した s_A
        res = {ck: sc.judge_primary(vals(arm, ck), vals(ref, ck), 2.0, floor[ck]) for ck in ('best', 'last')}
        verdicts['%s_vs_%s' % (arm, ref)] = {'final': sc.final_verdict(res['best'], res['last']),
                                             'class_best': res['best']['class'], 'class_last': res['last']['class'],
                                             'diff_best': round(-res['best']['gain_us'], 3), 'thr_best': round(res['best']['threshold_us'], 3),
                                             'diff_last': round(-res['last']['gain_us'], 3), 'thr_last': round(res['last']['threshold_us'], 3)}
    # evaluation.json との差
    mism = []
    for k in PRED:
        for ck in ('best', 'last'):
            if abs(round(L[k + (ck,)], 3) - evj['primary']['L_us'][ck][EVNAME[k]]) > 0.0015:
                mism.append(('L', k, ck, L[k + (ck,)], evj['primary']['L_us'][ck][EVNAME[k]]))
    for name, v in verdicts.items():
        ev_v = evj['verdict'].get(name)
        if ev_v is not None and ev_v != v['final']:
            mism.append(('verdict', name, v['final'], ev_v))
    # 種どうしの順位(2 つの種の両方が A の 2 本より上: 差が無いときの確率 = 1/6)
    order = {}
    for arm in ('C', 'C+FT', 'A+FT'):
        order[arm] = {ck: {'all_above_A': min(vals(arm, ck)) > max(vals('A', ck)),
                           'all_below_A': max(vals(arm, ck)) < min(vals('A', ck))} for ck in ('best', 'last')}
    real_only = [L[(a, s, 'best')] for a, s in (('A', 0), ('A', 1), ('A+FT', 0), ('A+FT', 1), ('B', 0))]
    mixed = [L[(a, s, 'best')] for a, s in (('C', 0), ('C', 1), ('C+FT', 0), ('C+FT', 1))]
    # 主指標のフレームの一致(genre 別の数)
    frames = {'clean': int(clean.sum()), 'g4': int((clean & (mode == 4)).sum()), 'g5': int((clean & (mode == 5)).sum())}
    return {'frames': frames, 'L': {'%s_s%d_%s' % k: round(v, 4) for k, v in L.items()},
            's_A_recomputed': {k: round(v, 4) for k, v in sA.items()},
            'arm_mean': {a: {ck: round(float(np.mean(vals(a, ck))), 4) for ck in ('best', 'last')} for a in ('A', 'C', 'C+FT', 'A+FT')},
            'seed_range_best': {a: round(max(vals(a, 'best')) - min(vals(a, 'best')), 4) for a in ('A', 'C', 'C+FT', 'A+FT')},
            'verdicts': verdicts, 'seed_order_vs_A': order,
            'pairs_mixed_gt_realonly_best': '%d/%d' % (sum(m > r for m, r in itertools.product(mixed, real_only)),
                                                       len(mixed) * len(real_only)),
            'mismatch_vs_evaluation_json': mism}


def recompute_robust_sim(tab, y, clean, P):
    evj = json.load(open(REPORT + '/evaluation.json', encoding='utf-8'))
    rows = None
    dl1, per = {}, {}
    for arm, key in (('A', None), ('C', 'C'), ('C+FT', 'CFT'), ('A+FT', 'AFT')):
        v, pp_ = [], []
        for s in (0, 1):
            pp = (Z0 + 'pert/jetformer_v0_s%d.npz' % s) if key is None else (Z + 'pert/%s_s%d.npz' % (key, s))
            zz = np.load(pp)
            assert str(zz['run']) == RUNS[(arm, s)] and str(zz['ckpt']) == 'best'
            if rows is None:
                rows = zz['rows']
            assert np.array_equal(zz['rows'], rows)
            l0 = np.abs(P[(arm, s, 'best')][rows] - y[rows]).mean()
            d = {k: float(np.abs(zz['pred_' + k].astype(np.float64) - y[rows]).mean() - l0) for k in PERT}
            pp_.append(d)
            v.append(float(np.mean(list(d.values()))))
        dl1[arm] = v
        per[arm] = {k: float(np.mean([d[k] for d in pp_])) for k in PERT}
        per[arm + '_seeds'] = {k: [round(d[k], 3) for d in pp_] for k in PERT}
    rows_in_clean = bool(clean[rows].all())
    ref = float(np.mean(dl1['A']))
    contrib = {}
    for arm in ('C', 'C+FT'):
        gain = {k: per['A'][k] - per[arm][k] for k in PERT}
        tot = sum(gain.values())
        contrib[arm] = {k: {'gain_us': round(g, 3), 'share': round(g / tot, 2),
                            'both_seeds_below_both_A': max(per[arm + '_seeds'][k]) < min(per['A_seeds'][k])}
                        for k, g in gain.items()}
    rob = {'frames': int(len(rows)), 'rows_subset_of_clean': rows_in_clean,
           'dL1_seeds': {a: [round(x, 3) for x in v] for a, v in dl1.items()},
           'ratio_to_A': {a: round(float(np.mean(v)) / ref, 3) for a, v in dl1.items()},
           'robust_good': {a: bool(np.mean(v) <= 0.8 * ref and max(v) <= ref) for a, v in dl1.items() if a != 'A'},
           'per_pert_mean': {a: {k: round(x, 3) for k, x in per[a].items()} for a in ('A', 'C', 'C+FT', 'A+FT')},
           'contribution_to_gain': contrib,
           'ft_change_dL1_per_seed': [round(dl1['C+FT'][s] - dl1['C'][s], 3) for s in (0, 1)],
           'ft_lost_share_of_C_gain': round((np.mean(dl1['C+FT']) - np.mean(dl1['C'])) / (ref - np.mean(dl1['C'])), 3),
           'eval_json_ratio': {a: evj['robustness']['per_arm'][a]['ratio_to_ref'] for a in ('C', 'C+FT', 'A+FT')}}
    # シミュレーターの L1
    ss = splits(SIM)['val']
    ys, oks, scn, frm = [], [], [], []
    for s in ss:
        a = aligned(SIM, s, ['future_actions', 'future_valid', 'vehicle_valid', 'timestamps_us', 'imu'])
        ys.append(a['future_actions'])
        oks.append(a['future_valid'] & a['vehicle_valid'] & ~impact_mask(a['timestamps_us'], a['imu'], 2.5))
        scn += [s] * len(a['future_valid'])
        frm += list(range(len(a['future_valid'])))
    ysim, ok = np.concatenate(ys).astype(np.float64), np.concatenate(oks)
    sim, order_ok = {}, True
    for arm, key in (('A', 'A'), ('C', 'C'), ('C+FT', 'CFT'), ('A+FT', 'AFT')):
        for h in (4, 5, 8):
            v, fb = [], []
            for s in (0, 1):
                zz = np.load(Z + 'sim/%s_s%d.npz' % (key, s))
                order_ok &= list(zz['scene'].astype(str)) == scn and list(zz['frame']) == frm
                order_ok &= str(zz['run']) == RUNS[(arm, s)]
                if 'pred_h%d' % h in zz.files:
                    p = zz['pred_h%d' % h][ok].astype(np.float64)
                    v.append(float(np.abs(p - ysim[ok]).mean()))
                    fb.append(float((p[:, 2, 1] - ysim[ok][:, 2, 1]).mean()))
            if v:
                sim['%s_h%d' % (arm, h)] = round(float(np.mean(v)), 3)
                sim['%s_h%d_seeds' % (arm, h)] = [round(x, 2) for x in v]
                sim['%s_h%d_fwd_bias' % (arm, h)] = round(float(np.mean(fb)), 1)
    sim['share_lost'] = round((sim['C+FT_h8'] - sim['C_h8']) / (sim['A_h5'] - sim['C_h8']), 4)
    sim['frames'] = int(ok.sum())
    sim['npz_order_and_run_ok'] = bool(order_ok)
    return {'robust': rob, 'sim': sim}


# ---------------------------------------------------------------- 4
def breakdown(tab, y, clean, mode, loc, P):
    E = {k: np.abs(p - y) for k, p in P.items() if k[2] == 'best'}
    scn = tab['scene']
    names = ev.LOC_NAMES

    def seeds(arm):
        return (0,) if arm == 'B' else (0, 1)

    def mean_l1(arm, m):
        return float(np.mean([E[(arm, s, 'best')][m].mean() for s in seeds(arm)]))

    def fb(arm, m):
        return float(np.mean([(P[(arm, s, 'best')][m][:, 2, 1] - y[m][:, 2, 1]).mean() for s in seeds(arm)]))
    per_scene = {}
    for s in sorted(set(scn[clean])):
        m = clean & (scn == s)
        per_scene[s] = {'n': int(m.sum()), 'genre': int(np.bincount(mode[m]).argmax()), 'A_L1': round(mean_l1('A', m), 2),
                        'C-A': round(mean_l1('C', m) - mean_l1('A', m), 2), 'C+FT-A': round(mean_l1('C+FT', m) - mean_l1('A', m), 2),
                        'A+FT-A': round(mean_l1('A+FT', m) - mean_l1('A', m), 2)}
    worst = sorted(per_scene, key=lambda s: -per_scene[s]['C-A'])[:2]
    for s in worst:
        m = clean & (scn == s)
        per_scene[s]['per_run_L1'] = {'%s_s%d' % k[:2]: round(float(E[k][m].mean()), 2) for k in E}
        per_scene[s]['HS_frames'] = int((m & (loc == names.index('HS'))).sum())
        mh = m & (loc == names.index('HS'))
        per_scene[s]['HS_C-A_L1'] = round(mean_l1('C', mh) - mean_l1('A', mh), 2) if mh.sum() else None
    contrib = {a: {s: round(float(per_scene[s]['n'] * per_scene[s][a + '-A'] / clean.sum()), 3) for s in worst}
               for a in ('C', 'C+FT')}
    m2 = clean & ~np.isin(scn, worst)
    cells = {}
    for g in (4, 5):
        for c in range(len(names)):
            m = clean & (mode == g) & (loc == c)
            if m.sum() < 100:
                continue
            cells['g%d_%s' % (g, names[c])] = {
                'n': int(m.sum()), 'fwd_shift_C': round(fb('C', m) - fb('A', m), 1), 'fwd_shift_CFT': round(fb('C+FT', m) - fb('A', m), 1),
                'fwd_shift_AFT': round(fb('A+FT', m) - fb('A', m), 1), 'A_seed_gap_fwd': round(
                    abs((P[('A', 0, 'best')][m][:, 2, 1] - P[('A', 1, 'best')][m][:, 2, 1]).mean()), 1),
                'L1_C-A': round(mean_l1('C', m) - mean_l1('A', m), 2),
                'fwd_bias_per_run': {'%s_s%d' % k[:2]: round(float((P[k][m][:, 2, 1] - y[m][:, 2, 1]).mean()), 1)
                                     for k in P if k[2] == 'best'}}
    return {'per_scene': per_scene, 'scenes_C_better': sum(v['C-A'] < 0 for v in per_scene.values()),
            'scenes_CFT_better': sum(v['C+FT-A'] < 0 for v in per_scene.values()),
            'scenes_AFT_better': sum(v['A+FT-A'] < 0 for v in per_scene.values()), 'two_worst_scenes_for_C': worst,
            'without_two_worst': {a: round(mean_l1(a, m2) - mean_l1('A', m2), 3) for a in ('C', 'C+FT', 'A+FT')},
            'two_worst_contribution_to_overall_diff_us': contrib,
            'cells': cells, 'max_abs_fwd_shift_cell': max(max(abs(v['fwd_shift_C']), abs(v['fwd_shift_CFT'])) for v in cells.values()),
            'max_A_seed_gap_fwd_cell': max(v['A_seed_gap_fwd'] for v in cells.values())}


# ---------------------------------------------------------------- 5・6
def weight_change():
    import torch
    from jetformer.policies import runtime
    out = {}
    for src, dst in ((('C', 0), ('C+FT', 0)), (('C', 1), ('C+FT', 1)), (('A', 0), ('A+FT', 0)), (('A', 1), ('A+FT', 1))):
        s1 = runtime.load_state('data/runs/%s/best.pth' % RUNS[src])
        s2 = runtime.load_state('data/runs/%s/best.pth' % RUNS[dst])
        g = {'params': [0.0, 0.0], 'bn_stats': [0.0, 0.0]}
        for k in s1:
            if not torch.is_floating_point(s1[k]):
                continue
            key = 'bn_stats' if 'running_' in k else 'params'
            g[key][0] += float((s2[k].float() - s1[k].float()).norm() ** 2)
            g[key][1] += float(s1[k].float().norm() ** 2)
        out['%s_s%d->%s' % (src[0], src[1], dst[0])] = {k: round((v[0] / v[1]) ** 0.5, 4) for k, v in g.items()}
        # 頭 8(シミュレーター)の重みが C+FT で残っているか: ジャンルの埋め込みの 8 列目の差
        if src[0] == 'C':
            emb = [k for k in s1 if s1[k].ndim == 2 and s1[k].shape[-1] == 9]
            out['%s_s%d->%s' % (src[0], src[1], dst[0])]['mode_weight_keys'] = emb
            out['%s_s%d->%s' % (src[0], src[1], dst[0])]['mode8_col_rel_change'] = [
                round(float((s2[k][:, 8] - s1[k][:, 8]).norm() / (s1[k][:, 8].norm() + 1e-12)), 4) for k in emb]
    return out


def cpu_predict(tab):
    import torch
    import policy_zoo as pz
    from jetformer.policies import runtime
    torch.set_num_threads(4)
    cv = set(splits(REAL)['val'])
    cand = np.nonzero((tab['part'] == 0) & np.isin(tab['scene'], list(cv)))[0]
    rows = np.sort(np.random.default_rng(1).choice(cand, 24, replace=False))
    fs = pz._Frames(tab, rows)
    b = [fs[i] for i in range(len(fs))]
    x0 = {k: torch.stack([t[j] for t in b]) for j, k in enumerate(('image', 'tof', 'imu', 'state'))}
    out = {}
    for k in [k for k in RUNS if k[0] in ('C', 'C+FT', 'A+FT')]:
        for ck in ('best', 'last'):
            model, cfg, _, _ = runtime.load_run('data/runs/' + RUNS[k], ck, 'cpu')
            z = np.load(PRED[k] + ('' if ck == 'best' else '_last') + '.npz')
            g = torch.from_numpy(z['genre'][rows].astype(np.int64))
            with torch.no_grad():
                p = runtime.forward(model, dict(x0, mode=torch.nn.functional.one_hot(g, cfg['num_modes']).float()))['actions']
            out['%s_s%d_%s' % (k + (ck,))] = round(float(np.abs(p.float().numpy() * 500.0 + 1500.0 - z['pred'][rows]).max()), 3)
            # 違う学習結果の予測とは大きく違うこと(比べ方が効いているかの確かめ)
            if ck == 'best' and k == ('C+FT', 0):
                zo = np.load(PRED[('C', 0)] + '.npz')
                out['control_CFT_s0_weights_vs_C_s0_pred'] = round(float(np.abs(p.float().numpy() * 500.0 + 1500.0 - zo['pred'][rows]).max()), 3)
            del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu-predict', action='store_true')
    args = ap.parse_args()
    OUT = {}
    OUT['files'] = part('0_files', check_files)
    OUT['leak'] = part('1_leak', check_leak)
    OUT['sim_content'] = part('1b_sim_content', check_sim_content)
    OUT['real_content'] = part('1c_real_content', check_real_content)
    OUT['drop'] = part('2_drop', check_drop)
    tab, y, clean, mode, loc = primary_frames()
    P = load_preds(tab, clean, mode)
    OUT['recompute'] = part('3_recompute', recompute, tab, y, clean, mode, P)
    OUT['robust_sim'] = part('3b_robust_sim', recompute_robust_sim, tab, y, clean, P)
    OUT['breakdown'] = part('4_breakdown', breakdown, tab, y, clean, mode, loc, P)
    del P
    OUT['weight_change'] = part('5_weight_change', weight_change)
    if args.cpu_predict:
        OUT['cpu_predict_max_abs_us'] = part('6_cpu_predict', cpu_predict, tab)
    path = os.path.join(HERE, 'review_check.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, ensure_ascii=False, indent=1)
    os.replace(path + '.tmp', path)
    print('書きました:', path)


if __name__ == '__main__':
    main()
