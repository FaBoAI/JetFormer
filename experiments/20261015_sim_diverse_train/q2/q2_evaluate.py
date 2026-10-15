#!/usr/bin/env python3
"""Q2 の評価(CPU): 予測の npz から、protocol.json の eval_Q2 の規則で C-DIV を A・C と比べる → eval_real.json

q2_eval.py evaluate から呼ぶ(CUDA_VISIBLE_DEVICES= で)。10/13 の sim_cotrain_eval.evaluate_all の写しを、腕 A・C・C-DIV にしたもの。
  - 主指標: 保留した実機の人のきれいな走行 11 シーン・4,670 フレームの L1(best・last)
  - A と比べる: judge_primary(spread_floor = s_A。閾値 best 0.626・last 1.322 µs)→ final_verdict
  - C と比べる: 同じ規則で基準を C の 2 つの種に(s = max(C の種の範囲 0.477, s_A)→ 閾値 best 0.954・last 1.322 µs)
  - 種 1 が無いときは「暫定(種 1 つ)」: judge_primary に種 0 の 1 本だけを渡す(protocol の single_seed_fallback と同じ式)
  - 参考: ジャンル別・速さの癖(ジャンル平均 + 低速の直線 HS)・R1/DE・崩れ・ずれへの強さ・シミュレーターの L1(前の教師の val 12 シーン、
    5 モードの val 15 シーン)・C が悪かった 2 シーン・ゲート(エンジンは作らない = protocol の no_engine)・学習
"""

import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

import q2_eval as qe
from q2_eval import _r, arm_names, name_table, pz, ev, rp, sc

CKPTS = qe.CKPTS
FLOOR_A = None          # protocol から読む(10/13 の baseline_A.spread_us)
C_RANGE_FIXED = 0.477   # protocol eval_Q2.primary.vs_C: s = max(C の種の範囲 0.477, s_A)


def _proto():
    p = qe.REPORT_DIR / 'protocol.json'
    sha = qe._sha256(p)
    frozen = (qe.REPORT_DIR / 'protocol.sha256').read_text().split()[0]
    return json.loads(p.read_text(encoding='utf-8')), sha, frozen


def _load_preds(names, key):
    preds, extras = {ck: {} for ck in CKPTS}, {}
    for n in names:
        for ck in CKPTS:
            p = qe.pred_file(n, ck)
            if not qe.npz_ok(p, ('pred',)):
                raise SystemExit('予測がありません: %s(predict を実行)' % p)
            with np.load(p) as z:
                if str(z['key']) != key:
                    raise SystemExit('%s は別のフレームの表で作られています' % p)
                if 'run' in z.files and str(z['run']) != name_table()[n]['run']:
                    raise SystemExit('%s の学習結果の名前が違います: %s' % (p, z['run']))
                preds[ck][n] = z['pred'].astype(np.float64)
                if ck == 'best':
                    extras[n] = {k: z[k] for k in z.files if k != 'pred'}
    return preds, extras


def compare(arm_vals, ref_vals, s_floor):
    out = {ck: sc.judge_primary(arm_vals[ck], ref_vals[ck], sc.SPREAD_MULT, s_floor[ck], sc.PRACTICAL_US) for ck in CKPTS}
    out['final_class'] = sc.final_verdict(out['best'], out['last'])
    out['practical'] = bool(out['best']['practical'])
    out['n_seeds'] = len(arm_vals['best'])
    out['provisional'] = len(arm_vals['best']) < 2
    out['consistent_direction'] = pz.consistent_direction(arm_vals, ref_vals)
    out['diff_best_us'] = -out['best']['gain_us']          # + = 腕のほうが大きい(悪い)
    out['diff_last_us'] = -out['last']['gain_us']
    out['diff_pct_best'] = 100.0 * out['diff_best_us'] / out['best']['ref_mean']
    return out


def speed_habit(fb_arm, fb_ref, th=qe.SPEED_HABIT_US):
    shift = float(np.mean(fb_arm)) - float(np.mean(fb_ref))
    return {'fwd_bias_us': float(np.mean(fb_arm)), 'fwd_bias_seeds_us': [float(x) for x in fb_arm],
            'ref_fwd_bias_us': float(np.mean(fb_ref)), 'shift_us': shift, 'moved': bool(abs(shift) >= th)}


def _r1_de(names, arms_cmp, outs, tab, cols, f, held, prm, scene, m0):
    """R1・DE(10/13 と同じ計算)。arms_cmp = [(ラベル, 腕の名前, 基準の名前)]"""
    with open(rp.rl_dir(ev.DATASET) / 'passes.json') as fh:
        passes = json.load(fh)
    r1p, dep = ev.heldout_passes(passes, f['log'], f['timestamps_us'], f['split'])
    if len(r1p) != held['counts']['r1_passes'] or len(dep) != held['counts']['de_passes']:
        raise SystemExit('保留の通過の数が heldout.json と違います')
    prow = np.full(len(f['scene']), -1, int)
    prow[cols['frow'][m0]] = np.nonzero(m0)[0]
    nb = int(prm['bootstrap_n'])
    items = {}
    for item, ps, ch in (('r1', r1p, 1), ('de', dep, 0)):
        idx = pz.bootstrap_index(len(ps), nb, prm['bootstrap_seed'])
        wv = {ck: pz._window_values(ps, prow, outs[ck], ch) for ck in CKPTS}
        rows = wv['best'][1]

        def judge(p_names, r_names, ck='best'):
            P = np.array([wv[ck][0][x] for x in p_names])
            R = np.array([wv[ck][0][x] for x in r_names])
            d_row = (np.mean([outs[ck][x][:, ch] for x in p_names], axis=0)
                     - np.mean([outs[ck][x][:, ch] for x in r_names], axis=0))
            adj = pz._adjusted_windows(d_row, scene, m0, rows)
            return pz.judge_pass_item(P, R, adj, idx, prm['ci_good'], prm['ci_bad'], prm['r1_de_good_min_pair_d_us'])

        it = {'passes': len(ps), 'ai_passes': int(sum(p['ai'] for p in ps)),
              'window_mean_us': {ck: {n: _r(float(np.mean(v)), 1) for n, v in wv[ck][0].items()} for ck in CKPTS},
              'compare': {}, 'compare_last': {}}
        for lab, pn, rn in arms_cmp:
            it['compare'][lab] = pz._round_tree(judge(pn, rn))
            it['compare_last'][lab] = pz._round_tree(judge(pn, rn, 'last'))
        it['model_B_vs_A'] = pz._round_tree(judge(['B'], arm_names('A')))
        it['nc_1012'] = {'NC-A': pz._round_tree(judge(['A_s1'], ['A_s0', 'B'])),
                         'NC-B': pz._round_tree(judge(['A_s0'], ['A_s1', 'B']))}
        items[item] = it
    return items


def _sim_eval(names, cfgs, datasets, file_of, label):
    """シミュレーターの L1(参考)。フレーム = future_valid & vehicle_valid & 2.5G の衝突の前後でない & ジャンル 8"""
    from jetformer.train_data import impact_mask
    ys, oks, ds_, sc_, frs = [], [], [], [], []
    scenes = []
    for ds in datasets:
        for s in ev._splits(ds)['val']:
            a = ev._aligned(ds, s, ['timestamps_us', 'imu', 'future_actions', 'future_valid', 'vehicle_valid', 'mode'])
            imp = impact_mask(a['timestamps_us'], a['imu'], sc.TRAIN_KW['impact_g'])
            ys.append(a['future_actions'])
            oks.append(a['future_valid'] & a['vehicle_valid'] & ~imp & (a['mode'] == sc.SIM_MODE))
            ds_.append(np.full(len(imp), ds))
            sc_.append(np.full(len(imp), s))
            frs.append(np.arange(len(imp)))
            scenes.append('%s/%s' % (ds, s))
    y, ok = np.concatenate(ys), np.concatenate(oks)
    dset, scene, frame = np.concatenate(ds_), np.concatenate(sc_), np.concatenate(frs)
    out = {'label': label, 'datasets': list(datasets), 'scenes': len(scenes), 'frames_all': int(len(y)),
           'frames': int(ok.sum()), 'excluded_frames': int((~ok).sum()), 'per_run': {}, 'per_arm': {}}
    for n in names:
        p = file_of(n)
        if not qe.npz_ok(p, ('heads', 'scene', 'frame')):
            out['per_run'][n] = None
            continue
        with np.load(p) as z:
            if not (np.array_equal(z['scene'].astype(str), scene.astype(str)) and np.array_equal(z['frame'], frame)):
                raise SystemExit('%s のフレームが今のシミュレーターのデータセットと合いません' % p)
            if 'run' in z.files and str(z['run']) != name_table()[n]['run']:
                raise SystemExit('%s の学習結果の名前が違います' % p)
            heads = [int(h) for h in z['heads']]
            row = {}
            for h in heads:
                pr = z['pred_h%d' % h]
                m = qe.masked_l1(pr, y, ok)
                if len(datasets) > 1:
                    m['per_dataset_l1_us'] = {qe.MODE_LABELS.get(d, d): _r(qe.masked_l1(pr, y, ok & (dset == d))['l1_us'], 2)
                                              for d in datasets}
                row['h%d' % h] = {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
            row['num_modes'] = int(cfgs[n]['num_modes'])
            out['per_run'][n] = row
    for arm in ('A', 'B', 'C', 'C-DIV'):
        nm = [x for x in arm_names(arm, names) if out['per_run'].get(x)]
        row = {}
        for h in qe.SIM_HEADS:
            key = 'h%d' % h
            if nm and all(key in out['per_run'][x] for x in nm):
                v = [out['per_run'][x][key]['l1_us'] for x in nm]
                row[key] = {'l1_mean_us': _r(np.mean(v), 3), 'l1_seeds_us': v,
                            'fwd_bias_mean_us': _r(np.mean([out['per_run'][x][key]['fwd_bias_us'] for x in nm]), 2)}
                if len(datasets) > 1:
                    row[key]['per_dataset_l1_mean_us'] = {
                        lab: _r(np.mean([out['per_run'][x][key]['per_dataset_l1_us'][lab] for x in nm]), 2)
                        for lab in out['per_run'][nm[0]][key]['per_dataset_l1_us']}
        out['per_arm'][arm] = row
    return out


def evaluate_all():
    import compare_models as cm
    proto, proto_sha, frozen_sha = _proto()
    p13 = json.loads(qe.PROTO_1013.read_text(encoding='utf-8'))
    e13 = json.loads(qe.EVAL_1013.read_text(encoding='utf-8'))
    base = p13['baseline_A']
    rules = pz.load_rules()
    prm = rules['parameters']
    table = name_table()
    names = qe.available_names()
    cdiv = arm_names('C-DIV', names)
    if not cdiv:
        raise SystemExit('C-DIV の学習結果がまだありません')
    single = len(cdiv) < 2
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key'] or tab['key'] != base['frames']['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json・10/13 の protocol.json と違います')
    preds, extras = _load_preds(names, tab['key'])
    cols, f = ev._val_arrays(tab)
    cfgs = {n: rp._read_json(rp.RUNS_DIR / table[n]['run'] / 'model_config.json') for n in names}
    resj = {n: rp._read_json(rp.RUNS_DIR / table[n]['run'] / 'result.json') or {} for n in names}
    refs = {n: (float(cfgs[n]['steer_center_us']), float(cfgs[n]['throttle_neutral_us'])) for n in names}
    dset = prm['drive_settings']

    def outs_of(ref_map):
        return {ck: {n: ev.drive_outputs(preds[ck][n], ref_map[n], dset['steer_gain'], dset['throttle_gain'],
                                         dset['lookahead'], dset['steer_trim_us']) for n in names} for ck in CKPTS}
    outs = outs_of(refs)
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    scene = tab['scene']
    res = {'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'question': proto['questions']['Q2_REAL_ACCURACY'],
           'protocol': {'path': str(qe.REPORT_DIR / 'protocol.json'), 'sha256': proto_sha, 'frozen_sha256': frozen_sha,
                        'unchanged': proto_sha == frozen_sha, 'written_at': proto['written_at']},
           'protocol_1013_baseline': str(qe.PROTO_1013),
           'zoo_rules': {'path': str(pz.rules_path()), 'written_at': rules['written_at']},
           'names': {n: dict(v, label=qe.ARM_LABELS[v['arm']],
                             pred_source={'1012': 'data/zoo/policy_zoo_20261012(10/12)', '1013': 'data/zoo/sim_cotrain_20261013(10/13)',
                                          None: 'data/zoo/%s(今日)' % qe.EVAL_NAME}[v['src']])
                     for n, v in table.items() if n in names},
           'missing': [n for n in table if n not in names],
           'single_seed': single,
           'refs_us': refs, 'drive_settings': dset,
           'frames': {'pred_table': int(len(tab['part'])), 'human_clean': int(clean.sum()), 'pred_table_key': tab['key'],
                      'human_clean_genre4': int((clean & (cols['mode'] == 4)).sum()),
                      'human_clean_genre5': int((clean & (cols['mode'] == 5)).sum()),
                      'human_clean_scenes': len(held['human_clean_scenes'])},
           'genre_substituted': {n: int(np.asarray(extras[n]['substituted']).sum()) for n in names}}
    finite = {n: bool(all(np.isfinite(preds[ck][n]).all() for ck in CKPTS)) for n in names}
    res['finite'] = finite
    res['crosscheck'] = rp._read_json(qe.eval_dir() / 'crosscheck.json')
    # ---- 主指標
    l1 = {ck: {n: pz.l1_rows(preds[ck][n], cols['y']) for n in names} for ck in CKPTS}
    L = {ck: {n: float(l1[ck][n][clean].mean()) for n in names} for ck in CKPTS}
    vals = {arm: {ck: [L[ck][x] for x in arm_names(arm, names)] for ck in CKPTS} for arm in qe.ARMS}
    s_a = {ck: float(base['spread_us'][ck]) for ck in CKPTS}
    s_c = {'best': max(C_RANGE_FIXED, s_a['best']), 'last': max(C_RANGE_FIXED, s_a['last'])}
    prim = {'L_us': {ck: {n: _r(v, 3) for n, v in L[ck].items()} for ck in CKPTS},
            's_A_us': {ck: _r(v, 3) for ck, v in s_a.items()},
            'threshold_vs_A_us': {ck: _r(sc.SPREAD_MULT * v, 3) for ck, v in s_a.items()},
            'threshold_vs_C_us': {ck: _r(sc.SPREAD_MULT * max(v, max(vals['C'][ck]) - min(vals['C'][ck])), 3)
                                  for ck, v in s_c.items()},
            'arm_mean_us': {arm: {ck: _r(np.mean(vals[arm][ck]), 3) for ck in CKPTS} for arm in qe.ARMS},
            'arm_range_us': {arm: {ck: [_r(min(vals[arm][ck]), 3), _r(max(vals[arm][ck]), 3)] for ck in CKPTS}
                             for arm in qe.ARMS},
            'seed_range_us': {arm: {ck: _r(max(vals[arm][ck]) - min(vals[arm][ck]), 3) for ck in CKPTS} for arm in qe.ARMS},
            'best_minus_last_us': {n: _r(L['best'][n] - L['last'][n], 3) for n in names}}
    prim['CDIV_vs_A'] = pz._round_tree(compare(vals['C-DIV'], vals['A'], s_a))
    prim['CDIV_vs_C'] = pz._round_tree(compare(vals['C-DIV'], vals['C'], s_c))
    prim['C_vs_A'] = pz._round_tree(compare(vals['C'], vals['A'], s_a))            # 10/13 の判定の再現
    prim['B_minus_A_best_us'] = _r(L['best']['B'] - np.mean(vals['A']['best']), 3)
    # 種 0 だけで判定したら(種 1 があっても、削ったときの判定がどうなったかを並べる)
    s0 = {ck: [L[ck]['CDIV_s0']] for ck in CKPTS}
    prim['seed0_only'] = {'vs_A': pz._round_tree(compare(s0, vals['A'], s_a)),
                          'vs_C': pz._round_tree(compare(s0, vals['C'], s_c))}
    prim['per_scene_L_us'] = {n: {s: _r(l1['best'][n][clean & (scene == s)].mean(), 2) for s in held['human_clean_scenes']}
                              for n in names}
    res['primary'] = prim
    # ---- ジャンル別
    acc = pz._accuracy(preds, cols, tab, held, refs, l1, names)
    res['accuracy'] = acc
    hab = {}
    for arm in ('C', 'C-DIV'):
        hab[arm] = {}
        for g in (4, 5):
            k = 'human_clean_genre%d' % g
            hab[arm][str(g)] = speed_habit([acc['metrics'][x][k]['fwd_bias_us'] for x in arm_names(arm, names)],
                                           [acc['metrics'][x][k]['fwd_bias_us'] for x in arm_names('A')])
    # 低速の直線 HS(10/13 の確認で見つかった所。今日は先に決めた参考の測り)と、場所 × ジャンルの全部のマス(100 フレーム以上)
    loc = np.full(len(tab['part']), -1, int)
    loc[m0] = f['loc_code'][cols['frow'][m0]]
    y = cols['y']

    def fb(n, m):
        return float((preds['best'][n][m][:, 2, 1] - y[m][:, 2, 1]).mean())

    def ml1(n, m):
        return float(l1['best'][n][m].mean())
    cells = {}
    for g in (4, 5):
        for c, cname in enumerate(ev.LOC_NAMES):
            m = clean & (cols['mode'] == g) & (loc == c)
            if m.sum() < 100:
                continue
            row = {'n': int(m.sum()), 'fwd_bias_per_run_us': {n: _r(fb(n, m), 1) for n in names},
                   'l1_per_run_us': {n: _r(ml1(n, m), 2) for n in names}}
            for arm in ('C', 'C-DIV'):
                an = arm_names(arm, names)
                row['fwd_shift_%s_us' % arm] = _r(np.mean([fb(x, m) for x in an]) - np.mean([fb(x, m) for x in arm_names('A')]), 1)
                row['L1_%s_minus_A_us' % arm] = _r(np.mean([ml1(x, m) for x in an]) - np.mean([ml1(x, m) for x in arm_names('A')]), 2)
            row['A_seed_gap_fwd_us'] = _r(abs(fb('A_s0', m) - fb('A_s1', m)), 1)
            cells['g%d_%s' % (g, cname)] = row
    hs = cells.get('g5_HS')
    hab['low_speed_straight_HS'] = None if hs is None else {
        'n': hs['n'], 'fwd_bias_per_run_us': hs['fwd_bias_per_run_us'],
        'A_mean_us': _r(np.mean([hs['fwd_bias_per_run_us'][x] for x in arm_names('A')]), 1),
        'C_mean_us': _r(np.mean([hs['fwd_bias_per_run_us'][x] for x in arm_names('C')]), 1),
        'CDIV_mean_us': _r(np.mean([hs['fwd_bias_per_run_us'][x] for x in cdiv]), 1),
        'CDIV_shift_us': hs['fwd_shift_C-DIV_us'], 'C_shift_us': hs['fwd_shift_C_us'],
        'CDIV_moved': bool(abs(hs['fwd_shift_C-DIV_us']) >= qe.SPEED_HABIT_US),
        'CDIV_seeds_beyond_A_by_10us': [bool(abs(hs['fwd_bias_per_run_us'][x] - np.mean(
            [hs['fwd_bias_per_run_us'][a] for a in arm_names('A')])) >= qe.SPEED_HABIT_US) for x in cdiv],
        'L1_CDIV_minus_A_us': hs['L1_C-DIV_minus_A_us'], 'L1_C_minus_A_us': hs['L1_C_minus_A_us'],
        'contribution_to_overall_us': {
            'C-DIV': _r(hs['n'] * hs['L1_C-DIV_minus_A_us'] / float(clean.sum()), 3),
            'C': _r(hs['n'] * hs['L1_C_minus_A_us'] / float(clean.sum()), 3),
            'note': 'この直線の L1 の差 × フレーム数 / 4,670 = 全体の差(best)のうちこの直線から来た分'},
        'note': '負 = 人より強く前進。10/13 の確認の値: A −1.0 / +0.8、C −11.3 / −22.9 µs(判定には使わない参考)'}
    hab['cells'] = cells
    hab['max_abs_fwd_shift_cell_us'] = {arm: _r(max(abs(v['fwd_shift_%s_us' % arm]) for v in cells.values()), 1)
                                        for arm in ('C', 'C-DIV')}
    res['speed_habit'] = pz._round_tree(hab)
    # ---- C が悪かった 2 シーン(10/13)
    ps = {}
    for s in qe.C_BAD_SCENES:
        m = clean & (scene == s)
        ps[s] = {'n': int(m.sum()), 'L_per_run_us': {n: _r(ml1(n, m), 2) for n in names},
                 'C_minus_A_us': _r(np.mean([ml1(x, m) for x in arm_names('C')]) - np.mean([ml1(x, m) for x in arm_names('A')]), 2),
                 'CDIV_minus_A_us': _r(np.mean([ml1(x, m) for x in cdiv]) - np.mean([ml1(x, m) for x in arm_names('A')]), 2),
                 'CDIV_minus_C_us': _r(np.mean([ml1(x, m) for x in cdiv]) - np.mean([ml1(x, m) for x in arm_names('C')]), 2)}
    per_scene_diff = {}
    for s in held['human_clean_scenes']:
        m = clean & (scene == s)
        per_scene_diff[s] = {'n': int(m.sum()), 'genre': int(np.bincount(cols['mode'][m]).argmax()),
                             'A_L1_us': _r(np.mean([ml1(x, m) for x in arm_names('A')]), 2),
                             'C_minus_A_us': _r(np.mean([ml1(x, m) for x in arm_names('C')]) - np.mean([ml1(x, m) for x in arm_names('A')]), 2),
                             'CDIV_minus_A_us': _r(np.mean([ml1(x, m) for x in cdiv]) - np.mean([ml1(x, m) for x in arm_names('A')]), 2)}
    two = np.isin(scene, list(qe.C_BAD_SCENES)) & clean
    rest = clean & ~np.isin(scene, list(qe.C_BAD_SCENES))
    contrib = {}
    for arm in ('C', 'C-DIV'):
        an = arm_names(arm, names)
        d_all = np.mean([l1['best'][x] for x in an], axis=0) - np.mean([l1['best'][x] for x in arm_names('A')], axis=0)
        contrib[arm] = {'two_scenes_contribution_us': _r(float(d_all[two].sum() / clean.sum()), 3),
                        'rest_9_scenes_diff_us': _r(float(d_all[rest].mean()), 3),
                        'overall_diff_us': _r(float(d_all[clean].mean()), 3)}
    res['per_scene_two_C_bad'] = {'frames': int(two.sum()), 'per_arm': contrib,
                                  'note': '10/13 で C が悪かった 2 シーンが全体の差(best、腕 − A)に足した分と、残り 9 シーンだけの差'}
    res['per_scene'] = {'C_bad_scenes_1013': ps, 'all': per_scene_diff,
                        'scenes_CDIV_better_than_A': int(sum(v['CDIV_minus_A_us'] < 0 for v in per_scene_diff.values())),
                        'scenes_C_better_than_A': int(sum(v['C_minus_A_us'] < 0 for v in per_scene_diff.values())),
                        'scenes': len(per_scene_diff)}
    # ---- R1・DE(参考)
    cmp_list = [('C-DIV − A', cdiv, arm_names('A')), ('C − A', arm_names('C'), arm_names('A')),
                ('C-DIV − C', cdiv, arm_names('C'))]
    items = _r1_de(names, cmp_list, outs, tab, cols, f, held, prm, scene, m0)
    res['r1'], res['de'] = items['r1'], items['de']
    ref_a = refs['A_s0']
    diff_ref = [n for n in cdiv if tuple(refs[n]) != tuple(ref_a)]
    res['r1_de_center'] = {'CDIV_refs_us': {n: refs[n] for n in cdiv}, 'A_ref_us': ref_a, 'differs': diff_ref}
    if diff_ref:          # protocol: C-DIV の中心が A と違えば、A の中心でも出して並べる
        refs_a = {n: (ref_a if n in cdiv else refs[n]) for n in names}
        items2 = _r1_de(names, cmp_list[:1] + cmp_list[2:], outs_of(refs_a), tab, cols, f, held, prm, scene, m0)
        res['r1_de_center']['with_A_center'] = {'r1': items2['r1']['compare'], 'de': items2['de']['compare']}
    # ---- 崩れ
    fr = cols['frow']
    stm = np.zeros(len(fr), bool)
    stm[m0] = ev.straight_mask(f['loc_code'][fr[m0]], f['gz'][fr[m0]], f['erpm'][fr[m0]], f['near_collision'][fr[m0]])
    met = {n: cm.metrics(preds['best'][n][clean], cols['y'][clean], cols['cur'][clean], refs[n][0], refs[n][1]) for n in names}
    fwd = {n: float((pz.NEUTRAL_US - outs['best'][n][stm, 1]).mean()) for n in names}
    col = {'straight_frames': int(stm.sum()), 'per_run': {}, 'per_arm': {}}
    for n in names:
        col['per_run'][n] = {'straight_fwd_strength_us': _r(fwd[n], 1), 'fwd_bias_us': met[n]['fwd_bias_us'],
                             'steer_sign_acc': met[n]['steer_sign_acc']}

    def collapse_of(p_names, r_names):
        fp = np.mean([fwd[x] for x in p_names])
        fr_ = np.mean([fwd[x] for x in r_names])
        bp = np.mean([met[x]['fwd_bias_us'] for x in p_names])
        sp = np.mean([met[x]['steer_sign_acc'] for x in p_names])
        sr = np.mean([met[x]['steer_sign_acc'] for x in r_names])
        c, parts, drop = pz.judge_collapse(fp, fr_, bp, sp, sr)
        return {'collapse': c, 'parts': parts, 'straight_drop': _r(drop, 4), 'fwd_strength_us': _r(fp, 1),
                'ref_fwd_strength_us': _r(fr_, 1), 'fwd_bias_us': _r(bp, 1), 'sign_acc': _r(sp, 4), 'ref_sign_acc': _r(sr, 4)}
    for arm in ('C', 'C-DIV'):
        col['per_arm'][arm] = collapse_of(arm_names(arm, names), arm_names('A'))
    col['model_B_vs_A'] = collapse_of(['B'], arm_names('A'))
    res['collapse'] = pz._round_tree(col)
    # ---- ずれへの強さ
    rob = {'per_run': {}, 'per_arm': {}}
    dl1, per_all, rows0 = {}, {}, None
    for n in names:
        p = qe.pert_file(n)
        if not qe.npz_ok(p, ('rows',)):
            raise SystemExit('ずれの予測がありません: %s' % p)
        with np.load(p) as z:
            rows = z['rows']
            if rows0 is None:
                rows0 = rows
            elif not np.array_equal(rows, rows0):
                raise SystemExit('%s のずれの行が 10/12 と違います' % p)
            b = pz.l1_rows(preds['best'][n][rows], cols['y'][rows])
            per = {k: float(pz.l1_rows(z['pred_' + k].astype(np.float64), cols['y'][rows]).mean() - b.mean()) for k in pz.PERTS}
        per_all[n] = per
        dl1[n] = float(np.mean(list(per.values())))
        rob['per_run'][n] = {'frames': int(len(rows)), 'L_clean_us': _r(b.mean(), 3), 'dL1_us': _r(dl1[n], 3),
                             'per_pert_dL1_us': {k: _r(v, 3) for k, v in per.items()}}
    dl1_r = float(np.mean([dl1[x] for x in arm_names('A')]))
    for arm in ('C', 'C-DIV'):
        an = arm_names(arm, names)
        v = [dl1[x] for x in an]
        rob['per_arm'][arm] = {'robust_good': pz.judge_robust(v, dl1_r), 'dL1_mean_us': _r(np.mean(v), 3),
                               'dL1_seeds_us': [_r(x, 3) for x in v], 'ref_dL1_us': _r(dl1_r, 3),
                               'ratio_to_ref': _r(np.mean(v) / dl1_r, 3) if dl1_r else None,
                               'per_pert_mean_us': {k: _r(np.mean([per_all[x][k] for x in an]), 3) for k in pz.PERTS}}
    rob['per_arm']['A'] = {'dL1_mean_us': _r(dl1_r, 3), 'dL1_seeds_us': [_r(dl1[x], 3) for x in arm_names('A')],
                           'per_pert_mean_us': {k: _r(np.mean([per_all[x][k] for x in arm_names('A')]), 3) for k in pz.PERTS}}
    rob['ref_dL1_us'] = _r(dl1_r, 3)
    rob['rule'] = '「ずれに強い」= ΔL1(種の平均)≤ 0.8 × ΔL1_A かつ両方の種で ΔL1 ≤ ΔL1_A(参考。判定を変えない)'
    res['robustness'] = rob
    # ---- シミュレーターの L1(参考)
    res['sim_old_val12'] = _sim_eval(names, cfgs, [sc.SIM_DATASET], qe.sim_file,
                                     '前の教師のデータの val 12 シーン(10/13 と同じ。C は頭 8 が学習のデータと同じ方策・同じ線)')
    res['sim_modes_val15'] = _sim_eval(names, cfgs, list(qe.MODE_DATASETS), qe.sim5_file,
                                       '5 モードの見本の val 15 シーン(どの腕の学習にも入っていない = 保留)')
    # ---- ゲート(FP16 のエンジンは作らない = protocol eval_Q2.secondary.no_engine)
    gates = {'per_run': {}, 'note': ('protocol の no_engine のとおり、C-DIV の FP16 エンジンは作っていない(推奨を変える条件を満たしたときだけ'
                                     '次の作業で作る)。エンジン・速さの項目は未判定。判定できる項目(予測が有限・学習のピークメモリ ≤ 4096 MB・'
                                     'パラメーター数・出力が PWM の範囲)だけを見た。A・B・C は前の評価の値')}
    e13g = (e13.get('gates') or {}).get('per_run', {})
    e13l = (e13.get('latency') or {}).get('per_run', {})
    for n in names:
        v = table[n]
        if v['src'] is not None:
            g = e13g.get(n, {})
            gates['per_run'][n] = {'pass': g.get('pass'), 'source': '10/13 の evaluation.json(エンジンあり)',
                                   'run_numpy_median_ms': (e13l.get(n) or {}).get('run_numpy_median_ms'),
                                   'params_total_m': (e13l.get(n) or {}).get('params_total_m')}
            continue
        pk = float(resj[n].get('peak_mem_mb') or np.nan)
        lo = min(float(preds[ck][n].min()) for ck in CKPTS)
        hi = max(float(preds[ck][n].max()) for ck in CKPTS)
        parts = {'finite': finite[n], 'train_mem': bool(np.isfinite(pk) and pk <= pz.GATES['peak_mem_mb']),
                 'pwm_range': bool(lo >= 900 and hi <= 2100), 'engine': None, 'speed': None}
        gates['per_run'][n] = {'parts': parts, 'pass_checked_items': bool(all(x for x in parts.values() if x is not None)),
                               'peak_mem_mb': _r(pk, 1), 'pred_min_us': _r(lo, 1), 'pred_max_us': _r(hi, 1),
                               'params_total_m': _r(float(extras[n]['params_total']) / 1e6, 2),
                               'onnx_exists': (rp.RUNS_DIR / v['run'] / 'jetformer.onnx').exists(),
                               'engine_built': (rp.RUNS_DIR / v['run'] / 'jetformer_fp16.engine').exists(),
                               'onnx_check': rp._read_json(qe.REPORT_DIR / 'q2' / 'onnx_check' / (v['run'] + '.json'))}
    res['gates'] = gates
    # ---- 学習(学習の段の summary.json)
    summ = rp._read_json(qe.REPORT_DIR / 'train' / 'summary.json') or {}
    tr = {}
    for v in (summ.get('q2') or []):
        if isinstance(v, dict) and v.get('key'):
            tr[v['key']] = {k: v.get(k) for k in v if not isinstance(v.get(k), (list, dict))}
    for key, v in (e13.get('training') or {}).items():
        if (key.startswith('A/') or key.startswith('C/')) and key not in tr:
            tr['%s(10/13)' % key] = v
    for n in cdiv:
        r = resj[n]
        tr.setdefault('result_%s' % n, {k: r.get(k) for k in ('state', 'epochs_done', 'best_val_us', 'best_epoch', 'peak_mem_mb',
                                                               'duration_s', 'action_reference') if k in r})
    res['training'] = tr
    # ---- 前と同じ値になるか(A・B は 10/12、C は 10/13)
    rep = {'L_best': {}, 'L_last': {}}
    for n in ('A_s0', 'A_s1', 'B'):
        z = table[n]['zoo']
        rep['L_best'][n] = [_r(L['best'][n], 3), base['L_us']['best'][z]]
        rep['L_last'][n] = [_r(L['last'][n], 3), base['L_us']['last'][z]]
    for n in ('C_s0', 'C_s1'):
        rep['L_best'][n] = [_r(L['best'][n], 3), e13['primary']['L_us']['best'][n]]
        rep['L_last'][n] = [_r(L['last'][n], 3), e13['primary']['L_us']['last'][n]]
    rep['per_genre'] = {}
    for n in ('A_s0', 'A_s1', 'B', 'C_s0', 'C_s1'):
        rep['per_genre'][n] = {}
        for k in ('human_clean', 'human_clean_genre4', 'human_clean_genre5'):
            old = (base['per_genre'][table[n]['zoo']][k] if table[n]['src'] == '1012' else e13['accuracy']['metrics'][n][k])
            rep['per_genre'][n][k] = [acc['metrics'][n][k]['l1_us'], old['l1_us'], acc['metrics'][n][k]['fwd_bias_us'],
                                      old['fwd_bias_us']]
    rep['r1_window_best'] = {n: [items['r1']['window_mean_us']['best'][n], e13['r1']['window_mean_us']['best'][n]]
                             for n in ('A_s0', 'A_s1', 'B', 'C_s0', 'C_s1')}
    rep['de_window_best'] = {n: [items['de']['window_mean_us']['best'][n], e13['de']['window_mean_us']['best'][n]]
                             for n in ('A_s0', 'A_s1', 'B', 'C_s0', 'C_s1')}
    rep['straight_fwd'] = {n: [_r(fwd[n], 1), e13['collapse']['per_run'][n]['straight_fwd_strength_us']]
                           for n in ('A_s0', 'A_s1', 'B', 'C_s0', 'C_s1')}
    rep['robust_dL1'] = {n: [_r(dl1[n], 3), e13['robustness']['per_run'][n]['dL1_us']] for n in ('A_s0', 'A_s1', 'B', 'C_s0', 'C_s1')}
    rep['sim_old_h8_C'] = {n: [res['sim_old_val12']['per_run'][n]['h8']['l1_us'], e13['sim']['per_run'][n]['h8']['l1_us']]
                           for n in ('C_s0', 'C_s1')}
    rep['verdict_C_vs_A'] = [prim['C_vs_A']['final_class'], e13['verdict']['C_vs_A']]
    ok = all(abs(a - b) <= 0.0015 for a, b in list(rep['L_best'].values()) + list(rep['L_last'].values()))
    ok &= all(abs(v[0] - v[1]) <= 0.0015 and abs(v[2] - v[3]) <= 0.051 for d in rep['per_genre'].values() for v in d.values())
    ok &= all(abs(a - b) <= 0.051 for a, b in list(rep['r1_window_best'].values()) + list(rep['de_window_best'].values()))
    ok &= all(abs(a - b) <= 0.051 for a, b in rep['straight_fwd'].values())
    ok &= all(abs(a - b) <= 0.0015 for a, b in rep['robust_dL1'].values())
    ok &= all(abs(a - b) <= 0.0015 for a, b in rep['sim_old_h8_C'].values())
    ok &= rep['verdict_C_vs_A'][0] == rep['verdict_C_vs_A'][1]
    rep['all_match'] = bool(ok)
    rep['note'] = '[今日の値, 前の値]。A・B は 10/13 の protocol.json の baseline_A(10/12)、C は 10/13 の evaluation.json'
    res['reproduce'] = rep
    # ---- 判定(protocol eval_Q2.conclusion_rules)
    res['verdict'] = conclusion(res)
    return pz._round_tree(res)


CLASS_JA = {'better': '良い', 'equivalent': '同等', 'worse': '悪い'}


def conclusion(res):
    P = res['primary']
    va, vc = P['CDIV_vs_A'], P['CDIV_vs_C']
    prov = '(暫定・種 1 つ)' if res['single_seed'] else ''
    cls_a, cls_c = va['final_class'], vc['final_class']
    collapse = bool(res['collapse']['per_arm']['C-DIV']['collapse'])
    lines = []
    lines.append('1. C-DIV は主指標で A と比べて「%s%s」(best の差 %+.2f µs・閾値 %.3f µs、last の差 %+.2f µs・閾値 %.3f µs)。%s'
                 % (CLASS_JA[cls_a], prov, va['diff_best_us'], va['best']['threshold_us'], va['diff_last_us'],
                    va['last']['threshold_us'],
                    '実用上の差(≥ 10 µs)' if va['practical'] else '差は 10 µs 未満(実用上の差ではない)'))
    lines.append('2. C-DIV は同じ規則で C(前の教師のくり返しのデータを混ぜた)と比べて「%s%s」(best の差 %+.2f µs・閾値 %.3f µs、'
                 'last の差 %+.2f µs・閾値 %.3f µs)。シミュレーターのデータの量は C と同じなので、差があれば中身(いろいろな走り。'
                 'ただし先生が RC 教師から RL に替わったことも含む)の効果' % (CLASS_JA[cls_c], prov, vc['diff_best_us'],
                                                                vc['best']['threshold_us'], vc['diff_last_us'], vc['last']['threshold_us']))
    if cls_a == 'equivalent' and cls_c == 'equivalent':
        key = 'no_change'
        lines.append('3. どちらも「同等」: いろいろな走りのシミュレーターのデータを混ぜても、実機の保留データの精度は変わらなかった(A・C と同等)%s' % prov)
    elif cls_a == 'better':
        key = 'better_than_A'
        lines.append('3. A より「良い」%s' % prov)
    elif cls_a == 'worse':
        key = 'worse_than_A'
        lines.append('3. A より「悪い」%s' % prov)
    else:
        key = 'vs_C_%s' % cls_c
        lines.append('3. A とは「同等」、C とは「%s」%s' % (CLASS_JA[cls_c], prov))
    rob = res['robustness']['per_arm']['C-DIV']
    lines.append('4. 参考(判定を変えない): ずれへの強さ ΔL1 %.3f µs = A の %.2f 倍(「ずれに強い」%s。C は %.2f 倍)'
                 % (rob['dL1_mean_us'], rob['ratio_to_ref'], 'はい' if rob['robust_good'] else 'いいえ',
                    res['robustness']['per_arm']['C']['ratio_to_ref']))
    rec = bool(cls_a == 'better' and va['practical'] and not collapse and not res['single_seed'])
    if rec:
        lines.append('5. 「良い」かつ実用上の差かつ崩れなし → 実車の A/B/A の候補(今日は実車で試さない = 範囲の外)。推奨はまだモデル B のまま')
    else:
        lines.append('5. 推奨はモデル B のまま(「良い」かつ 10 µs 以上かつ崩れなし、の条件を満たさない)')
    lines.append('6. 結論は jetformer_v0・この学習の条件(調整なし)・このシミュレーターのデータに限る。机上の結論は実車の改善を示さない')
    return {'CDIV_vs_A': cls_a, 'CDIV_vs_C': cls_c, 'C_vs_A': P['C_vs_A']['final_class'], 'provisional': res['single_seed'],
            'practical_vs_A': bool(va['practical']), 'collapse': collapse, 'key': key, 'accuracy_went_up': cls_a == 'better',
            'real_car_candidate': rec, 'recommended': 'モデル B のまま', 'lines': lines}


def cmd_evaluate(args):
    t0 = time.time()
    res = evaluate_all()
    res['seconds'] = round(time.time() - t0, 1)
    rp._save_json(qe.eval_dir() / 'eval_real.json', res)
    rp._save_json(qe.REPORT_DIR / 'eval_real.json', res)
    print(json.dumps({'verdict': res['verdict'], 'L_best': res['primary']['L_us']['best'], 'L_last': res['primary']['L_us']['last'],
                      'reproduce': res['reproduce']['all_match'], 'protocol_unchanged': res['protocol']['unchanged']},
                     ensure_ascii=False, indent=1))


def cmd_figures(args):
    import q2_figures
    res = rp._read_json(qe.REPORT_DIR / 'eval_real.json')
    if not res:
        raise SystemExit('先に evaluate を実行してください')
    for p in q2_figures.make_all(res, qe.REPORT_DIR / 'fig'):
        print(p)


def cmd_status(args):
    for n, v in name_table().items():
        print(n, v['run'], 'ready' if (v['src'] or qe.run_ready(n)) else 'not-ready',
              {k: qe.npz_ok(fn(n)) for k, fn in (('pred', qe.pred_file), ('pert', qe.pert_file), ('sim', qe.sim_file),
                                                  ('sim5', qe.sim5_file))})
    print('crosscheck:', rp._read_json(qe.eval_dir() / 'crosscheck.json'))


if __name__ == '__main__':
    sys.exit('q2_eval.py evaluate|figures|status から使ってください')
