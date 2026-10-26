#!/usr/bin/env python3
"""段 Desk(2026-10-23・A10)の机上のベンチの評価(CPU・numpy だけ。torch を読まない)

    cd ~/jetformer
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261022_prep/desk/desk_eval2.py

10/21 の desk_eval.py(experiments/20261021_bench_prep/desk/)の関数(load_table・r1_anchors・_load・engine_fields・sanity_fields)
を読み、1 本ずつの欄の作り方(desk_eval.main の候補ごとの行)を build_row に写した。欄と定義は desk_bench.json と同じ。
足すもの(A10): nonzone_thr_diff_us(desk_lib2.py の注)と、C1 の決め方の結果(a10_c1。計算するだけ。bench_candidates.json は変えない)。
対象: B・SLOW(10/21 の予測を読むだけ)と新しい 3 本(desk_gpu2.py の予測・sanity)。
確かめ: B・SLOW の 10/21 の欄を同じ値で作れるか(check_vs_20261021)。
出力: /home/jetson/jetformer_reports/prep_20261022/desk/desk_bench_20261022.json・pred/desk_inputs_20261022.npz
(10/21 の desk_bench.json・desk/pred は変えない)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_lib2 as D2  # noqa: E402

D = D2.register()
import desk_eval as E  # noqa: E402  (10/21 の台本)

OUT = Path('/home/jetson/jetformer_reports/prep_20261022/desk')
PRED = OUT / 'pred'
LOGS = HERE / 'logs'
OLD_OUT = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
OLD_PRED = OLD_OUT / 'pred'
OLD_LOGS = D2.OLD_DESK / 'logs'
ZONES = PRED / 'zones_heldout.npz'
rnd = E.rnd


def pred_paths(cid):
    """→ (best, last) のファイル。B は 10/12 の保存、SLOW は 10/21 の desk/pred、新しい 3 本は この段の pred"""
    if cid == 'B':
        return E.SAVED['B']
    run = D.RUN_OF[cid]
    base = OLD_PRED if cid == 'SLOW' else PRED
    return base / ('%s.npz' % run), base / ('%s_last.npz' % run)


def load_preds(ids):
    P, src = {}, {}
    for cid in ids:
        P[cid], src[cid] = {}, {}
        for path, ck in zip(pred_paths(cid), E.CKPTS):
            if path.exists():
                P[cid][ck] = E._load(path, D.RUN_OF[cid])
                src[cid][ck] = {'files': [str(path)], 'sha256': [E.sha256(path)]}
            else:
                P[cid][ck] = None
                src[cid][ck] = {'files': [], 'missing': str(path)}
    return P, src


def sanity_of(cid):
    """→ (sanity.json の dict, sanity_<id>.npz のあるフォルダ)。B・SLOW は 10/21、新しい 3 本はこの段"""
    p, d = (OLD_LOGS / 'sanity.json', OLD_PRED) if cid in ('B', 'SLOW') else (LOGS / 'sanity.json', PRED)
    return (json.load(open(p, encoding='utf-8')) if p.exists() else None), d


def ai_brake(cid, pred_dir):
    """desk_eval.ai_brake と同じ(フォルダだけ替える)"""
    p = pred_dir / ('sanity_%s.npz' % cid)
    if not p.exists():
        return None, None, None, None, None
    with np.load(p) as z:
        pred = z['pred'].astype(np.float64)
        v = D.speed_mps(z['erpm'])
    ok = np.isfinite(pred[:, 0, 1])
    k, n = D.brake_lowspeed(pred[:, 0, 1], v, ok)
    ka, na = D.brake_lowspeed(pred[:, 0, 1], np.zeros_like(v), ok)
    return k, n, str(p), ka, na


def build_row(cid, pb, pl, src_c, ctx):
    """desk_eval.main の候補ごとの行と同じ(門は後で)"""
    y, primary, interv, pre, aid, v_tab, low_hold, pb_B = (ctx[k] for k in ('y', 'primary', 'interv', 'pre', 'aid', 'v_tab',
                                                                           'low_hold', 'pb_B'))
    fam = D.family_by_run(D.RUN_OF[cid], D.FAMILY_OF[cid]) if cid != 'B' else None
    r = {'id': cid, 'label': D.LABEL_OF[cid], 'run': D.RUN_OF[cid], 'family': fam}
    notes = []
    if pb is not None:
        lp = D.l1_parts(pb, y, primary)
        r['l1_best_us'] = rnd(lp['l1'])
        r['l1_steer_best_us'] = rnd(lp['l1_steer'])
        r['l1_thr_best_us'] = rnd(lp['l1_thr'])
        r['l1_interv_us'] = rnd(D.l1_parts(pb, y, interv)['l1'])
        r['r1_app_thr_diff_us'] = rnd(D.thr_diff(pb, y, pre, D.LOOKAHEAD_ROW))
        r['steer_amp_ratio'] = rnd(D.steer_amp_ratio(pb, y, primary), 4)
        r['ref'] = {'r1_app_thr_diff_allrows_us': rnd(D.thr_diff(pb, y, pre, None)),
                    'r1_app_pred_thr_row2_us': rnd(float(pb[pre][:, 2, 1].mean())) if pre.any() else None,
                    'r1_app_human_thr_row2_us': rnd(float(y[pre][:, 2, 1].mean())) if pre.any() else None,
                    'steer_amp_ratio_row2': rnd(D.steer_amp_ratio(pb, y, primary, D.LOOKAHEAD_ROW), 4),
                    'l1_interv_steer_us': rnd(D.l1_parts(pb, y, interv)['l1_steer']),
                    'l1_interv_thr_us': rnd(D.l1_parts(pb, y, interv)['l1_thr'])}
        mu, lo, hi, kk = D.cluster_boot_ci(pb[:, 2, 1] - y[:, 2, 1], aid, seed=0)
        r['ref']['r1_app_thr_diff_ci90_us'] = [rnd(lo), rnd(hi)]
        r['ref']['r1_app_anchors'] = kk
        if cid != 'B':
            mu, lo, hi, kk = D.cluster_boot_ci(pb[:, 2, 1] - pb_B[:, 2, 1], aid, seed=0)
            r['ref']['r1_app_thr_vs_B_us'] = rnd(mu)
            r['ref']['r1_app_thr_vs_B_ci90_us'] = [rnd(lo), rnd(hi)]
    else:
        for k in ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_interv_us', 'r1_app_thr_diff_us', 'steer_amp_ratio'):
            r[k] = None
        notes.append('best の予測がありません(%s)' % src_c['best'].get('missing'))
    if pl is not None:
        r['l1_last_us'] = rnd(D.l1_parts(pl, y, primary)['l1'])
        r.setdefault('ref', {})['l1_interv_last_us'] = rnd(D.l1_parts(pl, y, interv)['l1'])
    else:
        r['l1_last_us'] = None
        notes.append('last の予測がありません')
    r.update(E.engine_fields(cid))
    sanity, sdir = sanity_of(cid)
    r.update(E.sanity_fields(cid, sanity))
    if pb is not None:
        kh, nh = D.brake_lowspeed(pb[:, 0, 1], v_tab, low_hold)
        kha, nha = D.brake_lowspeed(pb[:, 0, 1], np.zeros(len(v_tab)), primary)
    else:
        kh = nh = kha = nha = None
    ka, na, fa, kaa, naa = ai_brake(cid, sdir)
    if kh is not None and ka is not None:
        r['brake_lowspeed_frac'] = rnd(D.frac(kh + ka, nh + na), 5)
    else:
        r['brake_lowspeed_frac'] = None
        notes.append('brake_lowspeed_frac: 保留か AI 走行の予測がありません')
    r['brake_lowspeed'] = {'heldout': [kh, nh], 'ai_1010': [ka, na], 'ai_file': fa,
                           'heldout_frac': rnd(D.frac(kh, nh), 5) if kh is not None else None,
                           'ai_frac': rnd(D.frac(ka, na), 5) if ka is not None else None,
                           'ref_any_speed': {'heldout': [kha, nha] if pb is not None else None,
                                             'ai_1010': [kaa, naa] if kaa is not None else None}}
    r['pred_sources'] = src_c
    r['notes'] = notes
    return r


def add_nonzone(r, pb, ctx):
    """A10 の nonzone_thr_diff_us と参考の値"""
    if pb is None:
        r['nonzone_thr_diff_us'] = None
        r['notes'].append('nonzone_thr_diff_us: best の予測がありません')
        return
    y, primary, zone, nzm, scid, pb_B = (ctx[k] for k in ('y', 'primary', 'zone', 'nonzone', 'scene_id', 'pb_B'))
    r['nonzone_thr_diff_us'] = rnd(D2.nonzone_thr_diff(pb, y, primary, zone))
    ref = {'frames': int(nzm.sum()),
           'pred_thr_row0_us': rnd(float(pb[nzm][:, 0, 1].mean())),
           'human_thr_row0_us': rnd(float(y[nzm][:, 0, 1].mean())),
           'row2_us': rnd(D2.nonzone_thr_diff(pb, y, primary, zone, row=2)),
           'allrows_us': rnd(D2.nonzone_thr_diff(pb, y, primary, zone, row=None)),
           'by_zone_row0_us': {k: {'n': v['n'], 'thr_diff_us': rnd(v['thr_diff_us'])}
                               for k, v in D2.zone_thr_diff(pb, y, primary, zone).items()},
           'primary_all_row0_us': rnd(D.thr_diff(pb, y, primary, 0))}
    cl = np.where(nzm, scid, -1)
    mu, lo, hi, kk = D.cluster_boot_ci(pb[:, 0, 1] - y[:, 0, 1], cl, seed=0)
    ref['ci90_scene_boot_us'] = [rnd(lo), rnd(hi)]
    ref['scenes'] = kk
    if r['id'] != 'B':
        mu, lo, hi, kk = D.cluster_boot_ci(pb[:, 0, 1] - pb_B[:, 0, 1], cl, seed=0)
        ref['vs_B_us'] = rnd(mu)
        ref['vs_B_ci90_scene_boot_us'] = [rnd(lo), rnd(hi)]
    r['ref_nonzone'] = ref


CHECK_KEYS = ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_interv_us', 'r1_app_thr_diff_us', 'steer_amp_ratio',
              'l1_last_us', 'engine_ok', 'trt_median_ms', 'trt_p99_ms', 'sanity_ok', 'brake_lowspeed_frac', 'gates_pass',
              'gate_fail_reasons')


def check_vs_old(rows):
    """B・SLOW の欄が 10/21 の desk_bench.json と同じか(ref の値も)"""
    old = {r['id']: r for r in json.load(open(OLD_OUT / 'desk_bench.json', encoding='utf-8'))['candidates']}
    out = {'file': str(OLD_OUT / 'desk_bench.json'), 'diffs': [], 'compared': 0}
    for cid in ('B', 'SLOW'):
        a, b = rows[cid], old[cid]
        keys = list(CHECK_KEYS) + ['ref.' + k for k in (b.get('ref') or {})] + ['brake_lowspeed']
        for k in keys:
            if k.startswith('ref.'):
                va, vb = (a.get('ref') or {}).get(k[4:]), (b.get('ref') or {}).get(k[4:])
            else:
                va, vb = a.get(k), b.get(k)
            out['compared'] += 1
            if va != vb:
                out['diffs'].append({'id': cid, 'key': k, 'now': va, '20261021': vb})
    out['pass'] = not out['diffs']
    return out


def main():
    t0 = time.time()
    tab, held, cols, primary, interv = E.load_table()
    y = np.nan_to_num(cols['y'])
    logs = sorted({s[:15] for s in held['human_clean_scenes']})
    A = E.r1_anchors(logs)
    t_s = cols['t_us'] / 1e6
    scn = tab['scene'].astype(str)
    pre = np.zeros(len(t_s), bool)
    for lg, anc in A.items():
        m = primary & np.char.startswith(scn, lg)
        pre[m] = D.pre_anchor_mask(t_s[m], anc)
    pre &= primary
    aid = np.full(len(t_s), -1, np.int64)
    k0 = 0
    for lg, anc in A.items():
        m = primary & np.char.startswith(scn, lg)
        aid[m] = D.anchor_ids(t_s[m], anc, start=k0)
        k0 += len(anc)
    aid[~pre] = -1
    n_anchor_used = sum(1 for lg, anc in A.items() for ta in anc
                        if (primary & np.char.startswith(scn, lg) & (t_s >= ta - D.R1_LEAD_S) & (t_s < ta)).any())
    v_tab = D.speed_mps(cols['erpm'])
    low_hold = primary & np.isfinite(v_tab) & (v_tab < D.LOWSPEED_MPS)
    # 区間(zones2.py)を表の行に当てる
    zone = np.full(len(t_s), '', dtype='<U8')
    with np.load(ZONES) as z:
        zkey = {(s, int(f)): (zz, int(t)) for s, f, zz, t in zip(z['scene'].astype(str), z['frame'], z['zone'].astype(str), z['t_us'])}
    t_mismatch = 0
    for i in np.nonzero((tab['part'] == 0) & np.isin(scn, held['human_clean_scenes']))[0]:
        zz, tt = zkey[(scn[i], int(tab['frame'][i]))]
        zone[i] = zz
        t_mismatch += int(tt != int(cols['t_us'][i]))
    if t_mismatch or (zone[primary] == '').any():
        raise SystemExit('区間を表の行に当てられません(時刻の違い %d)' % t_mismatch)
    nz = D2.nonzone_mask(primary, zone)
    scene_names = sorted(held['human_clean_scenes'])
    scene_id = np.array([scene_names.index(s) if s in scene_names else -1 for s in scn])
    ids = list(D2.IDS2)
    P, src = load_preds(ids)
    ctx = {'y': y, 'primary': primary, 'interv': interv, 'pre': pre, 'aid': aid, 'v_tab': v_tab, 'low_hold': low_hold,
           'pb_B': P['B']['best'], 'zone': zone, 'nonzone': nz, 'scene_id': scene_id}
    rows = {}
    for cid in ids:
        rows[cid] = build_row(cid, P[cid]['best'], P[cid]['last'], src[cid], ctx)
        add_nonzone(rows[cid], P[cid]['best'], ctx)
    base = rows['B']
    for cid in ids:
        r = rows[cid]
        if cid == 'B':
            r['gates_pass'] = None
            r['gate_fail_reasons'] = []
            r['gate_note'] = '基準。門に掛けない(protocol gates.baseline)'
            ok, why = D.gates(r, base, None)
            r['gates_if_applied'] = {'pass': ok, 'fail': why}
            r['baseline_ok'] = bool(r.get('engine_ok') is True and r.get('sanity_ok') is True)
            continue
        ok, why = D.gates(r, base, r['family'])
        r['gates_pass'] = bool(ok)
        r['gate_fail_reasons'] = why
        r['gate_rule'] = ('protocol gates + A1(5 番目 = l1_steer_best_us の差 < 10.0、brake_lowspeed_frac ≤ 0.01)'
                          if r['family'] == 'label_r1' else 'protocol gates(5 つ)')
        if r['family'] == 'label_r1':
            ok0, why0 = D.gates(r, base, None)
            r['gates_without_A1'] = {'pass': ok0, 'fail': why0}
    chk = check_vs_old(rows)
    c1 = D2.c1_rule_a10(rows)
    c1.update({'rule': 'protocol_amendments.md A10 の「C1 の枠の決め方」1〜3(A1 の門・r1_app_thr_diff_us ≥ +20.0・'
                       'nonzone_thr_diff_us を小数第 1 位で小さい順 → r1_app 大きい順 → id)',
               'pick_run': D.RUN_OF.get(c1['pick']),
               'applied': False,
               'note': '計算しただけ。data/bench_candidates.json は変えていない(C1 は今も SLOW = bench_20261021-slow-s0)'})
    proto_sha = E.sha256(E.PLAN / 'protocol.json')
    want_sha = (E.PLAN / 'protocol.sha256').read_text().split()[0]
    amend = E.PLAN / 'protocol_amendments.md'
    zrep = json.load(open(LOGS / 'zones2.json', encoding='utf-8'))
    out = {
        'version': 1, 'bench': 'real_bench_20261023', 'stage': 'Desk(prep_20261022・A10)',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'),
        'protocol_sha256': proto_sha, 'protocol_sha256_expected': want_sha, 'protocol_sha256_ok': proto_sha == want_sha,
        'amendments': 'protocol_amendments.md A1(label_r1 の門)・A10(候補の追加・nonzone_thr_diff_us・C1 の決め方)',
        'amendments_sha256': E.sha256(amend),
        'definitions': {
            'same_as': str(OLD_OUT / 'desk_bench.json') + ' の definitions と同じ(欄・集合・門)',
            'nonzone_thr_diff_us': ('A10: primary のうち v1.1 の区間の判定が none か unknown の %d フレームで、予測の 1 行目(index 0)の'
                                    'スロットル − 人のスロットル(正解の index 0)の平均 [µs]。正 = 区間の外でも人より緩める(PWM は'
                                    '前進が小さい値)。区間は make_bench_tables.py と同じ(zones2.py)' % int(nz.sum())),
            'ref_nonzone': '参考(門・選び方に使わない): row2 = 行 index 2、allrows = 10 行の平均、by_zone = 区間ごと、'
                           'ci90_scene_boot = シーン(11 個)を復元抽出した 90% 区間、vs_B = 同じフレームの B との差',
        },
        'zones': {'file': str(ZONES), 'sha256': E.sha256(ZONES), 'report': str(LOGS / 'zones2.json'),
                  'check_train_counts_same_as_card': zrep.get('check_train_counts_same_as_card'),
                  'primary_by_zone': {k: int((primary & (zone == k)).sum()) for k in ('none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE')},
                  'nonzone_frames': int(nz.sum()),
                  'r1_pre1s_by_zone': {k: int((pre & (zone == k)).sum()) for k in ('none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE')}},
        'counts': {'primary': int(primary.sum()), 'interv': int(interv.sum()), 'r1_pre1s_frames': int(pre.sum()),
                   'r1_anchors_used': n_anchor_used, 'lowspeed_heldout': int(low_hold.sum()), 'nonzone': int(nz.sum())},
        'check_vs_20261021': chk,
        'a10_c1': c1,
        'candidates': [rows[c] for c in ids],
    }
    bundle = {'key': np.array(E.TABLE_KEY), 'part': tab['part'], 'scene': scn, 'frame': tab['frame'],
              'y': cols['y'].astype(np.float32), 'erpm': cols['erpm'].astype(np.float32), 't_us': cols['t_us'],
              'primary': primary, 'interv': interv, 'r1_pre1s': pre, 'r1_anchor_id': aid, 'zone': zone, 'nonzone': nz,
              'ids': np.array(ids)}
    for cid in ids:
        for ck in E.CKPTS:
            if P[cid][ck] is not None:
                bundle['pred_%s_%s' % (cid, ck)] = P[cid][ck].astype(np.float32)
    PRED.mkdir(parents=True, exist_ok=True)
    tmp = PRED / 'desk_inputs_20261022.tmp.npz'
    np.savez_compressed(tmp, **bundle)
    os.replace(tmp, PRED / 'desk_inputs_20261022.npz')
    out['inputs_bundle'] = {'file': str(PRED / 'desk_inputs_20261022.npz'), 'sha256': E.sha256(PRED / 'desk_inputs_20261022.npz')}
    out['seconds'] = rnd(time.time() - t0, 1)
    E.write_json(OUT / 'desk_bench_20261022.json', out)
    print(time.strftime('%H:%M:%S'), 'desk_bench_20261022.json を書きました(%.0f 秒)' % (time.time() - t0), flush=True)
    for r in out['candidates']:
        print(r['id'], {k: r.get(k) for k in ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'r1_app_thr_diff_us',
                                              'nonzone_thr_diff_us', 'brake_lowspeed_frac', 'steer_amp_ratio', 'trt_p99_ms',
                                              'engine_ok', 'sanity_ok', 'gates_pass', 'gate_fail_reasons')}, flush=True)
    print('check_vs_20261021', chk['pass'], chk['diffs'][:5])
    print('a10_c1', json.dumps(c1, ensure_ascii=False))


if __name__ == '__main__':
    main()
