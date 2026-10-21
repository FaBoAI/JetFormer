#!/usr/bin/env python3
"""段 Desk(2026-10-21)の机上のベンチマークの評価(CPU・numpy だけ。torch を読まない)

    cd ~/jetformer
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261021_bench_prep/desk/desk_eval.py

入力(読むだけ):
  - 予測の表 tools/rl_evaluate.pred_table(key 3d339512…)・heldout.json・ai_mix_20261011 / intervention_val_20261011 の aligned
  - 予測(PyTorch FP32)
      B・MSF・TCP・ACT: data/zoo/policy_zoo_20261012/pred(10/12)、SIM: data/zoo/sim_cotrain_20261013/pred/CFT_s0(10/13)、
      ENS: data/zoo/real_accuracy_20261015/pred の v0-s2 と msf-s2 の平均(10/15。S4 で、まとめた best.pth が平均と 0.00009 µs 以内)、
      SLOW・WGT・AUG: desk/pred(この段の desk_gpu.py predict)
  - TensorRT: data/runs/<学習結果>/engine_check.json、desk/pred/sanity_<id>.npz・logs/sanity.json(desk_gpu.py sanity)
  - R1 の起点: tools/label_events.course_anchors(ログ全体の IMU。make_bench_tables.py の phase_of_log と同じ手順)
出力: /home/jetson/jetformer_reports/bench_prep_20261021/desk/desk_bench.json・desk_bench.md・fig/・pred/desk_inputs.npz
定義は desk_lib.py の先頭の注。
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
sys.path.insert(0, str(HERE))
import desk_lib as D  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
PRED = OUT / 'pred'
FIG = OUT / 'fig'
LOGS = HERE / 'logs'
RUNS = REPO / 'data' / 'runs'
ZOO = REPO / 'data' / 'zoo'
PLAN = Path('/home/jetson/jetformer_reports/bench_prep_20261021/plan')
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
REALACC_PROTOCOL = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/protocol.json')
ENGINE = 'jetformer_fp16.engine'
CKPTS = ('best', 'last')

SAVED = {
    'B': (ZOO / 'policy_zoo_20261012' / 'pred' / 'B.npz', ZOO / 'policy_zoo_20261012' / 'pred' / 'B_last.npz'),
    'MSF': (ZOO / 'policy_zoo_20261012' / 'pred' / 'msfusion_s0.npz', ZOO / 'policy_zoo_20261012' / 'pred' / 'msfusion_s0_last.npz'),
    'TCP': (ZOO / 'policy_zoo_20261012' / 'pred' / 'tcp_s0.npz', ZOO / 'policy_zoo_20261012' / 'pred' / 'tcp_s0_last.npz'),
    'ACT': (ZOO / 'policy_zoo_20261012' / 'pred' / 'act_s0.npz', ZOO / 'policy_zoo_20261012' / 'pred' / 'act_s0_last.npz'),
    'SIM': (ZOO / 'sim_cotrain_20261013' / 'pred' / 'CFT_s0.npz', ZOO / 'sim_cotrain_20261013' / 'pred' / 'CFT_s0_last.npz'),
}
ENS_SRC = {p: (ZOO / 'real_accuracy_20261015' / 'pred' / ('%s.npz' % p),
               ZOO / 'real_accuracy_20261015' / 'pred' / ('%s_last.npz' % p)) for p in D.ENS_PARTS}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def rnd(x, nd=3):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return round(v, nd) if math.isfinite(v) else None


# ---------------------------------------------------------------- 表と正解

def load_table():
    import rl_evaluate as ev
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key'] or tab['key'] != TABLE_KEY:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    n = len(tab['part'])
    cols = {'y': np.full((n, 10, 2), np.nan), 'fv': np.zeros(n, bool), 'impact': np.zeros(n, bool),
            'mode': np.zeros(n, int), 'erpm': np.full(n, np.nan), 't_us': np.zeros(n, np.int64), 'source': np.full(n, -1)}
    for part, ds in ((0, ev.DATASET), (1, ev.IV_DATASET)):
        for s in np.unique(tab['scene'][tab['part'] == part]):
            a = ev._aligned(ds, s)
            m = np.nonzero((tab['part'] == part) & (tab['scene'] == s))[0]
            i = tab['frame'][m]
            imp = D.impact_mask(a['timestamps_us'], a['imu'], 2.5)
            cols['y'][m] = a['future_actions'][i]
            cols['fv'][m] = a['future_valid'][i] & a['vehicle_valid'][i]
            cols['impact'][m] = imp[i]
            cols['mode'][m] = a['mode'][i]
            cols['erpm'][m] = a['vehicle'][i, 3]
            cols['t_us'][m] = a['timestamps_us'][i]
            if 'source' in a:
                cols['source'][m] = a['source'][i]
    part, scene = tab['part'], tab['scene']
    yfin = np.isfinite(cols['y']).all(axis=(1, 2))
    primary = np.isin(scene, held['human_clean_scenes']) & (part == 0) & cols['fv'] & ~cols['impact']
    interv = (part == 1) & cols['fv'] & yfin
    want = json.loads(REALACC_PROTOCOL.read_text(encoding='utf-8'))['secondary']['frames']
    if int(primary.sum()) != int(want['primary']) or int(interv.sum()) != int(want['S3_intervention_part1']):
        raise SystemExit('集合のフレーム数が 10/15 と違います: primary %d・interv %d' % (primary.sum(), interv.sum()))
    return tab, held, cols, primary, interv


def r1_anchors(logs):
    """make_bench_tables.phase_of_log と同じ手順で、ログ全体の IMU から R1 の起点 [s](label_events.course_anchors)"""
    import label_events as le
    out = {}
    for lg in logs:
        L = le.load_log(lg)
        I, V = L['IMU'], L['VEHICLE']
        o = np.argsort(I['timestamps_us'], kind='stable')
        it = I['timestamps_us'][o].astype(np.int64) / 1e6
        gz = I['gyro'][o][:, 2].astype(np.float64)
        ov = np.argsort(V['timestamps_us'], kind='stable')
        vt = V['timestamps_us'][ov].astype(np.int64) / 1e6
        erpm = V['erpm'][ov].astype(np.float64)
        still = (np.abs(np.interp(it, vt, erpm)) < 50) & (np.abs(gz) < 5)
        hd, gzu = le.heading_deg(it, gz, still)
        anchors, _, _ = le.course_anchors(it, hd, gzu, vt, erpm)
        out[lg] = np.asarray(anchors, float)
    return out


# ---------------------------------------------------------------- 予測

def _load(path, run=None):
    with np.load(path) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('%s: 表の key が違います' % path)
        if run is not None and str(z['run']) != run:
            raise SystemExit('%s: 学習結果 %s が %s と違います' % (path, z['run'], run))
        p = z['pred'].astype(np.float64)
    if p.shape != (10985, 10, 2) or not np.isfinite(p).all():
        raise SystemExit('%s: 形か値が違います' % path)
    return p


def load_preds():
    P, src = {}, {}
    for cid in D.IDS:
        P[cid], src[cid] = {}, {}
        for k, ck in enumerate(CKPTS):
            if cid in SAVED:
                path = SAVED[cid][k]
                P[cid][ck] = _load(path, D.RUN_OF[cid])
                src[cid][ck] = {'files': [str(path)], 'sha256': [sha256(path)]}
            elif cid == 'ENS':
                paths = [ENS_SRC[p][k] for p in D.ENS_PARTS]
                P[cid][ck] = np.mean([_load(p, r) for p, r in zip(paths, D.ENS_PARTS)], axis=0)
                src[cid][ck] = {'files': [str(p) for p in paths], 'sha256': [sha256(p) for p in paths],
                                'how': '2 本の予測の平均(ens_average と同じ。S4 の確かめ: まとめた best.pth と 0.00009 µs 以内)'}
            else:
                path = PRED / ('%s%s.npz' % (D.RUN_OF[cid], '' if ck == 'best' else '_last'))
                if path.exists():
                    P[cid][ck] = _load(path, D.RUN_OF[cid])
                    src[cid][ck] = {'files': [str(path)], 'sha256': [sha256(path)]}
                else:
                    P[cid][ck] = None
                    src[cid][ck] = {'files': [], 'missing': str(path)}
    return P, src


# ---------------------------------------------------------------- TensorRT

def engine_fields(cid):
    run = RUNS / D.RUN_OF[cid]
    eng = run / ENGINE
    out = {'engine_ok': None, 'trt_median_ms': None, 'trt_p99_ms': None, 'engine_check': None}
    if not eng.exists():
        out['engine_why'] = 'エンジンがありません'
        out['engine_ok'] = False
        return out
    st = eng.stat()
    p = run / 'engine_check.json'
    c = json.load(open(p, encoding='utf-8')) if p.exists() else None
    cur = D.engine_current(c, st.st_size, st.st_mtime_ns)
    out['engine_ok'] = D.engine_ok_of(c, cur)
    out['engine_sha256'] = sha256(eng)
    out['engine_check'] = {'file': str(p) if c else None, 'current': bool(cur),
                           **({k: c.get(k) for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'actions_p99_us',
                                                     'run_numpy_median_ms', 'run_numpy_p99_ms', 'checked_at', 'source')}
                              if c else {})}
    if cur:
        out['trt_median_ms'] = rnd(c.get('run_numpy_median_ms'), 3)
        out['trt_p99_ms'] = rnd(c.get('run_numpy_p99_ms'), 3)
    else:
        out['engine_why'] = 'engine_check.json が無いか、今のエンジンのものではありません'
    return out


def sanity_fields(cid, sanity):
    st = (sanity or {}).get('candidates', {}).get(cid)
    if not st:
        return {'sanity_ok': None, 'sanity': {'why': 'sanity の結果がありません'}}
    keep = ('frames', 'check_io_ok', 'nonfinite_frames', 'out_of_range_values', 'sanity_range_us', 'outside_drive_range_values',
            'values', 'steer_min_us', 'steer_max_us', 'thr_min_us', 'thr_max_us', 'post_kind', 'run_numpy_median_ms',
            'run_numpy_p99_ms', 'error', 'drivable', 'engine')
    s = {k: st.get(k) for k in keep if k in st}
    if st.get('values'):
        s['outside_drive_range_share'] = rnd(st.get('outside_drive_range_values', 0) / st['values'], 5)
    return {'sanity_ok': D.sanity_of(st), 'sanity': s}


def ai_brake(cid):
    """10/10 の AI 走行のフレーム(TensorRT・ジャンル 5)の低速のブレーキ側の数 → (数, 分母, ファイル)"""
    p = PRED / ('sanity_%s.npz' % cid)
    if not p.exists():
        return None, None, None, None, None
    with np.load(p) as z:
        pred = z['pred'].astype(np.float64)
        v = D.speed_mps(z['erpm'])
    ok = np.isfinite(pred[:, 0, 1])
    k, n = D.brake_lowspeed(pred[:, 0, 1], v, ok)
    ka, na = D.brake_lowspeed(pred[:, 0, 1], np.zeros_like(v), ok)      # 参考: 速さを問わない
    return k, n, str(p), ka, na


# ---------------------------------------------------------------- 本体

def main():
    t0 = time.time()
    tab, held, cols, primary, interv = load_table()
    y = np.nan_to_num(cols['y'])
    logs = sorted({s[:15] for s in held['human_clean_scenes']})
    log('R1 の起点: ログ %d 本' % len(logs))
    A = r1_anchors(logs)
    t_s = cols['t_us'] / 1e6
    pre = np.zeros(len(t_s), bool)
    for lg, anc in A.items():
        m = primary & (np.char.startswith(tab['scene'].astype(str), lg))
        pre[m] = D.pre_anchor_mask(t_s[m], anc)
    pre &= primary
    aid = np.full(len(t_s), -1, np.int64)
    k0 = 0
    for lg, anc in A.items():
        m = primary & np.char.startswith(tab['scene'].astype(str), lg)
        aid[m] = D.anchor_ids(t_s[m], anc, start=k0)
        k0 += len(anc)
    aid[~pre] = -1
    n_anchor_used = 0
    for lg, anc in A.items():
        m = primary & np.char.startswith(tab['scene'].astype(str), lg)
        for ta in anc:
            if (m & (t_s >= ta - D.R1_LEAD_S) & (t_s < ta)).any():
                n_anchor_used += 1
    log('R1 の手前 1 秒のフレーム %d(起点 %d)' % (int(pre.sum()), n_anchor_used))
    v_tab = D.speed_mps(cols['erpm'])
    low_hold = primary & np.isfinite(v_tab) & (v_tab < D.LOWSPEED_MPS)
    P, src = load_preds()
    pb_B = P['B']['best']
    sanity_path = LOGS / 'sanity.json'
    sanity = json.load(open(sanity_path, encoding='utf-8')) if sanity_path.exists() else None
    proto_sha = sha256(PLAN / 'protocol.json')
    want_sha = (PLAN / 'protocol.sha256').read_text().split()[0]
    rows = {}
    for cid in D.IDS:
        fam = D.family_by_run(D.RUN_OF[cid], D.FAMILY_OF[cid]) if cid != 'B' else None
        r = {'id': cid, 'label': D.LABEL_OF[cid], 'run': D.RUN_OF[cid], 'family': fam}
        pb, pl = P[cid]['best'], P[cid]['last']
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
            notes.append('best の予測がありません(%s)' % src[cid]['best'].get('missing'))
        if pl is not None:
            r['l1_last_us'] = rnd(D.l1_parts(pl, y, primary)['l1'])
            r.setdefault('ref', {})['l1_interv_last_us'] = rnd(D.l1_parts(pl, y, interv)['l1'])
        else:
            r['l1_last_us'] = None
            notes.append('last の予測がありません')
        r.update(engine_fields(cid))
        r.update(sanity_fields(cid, sanity))
        # A1: brake_lowspeed_frac(保留 FP32 + 10/10 の AI 走行 TensorRT)
        if pb is not None:
            kh, nh = D.brake_lowspeed(pb[:, 0, 1], v_tab, low_hold)
        else:
            kh, nh = None, None
        ka, na, fa, kaa, naa = ai_brake(cid)
        if pb is not None:
            kha, nha = D.brake_lowspeed(pb[:, 0, 1], np.zeros(len(v_tab)), primary)
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
        r['pred_sources'] = src[cid]
        r['notes'] = notes
        rows[cid] = r
    base = rows['B']
    for cid in D.IDS:
        r = rows[cid]
        if cid == 'B':
            r['gates_pass'] = None
            r['gate_fail_reasons'] = []
            r['gate_note'] = '基準。門に掛けない(protocol gates.baseline)。engine_ok・sanity_ok が false なら 10/23 は走らずにユーザーに知らせる'
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
    out = {
        'version': 1, 'bench': 'real_bench_20261023', 'stage': 'Desk(bench_prep_20261021)',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S+09:00'),
        'protocol_sha256': proto_sha, 'protocol_sha256_expected': want_sha, 'protocol_sha256_ok': proto_sha == want_sha,
        'amendments': 'protocol_amendments.md A1(label_r1 の門・l1_steer_best_us・l1_thr_best_us・brake_lowspeed_frac)',
        'definitions': {
            'sets': {'primary': '保留の人の走行 %d フレーム(pred_table part 0・human_clean_scenes・future_valid&vehicle_valid・2.5G の前後を除く)'
                                % int(primary.sum()),
                     'interv': '介入の集合 intervention_val_20261011 の採点するフレーム %d' % int(interv.sum()),
                     'r1_pre1s': 'primary のうち R1 の起点(label_events.course_anchors)の手前 1 秒 %d フレーム・起点 %d'
                                 % (int(pre.sum()), n_anchor_used),
                     'lowspeed_heldout': 'primary のうち速さ < 1.0 m/s の %d フレーム' % int(low_hold.sum())},
            'l1': '10 ステップ × 2 の |予測 − 人| の平均 [µs](PyTorch FP32。best・last の重み)',
            'r1_app_thr_diff_us': '[ta − 1.0, ta) の 予測スロットル − 人のスロットル(行 index 2 = 先読み 3)の平均。正 = 人より緩める',
            'steer_amp_ratio': 'primary の予測ステアの標準偏差 ÷ 人のステアの標準偏差(10 ステップまとめて・best)',
            'brake_lowspeed_frac': 'v < 1.0 m/s のフレームで、予測の 1 行目のスロットル > 1598 µs の割合(保留 FP32 + 10/10 の AI 走行 TensorRT)',
            'engine_ok': 'engine_check.json が今のエンジンのもので accuracy_ok・actions_max_us ≤ 2.5',
            'sanity_ok': '10/10 の AI 走行 11 本の AI のフレームを TensorRT(ジャンル 5・基準の設定)に入れ、NaN・Inf なし・'
                         '[10,2] がすべて 500〜2500 µs・check_io が通る',
            'trt': 'engine_check.json の run_numpy(走行と同じ呼び方)の中央値・99% 点 [ms]',
        },
        'counts': {'primary': int(primary.sum()), 'interv': int(interv.sum()), 'r1_pre1s_frames': int(pre.sum()),
                   'r1_anchors_used': n_anchor_used, 'r1_anchors_by_log': {k: int(len(v)) for k, v in A.items()},
                   'lowspeed_heldout': int(low_hold.sum()),
                   'ai_1010_frames': (sanity or {}).get('frames')},
        'check_realacc_B': None,
        'candidates': [rows[c] for c in D.IDS],
    }
    # 10/15 の B の値(protocol の baseline_values)と同じか
    bv = json.loads(REALACC_PROTOCOL.read_text(encoding='utf-8'))['secondary']['baseline_values']
    out['check_realacc_B'] = {
        'l1_best_us': [base['l1_best_us'], bv['best']['B']['primary']],
        'l1_last_us': [base['l1_last_us'], bv['last']['B']['primary']],
        'l1_interv_us': [base['l1_interv_us'], bv['best']['B']['S3_intervention_part1']],
        'pass': bool(abs(base['l1_best_us'] - bv['best']['B']['primary']) <= 0.0015
                     and abs(base['l1_last_us'] - bv['last']['B']['primary']) <= 0.0015
                     and abs(base['l1_interv_us'] - bv['best']['B']['S3_intervention_part1']) <= 0.0015)}
    # 参考: 同じ作り方で種だけ違う学習結果の幅(机上の差が種の違いより大きいかを読むため。門・選び方には使わない)
    seeds = {'jetformer_v0(B と同じ形・同じデータ)': [('A_s0', ZOO / 'policy_zoo_20261012' / 'pred' / 'jetformer_v0_s0.npz'),
                                              ('A_s1', ZOO / 'policy_zoo_20261012' / 'pred' / 'jetformer_v0_s1.npz'),
                                              ('v0-s2', ENS_SRC['realacc_20261015-v0-s2'][0]),
                                              ('v0-s3', ZOO / 'real_accuracy_20261015' / 'pred' / 'realacc_20261015-v0-s3.npz'),
                                              ('B', SAVED['B'][0])],
             'msfusion(MSF の形)': [('msf_s0', SAVED['MSF'][0]), ('msf_s1', ZOO / 'policy_zoo_20261012' / 'pred' / 'msfusion_s1.npz'),
                                  ('msf-s2', ENS_SRC['realacc_20261015-msf-s2'][0]),
                                  ('msf-s3', ZOO / 'real_accuracy_20261015' / 'pred' / 'realacc_20261015-msf-s3.npz')],
             'tcp': [('tcp_s0', SAVED['TCP'][0]), ('tcp_s1', ZOO / 'policy_zoo_20261012' / 'pred' / 'tcp_s1.npz')],
             'act': [('act_s0', SAVED['ACT'][0]), ('act_s1', ZOO / 'policy_zoo_20261012' / 'pred' / 'act_s1.npz')]}
    spread = {}
    for fam, items in seeds.items():
        vals = {}
        for nm, path in items:
            pp = _load(path)
            vals[nm] = {'l1_best_us': rnd(D.l1_parts(pp, y, primary)['l1']), 'l1_interv_us': rnd(D.l1_parts(pp, y, interv)['l1']),
                        'r1_app_thr_diff_us': rnd(D.thr_diff(pp, y, pre, D.LOOKAHEAD_ROW)),
                        'steer_amp_ratio': rnd(D.steer_amp_ratio(pp, y, primary), 4), 'file': str(path)}
        rng = {k: rnd(max(v[k] for v in vals.values()) - min(v[k] for v in vals.values()))
               for k in ('l1_best_us', 'l1_interv_us', 'r1_app_thr_diff_us', 'steer_amp_ratio')}
        spread[fam] = {'runs': vals, 'range': rng}
    out['seed_spread_ref'] = spread
    # あとで別のコードで数え直すための入力(予測・正解・集合)
    bundle = {'key': np.array(TABLE_KEY), 'part': tab['part'], 'scene': tab['scene'].astype(str), 'frame': tab['frame'],
              'y': cols['y'].astype(np.float32), 'erpm': cols['erpm'].astype(np.float32), 't_us': cols['t_us'],
              'primary': primary, 'interv': interv, 'r1_pre1s': pre, 'r1_anchor_id': aid, 'ids': np.array(D.IDS)}
    for cid in D.IDS:
        for ck in CKPTS:
            if P[cid][ck] is not None:
                bundle['pred_%s_%s' % (cid, ck)] = P[cid][ck].astype(np.float32)
    PRED.mkdir(parents=True, exist_ok=True)
    tmp = PRED / 'desk_inputs.tmp.npz'
    np.savez_compressed(tmp, **bundle)
    os.replace(tmp, PRED / 'desk_inputs.npz')
    out['inputs_bundle'] = {'file': str(PRED / 'desk_inputs.npz'), 'sha256': sha256(PRED / 'desk_inputs.npz')}
    out['seconds'] = rnd(time.time() - t0, 1)
    write_json(OUT / 'desk_bench.json', out)
    log('desk_bench.json を書きました(%.0f 秒)' % (time.time() - t0))
    for r in out['candidates']:
        log(r['id'], {k: r.get(k) for k in ('l1_best_us', 'l1_last_us', 'l1_interv_us', 'r1_app_thr_diff_us', 'steer_amp_ratio',
                                              'l1_steer_best_us', 'brake_lowspeed_frac', 'trt_p99_ms', 'engine_ok', 'sanity_ok',
                                              'gates_pass', 'gate_fail_reasons')})
    log('B の 10/15 との一致', out['check_realacc_B'])


if __name__ == '__main__':
    main()
