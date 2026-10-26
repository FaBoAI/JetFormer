#!/usr/bin/env python3
"""verify_c3(CPU・numpy): C3 の欄 robust_dl1_us・G-b・G-d・止まった場面・低速のブレーキ を自分のコードで数え直す。

desk_v2.py・eval_v2.py・tools/policy_zoo.py・inventory.py・verify_d3 の台本は import しない。集合と式は protocol_v2.json の
sets・fields_added・gates_v2・gates_bfast・sign_rules、deviations.md 7、10/21 の desk_bench の定義(brake)の文から書いた。
jetformer の本体からは tof_guard.ToFGuard(protocol が名指しする道具)と train_data.top_ramp(照らすだけ)を使う。

  1. 集合を生のラベル(data/labels)と aligned(dataset_v2_20261025・clean_driving_20261011・ai_mix_20261011)から作る
     - 止まった場面: 学習の回 46 本・AI の側(ラベル ai・dry run でない・aligned source == 1・vehicle_valid)・
       |ラベルの eRPM| < 400・ラベルの near_collision でない・ToFGuard.features(tof, 0, 0, 中心)['x_free'] > 1.0 m
     - gb_main: 10/23 公式の B の 3 本(No.8・11・15)の AI の側・near_collision でない・間引きなし。G-b はラベルの
       loc_detail が DE(4)・SW(5)・HS(6)
     - 低速のブレーキ: H1(primary)の v < 1.0 m/s(v = −eRPM × 0.5e-3、aligned の eRPM)+ 10/10 の AI 走行
       (ai_mix_20261011 の source == 1 & vehicle_valid、21,759)の v < 1.0 m/s
     - H1: desk_inputs_20261022.npz の primary。正解は clean_driving_20261011 の aligned の future_actions から読み直して照らす
  2. C3 の集合(sets_v2.npz)・TensorRT の保存(trt/<仕事>/<名前>.npz)・バンドルと、フレームが同じかを照らす
  3. 保存した予測(desk/pred の tab・shift、eval/pred、trt)に自分の式を当て、desk_v2.json の値と照らす(全部の行)

書くのは c3/verify/ の下(recount_cpu.json・sets_mine.npz)だけ。data/・protocol・desk/ は読むだけ。
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
DESK = REP / 'desk'
HERE = REPO / 'experiments' / '20261025_train_v2' / 'c3' / 'verify'
DATA = REPO / 'data'
LAB = DATA / 'labels'
DS2 = DATA / 'datasets' / 'dataset_v2_20261025' / 'aligned'
CLEAN = DATA / 'datasets' / 'clean_driving_20261011'
AIMIX = DATA / 'datasets' / 'ai_mix_20261011'
RUNS = DATA / 'runs'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
GB_LOGS = ('20261023-210840', '20261023-211209', '20261023-211547')   # protocol sets.gb_main
DE, SW, HS = 4, 5, 6                                                 # tools/label_events.LOC_DETAIL の番号
KV = 0.5e-3                                                          # m/s / eRPM(前進 = eRPM 負)
SHIFTS = ('img_up3', 'img_down3', 'bright_0.8', 'bright_1.2', 'tof_down1', 'top_0.85', 'top_1.15')
GAIN = 0.90                                                          # B・候補のジャンル 5・B-FAST(sign_rules.gains)
LO, HI = 1000.0, 2000.0                                              # config.DRIVE min_us・max_us


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- 式(自分で書いた)

def car_thr(raw_thr, neutral, gain):
    """drive.apply_gains のスロットル: 前進側(中立より小さい)だけゲイン、1000〜2000 に切る"""
    raw = np.asarray(raw_thr, np.float64)
    d = raw - neutral
    out = np.where(d < 0, neutral + gain * d, raw)
    return np.clip(out, LO, HI)


def go_frac(pred, neutral, rows):
    """止まった場面: 行 0 の車に出るスロットル < 中立 − 30 µs の割合"""
    u = car_thr(pred[rows, 0, 1], neutral, GAIN)
    return float((u < neutral - 30.0).mean())


def fwd_diff(pc, pb, neu_c, neu_b, rows, gc=GAIN, gb=GAIN):
    """行 2 の車に出る前進の量の差(候補 − B)= −(車に出る PWM の差)、フレームの平均"""
    uc = car_thr(pc[rows, 2, 1], neu_c, gc)
    ub = car_thr(pb[rows, 2, 1], neu_b, gb)
    return float(np.mean(-(uc - ub)))


def fwd_diff_own_neutral(pc, pb, neu_c, neu_b, rows):
    """参考(別の読み方): 前進の量を各モデルの中立から測る (neu − 車に出る PWM) の差"""
    uc = car_thr(pc[rows, 2, 1], neu_c, GAIN)
    ub = car_thr(pb[rows, 2, 1], neu_b, GAIN)
    return float(np.mean((neu_c - uc) - (neu_b - ub)))


def l1(p, y):
    return float(np.abs(np.asarray(p, np.float64) - np.asarray(y, np.float64)).mean())


def brake_count(row0_thr, v, valid=None):
    ok = np.isfinite(v) & (v < 1.0)
    if valid is not None:
        ok &= valid
    return int((ok & (np.asarray(row0_thr, np.float64) > 1598.0)).sum()), int(ok.sum())


# ---------------------------------------------------------------- 集合

def aligned_scenes(log_name):
    return sorted(p for p in DS2.glob(log_name + '-*.npz'))


def build_sets(proto, guard):
    seal = set(proto['seal']['logs'])
    logs = [x for v in proto['tables']['dataset']['ai_logs'].values() for x in v]
    assert len(logs) == 46 and not (set(logs) & seal), '学習の回の記録が 46 本でないか、封がある'
    stand_rows_label = {}      # ToF の前・ラベルの行の数(シーンの範囲の中)
    stand_label_any = 0        # ToF の前・ラベルの行(シーンの範囲を問わない)
    S = {'stand': [], 'stand_pre_tof': [], 'gb_de': [], 'gb_sw': [], 'gb_hs': [], 'gb_all': []}
    diag = {'stand_near_aligned_vs_label_mismatch': 0, 'stand_xfree': [], 'gb_excluded_by_vehicle_valid': 0,
            'stand_excluded_by_vehicle_valid': 0, 'stand_with_aligned_erpm': 0, 'stand_pre_tof_label_rows_in_aligned': 0}
    for lg in logs:
        assert lg not in seal
        L = np.load(LAB / (lg + '.npz'), allow_pickle=False)
        ai = L['ai'].astype(bool)
        dry = L['dry_run'].astype(bool)
        near = L['near_collision'].astype(bool)
        erpm = L['erpm'].astype(np.float64)
        locd = L['loc_detail'].astype(int)
        t_us = L['t_us'].astype(np.int64)
        lab_stand = ai & ~dry & ~near & (np.abs(erpm) < 400.0)
        stand_label_any += int(lab_stand.sum())
        files = aligned_scenes(lg)
        if not files:
            diag.setdefault('logs_without_aligned', []).append(lg)
        # シーンの範囲(生の記録の CAM_FRONT の時刻 ± 50 ms)の中のラベルの行 = 「ディスクにある」
        ondisk = np.zeros(len(t_us), bool)
        for sd in sorted((DATA / 'raw' / lg).glob(lg + '-*')):
            try:
                with np.load(sd / 'CAM_FRONT.npz', allow_pickle=False) as zc:
                    cts = zc['timestamps_us'].astype(np.int64)
            except Exception:          # noqa: BLE001
                cts = np.array(sorted(int(x.stem) for x in (sd / 'CAM_FRONT').glob('*.jpg') if x.stem.isdigit()), np.int64)
            if len(cts):
                ondisk |= (t_us >= cts[0] - 50000) & (t_us <= cts[-1] + 50000)
        ondisk_al = np.zeros(len(t_us), bool)
        used_label_rows = set()
        for f in files:
            sc = f.stem
            with np.load(f, allow_pickle=False) as z:
                src = z['source'].astype(int)
                vv = z['vehicle_valid'].astype(bool)
                li = z['label_index'].astype(np.int64)
                ts = z['timestamps_us'].astype(np.int64)
                tof = z['tof']
                veh_erpm = z['vehicle'][:, 3].astype(np.float64)
                near_al = z['near_collision'].astype(bool)
            ondisk_al |= (t_us >= ts[0] - 50000) & (t_us <= ts[-1] + 50000)
            has = li >= 0
            lic = np.where(has, li, 0)
            side = has & (src == 1) & ai[lic] & ~dry[lic]
            # 止まった場面
            st0 = side & ~near[lic] & (np.abs(erpm[lic]) < 400.0)
            diag['stand_excluded_by_vehicle_valid'] += int((st0 & ~vv).sum())
            st0 &= vv
            diag['stand_near_aligned_vs_label_mismatch'] += int((side & vv & (near_al != near[lic])).sum())
            diag['stand_with_aligned_erpm'] += int((side & vv & ~near[lic] & (np.abs(veh_erpm) < 400.0)).sum())
            for i in np.nonzero(st0)[0]:
                used_label_rows.add(int(li[i]))
                S['stand_pre_tof'].append((sc, int(i)))
                xf = guard.features(tof[i], 0.0, 0.0, guard.center)['x_free']
                if xf > 1.0:
                    S['stand'].append((sc, int(i)))
                    diag['stand_xfree'].append(float(xf))
            # gb_main
            if lg in GB_LOGS:
                g0 = side & ~near[lic]
                diag['gb_excluded_by_vehicle_valid'] += int((g0 & ~vv & np.isin(locd[lic], (DE, SW, HS))).sum())
                g0 &= vv
                for i in np.nonzero(g0)[0]:
                    S['gb_all'].append((sc, int(i)))
                    d = locd[int(li[i])]
                    if d == DE:
                        S['gb_de'].append((sc, int(i)))
                    elif d == SW:
                        S['gb_sw'].append((sc, int(i)))
                    elif d == HS:
                        S['gb_hs'].append((sc, int(i)))
        stand_rows_label[lg] = int((lab_stand & ondisk).sum())
        diag['stand_label_rows_in_aligned_spans'] = diag.get('stand_label_rows_in_aligned_spans', 0) + int((lab_stand & ondisk_al).sum())
        if int((lab_stand & ondisk).sum()) and not files:
            diag.setdefault('stand_label_rows_logs_without_aligned', {})[lg] = int((lab_stand & ondisk).sum())
        diag['stand_pre_tof_label_rows_in_aligned'] += len(used_label_rows)
    diag['stand_label_rows_ondisk'] = int(sum(stand_rows_label.values()))
    diag['stand_label_rows_any'] = stand_label_any
    xf = np.array(diag.pop('stand_xfree'))
    diag['stand_xfree_min'] = float(xf.min()) if len(xf) else None
    return S, diag, logs


def h1_rows():
    with np.load(BUNDLE, allow_pickle=False) as z:
        prim = z['primary'].astype(bool)
        scene = z['scene'].astype(str)
        frame = z['frame'].astype(np.int64)
        y = z['y'].astype(np.float64)
        erpm = z['erpm'].astype(np.float64)
        key = str(z['key'])
    rows = np.nonzero(prim)[0]
    # 正解・eRPM を aligned から読み直す
    y2 = np.full((len(scene), 10, 2), np.nan)
    e2 = np.full(len(scene), np.nan)
    cache = {}
    for i in rows:
        s = scene[i]
        if s not in cache:
            with np.load(CLEAN / 'aligned' / (s + '.npz'), allow_pickle=False) as a:
                cache[s] = (a['future_actions'].astype(np.float64), a['vehicle'][:, 3].astype(np.float64))
        fa, ve = cache[s]
        y2[i] = fa[frame[i]]
        e2[i] = ve[frame[i]]
    return {'rows': rows, 'y': y, 'y_mine': y2, 'erpm': erpm, 'erpm_mine': e2, 'key': key, 'scene': scene, 'frame': frame}


def ai1010_frames(proto):
    logs = set(proto['tables']['dataset']['ai_logs']['10/10'])
    sp = json.load(open(AIMIX / 'splits.json'))
    rows = []
    for split in ('train', 'val'):
        for s in sp[split]:
            if s[:15] not in logs:
                continue
            with np.load(AIMIX / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                ok = (z['source'] == 1) & z['vehicle_valid'].astype(bool)
                for i in np.nonzero(ok)[0]:
                    rows.append((s, int(i), int(z['timestamps_us'][i]), float(z['vehicle'][i, 3])))
    rows.sort(key=lambda r: (r[0][:15], r[2], r[0], r[1]))
    return rows


# ---------------------------------------------------------------- 照らす

def keyset(lst):
    return {(str(a), int(b)) for a, b in lst}


def main():
    t0 = time.time()
    proto = json.load(open(REP / 'protocol_v2.json', encoding='utf-8'))
    desk = json.load(open(DESK / 'desk_v2.json', encoding='utf-8'))
    out = {'schema': 'jf-verify-c3-cpu/v1', 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'script': str(Path(__file__).resolve()), 'script_sha256': sha(Path(__file__).resolve()),
           'desk_v2_json_sha256': sha(DESK / 'desk_v2.json'), 'desk_made_at': desk['made_at'],
           'protocol_v2_sha256': sha(REP / 'protocol_v2.json'), 'checks': []}

    def check(name, ok, mine=None, c3=None, note=''):
        out['checks'].append({'name': name, 'ok': bool(ok), 'mine': mine, 'c3': c3, 'note': note})
        log('%s %s  mine=%s c3=%s %s' % ('OK' if ok else 'NG', name, mine, c3, note))

    from jetformer.tof_guard import ToFGuard
    from jetformer.train_data import top_ramp, TOP_FRACTION
    guard = ToFGuard(np.load(DATA / 'tof_floor_ref.npy'))
    # deviations 7 の重み: 一番上の行 1、上 1/3 の境で 0 の直線(自分の式と本体の top_ramp を照らす)
    H = 224
    w_mine = np.clip(1.0 - np.arange(H) / (H / 3.0), 0.0, 1.0)
    w_lib = top_ramp(H).reshape(-1).astype(np.float64)
    check('deviations7: top_ramp = 一番上 1・上 1/3 の境で 0 の直線', np.abs(w_mine - w_lib).max() < 1e-6 and
          abs(TOP_FRACTION - 1 / 3) < 1e-12 and w_lib[0] == 1.0 and w_lib[75] == 0.0,
          mine={'w0': float(w_mine[0]), 'w37': round(float(w_mine[37]), 4), 'w74': round(float(w_mine[74]), 4),
                'w75': float(w_mine[75])}, c3=None)

    # ---- 1. 集合
    log('集合を作ります(46 本)')
    S, diag, logs = build_sets(proto, guard)
    out['sets_diag'] = diag
    cnt = {k: len(v) for k, v in S.items()}
    out['sets_counts'] = cnt
    log('数', cnt, diag)
    check('止まった場面のラベルの行(ToF の前) = protocol の上限 1,622', diag['stand_label_rows_ondisk'] == 1622,
          mine=diag['stand_label_rows_ondisk'], c3=desk['stage0']['4'].get('label_rows_before_tof'))
    check('止まった場面の aligned(ToF の前)', cnt['stand_pre_tof'] == desk['stage0']['4'].get('aligned_before_tof'),
          mine=cnt['stand_pre_tof'], c3=desk['stage0']['4'].get('aligned_before_tof'))
    check('止まった場面(ToF の後)= desk の standstill_frames', cnt['stand'] == desk['standstill_frames'],
          mine=cnt['stand'], c3=desk['standstill_frames'])
    check('止まった場面 ≥ 300 → 門', (cnt['stand'] >= 300) == (desk['standstill_gate'] == '門'),
          mine='門' if cnt['stand'] >= 300 else '参考', c3=desk['standstill_gate'])
    for k, ck in (('gb_de', 'gb_de'), ('gb_sw', 'gb_sw'), ('gb_hs', 'gb_hs')):
        check('%s の数' % k, cnt[k] == desk['set_counts'][ck], mine=cnt[k], c3=desk['set_counts'][ck])
    check('gb_main の全部の数', cnt['gb_all'] == desk['set_counts']['gb'], mine=cnt['gb_all'], c3=desk['set_counts']['gb'])

    # C3 の集合と同じフレームか
    with np.load(DESK / 'sets_v2.npz', allow_pickle=False) as z:
        T = {k: z[k] for k in z.files}
    tsc, tfr = T['scene'].astype(str), T['frame'].astype(np.int64)
    idx = {(s, int(f)): i for i, (s, f) in enumerate(zip(tsc, tfr))}
    for mine_k, c3_k in (('stand', 'stand'), ('gb_de', 'gb_de'), ('gb_sw', 'gb_sw'), ('gb_hs', 'gb_hs'), ('gb_all', 'gb')):
        c3set = keyset(zip(tsc[T[c3_k]], tfr[T[c3_k]]))
        ms = keyset(S[mine_k])
        check('集合 %s のフレームが C3 と同じ' % mine_k, ms == c3set, mine={'only_mine': len(ms - c3set)},
              c3={'only_c3': len(c3set - ms)})
    # 表の行の番号(保存した予測を引くため)
    R = {}
    for k in ('stand', 'gb_de', 'gb_sw', 'gb_hs'):
        R[k] = np.array([idx[x] for x in S[k] if x in idx], np.int64)
        assert len(R[k]) == len(S[k]), '%s のフレームが C3 の表に無い' % k
    R['gb_dsh'] = np.concatenate([R['gb_de'], R['gb_sw'], R['gb_hs']])
    np.savez_compressed(HERE / 'sets_mine.npz',
                        **{'%s_scene' % k: np.array([a for a, _ in S[k]]) for k in ('stand', 'gb_de', 'gb_sw', 'gb_hs')},
                        **{'%s_frame' % k: np.array([b for _, b in S[k]], np.int32) for k in ('stand', 'gb_de', 'gb_sw', 'gb_hs')})

    # ---- H1・低速
    log('H1 を読みます')
    Hh = h1_rows()
    rows = Hh['rows']
    check('H1 = 4,670 フレーム', len(rows) == 4670, mine=int(len(rows)), c3=4670)
    dy = np.nanmax(np.abs(Hh['y_mine'][rows] - Hh['y'][rows]))
    check('H1 の正解 = aligned の future_actions(読み直し)', dy == 0.0 and np.isfinite(Hh['y'][rows]).all(),
          mine=float(dy), c3=None)
    de = np.nanmax(np.abs(Hh['erpm_mine'][rows] - Hh['erpm'][rows]))
    check('H1 の eRPM = aligned の vehicle[:,3]', de == 0.0, mine=float(de))
    v_h1 = -Hh['erpm_mine'] * KV
    low_h1 = np.zeros(len(v_h1), bool)
    low_h1[rows] = v_h1[rows] < 1.0
    check('低速のブレーキ: H1 の低速 = 222', int(low_h1.sum()) == 222, mine=int(low_h1.sum()), c3=222)
    log('10/10 の AI 走行のフレームを読みます')
    A = ai1010_frames(proto)
    check('10/10 の AI 走行 = 21,759 フレーム', len(A) == 21759, mine=len(A), c3=21759)
    e_ai = np.array([r[3] for r in A], np.float64)
    v_ai = -e_ai * KV
    low_ai = v_ai < 1.0
    check('低速のブレーキ: 10/10 の低速 = 2,326', int(low_ai.sum()) == 2326, mine=int(low_ai.sum()), c3=2326)

    # ---- 3. 保存した予測に自分の式
    ref = {}
    for name in desk['runs']:
        cfgp = RUNS / name / 'model_config.json'
        if cfgp.exists():
            c = json.load(open(cfgp))
            ref[name] = (float(c['steer_center_us']), float(c['throttle_neutral_us']))
    tabs = {}
    for name in desk['runs']:
        p = DESK / 'pred' / ('%s_tab.npz' % name)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                assert str(z['key']) == str(T['key'])
                tabs[name] = (z['pred'].astype(np.float64), z['genre'].astype(int))
    pB = tabs[B_RUN][0]
    nB = ref[B_RUN][1]
    fields = {}
    for name, row in desk['runs'].items():
        if name not in tabs or name not in ref:
            continue
        pc, gc = tabs[name]
        neu = ref[name][1]
        f = {}
        f['stand_genre_all_5'] = bool((gc[R['stand']] == 5).all())
        f['standstill_go_frac'] = go_frac(pc, neu, R['stand'])
        if row.get('family') == 'bfast':
            f['gb_genre_all_5'] = bool((gc[R['gb_dsh']] == 5).all() and (tabs[B_RUN][1][R['gb_dsh']] == 5).all())
            f['bfast_gb_fwd_vs_b_us'] = fwd_diff(pc, pB, neu, nB, R['gb_dsh'])
            f['bfast_gb_fwd_vs_b_us_de'] = fwd_diff(pc, pB, neu, nB, R['gb_de'])
            f['bfast_gb_fwd_vs_b_us_sw'] = fwd_diff(pc, pB, neu, nB, R['gb_sw'])
            f['bfast_gb_fwd_vs_b_us_hs'] = fwd_diff(pc, pB, neu, nB, R['gb_hs'])
            f['ref_gb_own_neutral_us'] = fwd_diff_own_neutral(pc, pB, neu, nB, R['gb_dsh'])
            f['ref_gb_mean_of_three_us'] = float(np.mean([f['bfast_gb_fwd_vs_b_us_de'], f['bfast_gb_fwd_vs_b_us_sw'],
                                                          f['bfast_gb_fwd_vs_b_us_hs']]))
        fields[name] = f
    # robust_dl1_us(shift.npz + eval/pred の保存した best)
    yH = Hh['y'][rows]
    for name, row in desk['runs'].items():
        p = DESK / 'pred' / ('%s_shift.npz' % name)
        e = REP / 'eval' / 'pred' / ('%s.npz' % name)
        if not p.exists() or not e.exists():
            continue
        with np.load(e, allow_pickle=False) as z:
            assert str(z['key']) == Hh['key'] and str(z['ckpt']) == 'best'
            base = z['pred'].astype(np.float64)[rows]
        with np.load(p, allow_pickle=False) as z:
            ok_rows = bool((z['rows'].astype(np.int64) == rows).all())
            per = {k: l1(z['pred_' + k], yH) - l1(base, yH) for k in SHIFTS}
            nr = z['pred_none_rows'].astype(np.int64)
            none_chk = float(np.abs(z['pred_none'].astype(np.float64) - np.asarray(
                np.load(e)['pred'], np.float64)[rows][nr]).max())
            bsha = str(z['best_sha256'])
        fields.setdefault(name, {})
        fields[name]['robust_rows_same'] = ok_rows
        fields[name]['robust_dl1_us'] = float(np.mean(list(per.values())))
        fields[name]['robust_per_shift'] = per
        fields[name]['robust_none_rows_vs_saved_max_abs_us'] = none_chk
        fields[name]['robust_best_sha_same'] = bsha == sha(RUNS / name / 'best.pth')
    # 低速のブレーキ(eval/pred の best + trt の 10/10)
    for name, row in desk['runs'].items():
        e = REP / 'eval' / 'pred' / ('%s.npz' % name)
        job = (row.get('trt') or {}).get('trt_job')
        if not e.exists() or not job:
            continue
        tp = DESK / 'trt' / job / ('%s.npz' % name)
        if not tp.exists():
            continue
        with np.load(e, allow_pickle=False) as z:
            pb = z['pred'].astype(np.float64)
        kh, nh = brake_count(pb[:, 0, 1], v_h1, low_h1 & np.isfinite(pb[:, 0, 1]))
        with np.load(tp, allow_pickle=False) as z:
            tpred = z['pred'].astype(np.float64)
            terpm = z['erpm'].astype(np.float64)
            part = z['part'].astype(int)
            tmp = z['model_pred'].astype(np.float64)
        sel = part == 0
        erpm_same = bool(sel.sum() == len(e_ai) and np.array_equal(terpm[sel], e_ai))
        ka, na = brake_count(tpred[sel, 0, 1], v_ai, np.isfinite(tpred[sel, 0, 1]))
        fields.setdefault(name, {})
        fields[name]['brake_heldout'] = [kh, nh]
        fields[name]['brake_ai_1010'] = [ka, na]
        fields[name]['brake_trt_erpm_order_same'] = erpm_same
        fields[name]['brake_pred_eq_model_pred'] = bool(np.nanmax(np.abs(tpred - tmp)) == 0.0)
        fields[name]['brake_lowspeed_frac'] = (kh + ka) / float(nh + na)
        fields[name]['brake_ref_own_neutral_plus_100'] = None
        neu = ref.get(name, (None, None))[1]
        if neu is not None:
            k1 = int((low_h1 & (pb[:, 0, 1] > neu + 100.0)).sum())
            k2 = int((low_ai & (tpred[sel, 0, 1] > neu + 100.0)).sum())
            fields[name]['brake_ref_own_neutral_plus_100'] = (k1 + k2) / float(nh + na)
    # ---- desk と照らす
    cmp = {}
    tol = {'standstill_go_frac': 5e-6, 'bfast_gb_fwd_vs_b_us': 5e-6, 'bfast_gb_fwd_vs_b_us_de': 5e-6,
           'bfast_gb_fwd_vs_b_us_sw': 5e-6, 'bfast_gb_fwd_vs_b_us_hs': 5e-6, 'robust_dl1_us': 5e-6,
           'brake_lowspeed_frac': 5e-6}
    for name, f in fields.items():
        row = desk['runs'][name]
        c = {}
        for k, t in tol.items():
            if k in f:
                cv = row.get(k)
                mv = f[k]
                ok = cv is not None and abs(round(mv, 5) - cv) <= t + 1e-12
                c[k] = {'mine': mv, 'c3': cv, 'ok': ok}
        if 'brake_heldout' in f:
            bl = (row.get('trt') or {}).get('brake_lowspeed') or {}
            c['brake_counts'] = {'mine': [f['brake_heldout'], f['brake_ai_1010']],
                                 'c3': [bl.get('heldout'), bl.get('ai_1010')],
                                 'ok': [f['brake_heldout'], f['brake_ai_1010']] == [bl.get('heldout'), bl.get('ai_1010')]}
        if 'robust_per_shift' in f and row.get('robust_dl1_per_shift_us'):
            mx = max(abs(round(f['robust_per_shift'][k], 4) - row['robust_dl1_per_shift_us'][k]) for k in SHIFTS)
            c['robust_per_shift'] = {'max_abs_diff_4dp': mx, 'ok': mx <= 1.1e-4}
        cmp[name] = c
    out['fields'] = fields
    out['compare'] = cmp
    n_ng = 0
    for name, c in cmp.items():
        for k, v in c.items():
            if not v['ok']:
                n_ng += 1
                log('NG %s %s %s' % (name, k, v))
    focus = [B_RUN, 'v2_20261025-v2h-e60-s0', 'v2_20261025-v2h-e60-s1', 'bfast_20261025-e15-s0']
    for name in focus:
        log(name, json.dumps({k: (v['mine'] if 'mine' in v else v) for k, v in cmp.get(name, {}).items()}, ensure_ascii=False))
    # G-d・門(自分の値で)
    gB = fields[B_RUN]['standstill_go_frac']
    gates = {}
    for name, f in fields.items():
        fam = desk['runs'][name].get('family')
        g = {}
        if 'standstill_go_frac' in f and name != B_RUN:
            g['standstill_or_G-d'] = f['standstill_go_frac'] >= gB - 0.10
        if 'brake_lowspeed_frac' in f:
            g['brake'] = f['brake_lowspeed_frac'] <= 0.01
        if fam == 'bfast' and 'bfast_gb_fwd_vs_b_us' in f:
            g['G-b'] = f['bfast_gb_fwd_vs_b_us'] >= 10.0
        dg = desk['runs'][name].get('gates') or {}
        same = {}
        for k, v in g.items():
            ck = {'standstill_or_G-d': 'G-d' if fam == 'bfast' else 'standstill'}.get(k, k)
            if ck in dg:
                same[k] = dg[ck] == v
        gates[name] = {'mine': g, 'same_as_desk_gates': same}
    out['gates_mine'] = gates
    out['compare_ng'] = n_ng
    out['seconds'] = round(time.time() - t0, 1)
    p = HERE / 'recount_cpu.json'
    tmp = str(p) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, p)
    log('書きました', p, '照らしの NG', n_ng, '確かめの NG', sum(1 for x in out['checks'] if not x['ok']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
