#!/usr/bin/env python3
"""原因 B の診断: B-FAST が同じゲインで B より踏まない理由(2026-10-26。読むだけ・numpy だけ・torch を読まない)。

確かめること
  A. 手本(No.9 20261024-162033・No.10 162305 の 10 周)の正解 future_actions が、AI_CMD の
       raw(モデル B の行 2 の予測 = ゲインの前)・model_out(apply_gains のあと・ガードや補助の前)・out(車に出た値)の
     どれと同じか。ゲイン(gains[:, 1])と、前進の量の比 (中立 − 正解) / (中立 − raw)。
  B. 机上の予測(desk/pred/*_tab.npz。C3 が保存した PyTorch FP32 best)で
       ・G-b(B-FAST 0.90 対 B 0.90)と g100(B-FAST 1.00 対 B 0.90)を desk_v2.py と同じ式で出し直す(照らし合わせ)
       ・「何も覚えなかった B-FAST」(= B そのもの)と「正解を完全に覚えた B-FAST」(= B の前進を 0.9 倍)の値
       ・観測の B-FAST が、正解の縮みをどれだけ覚えたか(覚えた割合 λ)を集合ごとに
       ・直し方 3 つの見込み(A 正解をゲインの前に・B 正解を手本の速さ(走らせるゲイン 0.90)に・C 走らせるゲイン 1.00)
       ・公平な門の候補(同じゲイン・形の門)を、手本を見ていない 4 本(A_s0・A_s1・v0-s2・v0-s3)と v2 19 本に掛けた値
  C. 正解の行のずれ: AI の正解の行 k = 車に出た値 out[i + k]。out[i] は B の行 2(先読み 3)なので、
     B-FAST を先読み 3 で走らせると「B の 2 フレーム先の指令」を出す。その大きさ(B の予測で)。

車・PCA9685・I2C・カメラ・ToF・基板に触れない。走行の API を呼ばない。data/・protocol・lock2・desk は読むだけ。
書くのは /home/jetson/jetformer_reports/fix_20261026/diag_bfast.json だけ。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 \
      /home/jetson/miniconda/envs/jetracer/bin/python -I \
      /home/jetson/jetformer/experiments/20261026_fix/diag_bfast/diag_bfast.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data/datasets/dataset_v2_20261025'
TABLES = REPO / 'data/derived/dataset_v2_20261025'
RAW = REPO / 'data/raw'
RUNS = REPO / 'data/runs'
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
DESK = REP / 'desk'
OUT = Path('/home/jetson/jetformer_reports/fix_20261026/diag_bfast.json')

DEMO_LOGS = ('20261024-162033', '20261024-162305')     # No.9(ゲイン 0.90)・No.10(1.00)
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
BF_RUNS = ['bfast_20261025-e15-s%d' % i for i in range(3)]
REF4 = {'A_s0': 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
        'A_s1': 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
        'v0-s2': 'realacc_20261015-v0-s2', 'v0-s3': 'realacc_20261015-v0-s3'}
G = 0.90                    # protocol: B・B-FAST・ジャンル 5 のゲイン
ROW2 = 2
FWD_MIN_US = 20.0           # 前進の比を出すフレーム: raw が中立より 20 µs 以上前
LO, HI = 1000.0, 2000.0
FRAME_US = 1e6 / 30.0       # config.CAMERA fps(build_ai_dataset.FRAME_US と同じ)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r(x, n=3):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def car_out_thr(pred_thr, neutral, gain):
    """drive.apply_gains のスロットル(前進側だけにゲイン・1000〜2000 に切る)"""
    p = np.asarray(pred_thr, np.float64)
    dt = p - neutral
    return np.clip(neutral + np.where(dt < 0, gain * dt, dt), LO, HI)


def ref_of(run):
    c = json.loads((RUNS / run / 'model_config.json').read_text(encoding='utf-8'))
    return float(c['steer_center_us']), float(c['throttle_neutral_us'])


# ============================================================ A. 手本の正解
def part_a():
    d = np.load(TABLES / 'drop_bfast.npz', allow_pickle=False)
    w = np.load(TABLES / 'weights_bfast.npz', allow_pickle=False)
    assert np.array_equal(d['scene'], w['scene']) and np.array_equal(d['frame'], w['frame'])
    keep = ~d['drop']
    res = {'by_log': {}, 'note': '手本のフレーム = drop_bfast で残した・No.9/No.10 のシーン・aligned の source 1・future_valid'
                                  '(train.py が掛ける vehicle_valid・衝突の前後も外した)'}
    demo_keys = set()
    missing_total = [0]
    pooled = {'lab0': [], 'raw': [], 'mo': [], 'out': [], 'neutral': [], 'w': [], 'log': [], 'gain': []}
    for lg in DEMO_LOGS:
        info = json.loads((REPO / 'data/labels' / (lg + '.json')).read_text(encoding='utf-8'))
        ref = info.get('ref_us') or {}
        neutral = float(ref.get('throttle_neutral', 1500.0))
        acc = {k: [] for k in ('lab', 'lab0', 'out', 'out_k', 'mo', 'raw', 'pred2', 'mpred', 'gain', 'w', 'assist',
                               'guard', 'la')}
        scenes = sorted(set(s for s in np.unique(d['scene']) if s.startswith(lg + '-')))
        for sc in scenes:
            ms = (d['scene'] == sc)
            fr = d['frame'][ms]
            kp = keep[ms]
            ww = w['weight'][ms]
            z = np.load(DS / 'aligned' / (sc + '.npz'), allow_pickle=False)
            a = np.load(RAW / lg / sc / 'AI_CMD.npz', allow_pickle=False)
            tz, ta = z['timestamps_us'].astype(np.int64), a['timestamps_us'].astype(np.int64)
            n = len(tz)
            ai_i = np.clip(np.searchsorted(ta, tz), 0, len(ta) - 1)
            has_ai = ta[ai_i] == tz          # カメラのフレームに同じ時刻の AI_CMD がある
            n_missing = int((~has_ai).sum())
            # 行 k の out: build_ai_dataset.lookahead と同じ(撮像から k+0.5 フレーム以内の最新の AI_CMD)
            jk_all = np.stack([np.searchsorted(ta, tz + int(round((k + 0.5) * FRAME_US)), side='right') - 1
                               for k in range(10)], axis=1)
            kk = np.zeros(n, bool)
            kk[fr[kp]] = True
            wf = np.ones(n, np.float32)
            wf[fr] = ww
            sel = kk & (z['source'] == 1) & z['future_valid'] & z['vehicle_valid'] & ~z['near_collision'] & has_ai
            idx = np.nonzero(sel)[0]
            missing_total[0] += n_missing
            for i in idx:
                demo_keys.add((sc, int(i)))
            ia = ai_i[idx]
            la = a['lookahead'].astype(int)
            acc['lab'].append(z['future_actions'][idx])
            acc['lab0'].append(z['future_actions'][idx, 0])
            acc['out'].append(a['out'][ia])
            jk = np.clip(jk_all[idx], 0, len(ta) - 1)
            acc['out_k'].append((z['future_actions'][idx][:, :, 1], a['out'][jk][:, :, 1]))
            acc['mo'].append(a['model_out'][ia])
            acc['raw'].append(a['raw'][ia])
            acc['pred2'].append(a['pred'][ia, np.clip(la[ia] - 1, 0, 9)])
            acc['mpred'].append(np.abs(a['pred'][ia] - a['model_pred'][ia]).max(axis=(1, 2)))
            acc['gain'].append(a['gains'][ia, 1])
            acc['w'].append(wf[idx])
            acc['assist'].append(a['assist'][ia])
            acc['guard'].append(a['guard_applied'][ia])
            acc['la'].append(la[ia])
        cat = {k: (np.concatenate(v) if k != 'out_k' else v) for k, v in acc.items()}
        lab0, out, mo, raw = cat['lab0'], cat['out'], cat['mo'], cat['raw']
        lk = np.concatenate([x[0] for x in cat['out_k']])
        ok_ = np.concatenate([x[1] for x in cat['out_k']])
        fwd = (neutral - raw[:, 1]) > FWD_MIN_US
        ratio = (neutral - lab0[fwd, 1]) / (neutral - raw[fwd, 1])
        res['by_log'][lg] = {
            'neutral_us': neutral, 'frames': int(len(lab0)), 'weight_sum': r(cat['w'].sum(), 1),
            'gains_thr_unique': sorted(set(round(float(x), 3) for x in cat['gain'])),
            'lookahead_unique': sorted(set(int(x) for x in cat['la'])),
            'assist_nonzero_frames': int((cat['assist'] != 0).sum()), 'guard_applied_frames': int(cat['guard'].sum()),
            'pred_eq_model_pred_max_us': r(cat['mpred'].max(), 4),
            'raw_eq_pred_row_lookahead_max_us': r(np.abs(raw - cat['pred2']).max(), 4),
            'label_row0_minus_out_maxabs_us': [r(np.abs(lab0 - out)[:, 0].max(), 4), r(np.abs(lab0 - out)[:, 1].max(), 4)],
            'label_row0_minus_model_out_maxabs_us': [r(np.abs(lab0 - mo)[:, 0].max(), 4), r(np.abs(lab0 - mo)[:, 1].max(), 4)],
            'label_row0_minus_raw_thr_mean_us': r(np.mean(lab0[:, 1] - raw[:, 1])),
            'label_row0_minus_raw_thr_maxabs_us': r(np.abs(lab0[:, 1] - raw[:, 1]).max()),
            'label_rowk_eq_out_i_plus_k_frac': r(np.mean(np.abs(lk - ok_) <= 0.01), 4),
            'fwd_frames': int(fwd.sum()),
            'fwd_ratio_label_over_raw': {'median': r(np.median(ratio), 4), 'p05': r(np.percentile(ratio, 5), 4),
                                         'p95': r(np.percentile(ratio, 95), 4)},
            'mean_fwd_raw_us': r(np.mean(neutral - raw[:, 1])),
            'mean_fwd_label_us': r(np.mean(neutral - lab0[:, 1])),
            'mean_fwd_removed_by_label_us': r(np.mean(lab0[:, 1] - raw[:, 1])),
        }
        for k_, v_ in (('lab0', lab0), ('raw', raw), ('mo', mo), ('out', out)):
            pooled[k_].append(v_)
        pooled['neutral'].append(np.full(len(lab0), neutral))
        pooled['w'].append(cat['w'])
        pooled['log'].append(np.full(len(lab0), lg))
        pooled['gain'].append(cat['gain'])
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    fw = (P['neutral'] - P['raw'][:, 1]) > FWD_MIN_US
    wts = P['w'][fw]
    rat = (P['neutral'][fw] - P['lab0'][fw, 1]) / (P['neutral'][fw] - P['raw'][fw, 1])
    res['camera_frames_without_ai_cmd'] = missing_total[0]
    res['pooled'] = {
        'frames': int(len(P['w'])), 'weight_sum': r(P['w'].sum(), 1),
        'weighted_mean_fwd_ratio_label_over_raw': r(np.sum(wts * rat) / np.sum(wts), 4),
        'share_no10_weight': r(P['w'][P['log'] == DEMO_LOGS[1]].sum() / P['w'].sum(), 4),
    }
    return res, demo_keys, P


# ============================================================ B. 机上の予測
def load_tab(run):
    z = np.load(DESK / 'pred' / (run + '_tab.npz'), allow_pickle=False)
    return z['pred'].astype(np.float64), str(z['key'])


def part_b(demo_keys):
    S = np.load(DESK / 'sets_v2.npz', allow_pickle=False)
    key = str(S['key'])
    scene, frame = S['scene'], S['frame']
    pB, kB = load_tab(B_RUN)
    assert kB == key
    refB = ref_of(B_RUN)
    nB = refB[1]
    rows = {(str(s), int(f)): i for i, (s, f) in enumerate(zip(scene, frame))}
    demo = np.zeros(len(scene), bool)
    for kk in demo_keys:
        if kk in rows:
            demo[rows[kk]] = True
    masks = {'gb_dsh': S['gb_dsh'], 'gb_de': S['gb_de'], 'gb_sw': S['gb_sw'], 'gb_hs': S['gb_hs'],
             'gb_r1pre': S['gb_r1pre'], 'gbref_dsh': S['gbref_dsh'], 'straight': S['straight'], 'demo': demo,
             'demo_dsh': demo & np.isin(S['loc_detail'], [4, 5, 6])}
    desk = json.loads((DESK / 'desk_v2.json').read_text(encoding='utf-8'))['runs']

    def fwd(pred_thr, neutral, gain, m):
        """車に出る前進の量の平均(B の中立 1496 から測る)[µs]"""
        return float(np.mean(nB - car_out_thr(pred_thr[m], neutral, gain)))

    out = {'sets_key': key, 'demo_rows_in_tab': int(demo.sum()), 'demo_keys': len(demo_keys),
           'set_counts': {k: int(v.sum()) for k, v in masks.items()}, 'refB': refB}
    # B の生の前進の量・0.90 のあと
    base = {}
    for name, m in masks.items():
        m = np.asarray(m, bool)
        if not m.any():
            continue
        thr = pB[:, ROW2, 1]
        base[name] = {'B_raw_fwd_us': r(fwd(thr, nB, 1.0, m)), 'B_car_fwd_g090_us': r(fwd(thr, nB, G, m)),
                      'B_car_fwd_g100_us': r(fwd(thr, nB, 1.0, m))}
        # 何も覚えなかった B-FAST(= B)を 1.00 で: 1.00 対 0.90 の差
        base[name]['null_g100_vs_b090_us'] = r(fwd(thr, nB, 1.0, m) - fwd(thr, nB, G, m))
        # 正解(ゲイン 0.90 のあと)を完全に覚えた B-FAST = B の前進側を 0.9 倍した予測
        shr = car_out_thr(thr, nB, G)
        base[name]['perfect_current_labels_gb_g090_us'] = r(fwd(shr, nB, G, m) - fwd(thr, nB, G, m))
        base[name]['perfect_current_labels_g100_vs_b090_us'] = r(fwd(shr, nB, 1.0, m) - fwd(thr, nB, G, m))
    out['B_and_bounds'] = base
    # B-FAST・手本を見ていないモデル
    models = {}
    names = [(x, x) for x in BF_RUNS] + list(REF4.items())
    v2 = sorted(k for k in desk if k.startswith('v2_20261025-'))
    names += [(x.replace('v2_20261025-', ''), x) for x in v2]
    for label, run in names:
        p, k = load_tab(run)
        assert k == key
        ref = ref_of(run)
        thr = p[:, ROW2, 1]
        e = {'run': run, 'ref': ref}
        for name, m in masks.items():
            m = np.asarray(m, bool)
            if not m.any():
                continue
            f9 = fwd(thr, ref[1], G, m)
            fb9 = fwd(pB[:, ROW2, 1], nB, G, m)
            e[name] = {'gb_g090_us': r(f9 - fb9), 'g100_vs_b090_us': r(fwd(thr, ref[1], 1.0, m) - fb9),
                       'raw_fwd_us': r(fwd(thr, ref[1], 1.0, m)),
                       # 覚えた割合: 観測の縮み / 完全に覚えたときの縮み
                       'lambda_vs_perfect': r((f9 - fb9) / base[name]['perfect_current_labels_gb_g090_us'], 3)
                       if base[name]['perfect_current_labels_gb_g090_us'] else None,
                       'scale_vs_B': r(fwd(thr, ref[1], 1.0, m) / fwd(pB[:, ROW2, 1], nB, 1.0, m), 4)}
        # 形の門: 直線の前進の比で B を伸ばしたものと比べる(ゲインだけの違いは 0 になる)
        sc = e['straight']['scale_vs_B']
        fd = fwd(thr, ref[1], G, np.asarray(masks['gb_dsh'], bool))
        fbd = fwd(pB[:, ROW2, 1], nB, G, np.asarray(masks['gb_dsh'], bool))
        e['gb_shape_us'] = r(fd - sc * fbd)
        # 中立が違う分(B-FAST 1500・B 1496)が G-b に効く量(0.90 で): (1 − g) × 4 µs のうちの前進側
        if run in desk:
            dk = desk[run]
            e['desk_check'] = {'gb': dk.get('bfast_gb_fwd_vs_b_us'), 'g100': dk.get('bfast_gb_fwd_g100_vs_b090_us'),
                               'straight_fwd': dk.get('straight_fwd_vs_b_us')}
        models[label] = e
    out['models'] = models
    # B-FAST の手本のフレームで、予測が正解(0.9 倍)にどれだけ寄ったか(学習の中)
    out['neutral_shift_effect_us'] = r(-(1 - G) * (ref_of(BF_RUNS[0])[1] - nB))
    return out, S, pB, masks


# ============================================================ C. 行のずれ
def part_c(S, pB, masks):
    """B の行 2 の予測を、同じシーンの 2 フレーム先と比べる(B-FAST が正解を完全に覚え、先読み 3 で走ったときの前倒し)"""
    scene, frame, t_us = S['scene'], S['frame'], S['t_us']
    rows = {(str(s), int(f)): i for i, (s, f) in enumerate(zip(scene, frame))}
    nB = ref_of(B_RUN)[1]
    res = {}
    for name in ('gb_dsh', 'gb_de', 'gb_sw', 'gb_hs', 'gb_r1pre', 'demo_dsh'):
        m = np.nonzero(np.asarray(masks[name], bool))[0]
        d2, n = [], 0
        for i in m:
            j = rows.get((str(scene[i]), int(frame[i]) + 2))
            if j is None or not (40_000 <= t_us[j] - t_us[i] <= 100_000):
                continue
            a = nB - car_out_thr(pB[j, ROW2, 1], nB, G)
            b = nB - car_out_thr(pB[i, ROW2, 1], nB, G)
            d2.append(a - b)
            n += 1
        d2 = np.asarray(d2)
        res[name] = {'frames': n, 'mean_fwd_shift2_us': r(d2.mean()) if n else None,
                     'mean_abs_us': r(np.abs(d2).mean()) if n else None}
    return res


# ============================================================ D. 正解の段の上限(学習の前に分かる)
def part_d(S, pB, masks):
    """手本のフレームで、正解の行 2 を 0.90 で車に出した値 − B の行 2 を 0.90 で出した値(前進 +)。
    完全に覚えても、手本の場面で B よりこれ以上は踏めない(10/23 の場面では、ふつう覚えた割合 λ < 1 だけ小さい)。
    今の正解 = out[i+2]、直し方 A = raw[i+2](ゲインの前)、直し方 B = out[i+2] を 0.90 で割り戻した値(走らせるゲインの尺度)"""
    rows = {(str(s), int(f)): i for i, (s, f) in enumerate(zip(S['scene'], S['frame']))}
    nB = ref_of(B_RUN)[1]
    acc = {'cur': [], 'fixA': [], 'fixB': [], 'B': [], 'raw_i': [], 'dsh': [], 'log': []}
    for lg in DEMO_LOGS:
        for sc in sorted(set(str(x) for x in S['scene'] if str(x).startswith(lg + '-'))):
            z = np.load(DS / 'aligned' / (sc + '.npz'), allow_pickle=False)
            a = np.load(RAW / lg / sc / 'AI_CMD.npz', allow_pickle=False)
            tz, ta = z['timestamps_us'].astype(np.int64), a['timestamps_us'].astype(np.int64)
            for i in range(len(tz)):
                j = rows.get((sc, i))
                if j is None or not masks['demo'][j]:
                    continue
                ii = int(np.searchsorted(ta, tz[i]))
                j2 = int(np.searchsorted(ta, tz[i] + int(round(2.5 * FRAME_US)), side='right') - 1)
                out2, raw2 = float(a['out'][j2, 1]), float(a['raw'][j2, 1])
                d = out2 - nB
                acc['cur'].append(float(z['future_actions'][i, 2, 1]))
                acc['fixA'].append(raw2)
                acc['fixB'].append(nB + d / G if d < 0 else out2)
                acc['B'].append(float(pB[j, ROW2, 1]))
                acc['raw_i'].append(float(a['raw'][ii, 1]))
                acc['dsh'].append(bool(masks['demo_dsh'][j]))
                acc['log'].append(lg)
    A = {k: np.asarray(v) for k, v in acc.items()}
    res = {'frames': int(len(A['B'])),
           'B_desk_pred_row2_minus_recorded_raw_us': {'median_abs': r(np.median(np.abs(A['B'] - A['raw_i'])), 4),
                                                      'p99_abs': r(np.percentile(np.abs(A['B'] - A['raw_i']), 99), 4)}}
    for name, m in (('demo', np.ones(len(A['B']), bool)), ('demo_dsh', A['dsh'])):
        e = {}
        fb = nB - car_out_thr(A['B'][m], nB, G)
        for opt in ('cur', 'fixA', 'fixB'):
            e[opt + '_bound_g090_us'] = r(np.mean(nB - car_out_thr(A[opt][m], nB, G)) - np.mean(fb))
        res[name] = e
    return res


# ============================================================ E. 門の自己試験(作り物のモデル)
def part_e(pB, masks):
    """B の行 2 の前進側を c 倍した作り物のモデルを、同じゲイン 0.90 の G-b1(候補 − B)と形の門 G-b3 に掛ける。
    c = 1(何も覚えない)・0.9098(今の正解を完全に覚えた)・0.95/0.90(ゲイン 0.95 の B と同じ)。G-b3 はどれも 0 のはず"""
    nB = ref_of(B_RUN)[1]
    thr = pB[:, ROW2, 1]
    dsh, st = np.asarray(masks['gb_dsh'], bool), np.asarray(masks['straight'], bool)

    def fw(t, m, g):
        return float(np.mean(nB - car_out_thr(t[m], nB, g)))
    res = {}
    for name, c in (('B_itself', 1.0), ('B_x0.9098_perfect_current_labels', 0.9098), ('B_x0.90', 0.90),
                    ('B_at_gain0.95_equiv', 0.95 / 0.90), ('B_at_gain1.00_equiv', 1.0 / 0.90)):
        t = car_out_thr(thr, nB, c)          # 前進側だけ c 倍(apply_gains と同じ形)
        gb1 = fw(t, dsh, G) - fw(thr, dsh, G)
        sc = fw(t, st, 1.0) / fw(thr, st, 1.0)
        gb3 = fw(t, dsh, G) - sc * fw(thr, dsh, G)
        res[name] = {'c': r(c, 4), 'gb1_us': r(gb1), 'gb3_shape_us': r(gb3), 'straight_scale': r(sc, 4)}
    return res


def main():
    a, demo_keys, P = part_a()
    b, S, pB, masks = part_b(demo_keys)
    c = part_c(S, pB, masks)
    dd = part_d(S, pB, masks)
    ee = part_e(pB, masks)
    res = {'made_by': str(Path(__file__)), 'script_sha256': sha(__file__),
           'inputs_sha256': {'drop_bfast.npz': sha(TABLES / 'drop_bfast.npz'),
                             'weights_bfast.npz': sha(TABLES / 'weights_bfast.npz'),
                             'sets_v2.npz': sha(DESK / 'sets_v2.npz'), 'desk_v2.json': sha(DESK / 'desk_v2.json')},
           'A_demo_labels': a, 'B_desk': b, 'C_row_shift': c, 'D_label_bound': dd, 'E_gate_selftest': ee}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'A': a, 'B_bounds': b['B_and_bounds'], 'C': c}, ensure_ascii=False, indent=1))
    for k in ['bfast_20261025-e15-s0', 'bfast_20261025-e15-s1', 'bfast_20261025-e15-s2', 'A_s0', 'A_s1', 'v0-s2',
              'v0-s3']:
        e = b['models'][k]
        print(k, 'gb', e['gb_dsh'], 'shape', e['gb_shape_us'], 'straight', e['straight'], 'gbref', e['gbref_dsh'],
              'demo', e.get('demo'), 'check', e.get('desk_check'))
    print('neutral shift', b['neutral_shift_effect_us'], 'demo rows', b['demo_rows_in_tab'], b['demo_keys'])
    print('D', json.dumps(dd, ensure_ascii=False))
    print('E', json.dumps(ee, ensure_ascii=False))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
