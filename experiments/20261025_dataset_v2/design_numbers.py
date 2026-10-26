#!/usr/bin/env python3
"""データセット v2 の設計 A(中身)のための数(2026-10-25)。読むだけ。

numpy だけ(torch・GPU・OpenCV は使わない)。data/ は変えない。出力は 1 つだけ:
  /home/jetson/jetformer_reports/dataset_v2_plan_20261025/design_numbers.json

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        python3 -I /home/jetson/jetformer/experiments/20261025_dataset_v2/design_numbers.py

入力:
  inventory.json(棚卸し。inventory.py が書いたもの)
  data/labels/<記録>.npz・.json(label_events のラベル)、data/raw/<記録>/<シーン>/IMU.npz・CAM_FRONT.npz(時刻だけ)
  experiments/20261015_real_accuracy の SIZE の点(evaluation.json の size)

数えるもの:
  1. 保留の分け方: 10/24 1 回目(13:46〜・昼)と 3 回目(18:31〜・夕方)を封をした保留にしたときの、学習の候補と保留の数
  2. AI のきれいな周: ラップの帯ごと(〜8.5・8.5〜9.0・9.0〜9.5・9.5〜10・10〜)の周・サンプル・軽い当たり・直線の速さ
  3. スロットルの効き k = 前進の eRPM ÷ (中立 − スロットル µs)。直線・前へ走る・スロットルが 0.33 秒ほぼ一定・衝突の前後でない
     フレーム。回ごと・人/AI ごと、clean_driving_20261011 のシーンごと。効きで正解を直したときの正解の動き [µs]
  4. ジャンルと速さ: 人のシーンの宣言したジャンルと直線の速さ(eRPM の 90% 点)、人のきれいな周のラップと速さ
  5. 大きさの見込み: SIZE の 4 点に 2 つの形(log2 の直線・べき + 床)を当てて、v2 の大きさでの L1 の見込み
  6. 介入の検証セット(intervention_val_20261011)の窓に入る、10/10 の AI 走行の記録の人の前へ走るフレームの割合
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import label_events as le  # noqa: E402  (numpy だけ。torch は読まない)

RAW = ROOT / 'data' / 'raw'
LABELS = ROOT / 'data' / 'labels'
OUT = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')
INV = OUT / 'inventory.json'
SIZE_EVAL = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/evaluation.json')
BASE_CARD = ROOT / 'data' / 'datasets' / 'clean_driving_20261011' / 'splits.json'

SEALED_SESSIONS = ('10/24 1 回目(13:46〜)', '10/24 3 回目(18:31〜)')
INTERVENTION_LOGS = ('20261010-171639', '20261010-172505', '20261010-172910', '20261010-173445', '20261010-181504',
                     '20261010-203924', '20261010-204208', '20261010-204605')   # make_intervention_set.py の AI_RUN_LOGS
LAP_BINS = ((0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 99.0))
STRAIGHT_ERPM = -1000.0
STRAIGHT_GZ = 40.0
U_MIN = 100.0                # 中立からこれ以上前へ入れている(µs)
STEADY_RANGE = 30.0          # 直前 10 フレームのスロットルの幅がこれ以下 = ほぼ一定
STEADY_N = 10
U_BINS = ((100, 200), (200, 300), (300, 400), (400, 600))
STRIDE = 3
LAG = 3                      # 当てはめの先 [フレーム](約 0.1 秒)


def fit_dyn(e0, e1, u0):
    """e1 = a·e0 + b·u0 + d → 行き着く速さ e_ss(u) = (b·u + d)/(1 − a)。u 250・350 µs の e_ss と、効き c = b/(1 − a)"""
    if len(e0) < 50:
        return None
    A = np.vstack([e0, u0, np.ones_like(e0)]).T
    (a, b, d), *_ = np.linalg.lstsq(A, e1, rcond=None)
    if not (0 < a < 0.999):
        return None
    c = b / (1 - a)
    return {'n': int(len(e0)), 'a': round(float(a), 4), 'c_erpm_per_us': round(float(c), 2),
            'u_deadband': round(float(-d / b), 1) if b != 0 else None,
            'e_ss_u250': round(float((b * 250 + d) / (1 - a)), 0), 'e_ss_u350': round(float((b * 350 + d) / (1 - a)), 0)}


def rd(x, n=1):
    if x is None:
        return None
    x = float(x)
    return round(x, n) if np.isfinite(x) else None


def pct(x, q, min_n=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) >= min_n else None


def csum(recs, key):
    out = {'frames': 0, 'samples': 0, 'seconds': 0.0}
    for r in recs:
        c = (r.get('candidates') or {}).get(key) or {}
        for k in out:
            out[k] += c.get(k, 0) or 0
    out['seconds'] = rd(out['seconds'])
    return out


# ---------------------------------------------------------------- 1. 保留の分け方
CAND_KEYS = ('a_nc_new_track', 'a_nc_new', 'a_new_track', 'a_after_takeover', 'a_other_than_takeover',
             'b_takeover_3s_drive', 'c_ai_clean_laps', 'c_ai_clean_sub9', 'd_ai_near_collision', 'ai_all', 'human_all')


def holdout_split(recs):
    use = [r for r in recs if r.get('usable')]
    groups = {
        'train_pool_10/10': [r for r in use if r['day'] == '20261010'],
        'train_pool_10/23': [r for r in use if r['day'] == '20261023'],
        'train_pool_10/24_2回目': [r for r in use if r['day'] == '20261024' and r['session'] not in SEALED_SESSIONS],
        'sealed_10/24_1回目_昼': [r for r in use if r['session'] == SEALED_SESSIONS[0]],
        'sealed_10/24_3回目_夕方': [r for r in use if r['session'] == SEALED_SESSIONS[1]],
        '10/10_intervention_logs': [r for r in use if r['log'] in INTERVENTION_LOGS],
        '10/10_ai_logs_other': [r for r in use if r['day'] == '20261010' and r.get('ai_log')
                                and r['log'] not in INTERVENTION_LOGS],
    }
    out = {}
    for g, rs in groups.items():
        out[g] = {'logs': [r['log'] for r in rs], 'n_logs': len(rs),
                  'ai_s': rd(sum(r.get('ai_s') or 0 for r in rs)), 'human_s': rd(sum(r.get('human_s') or 0 for r in rs)),
                  'takeovers': sum((r.get('takeovers') or {}).get('all', 0) for r in rs),
                  'collisions_ai': sum((r.get('collisions') or {}).get('ai', 0) for r in rs),
                  'cand': {k: csum(rs, k) for k in CAND_KEYS}}
    return out


# ---------------------------------------------------------------- 2. AI のきれいな周
def lap_tables(recs):
    laps = []
    for r in recs:
        for lp in r.get('lap_list') or []:
            x = dict(lp)
            x['log'] = r['log']
            x['day'] = r['day']
            x['session'] = r['session']
            x['ai_no'] = r.get('ai_no')
            x['battery'] = r.get('battery_note')
            laps.append(x)
    ai = [x for x in laps if x['who'] == 'ai' and x['clean']]
    hu = [x for x in laps if x['who'] == 'human' and x['clean']]

    def table(rows, key=None):
        t = []
        for lo, hi in LAP_BINS:
            w = [x for x in rows if lo <= x['lap_s'] < hi]
            t.append({'bin': '%.1f〜%.1f' % (lo, hi) if hi < 50 else '%.1f〜' % lo, 'laps': len(w),
                      'samples': sum(x['samples'] for x in w), 'frames': sum(x['frames'] for x in w),
                      'laps_with_bump': sum(1 for x in w if x.get('bumps')),
                      'samples_no_bump': sum(x['samples'] for x in w if not x.get('bumps')),
                      'straight_p90_median': rd(pct([x['straight_erpm_p90'] for x in w if x.get('straight_erpm_p90')], 50), 0),
                      'by_session': {s: sum(1 for x in w if x['session'] == s) for s in sorted({x['session'] for x in w})},
                      'by_model': {m: sum(1 for x in w if x['model'] == m) for m in sorted({str(x['model']) for x in w})}})
        return t
    sub9 = sorted([x for x in ai if x['lap_s'] < 9.0], key=lambda x: x['lap_s'])
    sub9_rows = [{'log': x['log'], 'ai_no': x['ai_no'], 'session': x['session'], 'lap_s': x['lap_s'], 'samples': x['samples'],
                  'bumps': x['bumps'], 'model': x['model'], 'throttle_gain': x.get('throttle_gain'),
                  'steer_gain': x.get('steer_gain'), 'straight_p90': x.get('straight_erpm_p90')} for x in sub9]
    # ラップと直線の速さの相関(AI のきれいな周・B)
    b = [x for x in ai if x['model'] == 'B' and x.get('straight_erpm_p90')]
    corr = None
    if len(b) >= 5:
        corr = rd(np.corrcoef([x['lap_s'] for x in b], [x['straight_erpm_p90'] for x in b])[0, 1], 3)
    hcorr = None
    hb = [x for x in hu if x.get('straight_erpm_p90')]
    if len(hb) >= 5:
        hcorr = rd(np.corrcoef([x['lap_s'] for x in hb], [x['straight_erpm_p90'] for x in hb])[0, 1], 3)
    return {'ai_clean_by_lap_bin': table(ai), 'human_clean_by_lap_bin': table(hu), 'ai_sub9_laps': sub9_rows,
            'corr_lap_vs_straight_ai_B': corr, 'n_ai_B': len(b), 'corr_lap_vs_straight_human': hcorr, 'n_human': len(hb),
            'ai_clean_total': {'laps': len(ai), 'samples': sum(x['samples'] for x in ai)},
            'ai_clean_B_sealed_vs_train': {
                g: {'laps': len(w), 'samples': sum(x['samples'] for x in w),
                    'sub9_laps': sum(1 for x in w if x['lap_s'] < 9.0),
                    'sub9_samples': sum(x['samples'] for x in w if x['lap_s'] < 9.0),
                    'lt95_samples': sum(x['samples'] for x in w if x['lap_s'] < 9.5)}
                for g, w in (('sealed', [x for x in ai if x['session'] in SEALED_SESSIONS]),
                             ('train_pool', [x for x in ai if x['session'] not in SEALED_SESSIONS]))}}


# ---------------------------------------------------------------- 3. スロットルの効き
def scene_spans(log):
    root = RAW / log
    out = []
    for d in sorted(x for x in root.iterdir() if x.is_dir() and x.name.startswith(log + '-')):
        try:
            with np.load(d / 'CAM_FRONT.npz', allow_pickle=False) as z:
                ts = z['timestamps_us'].astype(np.int64)
            out.append((d.name, int(ts[0]), int(ts[-1])) if len(ts) else (d.name, None, None))
        except Exception:
            out.append((d.name, None, None))
    return out


def effectiveness(recs, base_split):
    rows = []      # フレームごとの (log, session, who, scene, k, u, erpm)
    dyn = []       # 1 次の遅れの当てはめ用
    straight_rows = []   # 直線のフレームの (u, eRPM)
    per_scene_u = {}
    for r in recs:
        if not r.get('usable'):
            continue
        log = r['log']
        try:
            with np.load(LABELS / (log + '.npz'), allow_pickle=False) as z:
                lab = {k: z[k] for k in ('t_s', 't_us', 'ai', 'dry_run', 'erpm', 'valid_action', 'near_collision',
                                         'action_exec')}
            info = json.load(open(LABELS / (log + '.json'), encoding='utf-8'))
        except Exception:
            continue
        L = le.load_log(log)
        I = L['IMU']
        if I is None or len(I['timestamps_us']) < 20:
            continue
        t0 = info['t0_us']
        it = (I['timestamps_us'].astype(np.int64) - t0) / 1e6
        gz = I['gyro'][:, 2].astype(np.float64)
        V = L['VEHICLE']
        vt = (V['timestamps_us'].astype(np.int64) - t0) / 1e6
        still = (np.abs(np.interp(it, vt, V['erpm'].astype(np.float64))) < 50) & (np.abs(gz) < 5)
        _, gzu = le.heading_deg(it, gz, still)
        gzs = np.convolve(gzu, np.ones(11) / 11, 'same') if len(gzu) >= 11 else gzu
        t = lab['t_s'].astype(np.float64)
        gzf = np.interp(t, it, gzs)
        ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)
        hum = ~lab['ai'].astype(bool)
        neutral_ai = float((info.get('ref_us') or {}).get('throttle_neutral') or 1500.0)
        neutral_h = float(r.get('human_throttle_neutral_us') or 1500.0)
        thr = lab['action_exec'][:, 1].astype(np.float64)
        u = np.where(ai, neutral_ai - thr, neutral_h - thr)
        n = len(u)
        # 直前 STEADY_N フレームのスロットルの幅
        rng = np.full(n, np.inf)
        if n > STEADY_N:
            from numpy.lib.stride_tricks import sliding_window_view
            w = sliding_window_view(u, STEADY_N)
            rng[STEADY_N - 1:] = np.nanmax(w, 1) - np.nanmin(w, 1)
        erpm = lab['erpm'].astype(np.float64)
        ok = (lab['valid_action'].astype(bool) & (erpm < STRAIGHT_ERPM) & (np.abs(gzf) < STRAIGHT_GZ) & (u >= U_MIN)
              & (rng <= STEADY_RANGE) & ~lab['near_collision'].astype(bool) & np.isfinite(u))
        spans = scene_spans(log)
        t_us = lab['t_us'].astype(np.int64)
        scene = np.array([''] * n, dtype=object)
        for name, a, b in spans:
            if a is not None:
                scene[(t_us >= a - 50000) & (t_us <= b + 50000)] = name
        ok &= scene != ''
        k = -erpm / np.maximum(u, 1.0)
        # 1 次の遅れの当てはめ用: e[t+LAG] = a·e[t] + b·u[t] + d(直線・前へ・同じシーン・衝突の前後でない・0.2 秒以内)
        e = -erpm
        base = (lab['valid_action'].astype(bool) & (e > 500) & (np.abs(gzf) < STRAIGHT_GZ) & (u > -50) & (u < 600)
                & ~lab['near_collision'].astype(bool) & np.isfinite(u) & (scene != ''))
        i0 = np.nonzero(base[:-LAG])[0]
        i1 = i0 + LAG
        good = base[i1] & (scene[i0] == scene[i1]) & ((t[i1] - t[i0]) < 0.2) & (ai[i0] == ai[i1])
        i0, i1 = i0[good], i1[good]
        st_all = (lab['valid_action'].astype(bool) & (erpm < STRAIGHT_ERPM) & (np.abs(gzf) < STRAIGHT_GZ)
                  & ~lab['near_collision'].astype(bool) & np.isfinite(u) & (scene != ''))
        for who, m in (('ai', ai), ('human', hum)):
            sm = st_all & m
            if sm.sum() >= 30:
                straight_rows.append({'log': log, 'session': r['session'], 'who': who, 'u': u[sm], 'e': -erpm[sm],
                                      'model': (r.get('models') or [''])[0] if who == 'ai' else '人',
                                      'tg': (r.get('throttle_gain') or {}).get('median') if who == 'ai' else None})
        for who, m in (('ai', ai), ('human', hum)):
            mm = m[i0]
            if mm.sum() >= 50:
                dyn.append({'log': log, 'session': r['session'], 'who': who, 'scene': scene[i0][mm],
                            'e0': e[i0][mm], 'e1': e[i1][mm], 'u0': u[i0][mm],
                            'model': (r.get('models') or [''])[0] if who == 'ai' else '人',
                            'tg': (r.get('throttle_gain') or {}).get('median') if who == 'ai' else None})
        for who, m in (('ai', ai), ('human', hum)):
            sel = ok & m
            if sel.any():
                rows.append({'log': log, 'session': r['session'], 'who': who, 'band': r['band'],
                             'k': k[sel], 'u': u[sel], 'erpm': -erpm[sel], 'scene': scene[sel],
                             'model': (r.get('models') or [''])[0] if who == 'ai' else '人',
                             'tg': (r.get('throttle_gain') or {}).get('median') if who == 'ai' else None})
        # 人の (a) に近いフレーム(前へ・スロットル前進側)の u をシーンごとに(正解の動きの見積もり用。間引き 3)
        fwd_h = hum & lab['valid_action'].astype(bool) & (erpm < -200) & (u > 0) & (scene != '')
        idx = np.arange(n)
        for s in np.unique(scene[fwd_h]):
            mm = fwd_h & (scene == s) & (idx % STRIDE == 0)
            per_scene_u[s] = u[mm]
        del L
        print('効き', log, int(ok.sum()), flush=True)

    def summ(sel_rows):
        if not sel_rows:
            return None
        k = np.concatenate([x['k'] for x in sel_rows])
        uu = np.concatenate([x['u'] for x in sel_rows])
        ee = np.concatenate([x['erpm'] for x in sel_rows])
        res = {'n': int(len(k)), 'k_p25': rd(pct(k, 25), 2), 'k_p50': rd(pct(k, 50), 2), 'k_p75': rd(pct(k, 75), 2)}
        for lo, hi in U_BINS:
            m = (uu >= lo) & (uu < hi)
            res['erpm_p50_u%d_%d' % (lo, hi)] = rd(pct(ee[m], 50, 20), 0)
            res['n_u%d_%d' % (lo, hi)] = int(m.sum())
        return res

    by_session = {}
    for s in sorted({x['session'] for x in rows}):
        for who in ('human', 'ai'):
            v = summ([x for x in rows if x['session'] == s and x['who'] == who])
            if v:
                by_session['%s|%s' % (s, who)] = v
    # 同じモデル B・同じ速さのゲインで比べる(AI)
    by_model_gain = {}
    for x in rows:
        if x['who'] != 'ai':
            continue
        key = '%s|%s|%s' % (x['session'], x['model'], x['tg'])
        by_model_gain.setdefault(key, []).append(x)
    by_model_gain = {k: summ(v) for k, v in by_model_gain.items()}
    by_model_gain = {k: v for k, v in by_model_gain.items() if v and v['n'] >= 30}

    # clean_driving のシーンごと(人)
    train = set(base_split['train'])
    val = set(base_split['val'])
    scene_k = {}
    for x in rows:
        if x['who'] != 'human':
            continue
        for s in np.unique(x['scene']):
            m = x['scene'] == s
            scene_k.setdefault(s, []).append(x['k'][m])
    scene_tab = []
    for s, ks in sorted(scene_k.items()):
        kk = np.concatenate(ks)
        scene_tab.append({'scene': s, 'split': 'train' if s in train else ('val' if s in val else 'new'), 'n': int(len(kk)),
                          'k_p50': rd(pct(kk, 50), 2)})
    ref_rows = [x for x in rows if x['who'] == 'human' and any(sc in train for sc in np.unique(x['scene']))]
    kref_all = np.concatenate([x['k'][np.isin(x['scene'], list(train))] for x in ref_rows]) if ref_rows else np.zeros(0)
    k_ref = pct(kref_all, 50)
    # 効きで正解を直したときの動き: Δ = u × (k_scene / k_ref − 1)(前へのスロットルを、基準の効きのときに同じ速さになる量に)
    shift = {}
    for split in ('train', 'val', 'new'):
        d_all = []
        for row in scene_tab:
            if row['split'] != split or row['n'] < 30 or row['scene'] not in per_scene_u or k_ref is None:
                continue
            uu = per_scene_u[row['scene']]
            d_all.append(np.abs(uu * (row['k_p50'] / k_ref - 1.0)))
        if d_all:
            d = np.concatenate(d_all)
            shift[split] = {'n_samples': int(len(d)), 'abs_p50': rd(pct(d, 50)), 'abs_mean': rd(np.mean(d)),
                            'abs_p90': rd(pct(d, 90))}
    ks = [x['k_p50'] for x in scene_tab if x['split'] in ('train', 'val') and x['n'] >= 30]

    def dfit(sel):
        if not sel:
            return None
        return fit_dyn(np.concatenate([x['e0'] for x in sel]), np.concatenate([x['e1'] for x in sel]),
                       np.concatenate([x['u0'] for x in sel]))
    def ssum(sel):
        uu = np.concatenate([x['u'] for x in sel])
        ee = np.concatenate([x['e'] for x in sel])
        hi = uu >= 300
        return {'n': int(len(uu)), 'u_p50': rd(pct(uu, 50), 0), 'u_p90': rd(pct(uu, 90), 0),
                'erpm_p50': rd(pct(ee, 50), 0), 'erpm_p90': rd(pct(ee, 90), 0),
                'n_u_ge300': int(hi.sum()), 'erpm_p50_u_ge300': rd(pct(ee[hi], 50, 20), 0),
                'erpm_p90_u_ge300': rd(pct(ee[hi], 90, 20), 0)}
    straight_tab = {}
    for key in sorted({'%s|%s|%s|%s' % (x['session'], x['who'], x['model'], x['tg']) for x in straight_rows}):
        sel = [x for x in straight_rows if '%s|%s|%s|%s' % (x['session'], x['who'], x['model'], x['tg']) == key]
        v = ssum(sel)
        if v['n'] >= 100:
            straight_tab[key] = v
    straight_log_h = {}
    for lg in sorted({x['log'] for x in straight_rows if x['who'] == 'human'}):
        v = ssum([x for x in straight_rows if x['who'] == 'human' and x['log'] == lg])
        if v['n'] >= 100:
            straight_log_h[lg] = v
    dyn_session = {}
    for s_ in sorted({x['session'] for x in dyn}):
        for who in ('human', 'ai'):
            v = dfit([x for x in dyn if x['session'] == s_ and x['who'] == who])
            if v:
                dyn_session['%s|%s' % (s_, who)] = v
    dyn_model = {}
    for key in sorted({'%s|%s|%s' % (x['session'], x['model'], x['tg']) for x in dyn if x['who'] == 'ai'}):
        v = dfit([x for x in dyn if x['who'] == 'ai' and '%s|%s|%s' % (x['session'], x['model'], x['tg']) == key])
        if v and v['n'] >= 150:
            dyn_model[key] = v
    # clean_driving の記録ごと(人。シーンだけだと少ないので記録ごと)
    dyn_base_log = {}
    for lg in sorted({x['log'] for x in dyn if x['who'] == 'human'}):
        sel = [x for x in dyn if x['who'] == 'human' and x['log'] == lg]
        e0 = np.concatenate([x['e0'] for x in sel]); e1 = np.concatenate([x['e1'] for x in sel])
        u0 = np.concatenate([x['u0'] for x in sel]); sc = np.concatenate([x['scene'] for x in sel])
        for name, m in (('train', np.isin(sc, list(train))), ('val', np.isin(sc, list(val))),
                        ('new', ~np.isin(sc, list(train | val)))):
            v = fit_dyn(e0[m], e1[m], u0[m])
            if v and v['n'] >= 150:
                dyn_base_log['%s|%s' % (lg, name)] = v
    dyn_val_battery = fit_dyn(*[np.concatenate([x[k][x['scene'] == '20261010-193638-0003'] for x in dyn
                                                 if x['who'] == 'human' and x['log'] == '20261010-193638'])
                                for k in ('e0', 'e1', 'u0')]) if any(x['log'] == '20261010-193638' for x in dyn) else None
    return {'rule': '効き k = 前進の eRPM ÷ (中立 − スロットル µs)。直線(eRPM < −1000・|ヨー| < 40 deg/s)・前へ %d µs 以上・'
                    '直前 %d フレームのスロットルの幅 ≤ %d µs・衝突の前後でない。人はスティック、AI は出した値(out)'
                    % (U_MIN, STEADY_N, STEADY_RANGE),
            'by_session': by_session, 'ai_by_session_model_gain': by_model_gain, 'base_scenes': scene_tab,
            'k_ref_base_train_human': rd(k_ref, 2),
            'base_scene_k_spread': {'n_scenes': len(ks), 'min': rd(min(ks), 2) if ks else None,
                                    'p10': rd(pct(ks, 10), 2), 'p50': rd(pct(ks, 50), 2), 'p90': rd(pct(ks, 90), 2),
                                    'max': rd(max(ks), 2) if ks else None},
            'label_shift_if_normalized_us': shift,
            'dyn_rule': '1 次の遅れ e[t+%d] = a·e[t] + b·u[t] + d(e = 前進の eRPM、u = 中立 − スロットル µs)。直線・前へ(e > 500)・'
                        '衝突の前後でない・同じシーン・同じ操作者。行き着く速さ e_ss(u) = (b·u + d)/(1 − a)' % LAG,
            'straight_rule': '直線(eRPM < −1000・|ヨー| < 40 deg/s)・操作値が有効・衝突の前後でないフレームの u(中立 − スロットル µs)と前進の eRPM',
            'straight_by_session_who_model_gain': straight_tab, 'straight_by_log_human': straight_log_h,
            'dyn_by_session': dyn_session, 'dyn_ai_by_session_model_gain': dyn_model, 'dyn_by_log_human': dyn_base_log,
            'dyn_val_battery_scene_193638_0003': dyn_val_battery}


# ---------------------------------------------------------------- 4. ジャンルと速さ
def genre_speed(recs, base_split):
    train = set(base_split['train'])
    val = set(base_split['val'])
    out = []
    for r in recs:
        if not r.get('usable') or r.get('ai_log'):
            continue
        for s in r.get('scenes') or []:
            p = s.get('straight_erpm_p90_human')
            out.append({'scene': s['scene'], 'genre': '・'.join(s.get('genres') or []), 'band': s.get('band'),
                        'split': 'train' if s['scene'] in train else ('val' if s['scene'] in val else 'new'),
                        'straight_p90': p, 'a_samples': (s.get('candidates') or {}).get('a', {}).get('samples'),
                        'human_clean_laps': s.get('laps_human_clean_s')})
    summ = {}
    for g in sorted({x['genre'] for x in out}):
        for sp in ('train', 'val', 'new'):
            w = [x for x in out if x['genre'] == g and x['split'] == sp and x['straight_p90']]
            if w:
                v = [x['straight_p90'] for x in w]
                summ['%s|%s' % (g, sp)] = {'scenes': len(w), 'p90_min': min(v), 'p90_median': rd(np.median(v), 0),
                                           'p90_max': max(v), 'a_samples': sum(x['a_samples'] or 0 for x in w)}
    # 速さの帯で付け直したら(学習 41 シーン): 帯の境目 6,500・7,500 eRPM
    bands = {'〜6,500': (0, 6500), '6,500〜7,500': (6500, 7500), '7,500〜': (7500, 1e9)}
    rebanded = {}
    for name, (lo, hi) in bands.items():
        w = [x for x in out if x['split'] == 'train' and x['straight_p90'] and lo <= x['straight_p90'] < hi]
        rebanded[name] = {'scenes': len(w), 'a_samples': sum(x['a_samples'] or 0 for x in w),
                          'genres': {g: sum(1 for x in w if x['genre'] == g) for g in sorted({x['genre'] for x in w})}}
    return {'scenes': out, 'by_genre_split': summ, 'rebanded_train': rebanded}


# ---------------------------------------------------------------- 5. 大きさの見込み
def size_curve():
    ev = json.load(open(SIZE_EVAL, encoding='utf-8'))['size']
    N = ev['N']
    L = ev['by_metric']['primary/best']['L']
    pts = [(N[k], L[k]) for k in ('Q', 'H', 'T', 'full')]
    x = np.log2([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    # 最後の 3 点(H・T・全部)だけの直線(曲線が平らになりつつあるかを見る)
    (a3, b3), *_ = np.linalg.lstsq(A[1:], y[1:], rcond=None)
    # べき + 床: L = Linf + c·N^(−α)。床を 0〜45 で探して 2 乗誤差が最小のもの、床 31(人の揺れ)に固定したもの
    def fit_pow(linf):
        yy = np.log(np.maximum(y - linf, 1e-6))
        Ap = np.vstack([np.ones_like(x), np.log(2 ** x)]).T
        (lc, mal), *_ = np.linalg.lstsq(Ap, yy, rcond=None)
        pred = linf + np.exp(lc) * (2 ** x) ** mal
        return float(np.sum((pred - y) ** 2)), float(np.exp(lc)), float(-mal)
    best = None
    for linf in np.arange(0.0, 48.0, 0.25):
        e, c, al = fit_pow(linf)
        if best is None or e < best[0]:
            best = (e, linf, c, al)
    e31, c31, al31 = fit_pow(31.0)

    def pred(n):
        return {'loglin_all4': rd(a + b * np.log2(n), 2), 'loglin_last3': rd(a3 + b3 * np.log2(n), 2),
                'pow_free': rd(best[1] + best[2] * n ** (-best[3]), 2), 'pow_floor31': rd(31.0 + c31 * n ** (-al31), 2)}
    return {'points': [{'N': p[0], 'L': rd(p[1], 3)} for p in pts],
            'fits': {'loglin_all4': {'a': rd(a, 3), 'slope_per_doubling': rd(b, 3)},
                     'loglin_last3': {'a': rd(a3, 3), 'slope_per_doubling': rd(b3, 3)},
                     'pow_free': {'Linf': rd(best[1], 2), 'c': rd(best[2], 2), 'alpha': rd(best[3], 3), 'sse': rd(best[0], 4)},
                     'pow_floor31': {'Linf': 31.0, 'c': rd(c31, 2), 'alpha': rd(al31, 3), 'sse': rd(e31, 4)}},
            'pred': {str(n): pred(n) for n in (5912, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 20000)}}


# ---------------------------------------------------------------- 6. 介入の検証セットとの重なり
def intervention_overlap(recs):
    """10/10 の介入の検証セット(intervention_val_20261011)の窓(交代 → 最大 5 秒)に入る、人の前へ走るフレームの割合。
    人の前へ = CH3 が人側・操作値が有効・eRPM < −200(棚卸しの (a) の近い形。押し・持ち上げは外していない)"""
    card = json.load(open(ROOT / 'data' / 'datasets' / 'intervention_val_20261011' / 'dataset_card.json', encoding='utf-8'))
    iv = card['intervention']
    out = {'max_s': iv.get('max_s'), 'windows': 0, 'by_log': {}}
    tot_in = tot_all = 0
    for log in INTERVENTION_LOGS:
        wins = [w for w in iv['windows'] if w['log'] == log and w.get('included')]
        try:
            with np.load(LABELS / (log + '.npz'), allow_pickle=False) as z:
                t = z['t_s'].astype(np.float64)
                hum = ~z['ai'].astype(bool)
                fwd = hum & z['valid_action'].astype(bool) & (z['erpm'] < -200)
        except Exception:
            continue
        samp = (np.arange(len(t)) % STRIDE) == 0
        inw = np.zeros(len(t), bool)
        for w in wins:
            inw |= (t >= w['t_s']) & (t < w['t_s'] + float(iv.get('max_s') or 5.0))
        a_all = int((fwd & samp).sum())
        a_in = int((fwd & samp & inw).sum())
        out['by_log'][log] = {'windows': len(wins), 'fwd_samples': a_all, 'fwd_samples_in_windows': a_in}
        out['windows'] += len(wins)
        tot_in += a_in
        tot_all += a_all
    out['fwd_samples'] = tot_all
    out['fwd_samples_in_windows'] = tot_in
    out['share_in_windows'] = rd(tot_in / tot_all, 3) if tot_all else None
    return out


# ---------------------------------------------------------------- 7. 組み合わせ(レシピ)の大きさと L1 の見込み
def recipes(res, recs):
    hs = res['holdout_split']
    ov = res['intervention_overlap']['share_in_windows'] or 0.0
    base_build = 5912                     # 今の学習(build_dataset の数え方・衝突の前後を除く)
    base_a = inv_check = 5652             # 同じ 41 シーンを (a) の数え方で(棚卸しの checks)
    scale = base_build / base_a           # (a) の数え方 → build の数え方
    g = lambda k, key: hs[k]['cand'][key]['samples']          # noqa: E731
    human_new = {
        '10/10 AI 走行の記録(介入の窓を除く)': round(g('10/10_intervention_logs', 'a_nc_new_track') * (1 - ov)),
        '10/10 AI 走行の記録(介入の検証の外の 3 本)': g('10/10_ai_logs_other', 'a_nc_new_track'),
        '10/23': g('train_pool_10/23', 'a_nc_new_track'),
        '10/24 2 回目': g('train_pool_10/24_2回目', 'a_nc_new_track'),
    }
    human_sealed = g('sealed_10/24_1回目_昼', 'a_nc_new_track') + g('sealed_10/24_3回目_夕方', 'a_nc_new_track')
    # AI のきれいな周(学習に使える回の B だけ)
    laps = []
    for r in recs:
        for lp in r.get('lap_list') or []:
            if lp['who'] == 'ai' and lp['clean']:
                laps.append(dict(lp, session=r['session']))
    pool = [x for x in laps if x['session'] not in SEALED_SESSIONS]
    sealed = [x for x in laps if x['session'] in SEALED_SESSIONS]

    def lsum(rows, lo, hi, model=None):
        w = [x for x in rows if lo <= x['lap_s'] < hi and (model is None or x['model'] == model)]
        return {'laps': len(w), 'samples': sum(x['samples'] for x in w)}
    ai_pool = {'B <9.0': lsum(pool, 0, 9.0, 'B'), 'B 9.0〜9.5': lsum(pool, 9.0, 9.5, 'B'),
               'B 9.5〜10.0': lsum(pool, 9.5, 10.0, 'B'), 'ほかのモデル <9.5': {
                   'laps': lsum(pool, 0, 9.5)['laps'] - lsum(pool, 0, 9.5, 'B')['laps'],
                   'samples': lsum(pool, 0, 9.5)['samples'] - lsum(pool, 0, 9.5, 'B')['samples']}}
    ai_sealed = {'全部': lsum(sealed, 0, 99), '<9.0': lsum(sealed, 0, 9.0), '9.0〜9.5': lsum(sealed, 9.0, 9.5),
                 'B 全部': lsum(sealed, 0, 99, 'B')}
    hn = sum(human_new.values())
    fits = res['size_curve']['fits']

    def proj(n):
        a3, b3 = fits['loglin_last3']['a'], fits['loglin_last3']['slope_per_doubling']
        pf, p31 = fits['pow_free'], fits['pow_floor31']
        vals = [a3 + b3 * np.log2(n), pf['Linf'] + pf['c'] * n ** (-pf['alpha']), 31.0 + p31['c'] * n ** (-p31['alpha'])]
        return {'loglin_last3': rd(vals[0], 2), 'pow_free': rd(vals[1], 2), 'pow_floor31': rd(vals[2], 2),
                'range': [rd(min(vals), 1), rd(max(vals), 1)]}
    b90 = ai_pool['B <9.0']['samples']
    b95 = ai_pool['B 9.0〜9.5']['samples']
    rec_tab = {}
    for name, human_extra, ai_n, ai_w in (
            ('今(B と同じ)', 0, 0, 0.0),
            ('人だけ(封をしたうえで)', hn, 0, 0.0),
            ('主: 人 + AI の B <9.0(重み 1)+ 9.0〜9.5(重み 0.5)', hn, b90 + b95, None),
            ('封をしない場合の人だけ', hn + human_sealed, 0, 0.0)):
        n_build = base_build + round(human_extra * scale)
        if ai_w is None:
            ai_eff = b90 + 0.5 * b95
        else:
            ai_eff = ai_w * ai_n
        rec_tab[name] = {'human_samples': n_build, 'ai_samples': ai_n, 'total_samples': n_build + ai_n,
                         'x_of_now': rd((n_build + ai_n) / base_build, 2),
                         'l1_if_ai_counts_0': proj(n_build), 'l1_if_ai_counts_half': proj(n_build + 0.5 * ai_eff),
                         'l1_if_ai_counts_full': proj(n_build + ai_eff) if ai_n else None}
    need_2x = 2 * base_build - (base_build + round(hn * scale))
    return {'scale_a_to_build': rd(scale, 3), 'intervention_share': ov, 'human_new_pool': human_new,
            'human_new_pool_total': hn, 'human_sealed': human_sealed, 'ai_pool': ai_pool, 'ai_sealed': ai_sealed,
            'recipes': rec_tab, 'human_needed_for_2x_build_samples': need_2x,
            'human_needed_for_2x_minutes_forward': rd(need_2x / scale * STRIDE / 30.0 / 60.0, 1)}


def main():
    inv = json.load(open(INV, encoding='utf-8'))
    recs = inv['records']
    base_split = json.load(open(BASE_CARD, encoding='utf-8'))
    res = {'schema': 'jf-dataset-v2-design-numbers/v1', 'tool': 'experiments/20261025_dataset_v2/design_numbers.py',
           'sealed_sessions': SEALED_SESSIONS}
    res['holdout_split'] = holdout_split(recs)
    res['laps'] = lap_tables(recs)
    res['genre_speed'] = genre_speed(recs, base_split)
    res['size_curve'] = size_curve()
    res['intervention_overlap'] = intervention_overlap(recs)
    res['recipes'] = recipes(res, recs)
    res['effectiveness'] = effectiveness(recs, base_split)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(OUT / 'design_numbers.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=default)
    print('書いた:', OUT / 'design_numbers.json')


if __name__ == '__main__':
    main()
