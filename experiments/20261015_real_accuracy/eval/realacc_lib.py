#!/usr/bin/env python3
"""real_accuracy_20261015 の評価の計算(numpy だけ。テストする)。protocol.json の式をそのまま書いたもの。

  - judge・compare: tools/sim_cotrain.py の judge_primary・final_verdict(10/13 の規則)を best・last の両方に当てる
  - gain_correct: GAIN の後処理(ideas_data/probe_gain.py の correct と同じ式)
  - prev_index・temporal_ensemble: TE(jetformer/policies/ensemble.py の TemporalEnsembler と同じ計算、j = 2・m = 0)
  - size_reading: SIZE の読み(analysis.readings)
"""

import numpy as np

SPREAD_MULT = 2.0
PRACTICAL_US = 10.0
CKPTS = ('best', 'last')
NEUTRAL_US = 1498.0


# ---------------------------------------------------------------- 判定(10/13 の規則)

def judge_primary(arm_vals, ref_vals, spread_mult=SPREAD_MULT, spread_floor=0.0, practical_us=PRACTICAL_US):
    """tools/sim_cotrain.py の judge_primary と同じ(写し。テストで一致を確かめる)"""
    p, r = np.asarray(arm_vals, np.float64), np.asarray(ref_vals, np.float64)
    spread = max(float(r.max() - r.min()), float(spread_floor))
    thr = spread_mult * spread
    gain = float(r.mean() - p.mean())
    if p.max() < r.min() and gain > thr:
        cls = 'better'
    elif p.min() > r.max() and -gain > thr:
        cls = 'worse'
    else:
        cls = 'equivalent'
    return {'class': cls, 'gain_us': gain, 'threshold_us': thr, 'ref_spread_us': spread, 'practical': abs(gain) >= practical_us,
            'arm_mean': float(p.mean()), 'arm_min': float(p.min()), 'arm_max': float(p.max()),
            'ref_mean': float(r.mean()), 'ref_min': float(r.min()), 'ref_max': float(r.max())}


def final_verdict(best, last):
    """「良い」は best・last の両方で better のときだけ。「悪い」は best のまま(10/13)"""
    if best['class'] == 'better':
        return 'better' if last['class'] == 'better' else 'equivalent'
    return best['class']


def compare(arm_vals, ref_vals, s_floor):
    """arm_vals・ref_vals = {'best': [種], 'last': [種]}、s_floor = {'best': s_A, 'last': s_A(last)} → best・last・最終"""
    out = {ck: judge_primary(arm_vals[ck], ref_vals[ck], SPREAD_MULT, s_floor[ck], PRACTICAL_US) for ck in CKPTS}
    out['final_class'] = final_verdict(out['best'], out['last'])
    out['practical'] = bool(out['best']['practical'])
    out['paired_best_us'] = [float(a - b) for a, b in zip(arm_vals['best'], ref_vals['best'])]
    out['paired_last_us'] = [float(a - b) for a, b in zip(arm_vals['last'], ref_vals['last'])]
    out['arm_best'] = [float(x) for x in arm_vals['best']]
    out['arm_last'] = [float(x) for x in arm_vals['last']]
    out['ref_best'] = [float(x) for x in ref_vals['best']]
    out['ref_last'] = [float(x) for x in ref_vals['last']]
    return out


# ---------------------------------------------------------------- 主指標・副指標

def l1_rows(pred, y):
    """[N,10,2] µs → 行ごとの L1(10 ステップ × 2 の平均)。policy_zoo.l1_rows と同じ"""
    return np.abs(np.asarray(pred, np.float64) - np.nan_to_num(np.asarray(y, np.float64))).mean(axis=(1, 2))


def metric_values(pred, y, sets, k3_mask):
    """protocol_baseline_secondary.py と同じ計算 → {集合の名前: L1, 'S2_k3_steer', 'k3_throttle'}。
    sets = {名前: 行の mask}、k3_mask = 主指標の行(3 フレーム先 = index 2)"""
    d = np.abs(np.asarray(pred, np.float64) - np.nan_to_num(np.asarray(y, np.float64)))
    row = d.mean(axis=(1, 2))
    out = {name: float(row[m].mean()) for name, m in sets.items()}
    out['S2_k3_steer'] = float(d[k3_mask][:, 2, 0].mean())
    out['k3_throttle'] = float(d[k3_mask][:, 2, 1].mean())
    return out


# ---------------------------------------------------------------- GAIN(後処理)

def gain_scale(gt, gref_row, lo=0.8, hi=2.0):
    """行ごとの掛け算の比 clip(g_t / g_ref, lo, hi)。g_t・g_ref が無い行は 1(補正なし)"""
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.clip(np.asarray(gt, np.float64) / np.asarray(gref_row, np.float64), lo, hi)
    return np.where(np.isfinite(s), s, 1.0)


def gain_correct(pred, scale, neutral=NEUTRAL_US):
    """probe_gain.py の correct: 予測のスロットル < 中立 のステップだけ 中立 − (中立 − p) × scale。ステアは変えない"""
    q = np.array(pred, np.float64, copy=True)
    d = neutral - q[:, :, 1]
    q[:, :, 1] = np.where(d > 0, neutral - d * np.asarray(scale, np.float64)[:, None], q[:, :, 1])
    return q


def gain_label(y, scale, neutral=NEUTRAL_US):
    """GAINL のラベル: スロットルのラベルのうち < 中立 を 中立 − (中立 − y) / r。gain_correct の逆(r = 1 なら元のまま)"""
    q = np.array(y, np.float64, copy=True)
    d = neutral - q[..., 1]
    r = np.asarray(scale, np.float64)
    r = r.reshape(r.shape + (1,) * (d.ndim - r.ndim))
    q[..., 1] = np.where(d > 0, neutral - d / r, q[..., 1])
    return q


# ---------------------------------------------------------------- TE(時間方向の平均)

def prev_index(part, scene, frame, j_max=2):
    """表の行ごとに、同じ part・シーンの d フレーム前(d = 0..j_max)の行の番号 [N, j_max+1](無ければ −1)"""
    key = {(int(p), str(s), int(f)): k for k, (p, s, f) in enumerate(zip(part, scene, frame))}
    out = np.full((len(part), j_max + 1), -1, np.int64)
    for k, (p, s, f) in enumerate(zip(part, scene, frame)):
        for d in range(j_max + 1):
            out[k, d] = key.get((int(p), str(s), int(f) - d), -1)
    return out


def temporal_ensemble(pred, prev, m=0.0):
    """TemporalEnsembler(j_max = prev の列数 − 1、重み exp(−m·j))と同じ: 出力の行 s = Σ_j w_j·p_{t−j}[s+j] / Σ w_j
    (s + j ≤ 9、前の行が無い j は飛ばす)"""
    pred = np.asarray(pred, np.float64)
    n, h, _ = pred.shape
    num = np.zeros_like(pred)
    den = np.zeros((n, h, 1))
    for j in range(prev.shape[1]):
        r = prev[:, j]
        ok = r >= 0
        w = float(np.exp(-m * j))
        if j >= h:
            break
        num[ok, :h - j] += w * pred[r[ok], j:]
        den[ok, :h - j] += w
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------- SIZE

def size_slopes(LQ, LH, Lfull, NQ, NH, Nfull):
    """s1 = (L(Q) − L(H)) / log2(N_H/N_Q)、s2 = (L(H) − L(全部)) / log2(N_全部/N_H)、
    G2_lin = s2、G2_pow = s2 × clip(s2/s1, 0, 1)(s1 ≤ 0 なら 0)"""
    s1 = (float(LQ) - float(LH)) / np.log2(float(NH) / float(NQ))
    s2 = (float(LH) - float(Lfull)) / np.log2(float(Nfull) / float(NH))
    g_pow = 0.0 if s1 <= 0 else s2 * float(np.clip(s2 / s1, 0.0, 1.0))
    return {'s1': float(s1), 's2': float(s2), 'G2_lin': float(s2), 'G2_pow': float(g_pow),
            'LH_minus_Lfull': float(LH) - float(Lfull)}


def size_reading(sl, low_thr=0.626):
    """analysis.readings(low_value を先に調べる)→ 'low_value' | 'high_value' | 'middle'"""
    if sl['LH_minus_Lfull'] <= low_thr or sl['G2_lin'] < 1.0:
        return 'low_value'
    if sl['G2_pow'] >= 2.0:
        return 'high_value'
    return 'middle'


SIZE_TEXT = {
    'low_value': '2 倍でも 1 µs 未満の見込み。精度のために録画を増やす価値は低い',
    'high_value': '2 倍で 2 µs 以上の見込み。きれいな周回の録画を勧める(電池の残りも記録)',
    'middle': '1〜2 µs の見込み(中くらい)',
}


def scene_bootstrap(per_scene_sum, per_scene_n, n_boot=2000, seed=0):
    """シーン単位の選び直しの添え字 [n_boot, S] と、それで計算する関数を返す。
    per_scene_sum: {名前: [S] の L1 の和}、per_scene_n: [S] フレーム数 → {名前: [n_boot] の L(フレームで重み)}"""
    S = len(per_scene_n)
    idx = np.random.default_rng(seed).integers(0, S, size=(n_boot, S))
    n = np.asarray(per_scene_n, np.float64)
    den = n[idx].sum(axis=1)
    return {k: np.asarray(v, np.float64)[idx].sum(axis=1) / den for k, v in per_scene_sum.items()}
