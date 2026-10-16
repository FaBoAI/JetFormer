#!/usr/bin/env python3
"""B2 の机上の判定の計算(numpy だけ。test_b2_lib.py で試験する)。

protocol.json(b2_curation_20261016)の evaluation.gates・decision の式を、そのまま関数にしたもの。
  - 通過の値 u_m(p) = 通過 p の窓のフレームの、モデル m の走行の出力の平均
  - Δ_m(p) = u_m(p) − u_B(p)(スロットル: + = 遅い。ステア: + = 左)。B2 は種 0・1 の Δ の平均
  - 集合の値 D = 通過の平均。区間は通過の選び直し 10,000 回(乱数の種 0)の 95% パーセンタイル
  - L1 の判定は tools/sim_cotrain.py の judge_primary・final_verdict(10/13 の規則)をそのまま呼ぶ
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join('/home/jetson/jetformer', 'tools'))
import sim_cotrain as sc  # noqa: E402  10/13 の judge_primary・final_verdict(固定した規則)

NEUTRAL_US = 1498.0                # 前進の強さ(中立 − 出力のスロットル)の基準(rl_evaluate.NEUTRAL_US)
FAST_ERPM = 5500.0                 # 速い進入 = entry_erpm ≥ 5500(10/12 の ENTRY_BAND)
VERY_FAST_ERPM = 6200.0
N_BOOT = 10000
CKPTS = ('best', 'last')
TH = {
    'r1_held': {'mean': 30.0, 'each_seed': 20.0, 'vs_A': 15.0, 'last': 20.0},
    'r1_train': {'mean': 30.0, 'each_seed': 20.0, 'vs_A': 15.0},
    'r1_slow': {'mean': -10.0},
    'de': {'floor': -15.0, 'margin': 5.0},
    'straight_drop_max': 0.05,
    'fwd_bias_min_us': -30.0,
    'sign_acc_margin': 0.01,
    'l1_rest_band_best': 2.798,     # 5 本 {A_s0〜s3, B} の rest の範囲の 2 倍(baseline.json)
    'l1_practical_us': 10.0,
    'iv_band_best': 5.822,          # 介入の集合の 5 本の範囲の 2 倍
}


# ---------------------------------------------------------------- 通過の値

def window_means(out, rows_list, ch):
    """out [N,2](走行の出力)・rows_list = 通過ごとの行の配列 → [通過] の窓の平均(行が無い通過は NaN)"""
    out = np.asarray(out, np.float64)
    return np.array([out[np.asarray(r, int), ch].mean() if len(r) else np.nan for r in rows_list])


def boot_ci(d, n_boot=N_BOOT, seed=0, level=95.0):
    """通過を選び直して平均の区間 [下, 上](d は通過ごとの値)"""
    d = np.asarray(d, np.float64)
    if len(d) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    a = (100.0 - level) / 2.0
    return [float(np.percentile(b, a)), float(np.percentile(b, 100.0 - a))]


def set_summary(delta_seeds, delta_A=None, delta_last_seeds=None, n_boot=N_BOOT, seed=0):
    """delta_seeds = {種: [通過] の Δ}、delta_A = {A_i: [通過]}、delta_last_seeds = {種: [通過]}(last.pth の B2 − best の B)
    → D(種の平均の通過の平均)・種ごとの D・区間・D_Ai・last の D"""
    keys = sorted(delta_seeds)
    mat = np.stack([np.asarray(delta_seeds[k], np.float64) for k in keys])
    per_pass = mat.mean(axis=0)
    res = {'n': int(mat.shape[1]), 'D': float(per_pass.mean()) if mat.shape[1] else None,
           'D_seeds': {str(k): float(mat[i].mean()) for i, k in enumerate(keys)},
           'ci95': boot_ci(per_pass, n_boot, seed), 'per_pass_mean_delta': per_pass.tolist()}
    if delta_A is not None:
        res['D_A'] = {k: float(np.mean(v)) for k, v in delta_A.items()}
    if delta_last_seeds is not None:
        ml = np.stack([np.asarray(delta_last_seeds[k], np.float64) for k in sorted(delta_last_seeds)])
        res['D_last'] = float(ml.mean(axis=0).mean())
        res['D_last_seeds'] = {str(k): float(ml[i].mean()) for i, k in enumerate(sorted(delta_last_seeds))}
    return res


# ---------------------------------------------------------------- ゲート(protocol.evaluation.gates)

def gate_r1_held(s, th=TH['r1_held']):
    """D ≥ +30 かつ 両方の種で ≥ +20 かつ 区間の下限 > 0 かつ D − max_i D_Ai ≥ +15 かつ last の D ≥ +20"""
    parts = {'D_ge_30': s['D'] >= th['mean'],
             'each_seed_ge_20': all(v >= th['each_seed'] for v in s['D_seeds'].values()),
             'ci_low_gt_0': s['ci95'][0] is not None and s['ci95'][0] > 0,
             'minus_maxA_ge_15': s['D'] - max(s['D_A'].values()) >= th['vs_A'],
             'last_ge_20': s.get('D_last') is not None and s['D_last'] >= th['last']}
    return {'pass': all(parts.values()), 'parts': parts, 'D_minus_maxA': s['D'] - max(s['D_A'].values())}


def gate_r1_train(s, th=TH['r1_train']):
    """D ≥ +30 かつ 両方の種で ≥ +20 かつ D − max_i D_Ai ≥ +15"""
    parts = {'D_ge_30': s['D'] >= th['mean'],
             'each_seed_ge_20': all(v >= th['each_seed'] for v in s['D_seeds'].values()),
             'minus_maxA_ge_15': s['D'] - max(s['D_A'].values()) >= th['vs_A']}
    return {'pass': all(parts.values()), 'parts': parts, 'D_minus_maxA': s['D'] - max(s['D_A'].values())}


def gate_r1_slow(s, th=TH['r1_slow']):
    """D ≥ −10(遅い近づき方で、B より強く前へ出さない)"""
    parts = {'D_ge_minus10': s['D'] >= th['mean']}
    return {'pass': all(parts.values()), 'parts': parts}


def de_threshold(D_A, th=TH['de']):
    """min(−15, min_i D_Ai − 5)"""
    return float(min(th['floor'], min(D_A.values()) - th['margin']))


def gate_de(s_steer, s_thr, th=TH['de']):
    """ステアの D ≥ min(−15, min_i D_Ai − 5)(右へ寄らない)かつ スロットルの D ≥ 同じ式(強く前へ出さない)"""
    t_s, t_t = de_threshold(s_steer['D_A'], th), de_threshold(s_thr['D_A'], th)
    parts = {'steer_ok': s_steer['D'] >= t_s, 'throttle_ok': s_thr['D'] >= t_t}
    return {'pass': all(parts.values()), 'parts': parts, 'threshold_steer_us': t_s, 'threshold_throttle_us': t_t}


def straight_drop(fwd_arm_seeds, fwd_ref):
    """前進の強さ(1498 − 出力のスロットル)の種の平均の、B に対する低下の割合"""
    return 1.0 - float(np.mean(fwd_arm_seeds)) / float(fwd_ref)


def gate_straight(fwd_arm_seeds, fwd_ref, th=TH['straight_drop_max']):
    d = straight_drop(fwd_arm_seeds, fwd_ref)
    return {'pass': bool(d <= th), 'drop': d, 'fwd_arm_seeds': [float(x) for x in fwd_arm_seeds],
            'fwd_arm_mean': float(np.mean(fwd_arm_seeds)), 'fwd_ref': float(fwd_ref)}


def gate_no_collapse(fwd_bias_seeds, sign_seeds, sign_ref, th=TH):
    """fwd_bias(3 フレーム先の予測 − 人、種の平均)≥ −30 かつ 向きの正解率(種の平均)≥ B − 0.01"""
    fb, sg = float(np.mean(fwd_bias_seeds)), float(np.mean(sign_seeds))
    parts = {'fwd_bias_ok': fb >= th['fwd_bias_min_us'], 'sign_acc_ok': sg >= float(sign_ref) - th['sign_acc_margin']}
    return {'pass': all(parts.values()), 'parts': parts, 'fwd_bias_mean': fb, 'sign_acc_mean': sg,
            'sign_acc_ref': float(sign_ref)}


def compare(arm_vals, ref_vals, s_floor):
    """10/13 の規則: arm_vals・ref_vals = {'best': [種], 'last': [種]}、s_floor = {'best': s, 'last': s}
    → best・last の judge_primary と最終の分類(「良い」は両方で better のときだけ、「悪い」は best のまま)"""
    out = {ck: sc.judge_primary(arm_vals[ck], ref_vals[ck], 2.0, s_floor[ck], 10.0) for ck in CKPTS}
    out['final_class'] = sc.final_verdict(out['best'], out['last'])
    out['worsening_best_us'] = -out['best']['gain_us']        # + = 腕のほうが大きい(悪い)
    out['worsening_last_us'] = -out['last']['gain_us']
    out['arm'] = {ck: [float(x) for x in arm_vals[ck]] for ck in CKPTS}
    out['ref'] = {ck: [float(x) for x in ref_vals[ck]] for ck in CKPTS}
    return out


def gate_l1(primary, rest, th=TH):
    """NOT(rest が「悪い」かつ rest の悪化(best)> 2.798 µs)かつ 主指標の悪化(best)< 10 µs"""
    rest_bad = rest['final_class'] == 'worse' and rest['worsening_best_us'] > th['l1_rest_band_best']
    parts = {'rest_not_broken': not rest_bad, 'primary_worsening_lt_10': primary['worsening_best_us'] < th['l1_practical_us']}
    return {'pass': all(parts.values()), 'parts': parts, 'reading': l1_reading(primary, rest)}


def l1_reading(primary, rest, band_primary=2.49, band_rest=2.798, practical=10.0):
    """protocol の G3-L1.interpretation(先に決めた読み方)"""
    c, w = primary['final_class'], primary['worsening_best_us']
    if w >= practical:
        return 'no-go: 主指標の悪化が 10 µs 以上'
    if c in ('equivalent', 'better'):
        return '同等・良い: 精度を落とさずに選別できた'
    if w <= band_primary:
        return '悪い(種の帯の内): 種のばらつきの内。走行の判断には使わない'
    if rest['final_class'] != 'worse' and w < practical:
        return '悪い(種の帯の外)だが rest は「悪い」でない: 速い進入をまねなくなった分(期待どおり)'
    if rest['final_class'] == 'worse' and rest['worsening_best_us'] > band_rest:
        return 'no-go: rest も「悪い」で、悪化が rest の種の帯を越えた'
    return '悪い(種の帯の外)・rest も「悪い」だが rest の帯の内'


def gate_intervention(cmp, th=TH['iv_band_best']):
    """NOT(「悪い」かつ 悪化(best)> 5.822 µs)"""
    bad = cmp['final_class'] == 'worse' and cmp['worsening_best_us'] > th
    return {'pass': not bad, 'parts': {'not_worse_beyond_band': not bad}}


GATE_ORDER = ('G3-R1-held', 'G3-R1-train', 'G3-R1-slow', 'G3-DE', 'G3-straight', 'G3-no-collapse', 'G3-L1',
              'G3-intervention')


def desk_go(gates):
    """desk_go(g) = 8 つのゲートの積 → (合否, 落ちたゲート)"""
    failed = [k for k in GATE_ORDER if not gates[k]['pass']]
    return (len(failed) == 0), failed


def drive_genre(go5, go4):
    """g = 5 で desk_go なら 5。だめで 4 で desk_go なら 4。どちらもだめなら None(no-go)"""
    if go5:
        return 5
    if go4:
        return 4
    return None


def gate_engine(e, b_run_numpy_median, g, ref_b=(1441.0, 1496.0)):
    """FP16 と FP32 の差の最大 ≤ 2.5 µs・run_numpy 中央値 ≤ 6.0 ms・p99 ≤ 8.0 ms・peak_mem ≤ 4,096 MB・NaN/Inf なし・
    中央値 ≤ B + 0.5 ms・list_models に出て trained_modes に g・中心・中立が B と同じ"""
    mi = e.get('model_info') or {}
    lat = e.get('latency_ms') or {}
    acc = e.get('accuracy') or {}
    parts = {
        'fp16_vs_fp32_le_2p5us': acc.get('actions_max_us') is not None and acc['actions_max_us'] <= 2.5,
        'finite': bool(acc.get('finite')),
        'run_numpy_median_le_6ms': lat.get('run_numpy_median') is not None and lat['run_numpy_median'] <= 6.0,
        'run_numpy_p99_le_8ms': lat.get('run_numpy_p99') is not None and lat['run_numpy_p99'] <= 8.0,
        'peak_mem_le_4096mb': e.get('peak_mem_mb') is not None and e['peak_mem_mb'] <= 4096,
        'median_le_B_plus_0p5': (lat.get('run_numpy_median') is not None and b_run_numpy_median is not None
                                 and lat['run_numpy_median'] <= float(b_run_numpy_median) + 0.5),
        'listed_and_genre_trained': bool(e.get('listed')) and g is not None and int(g) in [int(x) for x in
                                                                                          (mi.get('trained_modes') or [])],
        'drivable': bool(mi.get('drivable')),
        'ref_same_as_B': (mi.get('steer_center_us') is not None
                          and (float(mi['steer_center_us']), float(mi['throttle_neutral_us'])) == tuple(map(float, ref_b))),
    }
    return {'pass': all(parts.values()), 'parts': parts}


# ---------------------------------------------------------------- 参考の値

def fwd_strength(out_rows_thr):
    return float(np.mean(NEUTRAL_US - np.asarray(out_rows_thr, np.float64)))


def erpm_from_throttle(x_us, y_erpm, u):
    return np.interp(np.asarray(u, np.float64), np.asarray(x_us, np.float64), np.asarray(y_erpm, np.float64))


def entry_estimate(entry_actual, u_model, u_actual, x_us, y_erpm):
    """rl_evaluate.entry_estimate と同じ: 実際の進入 + 対応表での差(モデルの出力 − 実際に出した指令)"""
    return (np.asarray(entry_actual, np.float64) + erpm_from_throttle(x_us, y_erpm, u_model)
            - erpm_from_throttle(x_us, y_erpm, u_actual))
