"""評価の結果(eval/zoo_eval.json = tools/policy_zoo.py evaluate の出力)から、報告の表(Markdown)と evaluation.json(要約)を作る(CPU)。

    python /home/jetson/jetformer_reports/policy_zoo_20261012/eval/make_tables.py

書くもの: eval/tables.md(evaluation.md に貼る表)、../evaluation.json(要約 + 全部の結果)。数字は手で写さない。
"""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
sys.path.insert(0, '/home/jetson/jetformer/tools')
POL = ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']
LAB = {'B': 'モデル B', 'jetformer_v0': 'jetformer_v0', 'act': 'act', 'tcp': 'tcp', 'msfusion': 'msfusion', 'dinov2': 'dinov2'}
CLS = {'good': '良い', 'bad': '悪い', 'equivalent': '同等'}


def f(x, nd=1, sign=False):
    if x is None:
        return '—'
    s = ('%+.' + str(nd) + 'f') % x if sign else ('%.' + str(nd) + 'f') % x
    return s


def seeds(p):
    return ['%s_s0' % p, '%s_s1' % p]


def yn(b):
    return '○' if b else '×'


def tables(r):
    out = []
    P = r['primary']
    L, Ll = P['L_us']['best'], P['L_us']['last']
    # T1: v0 の3本
    v0 = ['B'] + seeds('jetformer_v0')
    out.append('### 表1 基準(v0)の3本と δ\n')
    out.append('| 学習結果 | L(best.pth) | L(last.pth) | 学習中の検証 L1 |')
    out.append('|---|---|---|---|')
    for n in v0:
        out.append('| %s | %s | %s | %s |' % ({'B': 'モデル B(旧コード・再開あり)', 'jetformer_v0_s0': 'v0 種 0',
                                               'jetformer_v0_s1': 'v0 種 1'}[n], f(L[n], 2), f(Ll[n], 2),
                                              f(P['train_val_us'].get(n), 1)))
    out.append('\n3本の範囲 %s µs → **δ = max(10, 範囲) = %s µs**(last.pth では %s µs)。この幅より小さい差は区別できない。\n'
               % (f(P['v0_three_runs_range_us'], 2), f(P['delta_us']['best'], 1), f(P['delta_us']['last'], 1)))
    # T2: 主指標
    out.append('### 表2 主指標(保留した人のきれいな走行の L1、4,670 フレーム、PyTorch FP32)\n')
    out.append('| 方策 | 種 0 best / last | 種 1 best / last | L̄(best) | 範囲 | Δ = L̄ − L̄_v0 | Δ% | 分類 best / last → 判定 | 全部の種 × ckpt で向きがそろう |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for p in POL:
        pp = P['per_policy'][p]
        s0, s1 = seeds(p)
        b = pp['best']
        out.append('| %s | %s / %s | %s / %s | **%s** | %s〜%s | %s | %s | %s / %s → **%s** | %s |' % (
            LAB[p], f(L[s0], 2), f(Ll[s0], 2), f(L[s1], 2), f(Ll[s1], 2), f(b['mean'], 2), f(b['min'], 2), f(b['max'], 2),
            '—' if p == 'jetformer_v0' else f(b['diff'], 2, True), '—' if p == 'jetformer_v0' else f(pp['diff_pct'], 1, True),
            CLS[b['class']], CLS[pp['last']['class']], CLS[pp['final_class']],
            {'worse': '悪い向き', 'better': '良い向き', None: '—'}[pp['consistent']] if p != 'jetformer_v0' else '—'))
    out.append('| モデル B | %s / %s | — | %s | — | %s | %s | (別の行) | — |' % (
        f(L['B'], 2), f(Ll['B'], 2), f(L['B'], 2), f(L['B'] - P['per_policy']['jetformer_v0']['best']['mean'], 2, True),
        f(100 * (L['B'] - P['per_policy']['jetformer_v0']['best']['mean']) / P['per_policy']['jetformer_v0']['best']['mean'], 1, True)))
    out.append('')
    # T2b: ばらつき
    out.append('### 表3 ばらつきの実測(主指標)\n')
    tv = {x['run']: x for x in json.load(open(REPORT / 'train' / 'summary.json', encoding='utf-8'))['runs']} \
        if isinstance(json.load(open(REPORT / 'train' / 'summary.json', encoding='utf-8'))['runs'], list) else \
        {x['run']: x for x in json.load(open(REPORT / 'train' / 'summary.json', encoding='utf-8'))['runs'].values()}
    out.append('| 方策 | 種どうしの範囲 best / last [µs] | best − last 種 0 / 種 1 [µs] | 学習中の検証 best / 最後のエポック 種 0;種 1 [µs] | 境目の前後 1 秒を除いた L(D8)種 0 / 種 1 |')
    out.append('|---|---|---|---|---|')
    for p in POL:
        s0, s1 = seeds(p)
        sr = P['seed_range_us'][p]
        t0, t1 = tv.get(r['runs'][s0], {}), tv.get(r['runs'][s1], {})
        out.append('| %s | %s / %s | %s / %s | %s / %s;%s / %s | %s / %s |' % (
            LAB[p], f(sr['best'], 2), f(sr['last'], 2), f(P['best_minus_last_us'][s0], 2, True),
            f(P['best_minus_last_us'][s1], 2, True), f(t0.get('best_val_us')), f(t0.get('last_val_us')), f(t1.get('best_val_us')),
            f(t1.get('last_val_us')), f(P['d8_boundary_excluded']['L_us'][s0], 2), f(P['d8_boundary_excluded']['L_us'][s1], 2)))
    out.append('| モデル B | — | %s | 49.0 / — | %s |' % (f(P['best_minus_last_us']['B'], 2, True), f(P['d8_boundary_excluded']['L_us']['B'], 2)))
    out.append('\nD8 の除いたフレーム数: %d。\n' % P['d8_boundary_excluded']['frames'])
    # T4: ゲート
    G = r['gates']
    out.append('### 表4 ゲート(spec §5.1、判定は2つの種の両方)\n')
    out.append('| 学習結果 | エンジン | 精度 最大 [µs] | run_numpy 中央 / p99 [ms] | 学習のメモリ [MB] | 保留で NaN なし | 合格 |')
    out.append('|---|---|---|---|---|---|---|')
    for p in POL:
        for n in seeds(p):
            g = G['per_run'][n]
            out.append('| %s 種 %s | %s | %s | %s / %s | %s | %s | %s |' % (
                LAB[p], n[-1], g['engine_precision'], f(g['actions_max_us'], 2), f(g['run_numpy_median_ms'], 2),
                f(g['run_numpy_p99_ms'], 2), f(g['peak_mem_mb'], 0), yn(g['parts']['finite']), '**合格**' if g['pass'] else '**不合格**'))
    for n, g in G['dinov2_rebuilt'].items():
        out.append('| dinov2 種 %s(作り直し・事後) | %s | %s | %s / %s | — | %s | %s |' % (
            n[-1], g['engine_precision'], f(g['actions_max_us'], 2), f(g['run_numpy_median_ms'], 2), f(g['run_numpy_p99_ms'], 2),
            yn(g['parts']['finite']), '合格(走行可)' if g['pass'] else '不合格'))
    gb = G['per_run']['B']
    out.append('| モデル B(参考) | fp16 | %s | %s / %s | — | %s | %s |' % (f(gb['actions_max_us'], 2), f(gb['run_numpy_median_ms'], 2),
                                                                   f(gb['run_numpy_p99_ms'], 2), yn(gb['parts']['finite']),
                                                                   '合格' if gb['pass'] else '不合格'))
    out.append('\n方策のゲート(判定用): ' + '、'.join('%s %s' % (LAB[p], '合格' if G['per_policy'][p] else '**不合格**') for p in POL) + '\n')
    # T5: R1・DE
    for item, title, unit in (('r1', 'R1(進入前 0.4 秒のスロットル。+ = 遅い)', 'スロットル'),
                              ('de', 'DE(緑マット出口〜0.8 秒のステア。+ = 左)', 'ステア')):
        it = r[item]
        out.append('### 表%s %s、%d 通過(AI %d)\n' % ('5' if item == 'r1' else '6', title, it['passes'], it['ai_passes']))
        out.append('| 方策 − v0 | d̄ [µs] | 99.375% 区間 | 95% 区間 | 種ごと d̄_i | 4 組 d_ij の最小 | 差の差 99.375% 区間 | 良い | 悪い | last.pth で 良い / 悪い |')
        out.append('|---|---|---|---|---|---|---|---|---|---|')
        rows = [(p, it['per_policy'][p], it['per_policy_last'][p]) for p in POL if p != 'jetformer_v0']
        rows += [('モデル B − v0', it['model_B_vs_v0'], None), ('陰性対照 A(v0 種1 − {種0, B})', it['nc']['NC-A'], None),
                 ('陰性対照 B(v0 種0 − {種1, B})', it['nc']['NC-B'], None)]
        for name, j, jl in rows:
            mn = min(x for row in j['d_pairs'] for x in row)
            out.append('| %s | %s | [%s, %s] | [%s, %s] | %s | %s | [%s, %s] | %s | %s | %s |' % (
                LAB.get(name, name), f(j['d_mean'], 1, True), f(j['ci_good'][1], 1, True), f(j['ci_good'][2], 1, True),
                f(j['ci_bad'][1], 1, True), f(j['ci_bad'][2], 1, True), ' / '.join(f(x, 1, True) for x in j['d_seed']),
                f(mn, 1, True), f(j['adj_ci_good'][1], 1, True), f(j['adj_ci_good'][2], 1, True), yn(j['good']), yn(j['bad']),
                '—' if jl is None else '%s / %s' % (yn(jl['good']), yn(jl['bad']))))
        out.append('')
    # 進入の見積もり
    e = r['r1']['entry_estimate']
    out.append('### 表7 R1 の進入の見積もり(参考。対応表は粗い: 実際の進入との順位相関 %s)\n' % f(e['map']['validation_heldout_r1']['spearman_map_vs_entry'], 2))
    out.append('| | 窓の出力のスロットル [µs] | 見積もった進入 eRPM の中央値 | 5,500 未満の割合 | 人の通過(13)の中央値 | 実際との差の平均 [eRPM] |')
    out.append('|---|---|---|---|---|---|')
    a = e['actual']
    out.append('| 実際(人 13・AI 4) | %s(実際の指令) | %s | %s | %s | — |' % (f(a['exec_thr_us_mean'], 1), f(a['median_erpm'], 0),
                                                                 f(a['share_lt5500'], 2), f(a['median_erpm_human'], 0)))
    for n in ['B'] + [x for p in POL for x in seeds(p)]:
        q = e['per_run'][n]
        out.append('| %s | %s | %s | %s | %s | %s |' % (n, f(q['thr_out_us_mean'], 1), f(q['median_erpm'], 0), f(q['share_lt5500'], 2),
                                                     f(q['median_erpm_human_passes'], 0), f(q['mean_minus_actual_erpm'], 0, True)))
    out.append('')
    # T8: 崩れ
    C = r['collapse']
    out.append('### 表8 崩れ(点推定、種の平均どうし。直線 %d フレーム)\n' % C['straight_frames'])
    out.append('| 方策 | 直線の前進の強さ [µs](v0) | 低下 | 前進の偏り [µs] | 向きの正解率(v0) | 崩れ |')
    out.append('|---|---|---|---|---|---|')
    rows = [(p, C['per_policy'][p]) for p in POL if p != 'jetformer_v0'] + [('モデル B', C['model_B_vs_v0']),
                                                                            ('陰性対照 A', C['nc']['NC-A']), ('陰性対照 B', C['nc']['NC-B'])]
    for name, c in rows:
        out.append('| %s | %s(%s) | %s%% | %s | %s(%s) | %s |' % (LAB.get(name, name), f(c['fwd_strength_us'], 1),
                                                                f(c['ref_fwd_strength_us'], 1), f(100 * c['straight_drop'], 1, True),
                                                                f(c['fwd_bias_us'], 1, True), f(c['sign_acc'], 3), f(c['ref_sign_acc'], 3),
                                                                '**崩れ**' if c['collapse'] else 'なし'))
    out.append('')
    # T9: ずれ
    R = r['robustness']
    out.append('### 表9 ずれへの強さ(ΔL1 = ずれの L1 − ずれなしの L1、%d フレーム)\n' % R['per_run']['B']['frames'])
    out.append('| 学習結果 | 3 行下 | 3 行上 | 明るさ ×0.8 | ×1.2 | ToF 1 行下 | 平均 ΔL1 | 判定(v0 の %s の 0.8 倍以下 & 両方の種で v0 以下) |' % f(R['ref_dL1_us'], 2))
    out.append('|---|---|---|---|---|---|---|---|')
    for n in ['B'] + [x for p in POL for x in seeds(p)]:
        q = R['per_run'][n]
        pp = q['per_pert_dL1_us']
        p = n.rsplit('_s', 1)[0]
        judge = ''
        if n.endswith('_s0') and p != 'jetformer_v0':
            judge = '強い' if R['per_policy'][p]['robust_good'] else '—(比 %s)' % f(R['per_policy'][p]['ratio_to_ref'], 2)
        out.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (n, f(pp['img_down3'], 2, True), f(pp['img_up3'], 2, True),
                                                           f(pp['bright_0.8'], 2, True), f(pp['bright_1.2'], 2, True),
                                                           f(pp['tof_down1'], 2, True), f(q['dL1_us'], 2, True), judge))
    out.append('')
    # T10: 陰性対照
    N = r['negative_controls']
    out.append('### 表10 陰性対照(D4)\n')
    out.append('| | 主指標 L(方策)/ 基準の平均 | 分類 | R1 良い | DE 良い | ずれに強い | 崩れ | 規則 3 を無効にするか |')
    out.append('|---|---|---|---|---|---|---|---|')
    for k in ('NC-A', 'NC-B'):
        v = N[k]
        out.append('| %s(%s − {%s}) | %s / %s | %s | %s | %s | %s | %s | %s |' % (
            k, v['policy'], ', '.join(v['ref']), f(v['primary']['best']['mean'], 2), f(v['primary']['best']['ref_mean'], 2),
            CLS[v['primary']['final_class']], yn(v['flags']['r1_good']), yn(v['flags']['de_good']), yn(v['flags']['robust_good']),
            yn(v['flags']['collapse']), 'はい' if v['invalidates_rule3'] else 'いいえ'))
    out.append('')
    # T11: 参考の精度
    A = r['accuracy']
    M = A['metrics']
    out.append('### 表11 参考: 走行の精度(compare_models.metrics、自己入力 = 教師あり、best.pth。種 0 / 種 1)\n')
    out.append('| 学習結果 | ステア k1 / k3 / k10 | スロットル k1 / k3 / k10 | 向きの正解率 | 前進の偏り | L1 ジャンル 4 / 5 | 衝突の前後も含む L1 |')
    out.append('|---|---|---|---|---|---|---|')
    for n in ['B'] + [x for p in POL for x in seeds(p)]:
        m = M[n]['human_clean']
        out.append('| %s | %s / %s / %s | %s / %s / %s | %s | %s | %s / %s | %s |' % (
            n, f(m['steer_mae'][0]), f(m['steer_mae'][2]), f(m['steer_mae'][9]), f(m['throttle_mae'][0]), f(m['throttle_mae'][2]),
            f(m['throttle_mae'][9]), f(m['steer_sign_acc'], 3), f(m['fwd_bias_us'], 1, True), f(M[n]['human_clean_genre4']['l1_us'], 2),
            f(M[n]['human_clean_genre5']['l1_us'], 2), f(M[n]['human_all']['l1_us'], 2)))
    out.append('\nフレーム数: ' + '、'.join('%s %d' % (k, v) for k, v in A['sets'].items()) + '\n')
    out.append('### 表12 参考: 交代の後の人の修正(intervention_val、L1 [µs]。D12: 学習データの中の参考)\n')
    out.append('| 学習結果 | 全部(%d) | ログ単位で保留の 3 窓(%d) | 学習のログの窓(%d) | AI のフレームとの一致(参考) |' % (
        A['sets']['iv_all'], A['sets']['iv_log_heldout'], A['sets']['iv_train_logs']))
    out.append('|---|---|---|---|---|')
    for n in ['B'] + [x for p in POL for x in seeds(p)]:
        out.append('| %s | %s | %s | %s | %s |' % (n, f(M[n]['iv_all']['l1_us'], 2), f(M[n]['iv_log_heldout']['l1_us'], 2),
                                                 f(M[n]['iv_train_logs']['l1_us'], 2), f(M[n]['ai_val_clean_ref']['l1_us'], 2)))
    out.append('')
    # T13: 速さ
    T = r['latency']
    out.append('### 表13 速さ・大きさ(同じ道具。走行ループ・カメラ → PWM は見積もり)\n')
    out.append('| 学習結果 | パラメータ 全体 / 学習 [M] | エンジンだけ 中央 [ms] | run_numpy 中央 / p99 [ms] | 走行ループの見積もり [ms] | カメラ → PWM の見積もり [ms] |')
    out.append('|---|---|---|---|---|---|')
    for n in ['B'] + [x for p in POL for x in seeds(p)]:
        q = T['per_run'][n]
        out.append('| %s | %s / %s | %s | %s / %s | %s | %s |' % (n, f(q['params_total_m'], 2), f(q['params_trainable_m'], 2),
                                                             f(q['engine_median_ms'], 2), f(q['run_numpy_median_ms'], 2),
                                                             f(q['run_numpy_p99_ms'], 2), f(q['loop_estimate_ms'], 1),
                                                             f(q['cam_to_pwm_estimate_ms'], 1)))
    out.append('')
    # T14: tcp
    tc = r['tcp']
    out.append('### 表14 tcp の追加(参考、§5.5)\n')
    out.append('通過点・目標速度(clean_driving の val のラベル、wp_valid の %d フレーム。ラベルと予測のフレームが同じ: %s)\n'
               % (tc['wp_valid_frames'], tc['labels_same_frames_as_ai_mix']))
    out.append('| 学習結果 | ADE [m] | FDE [m] | 目標速度の誤差 MAE / 偏り [m/s] | ラベルの平均 / 予測の平均 [m/s] |')
    out.append('|---|---|---|---|---|')
    for n, q in sorted(tc['per_run'].items()):
        out.append('| %s | %s | %s | %s / %s | %s / %s |' % (n, f(q['waypoints']['ade_m'], 3), f(q['waypoints']['fde_m'], 3),
                                                       f(q['target_speed']['mae_mps'], 3), f(q['target_speed']['bias_mps'], 3, True),
                                                       f(q['target_speed']['v_label_mean_mps'], 2), f(q['target_speed']['v_hat_mean_mps'], 2)))
    out.append('')
    out.append('| 学習結果 | 出し方 | L1 [µs] | ステア / スロットル L1 | R1 窓のスロットル [µs] | R1 進入の見積もり 中央 [eRPM] | 5,500 未満 | 直線の前進の強さ [µs] |')
    out.append('|---|---|---|---|---|---|---|---|')
    for n, q in sorted(tc['per_run'].items()):
        for s, v in q['settings'].items():
            out.append('| %s | %s | %s | %s / %s | %s | %s | %s | %s |' % (n, s, f(v['L1_human_clean_us'], 2), f(v['L1_steer_us'], 1),
                                                                     f(v['L1_throttle_us'], 1), f(v['r1_window_thr_us'], 1),
                                                                     f(v['r1_entry_est_median_erpm'], 0), f(v['r1_entry_share_lt5500'], 2),
                                                                     f(v['straight_fwd_strength_us'], 1)))
    out.append('')
    out.append('| 学習結果 | 比べたもの | R1 窓のスロットルの差 95% 区間 [µs](+ = 弱まる) | 直線の前進の強さの低下 | 「上限が使える」(机上) |')
    out.append('|---|---|---|---|---|')
    for n, q in sorted(tc['per_run'].items()):
        for k, v in q['cap_effect'].items():
            u = q['speed_cap_usable_offline'].get(k.split('-')[0], None)
            ci = v['r1_d_thr_us_ci95']
            out.append('| %s | %s | %s [%s, %s] | %s%% | %s |' % (n, k, f(ci[0], 1, True), f(ci[1], 1, True), f(ci[2], 1, True),
                                                               f(100 * v['straight_drop'], 1, True), '—' if u is None else yn(u)))
    out.append('')
    if r.get('crosscheck'):
        out.append('### 表15 走行と同じ手順(compare_models.predict_scene)との一致(シーン %s)\n' % list(r['crosscheck'].values())[0]['scene'])
        out.append('| 学習結果 | フレーム | 教師あり − まとめた予測 最大 [µs] | 自己入力 − まとめた予測 最大 [µs] | 教師あり − 自己入力 最大 [µs] |')
        out.append('|---|---|---|---|---|')
        for n, v in r['crosscheck'].items():
            out.append('| %s | %d | %s | %s | %s |' % (n, v['frames'], f(v['max_abs_tf_vs_cached_us'], 4), f(v['max_abs_sf_vs_cached_us'], 4),
                                                      f(v['max_abs_tf_vs_sf_us'], 4)))
        out.append('')
    return '\n'.join(out)


def summary(r):
    P = r['primary']
    v = r['verdict']
    return {
        'title': '方策の比較(policy zoo)の評価のまとめ',
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'verdict': {'best_policy': v['best'], 'rule': v['rule'], 'note': v['note'], 'candidates': v['candidates'],
                    'rule2_good': v['rule2'], 'rule3': v['rule3'], 'rule3_valid': v['rule3_valid']},
        'recommended_engine': r['recommended'],
        'delta_us': P['delta_us'],
        'primary_L_us': {p: {'seeds_best': [P['L_us']['best'][n] for n in seeds(p)],
                             'seeds_last': [P['L_us']['last'][n] for n in seeds(p)],
                             'mean_best': P['per_policy'][p]['best']['mean'], 'diff_vs_v0': P['per_policy'][p]['best']['diff'],
                             'class': P['per_policy'][p]['final_class'], 'consistent': P['per_policy'][p]['consistent']}
                         for p in POL},
        'model_B_L_us': {'best': P['L_us']['best']['B'], 'last': P['L_us']['last']['B']},
        'gates': {'per_policy': r['gates']['per_policy'], 'dinov2_drivable_after_rebuild': r['gates']['dinov2_drivable_after_rebuild']},
        'r1_good': {p: r['r1']['per_policy'][p]['good'] for p in r['r1']['per_policy']},
        'r1_bad': {p: r['r1']['per_policy'][p]['bad'] for p in r['r1']['per_policy']},
        'de_good': {p: r['de']['per_policy'][p]['good'] for p in r['de']['per_policy']},
        'de_bad': {p: r['de']['per_policy'][p]['bad'] for p in r['de']['per_policy']},
        'collapse': {p: r['collapse']['per_policy'][p]['collapse'] for p in r['collapse']['per_policy']},
        'robust_good': {p: r['robustness']['per_policy'][p]['robust_good'] for p in r['robustness']['per_policy']},
        'negative_controls_invalidate_rule3': r['negative_controls']['invalidates_rule3'],
        'verdict_sensitivity_dinov2_rebuilt': {'best': r['verdict_sensitivity_dinov2_rebuilt']['best'],
                                               'rule': r['verdict_sensitivity_dinov2_rebuilt']['rule']},
        'files': {'full_result': str(HERE / 'zoo_eval.json'), 'protocol': str(HERE / 'protocol.json'),
                  'rules': str(REPORT / 'results' / 'zoo_rules.json'), 'report': str(REPORT / 'evaluation.md')},
    }


def main():
    r = json.load(open(HERE / 'zoo_eval.json', encoding='utf-8'))
    (HERE / 'tables.md').write_text(tables(r), encoding='utf-8')
    s = summary(r)
    s['result'] = r
    tmp = REPORT / 'evaluation.json.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, REPORT / 'evaluation.json')
    print('書きました:', HERE / 'tables.md', REPORT / 'evaluation.json')
    print(json.dumps({k: s[k] for k in ('verdict', 'recommended_engine', 'delta_us')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
