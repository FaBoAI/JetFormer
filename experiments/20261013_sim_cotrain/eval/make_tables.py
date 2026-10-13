#!/usr/bin/env python3
"""evaluation.json → 表(Markdown)。手で写さないための道具(make_report.py から使う)

    cd ~/jetformer && python /home/jetson/jetformer_reports/sim_cotrain_20261013/eval/make_tables.py   # eval/tables.md を書く
"""

import json
import sys
from pathlib import Path

REPORT = Path('/home/jetson/jetformer_reports/sim_cotrain_20261013')
CLS = {'better': '良い', 'equivalent': '同等', 'worse': '悪い'}
ARMS3 = ('C', 'C+FT', 'A+FT')
SEEDN = {'A': ['A_s0', 'A_s1'], 'B': ['B'], 'C': ['C_s0', 'C_s1'], 'C+FT': ['CFT_s0', 'CFT_s1'], 'A+FT': ['AFT_s0', 'AFT_s1']}
LAB = {'A': 'A(実機だけ)', 'B': 'モデル B(今の走行)', 'C': 'C(混ぜて学習)', 'C+FT': 'C+FT(C → 実機で微調整)',
       'A+FT': 'A+FT(A → 実機で微調整・対照)'}
RUNLAB = {'A_s0': 'A 種 0', 'A_s1': 'A 種 1', 'B': 'モデル B', 'C_s0': 'C 種 0', 'C_s1': 'C 種 1', 'CFT_s0': 'C+FT 種 0',
          'CFT_s1': 'C+FT 種 1', 'AFT_s0': 'A+FT 種 0', 'AFT_s1': 'A+FT 種 1'}
PERTS = ('img_down3', 'img_up3', 'bright_0.8', 'bright_1.2', 'tof_down1')


def f(x, nd=2, sign=False):
    if x is None:
        return '—'
    s = ('%+.' + str(nd) + 'f') % x if sign else ('%.' + str(nd) + 'f') % x
    return s


def yn(b):
    return '○' if b else '×'


def t_primary(r):
    P = r['primary']
    L = P['L_us']
    out = ['| 腕 | 種 0 best / last | 種 1 best / last | L̄(best) | L̄(last) | 種の範囲 best | L̄_腕 − L̄_A(best / last。+ = 悪い向き) | 閾値 best / last | 分類 best / last → 判定 | 実用上の差(≥ 10 µs) |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for arm in ('A', 'C', 'C+FT', 'A+FT'):
        n = SEEDN[arm]
        mb, ml = P['arm_mean_us'][arm]['best'], P['arm_mean_us'][arm]['last']
        if arm == 'A':
            out.append('| %s | %s / %s | %s / %s | **%s** | %s | %s | — | %s / %s | (基準) | — |' % (
                LAB[arm], f(L['best'][n[0]], 3), f(L['last'][n[0]], 3), f(L['best'][n[1]], 3), f(L['last'][n[1]], 3),
                f(mb, 3), f(ml, 3), f(P['seed_range_us'][arm]['best'], 3), f(P['threshold_us']['best'], 3),
                f(P['threshold_us']['last'], 3)))
            continue
        v = P['vs_A'][arm]
        out.append('| %s | %s / %s | %s / %s | **%s** | %s | %s | %s / %s | %s / %s | %s / %s → **%s** | %s |' % (
            LAB[arm], f(L['best'][n[0]], 3), f(L['last'][n[0]], 3), f(L['best'][n[1]], 3), f(L['last'][n[1]], 3),
            f(mb, 3), f(ml, 3), f(P['seed_range_us'][arm]['best'], 3), f(-v['best']['gain_us'], 3, True),
            f(-v['last']['gain_us'], 3, True), f(v['best']['threshold_us'], 3), f(v['last']['threshold_us'], 3),
            CLS[v['best']['class']], CLS[v['last']['class']], CLS[v['final_class']], 'あり' if v['practical'] else 'なし'))
    out.append('| %s(別の行) | %s / %s | — | %s | %s | — | %s / %s | — | — | — |' % (
        LAB['B'], f(L['best']['B'], 3), f(L['last']['B'], 3), f(L['best']['B'], 3), f(L['last']['B'], 3),
        f(L['best']['B'] - P['arm_mean_us']['A']['best'], 3, True), f(L['last']['B'] - P['arm_mean_us']['A']['last'], 3, True)))
    return '\n'.join(out)


def t_secondary(r):
    P = r['primary']
    rows = [('C+FT − C(微調整が C に足したもの)', P['secondary']['C+FT_vs_C']),
            ('C+FT − A+FT(シミュレーターの効果)', P['secondary']['C+FT_vs_A+FT']),
            ('A+FT − A(微調整だけの効果。追加の参考)', P['vs_A']['A+FT'])]
    out = ['| 比べたもの | 平均の差 best / last [µs] | 閾値 best / last | 種ごとの組の差 best(種 0 / 種 1) | 分類 best / last → 判定 |',
           '|---|---|---|---|---|']
    for lab, v in rows:
        out.append('| %s | %s / %s | %s / %s | %s / %s | %s / %s → **%s** |' % (
            lab, f(-v['best']['gain_us'], 3, True), f(-v['last']['gain_us'], 3, True), f(v['best']['threshold_us'], 3),
            f(v['last']['threshold_us'], 3), f(v['paired_best_us'][0], 3, True), f(v['paired_best_us'][1], 3, True),
            CLS[v['best']['class']], CLS[v['last']['class']], CLS[v['final_class']]))
    return '\n'.join(out)


def t_genre(r):
    M = r['accuracy']['metrics']
    out = ['| 学習結果 | L1 全体 | L1 高速(4) | L1 低速(5) | ステア k3 高速 / 低速 | スロットル k3 高速 / 低速 | 向きの正解率 高速 / 低速 | 前進の偏り 高速 / 低速 [µs] |',
           '|---|---|---|---|---|---|---|---|']
    for n in ['A_s0', 'A_s1', 'B', 'C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'AFT_s0', 'AFT_s1']:
        a, g4, g5 = M[n]['human_clean'], M[n]['human_clean_genre4'], M[n]['human_clean_genre5']
        out.append('| %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s |' % (
            RUNLAB[n], f(a['l1_us'], 2), f(g4['l1_us'], 2), f(g5['l1_us'], 2), f(g4['steer_mae_k3'], 1),
            f(g5['steer_mae_k3'], 1), f(g4['throttle_mae_k3'], 1), f(g5['throttle_mae_k3'], 1), f(g4['steer_sign_acc'], 3),
            f(g5['steer_sign_acc'], 3), f(g4['fwd_bias_us'], 1, True), f(g5['fwd_bias_us'], 1, True)))
    return '\n'.join(out)


def t_habit(r):
    H = r['speed_habit']
    out = ['| 腕 | 高速(4)の前進の偏り: 腕 / A / ずれ [µs] | 低速(5): 腕 / A / ずれ [µs] | 速さの癖の移り(ずれの大きさ ≥ 10 µs) |',
           '|---|---|---|---|']
    for arm in ARMS3:
        a, b = H[arm]['4'], H[arm]['5']
        out.append('| %s | %s / %s / %s | %s / %s / %s | %s |' % (
            LAB[arm], f(a['fwd_bias_us'], 1, True), f(a['ref_fwd_bias_us'], 1, True), f(a['shift_us'], 1, True),
            f(b['fwd_bias_us'], 1, True), f(b['ref_fwd_bias_us'], 1, True), f(b['shift_us'], 1, True),
            'あり' if (a['moved'] or b['moved']) else 'なし'))
    return '\n'.join(out)


def t_pass(r, item):
    it = r[item]
    out = ['| 比べたもの | 差の平均 [µs] | 99.375% 区間 | 95% 区間 | 種ごと | 4 組の最小 | 差の差 99.375% 区間 | 良い | 悪い | last.pth で 良い / 悪い |',
           '|---|---|---|---|---|---|---|---|---|---|']

    def row(lab, j, jl=None):
        mp = min(min(x) for x in j['d_pairs'])
        return '| %s | %s | [%s, %s] | [%s, %s] | %s | %s | [%s, %s] | %s | %s | %s |' % (
            lab, f(j['d_mean'], 1, True), f(j['ci_good'][1], 1, True), f(j['ci_good'][2], 1, True), f(j['ci_bad'][1], 1, True),
            f(j['ci_bad'][2], 1, True), ' / '.join(f(x, 1, True) for x in j['d_seed']), f(mp, 1, True),
            f(j['adj_ci_good'][1], 1, True), f(j['adj_ci_good'][2], 1, True), yn(j['good']), yn(j['bad']),
            ('%s / %s' % (yn(jl['good']), yn(jl['bad']))) if jl else '—')
    for arm in ARMS3:
        out.append(row(arm + ' − A', it['per_arm'][arm], it['per_arm_last'][arm]))
    out.append(row('C+FT − C', it['CFT_vs_C']))
    out.append(row('C+FT − A+FT', it['CFT_vs_AFT']))
    out.append(row('モデル B − A', it['model_B_vs_A']))
    out.append(row('陰性対照 A(A 種 1 − {種 0, B}、10/12)', it['nc_1012']['NC-A']))
    out.append(row('陰性対照 B(A 種 0 − {種 1, B}、10/12)', it['nc_1012']['NC-B']))
    wm = it['window_mean_us']['best']
    out.append('')
    out.append('窓の出力の平均(best)[µs]: ' + '、'.join('%s %s' % (RUNLAB[n], f(wm[n], 1)) for n in RUNLAB))
    return '\n'.join(out)


def t_collapse(r):
    C = r['collapse']
    out = ['| 腕 | 直線の前進の強さ [µs](A) | 低下 | 前進の偏り [µs] | 向きの正解率(A) | 崩れ |', '|---|---|---|---|---|---|']
    for arm in ARMS3 + ('B',):
        j = C['per_arm'][arm] if arm != 'B' else C['model_B_vs_A']
        out.append('| %s | %s(%s) | %s%% | %s | %s(%s) | %s |' % (
            LAB[arm], f(j['fwd_strength_us'], 1), f(j['ref_fwd_strength_us'], 1), f(100 * j['straight_drop'], 1, True),
            f(j['fwd_bias_us'], 1, True), f(j['sign_acc'], 3), f(j['ref_sign_acc'], 3), 'あり' if j['collapse'] else 'なし'))
    out.append('')
    out.append('直線のフレーム: %d' % C['straight_frames'])
    return '\n'.join(out)


def t_robust(r):
    R = r['robustness']
    out = ['| 学習結果 | 3 行下 | 3 行上 | 明るさ ×0.8 | ×1.2 | ToF 1 行下 | 平均 ΔL1 |', '|---|---|---|---|---|---|---|']
    for n in RUNLAB:
        p = R['per_run'][n]['per_pert_dL1_us']
        out.append('| %s | %s | %s |' % (RUNLAB[n], ' | '.join(f(p[k], 2, True) for k in PERTS), f(R['per_run'][n]['dL1_us'], 3, True)))
    out.append('')
    out.append('| 腕 | 平均 ΔL1(種 0 / 種 1) | A との比 | 「ずれに強い」(≤ 0.8 × A かつ両方の種で ≤ A) |')
    out.append('|---|---|---|---|')
    for arm in ARMS3:
        j = R['per_arm'][arm]
        out.append('| %s | %s(%s / %s) | %s | %s |' % (LAB[arm], f(j['dL1_mean_us'], 3, True), f(j['dL1_seeds_us'][0], 3, True),
                                                  f(j['dL1_seeds_us'][1], 3, True), f(j['ratio_to_ref'], 2),
                                                  '強い' if j['robust_good'] else '—'))
    out.append('| A(基準) | %s | 1.00 | — |' % f(R['ref_dL1_us'], 3, True))
    return '\n'.join(out)


def t_sim(r):
    S = r['sim']
    out = ['| 学習結果 | ジャンル数 | 頭 4: L1(ステア / スロットル)| 頭 5: L1(ステア / スロットル)| 頭 8: L1(ステア / スロットル)| 前進の偏り 頭 4 / 5 / 8 [µs] |',
           '|---|---|---|---|---|---|']
    for n in RUNLAB:
        row = S['per_run'][n]
        cells = []
        for h in (4, 5, 8):
            k = 'h%d' % h
            cells.append('%s(%s / %s)' % (f(row[k]['l1_us'], 1), f(row[k]['l1_steer_us'], 1), f(row[k]['l1_throttle_us'], 1))
                         if k in row else '—(頭なし)')
        fb = ' / '.join(f(row['h%d' % h]['fwd_bias_us'], 1, True) if ('h%d' % h) in row else '—' for h in (4, 5, 8))
        out.append('| %s | %d | %s | %s |' % (RUNLAB[n], row['num_modes'], ' | '.join(cells), fb))
    return '\n'.join(out)


def t_iv(r):
    M = r['accuracy']['metrics']
    sets = r['accuracy']['sets']
    out = ['| 学習結果 | 全部(%d) | ログ単位で保留の 3 窓(%d) | 学習のログの窓(%d) | 衝突の後(%d) | 先回り(%d) |' % (
        sets['iv_all'], sets['iv_log_heldout'], sets['iv_train_logs'], sets['iv_after_collision'], sets['iv_preemptive']),
        '|---|---|---|---|---|---|']
    for n in RUNLAB:
        m = M[n]
        out.append('| %s | %s | %s | %s | %s | %s |' % (RUNLAB[n], f(m['iv_all']['l1_us'], 2), f(m['iv_log_heldout']['l1_us'], 2),
                                                    f(m['iv_train_logs']['l1_us'], 2), f(m['iv_after_collision']['l1_us'], 2),
                                                    f(m['iv_preemptive']['l1_us'], 2)))
    return '\n'.join(out)


def t_gate(r):
    G, LT = r['gates'], r['latency']
    out = ['| 学習結果 | エンジンの確認 | 精度 最大 / p99 [µs] | run_numpy 中央 / p99 [ms] | エンジンだけ 中央 [ms] | 走行ループ / カメラ → PWM の見積もり [ms] | 学習のメモリ [MB] | ゲート |',
           '|---|---|---|---|---|---|---|---|']
    for n in RUNLAB:
        g, l = G['per_run'][n], LT['per_run'][n]
        src = '10/12' if n in ('A_s0', 'A_s1', 'B') else (g.get('checked_at') or '')[5:16]
        out.append('| %s | %s | %s / %s | %s / %s | %s | %s / %s | %s | %s |' % (
            RUNLAB[n], src, f(g.get('actions_max_us'), 2), f(g.get('actions_p99_us'), 2), f(l['run_numpy_median_ms'], 2),
            f(l['run_numpy_p99_ms'], 2), f(l['engine_median_ms'], 2), f(l['loop_estimate_ms'], 1),
            f(l['cam_to_pwm_estimate_ms'], 1), f(g.get('peak_mem_mb'), 0), '合格' if g['pass'] else '**不合格**'))
    return '\n'.join(out)


def t_train(r):
    T = r['training']
    out = ['| 腕・種 | 学習結果 | エポック | 学習中の検証 best(エポック)/ 最後 [µs] | 1 エポック 中央 [秒] |', '|---|---|---|---|---|']
    for k, v in T.items():
        out.append('| %s | `%s` | %s | %s(%s)/ %s | %s |' % (k.replace('/s', ' 種 '), v['run'], v['epochs'] or '—', f(v['best_val_us'], 1),
                                                          v['best_epoch'], f(v['last_val_us'], 1), f(v['epoch_s_median'], 1)))
    return '\n'.join(out)


def t_repro(r):
    R = r['reproduce_1012']
    out = ['| 項目 | この評価 / 10/12 |', '|---|---|']
    for k in ('L_best', 'L_last', 'r1_window_best', 'de_window_best', 'straight_fwd', 'robust_dL1'):
        out.append('| %s | %s |' % (k, '、'.join('%s %s / %s' % (RUNLAB[n], v[0], v[1]) for n, v in R[k].items())))
    out.append('| s_A best / last | %s / %s、%s / %s |' % tuple(R['s_A']))
    out.append('| 陰性対照の差の平均 R1・DE | %s |' % '、'.join('%s %s %s / %s' % (it.upper(), k, v[0], v[1])
                                                    for it, d in R['nc_d_mean'].items() for k, v in d.items()))
    out.append('| すべて一致 | **%s** |' % ('はい' if R['all_match'] else 'いいえ'))
    return '\n'.join(out)


ALL = [('主指標', t_primary), ('腕どうし', t_secondary), ('ジャンル別', t_genre), ('速さの癖', t_habit),
       ('R1', lambda r: t_pass(r, 'r1')), ('DE', lambda r: t_pass(r, 'de')), ('崩れ', t_collapse), ('ずれ', t_robust),
       ('シミュレーター', t_sim), ('交代', t_iv), ('ゲート', t_gate), ('学習', t_train), ('再現', t_repro)]


def main():
    r = json.loads((REPORT / 'evaluation.json').read_text(encoding='utf-8'))
    text = '\n\n'.join('## %s\n\n%s' % (k, fn(r)) for k, fn in ALL)
    (REPORT / 'eval' / 'tables.md').write_text(text + '\n', encoding='utf-8')
    print(text if '-v' in sys.argv else REPORT / 'eval' / 'tables.md')


if __name__ == '__main__':
    main()
