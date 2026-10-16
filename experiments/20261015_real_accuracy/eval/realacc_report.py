#!/usr/bin/env python3
"""evaluation.json → evaluation.md(日本語)。realacc_eval.py evaluate から呼ぶ(CPU・numpy なし)"""

import json
from pathlib import Path

CLS = {'better': '良い', 'equivalent': '同等', 'worse': '悪い', None: '—'}
PROTOCOL = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/protocol.json')
SEC_LABEL = {'S1_no_battery_scene': 'S1 電池のシーンを除く', 'S2_k3_steer': 'S2 3 フレーム先のステア',
             'S3_intervention_part1': 'S3 介入の集合', 'battery_scene_only': '電池のシーンだけ', 'genre4': '高速(ジャンル 4)',
             'genre5': '低速(ジャンル 5)', 'k3_throttle': '3 フレーム先のスロットル'}


def f3(x):
    return '—' if x is None else '%.3f' % x


def f2(x):
    return '—' if x is None else '%.2f' % x


def seeds(v):
    return ' / '.join(f3(x) for x in v) if v else '—'


def q_rows(qs, include_pending=True):
    out = ['| 問い | 腕 best [µs] | 基準 best [µs] | 平均の差 best(閾値) | 腕 last | 基準 last | 差 last(閾値) | best | last | **最終** |'
           ' 実用上の差 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for q in qs:
        if q['pending']:
            if include_pending:
                out.append('| %s | (まだ) | | | | | | | | **未** | |' % q['label'])
            continue
        p = q['primary']
        out.append('| %s | %s | %s | %+.3f(%.3f) | %s | %s | %+.3f(%.3f) | %s | %s | **%s** | %s |' % (
            q['label'], seeds(p['arm_best']), seeds(p['ref_best']), -p['best']['gain_us'], p['best']['threshold_us'],
            seeds(p['arm_last']), seeds(p['ref_last']), -p['last']['gain_us'], p['last']['threshold_us'],
            CLS[p['best']['class']], CLS[p['last']['class']], CLS[p['final_class']], 'あり' if p['practical'] else 'なし'))
    return out


def sec_rows(qs):
    keys = ('S1_no_battery_scene', 'S2_k3_steer', 'S3_intervention_part1', 'battery_scene_only', 'genre4', 'genre5', 'k3_throttle')
    out = ['| 問い | ' + ' | '.join(SEC_LABEL[k] for k in keys) + ' | S3 の拒否権 |', '|---' * (len(keys) + 2) + '|']
    for q in qs:
        if q['pending']:
            continue
        cells = []
        for k in keys:
            s = q['secondary'][k]
            cells.append('%s(%+.2f)' % (CLS[s['final_class']], -s['best']['gain_us']))
        out.append('| %s | %s | %s |' % (q['label'], ' | '.join(cells), 'あり' if q['veto_S3'] else 'なし'))
    return out


def write_md(res, path):
    proto = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    Q, R = res['questions'], res['reference_questions']
    L = ['# 実機の精度の試し(real_accuracy_20261015)の評価', '',
         '- 作成: %s(`experiments/20261015_real_accuracy/eval/realacc_eval.py evaluate`、道具の sha256 %s…)' % (
             res['created_at'], res['tool']['sha256'][:12]),
         '- 手順: protocol.json の sha256 %s…(固定したものと%s)' % (res['protocol']['sha256'][:12],
                                                          '同じ' if res['protocol']['unchanged'] else '**違う**'),
         '- 主指標: 保留の人のきれいな走行 %d フレームの L1 [µs](10 ステップ × [ステア, スロットル])。小さいほど良い' % res['frames']['primary'],
         '- 判定(10/13 の規則のまま): 「良い」= 両方の種が基準の両方より小さく、平均の改善 > 閾値(best 0.626・last 1.322 µs)。'
         'best と last の両方で「良い」のときだけ最終「良い」。実用上の差 = 10 µs',
         '- 差 = 腕 − 基準(負 = 腕のほうが良い)。値は 2 つの種(左から順)', '']
    cc = res.get('crosscheck') or {}
    L.append('- crosscheck: A_s0 の best を新しい道具で予測し直し、10/12 の予測との最大の差 %s µs(上限 0.01)→ %s' % (
        cc.get('max_abs_us'), '合格' if cc.get('pass') else '**不合格**'))
    L.append('- 基準の再現: A_s0・A_s1・B の主指標・副指標が protocol.json の値と %s' % (
        '全部一致(±0.0015 µs)' if res['reproduce_baseline']['all_match'] else '**一致しない**'))
    L.append('- 予測がある学習結果: %s' % '・'.join(res['have_predictions']))
    L += ['', '## 1. 判定(主指標)', ''] + q_rows(list(Q.values()))
    if res.get('single_seed_reference') and Q.get('GAINL_vs_Apost', {}).get('pending'):
        L += ['', '- GAINL の行が「まだ」なのは、gainl-s1 が電源断(10/15 23:46)で 8 エポック目までしか進まず、削ったため。'
              '規則は 2 つの種を求めるので判定しない(種 0 だけの参考は 4b)']
    L += ['', '## 2. 副指標(同じ規則・閾値は A と B から先に決めた値)', '',
          '括弧の中は best の差(腕 − 基準)[µs]。S3(介入の集合)で「悪い」は拒否権(実車のカードに載せない)。', ''] + sec_rows(list(Q.values()))
    g = res['gain']['verdict']
    L += ['', '## 3. GAIN(電池の効きの補正)と TE(時間方向の平均)', '']
    if g:
        L.append('- GAIN: G1(主指標で組 (i)・(ii) の両方が「良い」)= %s、G2(S1 で「悪い」なし)= %s、G3(S3 で「悪い」なし)= %s → '
                 '**%s**' % ('○' if g['G1'] else '×', '○' if g['G2'] else '×', '○' if g['G3'] else '×',
                             '実車の A/B/A のカードの候補' if g['card'] else 'カードには載せない'))
        for qid, v in g['battery_scene_share_of_gain'].items():
            L.append('  - %s: 主指標の改善 %.2f µs のうち、電池の減ったシーン 1 つ(%s)から %.2f µs(%s)' % (
                qid, v['gain_us'], '20261010-193638-0003', v['from_battery_scene_us'],
                '—' if v['share'] is None else '%.0f%%' % (100 * v['share'])))
        L.append('  - **主指標の改善はほぼ 1 シーンから**。補正の比 g_t/g_ref(シーンの平均): ' + '、'.join(
            '%s %.2f' % (s[-11:], v) for s, v in res['reference']['gain_scale_by_scene'].items()))
        L.append('  - g_ref(固定)%s、学習の 41 シーンで計算し直すと %s' % (
            res['gain']['g_ref'], {k: round(v, 2) for k, v in res['reference']['g_ref_recomputed'].items()}))
    else:
        L.append('- GAIN: まだ(v0-s2・v0-s3 の予測が要る)')
    t = res['te']['verdict']
    if t:
        L.append('- TE(j = 2・m = 0): T1(主指標で両方の組が「良い」)= %s、T3(S3 で「悪い」なし)= %s → **%s**' % (
            '○' if t['T1'] else '×', '○' if t['T3'] else '×', '走行の既定を見直す候補' if t['turn_on'] else '走行の既定(オフ)のまま'))
    s = res.get('size')
    L += ['', '## 4. SIZE(データの量の曲線。測定で、良し悪しの判定はしない)', '']
    if s:
        L.append('| 指標 | L(Q) 29% | L(H) 52% | L(T) 79% | L(全部) | s1 [µs/2 倍] | s2 [µs/2 倍] | G2_lin | G2_pow | 読み |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')
        for k, row in s['by_metric'].items():
            sl = row['slopes']
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                k, f3(row['L'].get('Q')), f3(row['L'].get('H')), f3(row['L'].get('T')), f3(row['L'].get('full')), f2(sl['s1']),
                f2(sl['s2']), f2(sl['G2_lin']), f2(sl['G2_pow']), row['reading']))
        b = s['bootstrap_primary_best']
        t_rows = ['%s %.2f' % (k, row['s2_T']) for k, row in s['by_metric'].items() if row.get('s2_T') is not None]
        if t_rows:
            L.append('')
            L.append('- 参考(任意の点 T = 79%%): s2_T = (L(T) − L(全部)) / log2(N_全部 / N_T) [µs / 2 倍]: %s' % '、'.join(t_rows))
        L.append('')
        L.append('- 読み(主指標・best、protocol の analysis.readings): **%s**' % s['reading_text'])
        L.append('- シーン単位のブートストラップ(2,000 回): L(H) − L(全部) の 95%% 区間 [%.2f, %.2f]、G2_lin [%.2f, %.2f]、'
                 'G2_pow [%.2f, %.2f]。読みの割合 低い %.0f%%・中 %.0f%%・高い %.0f%%' % (
                     *b['LH_minus_Lfull_ci95'], *b['G2_lin_ci95'], *b['G2_pow_ci95'], 100 * b['reading_share']['low_value'],
                     100 * b['reading_share']['middle'], 100 * b['reading_share']['high_value']))
        L.append('- 注意: 小さい部分集合も 30 エポックのまま(勾配のステップは少ない)。点は 3 つ(T があれば 4 つ)なので見積もりは粗い')
        eq = s.get('equal_steps_reference')
        if eq:
            L.append('- **固定の外の参考 SIZE-EQ**(結果を見た後に足した。勾配のステップ数を全部とそろえた: H 59・Q 106 エポック。'
                     '読みは変えない): %s' % eq['text'])
            for k, row in eq['by_metric'].items():
                sl = row.get('slopes') or {}
                L.append('  - %s: L(Q_eq) %s・L(H_eq) %s・L(全部) %s、s2_eq %.2f%s' % (
                    k, f3(row['L'].get('Q')), f3(row['L'].get('H')), f3(row['L'].get('full')), row['s2_eq'],
                    '、s1 %.2f・G2_pow %.2f(同じ式なら %s)' % (sl['s1'], sl['G2_pow'], row['reading_same_formula']) if sl else ''))
    else:
        L.append('- まだ(sizeH・sizeQ の予測が要る)')
    ssr = res.get('single_seed_reference')
    if ssr:
        L += ['', '## 4b. 種 1 本だけの参考(電源断で 2 本目を削った腕。判定はしない)', '',
              '- %s。削った: %s' % (ssr['note'], '・'.join(ssr['cut']))]
        g1 = ssr.get('GAINL_s0')
        if g1:
            L.append('- **GAINL**(任意の腕): %s' % g1['text'])
            L.append('')
            L.append('| 指標 | ckpt | GAIN(gainl-s0) | GAIN(A_s0) | 差 | 閾値 | 後処理なし gainl-s0 − A_s0 |')
            L.append('|---|---|---|---|---|---|---|')
            for ck in ('best', 'last'):
                for m, lab in (('primary', '主指標'), ('S1_no_battery_scene', 'S1 電池のシーンを除く'),
                               ('S3_intervention_part1', 'S3 介入の集合'), ('battery_scene_only', '電池のシーンだけ')):
                    a = g1['post'][ck][m]
                    L.append('| %s | %s | %s | %s | %+.3f | %s | %+.3f |' % (
                        lab, ck, f3(a['arm']), f3(a['ref']), a['diff'], '—' if a['threshold'] is None else '%.3f' % a['threshold'],
                        g1['raw_vs_A_s0'][ck][m]['diff']))
        e1 = ssr.get('SIZE_EQ_s0')
        if e1:
            L.append('')
            L.append('- **SIZE-EQ**(固定の外の参考の腕。21:55 に決めた読み方を種 0 だけに当てはめた): %s' % e1['text'])
            for k, row in e1['by_metric'].items():
                L.append('  - %s: L(H, 30 エポック) %s・L(H, 59 エポック) %s・L(全部 = A_s0) %s → s2 %.2f・s2_eq %.2f µs / 2 倍'
                         '(H と全部の差のうちステップ数で埋まった割合 %s)' % (
                             k, f3(row['L_H_s0_30ep']), f3(row['L_Heq_s0_59ep']), f3(row['L_full_A_s0']), row['s2_s0'],
                             row['s2_eq_s0'], '—' if row['share_of_gap_from_steps'] is None
                             else '%.0f%%' % (100 * row['share_of_gap_from_steps'])))
    gate = res.get('ens_gate')
    L += ['', '## 5. ENS のエンジンのゲート(P2 = v0-s2 + msf-s2 を 1 つのエンジンに)', '']
    if gate:
        L.append('- 合格: **%s**。PyTorch FP32(CPU の 2 本の平均)との差 最大 %s µs(≤ 2.5)、run_numpy 中央値 %s ms(≤ 6)・p99 %s ms(≤ 8)、'
                 '有限 %s' % ('○' if gate.get('all_gates') else '×', (gate.get('accuracy') or {}).get('actions_max_us'),
                              (gate.get('latency_ms') or {}).get('run_numpy_median'), (gate.get('latency_ms') or {}).get('run_numpy_p99'),
                              (gate.get('accuracy') or {}).get('finite')))
    else:
        L.append('- まだ・または削った')
    L += ['', '## 6. 参考の行(判定には使わない)', ''] + q_rows([q for q in R.values()], include_pending=False)
    ref = res['reference']
    if ref.get('pairs_4x4'):
        p = ref['pairs_4x4']
        L.append('')
        L.append('- v0 × msfusion の組み合わせ全部(%d 組)の平均: best %.3f・last %.3f µs' % (len(p['pairs']), p['mean_best'], p['mean_last']))
    if ref.get('seed_disagreement_us'):
        L.append('- 2 つの種の予測の食い違い(主指標のフレームの |差| の平均): ' + '、'.join(
            '%s %.1f µs' % (k, v) for k, v in ref['seed_disagreement_us'].items()))
    cq = res['collapse_reference']['per_question']
    if cq:
        L.append('- 崩れ(10/13 の規則・参考): ' + '、'.join('%s %s' % (k, 'あり' if v['collapse'] else 'なし') for k, v in cq.items()))
    if res.get('sensitivity_rule3'):
        sr = res['sensitivity_rule3']
        L.append('- 感度(結論の規則 3): FRESH-A が同等でないので s_A\' = %s で同じ判定 → %s' % (
            sr['s_A_prime_us'], {k: CLS[v['final_class']] for k, v in sr['rows'].items()}))
    ps = ref.get('per_scene_best') or {}
    if ps:
        cols = [c for c in ('A_s0', 'A_s1', 'P2', 'P3', 'SAME_A', 'SAME_new', 'GAIN(A_s0)', 'rot-s0', 'rot-s1') if c in ps]
        L += ['', '### シーン別の L1(best)[µs]', '', '| シーン | フレーム | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 2) + '|']
        for sc_, n in ref['per_scene_frames'].items():
            L.append('| %s | %d | %s |' % (sc_, n, ' | '.join('%.1f' % ps[c][sc_] for c in cols)))
    ph = proto['posthoc_already_seen_not_verdicts']
    L += ['', '## 7. 下調べ(手順を固める前に見た数字。判定ではない)', '',
          '- ENS(v0 + msfusion、10/12 の種 0・1): best %s → 計算し直し %s' % (ph['ENS_v0_plus_msfusion']['best'],
                                                                   seeds(R['PH_vs_A']['primary']['arm_best']) if not R['PH_vs_A']['pending'] else '—'),
          '- GAIN(5 秒): best %s、TE(一様): best %s、同じ形の平均(v0 s0+s1): %s、今ある 10 本の平均: %s µs' % (
              ph['GAIN_W5']['best'], ph['TE_J2_uniform']['best'], ph['SEED_ENS_v0_s0_s1']['best'], ph['all10_mean']),
          '- 相手に msfusion を選んだのは、同じ 4,670 フレームを見て 4 つの候補から(確かめは新しい種だけ)']
    c = res['conclusions']
    L += ['', '## 8. 結論(protocol の conclusion_rules)', '']
    for k in ('ENS', 'FRESH_A', 'ROT', 'GAIN', 'TE', 'SIZE', 'recommendation', 'open_loop', 'multiple_comparisons'):
        if k in c:
            L.append('- **%s**: %s' % (k, c[k]))
    L += ['', '## 9. 答え合わせ(学習の前に書いた予想)', '']
    pr = proto['predictions_before_results']
    outcome = {
        'ENS': None if Q['ENS_vs_A']['pending'] else '%s(差 %+.2f µs)・ENS vs SAME %s' % (
            CLS[Q['ENS_vs_A']['final_class']], -Q['ENS_vs_A']['primary']['best']['gain_us'],
            CLS[Q['ENS_vs_SAME'].get('final_class')]),
        'FRESH_A': None if Q['FRESH_A']['pending'] else CLS[Q['FRESH_A']['final_class']],
        'FRESH_MSF': None if Q['FRESH_MSF']['pending'] else CLS[Q['FRESH_MSF']['final_class']],
        'SIZE': None if not s else 'L(H) − L(全部) = %+.2f µs、G2_pow %.2f µs' % (
            s['by_metric']['primary/best']['slopes']['LH_minus_Lfull'], s['by_metric']['primary/best']['slopes']['G2_pow']),
        'ROT': None if Q['ROT_vs_A']['pending'] else '%s(差 %+.2f µs)' % (CLS[Q['ROT_vs_A']['final_class']],
                                                                       -Q['ROT_vs_A']['primary']['best']['gain_us']),
        'GAIN': None if not g else 'G1 %s・G2 %s・G3 %s(組 (i) の差 %+.2f µs)' % (
            g['G1'], g['G2'], g['G3'], -Q['GAIN_i']['primary']['best']['gain_us']),
        'TE': None if not t else '%s(組 (i) %s・(ii) %s)' % ('オンの候補' if t['turn_on'] else '同等・オフのまま',
                                                             CLS[Q['TE_i']['final_class']], CLS[Q['TE_ii']['final_class']]),
        'ENS_GATE': None if not gate else '%s・中央値 %s ms' % ('合格' if gate.get('all_gates') else '不合格',
                                                               (gate.get('latency_ms') or {}).get('run_numpy_median')),
    }
    L.append('| 項目 | 予想 | 結果 |')
    L.append('|---|---|---|')
    for k, v in pr.items():
        L.append('| %s | %s | %s |' % (k, v, outcome.get(k) or '(まだ・削った)'))
    L += ['', '## 10. 注意', '',
          '- どの結論も開いたループ(保留の記録)の結果で、実車の走りの改善は示さない。車は動かしていない',
          '- 確かめは新しい種で同じ規則に当てはめる形だけ(まだ開けていない別の保留は無かった)。保留の 4,670 フレームは下調べで見たもの',
          '- 多重比較: 判定する問いは ENS・ROT・GAIN・TE(・GAINL)。偽陽性は小さいが 0 ではない', '']
    Path(path).write_text('\n'.join(L), encoding='utf-8')
