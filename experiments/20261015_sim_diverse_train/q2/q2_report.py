#!/usr/bin/env python3
"""eval_real.json(+ recount.json・crosscheck.json)→ eval_real.md(日本語)。CPU だけ

    cd ~/jetformer && python experiments/20261015_sim_diverse_train/q2/q2_report.py
"""

import json
import os
import sys

REPORT = '/home/jetson/jetformer_reports/sim_diverse_train_20261015'
CJ = {'better': '良い', 'equivalent': '同等', 'worse': '悪い', None: '—'}


def f(x, nd=2, sign=False):
    if x is None:
        return '—'
    return ('%+.*f' if sign else '%.*f') % (nd, x)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else REPORT + '/eval_real.json'
    dst = sys.argv[2] if len(sys.argv) > 2 else REPORT + '/eval_real.md'
    R = json.load(open(src, encoding='utf-8'))
    rc = {}
    if os.path.exists(REPORT + '/q2/recount.json'):
        rc = json.load(open(REPORT + '/q2/recount.json', encoding='utf-8'))
    P, V = R['primary'], R['verdict']
    names = list(R['names'])
    cd = [n for n in names if n.startswith('CDIV')]
    single = R['single_seed']
    prov = '(暫定・種 1 つ)' if single else ''
    out = []
    w = out.append
    w('# Q2 — 実機の保留データの精度: いろいろな走りのシミュレーターのデータを混ぜた C-DIV(sim_diverse_train_20261015)')
    w('')
    w('- 作成 %s。決まりは `protocol.json`(sha256 %s…、固定のまま: %s)の eval_Q2。数字は `eval_real.json`。'
      % (R['created_at'], R['protocol']['sha256'][:8], 'はい' if R['protocol']['unchanged'] else '**いいえ**'))
    w('- 問い: %s' % R['question'])
    w('- 範囲: 実車で記録した保留データのオフラインの誤差だけ。**実車を走らせる試験はこの作業の範囲の外**(実車・PCA9685・I2C には触れていない)。')
    if R['missing']:
        w('- **まだ無い学習結果**: %s(%s)' % ('・'.join(R['missing']),
          'C-DIV の種 1 が無いので、判定は protocol の single_seed_fallback の「暫定(種 1 つ)」' if single else ''))
    w('')
    w('## 0. 要点')
    w('')
    for line in V['lines']:
        w('- ' + line)
    w('')
    w('| 比べ | best の差 [µs] | 閾値 best | last の差 [µs] | 閾値 last | 判定 |')
    w('|---|---:|---:|---:|---:|---|')
    for lab, key in (('C-DIV − A', 'CDIV_vs_A'), ('C-DIV − C', 'CDIV_vs_C'), ('C − A(10/13 の再現)', 'C_vs_A')):
        c = P[key]
        w('| %s | %s | %.3f | %s | %.3f | **%s**%s |' % (lab, f(c['diff_best_us'], 3, True), c['best']['threshold_us'],
                                                       f(c['diff_last_us'], 3, True), c['last']['threshold_us'],
                                                       CJ[c['final_class']], prov if key != 'C_vs_A' else ''))
    w('')
    w('差 = 腕 − 基準(+ = 腕のほうが誤差が大きい = 悪い)。「良い / 悪い」には、腕の全部の種が基準の全部の種より小さい(大きい)こと、かつ平均の差が閾値を超えることの両方が要る。'
      'best だけ「良い」は「同等」に下げる(10/13 の final_verdict)。')
    w('')
    w('## 1. 比べたもの')
    w('')
    w('| 腕 | 学習データ | 学習結果 | 予測 |')
    w('|---|---|---|---|')
    data = {'A': '実機 clean_driving_20261011 の 52 シーン(学習 41・検証 11)', 'B': '実機(今の走行のモデル。参考)',
            'C': '実機 + 前の教師(作者の RC 教師)の 60 シーン・36,000 フレーム(ほぼ同じ 20 秒の走りのくり返し)',
            'C-DIV': '実機 + 5 モード(最速・速さいろいろ・絶対壁にぶつからない・離脱復活・経路いろいろ)の学習の 12 シーンずつ = 60 シーン・36,000 フレーム'}
    for n in names:
        v = R['names'][n]
        w('| %s | %s | `%s` | %s |' % (n, data[v['arm']], v['run'], v['pred_source']))
    w('')
    w('作り方は A・C と同じ(jetformer_v0・30 エポック・lr 3e-4・間引き 3・バッチ 64・今の操作値なし・2.5 G の衝突の除外・最良は実機の検証 1,559 サンプル)。'
      'C-DIV はシミュレーターの量を C と同じにしたので、C との差は「中身」(いろいろな走り。ただし先生が RC 教師から RL の方策に替わったことも含む)の差。')
    w('')
    w('## 2. 主指標(保留した実機の人のきれいな走行 %d シーン・%s フレーム)' % (R['frames']['human_clean_scenes'],
                                                       format(R['frames']['human_clean'], ',')))
    w('')
    w('![主指標](fig/q2_primary.png)')
    w('')
    w('| 学習結果 | L best [µs] | L last [µs] | best − last |')
    w('|---|---:|---:|---:|')
    for n in names:
        w('| %s | %.3f | %.3f | %s |' % (n, P['L_us']['best'][n], P['L_us']['last'][n], f(P['best_minus_last_us'][n], 3, True)))
    w('')
    w('| 腕 | 平均 best | [最小, 最大] | 平均 last | 種の範囲 best |')
    w('|---|---:|---|---:|---:|')
    for arm in ('A', 'C', 'C-DIV'):
        w('| %s | %.3f | [%.3f, %.3f] | %.3f | %.3f |' % (arm, P['arm_mean_us'][arm]['best'], P['arm_range_us'][arm]['best'][0],
                                                        P['arm_range_us'][arm]['best'][1], P['arm_mean_us'][arm]['last'],
                                                        P['seed_range_us'][arm]['best']))
    w('')
    w('- モデル B − A の平均(best): %s µs(参考)。' % f(P['B_minus_A_best_us'], 3, True))
    w('- 閾値: A と比べるときは 2 × s_A(best 0.313 → %.3f µs・last 0.661 → %.3f µs)、C と比べるときは 2 × max(C の種の範囲 0.477, s_A)'
      '(best %.3f µs・last %.3f µs)。どれも結果を見る前に protocol で固定した値。' % (
          P['threshold_vs_A_us']['best'], P['threshold_vs_A_us']['last'], P['CDIV_vs_C']['best']['threshold_us'],
          P['CDIV_vs_C']['last']['threshold_us']))
    w('- 実用上の差(≥ 10 µs): A との差 %s µs → %s。' % (f(P['CDIV_vs_A']['diff_best_us'], 2, True),
                                                 'あり' if P['CDIV_vs_A']['practical'] else 'なし'))
    w('- 種と checkpoint の向きがそろうか(記述): A と比べて %s、C と比べて %s。' % (
        {'worse': '全部の種 × {best, last} で A より大きい', 'better': '全部で A より小さい', None: 'そろわない'}[P['CDIV_vs_A']['consistent_direction']],
        {'worse': '全部で C より大きい', 'better': '全部で C より小さい', None: 'そろわない'}[P['CDIV_vs_C']['consistent_direction']]))
    w('')
    va = P['CDIV_vs_A']
    w('### 2.1 判定の際どさ(記述。判定は変えない)')
    w('')
    for ck in ('best', 'last'):
        b = va[ck]
        w('- %s: C-DIV の種 [%.4f, %.4f]・A の種 [%.4f, %.4f]。「全部の種が A の全部の種より大きい」の余裕(C-DIV の最小 − A の最大)= %+.4f µs、'
          '平均の差 %+.3f µs(閾値 %.3f µs)→ この checkpoint の分類「%s」。' % (
              ck, b['arm_min'], b['arm_max'], b['ref_min'], b['ref_max'], b['arm_min'] - b['ref_max'], -b['gain_us'],
              b['threshold_us'], CJ[b['class']]))
    if va['final_class'] == 'worse' and va['best']['arm_min'] - va['best']['ref_max'] < 0.05:
        w('- **「悪い」は固定した規則どおりの判定だが、際どい**: 種の並びの条件は %.3f µs の差でかろうじて満たされ(C-DIV 種 1 と A 種 1 がほぼ同じ)、'
          '平均の差の多くは C-DIV 種 0(%.3f µs)から来ている。last.pth では平均の差が閾値より小さいので「同等」(種の並びの条件もわずかに満たさない。規則では「悪い」は best.pth のまま)。'
          '差は %.1f%%・10 µs よりずっと小さい。読み方は「A より少し悪い側(実用上の差ではない)」。' % (
              va['best']['arm_min'] - va['best']['ref_max'], P['L_us']['best'].get('CDIV_s0', float('nan')), va['diff_pct_best']))
    w('')
    w('### 2.2 種 1 があるか・削ったときの判定')
    w('')
    s0 = P['seed0_only']
    w('- 種 0 だけで判定すると(protocol の single_seed_fallback = 種 0 の 1 本を基準の 2 本と比べる): A と「%s(暫定)」(best の差 %s µs)、'
      'C と「%s(暫定)」(best の差 %s µs)。' % (CJ[s0['vs_A']['final_class']], f(s0['vs_A']['diff_best_us'], 3, True),
                                         CJ[s0['vs_C']['final_class']], f(s0['vs_C']['diff_best_us'], 3, True)))
    rule_txt = ('種 1 を削ったときの規則(protocol の single_seed_fallback): 「良い(暫定)」= 種 0 が基準の 2 本の小さいほうより小さく、かつ基準の平均との差が閾値を超える。'
                '「悪い(暫定)」はその逆。ほかは「同等(暫定)」。2 本の規則の「全部の種が基準の全部の種の外」という並びの条件が種 0 の 1 本だけで判断され、'
                '腕の種どうしのばらつきが判定に入らない(閾値は同じ)。')
    if single:
        w('- **C-DIV の種 1 は無い(削った・まだ)**。上の判定は「暫定(種 1 つ)」。' + rule_txt +
          '10/13 の C を同じ規則で種 0 だけで判定すると「悪い」、種 1 だけなら「同等」になり、2 本そろえた本判定(同等)と食い違う。'
          'したがって種 1 つの「良い / 悪い」は弱い証拠で、種 1 を足すと「同等」に変わりうる。')
    else:
        same = (s0['vs_A']['final_class'] == P['CDIV_vs_A']['final_class'] and s0['vs_C']['final_class'] == P['CDIV_vs_C']['final_class'])
        w('- **種 1 は削らずに学習・評価できた**ので、判定は本判定(暫定ではない)。' + rule_txt)
        if same:
            w('- 種 0 だけでも分類は同じ(A と「%s」・C と「%s」)だった。ただし種 0 だけなら A との差は %s µs に見え、種 1(%.3f µs)を足すと平均の差は %s µs に縮み、'
              '種の並びの条件も 2.1 のとおり際どくなった。種 1 を削っていれば、同じ「悪い」でも実際より強い差として(暫定で)報告していたことになる。'
              'C-DIV の種どうしの差は %.2f µs で、A(%.2f)・C(%.2f)より大きく、種 1 つでは見えないこのばらつきが今回の判定の際どさの元になっている。' % (
                  CJ[P['CDIV_vs_A']['final_class']], CJ[P['CDIV_vs_C']['final_class']], f(s0['vs_A']['diff_best_us'], 3, True),
                  P['L_us']['best'].get('CDIV_s1', float('nan')), f(P['CDIV_vs_A']['diff_best_us'], 3, True),
                  P['seed_range_us']['C-DIV']['best'], P['seed_range_us']['A']['best'], P['seed_range_us']['C']['best']))
        else:
            w('- **種 0 だけの判定は本判定と違う**(種 0 だけ: A と「%s」・C と「%s」)。種 1 つの規則は種どうしのばらつきを見られないので、'
              '種 1 を削ると判定が変わっていたことを示している。' % (CJ[s0['vs_A']['final_class']], CJ[s0['vs_C']['final_class']]))
    w('')
    w('## 3. ジャンル別・速さの癖(参考)')
    w('')
    w('![ジャンル別](fig/q2_genre.png)')
    w('')
    M = R['accuracy']['metrics']
    w('| 学習結果 | 高速(4)L1 | 高速 前進の偏り | 低速(5)L1 | 低速 前進の偏り | ステアの向きの正解率 |')
    w('|---|---:|---:|---:|---:|---:|')
    for n in names:
        g4, g5 = M[n]['human_clean_genre4'], M[n]['human_clean_genre5']
        w('| %s | %.2f | %s | %.2f | %s | %.4f |' % (n, g4['l1_us'], f(g4['fwd_bias_us'], 1, True), g5['l1_us'],
                                                     f(g5['fwd_bias_us'], 1, True), M[n]['human_clean']['steer_sign_acc']))
    w('')
    H = R['speed_habit']
    for arm in ('C', 'C-DIV'):
        w('- %s: ジャンル平均の前進の偏りの A からのずれ 高速 %s µs・低速 %s µs → 速さの癖の移り(≥ 10 µs)%s。' % (
            arm, f(H[arm]['4']['shift_us'], 1, True), f(H[arm]['5']['shift_us'], 1, True),
            'あり' if (H[arm]['4']['moved'] or H[arm]['5']['moved']) else 'なし'))
    hs = H.get('low_speed_straight_HS')
    if hs:
        w('')
        w('![低速の直線](fig/q2_speed_hs.png)')
        w('')
        fb = hs['fwd_bias_per_run_us']
        w('- **低速の直線 HS(%d フレーム。先に決めた参考の測り)**: 前進の偏り A %s、C %s、C-DIV %s µs(負 = 人より強く前進)。'
          'A の平均からのずれ: C %s µs・C-DIV %s µs(目安 10 µs を%s)。この直線の L1 の A との差: C %s µs・C-DIV %s µs。' % (
              hs['n'], ' / '.join(f(fb[n], 1, True) for n in ('A_s0', 'A_s1')),
              ' / '.join(f(fb[n], 1, True) for n in ('C_s0', 'C_s1')), ' / '.join(f(fb[n], 1, True) for n in cd),
              f(hs['C_shift_us'], 1, True), f(hs['CDIV_shift_us'], 1, True), '超える' if hs['CDIV_moved'] else '超えない',
              f(hs['L1_C_minus_A_us'], 2, True), f(hs['L1_CDIV_minus_A_us'], 2, True)))
    if hs and hs.get('contribution_to_overall_us'):
        cb = hs['contribution_to_overall_us']
        w('- 全体の差(best)のうち、この直線(主指標のフレームの %.0f%%)から来た分: C-DIV %s µs(全体の差 %s µs の %.0f%%)、C %s µs(全体 %s µs の %.0f%%)。' % (
            100.0 * hs['n'] / R['frames']['human_clean'], f(cb['C-DIV'], 2, True), f(P['CDIV_vs_A']['diff_best_us'], 2, True),
            100.0 * cb['C-DIV'] / P['CDIV_vs_A']['diff_best_us'] if P['CDIV_vs_A']['diff_best_us'] else float('nan'),
            f(cb['C'], 2, True), f(P['C_vs_A']['diff_best_us'], 2, True),
            100.0 * cb['C'] / P['C_vs_A']['diff_best_us'] if P['C_vs_A']['diff_best_us'] else float('nan')))
    w('- 場所 × ジャンルの全部のマス(100 フレーム以上)で、前進の偏りの A からのずれの最大: C %s µs・C-DIV %s µs。' % (
        f(H['max_abs_fwd_shift_cell_us']['C'], 1), f(H['max_abs_fwd_shift_cell_us']['C-DIV'], 1)))
    w('')
    w('## 4. ずれへの強さ(参考)')
    w('')
    w('![ずれへの強さ](fig/q2_robust.png)')
    w('')
    rb = R['robustness']['per_arm']
    w('| 腕 | ΔL1 平均 [µs] | 種 | A の何倍 | 「ずれに強い」 |')
    w('|---|---:|---|---:|---|')
    w('| A | %.3f | %s | 1.00 | — |' % (rb['A']['dL1_mean_us'], ' / '.join('%.3f' % x for x in rb['A']['dL1_seeds_us'])))
    for arm in ('C', 'C-DIV'):
        w('| %s | %.3f | %s | %.2f | %s |' % (arm, rb[arm]['dL1_mean_us'], ' / '.join('%.3f' % x for x in rb[arm]['dL1_seeds_us']),
                                            rb[arm]['ratio_to_ref'], 'はい' if rb[arm]['robust_good'] else 'いいえ'))
    w('')
    w('規則: %s。ずれは画像 3 行下・上、明るさ ×0.8・×1.2、ToF 1 行下(1,559 フレーム、best.pth)。' % R['robustness']['rule'])
    w('')
    w('## 5. 難しい場所 R1・DE(参考。判定に使わない)')
    w('')
    w('![R1・DE](fig/q2_r1_de.png)')
    w('')
    w('| 比べ | R1 差の平均 [µs] | R1 99.375% 区間 | R1 良い/悪い | DE 差の平均 [µs] | DE 99.375% 区間 | DE 良い/悪い |')
    w('|---|---:|---|---|---:|---|---|')
    for lab in R['r1']['compare']:
        a, b = R['r1']['compare'][lab], R['de']['compare'][lab]
        w('| %s | %s | [%s, %s] | %s | %s | [%s, %s] | %s |' % (
            lab, f(a['d_mean'], 1, True), f(a['ci_good'][1], 1), f(a['ci_good'][2], 1), '良い' if a['good'] else ('悪い' if a['bad'] else '—'),
            f(b['d_mean'], 1, True), f(b['ci_good'][1], 1), f(b['ci_good'][2], 1), '良い' if b['good'] else ('悪い' if b['bad'] else '—')))
    w('')
    nparts = {it: {lab: sum(bool(x) for x in R[it]['compare'][lab]['good_parts'].values()) for lab in R[it]['compare']} for it in ('r1', 'de')}
    w('- どの比べも「良い」(4 つの条件が全部)にも「悪い」にもならなかった。満たした「良い」の条件の数: R1 %s、DE %s(4 つのうち)。'
      '同じ作り方の種どうしでも R1 7.7・DE 2.8 µs の差がある(10/12 の陰性対照)ので、参考にとどめる。' % (
          '・'.join('%s %d' % (k, v) for k, v in nparts['r1'].items()), '・'.join('%s %d' % (k, v) for k, v in nparts['de'].items())))
    c = R['r1_de_center']
    w('- C-DIV の中心(steer_center / throttle_neutral): %s。A は %s。%s' % (
        ' / '.join('%s %s' % (k, v) for k, v in c['CDIV_refs_us'].items()), c['A_ref_us'],
        '同じなので A の中心での出し直しは不要。' if not c['differs'] else 'A と違うので、A の中心でも出した(eval_real.json の r1_de_center.with_A_center)。'))
    w('')
    w('## 6. 崩れ(10/13 と同じ規則)')
    w('')
    for arm in ('C', 'C-DIV'):
        cc = R['collapse']['per_arm'][arm]
        w('- %s: 崩れ%s(直線 %d フレームの前進の強さ %.1f µs、A %.1f µs、低下 %.1f%%、前進の偏り %s µs、向きの正解率 %.4f、A %.4f)。' % (
            arm, 'あり' if cc['collapse'] else 'なし', R['collapse']['straight_frames'], cc['fwd_strength_us'], cc['ref_fwd_strength_us'],
            100 * cc['straight_drop'], f(cc['fwd_bias_us'], 1, True), cc['sign_acc'], cc['ref_sign_acc']))
    w('')
    w('## 7. シミュレーターの L1(参考)')
    w('')
    if os.path.exists(REPORT + '/fig/q2_sim_l1.png'):
        w('![シミュレーターの L1](fig/q2_sim_l1.png)')
        w('')
    for key in ('sim_old_val12', 'sim_modes_val15'):
        S = R[key]
        w('**%s**(%d シーン・%s フレーム)' % (S['label'], S['scenes'], format(S['frames'], ',')))
        w('')
        w('| 腕 | 頭 4 | 頭 5 | 頭 8 |')
        w('|---|---:|---:|---:|')
        for arm in ('A', 'B', 'C', 'C-DIV'):
            row = S['per_arm'].get(arm) or {}
            w('| %s | %s | %s | %s |' % (arm, *[('%.1f' % row['h%d' % h]['l1_mean_us']) if 'h%d' % h in row else '—'
                                                for h in (4, 5, 8)]))
        w('')
        if key == 'sim_modes_val15':
            w('モードごと(頭 8。A・B は頭 5)の L1 [µs]:')
            w('')
            labs = None
            rows = []
            for arm, h in (('A', 5), ('B', 5), ('C', 8), ('C-DIV', 8)):
                row = (S['per_arm'].get(arm) or {}).get('h%d' % h)
                if not row:
                    continue
                pdl = row['per_dataset_l1_mean_us']
                labs = labs or list(pdl)
                rows.append('| %s(頭 %d) | %s |' % (arm, h, ' | '.join('%.1f' % pdl[l] for l in labs)))
            if labs:
                w('| 腕 | %s |' % ' | '.join(labs))
                w('|---|%s|' % '|'.join(['---:'] * len(labs)))
                for r in rows:
                    w(r)
                w('')
    w('- 前の教師の val 12 シーンは C の頭 8 の学習と同じ方策・同じ線(保留ではあるが「同じ走りのくり返し」)。5 モードの val 15 シーンは C-DIV の学習に入っていない(保留)。'
      'A・B はジャンル数 6 で頭 8 が無いので、頭 4・5 で予測した値(シミュレーターの画像を実機の頭で見たときの誤差)。')
    w('')
    w('## 8. シーンごと(参考)')
    w('')
    ps = R['per_scene']
    w('- C-DIV が A より小さかったシーン: %d / %d(C は %d / %d)。' % (ps['scenes_CDIV_better_than_A'], ps['scenes'],
                                                         ps['scenes_C_better_than_A'], ps['scenes']))
    for s, v in ps['C_bad_scenes_1013'].items():
        w('- 10/13 で C が悪かった `%s`(%d フレーム): C − A %s µs、C-DIV − A %s µs、C-DIV − C %s µs。' % (
            s, v['n'], f(v['C_minus_A_us'], 2, True), f(v['CDIV_minus_A_us'], 2, True), f(v['CDIV_minus_C_us'], 2, True)))
    w('')
    w('## 9. ゲート・推論')
    w('')
    w('- ' + R['gates']['note'])
    for n in cd:
        g = R['gates']['per_run'][n]
        if 'parts' not in g:
            continue
        w('- %s: 予測が有限 %s・学習のピークメモリ %s MB(≤ 4096 %s)・出力の範囲 %.0f〜%.0f µs・パラメーター %.2f M(C と同じ大きさ)・ONNX %s・FP16 エンジン %s。' % (
            n, 'はい' if g['parts']['finite'] else 'いいえ', g['peak_mem_mb'], '合格' if g['parts']['train_mem'] else '不合格',
            g['pred_min_us'], g['pred_max_us'], g['params_total_m'], 'あり' if g['onnx_exists'] else 'なし',
            '作った' if g['engine_built'] else '作っていない(protocol)'))
        oc = g.get('onnx_check')
        if oc:
            w('  - ONNX(FP32・onnxruntime CPU)と PyTorch FP32 の差: 保留の実機 %d フレームで最大 %s µs(目安 ≤ %.1f µs → %s)。'
              'これは ONNX の書き出しの確かめで、FP16 エンジンのゲートの代わりにはならない。' % (
                  oc['frames'], oc['max_abs_us'], oc['gate_us'], '合格' if oc['pass'] else '不合格'))
    gc_ = R['gates']['per_run'].get('C_s0', {})
    w('- 参考: C の種 0 は 10/13 にエンジンを作ってゲート合格(run_numpy の中央値 %s ms)。C-DIV は同じ jetformer_v0・同じ大きさなので、速さは同じくらいと見込むが測っていない。'
      % gc_.get('run_numpy_median_ms'))
    w('')
    w('## 10. 学習(学習の段の値)')
    w('')
    for k, v in R['training'].items():
        if k.startswith('result_') or 'CDIV' in k or k.startswith('C-DIV') or k.startswith('C/') or k.startswith('A/'):
            items = ['%s %s' % (kk, vv) for kk, vv in v.items() if kk in ('train_samples', 'val_samples', 'best_val_us', 'best_epoch',
                                                                         'last_val_us', 'last_train_us', 'epoch_s_median', 'peak_mem_mb',
                                                                         'wall_s', 'workers', 'attempts', 'resumes',
                                                                         'steer_center_us_trained', 'throttle_neutral_us_trained')
                     and vv is not None]
            if items:
                w('- %s: %s' % (k, '・'.join(items)))
    w('')
    w('## 11. 確かめたこと')
    w('')
    cc = R.get('crosscheck') or {}
    w('- **crosscheck**: 写した道具で C の種 0 の best を予測し直すと、10/13 の npz との差は最大 %s µs(許し 0.01 µs)→ %s。' % (
        cc.get('max_abs_us'), '合格' if cc.get('pass') else '**不合格**'))
    w('- **前と同じ値か**: A・B の L・ジャンル別・R1/DE の窓・直線・ずれ、C の L・ジャンル別・R1/DE の窓・直線・ずれ・シミュレーターの頭 8、C − A の判定が'
      '10/12・10/13 の値と同じ → %s。' % ('全部同じ' if R['reproduce']['all_match'] else '**違うものがある**'))
    if rc:
        w('- **主指標の数え直し**(q2_recount.py。判定の道具を使わず、正解を ai_mix_20261011 の aligned から作り直し、判定の規則も書き直した): '
          '%s フレーム、eval_real.json との違い %d 件 → %s。' % (format(rc['frames']['human_clean'], ','), len(rc['mismatch_vs_eval_real']),
                                                         '同じ' if rc['all_match'] else '**違う**'))
    w('- protocol.json は固定のまま(sha256 が protocol.sha256 と同じ: %s)。' % ('はい' if R['protocol']['unchanged'] else 'いいえ'))
    w('')
    w('## 12. 読み取り(後から見た解釈。判定ではない)')
    w('')
    tq = {k: v for k, v in R['training'].items() if isinstance(v, dict)}
    lt = {a: [v.get('last_train_us') for k, v in tq.items() if k.startswith(a + '/') and v.get('last_train_us') is not None]
          for a in ('A', 'C', 'C-DIV')}
    if lt['C'] and lt['C-DIV']:
        w('- **学習データへの当てはまり**: 最後のエポックの学習 L1 は A %s、C %s、C-DIV %s µs。C-DIV はシミュレーターの部分の線・速さがいろいろなので、'
          'C(同じ走りのくり返し)ほど学習データに当てはまらない。それでも実機の検証 L1(最良)は C-DIV %s、C %s、A %s µs で、実機の側はよくならなかった。' % (
              '・'.join('%.1f' % x for x in lt['A']), '・'.join('%.1f' % x for x in lt['C']), '・'.join('%.1f' % x for x in lt['C-DIV']),
              '・'.join('%.1f' % v.get('best_val_us') for k, v in tq.items() if k.startswith('C-DIV/')),
              '・'.join('%.1f' % v.get('best_val_us') for k, v in tq.items() if k.startswith('C/') and '10/13' not in k),
              '・'.join('%.1f' % v.get('best_val_us') for k, v in tq.items() if k.startswith('A/') and '10/13' not in k)))
    s5 = R['sim_modes_val15']['per_arm']
    so = R['sim_old_val12']['per_arm']
    if 'h8' in s5.get('C-DIV', {}) and 'h8' in s5.get('C', {}):
        w('- **シミュレーターの側では効いている**: 5 モードの保留 15 シーンの頭 8 の L1 は C-DIV %.1f µs・C %.1f µs(実機だけの A は頭 5 で %.1f µs)。'
          '前の教師の val では逆に C-DIV %.1f・C %.1f µs(C はその走りのくり返しで学習)。いろいろな走りはシミュレーターの中の新しい走りのまねには効くが、'
          'それが実機の保留データの精度には移っていない。' % (s5['C-DIV']['h8']['l1_mean_us'], s5['C']['h8']['l1_mean_us'],
                                                s5['A']['h5']['l1_mean_us'], so['C-DIV']['h8']['l1_mean_us'], so['C']['h8']['l1_mean_us']))
    w('- **ずれへの強さ**は C-DIV も C と同じくらい A より強い(%.2f 倍・%.2f 倍)。中身が違っても同じくらいなので、強さは「いろいろな走り」より'
      '「シミュレーターの画像を混ぜること」から来ている見込み(後からの見方)。' % (rb['C-DIV']['ratio_to_ref'], rb['C']['ratio_to_ref']))
    if hs:
        w('- **低速の直線の速さの癖**は C-DIV にも C と同じくらい出た(A の平均からのずれ C-DIV %s・C %s µs)。5 モードに速さ 50〜100%% の走りが入っても、'
          'この直線で人より強く前進を出す癖は減らなかった。' % (f(hs['CDIV_shift_us'], 1, True), f(hs['C_shift_us'], 1, True)))
        cb = hs.get('contribution_to_overall_us') or {}
        if cb and P['CDIV_vs_A']['diff_best_us'] > 0:
            w('  主指標の A との差(best %s µs)のうち %s µs(%.0f%%)がこの直線の %d フレーム(全体の %.0f%%)から来ている。'
              'C-DIV の差の多くは、この 1 か所の速さの癖で説明できる。' % (
                  f(P['CDIV_vs_A']['diff_best_us'], 2, True), f(cb['C-DIV'], 2, True), 100.0 * cb['C-DIV'] / P['CDIV_vs_A']['diff_best_us'],
                  hs['n'], 100.0 * hs['n'] / R['frames']['human_clean']))
    tw = R.get('per_scene_two_C_bad') or {}
    if tw:
        cdv, cc2 = tw['per_arm']['C-DIV'], tw['per_arm']['C']
        w('- **差は C と同じ 2 シーンに集まっている**: C-DIV が A より小さかったのは %d / %d シーン(C は %d / %d)。10/13 で C が悪かった 2 シーン(%d フレーム)が'
          '全体の差 %s µs のうち %s µs を足していて、残り 9 シーンだけなら C-DIV − A は %s µs(C は全体 %s・2 シーンの分 %s・残り %s µs)。'
          '2 シーンのうち 20261010-152707-0001 は低速の直線 HS が多いシーン(10/13 の確認)で、上の速さの癖と同じ所を指している。' % (
              ps['scenes_CDIV_better_than_A'], ps['scenes'], ps['scenes_C_better_than_A'], ps['scenes'], tw['frames'],
              f(cdv['overall_diff_us'], 2, True), f(cdv['two_scenes_contribution_us'], 2, True), f(cdv['rest_9_scenes_diff_us'], 2, True),
              f(cc2['overall_diff_us'], 2, True), f(cc2['two_scenes_contribution_us'], 2, True), f(cc2['rest_9_scenes_diff_us'], 2, True)))
    w('')
    w('## 13. 注意')
    w('')
    w('- 主指標はオフラインの誤差(人の操作との近さ)。走行の結果ではない。実車を走らせる試験は範囲の外で、していない。')
    w('- C-DIV と C の差には、ばらつき(5 つの方策・24 か所の開始・速さ 50〜100%・離脱復活)のほかに、先生が作者の RC 教師から RL の方策に替わったこと、'
      'PWM の中心の約束の小さな違い(舵 ≤ 1 µs・スロットル ≤ 2 µs)も入っている。')
    w('- 学習のメモリの上限を 3G → 3800M にした(data loader の workers を保つため。計算は同じ。deviations.md)。')
    w('- 結論は jetformer_v0・この学習の条件(調整なし)・このシミュレーターのデータに限る。')
    with open(dst, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    print(dst)


if __name__ == '__main__':
    sys.exit(main())
