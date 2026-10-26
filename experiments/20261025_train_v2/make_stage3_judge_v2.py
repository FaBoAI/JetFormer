#!/usr/bin/env python3
"""段 3 と D3 の表(stage3_judge.md)を書く(2026-10-26。D3 の係)。numpy なし・torch なし。

    python3 -I /home/jetson/jetformer/experiments/20261025_train_v2/make_stage3_judge_v2.py \
        --judge /home/jetson/jetformer_reports/train_v2_20261025/stage3_judge.json \
        --desk-stage3 /home/jetson/jetformer_reports/train_v2_20261025/desk_v2_stage3.json \
        --meta <係のメモの JSON(待ち・予測のログ・段 4 の名前とコマンド)> \
        --out /home/jetson/jetformer_reports/train_v2_20261025/stage3_judge.md
読むだけ: stage3_judge.json(judge_d3_v2.py の出力)・desk_v2_stage3.json・protocol_v2.json。書くのは --out だけ。
"""
import argparse
import json
import os
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')


def f(v, nd=3, sign=False):
    if v is None:
        return '未'
    if isinstance(v, bool):
        return '○' if v else '×'
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return ('%+.*f' if sign else '%.*f') % (nd, x)


def yn(v):
    return '未' if v is None else ('満たす' if v else '**欠ける**')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True)
    ap.add_argument('--desk-stage3', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    J = json.load(open(a.judge, encoding='utf-8'))
    D = json.load(open(a.desk_stage3, encoding='utf-8'))
    M = json.load(open(a.meta, encoding='utf-8'))
    proto = json.load(open(REP / 'protocol_v2.json', encoding='utf-8'))
    T = proto['thresholds_us']['values']
    A, B = J['A_eval_v2_judge'], J['B_recount']
    names_a = ['v2_20261025-%s-s%d' % (J['arm_aug2'], s) for s in range(4)]
    names_n = ['v2_20261025-%s-s%d' % (J['arm_none'], s) for s in range(4)]
    aug = J['aug']
    cond = A['D3'].get('conditions') or {}
    L = []
    L += ['# 段 3(拡張 AUG2)と D3 の判定', '',
          '%s。D3 の係(workflow のサブエージェント)。依頼: 段 3(%s-s0〜s3)の予測・C3(机上のベンチ v2)・D3 の判定。' % (
              M['date'], J['arm_aug2']), '',
          '## 0. 要点', '',
          '| 決まり | 答え | どの決まりの行で |', '|---|---|---|']
    if aug is None:
        L.append('| **D3 拡張** | **決まりで決まらない** | %s |' % M['why_none'])
    else:
        L.append('| **D3 拡張** | **aug = %s** | protocol decisions.D3「%s」→ %s |' % (aug, J['rule'], A['D3'].get('why')))
    L += ['| 段 4(多段型 2 本) | %s | gpu_queue_v2.py status・resolve で {x}・{A}・{E}・{AUG_ARGS} が埋まった |' % ' ・ '.join(
        M.get('stage4_names') or ['未']), '']
    L += ['- 判定の台本 2 つ(A = eval_v2.py の judge() をそのまま呼ぶ・B = 別のコードで数え直し)で、欄の差の最大 %s µs(許す %s µs)・'
          '条件・答えが%s。' % (f(J['crosscheck']['max_abs_diff_us'], 6), J['crosscheck']['tol_us'],
                          '同じ' if J['crosscheck']['pass'] else '**違う(答えを出さない)**'),
          '- decisions.json は D3 の欄だけを書き換えた(D1・D2・D4・D5a・BF・D5b はそのまま)。%s' % M.get('decisions_note', ''),
          '- 机上の順位で実車の順位は決めない(plan.md 6.4)。', '']
    L += ['## 1. 固定', '', '| もの | 結果 |', '|---|---|']
    for k, v in M['seal_checks'].items():
        L.append('| %s | %s |' % (k, v))
    L += ['| eval_v2.py の sha256 | %s(lock2 の C4 と同じ) |' % J['checks']['eval_v2_py_sha256'],
          '| protocol_numbers.py の sha256 | %s(lock2 と同じ) |' % J['checks']['protocol_numbers_py_sha256'], '']
    L += ['## 2. 段 3 の予測(eval_v2.py predict。best・last)', '',
          '| 名前 | best | last | eval/pred の sha256(best の先頭 12) | ログ |', '|---|---|---|---|---|']
    for n in names_a:
        e = M['evalpred'].get(n, {})
        L.append('| %s | %s | %s | %s | %s |' % (n, e.get('best', '未'), e.get('last', '未'),
                                                (J['inputs']['eval_pred_sha256'].get(n) or '')[:12], e.get('log', '')))
    L += ['', '- jf-heavy --lock big で 1 本ずつ(別の係の C3 と順番に分け合う。あいだ 60 秒)。%s' % M.get('evalpred_note', ''), '']
    L += ['## 3. 学習ごとの欄(H1 4,670 フレーム・介入 1,570 フレーム。µs。判定 A の値)', '',
          '| 名前 | L1 best | L1 last | 介入 best | ステアの偏り | スロットルの偏り | 昼 | 夜 | robust_dl1_us | 長すぎ | desk の行 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    rowsA = M['rowsA']
    for n in names_n + names_a:
        r = rowsA.get(n, {})
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            n, f(r.get('l1_best_us')), f(r.get('l1_last_us')), f(r.get('l1_interv_us'), 2), f(r.get('steer_bias_us'), 2, True),
            f(r.get('thr_bias_us'), 2, True), f(r.get('l1_day_us')), f(r.get('l1_night_us')), f(r.get('robust_dl1_us'), 4),
            f(r.get('mark_too_long')), M['row_source'].get(n, '')))
    L += ['', '- robust_dl1_us = ずれの試し 7 つの H1 の L1 の増え(試し − そのまま)の平均。そのまま = eval/pred の保存した best'
          '(protocol fields_added・deviations.md 7)。', '']
    L += ['ずれの試しごとの増え(µs。判定 B で数え直した値):', '',
          '| 名前 | ' + ' | '.join(M['shift_labels']) + ' | 平均 |', '|---|' + '---|' * (len(M['shift_labels']) + 1)]
    for n in names_n + names_a:
        per = (B['runs'][n] or {}).get('robust_per_shift') or {}
        L.append('| %s | ' % n + ' | '.join(f(per.get(k), 3, True) for k in M['shift_keys']) + ' | %s |' % f(B['runs'][n].get('robust_dl1'), 4))
    L += ['']
    L += ['## 4. 同じ種の 4 組(AUG2 − なし)', '',
          '| 種 | L1 best の差 | L1 last の差 | robust AUG2 | robust なし | AUG2 < なし | 昼の差 | 夜の差 |',
          '|---|---|---|---|---|---|---|---|']
    ph = A['pair_h1'] or {}
    for s in range(4):
        ra, rn = rowsA.get(names_a[s], {}), rowsA.get(names_n[s], {})
        rba, rbn = ra.get('robust_dl1_us'), rn.get('robust_dl1_us')
        L.append('| s%d | %s | %s | %s | %s | %s | %s | %s |' % (
            s, f((ph.get('d_best') or [None] * 4)[s], 3, True), f((ph.get('d_last') or [None] * 4)[s], 3, True),
            f(rba, 4), f(rbn, 4), '未' if None in (rba, rbn) else ('○' if rba < rbn else '×'),
            f(None if None in (ra.get('l1_day_us'), rn.get('l1_day_us')) else ra['l1_day_us'] - rn['l1_day_us'], 3, True),
            f(None if None in (ra.get('l1_night_us'), rn.get('l1_night_us')) else ra['l1_night_us'] - rn['l1_night_us'], 3, True)))
    rb = B['robust']
    L.append('| **平均** | **%s** | **%s** | **%s** | **%s** | 比 %s | **%s** | **%s** |' % (
        f(ph.get('mean_d_best'), 3, True), f(ph.get('mean_d_last'), 3, True), f(rb.get('mean_aug2'), 4), f(rb.get('mean_none'), 4),
        f(rb.get('ratio'), 3), f(B['day_night']['mean_d_day'], 3, True), f(B['day_night']['mean_d_night'], 3, True)))
    L += ['', '- H1 の同じ種の判定(judgement.pairs、k = 4): best の閾値 ±%s・last の閾値 ±%s・「悪くない」= d の平均 ≤ +%s(best と last の両方)'
          '→ **%s**(悪くない %s)。' % (T['pairs']['k4']['l1_best'], T['pairs']['k4']['l1_last'],
                                    proto['thresholds_us']['not_worse_us'], ph.get('verdict'), f(ph.get('not_worse'))), '']
    L += ['## 5. D3 の条件(protocol decisions.D3 の文を 1 つずつ)', '',
          '| 条件(protocol の文) | 値 | 基準 | 結果 |', '|---|---|---|---|']
    vt = (A['arm_aug2'] or {}).get('veto') or {}
    vals = {
        'H1 が悪くない': ('d の平均 best %s・last %s' % (f(ph.get('mean_d_best'), 3, True), f(ph.get('mean_d_last'), 3, True)),
                      '両方 ≤ +0.5 µs(judgement.pairs.not_worse)'),
        'robust_dl1_us が全部の組で AUG2 < なし': ('AUG2 %s / なし %s' % (
            ' '.join(f(x, 3) for x in rb['aug2']), ' '.join(f(x, 3) for x in rb['none'])), '4 組とも AUG2 < なし'),
        '平均(AUG2) ÷ 平均(なし) ≤ 0.8': ('%s ÷ %s = %s' % (f(rb.get('mean_aug2'), 4), f(rb.get('mean_none'), 4), f(rb.get('ratio'), 3)),
                                     '≤ 0.8'),
        'l1_day_us の差の平均 ≤ +1.0': (f(B['day_night']['mean_d_day'], 3, True), '≤ +1.0 µs(昼 4 シーン)'),
        'l1_night_us の差の平均 ≤ +1.0': (f(B['day_night']['mean_d_night'], 3, True), '≤ +1.0 µs(夜 7 シーン)'),
        'AUG2 の腕に拒否権が無い': ('介入 %s(閾値 %s)・偏り %s(±%s)' % (f(vt.get('interv_diff'), 3, True), f(vt.get('interv_thr'), 3),
                                                       f(vt.get('bias_value'), 3, True), f(vt.get('bias_thr'), 3)),
                           'vetoes.interv・vetoes.bias(n = 4)'),
    }
    for c, (v, base) in vals.items():
        L.append('| %s | %s | %s | %s |' % (c, v, base, yn(cond.get(c)) if cond else '未'))
    L += ['', '→ %s' % (('**全部を満たした → AUG2**' if aug == 'AUG2' else ('**%s → なし**(「それ以外はなし」)' % A['D3'].get('why')))
                         if aug else '**決まりで決まらない**: %s' % M['why_none']), '']
    va = (A['arm_aug2'] or {}).get('vs5') or {}
    L += ['## 6. AUG2 の腕 対 5 本・拒否権・印(参考を含む)', '',
          '| 腕 | n | best の差 | 閾値 | last の差 | 閾値 | 判定(vs5) | 介入の差 | 閾値 | 拒否権 介入 | 偏り | 閾値 | 拒否権 偏り | 長すぎ | 上がり |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in (J['arm_none'], J['arm_aug2']):
        x = A['arm_aug2'] if arm == J['arm_aug2'] else A['arm_none']
        v, t = (x or {}).get('vs5') or {}, (x or {}).get('veto') or {}
        L.append('| %s | %s | %s | ±%s | %s | ±%s | **%s** | %s | %s | %s | %s | ±%s | %s | %s 本 | %s 本 |' % (
            arm, v.get('n'), f(v.get('diff_best'), 3, True), v.get('thr_best'), f(v.get('diff_last'), 3, True), v.get('thr_last'),
            v.get('verdict'), f(t.get('interv_diff'), 3, True), f(t.get('interv_thr'), 3), 'あり' if t.get('interv') else 'なし',
            f(t.get('bias_value'), 3, True), f(t.get('bias_thr'), 3), 'あり' if t.get('bias') else 'なし', x.get('too_long'), x.get('rise')))
    L += ['', '- 5 本の平均: 介入 %s µs・ステアの偏り %s µs(C3 と同じフレームで保存した予測から。丸めない)。' % (
        f(B['veto_aug2']['five_mean_interv'], 3), f(B['veto_aug2']['five_mean_steer_bias'], 3, True)),
          '- 印(長すぎ・上がり)は protocol decisions.D3 の文に無い。参考に書いただけで、D3 に使っていない(決まりの外の判断をしない)。', '']
    L += ['## 7. C3(desk_v2.py)の行の出どころと待ち', '']
    for s in M['desk_notes']:
        L.append('- %s' % s)
    L += ['', '| 待ちの時刻 | 別の係の desk/desk_v2.json | 行の数 | 足りない 16 本 | 足りない基準 |', '|---|---|---|---|---|']
    for w in M.get('wait_log', []):
        L.append('| %s | %s | %s | %s | %s |' % (w.get('at'), w.get('made_at'), w.get('rows'), w.get('missing16'), w.get('missing_base')))
    L += ['']
    L += ['段 3 の C3 の欄(desk_v2_stage3.json。参考。門は D5a で main session が見る):', '',
          '| 名前 | sanity | p99 [ms] | 低速のブレーキ | DE 手前 vs B | R1 手前 vs B | SW 手前 vs B | 直線 vs B | 止まった場面 | 門(desk) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for n in names_a:
        r = (D.get('runs') or {}).get(n) or {}
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            n, f(r.get('sanity_ok')), f(r.get('trt_p99_ms'), 3), f(r.get('brake_lowspeed_frac'), 4), f(r.get('de_pre_thr_vs_b_us'), 2, True),
            f(r.get('r1_pre_thr_vs_b_us'), 2, True), f(r.get('sw_pre_thr_vs_b_us'), 2, True), f(r.get('straight_thr_vs_b_us'), 2, True),
            f(r.get('standstill_go_frac'), 4), {True: '合格', False: '不合格', None: '未'}.get(r.get('gates_pass'), r.get('gates_pass'))))
    L += ['', '- desk_v2_stage3.json は desk_v2.py report --out で作った(%s)。別の係の desk/desk_v2.json・desk_v2.md には書いていない。' % D.get('made_at'),
          '- 段 3 の予測の npz(desk/pred/<名前>_shift.npz・_tab.npz)と TensorRT(desk/trt/)は desk_v2.py の決まった場所に置いた'
          '(名前が違うので別の係のファイルは上書きしない)。別の係のこのあとの report は、この 4 本の行も desk_v2.json に入れる。', '']
    L += ['## 8. 段 4(TransFuser 多段型 2 本)', '']
    for s in M.get('stage4_lines', []):
        L.append(s)
    L += ['']
    L += ['## 9. 2 つの台本の照らし', '',
          '| 項目 | A(eval_v2.judge) | B(数え直し) |', '|---|---|---|',
          '| aug | %s | %s |' % (A['D3'].get('aug'), B.get('aug')),
          '| 条件 | %s | %s |' % ('・'.join('%s %s' % (k, '○' if v else '×') for k, v in cond.items()) if cond else '未',
                                 '・'.join('%s %s' % (k, '○' if v else '×') for k, v in (B.get('conditions') or {}).items()) or '未'),
          '| 欄の差の最大(8 本 × 7 欄) | %s µs | |' % f(J['crosscheck']['max_abs_diff_us'], 6), '']
    L += ['## 10. 未・注意', '']
    for s in M.get('notes', []):
        L.append('- %s' % s)
    L += ['', '出典: %s・%s・%s' % (a.judge, a.desk_stage3, REP / 'protocol_v2.json'), '']
    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    os.replace(tmp, a.out)
    print('書きました: %s' % a.out)


if __name__ == '__main__':
    main()
