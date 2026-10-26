#!/usr/bin/env python3
"""段 1・2 と B-FAST の判定の表と decisions.json を書く(2026-10-25 夜)。numpy だけ(torch・GPU は使わない)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/make_stage12_judge_v2.py

読む: eval/eval_v2.json(eval_v2.py judge)・recount_stage12.json(recount_stage12_v2.py)・protocol_v2_lock2.json。
- 2 つの台本の答え(欄・判定・D1・D2)が合わなければ、decisions.json を書かずに止まる。
- D3 は段 3 の学習と C3 の robust_dl1_us が要る(protocol decisions.D3)。答えが無いので aug は null(キューは段 4 の前で止まる)。
書く: decisions.json(キューが読む D1.x・D2.E・D3.aug と、使った数)・stage12_judge.md。決まり・既定値は変えない。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_v2 as EV  # noqa: E402  (門の関数だけ使う。numpy)

REP = Path(os.environ.get('JUDGE_REP') or '/home/jetson/jetformer_reports/train_v2_20261025')
EVAL_JSON = Path(os.environ.get('JUDGE_EVAL_JSON') or REP / 'eval' / 'eval_v2.json')
RECOUNT = Path(os.environ.get('JUDGE_RECOUNT') or REP / 'recount_stage12.json')
OUT_DEC = Path(os.environ.get('JUDGE_OUT_DEC') or REP / 'decisions.json')
OUT_MD = Path(os.environ.get('JUDGE_OUT_MD') or REP / 'stage12_judge.md')
LOCK2 = Path('/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2_lock2.json')
PROTOCOL = Path('/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2.json')
TRAIN_PY = Path('/home/jetson/jetformer/tools/train.py')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def F(v, nd=2, sign=False):
    if v is None:
        return '未'
    if v == '参考':
        return '参考'
    if isinstance(v, bool):
        return '○' if v else '×'
    if isinstance(v, (int, float)):
        return ('%+.*f' if sign else '%.*f') % (nd, v)
    return str(v)


def V(v):
    """拒否権・印: True → 付く、False → なし"""
    return '未' if v is None else ('**付く**' if v else 'なし')


def main():
    proto = json.load(open(PROTOCOL, encoding='utf-8'))
    T = proto['thresholds_us']['values']
    E = json.load(open(EVAL_JSON, encoding='utf-8'))
    RC = json.load(open(RECOUNT, encoding='utf-8'))
    L2 = json.load(open(LOCK2, encoding='utf-8'))
    if E['meta'].get('dry_run') and 'JUDGE_OUT_DEC' not in os.environ:
        raise SystemExit('eval_v2.json が空回しです')
    J, rows, five = E['judge'], {r['run']: r for r in E['runs']}, E['five']
    cmp = RC['compare_eval_v2']
    ok = bool(cmp.get('fields_ok') and all((cmp.get('verdicts_same') or {}).values()) and cmp.get('D1_same')
              and cmp.get('D2_same'))
    if not ok:
        raise SystemExit('eval_v2.py と数え直しが合いません。decisions.json は書きません: %s' % json.dumps(cmp, ensure_ascii=False))
    D = J['decisions']
    x, Ep = D['D1'].get('x'), D['D2'].get('E')
    if x is None or Ep is None:
        raise SystemExit('D1・D2 が決まっていません: %s / %s' % (D['D1'].get('why'), D['D2'].get('why')))
    now = time.strftime('%Y-%m-%dT%H:%M:%S+09:00')
    lock2_sha = sha(LOCK2)
    b = five['B']
    thr_bias_plus = T['five_mean']['thr_bias_best'] + T['pairs']['k2']['thr_bias_best']
    no_c1 = '--aug-top-brightness' not in TRAIN_PY.read_text(encoding='utf-8')
    c1_when = time.strftime('%m/%d %H:%M', time.localtime(TRAIN_PY.stat().st_mtime))

    # v2 の門(参考): D3 が「なし」なら最終の腕になる {x}-e{E} の種
    ref_arm = '%s-e%d' % (x, int(Ep))
    ref_gates = {}
    for s in range(4):
        n = 'v2_20261025-%s-s%d' % (ref_arm, s)
        if n in rows:
            g = EV.gates_v2(rows[n], b, None, thr_bias_plus)
            ref_gates[n] = {'gates': g, 'pass': EV.gates_pass(g)}

    bf_g = {n: J['gates'][n] for n in sorted(J['gates']) if n.startswith('bfast_')}
    bf_veto = J['arms']['bfast-e15']['veto']
    dec = {
        'schema': 'jf-train-v2-decisions/v1',
        'written_at': now,
        'written_by': 'workflow のサブエージェント(予測と判定の係)。eval_v2.py judge と recount_stage12_v2.py(別のコード)が同じ答え',
        'protocol_v2_sha256': sha(PROTOCOL),
        'protocol_v2_lock2_sha256': lock2_sha,
        'note': 'キューが読むのは D1.x・D2.E・D3.aug だけ。D3.aug が null のあいだ、キューは段 4 の前で止まる',
        'D1': {'x': x, 'rule': 'protocol decisions.D1 ①②③', 'why': D['D1'].get('why'),
               'v2h_e30_vs5': J['arms']['v2h-e30']['vs5']['verdict'], 'v2h_e30_veto': J['arms']['v2h-e30']['veto']['any'],
               'v2m_conditions': D['D1'].get('v2m_conditions'), 'values': D['D1'].get('values')},
        'D2': {'E': int(Ep), 'rule': 'protocol decisions.D2', 'why': D['D2'].get('why'),
               'pair_last': {k: J['pairs']['D2: %s-e60 対 %s-e30(last)' % (x, x)].get(k)
                             for k in ('k', 'd_last', 'mean_d_last', 'thr_last', 'verdict')},
               'too_long_60': D['D2'].get('too_long'), 'genre_skew_60_vs_30': D['D2'].get('genre_skew'),
               'veto_60': J['arms'].get('%s-e60' % x, {}).get('veto', {}).get('any')},
        'D3': {'aug': None, 'status': '決まりで決まらない(まだ)',
               'why': 'protocol decisions.D3 は {x}-aug2-e{E} と {x}-e{E} の同じ種 4 組と robust_dl1_us(C3 のずれの試し)で決める。'
                      '段 3 の学習がまだ・C3 がまだ。答えを置かない(段 4 は D3 のあと)'},
        'D4': {'reserve': None, 'why': '段 4(多段型)と D3 のあと'},
        'D5a': {'C1': None, 'C2': None, 'final_arm': None, 'why': 'D3 のあと。門の C3 の欄も要る'},
        'BF': {'BF1': D['BF'].get('BF1'), 'BF2': D['BF'].get('BF2'),
               'status': ('無し(B-FAST の腕に拒否権)' if bf_veto['any'] else '未(C3 の門)'),
               'rule': 'protocol decisions.BF「B-FAST の腕(n = そろった種の数)に拒否権があれば無し」・vetoes.when「B-FAST の腕(n = 3)にも同じ式で掛ける」',
               'why': D['BF'].get('why'),
               'veto': {k: bf_veto.get(k) for k in ('any', 'interv', 'interv_diff', 'interv_thr', 'bias', 'bias_value', 'bias_thr')},
               'veto_interv_seeds_us': {n: rows[n]['l1_interv_us'] for n in sorted(rows) if n.startswith('bfast_')},
               'five_mean_interv_best_us': RC['five_mean']['interv_best'],
               'gates_reference': {n: {'pass': g['pass'], **g['gates']} for n, g in bf_g.items()},
               'gates_note': '門は参考(拒否権で腕ごと候補にしない)。G-b・G-c・G-d・G-f・安全は C3 の欄が無く未'},
        'D5b': {'slot': None, 'when': '11/1 朝(封を 1 回だけ開ける)'},
    }
    tmp = str(OUT_DEC) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(dec, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT_DEC)

    # ---------------------------------------------------------------- 表
    order = (['bfast_20261025-e15-s%d' % s for s in range(3)]
             + ['v2_20261025-%s-e30-s%d' % (a, s) for a in ('v2h', 'v2m') for s in range(4)]
             + ['v2_20261025-v2h-e60-s%d' % s for s in range(4)] + ['v2_20261025-v2h-e100-s0'])
    fm = RC['five_mean']
    L = ['# 段 1・2 と B-FAST の判定(机上の予測と決まり)', '',
         '2026-10-25 夜。予測と判定の係。依頼: B-FAST とデータ v2 の学習(了承済み)の、終わった 16 本の予測・lock2・D1〜D3 の判定。', '',
         '## 0. 要点', '',
         '| 決まり | 答え | どの決まりの行で |', '|---|---|---|',
         '| **D1 データ** | **x = %s** | %s |' % (x, D['D1'].get('why')),
         '| **D2 エポック** | **E = %d** | %s |' % (int(Ep), D['D2'].get('why')),
         '| D3 拡張 | **決まりで決まらない(まだ)** | 段 3(AUG2 の 4 本)の学習と C3 の robust_dl1_us が要る(decisions.D3) |',
         ('| **BF B-FAST の候補** | **無し**(BF1・BF2 とも) | 拒否権「介入」: B-FAST 3 本の介入の L1 の平均 − 128.594 = %+.3f > 閾値 %.3f(n = 3)、かつ 3 本とも 128.594 より大きい(vetoes.interv)→ decisions.BF「腕に拒否権があれば無し」 |' % (
             bf_veto['interv_diff'], bf_veto['interv_thr']) if bf_veto['any'] else
          '| BF B-FAST の候補 | **未** | G-a・G-e は 3 本とも合格。G-b・G-c・G-d・G-f・安全の門は C3 の欄が無く「未」(合格にしない) |'),
         '| D4・D5a | 未 | D3 と段 4 のあと |', '',
         '- `decisions.json` に D1・D2 を書いた。D3.aug は null。キューは **段 4 の前で止まる**。',
         ('- 段 3 も今はまだ動かない。`tools/train.py` に `--aug-top-brightness`(C1)がまだ無い(キューの ⓪ で止まる)。' if no_c1 else
          '- `tools/train.py` に `--aug-top-brightness`(C1)がある(train.py の更新 %s。別の係)。キューの ⓪ の引数の確かめは通る。C1 のテストの結果は、この係では見ていない。' % c1_when),
         '- 予測は lock2 のあとに始めた(protocol lock.second_lock「最初の予測の前」)。依頼の番号の順(予測 → lock2)とは逆。',
         '- 2 つの別の台本(eval_v2.py と recount_stage12_v2.py)で、欄の差の最大 %s µs(%d 個)・判定・D1・D2 が同じ。' % (
             F(cmp['max_abs_diff_us'], 4), cmp['fields']),
         '- 机上の順位で実車の順位は決めない(plan.md 6.4)。', '',
         '## 1. 固定', '',
         '| もの | 結果 |', '|---|---|',
         '| `sha256sum -c protocol_v2.sha256` | OK(6 つ) |',
         '| `check_seal_v2.py` | すべて同じ(封の 14 本) |',
         '| lock2 `protocol_v2_lock2.json` | `%s`(chmod a-w。`protocol_v2_lock2.sha256`) |' % lock2_sha,
         '| eval_v2.json | %s・lock2 のあと |' % E['meta']['made_at'], '',
         '段 0 の答え(lock2 に書いた):', '',
         '| # | 確かめ | 答え |', '|---|---|---|']
    for k in sorted(L2['stage0']):
        s0 = L2['stage0'][k]
        L.append('| %s | %s | %s |' % (k, s0['what'], s0['status'] + ('。' + s0['why'] if s0.get('why') else '')))
    L += ['', '## 2. 学習ごとの欄(H1 4,670 フレーム・介入 1,570 フレーム。µs)', '',
          '| 名前 | L1 best | L1 last | 介入 best | 介入 last | ステアの偏り | スロットルの偏り | ジャンル 4(last) | ジャンル 5(last) | 振れ幅 | エンジン | 長すぎ |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for rid, r in five.items():
        L.append('| 5 本: %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            rid, F(r['l1_best_us'], 3), F(r['l1_last_us'], 3), F(r['l1_interv_us']), F(r['l1_interv_last_us']),
            F(r['steer_bias_us'], 2, True), F(r['thr_bias_us'], 2, True), F(r['l1_genre4_last_us']), F(r['l1_genre5_last_us']),
            F(r['steer_amp_ratio'], 3), F(r.get('engine_ok')), F(r.get('mark_too_long'))))
    L.append('| **5 本の平均** | **%.3f** | **%.3f** | **%.3f** | | **%+.3f** | **%+.3f** | | | | | |' % (
        fm['l1_best'], fm['l1_last'], fm['interv_best'], fm['steer_bias_best'], fm['thr_bias_best']))
    for n in order:
        r = rows[n]
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            n, F(r['l1_best_us'], 3), F(r['l1_last_us'], 3), F(r['l1_interv_us']), F(r['l1_interv_last_us']),
            F(r['steer_bias_us'], 2, True), F(r['thr_bias_us'], 2, True), F(r['l1_genre4_last_us']), F(r['l1_genre5_last_us']),
            F(r['steer_amp_ratio'], 3), F(r.get('engine_ok')), F(r.get('mark_too_long'))))
    L += ['', '- 符号: ステア + = 左。スロットルの偏り(_thr_)+ = 緩める。偏りは行 0 の 予測 − 人(best)。',
          '- エンジン = engine_check.json が今のエンジンのもので、TensorRT と PyTorch の差 ≤ 2.5 µs。', '',
          '## 3. 腕 対 5 本と拒否権(judgement.vs5・vetoes)', '',
          '| 腕 | n | best の差 | 閾値 | last の差 | 閾値 | 判定 | 介入の差 | 閾値 | 拒否権 介入 | 偏り | 閾値 | 拒否権 偏り | 長すぎ |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in ('v2h-e30', 'v2m-e30', 'v2h-e60', 'v2h-e100', 'bfast-e15'):
        a = J['arms'].get(arm)
        if not a:
            continue
        v, vt = a['vs5'], a['veto']
        L.append('| %s | %d | %s | ±%s | %s | ±%s | **%s** | %s | %s | %s | %s | ±%s | %s | %d 本 |' % (
            arm, v['n'], F(v.get('diff_best'), 3, True), F(v.get('thr_best'), 3), F(v.get('diff_last'), 3, True),
            F(v.get('thr_last'), 3), v['verdict'], F(vt.get('interv_diff'), 3, True), F(vt.get('interv_thr'), 3),
            V(vt.get('interv')), F(vt.get('bias_value'), 3, True), F(vt.get('bias_thr'), 3), V(vt.get('bias')), a['too_long']))
    L += ['', '- 判定「良い」= 差 ≤ −閾値 かつ全部の種が 5 本の平均より小さい(best と last の両方)。「悪い」= 差 ≥ +閾値 かつ全部の種が大きい(どちらか)。',
          '- 拒否権 介入 = 平均 − 128.594 > 閾値(n)かつ全部の種が大きい。偏り = 5 本の平均 +0.132 からの差が ±閾値(n)の外。v2m-e30 は同じ種の v2h-e30 との差(k = 4、±4.125)。',
          '- v2h-e100 は参考(D2 で選ばない)。B-FAST の結果は v2 の決まりに使わない(protocol bfast_recipe.independence)。', '',
          '## 4. D1 データ(decisions.D1)', '',
          '| 行 | 条件 | 値 | 結果 |', '|---|---|---|---|']
    h = J['arms']['v2h-e30']
    L.append('| ① | v2h-e30 が 5 本に「悪い」でない | 判定 %s | %s |' % (h['vs5']['verdict'], F(h['vs5']['verdict'] != '悪い')))
    L.append('| ① | v2h-e30 に拒否権が無い | 介入 %s・偏り %s | %s |' % (V(h['veto']['interv']), V(h['veto']['bias']),
                                                                 F(not h['veto']['any'])))
    ph = J['pairs'].get('D1: v2m-e30 対 v2h-e30(H1)', {})
    pi = J['pairs'].get('D1: v2m-e30 対 v2h-e30(介入)', {})
    vals = D['D1'].get('values') or {}
    conds = D['D1'].get('v2m_conditions') or {}
    show = {
        'H1 が悪くない': 'best の d の平均 %s・last %s(≤ +0.5)' % (F(ph.get('mean_d_best'), 3, True), F(ph.get('mean_d_last'), 3, True)),
        'steer_bias': '差の平均 %s(±4.125 の中)' % F(vals.get('steer_bias_diff'), 3, True),
        'l1_interv': '差の平均 %s(≤ +1.718)' % F(vals.get('interv_diff'), 3, True),
        'thr_bias': '差の平均 %s(≥ −2.887)' % F(vals.get('thr_bias_diff'), 3, True),
        '良い': 'H1: %s(d best %s・last %s、閾値 0.765・1.087)/ 介入: %s(d best %s・last %s、閾値 1.718・2.592)' % (
            ph.get('verdict'), F(ph.get('mean_d_best'), 3, True), F(ph.get('mean_d_last'), 3, True), pi.get('verdict'),
            F(pi.get('mean_d_best'), 3, True), F(pi.get('mean_d_last'), 3, True)),
        '拒否権': '介入 %s・偏り %s' % (V(J['arms']['v2m-e30']['veto']['interv']), V(J['arms']['v2m-e30']['veto']['bias'])),
    }
    for c, ok_ in conds.items():
        key = next((k for k in show if k in c), None)
        row = '③' if '拒否権' in c else '②'
        L.append('| %s | v2m-e30 対 v2h-e30: %s | %s | %s |' % (row, c, show.get(key, ''), F(bool(ok_))))
    if ph.get('d_best'):
        L += ['', '同じ種の差 d_k = v2m − v2h(H1、µs): best %s・last %s。介入 best %s。' % (
            ' / '.join(F(v, 3, True) for v in ph['d_best']), ' / '.join(F(v, 3, True) for v in ph.get('d_last') or []),
            ' / '.join(F(v, 3, True) for v in pi.get('d_best') or []))]
    L += ['', '→ **x = %s**。%s' % (x, D['D1'].get('why')), '']
    if x != 'v2h':
        L += ['- D1 が v2h 以外なので、段 2 を x でやり直す(先回りの v2h-e60・e100 は参考)。decisions.D1.after', '']
    p2 = J['pairs']['D2: %s-e60 対 %s-e30(last)' % (x, x)]
    L += ['## 5. D2 エポック(decisions.D2。%s-e60 対 %s-e30、last、同じ種 %d 組)' % (x, x, p2['k']), '',
          '| 種 | d_k(last)= 60 − 30 | 60 の last − best | 長すぎ(> 1.6) | ジャンル 4 の差(last) | ジャンル 5 の差(last) | ジャンルの片寄り |',
          '|---|---|---|---|---|---|---|']
    gs = RC['D2'].get('genre_skew') or {}
    for i, s in enumerate(p2['seeds']):
        r60 = rows['v2_20261025-%s-e60-s%d' % (x, s)]
        g = gs.get(str(s), {})
        L.append('| %d | %s | %s | %s | %s(> 1.938 で印) | %s(> 5.308 で印) | %s |' % (
            s, F(p2['d_last'][i], 3, True), F(r60['l1_last_us'] - r60['l1_best_us'], 3, True), F(bool(r60.get('mark_too_long'))),
            F(g.get('d4'), 3, True), F(g.get('d5'), 3, True), V(g.get('mark'))))
    v60 = J['arms'].get('%s-e60' % x, {})
    L += ['', '| 30 に戻す条件 | 値 | 当たる |', '|---|---|---|',
          '| 60 が「悪い」(d の平均 ≥ +%.3f かつ全部 > 0、last) | d の平均 %s | %s |' % (
              p2['thr_last'], F(p2['mean_d_last'], 3, True), F(p2['verdict'] == '悪い')),
          '| 「長すぎ」2 本以上 | %d 本 | %s |' % (D['D2'].get('too_long', 0), F(D['D2'].get('too_long', 0) >= 2)),
          '| 「ジャンルの片寄り」2 組以上 | %d 組 | %s |' % (sum(1 for v in (D['D2'].get('genre_skew') or {}).values() if v),
                                              F(sum(1 for v in (D['D2'].get('genre_skew') or {}).values() if v) >= 2)),
          '| 60 の腕に拒否権 | 介入 %s・偏り %s | %s |' % (V(v60.get('veto', {}).get('interv')), V(v60.get('veto', {}).get('bias')),
                                                  F(bool(v60.get('veto', {}).get('any')))),
          '', '→ **E = %d**。%s。100 は参考で選ばない。' % (int(Ep), D['D2'].get('why')), '',
          '- 参考: v2h-e100-s0 は L1 best %s・last %s(n = 1、対 5 本 %s)。' % (
              F(rows['v2_20261025-v2h-e100-s0']['l1_best_us'], 3), F(rows['v2_20261025-v2h-e100-s0']['l1_last_us'], 3),
              J['arms']['v2h-e100']['vs5']['verdict']), '',
          '## 6. D3 拡張(decisions.D3)', '',
          '- **決まりで決まらない(まだ)。** D3 は「%s-aug2-e%d と %s-e%d の同じ種 4 組」で、H1 が悪くない・robust_dl1_us(C3)が 4 組とも小さく平均で 0.8 倍以下・昼と夜の差・拒否権を見る。' % (x, int(Ep), x, int(Ep)),
          '- 段 3 の 4 本は学習していない。robust_dl1_us を出す C3 も無い。だから AUG2 も「なし」も選べない。decisions.json の D3.aug は null。',
          '- 段 4(多段型)の名前と引数は D3 の答えで決まる(`-msf{A}`)。D3 の前に段 4 を回すと決まりの外になる。', '',
          '## 7. B-FAST の門(gates_bfast。種ごと)', '',
          '| 門 | 条件 | 種 0 | 種 1 | 種 2 |', '|---|---|---|---|---|']
    names = ['bfast_20261025-e15-s%d' % s for s in range(3)]
    gate_txt = {'G-a': 'l1_best ≤ B %.3f + 1.0 = %.3f' % (b['l1_best_us'], b['l1_best_us'] + 1.0),
                'G-b': 'bfast_gb_fwd_vs_b_us ≥ +10(C3)', 'G-c': 'bfast_gc_fwd_vs_b_us ≤ +5(C3)',
                'G-d': 'standstill_go_frac ≥ B − 0.10(C3)', 'G-e': 'engine_ok(差 ≤ 2.5 µs)',
                'G-f': 'straight_fwd_vs_b_us < 0.10 × B の生の前進(C3)', 'sanity_ok': 'NaN・Inf なし・500〜2500(C3)',
                'inference': 'p99 ≤ 1.3 × 同じ日の B(C3)', 'amp': '振れ幅 0.80〜1.25', 'brake': 'brake_lowspeed_frac ≤ 0.01(C3)'}
    for gk, txt in gate_txt.items():
        cells = []
        for n in names:
            g = J['gates'][n]['gates'].get(gk)
            r = rows[n]
            extra = {'G-a': ' %.3f' % r['l1_best_us'], 'G-e': ' %s µs' % F(r.get('engine_actions_max_us'), 3),
                     'amp': ' %.3f' % r['steer_amp_ratio']}.get(gk, '')
            cells.append(F(g) + extra)
        L.append('| %s | %s | %s |' % (gk, txt, ' | '.join(cells)))
    L.append('| **全部** | | %s |' % ' | '.join({True: '合格', False: '**不合格**', None: '未'}[J['gates'][n]['pass']] for n in names))
    vb = J['arms']['bfast-e15']
    L += ['', '- B-FAST の腕(n = 3)の拒否権: 介入 %s(差 %s、閾値 %s)・偏り %s(差 %s、閾値 ±%s)。対 5 本は「%s」(参考)。' % (
        V(vb['veto']['interv']), F(vb['veto']['interv_diff'], 3, True), F(vb['veto']['interv_thr'], 3), V(vb['veto']['bias']),
        F(vb['veto']['bias_value'], 3, True), F(vb['veto']['bias_thr'], 3), vb['vs5']['verdict'])]
    if vb['veto']['any']:
        L += ['- 介入の L1(best): 種 0 %s・種 1 %s・種 2 %s(5 本の平均 %.3f。B は %s)。3 本とも 5 本の平均より大きい。' % (
            F(rows[names[0]]['l1_interv_us']), F(rows[names[1]]['l1_interv_us']), F(rows[names[2]]['l1_interv_us']),
            RC['five_mean']['interv_best'], F(b['l1_interv_us'])),
              '- **決まり(decisions.BF)で BF1・BF2 は無し。** 拒否権は腕ごとなので、C3 の門(G-b など)を待たずに決まる。上の門の表は参考。',
              '- 11/3 の 1 枠(slot_20261103: BF1 → BF2 → C1 → C2)は、C1・C2 だけが残る。', '']
    else:
        L += ['- 「未」は合格にしない(protocol)。**BF1・BF2 はまだ決まらない。** C3 の欄(G-b・G-c・G-d・G-f・sanity・推論・低速のブレーキ)が要る。', '']
    L += [
        '## 8. v2 の門(参考。gates_v2)', '',
        'D3 が「なし」なら最終の腕は %s になる。いま出せる門だけ(C3 の欄は「未」)。D5a(C1・C2)は D3 のあと。' % ref_arm, '',
        '| 名前 | engine_ok | 振れ幅 | L1 の差(B と < 10) | 人の走りのスロットル(≤ 15.964) | sanity・推論・低速のブレーキ・止まった場面・直線 | 全部 |',
        '|---|---|---|---|---|---|---|']
    for n, g in ref_gates.items():
        r, gg = rows[n], g['gates']
        L.append('| %s | %s | %s(%.3f) | %s(%+.2f) | %s(%+.2f) | 未(C3) | %s |' % (
            n, F(gg['engine_ok']), F(gg['amp']), r['steer_amp_ratio'], F(gg['l1_vs_b']), r['l1_best_us'] - b['l1_best_us'],
            F(gg['human_throttle']), r['thr_bias_us'], {True: '合格', False: '**不合格**', None: '未'}[g['pass']]))
    L += ['', '## 9. 数え直し(別のコード)', '',
          '| 照らすもの | 結果 |', '|---|---|',
          '| 欄(16 本 × 10 欄。L1・介入・偏り・ジャンル) | 差の最大 %s µs(≤ 0.001。eval_v2.json は小数 4 桁に丸めてある) |' % F(cmp['max_abs_diff_us'], 4),
          '| 腕の判定・拒否権・長すぎ | %s |' % ('全部同じ' if all(cmp['verdicts_same'].values()) else '違う'),
          '| D1・D2 | %s・%s |' % ('同じ' if cmp['D1_same'] else '違う', '同じ' if cmp['D2_same'] else '違う'),
          '| 段 0 の確かめ 3(ジャンルの片寄りの閾値) | %s |' % ' / '.join(
              '%s: %s(protocol %s)' % (k, v['recount'], v['protocol']) for k, v in RC['stage0_check3'].items()),
          '', '- 欄は protocol_numbers.py(protocol で固定した台本)の関数で出し、決まりは protocol_v2.json の文から書き直した(`recount_stage12_v2.py`)。', '',
          '## 10. main session に返すこと', '',
          '1. **D3 は決まりで決まらない(まだ)。** 段 3 の 4 本と C3(robust_dl1_us)のあと。段 4 は D3 のあと。今夜に段 3・4 の 6 本を続けて回すことは、決まりの中ではできない(段 4 の `{A}` が D3 の答え)。',
          ('2. **段 3 は今は動かない。** `tools/train.py` に `--aug-top-brightness`(C1)が無い。キューの ⓪ で止まる。C1 を足してテストするかは main session が決める。' if no_c1
           else '2. `tools/train.py` に C1(`--aug-top-brightness`)がある(更新 %s。別の係)。D1・D2 が書けたので、キューを同じコマンドでもう一度動かせば段 3 の 4 本(%s-aug2-e%d-s0〜s3)は始まる。C1 のテストが済んだかは main session が確かめる。' % (c1_when, x, int(Ep))),
          ('3. **B-FAST は候補なし(拒否権「介入」)。** decisions.BF のとおり BF1・BF2 は無し。11/3 の 1 枠は C1・C2(v2)だけを見る。' if bf_veto['any'] else
           '3. B-FAST の BF1・BF2 は C3 の門のあと。'),
          '4. **C3(机上のベンチ v2)がまだ無い。** 段 0 の 2・4・8 と、v2 の門の多く(sanity・推論・低速のブレーキ・止まった場面・直線)が「未」。C1・C2 は D3 と C3 のあと。',
          '5. 段 0 の 7(中心 1433・中立 1500 と B の 1441・1496)は、11/3 に出す前にユーザーに聞く(deviations.md の 4)。',
          '6. lock2 が学習のあとになったことは deviations.md の 6 に書いた。', '',
          '## 11. 守ったこと', '',
          '- 車・PCA9685・I2C・カメラ・ToF・基板に触れていない。jetformer-web を再起動していない。走行の API は呼んでいない(jf-heavy の状態の GET だけ)。',
          '- GPU・torch の予測は 16 本とも `jf-heavy --lock big`(1 本ずつ、あいだ 30 秒)。判定・数え直しは `--lock none --mem 1500M`(torch を読まない)。',
          '- protocol_v2.json は変えていない(sha256 OK)。既定値(drive.py・推奨・drive_settings.json)は変えていない。git commit していない。', '',
          '## 12. ファイル', '',
          '| ファイル | 中身 |', '|---|---|',
          '| `decisions.json` | キューが読む答え(D1.x・D2.E・D3.aug = null)と使った数 |',
          '| `protocol_v2_lock2.json`・`.sha256` | 第 2 の固定(chmod a-w) |',
          '| `stage0_README.md` | 段 0 の答えと lock2 の sha256(ユーザーが見る) |',
          '| `eval/eval_v2.json`・`eval/eval_v2.md`・`eval/decisions_proposed.json` | eval_v2.py judge の出力 |',
          '| `eval/pred/*.npz` | 16 本の予測(best・last) |',
          '| `recount_stage12.json`・`recount_stage0_check3.json` | 数え直し |',
          '| 台本 `experiments/20261025_train_v2/` | `write_lock2_v2.py`・`recount_stage12_v2.py`・`predict_stage12_v2.sh`・`make_stage12_judge_v2.py` |']
    tmp = str(OUT_MD) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    os.replace(tmp, OUT_MD)
    print('書いた:', OUT_DEC, OUT_MD)
    print(json.dumps({k: dec[k] for k in ('D1', 'D2', 'D3')}, ensure_ascii=False)[:600])
    return 0


if __name__ == '__main__':
    sys.exit(main())
