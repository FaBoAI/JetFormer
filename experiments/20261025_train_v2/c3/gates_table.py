#!/usr/bin/env python3
"""C3 の desk_v2.json から、v2 と B-FAST の門の表(C3 の欄で埋まったもの)を md と json に書く(2026-10-26)。

門の判定は desk_v2.py report が eval_v2.py の gates_v2・gates_bfast・gates_pass をそのまま呼んで desk_v2.json に入れた値を写すだけ
(この台本は判定しない。値と閾値を並べる)。ジャンル 9 の門(gates_v2.genre9)は eval_v2 に無いので、protocol の文
「g9_minus_g5_pre_thr_us ≥ −5 µs」で値を並べ、参考と書く(D1 は x = v2h)。
読むだけ: desk/desk_v2.json・decisions.json。書くもの: c3_gates_20261026.md・c3_gates_20261026.json(レポートの場所)。

    python3 -I experiments/20261025_train_v2/c3/gates_table.py
"""
import json
import sys
import time
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
DESK = REP / 'desk' / 'desk_v2.json'
DEC = REP / 'decisions.json'
OUT_MD = REP / 'c3_gates_20261026.md'
OUT_JSON = REP / 'c3_gates_20261026.json'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
INFER_FACTOR, BRAKE_FRAC, STAND_MARGIN, STRAIGHT_US = 1.3, 0.01, 0.10, 10.0
GB_MIN, GC_MAX, G9_MIN, GA_PLUS = 10.0, 5.0, -5.0, 1.0
V2_C3 = ('sanity_ok', 'inference', 'brake', 'standstill', 'straight')
V2_OTHER = ('engine_ok', 'amp', 'l1_vs_b', 'human_throttle')
BF_C3 = ('G-b', 'G-c', 'G-d', 'G-f', 'sanity_ok', 'inference', 'brake')
BF_OTHER = ('G-a', 'G-e', 'amp')
NAMES = {'sanity_ok': 'sanity', 'inference': '推論', 'brake': '低速のブレーキ', 'standstill': '止まった場面', 'straight': '直線',
         'engine_ok': 'engine_ok', 'amp': '振れ幅', 'l1_vs_b': 'L1 対 B', 'human_throttle': '人の走りのスロットル'}


def mark(v):
    if v is None:
        return '未'
    if v == '参考':
        return '参考'
    return '○' if v else '×'


def f(v, nd=2):
    if v is None:
        return '未'
    if isinstance(v, bool):
        return '○' if v else '×'
    if isinstance(v, (int, float)):
        return ('%%.%df' % nd) % v
    return str(v)


def passed(v):
    return {True: '**全部合格**', False: '**不合格**', None: '未'}[v]


def seed_of(n):
    return int(n.rsplit('-s', 1)[1])


def arm_of(n):
    return n.rsplit('-s', 1)[0]


def main():
    d = json.loads(DESK.read_text(encoding='utf-8'))
    dec = json.loads(DEC.read_text(encoding='utf-8')) if DEC.exists() else {}
    runs = d['runs']
    b = runs.get(B_RUN) or {}
    bp99 = d.get('b_trt_p99_ms_same_day')
    inf_lim = INFER_FACTOR * bp99 if bp99 else None
    b_go = b.get('standstill_go_frac')
    go_lim = (b_go - STAND_MARGIN) if b_go is not None else None
    gf_lim = d.get('bfast_gf_limit_us')
    b_l1 = b.get('l1_best_us')
    v2 = sorted([n for n, r in runs.items() if r.get('family') == 'v2'], key=lambda n: (arm_of(n), seed_of(n)))
    bf = sorted([n for n, r in runs.items() if r.get('family') == 'bfast'], key=seed_of)
    ref = [n for n, r in runs.items() if r.get('family') == 'ref']
    L = ['# C3 の門の表(v2・B-FAST)', '',
         '- 作った時刻: %s(desk_v2.json の made_at %s)' % (time.strftime('%Y-%m-%d %H:%M'), d.get('made_at')),
         '- もと: `%s`(C3 sha256 `%s`・protocol `%s`・lock2 `%s`)' % (DESK, (d.get('script_sha256') or '')[:12],
                                                                (d.get('protocol_v2_sha256') or '')[:12], (d.get('lock2_sha256') or '')[:12]),
         '- 判定は desk_v2.py report が eval_v2.py の gates_v2・gates_bfast・gates_pass をそのまま呼んだ値(この表は写すだけ)。'
         '「未」= 欄が無い(合格にしない)。○ = 合格・× = 不合格。',
         '- 決まりの数: 推論 ≤ 1.3 × 同じ日の B の p99 = %s ms(B %s ms・%s)・低速のブレーキ ≤ 0.01・止まった場面 ≥ B − 0.10 = %s'
         '(%s フレーム → %s)・直線 |差| ≤ 10 µs・G-b ≥ +10 µs・G-c ≤ +5 µs・G-f < %s µs・G-a ≤ B の l1 + 1.0 = %s µs' % (
             f(inf_lim, 3), f(bp99, 3), d.get('b_trt_p99_day'), f(go_lim, 4), d.get('standstill_frames'), d.get('standstill_gate'),
             f(gf_lim, 2), f((b_l1 + GA_PLUS) if b_l1 is not None else None, 3)),
         '- B の行(基準): standstill_go_frac %s・trt p99 %s ms・低速のブレーキ %s・l1_best %s µs' % (
             f(b_go, 4), f(b.get('trt_p99_ms'), 3), f(b.get('brake_lowspeed_frac'), 4), f(b_l1, 3)), '']
    out = {'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'desk_made_at': d.get('made_at'), 'desk_script_sha256': d.get('script_sha256'),
           'limits': {'inference_ms': inf_lim, 'b_trt_p99_ms_same_day': bp99, 'standstill_go_frac_min': go_lim,
                      'standstill_frames': d.get('standstill_frames'), 'standstill_gate': d.get('standstill_gate'),
                      'bfast_gf_limit_us': gf_lim, 'brake_max': BRAKE_FRAC, 'straight_abs_max_us': STRAIGHT_US,
                      'gb_min_us': GB_MIN, 'gc_max_us': GC_MAX, 'g9_min_us': G9_MIN},
           'v2': {}, 'bfast': {}}
    # ---- v2
    L += ['## v2 の門(gates_v2)', '',
          'C3 の欄で決まる門: sanity・推論・低速のブレーキ・止まった場面・直線。ほか(engine_ok・振れ幅・L1 対 B・人の走りのスロットル)は H1・engine_check で決まる(前から埋まっている)。', '',
          '| 学習 | sanity | 推論 p99 ms | 低速のブレーキ | 止まった場面 go_frac | 直線 thr 差 µs | engine_ok | 振れ幅 | L1 対 B | 人の走りのスロットル | 門 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for n in v2:
        r = runs[n]
        g = r.get('gates') or {}
        L.append('| %s | %s | %s %s | %s %s | %s %s | %s %s | %s | %s | %s | %s | %s |' % (
            n.replace('v2_20261025-', ''), mark(g.get('sanity_ok')),
            f(r.get('trt_p99_ms'), 3), mark(g.get('inference')),
            f(r.get('brake_lowspeed_frac'), 4), mark(g.get('brake')),
            f(r.get('standstill_go_frac'), 3), mark(g.get('standstill')),
            f(r.get('straight_thr_vs_b_us'), 2), mark(g.get('straight')),
            mark(g.get('engine_ok')), mark(g.get('amp')), mark(g.get('l1_vs_b')), mark(g.get('human_throttle')),
            passed(r.get('gates_pass'))))
        out['v2'][n] = {'gates': g, 'gates_pass': r.get('gates_pass'),
                        'values': {k: r.get(k) for k in ('sanity_ok', 'trt_p99_ms', 'brake_lowspeed_frac', 'standstill_go_frac',
                                                         'straight_thr_vs_b_us', 'steer_amp_ratio', 'l1_best_us', 'thr_bias_us',
                                                         'g9_minus_g5_pre_thr_us', 'robust_dl1_us')}}
    fails = {}
    for n in v2:
        g = runs[n].get('gates') or {}
        for k, v in g.items():
            if v is False:
                fails.setdefault(k, []).append(n.replace('v2_20261025-', ''))
    L += ['', '- 落ちた門の数え: ' + ('・'.join('%s %d 本(%s)' % (NAMES.get(k, k), len(v), ' '.join(v)) for k, v in fails.items()) or 'なし')]
    v2m = [n for n in v2 if '-v2m-' in n]
    if v2m:
        L += ['- ジャンル 9(gates_v2.genre9。x = v2m のときだけ。D1 は x = %s なので参考): g9_minus_g5_pre_thr_us ≥ −5 µs → %s' % (
            (dec.get('D1') or {}).get('x'),
            '・'.join('%s %s %s' % (n.replace('v2_20261025-', ''), f(runs[n].get('g9_minus_g5_pre_thr_us')),
                                    mark(None if runs[n].get('g9_minus_g5_pre_thr_us') is None else runs[n]['g9_minus_g5_pre_thr_us'] >= G9_MIN))
                     for n in v2m))]
    # D5a の見え方(D3 の答えごと。決めない)
    L += ['- 参考(決めない): 最終の腕は D1 x = v2h・D2 E = 60 と D3 の答えで、「なし」なら v2h-e60、AUG2 なら v2h-aug2-e60。'
          'D5a は種 0 → 3 の順に門に全部合格した最初 = C1・次 = C2(腕に拒否権があれば無し)。decisions.json の D3.aug = %s'
          % json.dumps((dec.get('D3') or {}).get('aug'), ensure_ascii=False)]
    for arm, d3 in (('v2_20261025-v2h-e60', 'なし'), ('v2_20261025-v2h-aug2-e60', 'AUG2')):
        rows_ = sorted([n for n in v2 if arm_of(n) == arm], key=seed_of)
        if not rows_:
            L.append('  - D3 = %s(%s): 行が無い(未)' % (d3, arm.replace('v2_20261025-', '')))
            continue
        ok = [n for n in rows_ if runs[n].get('gates_pass') is True]
        und = [n for n in rows_ if runs[n].get('gates_pass') is None]
        L.append('  - D3 = %s(%s): %s → 門に全部合格 %s%s' % (
            d3, arm.replace('v2_20261025-', ''),
            '・'.join('s%d %s' % (seed_of(n), {True: '○', False: '×', None: '未'}[runs[n].get('gates_pass')]) for n in rows_),
            ' '.join('s%d' % seed_of(n) for n in ok) or 'なし', ('(未 %s)' % ' '.join('s%d' % seed_of(n) for n in und)) if und else ''))
    # ---- B-FAST
    L += ['', '## B-FAST の門(gates_bfast。参考: decisions.json BF「無し」= 腕に拒否権)', '',
          'C3 の欄で決まる門: G-b・G-c・G-d・G-f・sanity・推論・低速のブレーキ。G-a(H1)・G-e(engine_check)・振れ幅は前から。', '',
          '| 学習 | G-a | G-b fwd 差 µs(DE・SW・HS) | G-c fwd 差 µs | G-d go_frac | G-e | G-f 直線 fwd 差 µs | sanity | 推論 p99 ms | 振れ幅 | 低速のブレーキ | 門 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n in bf:
        r = runs[n]
        g = r.get('gates') or {}
        L.append('| %s | %s | %s(%s・%s・%s)%s | %s %s | %s %s | %s | %s %s | %s | %s %s | %s | %s %s | %s |' % (
            n.replace('bfast_20261025-', 'bfast-'), mark(g.get('G-a')),
            f(r.get('bfast_gb_fwd_vs_b_us')), f(r.get('bfast_gb_fwd_vs_b_us_de')), f(r.get('bfast_gb_fwd_vs_b_us_sw')),
            f(r.get('bfast_gb_fwd_vs_b_us_hs')), mark(g.get('G-b')),
            f(r.get('bfast_gc_fwd_vs_b_us')), mark(g.get('G-c')),
            f(r.get('standstill_go_frac'), 3), mark(g.get('G-d')), mark(g.get('G-e')),
            f(r.get('straight_fwd_vs_b_us')), mark(g.get('G-f')), mark(g.get('sanity_ok')),
            f(r.get('trt_p99_ms'), 3), mark(g.get('inference')), mark(g.get('amp')),
            f(r.get('brake_lowspeed_frac'), 4), mark(g.get('brake')), passed(r.get('gates_pass'))))
        out['bfast'][n] = {'gates': g, 'gates_pass': r.get('gates_pass'),
                           'values': {k: r.get(k) for k in ('bfast_gb_fwd_vs_b_us', 'bfast_gb_fwd_vs_b_us_de', 'bfast_gb_fwd_vs_b_us_sw',
                                                            'bfast_gb_fwd_vs_b_us_hs', 'bfast_gc_fwd_vs_b_us', 'standstill_go_frac',
                                                            'straight_fwd_vs_b_us', 'trt_p99_ms', 'brake_lowspeed_frac',
                                                            'bfast_gb_ref_fwd_vs_b_us', 'bfast_gb_fwd_g100_vs_b090_us',
                                                            'bfast_de_line_vs_b_us', 'lap3_est_s', 'l1_best_us')}}
    if bf:
        L += ['', '- 参考の欄: ' + ' / '.join('%s: G-b 参考(10/24 2 回目)%s・1.00 対 0.90 %s・DE の線 %s・3 周の見込み %s 秒' % (
            n.replace('bfast_20261025-', 'bfast-'), f(runs[n].get('bfast_gb_ref_fwd_vs_b_us')),
            f(runs[n].get('bfast_gb_fwd_g100_vs_b090_us')), f(runs[n].get('bfast_de_line_vs_b_us')), f(runs[n].get('lap3_est_s')))
            for n in bf),
              '- B-FAST の腕は介入の拒否権で候補にしない(decisions.json BF)。この表は 11/3 の判断に使わない参考。']
    # ---- 基準(門に掛けない)
    L += ['', '## 基準の 5 本と 2 本の平均(門に掛けない。欄だけ)', '',
          '| 学習 | robust | 止まった場面 go_frac | 直線 thr 差 | sanity | p99 ms | 低速のブレーキ |', '|---|---|---|---|---|---|---|']
    for n in sorted(ref, key=lambda x: (x != B_RUN, x)):
        r = runs[n]
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (n, f(r.get('robust_dl1_us'), 3), f(r.get('standstill_go_frac'), 3),
                                                       f(r.get('straight_thr_vs_b_us')), mark(r.get('sanity_ok')),
                                                       f(r.get('trt_p99_ms'), 3), f(r.get('brake_lowspeed_frac'), 4)))
    OUT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('書きました: %s・%s(v2 %d 本・B-FAST %d 本)' % (OUT_MD, OUT_JSON, len(v2), len(bf)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
