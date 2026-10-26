#!/usr/bin/env python3
"""段 1・2 と B-FAST の判定の数え直し(2026-10-25 夜)。eval_v2.py とは別のコード。numpy だけ(torch・GPU は使わない)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/recount_stage12_v2.py [--stage0]

- 欄(H1 の L1・介入・偏り・ジャンル)は protocol_numbers.py(protocol で固定した台本)の h1_metrics・h1_genre・load_pred で出す。
- 決まり(judgement・vetoes・marks・decisions.D1・D2・gates_bfast の G-a)は protocol_v2.json の文から、この台本で書き直した。
- eval_v2.py judge の出力(eval/eval_v2.json)と、欄の値(差 ≤ 0.001 µs)・判定・D1・D2 の答えを照らす。
- 段 0 の確かめ 3: 5 本の last のジャンル別 L1 の幅 × 2 が protocol の 1.938・5.308 と同じか(差 ≤ 0.01 µs)。
出力: /home/jetson/jetformer_reports/train_v2_20261025/recount_stage12.json(これだけ)。data/・既定値は変えない。
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol_numbers as PN  # noqa: E402  (protocol で固定した台本。main は動かさない)

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PRED = Path(os.environ.get('RECOUNT_PRED_DIR') or REP / 'eval' / 'pred')            # 試しのときだけ別の場所
EVAL_JSON = Path(os.environ.get('RECOUNT_EVAL_JSON') or REP / 'eval' / 'eval_v2.json')
OUT = Path(os.environ.get('RECOUNT_OUT') or REP / 'recount_stage12.json')
RUNS = ([('bfast_20261025-e15-s%d' % s) for s in range(3)]
        + [('v2_20261025-%s-e30-s%d' % (a, s)) for s in range(4) for a in ('v2h', 'v2m')]
        + [('v2_20261025-v2h-e60-s%d' % s) for s in range(4)] + ['v2_20261025-v2h-e100-s0'])


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def arm_of(name):
    m = re.match(r'^(v2|bfast)_20261025-(.+)-s(\d+)$', name)
    return ('bfast-' if m.group(1) == 'bfast' else '') + m.group(2), int(m.group(3))


def main():
    proto = json.load(open(REP / 'protocol_v2.json', encoding='utf-8'))
    T = proto['thresholds_us']['values']
    with np.load(PN.BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != PN.TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        y = np.nan_to_num(np.asarray(z['y'], np.float64))
        prim, interv, scene, frame = z['primary'], z['interv'], z['scene'], z['frame']
    if int(prim.sum()) != 4670 or int(interv.sum()) != 1570:
        raise SystemExit('H1・介入のフレーム数が違います')
    genre = PN.h1_genre(scene, frame, prim)

    def both(pb, pl):
        return {'best': PN.h1_metrics(PN.load_pred(pb), y, prim, interv, genre),
                'last': PN.h1_metrics(PN.load_pred(pl), y, prim, interv, genre)}

    five = {rid: both(pb, pl) for rid, _run, pb, pl in PN.FIVE}
    seeds5 = {'A_s0': 0, 'A_s1': 1, 'v0-s2': 2, 'v0-s3': 3}
    f = {k: avg(five[r][ck][m] for r in five) for k, (ck, m) in {
        'l1_best': ('best', 'l1'), 'l1_last': ('last', 'l1'), 'interv_best': ('best', 'l1_interv'),
        'steer_bias_best': ('best', 'steer_bias'), 'thr_bias_best': ('best', 'thr_bias')}.items()}

    # 段 0 の確かめ 3(ジャンルの片寄りの閾値)
    st3 = {}
    for g, want in ((4, proto['marks']['genre_skew_threshold_us']['genre4_last']),
                    (5, proto['marks']['genre_skew_threshold_us']['genre5_last'])):
        vals = [five[r]['last']['l1_genre%d' % g] for r in five]
        got = 2 * (max(vals) - min(vals))
        st3['genre%d_last' % g] = {'recount': round(got, 4), 'protocol': want, 'diff': round(abs(got - want), 4),
                                   'ok': abs(got - want) <= 0.01}

    if '--stage0' in sys.argv:          # 段 0 の確かめ 3 だけ(予測の前。lock2 に書く)
        print(json.dumps({'stage0_check3': st3, 'five_mean': f}, ensure_ascii=False))
        return 0 if all(v['ok'] for v in st3.values()) else 1

    R = {}
    for n in RUNS:
        pb, pl = PRED / ('%s.npz' % n), PRED / ('%s_last.npz' % n)
        if not (pb.exists() and pl.exists()):
            raise SystemExit('予測がありません: %s' % n)
        R[n] = both(pb, pl)
    arms = {}
    for n in RUNS:
        a, s = arm_of(n)
        arms.setdefault(a, {})[s] = R[n]

    def v_vs5(rs):
        n = len(rs)
        th = T['vs5']['n%d' % n]
        res = {'n': n}
        bad = good = True
        for ck, key in (('best', 'l1_best'), ('last', 'l1_last')):
            vals = [r[ck]['l1'] for r in rs.values()]
            d = avg(vals) - f[key]
            g_ = d <= -th[key] and all(v < f[key] for v in vals)
            b_ = d >= th[key] and all(v > f[key] for v in vals)
            res[ck] = {'diff': d, 'thr': th[key], 'good': g_, 'bad': b_}
            good = good and g_
        bad = res['best']['bad'] or res['last']['bad']
        res['verdict'] = '悪い' if bad else ('良い' if good else '同等')
        return res

    def veto(rs, v2h=None):
        n = len(rs)
        th = T['vs5']['n%d' % n]
        iv = [r['best']['l1_interv'] for r in rs.values()]
        v_iv = (avg(iv) - f['interv_best'] > th['interv_best']) and all(x > f['interv_best'] for x in iv)
        if v2h is None:
            bias = avg(r['best']['steer_bias'] for r in rs.values()) - f['steer_bias_best']
            t_b = th['steer_bias_best']
        else:
            ks = sorted(set(rs) & set(v2h))
            bias = avg(rs[s]['best']['steer_bias'] - v2h[s]['best']['steer_bias'] for s in ks)
            t_b = T['pairs']['k%d' % len(ks)]['steer_bias_best']
        return {'interv': v_iv, 'interv_diff': avg(iv) - f['interv_best'], 'interv_thr': th['interv_best'],
                'bias': abs(bias) > t_b, 'bias_value': bias, 'bias_thr': t_b, 'any': bool(v_iv or abs(bias) > t_b)}

    def pair(a, b, metric, ck_list=('best', 'last')):
        ks = sorted(set(a) & set(b))
        k = len(ks)
        out = {'k': k}
        for ck in ck_list:
            d = [a[s][ck][metric] - b[s][ck][metric] for s in ks]
            out[ck] = {'d': d, 'mean': avg(d)}
        return out

    arm_res = {}
    for a, rs in arms.items():
        arm_res[a] = {'seeds': sorted(rs), 'vs5': v_vs5(rs),
                      'veto': veto(rs, arms['v2h-e30'] if a == 'v2m-e30' else None),
                      'too_long': sum(1 for r in rs.values() if r['last']['l1'] - r['best']['l1'] > 1.6)}

    # D1
    h = arm_res['v2h-e30']
    if h['vs5']['verdict'] == '悪い' or h['veto']['any']:
        x, why1 = 'base', '① v2h-e30 が「悪い」か拒否権'
        cond = None
    else:
        k4 = T['pairs']['k4']
        p_h1 = pair(arms['v2m-e30'], arms['v2h-e30'], 'l1')
        p_iv = pair(arms['v2m-e30'], arms['v2h-e30'], 'l1_interv')
        p_sb = pair(arms['v2m-e30'], arms['v2h-e30'], 'steer_bias', ('best',))
        p_tb = pair(arms['v2m-e30'], arms['v2h-e30'], 'thr_bias', ('best',))

        def good(p, tb, tl):
            return (p['best']['mean'] <= -tb and p['last']['mean'] <= -tl and all(d < 0 for d in p['best']['d'])
                    and all(d < 0 for d in p['last']['d']))
        cond = {
            'H1 が悪くない(best・last の d の平均 ≤ +0.5)': p_h1['best']['mean'] <= 0.5 and p_h1['last']['mean'] <= 0.5,
            'steer_bias の差の平均が ±4.125 の中': abs(p_sb['best']['mean']) <= k4['steer_bias_best'],
            'l1_interv の差の平均 ≤ +1.718(best)': p_iv['best']['mean'] <= k4['interv_best'],
            'thr_bias の差の平均 ≥ −2.887(best)': p_tb['best']['mean'] >= -k4['thr_bias_best'],
            'H1 か介入で「良い」': good(p_h1, k4['l1_best'], k4['l1_last']) or good(p_iv, k4['interv_best'], k4['interv_last']),
            'v2-M に拒否権が無い': not arm_res['v2m-e30']['veto']['any'],
        }
        x = 'v2m' if all(cond.values()) else 'v2h'
        why1 = '② 全部を満たした' if x == 'v2m' else '② 欠けた: ' + '・'.join(c for c, v in cond.items() if not v)
        d1_vals = {'H1': p_h1, '介入': p_iv, 'steer_bias': p_sb, 'thr_bias': p_tb}
    # D2(x = v2h のときだけこの台本で数える)
    d2 = {}
    if x == 'v2h':
        a60, a30 = arms['v2h-e60'], arms['v2h-e30']
        p = pair(a60, a30, 'l1', ('last',))
        thr = T['pairs']['k%d' % p['k']]['l1_last']
        bad = p['last']['mean'] >= thr and all(d > 0 for d in p['last']['d'])
        t4 = proto['marks']['genre_skew_threshold_us']['genre4_last']
        t5 = proto['marks']['genre_skew_threshold_us']['genre5_last']
        skew = {}
        for s in sorted(set(a60) & set(a30)):
            d4 = a60[s]['last']['l1_genre4'] - a30[s]['last']['l1_genre4']
            d5 = a60[s]['last']['l1_genre5'] - a30[s]['last']['l1_genre5']
            skew[s] = {'d4': d4, 'd5': d5, 'mark': bool((d4 > t4 and d5 <= t5) or (d5 > t5 and d4 <= t4))}
        tl = arm_res['v2h-e60']['too_long']
        vt = arm_res['v2h-e60']['veto']['any']
        reasons = [r for r, c in (('60 が「悪い」', bad), ('「長すぎ」2 本以上', tl >= 2),
                                  ('「ジャンルの片寄り」2 組以上', sum(v['mark'] for v in skew.values()) >= 2),
                                  ('60 の腕に拒否権', vt)) if c]
        d2 = {'E': 30 if reasons else 60, 'reasons': reasons, 'pair_last': p, 'thr_last': thr, 'bad': bad,
              'too_long': tl, 'genre_skew': skew, 'veto_60': vt}
    # B-FAST の G-a(この台本で出せる門だけ)
    ga = {s: {'l1_best': r['best']['l1'], 'limit': five['B']['best']['l1'] + 1.0,
              'pass': r['best']['l1'] <= five['B']['best']['l1'] + 1.0} for s, r in arms['bfast-e15'].items()}

    # eval_v2.json と照らす
    cmp = {'max_abs_diff_us': None, 'fields': 0, 'verdicts_same': None, 'D1_same': None, 'D2_same': None}
    if EVAL_JSON.exists():
        E = json.load(open(EVAL_JSON, encoding='utf-8'))
        rows = {r['run']: r for r in E['runs']}
        mx, nf = 0.0, 0
        mp = {'l1_best_us': ('best', 'l1'), 'l1_last_us': ('last', 'l1'), 'l1_interv_us': ('best', 'l1_interv'),
              'l1_interv_last_us': ('last', 'l1_interv'), 'steer_bias_us': ('best', 'steer_bias'),
              'thr_bias_us': ('best', 'thr_bias'), 'l1_genre4_last_us': ('last', 'l1_genre4'),
              'l1_genre5_last_us': ('last', 'l1_genre5'), 'l1_genre4_us': ('best', 'l1_genre4'),
              'l1_genre5_us': ('best', 'l1_genre5')}
        for n in RUNS:
            for k, (ck, m) in mp.items():
                mx = max(mx, abs(rows[n][k] - R[n][ck][m]))
                nf += 1
        J = E['judge']
        vs = {a: (J['arms'][a]['vs5']['verdict'] == arm_res[a]['vs5']['verdict']
                  and bool(J['arms'][a]['veto']['any']) == arm_res[a]['veto']['any']
                  and J['arms'][a]['too_long'] == arm_res[a]['too_long']) for a in arm_res}
        cmp = {'max_abs_diff_us': round(mx, 6), 'fields': nf, 'fields_ok': mx <= 0.001, 'verdicts_same': vs,
               'D1_same': J['decisions']['D1'].get('x') == x, 'D2_same': J['decisions']['D2'].get('E') == d2.get('E'),
               'eval_made_at': E['meta']['made_at']}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (float, np.floating)):
            return round(float(o), 4) if math.isfinite(o) else None
        return o
    out = {'made_at': __import__('time').strftime('%Y-%m-%d %H:%M:%S'), 'script': str(Path(__file__)),
           'five_mean': f, 'stage0_check3': st3, 'runs': R, 'arms': arm_res,
           'D1': {'x': x, 'why': why1, 'conditions': cond, 'values': d1_vals if cond is not None else None},
           'D2': d2, 'bfast_G_a': ga, 'compare_eval_v2': cmp}
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(clean(out), fh, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    print('段 0 の確かめ 3:', json.dumps(clean(st3), ensure_ascii=False))
    print('D1 x =', x, '/', why1)
    print('D2 E =', d2.get('E'), '/', d2.get('reasons'))
    print('照らし:', json.dumps(clean(cmp), ensure_ascii=False))
    print('書いた:', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
