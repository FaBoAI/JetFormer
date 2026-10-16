#!/usr/bin/env python3
"""電源断(10/15 23:46〜10/16 10:03)で 2 本目の種を削った腕の「種 1 本だけの参考」(判定はしない)。numpy なし。

- GAINL(protocol の任意の腕): gainl-s1 は 8 エポック目で電源断 → 削った。protocol の主の問いは
  [GAIN(gainl-s0), GAIN(gainl-s1)] vs [GAIN(A_s0), GAIN(A_s1)] で、2 つの種が要るので判定しない。
  ここでは同じ種 0 どうし GAIN(gainl-s0) − GAIN(A_s0) を、閾値と並べて参考に出すだけ。
- SIZE-EQ(固定の外の参考の腕・deviations.md 2026-10-15 21:55): sizeHeq-s0 だけ済んだ(s1・Q は削った)。
  21:55 に決めた読み方(s2_eq ≥ 1 µs なら「ステップ数をそろえてもデータの量の効果が残る」)を、同じ種 0 どうしの
  s2_eq(s0) = (L(sizeHeq-s0) − L(A_s0)) / log2(N_全部 / N_H) に当てはめる。比べる相手は同じ種の s2(s0)(30 エポック)。
"""
import math

N_H, N_FULL = 3046, 5912
CKPTS = ('best', 'last')
PRIMARY_THR = {'best': 0.626, 'last': 1.322}
SEC_THR = {'S1_no_battery_scene': {'best': 1.134, 'last': 1.98}, 'S3_intervention_part1': {'best': 3.458, 'last': 4.858}}
METRICS = ('primary', 'S1_no_battery_scene', 'S3_intervention_part1', 'battery_scene_only')


def _thr(metric, ck):
    return PRIMARY_THR[ck] if metric == 'primary' else (SEC_THR.get(metric) or {}).get(ck)


def pair_diff(pv, arm, ref):
    """同じ種どうしの差(腕 − 基準、負 = 腕が良い)。閾値がある指標は、差が閾値の中か外かも"""
    out = {}
    for ck in CKPTS:
        row = {}
        for m in METRICS:
            a, r = float(pv[arm][ck][m]), float(pv[ref][ck][m])
            t = _thr(m, ck)
            row[m] = {'arm': a, 'ref': r, 'diff': a - r, 'threshold': t,
                      'within_threshold': None if t is None else abs(a - r) <= t}
        out[ck] = row
    return out


def size_eq_s0(pv):
    d = math.log2(N_FULL / N_H)
    out = {}
    for ck in CKPTS:
        for m in ('primary', 'S1_no_battery_scene'):
            lh, lhe, lf = (float(pv[n][ck][m]) for n in ('sizeH-s0', 'sizeHeq-s0', 'A_s0'))
            gap = lh - lf
            out['%s/%s' % (m, ck)] = {'L_H_s0_30ep': lh, 'L_Heq_s0_59ep': lhe, 'L_full_A_s0': lf,
                                      's2_s0': gap / d, 's2_eq_s0': (lhe - lf) / d,
                                      'share_of_gap_from_steps': (lh - lhe) / gap if gap > 0 else None}
    return out


def single_seed_reference(pv):
    """evaluation.json の per_variant → 種 1 本だけの参考(無ければその項目は None)"""
    res = {'note': '電源断(10/15 23:46〜10/16 10:03)で 2 本目の種を削った腕の、種 1 本だけの参考。判定はしない(規則は 2 つの種を求める)',
           'cut': ['gainl-s1(8 エポック目で電源断)', 'sizeHeq-s1', 'sizeQeq-s0', 'sizeQeq-s1'],
           'GAINL_s0': None, 'SIZE_EQ_s0': None}
    if all(n in pv for n in ('GAIN(gainl-s0)', 'GAIN(A_s0)', 'gainl-s0', 'A_s0')):
        post = pair_diff(pv, 'GAIN(gainl-s0)', 'GAIN(A_s0)')
        raw = pair_diff(pv, 'gainl-s0', 'A_s0')
        inside = all(post[ck]['primary']['within_threshold'] for ck in CKPTS)
        res['GAINL_s0'] = {
            'question': 'GAINL + 後処理 vs A + 後処理(同じ種 0 どうし)', 'post': post, 'raw_vs_A_s0': raw,
            'verdict': None,
            'text': ('種 0 どうしの差 best %+.2f・last %+.2f µs は、どちらも閾値(0.626・1.322)の中。'
                     'ラベルの側に入れても、後処理だけより上という兆しは無い(種 1 本なので判定はしない)' if inside else
                     '種 0 どうしの差 best %+.2f・last %+.2f µs。閾値の外の差があるが、種 1 本なので判定はしない') % (
                         post['best']['primary']['diff'], post['last']['primary']['diff'])}
        worse = ['%s %+.2f µs(閾値 %.3f)' % (ck, post[ck]['S3_intervention_part1']['diff'],
                                             post[ck]['S3_intervention_part1']['threshold'])
                 for ck in CKPTS if post[ck]['S3_intervention_part1']['diff'] > post[ck]['S3_intervention_part1']['threshold']]
        if worse:
            res['GAINL_s0']['text'] += '。介入の集合(S3、拒否権の指標)では悪い向きで閾値を超えた: ' + '・'.join(worse)
    if all(n in pv for n in ('sizeH-s0', 'sizeHeq-s0', 'A_s0')):
        rows = size_eq_s0(pv)
        s2e = rows['primary/best']['s2_eq_s0']
        res['SIZE_EQ_s0'] = {
            'by_metric': rows, 'epochs': {'H': 59, 'full': 30}, 'N': {'H': N_H, 'full': N_FULL},
            'text': ('ステップ数をそろえてもデータの量の効果が残る(種 0 だけ: s2_eq = %.2f µs / 2 倍、30 エポックの s2 = %.2f)' if s2e >= 1.0
                     else 'データの量の効果は、ステップ数の差で説明できる部分が大きい(種 0 だけ: s2_eq = %.2f µs / 2 倍、30 エポックの s2 = %.2f)')
            % (s2e, rows['primary/best']['s2_s0'])}
    return res
