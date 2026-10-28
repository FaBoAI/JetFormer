"""fix8 係 safety 直す 8-1 回目: sweep_r1 の出力を数える(票 A の ana_A.py と同じ分け方: 両方当たる組の速さの差が 0.1 m/s より
大きいか・片方だけ当たるか)。案(qA・qB・qC の両方 ON)を、今の両方 OFF と今の両方 ON の両方と比べる。距離の帯ごと・ESC の模型ごと。
下がる距離・escape の間の前の壁までの最短・案の写しの両方 OFF が今の OFF と同じか・リカバリーの前が同じか。
    python3 -I -B ana_r1.py → out/ana_r1.txt・out/ana_r1.json
"""
import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TH = 0.1
PROPS = ('qA', 'qB', 'qC')


def cls(a, b):
    """a(比べる側)と b(基準)の当たりの速さ → 分け方"""
    if a is None and b is None:
        return 'neither'
    if b is None:
        return 'a_only'
    if a is None:
        return 'b_only'
    if a > b + TH:
        return 'a_faster'
    if b > a + TH:
        return 'b_faster'
    return 'same'


def summ(pairs):
    c = collections.Counter(cls(a, b) for a, b in pairs)
    dif = [a - b for a, b in pairs if cls(a, b) == 'a_faster']
    e_a = sum((a or 0.0) ** 2 for a, _ in pairs)
    e_b = sum((b or 0.0) ** 2 for _, b in pairs)
    return {'n': len(pairs), 'counts': dict(sorted(c.items())),
            'a_faster_by': None if not dif else {'mean': round(sum(dif) / len(dif), 3), 'max': round(max(dif), 3)},
            'hits_a': sum(1 for a, _ in pairs if a is not None), 'hits_b': sum(1 for _, b in pairs if b is not None),
            'mean_v2_a': round(e_a / max(len(pairs), 1), 3), 'mean_v2_b': round(e_b / max(len(pairs), 1), 3)}


def band(d):
    lo = math.floor(d * 2 + 1e-9) / 2
    return '%.1f-%.1f' % (lo, lo + 0.5)


def load():
    rows, ds = [], None
    for f in sorted(HERE.glob('out/sweep_r1_*of*.json')) or [HERE / 'out' / 'sweep_r1.json']:
        J = json.loads(f.read_text())
        ds = J['ds']
        rows += J['rows']
    return ds, rows


def main():
    ds, rows = load()
    out = {'cells': len(rows), 'esc_models': len({r['esc'] for r in rows}), 'pairs_per_form': len(rows) * len(ds)}
    lines = ['cells %d (ESC %d × 場面 %d)・距離 %d 点(%.2f〜%.2f m)= %d 組' % (
        len(rows), out['esc_models'], len(rows) // max(out['esc_models'], 1), len(ds), ds[0], ds[-1], len(rows) * len(ds))]
    # 0) 確かめ: 案の写しの OFF = 今の OFF(書いた値の列)、リカバリーの前は同じ
    off_bad = {p: [(r['esc'], r['m'], r['xf'], r['brake_us'], r['veh_dt']) for r in rows
                   if r['%s_off' % p]['w_sha'] != r['now_off']['w_sha']] for p in PROPS}
    pre_bad = {p: [(r['esc'], r['m'], r['xf'], r['brake_us'], r['veh_dt']) for r in rows
                   if r['%s_on' % p].get('pre_sha') != r['now_on'].get('pre_sha')
                   or (r['%s_on' % p].get('rec') or [None])[0] != (r['now_on'].get('rec') or [None])[0]] for p in PROPS}
    no_rec = [(r['esc'], r['m']) for r in rows if any(r[k].get('rec') is None for k in r if isinstance(r[k], dict))]
    out['off_differs'] = off_bad
    out['pre_recovery_differs'] = pre_bad
    out['no_recovery'] = no_rec
    lines.append('案の OFF が今の OFF と違う cells: %s / リカバリーの前が今の ON と違う: %s / リカバリーなし: %d' % (
        {p: len(v) for p, v in off_bad.items()}, {p: len(v) for p, v in pre_bad.items()}, len(no_rec)))
    # 1) 当たりの速さ
    for p in PROPS:
        for base in ('now_off', 'now_on'):
            allp, by_band, by_esc = [], collections.defaultdict(list), collections.defaultdict(list)
            for r in rows:
                for d in ds:
                    a, b = r['%s_on' % p]['hit'][str(d)], r[base]['hit'][str(d)]
                    allp.append((a, b))
                    by_band[band(d)].append((a, b))
                    by_esc[r['esc']].append((a, b))
            key = '%s_vs_%s' % (p, base)
            out[key] = {'all': summ(allp), 'by_band': {k: summ(v) for k, v in sorted(by_band.items())},
                        'by_esc': {k: summ(v) for k, v in sorted(by_esc.items())}}
            lines.append('== %s ON 対 %s: %s' % (p, base, json.dumps(out[key]['all'], ensure_ascii=False)))
            for k, v in out[key]['by_band'].items():
                lines.append('   %s %s faster_by %s v2 %.3f / %.3f' % (k, v['counts'], v['a_faster_by'], v['mean_v2_a'], v['mean_v2_b']))
            worst = sorted(((v['counts'].get('a_faster', 0) + v['counts'].get('a_only', 0), k) for k, v in out[key]['by_esc'].items()),
                           reverse=True)[:4]
            lines.append('   ESC ごとの「案が速い・案だけ当たる」の多い順: %s' % worst)
    # 今の ON 対 今の OFF(票 A の数の照らし)
    allp, by_band = [], collections.defaultdict(list)
    for r in rows:
        for d in ds:
            a, b = r['now_on']['hit'][str(d)], r['now_off']['hit'][str(d)]
            allp.append((a, b))
            by_band['lt1' if d < 1.0 - 1e-9 else 'ge1'].append((a, b))
    out['now_on_vs_now_off'] = {'all': summ(allp), 'by_band': {k: summ(v) for k, v in by_band.items()}}
    lines.append('== 今の ON 対 今の OFF: %s' % json.dumps(out['now_on_vs_now_off'], ensure_ascii=False))
    # 2) 下がる距離・最高の後ろへの速さ・後退側を書いた秒・escape の間の前の壁までの最短
    forms = ('now_on', 'now_off') + tuple('%s_on' % p for p in PROPS)
    stat = {}
    for f in forms:
        bk = [r[f]['back'] for r in rows]
        vp = [r[f]['vpk'] for r in rows]
        rc = [r[f]['rev_cmd_s'] for r in rows]
        fg = [r[f]['front_gap_in_escape'] for r in rows]
        push = [r[f]['esc_push'] for r in rows]
        stat[f] = {'back_min': min(bk), 'back_med': sorted(bk)[len(bk) // 2], 'back_max': max(bk),
                   'vpk_med': sorted(vp)[len(vp) // 2], 'vpk_max': max(vp), 'rev_cmd_med': sorted(rc)[len(rc) // 2],
                   'front_gap_min': min(fg), 'front_gap_lt_0p1': sum(1 for x in fg if x < 0.1),
                   'front_gap_lt_0p3': sum(1 for x in fg if x < 0.3), 'esc_push_cells': sum(1 for x in push if x > 0),
                   'esc_front_hits': sum(len(r[f]['esc_front_hits']) for r in rows)}
        lines.append('%-8s back %.3f / %.3f / %.3f m・vpk 中央 %.2f 最大 %.2f m/s・後退側 中央 %.2f 秒・escape 中の前の壁まで 最短 %.3f m'
                     '(0.1 m 未満 %d・0.3 m 未満 %d cells)・escape の壁押し %d cells・前の当たり %d' % (
                         f, stat[f]['back_min'], stat[f]['back_med'], stat[f]['back_max'], stat[f]['vpk_med'], stat[f]['vpk_max'],
                         stat[f]['rev_cmd_med'], stat[f]['front_gap_min'], stat[f]['front_gap_lt_0p1'],
                         stat[f]['front_gap_lt_0p3'], stat[f]['esc_push_cells'], stat[f]['esc_front_hits']))
    out['stats'] = stat
    # escape の間の前の壁までの最短が 0.3 m より近い ESC の模型(最短 m・0.1 m より近い cells)
    out['front_gap_by_esc'] = {}
    for f in forms:
        c = collections.defaultdict(list)
        for r in rows:
            c[r['esc']].append(r[f]['front_gap_in_escape'])
        out['front_gap_by_esc'][f] = {e: [round(min(v), 3), sum(1 for x in v if x < 0.1)] for e, v in sorted(c.items())
                                      if min(v) < 0.3}
        lines.append('escape の間に前の壁まで 0.3 m より近づいた ESC(%s): %s' % (f, out['front_gap_by_esc'][f]))
    # 下がる距離の ESC ごと(中央)
    by_esc_back = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for f in forms:
            by_esc_back[r['esc']][f].append(r[f]['back'])
    out['back_by_esc_median'] = {e: {f: sorted(v)[len(v) // 2] for f, v in d.items()} for e, d in sorted(by_esc_back.items())}
    lines.append('ESC ごとの下がる距離の中央 [m](%s):' % ' / '.join(forms))
    for e, d in out['back_by_esc_median'].items():
        lines.append('   %-16s %s' % (e, ' / '.join('%.3f' % d[f] for f in forms)))
    # 3) 10/23 No.6 に合わせた模型(fit)の例: 前進 120・壁 2.4・手前 0.45・ブレーキ側 200・100 Hz
    ex = next(r for r in rows if r['esc'] == 'fit' and r['m'] == 120.0 and r['xf'] == 2.4 and r['brake_us'] == 200.0
              and r['veh_dt'] == 0.01)
    out['example_fit'] = {f: {'back': ex[f]['back'], 'vpk': ex[f]['vpk'], 'phases': ex[f]['phases'], 'events': ex[f]['events'],
                              'by_d': {k: ex[f]['hit'][k] for k in ('0.3', '0.5', '0.7', '0.9', '1.1', '1.2', '1.3', '1.5')}}
                          for f in forms}
    lines.append('fit の例(前進 120・壁 2.4・100 Hz)の当たりの速さ [m/s] d = 0.3 / 0.5 / 0.7 / 0.9 / 1.1 / 1.2 / 1.3 / 1.5 m:')
    for f in forms:
        e = out['example_fit'][f]
        lines.append('   %-8s back %.3f %s' % (f, e['back'], ' / '.join(str(v) for v in e['by_d'].values())))
    # 10/23 No.6 と同じ距離(止まった所から約 1.04〜1.24 m)の fit の 16 場面
    near6 = [k for k in ex['now_on']['hit'] if 1.04 <= float(k) <= 1.24]
    fit_rows = [r for r in rows if r['esc'] == 'fit']
    out['fit_no6_band'] = {f: {'hits': sum(1 for r in fit_rows for k in near6 if r[f]['hit'][k] is not None),
                               'n': len(fit_rows) * len(near6),
                               'v_med': (lambda v: sorted(v)[len(v) // 2] if v else None)(
                                   [r[f]['hit'][k] for r in fit_rows for k in near6 if r[f]['hit'][k] is not None]),
                               'v_max': max([r[f]['hit'][k] for r in fit_rows for k in near6 if r[f]['hit'][k] is not None] or [0])}
                           for f in forms}
    lines.append('fit の 16 場面 × 10/23 No.6 と同じ距離 %s m: %s' % (near6, json.dumps(out['fit_no6_band'])))
    # fit(10/23 No.6 に合わせた模型)の 16 場面: 距離ごとに「今の ON が今の OFF より 0.1 m/s 速い場面の数」と速さの中央
    out['fit_by_d'] = {}
    lines.append('fit の 16 場面の距離ごと(今の ON が OFF より速い場面の数・中央の ON−OFF [m/s]・OFF だけ当たる数・qA / qC が OFF より速い数):')
    for d in ds:
        k = str(d)
        pr = [(r['now_on']['hit'][k], r['now_off']['hit'][k], r['qA_on']['hit'][k], r['qC_on']['hit'][k]) for r in fit_rows]
        dif = sorted(a - b for a, b, _, _ in pr if a is not None and b is not None)
        ent = {'on_faster': sum(1 for a, b, _, _ in pr if a is not None and b is not None and a > b + TH),
               'med_on_minus_off': None if not dif else round(dif[len(dif) // 2], 3),
               'off_only': sum(1 for a, b, _, _ in pr if a is None and b is not None),
               'qA_faster_than_off': sum(1 for _, b, qa, _ in pr if not (qa is None or (b is not None and qa <= b + TH))),
               'qC_faster_than_off': sum(1 for _, b, _, qc in pr if not (qc is None or (b is not None and qc <= b + TH)))}
        out['fit_by_d'][k] = ent
        lines.append('   %.2f %s' % (d, ent))
    (HERE / 'out' / 'ana_r1.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (HERE / 'out' / 'ana_r1.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
