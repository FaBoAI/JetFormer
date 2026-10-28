"""fix8 係 safety 直す 8-1 回目: exp_q12 の出力を数える。場面の組ごとに、案(qA・qC)の両方 ON が今の両方 ON とビット単位で同じ
場面の数、違う場面の中で: 案が今の OFF より悪い印(exp11.worse と同じ印: 後ろの当たりが多い・0.1 m/s より速い・前の壁押し +0.3 秒・
0 m の壁押し・前の当たり +0.3 m/s・後ろの物を押す +0.1 秒・中止が案だけ)、案が今の ON より悪い印、今の ON が案より悪い印。
最初のリカバリーの下がる距離(案 − 今の ON)も。
    python3 -I -B ana_q12.py → out/ana_q12.txt・out/ana_q12.json
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPS = ('qA', 'qC')


def flags(ws):
    return collections.Counter(w.split(':')[0] for w in ws)


def main():
    by_kind = collections.defaultdict(list)
    for f in sorted(HERE.glob('out/exp_q12_*of*.json')):
        J = json.loads(f.read_text())
        by_kind[J['kind']] += J['rows']
    out, lines = {}, []
    tot = collections.Counter()
    for kind, rows in sorted(by_kind.items()):
        ent = {'n': len(rows), 'now_on_worse_than_off_scenes': sum(1 for r in rows if r['now_on_worse_than_off'])}
        for q in PROPS:
            diff = [r for r in rows if not r['%s_same_as_now_on' % q]]
            w_off = [r for r in diff if r['%s_worse_than_off' % q]]
            w_on = [r for r in diff if r['%s_worse_than_now_on' % q]]
            b_on = [r for r in diff if r['now_on_worse_than_%s' % q]]
            fo = collections.Counter()
            for r in w_off:
                fo.update(flags(r['%s_worse_than_off' % q]))
            fn = collections.Counter()
            for r in w_on:
                fn.update(flags(r['%s_worse_than_now_on' % q]))
            fb = collections.Counter()
            for r in b_on:
                fb.update(flags(r['now_on_worse_than_%s' % q]))
            dback = [(r['%s_on' % q]['seg0']['back'] or 0) - (r['now_on']['seg0']['back'] or 0) for r in diff
                     if r['%s_on' % q]['seg0']['back'] is not None and r['now_on']['seg0']['back'] is not None]
            # 今の ON が OFF より悪かった場面のうち、案では OFF より悪くなくなった場面
            fixed = [r for r in rows if r['now_on_worse_than_off'] and not r['%s_same_as_now_on' % q] and not r['%s_worse_than_off' % q]]
            # 案が OFF より悪いのに、今の ON は(その印で)OFF より悪くない場面 = 案で新しく出た「OFF より悪い」
            new_w = [r for r in w_off if set(flags(r['%s_worse_than_off' % q])) - set(flags(r['now_on_worse_than_off']))]
            nf = collections.Counter()
            for r in new_w:
                nf.update(set(flags(r['%s_worse_than_off' % q])) - set(flags(r['now_on_worse_than_off'])))
            ent.setdefault('new_worse_than_off', {})[q] = {'n': len(new_w), 'flags': dict(nf)}
            ent[q] = {'same_as_now_on': len(rows) - len(diff), 'differs': len(diff),
                      'differs_with_back_early_now': sum(1 for r in diff if 'back_early' in r['now_on']['guards']),
                      'worse_than_off': len(w_off), 'worse_than_off_flags': dict(fo),
                      'worse_than_now_on': len(w_on), 'worse_than_now_on_flags': dict(fn),
                      'now_on_worse': len(b_on), 'now_on_worse_flags': dict(fb),
                      'fixed_vs_off': len(fixed),
                      'dback_min': round(min(dback), 3) if dback else None, 'dback_max': round(max(dback), 3) if dback else None,
                      'dback_med': round(sorted(dback)[len(dback) // 2], 3) if dback else None,
                      'examples_worse_than_off': [{'sd': {k: r['sd'].get(k) for k in ('form', 'xf', 'model', 'esc', 'rear', 'veh_dt',
                                                                                     'coast', 'lag', 'i')},
                                                   'flags': r['%s_worse_than_off' % q],
                                                   'q': {k: r['%s_on' % q][k] for k in ('rear_max_v', 'front_push_s', 'zero_push')},
                                                   'off': {k: r['now_off'][k] for k in ('rear_max_v', 'front_push_s', 'zero_push')},
                                                   'now_on': {k: r['now_on'][k] for k in ('rear_max_v', 'front_push_s', 'zero_push')}}
                                                  for r in w_off[:6]]}
            ent[q]['new_worse_than_off'] = ent['new_worse_than_off'][q]
            for k in ('differs', 'worse_than_off', 'worse_than_now_on', 'now_on_worse', 'fixed_vs_off'):
                tot['%s_%s' % (q, k)] += ent[q][k]
            tot['%s_new_worse_than_off' % q] += ent[q]['new_worse_than_off']['n']
        tot['n'] += len(rows)
        tot['now_on_worse_than_off_scenes'] += ent['now_on_worse_than_off_scenes']
        out[kind] = ent
        lines.append('== %s %d 場面(今の ON が OFF より悪い印 %d)' % (kind, len(rows), ent['now_on_worse_than_off_scenes']))
        for q in PROPS:
            e = ent[q]
            lines.append('   %s: 今の ON と同じ %d・違う %d(今の ON に back_early %d)| OFF より悪い %d %s | 今の ON より悪い %d %s | '
                         '今の ON のほうが悪い %d %s | 今の ON の「OFF より悪い」が消えた %d | 下がる距離の差 %s〜%s(中央 %s)m' % (
                             q, e['same_as_now_on'], e['differs'], e['differs_with_back_early_now'], e['worse_than_off'],
                             e['worse_than_off_flags'], e['worse_than_now_on'], e['worse_than_now_on_flags'], e['now_on_worse'],
                             e['now_on_worse_flags'], e['fixed_vs_off'], e['dback_min'], e['dback_max'], e['dback_med']))
            lines.append('      案で新しく出た「OFF より悪い」(今の ON にはその印が無い): %d %s' % (
                e['new_worse_than_off']['n'], e['new_worse_than_off']['flags']))
            for x in e['examples_worse_than_off'][:3]:
                lines.append('      例 %s' % json.dumps(x, ensure_ascii=False))
    out['total'] = dict(tot)
    lines.append('合計 %s' % dict(tot))
    (HERE / 'out' / 'ana_q12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (HERE / 'out' / 'ana_q12.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
