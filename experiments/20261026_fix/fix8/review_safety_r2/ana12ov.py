"""exp12 ov(後退の終わりのカメラの止まり)と tb の coast の組だけのまとめ: ON・OFF・乱れの無い OFF(base)の下がった距離・
後ろへの最高速・後ろの物の距離ごとの当たりの速さ。止まりの始め(coast の k フレーム前後)× 長さ n ごと。
    python -I -B ana12ov.py → out/ana12ov.json・out/ana12ov.txt
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
import ana12  # noqa: E402

DS_SHOW = (0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.6)


def main():
    d = ana12.load(HERE / 'out' / 'exp12_ov_0of1.jsonl')
    base = {}
    for b in d['bases']:
        base[(b['sd']['base'], b['sd']['esc'])] = b
    cells = collections.defaultdict(list)
    rows_out = []
    for r in d['rows']:
        sd = r['sd']
        b = base[(sd['base'], sd['esc'])]['row']
        bo = b['off']
        n = sd['pert'][0][2]
        on, off = r['on'], r['off']
        x = {'esc': sd['esc'], 'base': sd['base'], 'ev': sd['ev'], 'n': n,
             'back_on': on['back'], 'back_off': off['back'], 'back_base': bo['back'],
             'vback_on': on['vback'], 'vback_off': off['vback'], 'vback_base': bo['vback'],
             'hits': {str(dd): (on['prof'][str(dd)], off['prof'][str(dd)], bo['prof'][str(dd)]) for dd in DS_SHOW},
             'guards_on': on['guards'][:4], 'front_push': (on['front_push_s'], off['front_push_s'], bo['front_push_s'])}
        # ON が OFF・base のどちらより悪いか(同じものさし: 後ろの物の距離ごとの当たりの速さ +0.15 m/s・下がる距離 +0.15 m)
        rear_worse = [dd for dd in ana12.DS if on['prof'][str(dd)] > max(off['prof'][str(dd)], bo['prof'][str(dd)]) + 0.15
                      and on['prof'][str(dd)] > 0.3]
        x['on_worse_both'] = bool(rear_worse) or on['back'] > max(off['back'], bo['back']) + 0.15
        x['rear_worse_d'] = rear_worse
        x['on_only_hit'] = [dd for dd in ana12.DS if on['prof'][str(dd)] > 0.3 and off['prof'][str(dd)] == 0.0
                            and bo['prof'][str(dd)] == 0.0]
        rows_out.append(x)
        cells[(sd['ev'], n)].append(x)
    lines = []
    tot = collections.Counter()
    for x in rows_out:
        tot['rows'] += 1
        tot['on_worse_both'] += int(x['on_worse_both'])
        tot['on_only_hit_rows'] += int(bool(x['on_only_hit']))
    lines.append('ov rows %d・ON が OFF と base のどちらより悪い %d・ON だけ当たる距離がある %d' % (
        tot['rows'], tot['on_worse_both'], tot['on_only_hit_rows']))
    lines.append('止まりの始め・長さ: 組・ON が両方より悪い・下がった距離の差 ON−base(中央 / 最大)・OFF−base(中央 / 最大)')
    for (ev, n), xs in sorted(cells.items(), key=lambda kv: (int(kv[0][0][5:]), kv[0][1])):
        don = sorted(round(x['back_on'] - x['back_base'], 3) for x in xs)
        doff = sorted(round(x['back_off'] - x['back_base'], 3) for x in xs)
        lines.append('  %s n=%d: %d 組・悪い %d・ON−base %+.3f / %+.3f m・OFF−base %+.3f / %+.3f m' % (
            ev, n, len(xs), sum(1 for x in xs if x['on_worse_both']), don[len(don) // 2], don[-1], doff[len(doff) // 2], doff[-1]))
    # ESC ごと(coast+0)
    lines.append('coast+0(止まりが coast に移るフレームから)の ESC ごと: ON−base の下がった距離(n=3〜10 の最大)・後ろへの最高速 ON / base')
    by = collections.defaultdict(list)
    for x in rows_out:
        if x['ev'] == 'coast+0':
            by[(x['esc'], x['base'])].append(x)
    for k, xs in sorted(by.items()):
        m = max(xs, key=lambda x: x['back_on'] - x['back_base'])
        lines.append('  %s %s: +%.3f m(n=%d。ON %.3f・OFF %.3f・base %.3f m)・最高速 ON %.2f / OFF %.2f / base %.2f m/s・'
                     '1.0/1.2/1.4 m の当たり ON %s OFF %s base %s' % (
                         k[0], k[1], m['back_on'] - m['back_base'], m['n'], m['back_on'], m['back_off'], m['back_base'],
                         m['vback_on'], m['vback_off'], m['vback_base'],
                         [m['hits'][s][0] for s in ('1.0', '1.2', '1.4')], [m['hits'][s][1] for s in ('1.0', '1.2', '1.4')],
                         [m['hits'][s][2] for s in ('1.0', '1.2', '1.4')]))
    (HERE / 'out' / 'ana12ov.json').write_text(json.dumps({'tot': dict(tot), 'rows': rows_out}, ensure_ascii=False, indent=1))
    (HERE / 'out' / 'ana12ov.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
