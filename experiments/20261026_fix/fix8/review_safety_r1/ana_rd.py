"""exp11 rd(10/23 の形 × 後ろの物の距離)の表: くみ(中立の惰性・ESC・モデルの前進)ごとに、止まった所から後ろの物までの
距離 d [m] と、最初のリカバリーの後ろの当たりの速さ(ON / OFF。当たらなければ -)・下がった距離・後退の最高の速さ。
    python -I -B ana_rd.py [出力の txt] [rd|rd2]
"""
import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def first_rear(m, t0, t1):
    hs = [h for h in m['rear'] if t0 <= h[0] <= (t1 if t1 is not None else 99)]
    return -hs[0][2] if hs else None


def main(out=None, kind='rd'):
    rows = []
    for f in sorted(glob.glob(str(HERE / 'out' / ('exp11_%s_*.json' % kind)))):
        rows += json.loads(Path(f).read_text())['rows']
    L = []
    P = L.append
    groups = collections.defaultdict(list)
    for r in rows:
        sd = r['sd']
        groups[(sd.get('coast', 1.5), sd['esc'], sd['model']['m'], sd.get('lag', 0.0))].append(r)
    tot = collections.Counter()
    worse_list = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        rs = sorted(groups[key], key=lambda r: (r['sd']['rear'] is None, r['sd']['rear'] or 0))
        P('== 中立の惰性 %.1f m/s²・ESC %s・前進 %d µs・指令の遅れ %.2f 秒' % (key[0], key[1], key[2], key[3]))
        free = next((r for r in rs if r['sd']['rear'] is None), None)
        if free:
            so, sf = free['on']['segs'][:1], free['off']['segs'][:1]
            P('  後ろの物なし: ON 下がる %s m・最高 %s m/s | OFF 下がる %s m・最高 %s m/s' % (
                so[0].get('back') if so else None, so[0].get('vmin') if so else None,
                sf[0].get('back') if sf else None, sf[0].get('vmin') if sf else None))
        for r in rs:
            sd = r['sd']
            if sd['rear'] is None:
                continue
            a, b = r['on'], r['off']
            sa, sb = (a['segs'] or [{}])[0], (b['segs'] or [{}])[0]
            x0 = sa.get('x0')
            d = None if x0 is None else round(x0 - (sd['xf'] - sd['rear']), 3)
            ha = first_rear(a, sa.get('t0', 0), sa.get('t1')) if sa else None
            hb = first_rear(b, sb.get('t0', 0), sb.get('t1')) if sb else None
            tag = ''
            if ha is not None and hb is None:
                tag = 'ON だけ当たる'
                tot['on_only'] += 1
            elif hb is not None and ha is None:
                tag = 'OFF だけ当たる'
                tot['off_only'] += 1
            elif ha is not None and hb is not None:
                tot['both'] += 1
                if ha > hb + 0.1:
                    tag = 'ON が速い +%.2f' % (ha - hb)
                    tot['both_on_faster'] += 1
                elif hb > ha + 0.1:
                    tag = 'OFF が速い +%.2f' % (hb - ha)
                    tot['both_off_faster'] += 1
                else:
                    tot['both_same'] += 1
            else:
                tot['none'] += 1
            if tag.startswith('ON'):
                worse_list.append((key, sd['rear'], d, ha, hb))
            P('  後ろの物 壁から %.1f m(止まった所から %s m): ON %s | OFF %s  %s' % (
                sd['rear'], d, '-' if ha is None else '%.2f m/s' % ha, '-' if hb is None else '%.2f m/s' % hb, tag))
    P('')
    P('合計(最初のリカバリーの後ろの当たり): %s' % dict(tot))
    P('ON が悪い(ON だけ当たる・両方当たって ON が 0.1 m/s より速い): %d' % len(worse_list))
    by = collections.Counter((k[0], k[1], k[3]) for k, *_ in worse_list)
    P('  中立の惰性・ESC ごと: %s' % dict(by))
    txt = '\n'.join(L)
    print(txt)
    if out:
        Path(out).write_text(txt + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
