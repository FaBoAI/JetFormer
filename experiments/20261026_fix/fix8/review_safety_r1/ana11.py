"""exp11 の出力をまとめる: 組の数・OFF が写し f7b の OFF とビット単位で同じ数・ON と OFF がビット単位で同じ数・
「ON が OFF より悪い」印の数(種類ごと・くみごと)と例。
    python -I -B ana11.py <kind> [出力の txt]
"""
import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(kind):
    rows = []
    for f in sorted(glob.glob(str(HERE / 'out' / ('exp11_%s_*.json' % kind)))):
        d = json.loads(Path(f).read_text())
        rows += d['rows']
    return rows


def main(kind, out=None):
    rows = load(kind)
    lines = []
    P = lines.append
    n = len(rows)
    P('kind=%s 組 %d' % (kind, n))
    P('OFF が f7b の OFF とビット単位で同じ: %d / %d' % (sum(1 for r in rows if r.get('off_eq_f7b')), n))
    P('ON と OFF がビット単位で同じ: %d / %d' % (sum(1 for r in rows if r.get('same_on_off')), n))
    c = collections.Counter()
    for r in rows:
        for w in r['worse']:
            c[w.split(':')[0] if not w.startswith('giveup') else w] += 1
    P('ON が悪い印(のべ): %s' % dict(c))
    P('ON が悪い印のある組: %d' % sum(1 for r in rows if [w for w in r['worse'] if not w.startswith('giveup')]))
    # 逆向き(OFF が悪い)も数える(参考)
    rev = collections.Counter()
    for r in rows:
        a, b = r['on'], r['off']
        if b['rear_n'] > a['rear_n']:
            rev['off_rear_more'] += 1
        if b['rear_max_v'] > a['rear_max_v'] + 0.1:
            rev['off_rear_faster'] += 1
        if b['front_push_s'] > a['front_push_s'] + 0.3:
            rev['off_front_push_more'] += 1
        if b['zero_push'] > a['zero_push']:
            rev['off_zero_push'] += 1
        if b['front_max_v'] > a['front_max_v'] + 0.3:
            rev['off_front_faster'] += 1
        if b['rear_push_s'] > a['rear_push_s'] + 0.1:
            rev['off_rear_push_more'] += 1
        if b['giveup_t'] is not None and a['giveup_t'] is None:
            rev['giveup_only_off'] += 1
    P('(参考)OFF が悪い印: %s' % dict(rev))
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        sd = r['sd']
        key = (sd['form'], sd['model']['kind'], sd['esc'])
        for w in r['worse']:
            if not w.startswith('giveup'):
                by[key][w] += 1
    P('くみごと(ON が悪い印のある くみ):')
    for key, cc in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:40]:
        P('  %s %s' % (key, dict(cc)))
    P('例(ON が悪い印。最大 60):')
    k = 0
    for r in rows:
        ws = [w for w in r['worse'] if not w.startswith('giveup')]
        if not ws:
            continue
        sd = r['sd']
        a, b = r['on'], r['off']
        P('  %s | on rear %s fp %.2f zp %d gu %s | off rear %s fp %.2f zp %d gu %s | %s' % (
            ws, [(h[0], h[2]) for h in a['rear']][:3], a['front_push_s'], a['zero_push'], a['giveup_t'],
            [(h[0], h[2]) for h in b['rear']][:3], b['front_push_s'], b['zero_push'], b['giveup_t'],
            json.dumps({x: sd[x] for x in sd if x not in ('kind',)}, ensure_ascii=False)[:600]))
        k += 1
        if k >= 60:
            break
    txt = '\n'.join(lines)
    print(txt)
    if out:
        Path(out).write_text(txt + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
