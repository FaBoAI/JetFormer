"""票 A: sweep_rA.py の出力をまとめる。
    python -I -B ana_rA.py PART → 標準出力(と out/ana_PART.txt)
「ON が悪い」= ON の下がった距離が 止まりなし・OFF の大きい方より +0.05 m を超える、か、後ろ 0.8〜1.8 m の物のどれかで
ON の当たりの速さが 止まりなし・OFF の速い方より +0.1 m/s を超える。「OFF が悪い」も同じ(入れかえ)。
"""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


def load(part):
    rows = []
    for f in sorted(glob.glob(str(HERE / 'out' / ('%s_*of*.jsonl' % part)))):
        for ln in Path(f).read_text().splitlines():
            rows.append(json.loads(ln))
    return rows


def worse(a, b, c):
    """a が b・c の悪い方より悪いか。(距離で, 当たりで, 距離の差 a − max(b,c))"""
    dm = a['back'] - max(b['back'], c['back'])
    hm = max(a['hits'][i] - max(b['hits'][i], c['hits'][i]) for i in range(len(DS)))
    return dm > 0.05, hm > 0.1, dm, hm


def main():
    part = sys.argv[1]
    data = load(part)
    out = []

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        out.append(s)

    P('part', part, 'configs', len(data))
    P('止まりなしで ON == OFF(書いた値・AI_CMD の out/assist): %d/%d' % (sum(1 for d in data if d['base_same']), len(data)))
    notes = [d for d in data if 'note' in d]
    P('リカバリー・後退の終わりの無い設定:', len(notes), [d['cf'] for d in notes][:5])
    g = collections.defaultdict(lambda: collections.Counter())
    dd = collections.defaultdict(lambda: {'on': [], 'off': [], 'hm_on': []})
    by_esc = collections.defaultdict(collections.Counter)
    by_cf = collections.defaultdict(collections.Counter)
    ex = []
    other = collections.Counter()
    for d in data:
        if 'rows' not in d:
            continue
        b = d['base']
        for r in d['rows']:
            on, off = r['on'], r['off']
            key = (r['k'], r['L'])
            if on is None or off is None:
                g[key]['none'] += 1
                continue
            g[key]['n'] += 1
            wd, wh, dm, hm = worse(on, b, off)
            owd, owh, odm, ohm = worse(off, b, on)
            dd[key]['on'].append(on['back'] - b['back'])
            dd[key]['off'].append(off['back'] - b['back'])
            dd[key]['hm_on'].append(hm)
            if wd or wh:
                g[key]['on_worse'] += 1
                by_esc[d['cf']['esc']]['on_worse'] += 1
                by_cf[(d['cf']['model'], d['cf']['phase'], d['cf']['proc'], d['cf']['xf'])]['on_worse'] += 1
                if len(ex) < 6 and r['L'] in (3, 4) and r['k'] == 0:
                    ex.append((d['cf'], r['k'], r['L'], b['back'], on['back'], off['back'], b['hits'], on['hits'], off['hits']))
            if owd or owh:
                g[key]['off_worse'] += 1
            if on['giveup']:
                g[key]['on_giveup'] += 1
            if off['giveup']:
                g[key]['off_giveup'] += 1
            if on['fpush'] > b['fpush'] + 0.3:
                other['on_fpush+0.3'] += 1
            if off['fpush'] > b['fpush'] + 0.3:
                other['off_fpush+0.3'] += 1
            by_esc[d['cf']['esc']]['n'] += 1
            by_cf[(d['cf']['model'], d['cf']['phase'], d['cf']['proc'], d['cf']['xf'])]['n'] += 1
    P('')
    P('k(後退の終わりのフレームから) L(フレーム)  n  ON が悪い  OFF が悪い  ON 中止  OFF 中止 | ON−なし 中央/最小/最大 m | OFF−なし 中央/最小/最大 m')
    for key in sorted(g):
        c = g[key]
        a, o = np.array(dd[key]['on']), np.array(dd[key]['off'])
        if not len(a):
            P(key, dict(c))
            continue
        P('k %+d L %2d  n %4d  ON悪 %4d  OFF悪 %4d  ON中止 %3d  OFF中止 %3d | %+.3f %+.3f %+.3f | %+.3f %+.3f %+.3f' % (
            key[0], key[1], c['n'], c['on_worse'], c['off_worse'], c['on_giveup'], c['off_giveup'],
            np.median(a), a.min(), a.max(), np.median(o), o.min(), o.max()))
    P('')
    P('ESC ごとの ON が悪い / 組:', {k: '%d/%d' % (v['on_worse'], v['n']) for k, v in sorted(by_esc.items())})
    P('(モデル, 位相, 処理, 壁)ごと:', {str(k): '%d/%d' % (v['on_worse'], v['n']) for k, v in sorted(by_cf.items())})
    P('前の壁を止まりなしより 0.3 秒より長く押した組:', dict(other))
    P('')
    P('例(k 0・L 3/4。止まりなし / ON / OFF の下がった距離と、後ろ', DS, 'm の物に当たる速さ):')
    for e in ex:
        P('  ', e[0], 'k', e[1], 'L', e[2], 'back', e[3], e[4], e[5])
        P('       hits none', e[6], 'ON', e[7], 'OFF', e[8])
    (HERE / 'out' / ('ana_%s.txt' % part)).write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
