"""票 A: sweep_rA.py の出力を、モデルごと・止まりの置き方ごとに分けて、ON は ON の止まりなし、OFF は OFF の止まりなしと比べる。
    python -I -B ana2_rA.py PART [PART ...] → 標準出力(と out/ana2_<PART...>.txt)
  ON悪 = ON の止まりの走りが、ON の止まりなし・OFF の止まりの走りの両方より 0.05 m を超えて多く下がった か 後ろ 0.8〜1.8 m の
         どれかの物に 0.1 m/s を超えて速く当たった(指摘の「ON だけ・ON のほうが強く」)
  OFF悪 = 入れかえ(OFF の止まりの走りが、OFF の止まりなし・ON の止まりの走りの両方より)
"""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


def load(parts):
    rows = []
    for part in parts:
        for f in sorted(glob.glob(str(HERE / 'out' / ('%s_*of*.jsonl' % part)))):
            for ln in Path(f).read_text().splitlines():
                d = json.loads(ln)
                d['part'] = part
                rows.append(d)
    return rows


def bad(a, ref_self, other):
    dm = a['back'] - max(ref_self['back'], other['back'])
    hm = max(a['hits'][i] - max(ref_self['hits'][i], other['hits'][i]) for i in range(len(DS)))
    return dm > 0.05 or hm > 0.1


def main():
    parts = sys.argv[1:]
    data = load(parts)
    out = []

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        out.append(s)

    P('parts', parts, 'configs', len(data), ' 止まりなしで ON == OFF:', sum(d['base_same'] for d in data))
    grp = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in data:
        if 'rows' not in d:
            continue
        mdl = d['cf']['model']
        bo, bf = d['base'], d['base_off']
        for r in d['rows']:
            on, off = r['on'], r['off']
            if on is None or off is None:
                continue
            k = r['k'] if d['part'] != 's2' else max(min(r['k'], 6), -16)
            key = (mdl, k, r['L'])
            G = grp[key]
            G['n'].append(1)
            G['don'].append(on['back'] - bo['back'])
            G['doff'].append(off['back'] - bf['back'])
            G['on_off'].append(on['back'] - off['back'])
            G['onbad'].append(bad(on, bo, off))
            G['offbad'].append(bad(off, bf, on))
            G['ongu'].append(on['giveup'])
            G['eq'].append(on['back'] == off['back'] and on['hits'] == off['hits'])
            G['h14'].append((on['hits'][3], off['hits'][3], bo['hits'][3]))
    P('モデル k L | n | ON悪 OFF悪 ON=OFF ON中止 | ON−ONなし 中央[最小,最大] | OFF−OFFなし 中央[最小,最大] | ON−OFF 中央[最小,最大] | 1.4m の当たり ON>0.1 / OFF>0.1 / なし>0.1')
    for key in sorted(grp, key=lambda x: (x[0], x[1], x[2])):
        G = grp[key]
        a, o, oo = np.array(G['don']), np.array(G['doff']), np.array(G['on_off'])
        h = np.array(G['h14'])
        P('%-7s k%+3d L%2d | %4d | %4d %4d %4d %4d | %+.3f[%+.3f,%+.3f] | %+.3f[%+.3f,%+.3f] | %+.3f[%+.3f,%+.3f] | %d/%d/%d' % (
            key[0], key[1], key[2], len(G['n']), sum(G['onbad']), sum(G['offbad']), sum(G['eq']), sum(G['ongu']),
            np.median(a), a.min(), a.max(), np.median(o), o.min(), o.max(), np.median(oo), oo.min(), oo.max(),
            int((h[:, 0] > 0.1).sum()), int((h[:, 1] > 0.1).sum()), int((h[:, 2] > 0.1).sum())))
    tag = '_'.join(parts)
    (HERE / 'out' / ('ana2_%s.txt' % tag)).write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
