"""打ち消しの票 A: out/sweep_first_rec.json(自分の閉じたループ)を数える。
  1) 10/23 の形・後ろの物なし: ON / OFF の下がる距離(最小・中央・最大)、1.0 m 以上の場面の数、1.0 m に届くまでの秒とそのときの速さ
  2) ふつうの形(前の壁に当たって止まる): 同じ数(ON と OFF が同じか)
  3) 10/23 の形で 1.0〜2.0 m の後ろの物: 当たる組の数・悪いほう(0.1 m/s より速い・片方だけ当たる)の数
  4) 模型ごとの「10/23 の形の ON の下がり − ふつうの形の下がり」
    python -I -B ana_first_rec.py → 標準出力・out/ana_first_rec.json
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TH = 0.1


def st(x):
    x = sorted(v for v in x if v is not None)
    if not x:
        return None
    return {'n': len(x), 'min': round(x[0], 3), 'median': round(x[len(x) // 2], 3), 'max': round(x[-1], 3)}


def main():
    J = json.loads((HERE / 'out' / 'sweep_first_rec.json').read_text())
    rows = J['rows']
    out = {'sha': J['sha'], 'n_esc': len({r['esc'] for r in rows})}
    for form in ('1023', 'wall'):
        R = [r for r in rows if r['form'] == form]
        d = {'scenes': len(R), 'no_rec': sum(1 for r in R for s in ('on', 'off') if r[s]['rec'] is None)}
        for s in ('on', 'off'):
            ok = [r[s] for r in R if r[s]['rec'] is not None]
            d[s] = {'back': st([o['back'] for o in ok]), 'ge1m': sum(o['back'] >= 1.0 for o in ok),
                    't_1m': st([o['t_1m'] for o in ok]), 'v_1m': st([o['v_1m'] for o in ok]),
                    'events': dict(collections.Counter(e for o in ok for e in o['events']))}
            fit = [r[s] for r in R if r['esc'] == 'fit' and r[s]['rec'] is not None]
            d[s]['fit_back'] = st([o['back'] for o in fit])
        d['on_eq_off_back'] = sum(1 for r in R if r['on']['rec'] and r['off']['rec'] and abs(r['on']['back'] - r['off']['back']) < 1e-9)
        out[form] = d
    # 3) 10/23 の形の 1.0〜2.0 m の後ろの物
    R = [r for r in rows if r['form'] == '1023']
    c = collections.Counter()
    fitc = collections.Counter()
    for r in R:
        for k, von in r['on']['hit'].items():
            dd = float(k)
            if dd < 1.0 - 1e-9:
                continue
            voff = r['off']['hit'][k]
            for C_ in ((c,) if r['esc'] != 'fit' else (c, fitc)):
                C_['pairs'] += 1
                C_['on_hits'] += von is not None
                C_['off_hits'] += voff is not None
                if von is None and voff is None:
                    C_['neither'] += 1
                elif von is None:
                    C_['off_only'] += 1
                elif voff is None:
                    C_['on_only'] += 1
                elif von > voff + TH:
                    C_['on_faster'] += 1
                elif voff > von + TH:
                    C_['off_faster'] += 1
                else:
                    C_['same'] += 1
    out['rear_1to2m'] = dict(c)
    out['rear_1to2m_fit'] = dict(fitc)
    # 4) 模型ごと
    per = {}
    for esc in sorted({r['esc'] for r in rows}):
        a = [r['on']['back'] for r in rows if r['esc'] == esc and r['form'] == '1023' and r['on']['rec']]
        b = [r['off']['back'] for r in rows if r['esc'] == esc and r['form'] == '1023' and r['off']['rec']]
        w = [r['on']['back'] for r in rows if r['esc'] == esc and r['form'] == 'wall' and r['on']['rec']]
        per[esc] = {'1023_on': st(a), '1023_off': st(b), 'wall_on(=off)': st(w)}
    out['per_esc'] = per
    print(json.dumps({k: v for k, v in out.items() if k != 'per_esc'}, ensure_ascii=False, indent=1))
    print('== 模型ごとの下がる距離 [m](10/23 の形の ON / OFF・ふつうの形(前の壁に当たって止まる))')
    for esc, v in per.items():
        print('%-16s 1023 ON %s  OFF %s  wall %s' % (esc, v['1023_on'], v['1023_off'], v['wall_on(=off)']))
    (HERE / 'out' / 'ana_first_rec.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
