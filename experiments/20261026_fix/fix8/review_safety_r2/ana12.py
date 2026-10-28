"""exp12(tb・hm・hmh)・exp12c の出力をまとめる。ON が OFF より悪い印のある組を、乱れの無い同じ場面の OFF(base)とも比べて分ける:
  A    ON は乱れの無い OFF と同じかよい(許し: 後ろの物の距離 ±0.1 m・速さ +0.15 m/s・下がる距離 +0.15 m・前の押し +0.3 秒)
       = OFF がその乱れで得をした(中立止めでリカバリーが消えた・人が取った など)。ON は決めた手順のまま
  T    許しを広げる(距離 ±0.2 m・下がる距離 +0.2 m・前の押し +0.5 秒)と乱れの無い OFF と同じ = 中立止めで手順の始めが
       フレーム 1〜数個ずれた(下がる長さはフレームの刻みで 1 フレーム ≒ 0.07 m 変わる・止まりの待ちが中立止めのぶん延びる)
  Ks   知られている形: 段差の形の衝撃(spike)を見張りが後ろの当たり(rear_impact)とみた(fix7/README.md 3 章「IMU の止まらない衝撃
       (段差の形)を後ろの当たりとみる rear_impact は今回の外」)
  Kh   知られている形: 長い中立止めのあとの中止(hold_long。approvals 3「困ること(ON)」)・中立止めの間の衝撃(rear_impact_hold。
       fix7/README.md 3 章)
  D    そのほか(手で見る)
ほか: 全部 OFF と写し f7b の OFF のビット単位、記録の照らし(行の out と書いた値など)、夜の道具の印と本当の形。
    python -I -B ana12.py → out/ana12.json・out/ana12.txt(読むのは out/exp12_*.jsonl・out/exp12c_*.jsonl)
"""
import collections
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6]
REC_PH = ('brake', 'wait', 'gap', 'reverse', 'coast', 'escape', 'halt', 'giveup')


def near_max(prof, d, tol=0.1):
    return max(prof[str(x)] for x in DS if abs(x - d) <= tol + 1e-9)


def worse_vs_base(on, base, tol_d=0.1, tol_back=0.15, tol_push=0.3, only=None):
    """ON が乱れの無い OFF より悪い印。only = ON 対 OFF で悪かった印の種類(rear_faster・front_push_more・zero_push・front_faster)
    だけを見る(同じものさしで比べる。ON 対 OFF で前の押しが少ない組の前の押しは見ない)"""
    w = []
    for d in (DS if only is None or 'rear_faster' in only else ()):
        a = on['prof'][str(d)]
        if a > 0.3 and a > near_max(base['prof'], d, tol_d) + 0.15:
            w.append('rear@%s' % d)
            break
    if (only is None or 'rear_faster' in only) and on['back'] > base['back'] + tol_back:
        w.append('back+%.2f' % (on['back'] - base['back']))
    if (only is None or 'front_push_more' in only) and on['front_push_s'] > base['front_push_s'] + tol_push:
        w.append('front_push+%.2f' % (on['front_push_s'] - base['front_push_s']))
    if (only is None or 'zero_push' in only) and on['zero_push'] > base['zero_push']:
        w.append('zero_push')
    if (only is None or 'front_faster' in only) and on['front_max_v'] > base['front_max_v'] + 0.3:
        w.append('front_v')
    return w


def key_of(sd, kind):
    if kind == 'c':
        return ('c', sd['form'], sd['esc'])
    return (sd['base'], sd['esc'], sd.get('form', 'kai1'))


def load(path):
    """out/*.jsonl(1 組 1 行)→ {'kind', 'bases', 'rows'}。止まったときの書きかけの行・同じ組の 2 行目は読まない"""
    out = {'kind': None, 'bases': [], 'rows': []}
    seen = set()
    for line in Path(path).read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        k = (x.get('t'), json.dumps(x.get('sd'), sort_keys=True))
        if k in seen:
            continue
        seen.add(k)
        out['kind'] = x.get('kind') or out['kind']
        if x.get('t') == 'base':
            out['bases'].append(x)
        elif x.get('t') == 'row':
            out['rows'].append(x['row'])
    return out


def main():
    files = sorted(glob.glob(str(HERE / 'out' / 'exp12_*.jsonl'))) + sorted(glob.glob(str(HERE / 'out' / 'exp12c_*.jsonl')))
    tot = collections.Counter()
    cats = collections.Counter()
    cat_by_kind = collections.defaultdict(collections.Counter)
    flagged = []
    giveup = collections.Counter()
    off_neq = []
    rows_bad = collections.Counter()
    rows_bad_ex = []
    marks = collections.Counter()
    mark_ex = []
    worse_kinds = collections.Counter()
    for f in files:
        d = load(f)
        kind = d['kind']
        base, base_on = {}, {}
        for b in d['bases']:
            base[key_of(b['sd'], kind)] = b['row']['off']
            base_on[key_of(b['sd'], kind)] = b['row']['on']
            tot['bases'] += 1
            if b['row']['worse']:
                tot['bases_worse'] += 1
            if not b['row'].get('off_eq_f7b', True):
                off_neq.append({'file': Path(f).name, 'sd': b['sd'], 'why': b['row'].get('off_f7b_diff')})
            tot['idchecked'] += int('off_eq_f7b' in b['row'])
            tot['pairs'] += 1
        for r in d['rows']:
            tot['pairs'] += 1
            tot['rows_' + kind] += 1
            if 'off_eq_f7b' in r:
                tot['idchecked'] += 1
                if not r['off_eq_f7b']:
                    off_neq.append({'file': Path(f).name, 'sd': r['sd'], 'why': r.get('off_f7b_diff')})
            if r.get('same_on_off'):
                tot['same_on_off'] += 1
            for side in ('on', 'off'):
                if r[side]['rows_bad']:
                    rows_bad[side] += 1
                    if len(rows_bad_ex) < 10:
                        rows_bad_ex.append({'sd': r['sd'], 'side': side, 'bad': r[side]['rows_bad'][:3]})
                for mk in r[side].get('marks') or []:
                    tr = mk['truth']
                    m = mk['mark']
                    if tr in REC_PH and m is None:
                        marks[side + '_missed'] += 1
                        if len(mark_ex) < 12:
                            mark_ex.append({'sd': r['sd'], 'side': side, 'mk': mk})
                    elif tr in REC_PH:
                        marks[side + '_marked_' + str(m)] += 1
                    elif m is not None:
                        marks[side + '_extra_' + str(m)] += 1
                    else:
                        marks[side + '_none_ok'] += 1
            w = [x for x in r['worse'] if x != 'giveup_only_on']
            if 'giveup_only_on' in r['worse']:
                gs = r['on']['guards']
                giveup['hold_long' if 'hold_long' in gs else ('other:' + ','.join(gs[:3]))] += 1
            if not w:
                continue
            tot['worse_' + kind] += 1
            for x in w:
                worse_kinds[x.split('@')[0]] += 1
            b = base.get(key_of(r['sd'], kind))
            only = {x.split('@')[0] for x in w}
            wb = worse_vs_base(r['on'], b, only=only) if b is not None else ['no_base']
            gs = r['on']['guards']
            pk = [p[0] for p in r['sd'].get('pert', [])]
            if not wb:
                cat = 'A'
            elif not worse_vs_base(r['on'], b, 0.2, 0.2, 0.5, only=only):
                cat = 'T'
            elif 'spike' in pk and 'rear_impact' in gs:
                cat = 'Ks'
            elif 'hold_long' in gs or 'rear_impact_hold' in gs:
                cat = 'Kh'
            else:
                cat = 'D'
            cats[cat] += 1
            cat_by_kind[kind][cat] += 1
            if cat == 'D' or cat == 'Kh':
                flagged.append({'cat': cat, 'kind': kind, 'sd': r['sd'], 'worse': w, 'vs_base': wb, 'guards_on': gs[:8],
                                'guards_off': r['off']['guards'][:8],
                                'on': {k: r['on'][k] for k in ('back', 'vback', 'front_push_s', 'zero_push', 'recoveries', 'giveup_t')},
                                'off': {k: r['off'][k] for k in ('back', 'vback', 'front_push_s', 'zero_push', 'recoveries', 'giveup_t')},
                                'base_off': {k: b[k] for k in ('back', 'vback', 'front_push_s', 'zero_push')} if b else None,
                                'base_on_eq_off': base_on.get(key_of(r['sd'], kind), {}).get('prof') == (b or {}).get('prof'),
                                'prof_on': r['on']['prof'], 'prof_off': r['off']['prof'],
                                'prof_base': b['prof'] if b else None})
    # 全部 OFF と前の版(idold12): 版ごとに同じだった組の数
    idold = collections.Counter()
    idold_n = 0
    idold_diff = []
    for f in sorted(glob.glob(str(HERE / 'out' / 'idold12_*.jsonl'))):
        for line in Path(f).read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get('t') != 'row':
                continue
            idold_n += 1
            for v, c in x['row']['cmp'].items():
                idold[v] += int(bool(c['same']))
                if not c['same']:
                    idold_diff.append({'sd': x['sd'], 'ver': v, 'why': c['why']})
    res = {'files': [Path(f).name for f in files], 'tot': dict(tot), 'idold_rows': idold_n, 'idold_same': dict(idold),
           'idold_diff': idold_diff[:20], 'cats': dict(cats),
           'cats_by_kind': {k: dict(v) for k, v in cat_by_kind.items()}, 'worse_kinds': dict(worse_kinds),
           'giveup_only_on': dict(giveup), 'off_neq_f7b': off_neq[:50], 'off_neq_f7b_n': len(off_neq),
           'rows_bad': dict(rows_bad), 'rows_bad_ex': rows_bad_ex, 'marks': dict(marks), 'mark_ex': mark_ex,
           'flagged': flagged}
    (HERE / 'out' / 'ana12.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    lines = ['files %s' % res['files'], 'tot %s' % res['tot'], 'cats %s' % res['cats'],
             'cats_by_kind %s' % res['cats_by_kind'], 'worse_kinds %s' % res['worse_kinds'],
             'giveup_only_on %s' % res['giveup_only_on'], 'off_neq_f7b %d' % len(off_neq), 'rows_bad %s' % res['rows_bad'],
             'marks %s' % res['marks'], 'idold rows %d same %s' % (idold_n, dict(idold))]
    dcount = collections.Counter()
    for x in flagged:
        if x['cat'] == 'D':
            dcount[(x['kind'], x['sd'].get('ev'), tuple(p[0] for p in x['sd'].get('pert', [])), tuple(x['vs_base'][:1]))] += 1
    lines.append('D groups:')
    for k, v in dcount.most_common(60):
        lines.append('  %s %s' % (v, k))
    (HERE / 'out' / 'ana12.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
