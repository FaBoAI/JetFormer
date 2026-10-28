"""fix8 係 safety-proposal: out/exp10_*.json を表にする(案ごと: 今の ON と違う組・直した組・悪くした組と、その中身)。
失敗 = 最初のリカバリーで 0.1 m 未満しか下がらず escape が前の壁を 0.05 秒より押した(exp9 と同じ)。
「悪くなった」の見方(案の ON を今の ON と同じ組で比べる。直した組 = 今が失敗で案が失敗でない、は別に数える):
  後ろの当たり: 当たりの数が増えた・当たりの最大の速さが 0.05 m/s より速い / 下がる距離: 0.05 m より長く下がった /
  最大の後ろへの速さ: 0.05 m/s より速い / 止まるまで: 下がり終えるまでの秒・escape までの秒が 0.05 秒より長い / 後ろの物を押した秒が増えた
    python -I analyze10.py [くみ ...]   → 標準出力と out/analyze10.txt
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPS = ('pA', 'pB', 'pC')


def fail(m):
    b = m['back'] if m['back'] is not None else 0.0
    return b < 0.1 and (m['esc_push'] or 0) > 0.05


def rear_n(m):
    return len(m.get('rear_hits') or [])


def worse(a, b):
    """b(案)が a(今)より悪い所の印の並び"""
    w = []
    if rear_n(b) > rear_n(a):
        w.append('rear_more')
    if (b.get('rear_max_v') or 0) > (a.get('rear_max_v') or 0) + 0.05:
        w.append('rear_faster')
    if (b['back'] or 0) > (a['back'] or 0) + 0.05:
        w.append('back_longer')
    if (b.get('max_back_v') or 0) > (a.get('max_back_v') or 0) + 0.05:
        w.append('vmax_up')
    if a.get('t_back_end') is not None and b.get('t_back_end') is not None and b['t_back_end'] > a['t_back_end'] + 0.05:
        w.append('t_back_end_up')
    if a.get('t_to_escape') is not None and b.get('t_to_escape') is not None and b['t_to_escape'] > a['t_to_escape'] + 0.05:
        w.append('t_escape_up')
    if (b.get('rear_push') or 0) > (a.get('rear_push') or 0) + 0.02:
        w.append('rear_push_up')
    if fail(b) and not fail(a):
        w.append('NEW_FAIL')
    return w


def one(kind):
    fp = HERE / 'out' / ('exp10_%s.json' % kind)
    if not fp.exists():
        return None
    d = json.loads(fp.read_text(encoding='utf-8'))
    vs = d['variants']
    cells = d['cells']
    L = ['## %s(%d 組・%.0f 秒)形 %s' % (kind, len(cells), d.get('elapsed_s', 0), ','.join(vs))]
    fc = {v: sum(1 for r in cells.values() if fail(r[v])) for v in vs}
    rh = {v: sum(1 for r in cells.values() if rear_n(r[v])) for v in vs}
    rv = {v: max([r[v].get('rear_max_v') or 0 for r in cells.values()] + [0]) for v in vs}
    L.append('  失敗(0 m・壁押し): ' + '  '.join('%s %d' % (v, fc[v]) for v in vs))
    L.append('  後ろに当たった組: ' + '  '.join('%s %d' % (v, rh[v]) for v in vs) + '   最大 ' +
             '  '.join('%s %.2f' % (v, rv[v]) for v in vs))
    summ = {}
    # 跳ね・途切れなしの同じ場面(clean)がある くみ: 各形が clean からどれだけずれたか(理想 = ずれない)
    cl = {k[:-len('clean')]: r for k, r in cells.items() if k.endswith('clean')}
    if cl:
        def pref(k):
            for c in cl:
                if k.startswith(c):
                    return c
            return None
        L.append('  跳ね・途切れなしの同じ場面(clean %d)からのずれ(決めが変わった = やり直し・early_none の数が clean と違う。'
                 '距離は |案 − clean| の最大と 0.15 m を超えた組の数):' % len(cl))
        for v in vs:
            dec, dmax, d15, rmore, vfast, fails = 0, 0.0, 0, 0, 0, 0
            for k, r in cells.items():
                c = pref(k)
                if c is None or k.endswith('clean'):
                    continue
                a, b = cl[c][v], r[v]
                d = abs((b['back'] or 0) - (a['back'] or 0))
                dec += b.get('n_retry') != a.get('n_retry')
                dmax = max(dmax, d)
                d15 += d > 0.15
                rmore += rear_n(b) > rear_n(a)
                vfast += (b.get('rear_max_v') or 0) > (a.get('rear_max_v') or 0) + 0.05
                fails += fail(b) and not fail(a)
            L.append('    %-8s 決めが変わった %3d 組・距離の差 最大 %.3f m(0.15 m 超 %d 組)・後ろの当たりが増えた %d・当たりが速い %d・'
                     '失敗 %d' % (v, dec, dmax, d15, rmore, vfast, fails))
        summ['_clean_dev'] = True
    for p in PROPS:
        if p not in vs or 'on' not in vs:
            continue
        diff = [k for k, r in cells.items() if not r.get('same_' + p, True)]
        fixed = [k for k in diff if fail(cells[k]['on']) and not fail(cells[k][p])]
        other = [k for k in diff if k not in fixed]
        wc = collections.Counter()
        ex = collections.defaultdict(list)
        dback, dv, dte, dtb = [], [], [], []
        for k in diff:
            a, b = cells[k]['on'], cells[k][p]
            for w in worse(a, b):
                if k in fixed and w in ('back_longer', 't_back_end_up', 't_escape_up', 'vmax_up'):
                    continue            # 直した組は下がるようになったので長い・遅いのは当たり前(後ろの当たりは数える)
                wc[w] += 1
                if len(ex[w]) < 3:
                    ex[w].append(k)
            if k in fixed:
                continue
            dback.append(round((b['back'] or 0) - (a['back'] or 0), 3))
            dv.append(round((b.get('max_back_v') or 0) - (a.get('max_back_v') or 0), 3))
            if a.get('t_to_escape') is not None and b.get('t_to_escape') is not None:
                dte.append(round(b['t_to_escape'] - a['t_to_escape'], 3))
            if a.get('t_back_end') is not None and b.get('t_back_end') is not None:
                dtb.append(round(b['t_back_end'] - a['t_back_end'], 3))
        # 直した組の後ろの当たりは、跳ねなしの同じ場面(clean)と比べる
        rng = lambda xs: ('%+.3f〜%+.3f' % (min(xs), max(xs))) if xs else '—'
        L.append('  %s: 今の ON と違う %d 組 = 直した %d・ほか %d。悪くなった印 %s' % (
            p, len(diff), len(fixed), len(other), dict(wc) if wc else 'なし'))
        if other:
            L.append('      ほかの組の差(案 − 今): 下がる距離 %s m・最大の後ろへの速さ %s m/s・escape まで %s 秒・下がり終えるまで %s 秒'
                     % (rng(dback), rng(dv), rng(dte), rng(dtb)))
        for w, ks in ex.items():
            L.append('      例 %s: %s' % (w, '; '.join(ks)))
        summ[p] = {'diff': len(diff), 'fixed': len(fixed), 'other': len(other), 'worse': dict(wc)}
        if p + '_g' in vs:
            L.append('      見張りだけ(keep 0)の形が今と同じ組: %d / %d' % (
                sum(1 for r in cells.values() if r.get('same_%s_g' % p)), len(cells)))
        if p + '_off' in vs:
            L.append('      案の写しの OFF が今の OFF とビット単位で同じ: %d / %d' % (
                sum(1 for r in cells.values() if r.get('same_%s_off' % p)), len(cells)))
        if p + 'x_off' in vs:
            xd = [k for k, r in cells.items() if not r.get('same_%sx_off' % p, True)]
            xf = [k for k in xd if fail(cells[k]['off']) and not fail(cells[k][p + 'x_off'])]
            xw = collections.Counter(w for k in xd for w in worse(cells[k]['off'], cells[k][p + 'x_off'])
                                     if not (k in xf and w in ('back_longer', 't_back_end_up', 't_escape_up', 'vmax_up')))
            L.append('      案を OFF にも入れた形: 今の OFF と違う %d 組 = 直した %d・悪くなった印 %s' % (
                len(xd), len(xf), dict(xw) if xw else 'なし'))
    return '\n'.join(L), summ


def main(kinds):
    if not kinds:
        kinds = sorted(p.stem[len('exp10_'):] for p in (HERE / 'out').glob('exp10_*.json'))
    txt = []
    allsum = {}
    for k in kinds:
        r = one(k)
        if r is None:
            continue
        txt.append(r[0])
        allsum[k] = r[1]
    s = '\n'.join(txt)
    print(s)
    (HERE / 'out' / 'analyze10.txt').write_text(s + '\n', encoding='utf-8')
    (HERE / 'out' / 'analyze10.json').write_text(json.dumps(allsum, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1:])
