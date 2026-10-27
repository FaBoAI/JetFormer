"""fix7 係 tools 直す 7-1 回目: run_real.sh・real_s3qp.py の old / new を比べる(読むだけ)。出力は out/compare_real.txt"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE / 'out'
lines = []


def nopath(t):
    return '\n'.join(x for x in t.splitlines() if not x.startswith('出力 '))


# ⑤
for p in sorted((O / 'real_old').glob('launches*.json')):
    q = O / 'real_new' / p.name
    jo, jn = p.read_bytes(), q.read_bytes()
    so = nopath((O / 'real_old' / (p.stem + '.stdout')).read_text(encoding='utf-8'))
    sn = nopath((O / 'real_new' / (p.stem + '.stdout')).read_text(encoding='utf-8'))
    same = jo == jn
    lines.append('⑤ %-40s JSON %s・標準出力 %s' % (p.name, 'バイト単位で同じ' if same else '違う', '同じ' if so == sn else '違う'))
    if not same:
        a, b = json.loads(jo), json.loads(jn)
        marked = [(r['log'], r['t_ai'], r['after_abort']) for r in b['launches'] if r.get('after_abort')]
        lines.append('   印の付いた発進: %s' % marked)
        ra = [{k: v for k, v in r.items()} for r in a['launches']]
        rb = [{k: v for k, v in r.items() if k != 'after_abort'} for r in b['launches']]
        lines.append('   launches(after_abort を除く)%s' % ('同じ' if ra == rb else '違う'))
        for ga, gb in zip(a.get('summary') or [], b.get('summary') or []):
            if ga != gb:
                ks = sorted(k for k in set(ga) | set(gb) if ga.get(k) != gb.get(k))
                lines.append('   組 %s %s: 違う欄 %s' % (gb.get('session'), gb.get('model'), ks))
                for st in ('off', 'on'):
                    if st in (ga.get('by_launch') or {}):
                        x, y = ga['by_launch'][st], gb['by_launch'].get(st, {})
                        lines.append('     %s n %s→%s restarts %s→%s 起点まで %s→%s 失う %s→%s' % (
                            st, x.get('n'), y.get('n'), x.get('restarts'), y.get('restarts'), x.get('to_r1_median_s'),
                            y.get('to_r1_median_s'), x.get('loss_median_s'), y.get('loss_median_s')))
                lines.append('     abort_resume_excluded %s' % gb.get('abort_resume_excluded'))
        if len(a.get('summary') or []) != len(b.get('summary') or []):
            lines.append('   組の数 %d→%d' % (len(a.get('summary') or []), len(b.get('summary') or [])))
        do = [x for x in so.splitlines() if x not in set(sn.splitlines())]
        dn = [x for x in sn.splitlines() if x not in set(so.splitlines())]
        lines.append('   標準出力の前だけの行 %d・今だけの行 %d' % (len(do), len(dn)))
        for x in dn:
            lines.append('     + ' + x[:260])
        for x in do:
            lines.append('     - ' + x[:260])
# ③④
for p in sorted((O / 'real_s3qp_old').rglob('*')):
    if p.is_dir():
        continue
    q = O / 'real_s3qp_new' / p.relative_to(O / 'real_s3qp_old')
    if not q.exists():
        lines.append('③④ %s: 今の方に無い' % p.relative_to(O / 'real_s3qp_old'))
        continue
    a, b = p.read_bytes(), q.read_bytes()
    if a == b:
        lines.append('③④ %-34s バイト単位で同じ' % p.relative_to(O / 'real_s3qp_old'))
        continue
    lines.append('③④ %-34s 違う' % p.relative_to(O / 'real_s3qp_old'))
    ta, tb = a.decode('utf-8').splitlines(), b.decode('utf-8').splitlines()
    sa, sb = set(ta), set(tb)
    for x in tb:
        if x not in sa:
            lines.append('     + ' + x[:300])
    for x in ta:
        if x not in sb:
            lines.append('     - ' + x[:300])
(O / 'compare_real.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
