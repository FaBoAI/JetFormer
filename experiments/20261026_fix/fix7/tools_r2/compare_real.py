"""fix7 係 tools 直す 7-2 回目: run_real.sh の old(fix7 r1)/ new を比べる(読むだけ)。出力は out/compare_real.txt"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE / 'out'
A, B = O / 'real_old', O / 'real_new'
lines = []


def nopath(t):
    return '\n'.join(x for x in t.splitlines() if not x.startswith('出力 '))


for p in sorted(A.rglob('*')):
    if p.is_dir():
        continue
    rel = p.relative_to(A)
    q = B / rel
    if not q.exists():
        lines.append('%-44s 今の方に無い' % rel)
        continue
    a, b = p.read_bytes(), q.read_bytes()
    if p.suffix == '.stdout':
        a, b = nopath(a.decode('utf-8')).encode('utf-8'), nopath(b.decode('utf-8')).encode('utf-8')
    if a == b:
        lines.append('%-44s バイト単位で同じ' % rel)
        continue
    lines.append('%-44s 違う' % rel)
    if p.suffix == '.json' and p.name.startswith('launches'):
        ja, jb = json.loads(a), json.loads(b)
        marked = [(r['log'], r['t_ai'], r['after_abort']) for r in jb['launches'] if r.get('after_abort')]
        lines.append('   印の付いた発進(今): %s' % marked)
        ra = [{k: v for k, v in r.items() if k != 'after_abort'} for r in ja['launches']]
        rb = [{k: v for k, v in r.items() if k != 'after_abort'} for r in jb['launches']]
        lines.append('   launches(after_abort を除く)%s' % ('同じ' if ra == rb else '違う'))
        for ga, gb in zip(ja.get('summary') or [], jb.get('summary') or []):
            if ga != gb:
                ks = sorted(k for k in set(ga) | set(gb) if ga.get(k) != gb.get(k))
                lines.append('   組 %s %s: 違う欄 %s' % (gb.get('session'), gb.get('model'), ks))
                for k in ks:
                    lines.append('     %s: %s → %s' % (k, json.dumps(ga.get(k), ensure_ascii=False)[:300],
                                                     json.dumps(gb.get(k), ensure_ascii=False)[:300]))
        if len(ja.get('summary') or []) != len(jb.get('summary') or []):
            lines.append('   組の数 %d→%d' % (len(ja.get('summary') or []), len(jb.get('summary') or [])))
        continue
    ta, tb = a.decode('utf-8').splitlines(), b.decode('utf-8').splitlines()
    sa, sb = set(ta), set(tb)
    for x in tb:
        if x not in sa:
            lines.append('     + ' + x[:400])
    for x in ta:
        if x not in sb:
            lines.append('     - ' + x[:400])
(O / 'compare_real.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
