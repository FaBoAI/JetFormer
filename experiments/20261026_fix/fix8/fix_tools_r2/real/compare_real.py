"""fix8 係 tools 直す 8-2 回目: run_real.sh の old(直す前の写し)/ new(今)を比べる(読むだけ)。出力は ../out/compare_real.txt。
標準出力は「出力 …」の行(出力の場所)だけ除いて比べる。違えば、違う欄・行を出す"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parent / 'out'
A, B = O / 'real_old', O / 'real_new'
lines = []


def nopath(t):
    return '\n'.join(x for x in t.splitlines() if not x.startswith('出力 '))


n_same = n_diff = 0
for p in sorted(A.rglob('*')):
    if p.is_dir():
        continue
    rel = p.relative_to(A)
    q = B / rel
    if not q.exists():
        lines.append('%-48s 今の方に無い' % rel)
        n_diff += 1
        continue
    a, b = p.read_bytes(), q.read_bytes()
    if p.suffix == '.stdout':
        a, b = nopath(a.decode('utf-8')).encode('utf-8'), nopath(b.decode('utf-8')).encode('utf-8')
    if a == b:
        n_same += 1
        lines.append('%-48s バイト単位で同じ(%d バイト)' % (rel, len(b)))
        continue
    n_diff += 1
    lines.append('%-48s 違う' % rel)
    if p.suffix == '.json':
        ja, jb = json.loads(a), json.loads(b)
        for k in sorted(set(ja) | set(jb)):
            if ja.get(k) != jb.get(k):
                lines.append('   欄 %s が違う' % k)
    else:
        sa, sb = set(a.decode('utf-8').splitlines()), set(b.decode('utf-8').splitlines())
        for x in b.decode('utf-8').splitlines():
            if x not in sa:
                lines.append('     + ' + x[:600])
        for x in a.decode('utf-8').splitlines():
            if x not in sb:
                lines.append('     - ' + x[:600])
# 本物の出力の中身(今): 比べのある組・新しい欄
for p in sorted(B.glob('launches_*.json')):
    j = json.loads(p.read_text(encoding='utf-8'))
    S = j.get('summary') or []
    lines.append('%-48s 組 %d・比べのある組 %d・off_dropped %d・launch_stall の印 %d(うち中立の合計 %d)・launch_sum_short %d' % (
        p.name, len(S), sum(1 for g in S if 'compare' in g), sum(1 for g in S if g.get('off_dropped')),
        sum(1 for r in j['launches'] if r.get('launch_stall')),
        sum(1 for r in j['launches'] if (r.get('launch_stall') or {}).get('neutral_s')),
        sum(1 for g in S if g.get('launch_sum_short'))))
lines.append('バイト単位で同じ %d・違う %d' % (n_same, n_diff))
(O / 'compare_real.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
