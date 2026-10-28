"""fix8 係 tools 直す 8-1 回目: run_real.sh の old(直す前の写し)/ new(今)を比べる(読むだけ)。出力は ../out/compare_real.txt。
⑤ の JSON は、発進ごとの行の launch_stall(カメラの止まり)・組の other_group_on / on_faults_counted_in / launch_stall_* と、判定
(by_launch・compare)の違いを分けて出す"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parent / 'out'
A, B = O / 'real_old', O / 'real_new'
NEWK = ('launch_stall',)
lines = []


def nopath(t):
    return '\n'.join(x for x in t.splitlines() if not x.startswith('出力 '))


def strip_rows(rows):
    return [{k: v for k, v in r.items() if k not in NEWK} for r in rows]


n_same = n_diff = 0
for p in sorted(A.rglob('*')):
    if p.is_dir():
        continue
    rel = p.relative_to(A)
    q = B / rel
    if not q.exists():
        lines.append('%-44s 今の方に無い' % rel)
        continue
    a, b = p.read_bytes(), q.read_bytes()
    if p.suffix in ('.stdout', '.txt'):
        a, b = nopath(a.decode('utf-8')).encode('utf-8'), nopath(b.decode('utf-8')).encode('utf-8')
    if a == b:
        n_same += 1
        lines.append('%-44s バイト単位で同じ' % rel)
        continue
    n_diff += 1
    lines.append('%-44s 違う' % rel)
    if p.suffix == '.json' and p.name.startswith('launches'):
        ja, jb = json.loads(a), json.loads(b)
        ma = {(r['log'], r['t_ai']): r.get('launch_stall') for r in ja['launches'] if r.get('launch_stall')}
        mb = {(r['log'], r['t_ai']): r.get('launch_stall') for r in jb['launches'] if r.get('launch_stall')}
        lines.append('   launch_stall の付いた発進 前 %d → 今 %d %s' % (len(ma), len(mb), json.dumps(
            {'%s@%s' % k: v for k, v in mb.items()}, ensure_ascii=False)))
        lines.append('   launches(launch_stall を除く)%s' % ('同じ' if strip_rows(ja['launches']) == strip_rows(jb['launches']) else '違う'))
        sa, sb = ja.get('summary') or [], jb.get('summary') or []
        if len(sa) != len(sb):
            lines.append('   組の数 %d→%d' % (len(sa), len(sb)))
        for ga, gb in zip(sa, sb):
            if ga != gb:
                ks = sorted(k for k in set(ga) | set(gb) if ga.get(k) != gb.get(k))
                lines.append('   組 %s %s g%s: 違う欄 %s' % (gb.get('session'), gb.get('model'), gb.get('gain'), ks))
                for k in ks:
                    lines.append('     %s: %s → %s' % (k, json.dumps(ga.get(k), ensure_ascii=False)[:600],
                                                     json.dumps(gb.get(k), ensure_ascii=False)[:600]))
        for k in sorted(set(ja) | set(jb)):
            if k not in ('launches', 'summary') and ja.get(k) != jb.get(k):
                lines.append('   欄 %s が違う' % k)
        continue
    ta, tb = a.decode('utf-8').splitlines(), b.decode('utf-8').splitlines()
    sa, sb = set(ta), set(tb)
    for x in tb:
        if x not in sa:
            lines.append('     + ' + x[:900])
    for x in ta:
        if x not in sb:
            lines.append('     - ' + x[:900])
lines.append('バイト単位で同じ %d・違う %d' % (n_same, n_diff))
(O / 'compare_real.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
