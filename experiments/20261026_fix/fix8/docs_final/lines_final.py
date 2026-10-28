"""読むだけ: 文書の行ごとに、その行に掛かる前の new(spans_final)の持ち主と、new の始め・終わりの文(前後 30 字)を出す。
使い方: python3 -I -B lines_final.py <文書> <行> [<行> ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spans_final import spans, REPORTS, FILES
texts = {f: (REPORTS / f).read_text(encoding='utf-8') for f in FILES}
sp, bad = spans(texts)
if bad:
    print('new が 1 回でない:', bad)
f = sys.argv[1]
t = texts[f]
starts = [0]
for i, ch in enumerate(t):
    if ch == '\n':
        starts.append(i + 1)
for ln in map(int, sys.argv[2:]):
    a = starts[ln - 1]
    b = starts[ln] - 1 if ln < len(starts) else len(t)
    print('== 行', ln, '(%d 字)' % (b - a))
    for s, e, k, i in sorted(sp.get(f, [])):
        if s < b and a < e:
            print('  %-8s #%-3d [%7d,%7d) 始め «%s» … 終わり «%s»' % (k, i, s - a, e - a, t[s:s + 30].replace('\n', '⏎'),
                                                               t[max(s, e - 30):e].replace('\n', '⏎')))
