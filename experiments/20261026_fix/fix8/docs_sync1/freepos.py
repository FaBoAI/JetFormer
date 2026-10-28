"""読むだけ: 文書の行ごとに、行の始め・終わり(改行の前)が前の new の中かを出す(spans.py の持ち主つき)。
使い方: python3 -I -B freepos.py <文書> <最初の行> <最後の行>"""
import sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from spans import spans, inside, REPORTS

files = ('runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix3/disk.md', 'fix7/README.md', 'fix6/README.md')
texts = {f: (REPORTS / f).read_text(encoding='utf-8') for f in files}
sp, bad = spans(texts)
f, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = texts[f]
starts = [0]
for i, ch in enumerate(t):
    if ch == '\n':
        starts.append(i + 1)
for ln in range(a, b + 1):
    s = starts[ln - 1]
    e = (starts[ln] - 1) if ln < len(starts) else len(t)
    print('%4d 始め %-28s 終わり %-28s | %s' % (ln, inside(sp, f, s) or '-', inside(sp, f, e) or '-', t[s:e][:40]))
