"""読むだけ: 足す所の印 ‸ の位置に「★」を足すと、前の new(今の形)が崩れるか。使い方: trypos.py <文書> '<印つきの文>' ..."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spans import broken_by, REPORTS
files = ('runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix3/disk.md', 'fix7/README.md', 'fix6/README.md')
texts = {f: (REPORTS / f).read_text(encoding='utf-8') for f in files}
args = sys.argv[1:]
for f, x in zip(args[::2], args[1::2]):
    old = x.replace('‸', '')
    if texts[f].count(old) != 1:
        print(f, '|', old[:40].replace('\n', '⏎'), '→ 文書に %d 回' % texts[f].count(old))
        continue
    new = x.replace('‸', '★')
    b = broken_by(texts, f, old, new)
    print(f, '|', x[:60].replace('\n', '⏎'), '→', b or 'OK(崩れない)')
