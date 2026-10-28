"""反証 8-1 docs: 文書の控え(.bak-20261028-fix8-decided)と今の文書の違いを、行ごとに「足した・消した」切れ端で出す(読むだけ)。"""
import difflib
import sys
from pathlib import Path
REP = Path('/home/jetson/jetformer_reports/fix_20261026')
for name in sys.argv[1:]:
    cur = (REP / name).read_text(encoding='utf-8').splitlines()
    old = (REP / (name + '.bak-20261028-fix8-decided')).read_text(encoding='utf-8').splitlines()
    sm = difflib.SequenceMatcher(None, old, cur, autojunk=False)
    print('=====', name)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == 'equal':
            continue
        if op == 'insert':
            for j in range(b0, b1):
                print('[+%d 新しい行] %s' % (j + 1, cur[j]))
            continue
        if op == 'delete':
            for i in range(a0, a1):
                print('[-%d 消した行] %s' % (i + 1, old[i]))
            continue
        # replace: 行どうしを対にして文字の違い
        pairs = list(zip(range(a0, a1), range(b0, b1)))
        for i, j in pairs:
            s = difflib.SequenceMatcher(None, old[i], cur[j], autojunk=False)
            for o2, x0, x1, y0, y1 in s.get_opcodes():
                if o2 == 'equal':
                    continue
                ctx = old[i][max(0, x0 - 40):x0]
                print('[%d→%d %s] …%s【- %s】【+ %s】' % (i + 1, j + 1, o2, ctx, old[i][x0:x1], cur[j][y0:y1]))
        for j in range(b0 + len(pairs), b1):
            print('[+%d 新しい行] %s' % (j + 1, cur[j]))
        for i in range(a0 + len(pairs), a1):
            print('[-%d 消した行] %s' % (i + 1, old[i]))
