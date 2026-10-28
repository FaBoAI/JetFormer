"""fix8 係 tools 直す 8-2 回目(読むだけ): tests/test_fix8_docs_sync1.py::test_words_match_the_code が collect.py の中に引く文字列(app.js の
時刻の行のあとにあるので、app.js が変わると照らされない)を、その試験の文の並びのまま今の collect.py で照らす → out/docs_words_check.txt

    python3 -I -B experiments/20261026_fix/fix8/fix_tools_r2/docs_words_check.py
"""
import ast
import re
from pathlib import Path

JF = Path('/home/jetson/jetformer')
src = (JF / 'tests/test_fix8_docs_sync1.py').read_text(encoding='utf-8')
i = src.index("    co = (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py').read_text(encoding='utf-8')\n    for w in (")
j = src.index("        assert w in co, w", i)
block = src[i:j]
ns = {'STALL': ast.literal_eval(re.search(r"^STALL = (.+)$", src, re.M).group(1))}
words = eval(block[block.index('for w in (') + len('for w in '):block.rindex(':')], ns)
co = (JF / 'experiments/20261026_fix/diag_launch/collect.py').read_text(encoding='utf-8')
lines = ['%s %s' % ('ある' if w in co else '無い', w) for w in words]
lines.append('collect.py の文字列 %d・無い %d' % (len(words), sum(1 for w in words if w not in co)))
out = Path(__file__).resolve().parent / 'out' / 'docs_words_check.txt'
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
