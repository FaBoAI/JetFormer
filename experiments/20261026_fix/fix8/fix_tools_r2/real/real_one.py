"""fix8 係 tools 直す 8-2 回目(読むだけ): 本物の記録で ⑤ collect.py を、直す前(tests/fixtures/fix8_tools_r2_before/collect.py.orig)か
今で 1 回流す。③ standing_3lap は変えていない(今のものを読む)。書くのは ../out/real_<old|new>/ だけ(fix8/fix_tools_r1/real/real_one.py と
同じ流し方)。

  jf-heavy --lock none --mem 1500M -- python -B real_one.py old|new <出力の名前> [collect の引数 ...]
"""
import contextlib
import io
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
from test_fix3_r1_2 import load_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
FIX = JF / 'tests/fixtures/fix8_tools_r2_before'


def main(tag, name, *args):
    col = FIX / 'collect.py.orig' if tag == 'old' else JF / 'experiments/20261026_fix/diag_launch/collect.py'
    O = HERE.parent / 'out' / ('real_%s' % tag)
    O.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    m = load_mod('collect_r2_%s' % tag, col)
    with contextlib.redirect_stdout(buf):
        rc = m.main(list(args) + ['--out', str(O / ('launches_%s.json' % name))])
    (O / ('launches_%s.stdout' % name)).write_text(buf.getvalue(), encoding='utf-8')
    print(tag, name, 'rc', rc)


if __name__ == '__main__':
    main(*sys.argv[1:])
