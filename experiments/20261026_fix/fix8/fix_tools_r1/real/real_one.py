"""fix8 係 tools 直す 8-1 回目(読むだけ): 本物の記録で ⑤ collect.py か ③ standing_3lap・④ quick_eval_by_params を、直す前
(tests/fixtures/fix8_tools_r1_before/*.orig。⑤ の中で読む ③ も同じ版)か今で 1 回流す。書くのは ../out/real_<old|new>/ だけ
(fix8/decided_tools/real_one.py と同じ流し方)。

  jf-heavy --lock none --mem 1500M -- python -B real_one.py old|new c5 <出力の名前> [collect の引数 ...]
  jf-heavy --lock none --mem 1500M -- python -B real_one.py old|new s3qp <日>
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
FIX = JF / 'tests/fixtures/fix8_tools_r1_before'


def main(tag, what, *rest):
    if tag == 'old':
        s3 = load_mod('standing_3lap', FIX / 'standing_3lap.py.orig')
        sys.modules['standing_3lap'] = s3
        qp = load_mod('quick_eval_by_params', FIX / 'quick_eval_by_params.py.orig')
        col = FIX / 'collect.py.orig'
    else:
        import standing_3lap as s3  # noqa: F401
        import quick_eval_by_params as qp
        col = JF / 'experiments/20261026_fix/diag_launch/collect.py'
    O = HERE.parent / 'out' / ('real_%s' % tag)
    O.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    if what == 'c5':
        name, args = rest[0], list(rest[1:])
        m = load_mod('collect_%s' % tag, col)
        with contextlib.redirect_stdout(buf):
            rc = m.main(args + ['--out', str(O / ('launches_%s.json' % name))])
        (O / ('launches_%s.stdout' % name)).write_text(buf.getvalue(), encoding='utf-8')
        print(tag, 'c5', name, 'rc', rc)
    else:
        d = rest[0]
        with contextlib.redirect_stdout(buf):
            rc1 = s3.main(['--day', d, '--json', str(O / ('s3_%s.json' % d)), '--md', str(O / ('s3_%s.md' % d))])
            rc2 = qp.main(['--day', d, '--out', str(O / ('qp_%s' % d))])
            rc3 = qp.main(['--day', d, '--ignore', 'gain', '--out', str(O / ('qp_%s_ig' % d))])
        (O / ('s3qp_stdout_%s.txt' % d)).write_text(buf.getvalue(), encoding='utf-8')
        print(tag, 's3qp', d, 'rc', rc1, rc2, rc3)


if __name__ == '__main__':
    main(*sys.argv[1:])
