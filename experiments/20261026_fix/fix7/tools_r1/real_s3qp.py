"""fix7 係 tools 直す 7-1 回目: 本物の記録 10/10・10/23・10/24 で ③ standing_3lap・④ quick_eval_by_params を、直す前
(before/*.orig)と今で流す(読むだけ。書くのは out/real_s3qp_<old|new>/ だけ。data/raw・data/labels は読むだけ。GPU なし)。

  jf-heavy --lock none --mem 1500M -- python3 -I real_s3qp.py old|new

old は before/standing_3lap.py.orig を standing_3lap として読み(④ の写しもそれを使う)、new は tools/ の今の版。
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
DAYS = ('20261010', '20261023', '20261024')


def main(tag):
    if tag == 'old':
        s3 = load_mod('standing_3lap', HERE / 'before' / 'standing_3lap.py.orig')
        sys.modules['standing_3lap'] = s3
        qp = load_mod('quick_eval_by_params', HERE / 'before' / 'quick_eval_by_params.py.orig')
    else:
        import standing_3lap as s3  # noqa: F401
        import quick_eval_by_params as qp
    O = HERE / 'out' / ('real_s3qp_%s' % tag)
    O.mkdir(parents=True, exist_ok=True)
    for d in DAYS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc1 = s3.main(['--day', d, '--json', str(O / ('s3_%s.json' % d)), '--md', str(O / ('s3_%s.md' % d))])
            rc2 = qp.main(['--day', d, '--out', str(O / ('qp_%s' % d))])
            rc3 = qp.main(['--day', d, '--ignore', 'gain', '--out', str(O / ('qp_%s_ig' % d))])
        (O / ('stdout_%s.txt' % d)).write_text(buf.getvalue(), encoding='utf-8')
        print(tag, d, 'rc', rc1, rc2, rc3)


if __name__ == '__main__':
    main(sys.argv[1])
