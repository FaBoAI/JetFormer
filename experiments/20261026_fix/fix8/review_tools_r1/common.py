"""反証 8-1 回目・係 tools(読むだけ・直さない)の台本の共通部分。

作り物の記録の作り方と ⑤ ③ ④ の流し方は tests/test_fix7_tools.py・test_fix7_tools_r2.py・test_fix8_tools_decided.py をそのまま使う
(同じ作り方・同じ読み方)。作り物は ../tmp_review_tools_r1/ の下に作り、台本の終わりに消す。本物の data/raw・labels は読むだけ。
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix7_tools as t7  # noqa: E402
import test_fix7_tools_r2 as t72  # noqa: E402
import test_fix8_tools_decided as t8  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_review_tools_r1'
PRM, ON = t7.PRM, t7.ON
ON200 = dict(ON, launch_us=200.0)
IG = ['--ignore-gain']


def fresh(name):
    d = TMP / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def cleanup():
    if TMP.exists():
        shutil.rmtree(TMP)


def verdict(g):
    c = g.get('compare') or {}
    if not c:
        return '比べなし'
    return '使ってよい' if c.get('all_ok') else '満たさない'


def group_text(g):
    c = g.get('compare') or {}
    on = (g.get('by_launch') or {}).get('on') or {}
    off = (g.get('by_launch') or {}).get('off') or {}
    keys = ('trial_excluded', 'trial_on_collisions', 'trial_slip', 'trial_slip_unmeasured', 'midrun_excluded',
            'midrun_on_collisions', 'midrun_slip', 'first_lap_gap_on', 'abort_resume_excluded', 'abort_resume_on_collisions',
            'abort_resume_slip', 'not_reached', 'mixed_options', 'form_split')
    extra = {k: g[k] for k in keys if g.get(k)}
    if 'not_reached' in extra:
        extra['not_reached'] = {k: {'n': v['n'], 'logs': v['logs'], 'after_abort': v.get('after_abort')}
                                for k, v in extra['not_reached'].items()}
    return ('組 [回 %s・天井 %s・左 %s] ON n %s(中央値 %s)・OFF n %s(中央値 %s)・ON − OFF %s・%s・checks の False %s・%s' % (
        g.get('session'), g.get('speed_ceiling'), g.get('ceil_left_mps'), on.get('n'), on.get('to_r1_median_s'), off.get('n'),
        off.get('to_r1_median_s'), c.get('d_to_r1_s'), verdict(g),
        sorted(k for k, v in (c.get('checks') or {}).items() if v is not True), json.dumps(extra, ensure_ascii=False)))


def runc(D, ver='new', args=IG, tag='x'):
    return t8.runc(D, ver, args, tag)


def write(name, lines):
    txt = '\n'.join(lines) + '\n'
    (OUT / name).write_text(txt, encoding='utf-8')
    print(txt)
