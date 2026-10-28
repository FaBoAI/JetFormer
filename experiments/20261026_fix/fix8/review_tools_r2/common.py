"""反証 8-2 回目・係 tools(読むだけ・直さない)の台本の共通部分。

作り物の記録の作り方と ⑤ の流し方は tests/test_fix8_tools_r1.py(tfday・sday・runc の版 'r1new' 今 / 'r1old' 8-1 の直す前)・
test_fix7_tools.py・test_fix7_tools_r2.py・test_fix8_tools_decided.py をそのまま使う(同じ作り方・同じ読み方)。
作り物は ../tmp_review_tools_r2/ の下に作り、台本の終わりに消す。本物の data/raw・labels は読むだけ。
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix7_tools as t7  # noqa: E402
import test_fix7_tools_r2 as t72  # noqa: E402
import test_fix8_tools_decided as t8  # noqa: E402
import test_fix8_tools_r1 as T  # noqa: E402  ('r1new' / 'r1old' を t8.VER に足す)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_review_tools_r2'
PRM, ON = t7.PRM, t7.ON
ON200 = dict(ON, launch_us=200.0)
IG = ['--ignore-gain']
FO = ['--first-only']
IGF = ['--ignore-gain', '--first-only']
ARGS = (('--day', []), ('--ignore-gain', IG), ('--first-only', FO), ('--ignore-gain --first-only', IGF))
OFFOPT = {'rec_back_guard': 0, 'rec_hold_keep': 0}


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


EXTRA = ('trial_excluded', 'midrun_excluded', 'trial_on_collisions', 'midrun_on_collisions', 'trial_slip', 'midrun_slip',
         'trial_slip_unmeasured', 'midrun_slip_unmeasured', 'first_lap_gap_on', 'not_reached', 'form_split', 'gain_split',
         'other_group_on', 'launch_stall_excluded', 'launch_stall_on', 'launch_stall_on_collisions', 'launch_stall_slip',
         'on_faults_counted_in', 'launch_gap_short', 'mixed_options', 'abort_resume_excluded', 'abort_resume_on_collisions',
         'unreadable')


def gtext(g, keys=EXTRA):
    c = g.get('compare') or {}
    on = (g.get('by_launch') or {}).get('on') or {}
    off = (g.get('by_launch') or {}).get('off') or {}
    extra = {k: g[k] for k in keys if g.get(k)}
    if 'not_reached' in extra:
        extra['not_reached'] = {k: {'n': v['n'], 'logs': v['logs'], 'collisions': v['collisions'],
                                    'after_abort': v.get('after_abort')} for k, v in extra['not_reached'].items()}
    if 'other_group_on' in extra:
        extra['other_group_on'] = {k: extra['other_group_on'][k] for k in ('logs', 'checks')}
    if 'unreadable' in extra:
        extra['unreadable'] = {k: extra['unreadable'][k] for k in ('n', 'logs', 'launch_state')}
    return ('[%s・g %s・天井 %s(%s)・自動 %s(%s)] ON n %s(%s)・OFF n %s(%s)・ON − OFF %s・%s・False %s・%s' % (
        g['session'], g['gain'], g['speed_ceiling'], g['ceil_left_mps'], g['speed_auto'], g['auto_target_mps'], on.get('n'),
        on.get('to_r1_median_s'), off.get('n'), off.get('to_r1_median_s'), c.get('d_to_r1_s'), verdict(g),
        sorted(k for k, v in (c.get('checks') or {}).items() if v is not True), json.dumps(extra, ensure_ascii=False)))


def runc(D, ver='r1new', args=IG, tag='x'):
    return t8.runc(D, ver, args, tag)


def lines_with(out, *heads):
    return [x for x in out.splitlines() if any(x.startswith(h) for h in heads)]


def compare_lines(out):
    return [x for x in out.splitlines() if x.startswith('  ON − OFF')]


def load_new():
    return t8.load_mod('c8r2_new_unit', T.NEW_C)


def write(name, lines):
    txt = '\n'.join(lines) + '\n'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(txt, encoding='utf-8')
    print(txt)


# ------------------------------------------------------------------ 速さの形(本物に近い: 止まりの 0.1 秒あとから中立の惰性)
ACC, DEC = 2.0 / 0.6, 0.9


def sim_vk(dur, t_move=1.22, neutral=(), vmax=2.0, dt=0.001):
    """t_move に動き出し、ACC で vmax へ。neutral [(始め, 終わり)] の間は DEC で惰性(drive が中立を書いた間)。
    戻り値 vkeys [(秒, m/s)](10 ms ごと)"""
    ts = np.arange(0.0, dur + dt / 2, dt)
    v = np.zeros(len(ts))
    for i in range(1, len(ts)):
        t = ts[i]
        if t < t_move:
            v[i] = 0.0
            continue
        if any(a <= t < b for a, b in neutral):
            v[i] = max(0.0, v[i - 1] - DEC * dt)
        else:
            v[i] = min(vmax, v[i - 1] + ACC * dt)
    k = np.arange(0, len(ts), 10)
    return [(round(float(ts[j]), 3), float(v[j])) for j in k]


def gaps_to_frames(gaps):
    """フレームの間 g 秒(30 fps のフレーム k/30 から)の止まりを、write_log の stall/hold の間 [(a, b)] に。
    gaps = [(始めのフレームの番号 k, 抜くフレームの数 n)] → (k/30 + 0.001, (k + n)/30 + 0.001)"""
    return [(k / 30.0 + 0.001, (k + n) / 30.0 + 0.001) for k, n in gaps]


def neutral_of(gaps):
    """drive の中立の間(前のフレームの 0.1 秒あと〜次のフレーム)。gaps は gaps_to_frames と同じ"""
    out = []
    for k, n in gaps:
        a, b = k / 30.0, (k + n + 1) / 30.0
        if b - a > 0.1:
            out.append((a + 0.1, b))
    return out
