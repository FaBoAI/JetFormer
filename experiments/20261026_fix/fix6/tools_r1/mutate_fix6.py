"""直す 6-1 回目・係 tools: わざと壊した collect.py で tests/test_fix6_tools.py が落ちるかを見る。本物の collect.py は変えない
(写しを fix6/tmp_tools/r1_mut/ に作り、環境変数 JF_FIX6_COLLECT で試験に渡す)。写しと basetemp は最後に消す。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B mutate_fix6.py
        → out/mutate_fix6_out.txt
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
SRC = JF / 'experiments/20261026_fix/diag_launch/collect.py'
BEFORE = Path(__file__).resolve().parent / 'before' / 'collect.py.orig'
WORK = JF / 'experiments/20261026_fix/fix6/tmp_tools/r1_mut'
OUT = Path(__file__).resolve().parent / 'out' / 'mutate_fix6_out.txt'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'

MUT = [
    ('対照: CRIT の起点の差 0.15 → 0.50', 'CRIT = {\'to_r1_gain_s\': 0.15,', 'CRIT = {\'to_r1_gain_s\': 0.50,'),
    ('AI のフレームが無い本を読めない記録に入れない', "                if ls:\n                    st = launch_starts(d)",
     "                if False:\n                    st = launch_starts(d)"),
    ('lost_scenes がいつも空', "    if not names:\n        return []", "    if True:\n        return []"),
    ('フレームのある欠けたシーンも「欠けたシーン」に数える', 'if i is None or bool((fs == i).any()):', 'if i is None:'),
    ('events の start で ON にしない', "                        u['launch_starts'] = st\n                        u['launch_state'] = 'on'",
     "                        u['launch_starts'] = st"),
    ('start が無くても ON にする', "                    if st:             # 発進の前進を出した出来事がある",
     "                    if True:             # 発進の前進を出した出来事がある"),
    ('読めた本の欠けたシーンの間の start を見ない', "            st = launch_starts(d, ls)\n            if not st:",
     "            st = []\n            if not st:"),
    ('--first-only を見ない', 'later = bool(first_only and t_read and min(t_read) < min(st))', 'later = False'),
    ('--first-only で読めた発進の前の start も外す', 'later = bool(first_only and t_read and min(t_read) < min(st))',
     'later = bool(first_only and t_read)'),
    ('読めた本の欠けた発進を OFF に', "u.update(lost_scenes=sc, launch_starts=st, launch_state='on')",
     "u.update(lost_scenes=sc, launch_starts=st, launch_state='off')"),
    ('次のフレームの前の幅を 0 に', 'LOST_HEAD_S, LOST_TAIL_S = 0.1, 0.005', 'LOST_HEAD_S, LOST_TAIL_S = 0.1, 0.0'),
    ('前のフレームの前の幅を 0 に', 'LOST_HEAD_S, LOST_TAIL_S = 0.1, 0.005', 'LOST_HEAD_S, LOST_TAIL_S = 0.0, 0.005'),
    ('skip の行を出さない', "                    print('skip', log, why + ('。events.jsonl に発進の前進の start %s' % starts_text(st) if st else ''))",
     "                    pass"),
    ('「記録が欠けている本」を AI のフレームが無い本に出さない', 'for lg in sorted(set(data) | set(noai)):', 'for lg in sorted(data):'),
    ('problems に AI のフレームが無い本を入れない', 'for lg in day_logs if lg in data or lg in noai]', 'for lg in day_logs if lg in data]'),
    ('読めない記録の行に events の start を出さない', "'・events.jsonl の start %s' % starts_text(ss) if ss else ''", "''"),
    ('読めない記録を名前の順に並べ直さない', "            unreadable = sorted(unreadable, key=lambda u: u['log'])\n    res = {",
     "            pass\n    res = {"),
    ('行の後ろの 1 文を出さない', '                if lg in lost_ev:\n', '                if False:\n'),
    ('組の launch_starts を出さない', "            if any(u.get('launch_starts') for u in us_):", "            if False:"),
]


def run(path, tag, stop=True):
    bt = WORK / ('bt_' + tag)
    env = dict(os.environ, JF_FIX6_COLLECT=str(path), PYTHONDONTWRITEBYTECODE='1')
    p = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::DeprecationWarning'] + (['-x'] if stop else []) + ['--tb=line',
                        '--basetemp=%s' % bt, 'tests/test_fix6_tools.py'], cwd=str(JF), env=env, capture_output=True, text=True)
    shutil.rmtree(bt, ignore_errors=True)
    tail = [x for x in p.stdout.splitlines() if x.strip()][-3:]
    return p.returncode, tail


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    src = SRC.read_text(encoding='utf-8')
    lines = []
    try:
        rc, tail = run(SRC, 'now')
        lines.append('今の版: rc=%d %s' % (rc, tail[-1]))
        rc, tail = run(BEFORE, 'before', stop=False)
        lines.append('直す前の版: rc=%d %s' % (rc, ' / '.join(tail)))
        for i, (name, old, new) in enumerate(MUT):
            n = src.count(old)
            if n != 1:
                lines.append('[%02d] %s: 当たらない(%d 回)' % (i, name, n))
                continue
            mp = WORK / ('collect_m%02d.py' % i)
            mp.write_text(src.replace(old, new), encoding='utf-8')
            rc, tail = run(mp, 'm%02d' % i)
            fail = [x for x in tail if 'Error' in x or 'assert' in x or 'FAILED' in x or 'failed' in x]
            lines.append('[%02d] %s: %s rc=%d %s' % (i, name, '落ちる' if rc != 0 else '** 通ってしまう **', rc,
                                                     ' / '.join(fail or tail)[:300]))
            print(lines[-1], flush=True)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
