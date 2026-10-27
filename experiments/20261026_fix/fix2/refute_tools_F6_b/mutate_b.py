#!/usr/bin/env python3
"""指摘 tools:F6 の確かめ(2026-10-26 fix2・打ち消し係 b): 本物のファイルは変えず、写しに 1 か所ずつ壊しを入れて
対象の試験(test_standing_3lap.py・test_quick_eval_by_params.py・fix2/test_tools_collect.py)だけを流す。
    python3 mutate_b.py <写しの場所> <basetemp の親> [壊しの名前 …]
写しの場所・basetemp は終わったら消す(この台本の最後で消す)。"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
WORK = Path(sys.argv[1])
BASET = Path(sys.argv[2])
ONLY = sys.argv[3:]
DEBUG = bool(os.environ.get('MUT_DEBUG'))
COL = 'experiments/20261026_fix/diag_launch/collect.py'

MUT = [
    ('control_s3', 'tools/standing_3lap.py', "MOVE_ERPM = 400.0 ", "MOVE_ERPM = 4000.0 "),
    ('control_qp', 'tools/quick_eval_by_params.py', "txt.append('発進 OFF')", "txt.append('発進 X')"),
    ('control_col', COL, "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.5,"),
    ('frame_gap', 'tools/standing_3lap.py', "if _max_gap(t, t_ai, e3) > le.GAP_S:", "if False:"),
    ('vehicle_gap', 'tools/standing_3lap.py', "if _max_gap(vt, t_move, e3) > le.GAP_S:", "if False:"),
    ('line_lap_5_16', 'tools/standing_3lap.py', "if not all(LAP_MIN_S <= x <= LAP_MAX_S for x in row['laps_line_s']):", "if False:"),
    ('anchor_cont', 'tools/standing_3lap.py', "if len(A_) == 4 or not (LAP_MIN_S <= x - A_[-1] <= LAP_MAX_S):", "if len(A_) == 4:"),
    ('to_r1_15s', 'tools/standing_3lap.py', " or nxt[0] - t_ai > MAX_TO_R1_S:", ":"),
    ('cont_s', 'tools/standing_3lap.py', "if not all(abs(w[j]['b'] - w[j + 1]['a']) <= CONT_S for j in range(2)):", "if False:"),
    ('dry_run_frames', 'tools/standing_3lap.py', "ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)", "ai = lab['ai'].astype(bool)"),
    ('dist_dpsi', 'tools/standing_3lap.py', "if DIST_DPSI[0] <= dpsi <= DIST_DPSI[1] and b - a < LAP_MAX_S:", "if b - a < LAP_MAX_S:"),
    ('best_min', 'tools/standing_3lap.py', "return min(v, key=lambda s: s['from_move_s']) if v else None", "return max(v, key=lambda s: s['from_move_s']) if v else None"),
    ('qp_gain', 'tools/quick_eval_by_params.py', "key.append(('gain', g if g is not None else -1))", "key.append(('gain', 0))"),
    ('qp_steer', 'tools/quick_eval_by_params.py', "key.append(('steer', sg))", "key.append(('steer', 1.0))"),
    ('qp_zone', 'tools/quick_eval_by_params.py', "key.append(('zone', z))", "key.append(('zone', 0))"),
    ('qp_mode', 'tools/quick_eval_by_params.py', "key.append(('mode', m if isinstance(m, int) else -1))", "key.append(('mode', 0))"),
    ('qp_auto_target', 'tools/quick_eval_by_params.py', "key.append(('auto', (1, _num(S.get('auto_target_mps')) or 0)))", "key.append(('auto', (1, 0)))"),
    ('qp_launch_us', 'tools/quick_eval_by_params.py', "key.append(('launch', (2, _num(S.get('launch_us')) or 0,", "key.append(('launch', (2, 0,"),
    ('qp_valid', 'tools/quick_eval_by_params.py', "stand = [s for r in rs for s in r.get('standing') or [] if s.get('valid')]", "stand = [s for r in rs for s in r.get('standing') or [] if s.get('from_move_s') is not None]"),
    ('qp_fast9', 'tools/quick_eval_by_params.py', "'fast_laps': sum(1 for x in clean if x < FAST_LAP_S),", "'fast_laps': 0,"),
    ('col_on_none', COL, "        return False\n\n\ndef extended_launch", "        return x is None\n\n\ndef extended_launch"),
    ('col_on_s', COL, "row['launch_on_s'] = round(float(ts[on][-1] - t0) + 1 / 30.0, 3)", "row['launch_on_s'] = round(float(ts[on][-1] - t0), 3)"),
    ('col_ceil_key', COL, "return (d['model_name'], round(g, 2), d['session'], _on(S.get('speed_ceiling')), S.get('ceil_left_mps'),", "return (d['model_name'], round(g, 2), d['session'], 0, 0,"),
]


def collect_src(src):
    return src.replace("sys.path.insert(0, '/home/jetson/jetformer/tools')", "sys.path.insert(0, %r)" % str(WORK / 'tools'))


def put(rel, old=None, new=None):
    """写しの rel を本物(+ 壊し)で書き直す。全部を本物のコピーにする(symlink の resolve で本物の tools に戻らないように)"""
    src = (JF / rel).read_text(encoding='utf-8')
    if old is not None:
        assert src.count(old) == 1, (rel, old, src.count(old))
        src = src.replace(old, new)
    if rel == COL:
        src = collect_src(src)
    (WORK / rel).write_text(src, encoding='utf-8')


def build():
    if WORK.exists():
        shutil.rmtree(WORK)
    for d in ('tools', 'tests', 'experiments/20261026_fix/diag_launch', 'experiments/20261026_fix/fix2'):
        (WORK / d).mkdir(parents=True)
    os.symlink(JF / 'jetformer', WORK / 'jetformer')
    for p in (JF / 'tools').glob('*.py'):
        shutil.copy(p, WORK / 'tools' / p.name)
    for n in ('test_standing_3lap.py', 'test_quick_eval_by_params.py', 'test_lap_anchor.py'):
        shutil.copy(JF / 'tests' / n, WORK / 'tests' / n)
    put(COL)
    t = (JF / 'experiments/20261026_fix/fix2/test_tools_collect.py').read_text(encoding='utf-8')
    t = t.replace("JF = Path('/home/jetson/jetformer')", "JF = Path(%r)" % str(WORK))
    (WORK / 'experiments/20261026_fix/fix2/test_tools_collect.py').write_text(t, encoding='utf-8')


def pytest(tag):
    BASET.mkdir(parents=True, exist_ok=True)
    tests = ['tests/test_standing_3lap.py', 'tests/test_quick_eval_by_params.py',
             'experiments/20261026_fix/fix2/test_tools_collect.py']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp',
                        str(BASET / tag), '--rootdir', str(WORK)] + tests, cwd=str(WORK), capture_output=True, text=True,
                       env=env)
    shutil.rmtree(BASET / tag, ignore_errors=True)
    if r.returncode and DEBUG:
        print(r.stdout[-2500:], r.stderr[-1500:])
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    fails = [x for x in lines if x.startswith(('FAILED', 'ERROR'))]
    return r.returncode, (lines[-1] if lines else '?'), fails


try:
    build()
    rc, last, _ = pytest('base')
    print('base', rc, last, flush=True)
    for tag, rel, old, new in MUT:
        if ONLY and tag not in ONLY:
            continue
        put(rel, old, new)
        rc, last, fails = pytest(tag)
        print('%-15s %s | %s | %s' % (tag, 'SURVIVED(試験が通った)' if rc == 0 else 'killed', last,
                                     ' / '.join(f.split(' - ')[0] for f in fails)), flush=True)
        put(rel)
finally:
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(BASET, ignore_errors=True)
