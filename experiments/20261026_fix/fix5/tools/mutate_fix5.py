#!/usr/bin/env python3
"""直す 5 回目・係 tools(2026-10-27): 「わざと壊すと落ちる」の確かめ。本物のファイルは変えず、写しの場所に 1 か所ずつ壊しを入れて
(または直す前の写し before/*.orig に戻して)tests/test_fix5_tools.py だけを流す。
    python3 mutate_fix5.py <写しの場所> <basetemp の親> [壊しの名前 …]
写しの場所・basetemp はこの台本の最後で消す(どちらも作業用の一時の場所。本物の data/raw・labels・報告には書かない)"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
WORK = Path(sys.argv[1]).resolve()
BASET = Path(sys.argv[2]).resolve()
ONLY = sys.argv[3:]
assert 'tmp_tools' in str(WORK) and 'tmp_tools' in str(BASET), WORK          # 消してよい場所だけ
COL = 'experiments/20261026_fix/diag_launch/collect.py'
CC, S3, QP = 'tools/car_check.py', 'tools/standing_3lap.py', 'tools/quick_eval_by_params.py'
T = ['tests/test_fix5_tools.py']

# (名前, 指摘, ファイル, 前, 後)。前 = None なら直す前の写し(before/<名前>.orig)にまるごと戻す
MUT = [
    # 試験がちゃんと道具を読むか(効く壊し)
    ('control_s3', '-', S3, "MOVE_ERPM = 400.0 ", "MOVE_ERPM = 4000.0 "),
    ('control_col', '-', COL, "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.5,"),
    # 直す前の版に戻す
    ('cc_orig', 'c1', CC, None, None),
    ('s3_orig', 'c1・c2', S3, None, None),
    ('qp_orig', 'c1・c2', QP, None, None),
    ('col_orig', 'c1・c3', COL, None, None),
    # c1 car_check
    ('cc_no_dm', 'c1', CC, "    dm = data_missing_of(log, raw_dir, L.get('problems'))      # 2026-10-27 fix5 c1", "    dm = None  #"),
    ('cc_no_dm_static', 'c1', CC, "        dm = data_missing_of(log, raw_dir)          # c1", "        dm = None  #"),
    ('cc_judge_no_item', 'c1', CC, "        items.insert(0, ('data_missing', 'missing', missing_text(dm)))", "        pass"),
    ('cc_code_ignores_missing', 'c1', CC, "(3 if ('insufficient' in st or 'missing' in st) else", "(3 if ('insufficient' in st) else"),
    ('cc_missing_beats_fail', 'c1', CC, "    code = 2 if 'fail' in st else (3 if", "    code = 3 if 'missing' in st else 2 if 'fail' in st else (3 if"),
    ('cc_latest_no_stop', 'c1', CC, "    if args.latest and res.get('data_missing'):", "    if False:"),
    ('cc_coll_line', 'c1', CC, "        if dm:                  # 2026-10-27 fix5 c1: 記録が欠けた本", "        if False:                  #"),
    ('cc_none_kind_ignored', 'c1', CC, "            if 'ありません' in why:", "            if False:"),
    ('cc_ai_cmd_counted', 'c1', CC, "MISSING_SENSORS = ('VEHICLE', 'IMU', 'CAM_FRONT')     # 無ければ欠け",
     "MISSING_SENSORS = ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT')     # 無ければ欠け"),
    ('cc_static_no_zero', 'c1', CC, "                    if p.stat().st_size == 0:\n                        items.append([s, n, 'zero'])",
     "                    if False:\n                        items.append([s, n, 'zero'])"),
    ('cc_error_text', 'c1', CC, "        if res.get('data_missing'):             # c1(あるときだけ)", "        if False:"),
    # c1 standing_3lap・by_params・collect
    ('s3_no_missing', 'c1', S3, "    if mz:                               # 2026-10-27 fix5 c1", "    if False:  #"),
    ('s3_no_missing_line', 'c1', S3, "    if mz:\n        out += ", "    if False:\n        out += "),
    ('s3_ai_cmd_counted', 'c1', S3, "MISSING_SENSORS = ('VEHICLE', 'IMU', 'CAM_FRONT')   # 無ければ欠け",
     "MISSING_SENSORS = ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT')   # 無ければ欠け"),
    ('qp_no_missing_flag', 'c1', QP, "    if res.get('missing_npz'):", "    if False:"),
    ('col_no_problem_line', 'c1', COL, "            if not pr:\n                continue", "            continue"),
    # c2
    ('s3_c2_still_counted', 'c2', S3, "                s['midrun'] = True\n                s['counted'] = False", "                s['midrun'] = True"),
    ('s3_c2_after_end', 'c2', S3, "float(m['t_s']) < float(e3)", "float(m['t_s']) > float(e3)"),
    ('s3_c2_unknown_time', 'c2', S3, "if any(m.get('t_s') is None or float(m['t_s']) < float(e3)",
     "if any(m.get('t_s') is not None and float(m['t_s']) < float(e3)"),
    ('s3_c2_no_table', 'c2', S3, "            if s.get('midrun'):          # 2026-10-27 fix5 c2", "            if False:  #"),
    ('s3_c2_no_note', 'c2', S3, "    if any(s.get('midrun') for r in results for s in r.get('standing') or []):", "    if False:"),
    ('qp_c2_no_flag', 'c2', QP, "    if exm:", "    if False:"),
    ('qp_c2_wrong_reason', 'c2', QP, "            and (s.get('coll_before') or not s.get('midrun'))]", "            ]"),
    # c3
    ('col_c3_no_split', 'c3', COL, "        if len(fs) >= 2 and any(on for _, on in gs):", "        if False:"),
    ('col_c3_notice_only', 'c3', COL, "                    g['compare']['checks']['one_form'] = False\n                    g['compare']['all_ok'] = False",
     "                    pass"),
    ('col_c3_no_on_needed', 'c3', COL, "        if len(fs) >= 2 and any(on for _, on in gs):", "        if len(fs) >= 2:"),
    ('col_c3_trials_count', 'c3', COL, "        if allr or k in ur:", "        if True:"),
    ('col_c3_by_gain', 'c3', COL, "            forms.setdefault(k[:2], [])", "            forms.setdefault(k[:3], [])"),
    ('col_c3_no_line', 'c3', COL, "        if g.get('form_split'):        # 2026-10-27 fix5 c3", "        if False:  #"),
]


def fix_paths(rel, src):
    if rel == COL:
        src = src.replace("sys.path.insert(0, '/home/jetson/jetformer/tools')", "sys.path.insert(0, %r)" % str(WORK / 'tools'))
    return src


def put(rel, old=None, new=None, orig=False):
    """写しの rel を本物(+ 壊し)か、直す前の写しで書き直す"""
    if orig:
        src = (HERE / 'before' / (Path(rel).name + '.orig')).read_text(encoding='utf-8')
    else:
        src = (JF / rel).read_text(encoding='utf-8')
        if old is not None:
            assert src.count(old) == 1, (rel, old, src.count(old))
            src = src.replace(old, new)
    (WORK / rel).write_text(fix_paths(rel, src), encoding='utf-8')


def build():
    if WORK.exists():
        shutil.rmtree(WORK)
    for d in ('tools', 'tests/fixtures', 'experiments/20261026_fix/diag_launch'):
        (WORK / d).mkdir(parents=True)
    os.symlink(JF / 'jetformer', WORK / 'jetformer')
    os.symlink(JF / 'data', WORK / 'data')           # 読むだけ(bench_candidates.json の候補の名前)
    for p in (JF / 'tools').glob('*.py'):
        shutil.copy(p, WORK / 'tools' / p.name)
    for n in ('test_fix5_tools.py', 'test_fix3_r1_2.py', 'test_fix3_r3_2_tools.py', 'test_lap_anchor.py', 'test_standing_3lap.py'):
        shutil.copy(JF / 'tests' / n, WORK / 'tests' / n)
    shutil.copytree(JF / 'tests' / 'fixtures' / 'fix5_tools_before', WORK / 'tests' / 'fixtures' / 'fix5_tools_before')
    put(COL)


def pytest(tag):
    BASET.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::DeprecationWarning',
                        '--basetemp', str(BASET / tag), '--rootdir', str(WORK)] + T, cwd=str(WORK), capture_output=True,
                       text=True, env=env, timeout=900)
    shutil.rmtree(BASET / tag, ignore_errors=True)
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    fails = [x.split(' - ')[0].split('::')[-1] for x in lines if x.startswith(('FAILED', 'ERROR'))]
    return r.returncode, (lines[-1] if lines else '?'), fails


try:
    build()
    rc, last, fails = pytest('base')
    print('base %s | %s | %s' % (rc, last, ' / '.join(fails)), flush=True)
    for tag, fid, rel, old, new in MUT:
        if ONLY and tag not in ONLY:
            continue
        put(rel, old, new, orig=old is None)
        rc, last, fails = pytest(tag)
        print('%-26s %-8s %s | %s | %s' % (tag, fid, 'SURVIVED(試験が通った)' if rc == 0 else 'killed', last,
                                          ' / '.join(fails[:6]) + (' …' if len(fails) > 6 else '')), flush=True)
        put(rel)
finally:
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(BASET, ignore_errors=True)
