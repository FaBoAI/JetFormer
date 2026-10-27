#!/usr/bin/env python3
"""係 tools(fix3、2026-10-26 夜): 「わざと壊すと落ちる」の確かめ。本物のファイルは変えず、写しの場所に 1 か所ずつ壊しを入れて
(または直す前の写し before/*.orig に戻して)対象の試験だけを流す。
    python3 mutate_fix3.py <写しの場所> <basetemp の親> [壊しの名前 …]
写しの場所・basetemp はこの台本の最後で消す(どちらも作業用の一時の場所。本物の data/raw・labels・報告には書かない)。
bench_nightly.sh の試験は tmp の中だけで回る(試験の sandbox が本物の ~/jetformer_reports を tmp に置き換える)。"""
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
assert '/tmp' in str(WORK) or 'tmp_tools' in str(WORK), WORK          # 消してよい場所だけ
COL = 'experiments/20261026_fix/diag_launch/collect.py'
T_COL3 = 'experiments/20261026_fix/fix3/tools/test_fix3_collect.py'
T_COL2 = 'experiments/20261026_fix/fix2/test_tools_collect.py'
T_CRIT = 'experiments/20261026_fix/fix2/critic/test_critic_tools.py'
T_CF = ['tests/test_course_fit_fields.py', 'tests/test_broken_records.py']
T_BN = ['tests/test_bench_nightly_out.py']
T_S3 = ['tests/test_tools_fix3.py', 'tests/test_standing_3lap.py', 'tests/test_quick_eval_by_params.py']
T_C = [T_COL3, T_COL2, T_CRIT]
CF, BN, S3, QP = 'tools/course_fit.py', 'tools/bench_nightly.sh', 'tools/standing_3lap.py', 'tools/quick_eval_by_params.py'

# (名前, 指摘, ファイル, 前, 後, 試験)。前 = None なら直す前の写し(before/<名前>.orig)にまるごと戻す
MUT = [
    # 道具がちゃんと試験を読むか(効く壊し)
    ('control_s3', '-', S3, "MOVE_ERPM = 400.0 ", "MOVE_ERPM = 4000.0 ", T_S3),
    ('control_col', '-', COL, "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.5,", T_C),
    # code:F1
    ('cf_orig', 'code:F1', CF, None, None, T_CF),
    ('cf_all_fields', 'code:F1', CF, "return {k: z[k] for k in USED_FIELDS[name] if k in z.files}", "return {k: z[k] for k in z.files}", T_CF),
    ('cf_no_vehicle_select', 'code:F1', CF, "            if self.ai_bad:\n", "            if False:\n", T_CF),
    # code:F2(写しが JF_REPORTS を読まない)
    ('bn_no_jf_reports', 'code:F2', BN, "REPORTS_ROOT=${JF_REPORTS:-/home/jetson/jetformer_reports}", "REPORTS_ROOT=/home/jetson/jetformer_reports", T_BN),
    # code:F3〜F5
    ('bn_orig', 'code:F3-F5', BN, None, None, T_BN),
    ('bn_no_seed', 'code:F3', BN, 'if [ $REPORT_DEF -eq 1 ] && { [ ! -e "$REPORT/deviations.md" ]', 'if false && { [ ! -e "$REPORT/deviations.md" ]', T_BN),
    ('bn_seed_is_busy', 'code:F3', BN, ' && ! seed_only "$REPORT" && BUSY+=("$REPORT")', ' && BUSY+=("$REPORT")', T_BN),
    ('bn_no_early_check', 'code:F4', BN, 'if [ ${#LOGS[@]} -eq 0 ] && [ $OUT_DEF -eq 1 ]; then', 'if false; then', T_BN),
    ('bn_dir_is_busy', 'code:F4', BN, '[ $OUT_DEF -eq 1 ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && BUSY+=("$OUT")',
     '[ $OUT_DEF -eq 1 ] && [ -d "$OUT" ] && BUSY+=("$OUT")', T_BN),
    ('bn_rc3_ignored', 'code:F5', BN, 'if [ $RC3 -eq 0 ]; then CR=', 'if true; then CR=', T_BN),
    ('bn_rc4_ignored', 'code:F5', BN, 'if [ $RC4 -eq 0 ]; then BR=', 'if true; then BR=', T_BN),
    # tools:F4 の残り
    ('col_orig', 'tools:F4・F6・F9', COL, None, None, T_C),
    ('col_mixed_all_on', 'tools:F4', COL, "            on_use = agg('on', use)\n", "            on_use = on\n", T_C),
    ('col_last_max_value', 'tools:F4', COL, "last = max(rows_on, key=lambda r: (r['log'], r['t_ai']))", "last = max(rows_on, key=lambda r: float(us_of(r)))", T_C),
    ('col_last_min_value', 'tools:F4', COL, "last = max(rows_on, key=lambda r: (r['log'], r['t_ai']))", "last = min(rows_on, key=lambda r: float(us_of(r)))", T_C),
    ('col_coll_used_only', 'tools:F4', COL, "'first_lap_collisions_0': on['collisions_first_lap'] == 0,", "'first_lap_collisions_0': on_use['collisions_first_lap'] == 0,", T_C),
    # tools:F5
    ('qp_orig', 'tools:F5・F6・F7', QP, None, None, T_S3),
    ('qp_doubt_standing_counted', 'tools:F5', QP, "stand = [s for s in starts if s3.counted(s) and not doubtful(s)]", "stand = [s for s in starts if s3.counted(s)]", T_S3),
    ('qp_doubt_window_counted', 'tools:F5', QP, "wins = [w['total_s'] for r in rs for w in r.get('windows') or [] if not w.get('suspect')]",
     "wins = [w['total_s'] for r in rs for w in r.get('windows') or []]", T_S3),
    ('qp_no_standing_flag', 'tools:F5', QP, "    if chk:\n        out.append('確かめる(止まってから 3 周", "    if False:\n        out.append('確かめる(止まってから 3 周", T_S3),
    ('qp_no_window_flag', 'tools:F5', QP, "    if sw:\n", "    if False:\n", T_S3),
    # tools:F7
    ('qp_per_record_launch', 'tools:F7', QP, "if 'launch' in ignore or st_ == rec:", "if True:", T_S3),
    ('qp_no_mixed_flag', 'tools:F7', QP, "if 'on' in st_ and 'on_not_started' in st_:", "if False:", T_S3),
    # tools:F6 standing_3lap(反証 1 回目で試験が通った 9 か所)
    ('s3_frame_gap', 'tools:F6', S3, "if _max_gap(t, t_ai, e3) > le.GAP_S:", "if False:", T_S3),
    ('s3_vehicle_gap', 'tools:F6', S3, "if _max_gap(vt, t_move, e3) > le.GAP_S:", "if False:", T_S3),
    ('s3_line_lap_5_16', 'tools:F6', S3, "if not all(LAP_MIN_S <= x <= LAP_MAX_S for x in row['laps_line_s']):", "if False:", T_S3),
    ('s3_anchor_chain', 'tools:F6', S3, "if len(A_) == 4 or not (LAP_MIN_S <= x - A_[-1] <= LAP_MAX_S):", "if len(A_) == 4:", T_S3),
    ('s3_to_r1_15s', 'tools:F6', S3, " or nxt[0] - t_ai > MAX_TO_R1_S:", ":", T_S3),
    ('s3_cont_s', 'tools:F6', S3, "if not all(abs(w[j]['b'] - w[j + 1]['a']) <= CONT_S for j in range(2)):", "if False:", T_S3),
    ('s3_dry_run', 'tools:F6', S3, "ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)", "ai = lab['ai'].astype(bool)", T_S3),
    ('s3_dist_dpsi', 'tools:F6', S3, "if DIST_DPSI[0] <= dpsi <= DIST_DPSI[1] and b - a < LAP_MAX_S:", "if b - a < LAP_MAX_S:", T_S3),
    ('s3_best_max', 'tools:F6', S3, "return min(v, key=lambda s: s['from_move_s']) if v else None", "return max(v, key=lambda s: s['from_move_s']) if v else None", T_S3),
    # tools:F6 quick_eval_by_params(8 か所)
    ('qp_gain', 'tools:F6', QP, "key.append(('gain', g if g is not None else -1))", "key.append(('gain', 0))", T_S3),
    ('qp_steer', 'tools:F6', QP, "key.append(('steer', sg))", "key.append(('steer', 1.0))", T_S3),
    ('qp_zone', 'tools:F6', QP, "key.append(('zone', z))", "key.append(('zone', 0))", T_S3),
    ('qp_mode', 'tools:F6', QP, "key.append(('mode', m if isinstance(m, int) else -1))", "key.append(('mode', 0))", T_S3),
    ('qp_auto_target', 'tools:F6', QP, "key.append(('auto', (1, _num(S.get('auto_target_mps')) or 0)))", "key.append(('auto', (1, 0)))", T_S3),
    ('qp_launch_us', 'tools:F6', QP, "key.append(('launch', (2, _num(S.get('launch_us')) or 0,", "key.append(('launch', (2, 0,", T_S3),
    ('qp_counted', 'tools:F6', QP, "stand = [s for s in starts if s3.counted(s) and not doubtful(s)]",
     "stand = [s for s in starts if s.get('from_move_s') is not None and not doubtful(s)]", T_S3),
    ('qp_fast9', 'tools:F6', QP, "'fast_laps': sum(1 for x in clean if x < FAST_LAP_S),", "'fast_laps': 0,", T_S3),
    # tools:F6 collect.py(3 か所)と同じ設定の周の鍵(fix3 の直し)
    ('col_on_none', 'tools:F6', COL, "        return False\n\n\ndef gain_key", "        return x is None\n\n\ndef gain_key", T_C),
    ('col_on_s', 'tools:F6', COL, "row['launch_on_s'] = round(float(ts[on][-1] - t0) + 1 / 30.0, 3)", "row['launch_on_s'] = round(float(ts[on][-1] - t0), 3)", T_C),
    ('col_ceil_key', 'tools:F6', COL, "return (d['model_name'], gain_key(S, round(g, 2)), d['session'], _on(S.get('speed_ceiling')),",
     "return (d['model_name'], gain_key(S, round(g, 2)), d['session'], 0,", T_C),
    ('col_left_key_when_off', 'tools:F6', COL, "S.get('ceil_left_mps') if _on(S.get('speed_ceiling')) else None,          # 2026-10-26 fix3",
     "S.get('ceil_left_mps'),          # 2026-10-26 fix3", T_C),
    # tools:F9
    ('col_end_at_a2', 'tools:F9', COL, "                    te, kind = tl, 'line'\n", "                    te, kind = a2, 'line'\n", T_C),
    ('col_a2_in_ai_only', 'tools:F9', COL, "a2 = nx[0] if nx and 5.0 <= nx[0] - a1 <= 16.0 else None",
     "a2 = nx[0] if nx and nx[0] - a1 <= 16.0 and nx[0] <= t1 + 0.2 else None", T_C),
]


def fix_paths(rel, src):
    if rel == COL:
        src = src.replace("sys.path.insert(0, '/home/jetson/jetformer/tools')", "sys.path.insert(0, %r)" % str(WORK / 'tools'))
    if rel in (T_COL3, T_COL2, T_CRIT):
        src = src.replace("JF = Path('/home/jetson/jetformer')", "JF = Path(%r)" % str(WORK))
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
    for d in ('tools', 'tests', 'experiments/20261026_fix/diag_launch', 'experiments/20261026_fix/fix2/critic',
              'experiments/20261026_fix/fix3/tools/before', 'experiments/20261026_fix/plan'):
        (WORK / d).mkdir(parents=True)
    os.symlink(JF / 'jetformer', WORK / 'jetformer')
    os.symlink(JF / 'data', WORK / 'data')           # 読むだけ(test_broken_records は写しを tmp に作る)
    for p in list((JF / 'tools').glob('*.py')) + list((JF / 'tools').glob('*.sh')):
        shutil.copy(p, WORK / 'tools' / p.name)
    for n in ('test_course_fit_fields.py', 'test_broken_records.py', 'test_bench_nightly_out.py', 'test_tools_fix3.py',
              'test_standing_3lap.py', 'test_quick_eval_by_params.py', 'test_lap_anchor.py'):
        shutil.copy(JF / 'tests' / n, WORK / 'tests' / n)
    shutil.copy(JF / 'experiments/20261026_fix/plan/runs_1103.py', WORK / 'experiments/20261026_fix/plan/runs_1103.py')
    shutil.copy(HERE / 'before' / 'collect.py.orig', WORK / 'experiments/20261026_fix/fix3/tools/before/collect.py.orig')
    for rel in (COL, T_COL3, T_COL2, T_CRIT):
        put(rel)


def pytest(tag, tests):
    BASET.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::DeprecationWarning',
                        '--basetemp', str(BASET / tag), '--rootdir', str(WORK)] + tests, cwd=str(WORK), capture_output=True,
                       text=True, env=env, timeout=900)
    shutil.rmtree(BASET / tag, ignore_errors=True)
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    fails = [x.split(' - ')[0].split('::')[-1] for x in lines if x.startswith(('FAILED', 'ERROR'))]
    return r.returncode, (lines[-1] if lines else '?'), fails


try:
    build()
    for tests in (T_CF + T_BN + T_S3 + T_C,):
        rc, last, fails = pytest('base', tests)
        print('base %s | %s | %s' % (rc, last, ' / '.join(fails)), flush=True)
    for tag, fid, rel, old, new, tests in MUT:
        if ONLY and tag not in ONLY:
            continue
        put(rel, old, new, orig=old is None)
        rc, last, fails = pytest(tag, tests)
        print('%-26s %-14s %s | %s | %s' % (tag, fid, 'SURVIVED(試験が通った)' if rc == 0 else 'killed', last,
                                           ' / '.join(fails[:6]) + (' …' if len(fails) > 6 else '')), flush=True)
        put(rel)
finally:
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(BASET, ignore_errors=True)
