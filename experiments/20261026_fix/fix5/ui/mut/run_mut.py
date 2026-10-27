"""わざと壊す(1 つずつ)→ tests/test_ui_fix5.py を流し、落ちた試験を出す。本物の app.js・server.py・runs_1103.py は変えない(写しを壊す)"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
APP = (JF / 'jetformer/web/static/app.js').read_text(encoding='utf-8')
RUNS = (JF / 'experiments/20261026_fix/plan/runs_1103.py').read_text(encoding='utf-8')
M = {   # 名前: (ファイル, 元の文, 壊した文)
    'a_no_drLost': ('app', "catch (e) { st = null; drLost(); }", "catch (e) { st = null; }"),
    'b_no_boot_at_in_body': ('app', "boot_at: state.drive.bootAt ?? null,", "boot_at: null,"),
    'b_no_resync_after_refusal': ('app', "catch (e) { toast(e.message, true); pollDrive(); }", "catch (e) { toast(e.message, true); }"),
    'a_old_message_always': ('app', "state.drive.lostAt && state.drive.synced ?", "state.drive.lostAt ?"),
    't2_no_drPostN_keep': ('app', "!!pend[k] || (drPostN[k] || 0) > 0 ||", "!!pend[k] ||"),
    't2_no_drPostN_count': ('app', "for (const k of sent) drPostN[k] = (drPostN[k] || 0) + 1;", ""),
    't3_no_rev_line': ('app', "const trimLine = trimGuardStartLines(state.drive.st) + revStartLine();",
                       "const trimLine = trimGuardStartLines(state.drive.st);"),
    'version_8': ('app', "const body = { client: 9,", "const body = { client: 8,"),
    'server_no_check': ('server', None, None),
    'runs_no_rev_keys': ('runs', "        'zone_caps', 'rev_s', 'rev_us')", "        'zone_caps')"),
    'runs_no_round_rev': ('runs', "    round_rev(rows)\n", ""),
    'runs_no_rev_line': ('runs', "    rl = rev_line(rows)\n", "    rl = None\n"),
}
only = sys.argv[1:]
res = {}
for name, (what, a, b) in M.items():
    if only and name not in only:
        continue
    tmp = Path(tempfile.mkdtemp(prefix='mut_', dir=str(HERE)))
    env = dict(os.environ, PYTHONPATH='%s:%s' % (HERE, JF / 'tests'))
    if what == 'app':
        assert APP.count(a) == 1, name
        (tmp / 'app.js').write_text(APP.replace(a, b), encoding='utf-8')
        shutil.copy(JF / 'jetformer/web/static/index.html', tmp / 'index.html')
        env['MUT_STATIC'] = str(tmp)
    elif what == 'runs':
        assert RUNS.count(a) == 1, name
        (tmp / 'runs_1103.py').write_text(RUNS.replace(a, b), encoding='utf-8')
        env['MUT_RUNS'] = str(tmp / 'runs_1103.py')
    else:
        env['MUT_SERVER_NOCHECK'] = '1'
    r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p', 'mutplugin', '--basetemp', str(tmp / 'bt'),
                        '-rf', str(JF / 'tests/test_ui_fix5.py')], cwd=str(JF), env=env, capture_output=True, text=True)
    failed = [x.split('::')[1].split(' ')[0] for x in r.stdout.splitlines() if x.startswith('FAILED')]
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    res[name] = (failed, last)
    print('%-28s %s\n    %s' % (name, last, ' '.join(failed)), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
bad = [n for n, (f, _) in res.items() if not f]
print('落ちなかった壊し方:', bad or 'なし')
