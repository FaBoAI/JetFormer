"""fix7 画面の係(w1-ui-r7-ui-2 の確かめ): 前の試験の台本 9 本(tests/fixtures/app_sim_*.js)の runShortTimers だけを「長いタイマーを
捨てない・鳴らす前に消されたものは鳴らさない」形(tests/fixtures/app_sim_fix7.js と同じ)にした写しを keep/ に作り、前の画面の試験
9 本をその写しで流す(keep_plugin.py)。keepdrv は、さらに app_sim_fix6_r2 の ['ntimers'] から pollStatus・pollValues を除いたもの
(test_ui_fix6_r2 は ['ntimers'] の長さの列をそのまま書いて比べているので)。本物の台本・試験・app.js は変えない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix7/ui_r1/harness/run_keep.py > experiments/20261026_fix/fix7/ui_r1/harness/out_keep.txt
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
KEEP = HERE / 'keep'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TMP = JF / 'experiments' / '20261026_fix' / 'fix7' / 'tmp_ui' / 'keep'
NAMES = ['app_sim_round2.js', 'app_sim_fix3.js', 'app_sim_fix3_r1.js', 'app_sim_fix3_r1_2.js', 'app_sim_fix3_r2_1.js', 'app_sim_fix3_r3_1.js',
         'app_sim_fix5.js', 'app_sim_fix6.js', 'app_sim_fix6_r2.js']
TESTS = ['tests/test_bench_ui_round2.py', 'tests/test_ui_fix3.py', 'tests/test_ui_fix3_r1.py', 'tests/test_ui_fix3_r1_2.py',
         'tests/test_ui_fix3_r2_1.py', 'tests/test_ui_fix3_r3_1.py', 'tests/test_ui_fix5.py', 'tests/test_ui_fix6.py', 'tests/test_ui_fix6_r2.py']
OLD = "const runShortTimers = async () => { const t = timers.splice(0); for (const [fn, ms] of t) if (ms <= 200) await fn(); await flush(); };"
NEW = ("const runShortTimers = async () => { for (const x of timers.filter((x) => x[1] <= 200)) { const i = timers.indexOf(x); if (i < 0) continue; "
       "timers.splice(i, 1); await x[0](); } await flush(); };   // fix7 確かめ: 長いタイマーを捨てない(app_sim_fix7.js と同じ)")
NT_OLD = "    else if (op === 'ntimers') { out.push({ op, value: timers.map((x) => x[1]) }); }"
NT_NEW = "    else if (op === 'ntimers') { out.push({ op, value: timers.filter((x) => !['pollStatus', 'pollValues'].includes(x[0].name)).map((x) => x[1]) }); }"


def main():
    KEEP.mkdir(exist_ok=True)
    for n in NAMES:
        s = (JF / 'tests' / 'fixtures' / n).read_text(encoding='utf-8')
        assert s.count(OLD) == 1, n
        k = s.replace(OLD, NEW)
        (KEEP / ('keep_' + n)).write_text(k, encoding='utf-8')
        if n == 'app_sim_fix6_r2.js':
            assert k.count(NT_OLD) == 1
            (KEEP / ('keepdrv_' + n)).write_text(k.replace(NT_OLD, NT_NEW), encoding='utf-8')
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        for v in ('keep', 'keepdrv'):
            r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p', 'keep_plugin', '-rf', '--basetemp', str(TMP / v)] + TESTS,
                               cwd=str(JF), capture_output=True, text=True, timeout=1800,
                               env=dict(os.environ, JF_KEEP_VARIANT=v, PYTHONPATH=str(HERE)))
            print('==== 台本: %s' % v)
            print('\n'.join(x for x in r.stdout.splitlines() if x.startswith('SIM ->')))
            for x in r.stdout.splitlines():
                if x.startswith('FAILED') or x.startswith('ERROR'):
                    print(x[:400])
            print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:])
            if v == 'keep':
                # 落ちたものの中身(どの assert か)
                for m in re.finditer(r'^(E   .*)$', r.stdout, re.M):
                    print('   ', m.group(1)[:300])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
