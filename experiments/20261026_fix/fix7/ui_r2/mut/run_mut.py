"""直す 7-2 回目(画面の係)の「わざと壊す」台本。今の app.js の写しを 1 か所ずつ壊し(本物は変えない)、JF_UI72_APP で
tests/test_ui_fix7_r2.py の動きの試験(node で画面を流すもの)を流して、落ちる試験の数を数える。形を見るだけの 3 本
(test_before_copy_…・test_fill_is_…・test_sim_copy_…)は数えない(どの壊し方でも落ちるので)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix7/ui_r2/mut/run_mut.py <一時の場所>
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
APP = ROOT / 'jetformer' / 'web' / 'static' / 'app.js'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TMP = Path(sys.argv[1])

R1 = "      if (!synced && seq === state.drive.startSeq && (state.drive.bootToasts || 0) === nBoot) state.drive.startRefused = { seq, m, bootRefusal };\n"
RS = "  if (rs && rs.seq === state.drive.startSeq) toast(startRefusalText(st, rs.m, rs.bootRefusal), true);\n"
RT = "  if (!d.modelsMiss || d.modelsBusy > 0 || !drIdle() || state.tab !== 'drive' || !(d.modelsRetry > 0)) return;\n"
RL = "  state.drive.models = models; state.drive.modelsMiss = false;\n  fillDriveModels();\n  loadBench();\n}\n"
MUTANTS = {
    # w2-ui-w2-ui-r72-ui-1(▶ の断りの文の出し直し)
    '01 同期が通らなくても覚えない': [(R1, '')],
    '02 起き直しの 409 のときだけ覚える': [(R1, R1.replace('=== nBoot)', '=== nBoot && bootRefusal)'))],
    '03 pollDrive がいつも「同期できた」': [('  return !!st;', '  return true;')],
    '04 知らせの無い同期で覚えた文を捨てない': [('  const rs = state.drive.startRefused;\n  state.drive.startRefused = null;\n',
                                          '  const rs = state.drive.startRefused;\n'), (RS, RS + '  state.drive.startRefused = null;\n')],
    '05 ▶ の番号を見ない': [(RS, RS.replace('if (rs && rs.seq === state.drive.startSeq)', 'if (rs)'))],
    '06 知らせの前に出し直す': [('  toast(msg, true);\n' + RS, RS + '  toast(msg, true);\n')],
    '07 出し直しに「始まっていません」を付けない': [(RS, RS.replace('startRefusalText(st,', 'startRefusalText(null,'))],
    '08 走行中でも「始まっていません」を付ける': [("  return (st && !st.running ? '走行は始まっていません", "  return (true ? '走行は始まっていません")],
    '09 409 の出し直しにも知らせを付ける': [(RS, RS.replace('rs.bootRefusal)', 'false)'))],
    # w2-ui-w2-ui-r72-ui-2(一覧の読み直し・文)
    '10 読み直さない': [('  if (--d.modelsRetry === 0) reloadDriveModels();', '  --d.modelsRetry;')],
    '11 同期のたびに読み直す': [(RT + '  if (--d.modelsRetry === 0) reloadDriveModels();', RT.replace(' || !(d.modelsRetry > 0)', '') + '  reloadDriveModels();')],
    '12 失敗のすぐあとの同期で読み直す': [('state.drive.modelsRetry = 2;', 'state.drive.modelsRetry = 1;')],
    '13 一覧の GET を待っている間も読み直す': [(RT, RT.replace(' || d.modelsBusy > 0', ''))],
    '14 走行中も読み直す': [(RT, RT.replace(' || !drIdle()', ''))],
    '15 走行タブでなくても読み直す': [(RT, RT.replace(" || state.tab !== 'drive'", ''))],
    '16 基準の設定の文が前のまま': [('toast(drModelsMissing() ? DR_MODELS_MISS : `モデル ${run}', 'toast(false ? DR_MODELS_MISS : `モデル ${run}')],
    '17 ▶ の文が前のまま': [("toast(drModelsMissing() ? DR_MODELS_MISS : 'モデルを選んでください'", "toast(false ? DR_MODELS_MISS : 'モデルを選んでください'")],
    '18 走行中も文を変える': [('const drModelsMissing = () => !!state.drive.modelsMiss && drIdle();', 'const drModelsMissing = () => !!state.drive.modelsMiss;')],
    '19 loadDrive が通っても印を消さない': [('  state.drive.modelsBusy--; state.drive.modelsMiss = false;\n  clearTimeout(late);', '  state.drive.modelsBusy--;\n  clearTimeout(late);')],
    '20 読み直しが通っても印を消さない': [(RL, RL.replace(' state.drive.modelsMiss = false;', ''))],
    '21 選ぶ欄に短い文を出さない': [('  if (!sel.options.length) sel.innerHTML = ', '  if (false) sel.innerHTML = ')],
    '22 前の一覧を短い文で消す': [('  if (!sel.options.length) sel.innerHTML = ', '  sel.innerHTML = ')],
    '23 読み直しでベンチマークを読まない': [(RL, RL.replace('  loadBench();\n', ''))],
    '24 読み直しで pollDrive も呼ぶ': [(RL, RL.replace('  loadBench();\n', '  loadBench();\n  pollDrive();\n'))],
    '25 失敗のとき待っている数を減らさない': [('    state.drive.modelsBusy--; clearTimeout(late);', '    clearTimeout(late);')],
    '26 失敗・遅れの印を付けない': [('const lost = () => { state.drive.modelsMiss = true; state.drive.modelsRetry = 2; if', 'const lost = () => { if')],
}
SHAPE = 'not test_before_copy and not test_fill_is and not test_sim_copy'


def run(env_app, tmpdir, k=SHAPE):
    r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--basetemp={tmpdir}', 'tests/test_ui_fix7_r2.py', '-k', k],
                       cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, JF_UI72_APP=str(env_app)), timeout=1800)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    failed = int(re.search(r'(\d+) failed', tail).group(1)) if 'failed' in tail else 0
    passed = int(re.search(r'(\d+) passed', tail).group(1)) if 'passed' in tail else 0
    names = sorted({m.group(1) for m in re.finditer(r'FAILED tests/test_ui_fix7_r2.py::(\w+)', r.stdout)})
    return failed, passed, names


src = APP.read_text(encoding='utf-8')
TMP.mkdir(parents=True, exist_ok=True)
f0, p0, _ = run(APP, TMP / 'none')
print(f'(壊さない) failed={f0} passed={p0}', flush=True)
missed = 0
for i, (name, edits) in enumerate(MUTANTS.items()):
    s = src
    for old, new in edits:
        assert s.count(old) == 1, (name, old[:60], s.count(old))
        s = s.replace(old, new)
    d = TMP / f'm{i:02d}'
    d.mkdir(exist_ok=True)
    (d / 'app.js').write_text(s, encoding='utf-8')
    f, p, names = run(d / 'app.js', d / 'bt')
    missed += f == 0
    print(f'{name}: failed={f} passed={p} {"見逃し" if f == 0 else ""} {names}', flush=True)
print(f'見逃し {missed} / {len(MUTANTS)}')
