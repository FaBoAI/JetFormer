"""fix8 ui r1: わざと壊すと tests/test_ui_fix8_r1.py が落ちるか(本物のファイルは変えない。写しを 1 か所ずつ壊して JF_UI81_APP・
JF_UI81_RUNS・JF_UI81_CSS で渡す)。数えるのは動きの試験(形を見る 5 本 = 直す前の写し・app.js の足した行・style.css の 1 行・
runs_1103 の AST・台本の写し は数えない)。CSS は形でない 1 本(.toast の宣言)と、--firefox のときは Firefox の 1 本も。

    python run_mut.py            # app.js・runs_1103.py・style.css(Firefox なし)。jf-heavy --lock none --mem 1500M
    python run_mut.py --firefox  # style.css の壊しを Firefox の 1 本でも。jf-heavy --lock big --mem 3000M

出力 out_mut.txt(--firefox は out_mut_firefox.txt)。写しと basetemp は最後に消す。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
APP = JF / 'jetformer' / 'web' / 'static' / 'app.js'
CSS = JF / 'jetformer' / 'web' / 'static' / 'style.css'
RUNS = JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
SHAPE = 'not before_copies and not only_added and not only_the_toast and not only_round_options and not sim_copy'

APP_MUTS = {
    'A01 loadDrive で paramsLoaded を下ろさない(番号だけ進める)': (
        'if (drIdle()) { state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }',
        'if (drIdle()) { state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }'),
    'A02 走行中・token の画面でも下ろす(drIdle を見ない)': (
        'if (drIdle()) { state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }',
        '{ state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }'),
    'A03 戻る前に出した問い合わせの返事も「読み直した」に数える(番号を見ない)': (
        '  if (st && gen !== state.drive.drTabGen && drIdle()) state.drive.paramsLoaded = false;\n', ''),
    'A04 戻ったときに番号を進めない': (
        'state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }', 'state.drive.drTabGen = (state.drive.drTabGen || 0); }'),
    'A05 下ろすときに lostAt も付ける(文が「つながっていません」になる)': (
        'if (drIdle()) { state.drive.paramsLoaded = false;', 'if (drIdle()) { state.drive.paramsLoaded = false; state.drive.lostAt = Date.now();'),
    'A06 下ろさずに一覧の GET の前に pollDrive を出す(反証の別案。問い合わせが増える)': (
        'if (drIdle()) { state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }',
        'if (drIdle()) { pollDrive(); }'),
    'A07 番号を読まない(いつも違う = 同期しても下ろしたまま)': (
        '  const gen = state.drive.drTabGen;   // 走行タブに戻った番号', '  let gen;   // 走行タブに戻った番号'),
    'A08 番号が違うときに下ろさず上げる': (
        'if (st && gen !== state.drive.drTabGen && drIdle()) state.drive.paramsLoaded = false;',
        'if (st && gen !== state.drive.drTabGen && drIdle()) state.drive.paramsLoaded = true;'),
}
RUNS_MUTS = {
    'R01 回の最初の本と比べる(直す前)': ('        f = max(rs, key=lambda x: len(opt_on(x)))', '        f = rs[0]'),
    'R02 ON がいちばん少ない本と比べる': ('        f = max(rs, key=lambda x: len(opt_on(x)))', '        f = min(rs, key=lambda x: len(opt_on(x)))'),
    'R03 同じ数なら後の本': ('        f = max(rs, key=lambda x: len(opt_on(x)))', '        f = max(reversed(rs), key=lambda x: len(opt_on(x)))'),
    'R04 補助なしを数えない': ('        f = max(rs, key=lambda x: len(opt_on(x)))',
                         "        f = max(rs, key=lambda x: len([k for k in opt_on(x) if k != 'launch_no_kick']))"),
    'R05 印の中の ON/OFF を逆に書く': ("'ON' if k in a else 'OFF', f['run_id'], 'ON' if k in b else 'OFF')",
                               "'ON' if k in b else 'OFF', f['run_id'], 'ON' if k in a else 'OFF')"),
    'R06 回をまたいで 1 つの相手と比べる': ("rounds.setdefault(str(r['run_id']).split('-')[0], [])", "rounds.setdefault('1', [])"),
}
CSS_MUTS = {
    'C01 直す前の .toast(幅 auto)': ('width: max-content; max-width: 90%; max-width: min(90%, 960px);', 'max-width: 90%;'),
    'C02 max-width なし(長い文が画面の外へ)': ('width: max-content; max-width: 90%; max-width: min(90%, 960px);', 'width: max-content;'),
    'C03 left/right 5%(短い文も画面いっぱい)': (
        'left: 50%; transform: translateX(-50%);', 'left: 5%; right: 5%; transform: none;'),
    'C04 width: max-content を付けない': ('width: max-content; max-width: 90%;', 'max-width: 90%;'),
    'C05 960 px の上限なし(大きな画面で 1 行が長い)': ('max-width: 90%; max-width: min(90%, 960px);', 'max-width: 90%;'),
}


def run(env_key, path, kexpr, firefox, tag):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{env_key: str(path)})
    if firefox:
        env['JF_UI_FIREFOX'] = '1'
    bt = HERE / 'bt' / tag
    bt.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'tmp_path_retention_policy=failed', '--basetemp', str(bt),
                        str(JF / 'tests' / 'test_ui_fix8_r1.py'), '-k', kexpr, '-rf'], cwd=str(JF), env=env, capture_output=True,
                       text=True, timeout=1800)
    shutil.rmtree(bt, ignore_errors=True)
    tail = r.stdout.strip().splitlines()
    failed = sorted({x.split('::')[1].split(' ')[0] for x in tail if x.startswith('FAILED')})
    return r.returncode, tail[-1] if tail else r.stderr[-300:], failed


def main():
    firefox = '--firefox' in sys.argv
    work = HERE / 'copies'
    work.mkdir(exist_ok=True)
    lines = []
    groups = [('JF_UI81_CSS', CSS, CSS_MUTS, 'firefox')] if firefox else [
        ('JF_UI81_APP', APP, APP_MUTS, SHAPE + ' and not runs and not toast'), ('JF_UI81_RUNS', RUNS, RUNS_MUTS, 'runs and not only_round_options'),
        ('JF_UI81_CSS', CSS, CSS_MUTS, 'toast_rule and not only_the')]
    # 壊さないとき(どの組も通る)
    for env_key, src, _, kexpr in groups:
        rc, last, failed = run(env_key, src, kexpr, firefox, 'clean')
        lines.append('壊さない(%s = 本物・-k %s): %s %s' % (env_key, kexpr, last, failed))
    missed = 0
    for env_key, src, muts, kexpr in groups:
        text = src.read_text(encoding='utf-8')
        for i, (name, (a, b)) in enumerate(muts.items()):
            assert text.count(a) == 1, (name, text.count(a))
            p = work / ('%s_%d%s' % (env_key, i, src.suffix))
            p.write_text(text.replace(a, b), encoding='utf-8')
            rc, last, failed = run(env_key, p, kexpr, firefox, 'm%d' % i)
            ok = rc != 0
            missed += not ok
            lines.append('%s: %s(%s)%s' % (name, '落ちた' if ok else '見逃し', last, (' ' + ', '.join(failed)) if failed else ''))
            print(lines[-1], flush=True)
    shutil.rmtree(work)
    shutil.rmtree(HERE / 'bt', ignore_errors=True)
    n = sum(len(m) for _, _, m, _ in groups)
    lines.insert(len(groups), '見逃し %d / %d' % (missed, n))
    (HERE / ('out_mut_firefox.txt' if firefox else 'out_mut.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
