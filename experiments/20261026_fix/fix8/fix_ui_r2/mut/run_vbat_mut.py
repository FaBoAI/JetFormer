# -*- coding: utf-8 -*-
"""わざと壊す(直す 8-2 回目・係 ui の 2 件目: tests/test_vbat_ui.py の天井の確認の文の期待)。
test_vbat_ui の Firefox の試験(test_firefox_drive_tab_vbat_and_switches)を、今の版と直す前の写し(tests/fixtures/fix8_ui_r2_before/
test_vbat_ui.py)で、次の形で流す(本物の app.js・drive.py は変えない。app.js を壊すときは static の写しを作り、試験の STATIC をそこに向ける。
既定を変えるときは drive.DEFAULT_PARAMS をこの台本の中だけで書き換える):
  N0 そのまま / V01 既定の左を 4.2 に(app.js はそのまま = 画面はサーバーの値を出す)/ V02 app.js の ceilSpecText が全体と左を入れ違える /
  V03 app.js の ceilSpecText が toFixed(2) / V04 app.js の speedSpec が左にサーバーの値を使わず控えの 4.0(SPEED_FALLBACK)を出す /
  V05 既定の左を 4.25 に(ちょうど半分: 画面は toFixed(1) で 4.3)
  cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big --mem 3000M -- /home/jetson/miniconda/envs/jetracer/bin/python -B \\
      experiments/20261026_fix/fix8/fix_ui_r2/mut/run_vbat_mut.py
出力 out_vbat_mut.txt
"""
import importlib.util
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
HERE = Path(__file__).resolve().parent
SCRATCH = '/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad'
from jetformer import drive  # noqa: E402

NOW = ROOT / 'tests' / 'test_vbat_ui.py'
OLD = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r2_before' / 'test_vbat_ui.py'
CEIL = 'const ceilSpecText = (sp) => `全体 ${sp.all.toFixed(1)}・左旋回中 ${sp.left.toFixed(1)} m/s`;'
LEFT = 'all: num(c.all_mps, p.ceil_all_mps, F.all), left: num(c.left_mps, p.ceil_left_mps, F.left),'
APP_MUTS = {
    'V02': (CEIL, 'const ceilSpecText = (sp) => `全体 ${sp.left.toFixed(1)}・左旋回中 ${sp.all.toFixed(1)} m/s`;'),
    'V03': (CEIL, 'const ceilSpecText = (sp) => `全体 ${sp.all.toFixed(2)}・左旋回中 ${sp.left.toFixed(2)} m/s`;'),
    'V04': (LEFT, 'all: num(c.all_mps, p.ceil_all_mps, F.all), left: num(F.left),'),
}
CASES = [('N0 そのまま', None, None), ('V01 既定の左 4.2', None, 4.2), ('V02 app.js 全体と左を入れ違える', 'V02', None),
         ('V03 app.js toFixed(2)', 'V03', None), ('V04 app.js 左は控えの 4.0', 'V04', None), ('V05 既定の左 4.25', None, 4.25)]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_one(mod_path, tag, app_mut, left):
    m = load(mod_path, 'vbat_%s' % tag)
    m.ROOT = str(ROOT)                                   # 写し(tests/fixtures/…)は自分の場所から ROOT を作るので、本物の場所に向ける
    m.STATIC = str(ROOT / 'jetformer' / 'web' / 'static')
    saved_left = drive.DEFAULT_PARAMS['ceil_left_mps']
    tmp = None
    try:
        if app_mut:
            tmp = tempfile.mkdtemp(prefix='vbat-mut-', dir=SCRATCH)
            static = Path(tmp) / 'static'
            shutil.copytree(m.STATIC, static)
            a, b = APP_MUTS[app_mut]
            js = (static / 'app.js').read_text(encoding='utf-8')
            assert js.count(a) == 1, app_mut
            (static / 'app.js').write_text(js.replace(a, b), encoding='utf-8')
            m.STATIC = str(static)
        if left is not None:
            drive.DEFAULT_PARAMS['ceil_left_mps'] = left
        m.test_firefox_drive_tab_vbat_and_switches()
        return 'passed', ''
    except AssertionError:
        tb = traceback.extract_tb(sys.exc_info()[2])
        fr = [f for f in tb if f.filename == str(mod_path)][-1]
        return 'failed', '%s 行 %d: %s' % (Path(mod_path).name, fr.lineno, (fr.line or '')[:110])
    except Exception as e:      # noqa: BLE001
        return 'error', repr(e)[:200]
    finally:
        drive.DEFAULT_PARAMS['ceil_left_mps'] = saved_left
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    lines = ['# test_vbat_ui の Firefox の試験を、今の版(期待を既定から作る)と直す前の写し(固定の「左旋回中 4.0」)で',
             '# 列: 形 | 今の版 | 直す前の写し']
    for i, (name, app_mut, left) in enumerate(CASES):
        a = run_one(NOW, 'now%d' % i, app_mut, left)
        b = run_one(OLD, 'old%d' % i, app_mut, left)
        lines.append('%s | 今 %s %s | 前 %s %s' % (name, a[0], a[1], b[0], b[1]))
        print(lines[-1], flush=True)
    (HERE / 'out_vbat_mut.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
