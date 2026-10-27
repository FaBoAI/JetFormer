"""fix6 画面の係 直す 6-2 回目: わざと壊すと tests/test_ui_fix6_r2.py が落ちるか(本物の app.js は変えない。写しを壊して JF_UI6R2_APP で渡す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix6/ui_r2/mut/run_mut.py > experiments/20261026_fix/fix6/ui_r2/mut/out_mut.txt

壊し方ごとに、落ちた試験の名前を出す。「差分の形」の試験(test_before_copy_and_only_the_loop_places_changed)のほかに、動きを見る
試験が 1 本以上落ちなければ「見逃し」。一時の写しは experiments/20261026_fix/fix6/tmp_ui/mut_r2/ に置き、終わったら消す。
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
APP = JF / 'jetformer' / 'web' / 'static' / 'app.js'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TMP = JF / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_ui' / 'mut_r2'
SHAPE = 'test_before_copy_and_only_the_loop_places_changed'

LOST = "  const lost = () => { if (drIdle()) { drLost(); pollDrive(); } };"
CATCH = "catch (e) { clearTimeout(late); toast(e.message, true); lost(); return; }"
AC = "  const ac = drIdle() && typeof AbortController === 'function' ? new AbortController() : null;"
IDLE = "const drIdle = () => !state.drive.token && !(state.drive.st && state.drive.st.running);"
MUTS = {
    '一覧の失敗で輪を始めない(前の形)': (CATCH, "catch (e) { clearTimeout(late); toast(e.message, true); return; }"),
    '一覧の失敗・期限で drLost しない': (LOST, "  const lost = () => { if (drIdle()) { pollDrive(); } };"),
    '一覧の失敗・期限で輪を始めない(drLost だけ)': (LOST, "  const lost = () => { if (drIdle()) { drLost(); } };"),
    '一覧の期限なし': ("  const late = setTimeout(lost, DR_IDLE_WAIT_MS);", "  const late = null;"),
    '一覧の期限を返事で消さない': ("  clearTimeout(late);\n  if (!state.drive.st) {", "  if (!state.drive.st) {"),
    '走行中も一覧の期限・失敗で問い合わせる': (LOST, "  const lost = () => { if (true) { drLost(); pollDrive(); } };"),
    '打ち切らない': (AC, "  const ac = null;"),
    '走行中も打ち切る': (AC, "  const ac = typeof AbortController === 'function' ? new AbortController() : null;"),
    'token を見ない(走行を始めた画面でも打ち切る)': (IDLE, "const drIdle = () => !(state.drive.st && state.drive.st.running);"),
    '走行中の表示を見ない(ほかの画面の走行中も打ち切る)': (IDLE, "const drIdle = () => !state.drive.token;"),
    '打ち切りのタイマーを返事で消さない': ("  if (ac) clearTimeout(cut);\n", ""),
    '本文の途中の打ち切りを失敗にしない': ("    if (ac && ac.signal.aborted) throw new Error('打ち切り');\n", ""),
    'api が打ち切りの印を渡さない': ("  if (signal) opt.signal = signal;", "  if (false) opt.signal = signal;"),
    '期限が 60 秒': ("const DR_IDLE_WAIT_MS = 5000;", "const DR_IDLE_WAIT_MS = 60000;"),
}


def main():
    src = APP.read_text(encoding='utf-8')
    TMP.mkdir(parents=True, exist_ok=True)
    missed = 0
    try:
        for i, (name, (old, new)) in enumerate(MUTS.items()):
            assert src.count(old) == 1, name
            d = TMP / ('m%02d' % i)
            d.mkdir(exist_ok=True)
            (d / 'app.js').write_text(src.replace(old, new), encoding='utf-8')
            r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rf', '--basetemp', str(d / 'bt'),
                                'tests/test_ui_fix6_r2.py'], cwd=str(JF), capture_output=True, text=True,
                               env=dict(os.environ, JF_UI6R2_APP=str(d / 'app.js')), timeout=900)
            failed = sorted(set(re.findall(r'^FAILED tests/test_ui_fix6_r2\.py::(.+?)(?: - .*)?$', r.stdout, re.M)))
            behave = [f for f in failed if not f.startswith(SHAPE)]
            if not behave:
                missed += 1
            print('%s: 落ちた %d 本(動きの試験 %d 本)%s' % (name, len(failed), len(behave), '' if behave else '  ← 見逃し'))
            for f in behave:
                print('    ' + f)
            sys.stdout.flush()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print('見逃し %d / %d' % (missed, len(MUTS)))
    return 1 if missed else 0


if __name__ == '__main__':
    sys.exit(main())
