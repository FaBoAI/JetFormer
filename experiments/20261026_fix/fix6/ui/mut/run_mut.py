"""fix6 画面の係: わざと壊すと tests/test_ui_fix6.py が落ちるか(本物の app.js は変えない。写しを壊して JF_UI6_APP で渡す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix6/ui/mut/run_mut.py > experiments/20261026_fix/fix6/ui/mut/out_mut.txt

壊し方ごとに、落ちた試験の名前を出す。「差分の形」の試験(test_before_copy_and_only_three_places_changed)のほかに、動きを見る
試験が 1 本以上落ちなければ「見逃し」。一時の写しは experiments/20261026_fix/fix6/tmp_ui/mut/ に置き、終わったら消す。
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
TMP = JF / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_ui' / 'mut'
SHAPE = 'test_before_copy_and_only_three_places_changed'

MUTS = {
    '出し直さない': ("      if (seq === state.drive.startSeq && (bootRefusal ||", "      if (false && seq === state.drive.startSeq && (bootRefusal ||"),
    '同期を待たない': ('    try { await pollDrive(); } finally {', '    try { pollDrive(); } finally {'),
    '知らせを数えない': ('  state.drive.bootToasts = (state.drive.bootToasts || 0) + 1;\n', ''),
    '断りの文を見分けない': ('bootRefusal = /^管理画面のサーバーが .*に起動し直しました\\(/.test(m);', 'bootRefusal = false;'),
    '頭の文なし': ("toast((st && !st.running ? '走行は始まっていません(停止中)。' : '')", "toast((st && !st.running ? '' : '')"),
    '走行中でも頭の文': ("toast((st && !st.running ? '走行は始まっていません(停止中)。' : '')", "toast((true ? '走行は始まっていません(停止中)。' : '')"),
    '次の ▶ を見ない': ('      if (seq === state.drive.startSeq && (bootRefusal ||', '      if ((bootRefusal ||'),
    '知らせを後ろに付けない': ("(bootRefusal ? m : `${m.replace(/。$/, '')}。${state.drive.bootToast}`)", '(bootRefusal ? m : m)'),
    '断りの文の形が広すぎる': ('bootRefusal = /^管理画面のサーバーが .*に起動し直しました\\(/.test(m);', 'bootRefusal = /起動し直しました/.test(m);'),
    '番号を数えない': ('  const seq = state.drive.startSeq = (state.drive.startSeq || 0) + 1;', '  const seq = state.drive.startSeq = 1;'),
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
                                'tests/test_ui_fix6.py'], cwd=str(JF), capture_output=True, text=True,
                               env=dict(os.environ, JF_UI6_APP=str(d / 'app.js')), timeout=900)
            failed = sorted(set(re.findall(r'^FAILED tests/test_ui_fix6\.py::(\S+)', r.stdout, re.M)))
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
