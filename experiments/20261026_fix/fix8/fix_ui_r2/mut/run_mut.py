# -*- coding: utf-8 -*-
"""わざと壊す(直す 8-2 回目・係 ui)。本物の style.css は変えず、写しを 1 か所ずつ壊して JF_UI82_CSS で tests/test_ui_fix8_r2.py に渡す。
数えるのは形を見る試験(写し・差分の 3 本)のほかの 2 本: 宣言の試験と Firefox の試験(JF_UI_FIREFOX=1)。

  cd ~/jetformer && JF_UI_FIREFOX=1 /home/jetson/bin/jf-heavy --lock big --mem 3000M -- /home/jetson/miniconda/envs/jetracer/bin/python -B \\
      experiments/20261026_fix/fix8/fix_ui_r2/mut/run_mut.py
出力 out_mut.txt・junit/<id>.xml。
"""
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
CSS = ROOT / 'jetformer' / 'web' / 'static' / 'style.css'
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r2_before' / 'style.css'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
BT = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_ui_r2' / 'mut'
ADDED = ' width: max-content; max-width: 90%; }'
MEDIA = '@media (max-width: 600px) { body.driving .toast { bottom: 126px; } }'
COUNT = ('test_float_stop_takes_its_text_width_and_driving_toast_goes_above_it_on_phones',
         'test_firefox_float_stop_and_driving_toast_on_phone')
MUTS = [
    ('M00 壊さない(今の style.css)', None),
    ('M01 直す前の style.css(dbc8b79a)', 'before'),
    ('M02 .float-stop に width: max-content なし(max-width 90% だけ)', [(ADDED, ' max-width: 90%; }')]),
    ('M03 .float-stop に max-width: 90% なし', [(ADDED, ' width: max-content; }')]),
    ('M04 走行中のトーストを上げる @media なし', [('\n' + MEDIA, '')]),
    ('M05 @media の bottom 100 px(2 行の ■ に 6 px 重なる)', [(MEDIA, MEDIA.replace('126px', '100px'))]),
    ('M06 @media で囲まない(どの幅でも 126 px)', [(MEDIA, 'body.driving .toast { bottom: 126px; }')]),
    ('M07 @media が止まっているときの .toast も上げる', [(MEDIA, '@media (max-width: 600px) { .toast { bottom: 126px; } }')]),
    ('M08 .float-stop の max-width 50%', [(ADDED, ' width: max-content; max-width: 50%; }')]),
    ('M09 @media の幅 400 px まで', [(MEDIA, MEDIA.replace('600px', '400px'))]),
    ('M10 .float-stop を画面いっぱいの帯(left/right 5%・transform なし)', [(ADDED, ' left: 5%; right: 5%; transform: none; width: auto; max-width: none; }')]),
    ('M11 .float-stop を 1 行に固定(white-space: nowrap)', [(ADDED, ' width: max-content; max-width: 90%; white-space: nowrap; }')]),
    ('M12 @media の bottom 112 px(間 6 px。指摘の案の下の値)', [(MEDIA, MEDIA.replace('126px', '112px'))]),
    ('M13 .float-stop の幅を 300 px に固定', [(ADDED, ' width: 300px; max-width: 90%; }')]),
    ('M14 .toast の z-index を ■ より下に(■ がトーストを隠す)', [('z-index: 50; font-size: 13px;', 'z-index: 30; font-size: 13px;')]),
]


def make(mid, how):
    dst = HERE / 'css' / ('%s.css' % mid.split()[0])
    dst.parent.mkdir(exist_ok=True)
    if how is None:
        shutil.copy(CSS, dst)
    elif how == 'before':
        shutil.copy(BEFORE, dst)
    else:
        s = CSS.read_text(encoding='utf-8')
        for a, b in how:
            assert s.count(a) == 1, (mid, a)
            s = s.replace(a, b)
        dst.write_text(s, encoding='utf-8')
    return dst


def run(mid, css):
    junit = HERE / 'junit' / ('%s.xml' % mid.split()[0])
    junit.parent.mkdir(exist_ok=True)
    bt = BT / mid.split()[0]
    BT.mkdir(parents=True, exist_ok=True)          # --basetemp の親(無いと tmp_path の試験が setup で error になる。1 回目はこれで無効)
    env = dict(os.environ, JF_UI82_CSS=str(css), JF_UI_FIREFOX='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'tmp_path_retention_policy=failed', '--basetemp=%s' % bt,
                    '--junitxml=%s' % junit, 'tests/test_ui_fix8_r2.py'], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    res = {}
    for tc in ET.parse(junit).getroot().iter('testcase'):
        name = tc.get('name')
        st = ('failed' if tc.find('failure') is not None else 'error' if tc.find('error') is not None
              else 'skipped' if tc.find('skipped') is not None else 'passed')
        res[name] = st
    shutil.rmtree(bt, ignore_errors=True)
    return res


def main():
    lines = ['# わざと壊す(tests/test_ui_fix8_r2.py。JF_UI82_CSS・JF_UI_FIREFOX=1)。数えるのは宣言の試験(decl)と Firefox の試験(ff)',
             '# 列: 壊し方 | decl | ff | 形の試験(差分。どの壊し方でも落ちるはず) | ほかの試験']
    missed = 0
    for mid, how in MUTS:
        css = make(mid, how)
        res = run(mid, css)
        decl, ff = res.get(COUNT[0]), res.get(COUNT[1])
        shape = res.get('test_style_css_only_the_float_stop_rule_and_one_media_rule')
        other = sorted({v for k, v in res.items() if k not in COUNT and not k.startswith('test_style_css_only')})
        assert 'error' not in (decl, ff), (mid, res)               # setup の error は「落ちた」に数えない
        if how is None:
            assert decl == ff == shape == 'passed' and other == ['passed'], (mid, res)     # 壊さないときは全部通る
        if how is not None and decl != 'failed' and ff != 'failed':
            missed += 1
        lines.append('%s | decl %s | ff %s | 形 %s | ほか %s' % (mid, decl, ff, shape, other))
        print(lines[-1], flush=True)
    lines.append('見逃し(decl も ff も落ちない壊し方): %d / %d' % (missed, len(MUTS) - 1))
    (HERE / 'out_mut.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(lines[-1])


if __name__ == '__main__':
    sys.exit(main())
