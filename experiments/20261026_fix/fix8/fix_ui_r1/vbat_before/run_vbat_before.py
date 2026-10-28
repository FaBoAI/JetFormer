"""前からの Firefox の試験 tests/test_vbat_ui.py::test_firefox_drive_tab_vbat_and_switches を、直す前の app.js・style.css(この場所の
static/ = tests/fixtures/fix8_ui_r1_before の写しと今の index.html)で流す(この組の直しの前から落ちるかを見る)。STATIC を差し替えるだけ。
JF_UI_FIREFOX=1・jf-heavy --lock big --mem 3000M。出力は標準出力"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/tests')
import test_vbat_ui as V  # noqa: E402

V.STATIC = str(HERE / 'static')
print('STATIC', V.STATIC, sorted(os.listdir(V.STATIC)))
try:
    V.test_firefox_drive_tab_vbat_and_switches()
    print('RESULT passed')
except AssertionError as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print('RESULT failed at %s:%d %s' % (Path(tb.filename).name, tb.lineno, tb.line))
