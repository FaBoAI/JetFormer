# -*- coding: utf-8 -*-
"""わざと壊す(直す 8-2 回目・係 ui の 3 件目 rehearsal-r2-vbefore-screen-vs-server)。本物の app.js は変えず、写しの vbatSourceText を
1 か所ずつ壊して JF_UI82_APP で tests/test_ui_fix8_r2.py に渡す(Firefox は使わない)。数えるのは電圧の源の試験(形を見る
test_app_js_only_vbat_source_text_changed のほか): 入力の表・でたらめな説明欄・pyFloat と空白の字・範囲・確認の文(5 本)。

  cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B \\
      experiments/20261026_fix/fix8/fix_ui_r2/mut/run_mut_app.py
出力 out_mut_app.txt・junit_app/<id>.xml。
"""
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
APP = ROOT / 'jetformer' / 'web' / 'static' / 'app.js'
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r2_before' / 'app.js'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
BT = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_ui_r2' / 'mut_app'
COUNT = ('test_vbefore_screen_matches_server_table', 'test_vbefore_random_descriptions_match_server',
         'test_pyfloat_is_python_float_and_py_space_is_python_isspace', 'test_vdesc_range_is_the_server_range',
         'test_start_and_switch_confirm_texts_say_what_the_server_uses')
GRAMMAR = r"    if (!/^[+-]?(\d(_?\d)*(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$/.test(t)) return null;"
RANGE = 'Number.isFinite(vals[k]) && vals[k] >= VDESC[0] && vals[k] <= VDESC[1]'
MUTS = [
    ('A00 壊さない(今の app.js)', None),
    ('A01 直す前の app.js(aa275bb5。V_before= のあとが数字で始まるかだけ)', 'before'),
    ('A02 6.0〜9.0 V の範囲を見ない', [(RANGE, 'Number.isFinite(vals[k])')]),
    ('A03 Vb_before= を見ない', [("for (const key of ['V_before', 'Vb_before'])", "for (const key of ['V_before'])")]),
    ('A04 字を NFKC でそろえる(全角の点も点に)', [("    let t = '';\n    for (const ch of s) {", "    let t = '';\n    s = s.normalize('NFKC');\n    for (const ch of s) {")]),
    ('A05 JavaScript の空白(/\\s+/)で区切る', [('.split(PY_SPACE)) {', '.split(/\\s+/)) {')]),
    ('A06 parseFloat で読む(後ろのごみを捨てる)', [(GRAMMAR, '    return Number.isNaN(parseFloat(t)) ? null : parseFloat(t);')]),
    ('A07 範囲の端を入れない(6.0・9.0 は使わない)', [(RANGE, 'Number.isFinite(vals[k]) && vals[k] > VDESC[0] && vals[k] < VDESC[1]')]),
    ('A08 読めない最初の V_before= で決める', [('        if (x !== null) vals[key] = x;', '        vals[key] = x;')]),
    ('A09 末尾の V を除かない', [(".replace(/[Vv]+$/, '')", '')]),
    ('A10 鍵の大文字・小文字を見ない', [("if (tok.startsWith(key + '=') && !(key in vals))",
                                         "if (tok.toLowerCase().startsWith(key.toLowerCase() + '=') && !(key in vals))")]),
    ('A11 最後の V_before= を使う', [(" && !(key in vals))", ')')]),
    ('A12 下線を読まない', [(GRAMMAR, GRAMMAR.replace('(_?\\d)', '(\\d)'))]),
    ('A13 inf・nan を読めない字にする', [("    if (/^[+-]?(inf|infinity)$/i.test(t)) return t[0] === '-' ? -Infinity : Infinity;\n"
                                          "    if (/^[+-]?nan$/i.test(t)) return NaN;\n", '')]),
    ('A14 範囲を 6.5〜9.0 V に', [('const VDESC = [6.0, 9.0];', 'const VDESC = [6.5, 9.0];')]),
    ('A15 pyFloat で前後の空白を除かない', [("    t = t.replace(/^ +| +$/g, '');\n", '')]),
    ('A16 Unicode の数字の値を 10 で割った余りにしない', [('t += String((c - z) % 10);', 't += String(c - z);')]),
]


def make(mid, how):
    dst = HERE / 'app' / ('%s.js' % mid.split()[0])
    dst.parent.mkdir(exist_ok=True)
    if how is None:
        shutil.copy(APP, dst)
    elif how == 'before':
        shutil.copy(BEFORE, dst)
    else:
        s = APP.read_text(encoding='utf-8')
        for a, b in how:
            assert s.count(a) == 1, (mid, a)
            s = s.replace(a, b)
        dst.write_text(s, encoding='utf-8')
    r = subprocess.run(['node', '--check', str(dst)], capture_output=True, text=True)
    assert r.returncode == 0, (mid, r.stderr)
    return dst


def run(mid, app):
    junit = HERE / 'junit_app' / ('%s.xml' % mid.split()[0])
    junit.parent.mkdir(exist_ok=True)
    BT.mkdir(parents=True, exist_ok=True)
    bt = BT / mid.split()[0]
    env = dict(os.environ, JF_UI82_APP=str(app), PYTHONDONTWRITEBYTECODE='1')
    env.pop('JF_UI_FIREFOX', None)
    subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'tmp_path_retention_policy=failed', '--basetemp=%s' % bt,
                    '-W', 'ignore::DeprecationWarning', '--junitxml=%s' % junit, 'tests/test_ui_fix8_r2.py'],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    res = {}
    for tc in ET.parse(junit).getroot().iter('testcase'):
        st = ('failed' if tc.find('failure') is not None else 'error' if tc.find('error') is not None
              else 'skipped' if tc.find('skipped') is not None else 'passed')
        res[tc.get('name')] = st
    shutil.rmtree(bt, ignore_errors=True)
    return res


def main():
    lines = ['# わざと壊す(tests/test_ui_fix8_r2.py。JF_UI82_APP。Firefox なし)。数えるのは電圧の源の試験 5 つ(確認の文は 5 本)',
             '# 列: 壊し方 | 落ちた電圧の源の試験(本)| 形の試験 test_app_js_only_vbat_source_text_changed | ほかの試験']
    missed = 0
    for mid, how in MUTS:
        app = make(mid, how)
        res = run(mid, app)
        cnt = {k: v for k, v in res.items() if k.split('[')[0] in COUNT}
        failed = sorted(k for k, v in cnt.items() if v == 'failed')
        assert 'error' not in cnt.values() and len(cnt) == 9, (mid, cnt)
        shape = res.get('test_app_js_only_vbat_source_text_changed')
        other = sorted({v for k, v in res.items() if k.split('[')[0] not in COUNT and k != 'test_app_js_only_vbat_source_text_changed'})
        if how is None:
            assert not failed and shape == 'passed' and other == ['passed', 'skipped'], (mid, res)
        elif not failed:
            missed += 1
        lines.append('%s | %d 本 %s | 形 %s | ほか %s' % (mid, len(failed), failed, shape, other))
        print(lines[-1], flush=True)
    lines.append('見逃し(電圧の源の試験が 1 本も落ちない壊し方): %d / %d' % (missed, len(MUTS) - 1))
    (HERE / 'out_mut_app.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(lines[-1])


if __name__ == '__main__':
    sys.exit(main())
