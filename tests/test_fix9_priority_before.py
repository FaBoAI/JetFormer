"""優先のモデル(2026-10-29 fix9。main session)の「直す前の写し」と「変えた所」の試験(形と、直す前の写しとの比べ。動きは
tests/test_drive_priority.py が見る)。CPU だけ・サーバーは起動しない・車や装置には触れない。

ユーザー(10/29 08:4x)「今日のテスト検証で動作確認したい 5 つのモデル整理し、セレクトボックスの上から 5 つ目までは優先モデルを
並べること」で、main session が jetformer/drive.py(PRIORITY_NAME・PRIORITY_MAX・PRIORITY_MARKS・_priority_path・priority・
save_priority を足し、list_models の並べ方を変えた)と jetformer/web/static/app.js(fillDriveModels の option の先頭に ①〜⑤ と名前・
renderDriveModel の情報の欄に優先の行)を直した。data/drive_priority.json が無ければ、一覧も選ぶ欄の HTML も前とまったく同じ。

直す前の写し tests/fixtures/fix9_priority_before/(SHA256SUMS):
- drive.py = 3f2e31a2…(10/28 fix8 safety の説明の文のあとの版 = tests/test_fix3_r3_2_tools.py の LATER_FIX8_SAFETY_DOC)
- app.js = e61dfac9…(10/28 fix8 画面の係 8-2 回目のあとの版 = 同じ LATER_FIX8_UI_R2)
  (どちらも git の b53426e と同じ・main session の控え experiments/20261026_fix/fix9/priority/*.bak-20261029-priority と同じ)
- test_fix8_docs_final.py.orig = a0030643…(この直しで 3 つの確かめを直す前の tests/test_fix8_docs_final.py。文書
  fix8/README.md 10 章の表に sha256 が載っている試験なので、表のその行はこの写しで照らす)

前の直しの「版を sha256 で固定した試験」(LATER_* の鎖)は、今のファイルがこの直しの版(LATER_FIX9_PRIORITY)なら、この写しをたどって
前の版につなぐ。そのため、ここで「変わってよいのは優先のモデルの所だけ」を見る(緩めない):
- drive.py: 上の段の文を名前で照らす(AST)。足したのは定数 3 つ(RECOMMENDED_PATH のすぐあと)と関数 3 つ(list_models のすぐ前)、
  変えたのは list_models だけ、消したものは無い。説明の文・既定値・Driver(走行のループ)・ほかの関数は同じ。
- list_models: 写しの drive.py を読み込み、同じ作り物の学習結果で流す。優先のモデルのファイルが無い・壊れている・空なら答えがまったく
  同じ。あれば、そのモデルを書いた順で先頭に出し priority の項目を足しただけ(ほかの値・残りの順は同じ)。
- app.js: 消した行は 2 行、足した行は 6 行で、fillDriveModels と renderDriveModel の中だけ。前の行の先頭に優先の印を足しただけ(優先の
  モデルでなければ字は同じ)。足した行に送る物・問い合わせ・タイマー・トースト・確認の文は無い。版(DRIVE_CLIENT 9)は同じ。
  2 つの関数のほかは写しと同じ。
- 画面(node): 写しと今の fillDriveModels を同じ一覧で流す。優先のモデルが無ければ選ぶ欄の HTML がまったく同じ。あれば、優先の
  モデルの option だけ ①〜⑤ と名前が前に付き、ほかの字は同じ。
- tests/test_fix8_docs_final.py: 写しから変えたのは説明の文・足した定数と 1 つの関数・3 つの確かめ(test_words_match_the_code・
  test_restart_not_needed・test_fix8_readme)だけ。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> \\
        tests/test_fix9_priority_before.py
"""
import ast
import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive

import test_drive_priority as DP

ROOT = Path(__file__).resolve().parents[1]
FIX9 = ROOT / 'tests' / 'fixtures' / 'fix9_priority_before'
DRIVE_PY = ROOT / 'jetformer' / 'drive.py'
APP = ROOT / 'jetformer' / 'web' / 'static' / 'app.js'
DOCS_FINAL = ROOT / 'tests' / 'test_fix8_docs_final.py'
BAK = ROOT / 'experiments' / '20261026_fix' / 'fix9' / 'priority'          # main session の控え(*.bak-20261029-priority)
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')

# 直す前の写しの sha256(drive.py・app.js = 前の直しのあとの版。test_fix8_docs_final.py.orig = fix8/README.md 10 章の表の版)
BEFORE_SHA = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
              'app.js': 'e61dfac905cb8bc86ac76e07ff104c85df1c9433d5a5b889ac80a0af1190ce73',
              'test_fix8_docs_final.py.orig': 'a0030643a7833841d7c3c7bd3e7c27b5d3a4de2efb4a81c1b792a80e7d39d403'}
# この直しのあとの版の sha256(前の直しの試験の LATER_FIX9_PRIORITY と同じ値。このあと誰かが直せば、その試験と下の確かめが落ちて知らせる)
LATER_FIX9_PRIORITY = {'jetformer/drive.py': 'ecfb6656cae624950d5de221dfa71a5ac31b84ba82dacfbea2136832eae92315',
                       'jetformer/web/static/app.js': '6b9e160c43c870dc1f4c2a5e2c23e2ad7513fe7e925d76a12cb69d5854966233'}

# drive.py: 足した文(この順で。定数 3 つは RECOMMENDED_PATH のすぐあと・関数 3 つは list_models のすぐ前)と変えた文
DRIVE_ADDED = ['set:PRIORITY_NAME', 'set:PRIORITY_MAX', 'set:PRIORITY_MARKS', 'def:_priority_path', 'def:priority', 'def:save_priority']
DRIVE_CHANGED = ['def:list_models']
PRIORITY_KEYS = {'priority', 'priority_mark', 'priority_label', 'priority_reason', 'priority_title'}

# app.js: 消した行(fillDriveModels の option の 1 行・renderDriveModel の情報の欄の 1 行)と足した行(それぞれ 3 行)
APP_REMOVED = [
    '    `<option value="${esc(m.name)}" ${m.drivable === false ? \'disabled\' : \'\'}>${m.recommended ? \'★推奨 \' : \'\'}「${esc(m.policy_title || \'JetFormer v0\')}」${esc(m.name)}`',
    '  $(\'#dr-model-info\').innerHTML = (m.recommended ? `<div><span class="badge ok">★推奨のモデル</span>${m.recommended_reason ? ` ${esc(m.recommended_reason)}` : \'\'}`',
]
APP_ADDED = [
    '    // 優先のモデル(サーバーの data/drive_priority.json。一覧の上から 5 つまで)は ①〜⑤ と名前を先に出す(2026-10-29)',
    '    `<option value="${esc(m.name)}" ${m.drivable === false ? \'disabled\' : \'\'}>${m.priority ? `${esc(m.priority_mark || \'\')}${esc(m.priority_label || \'優先\')} ` : \'\'}`',
    "    + `${m.recommended ? '★推奨 ' : ''}「${esc(m.policy_title || 'JetFormer v0')}」${esc(m.name)}`",
    '  $(\'#dr-model-info\').innerHTML = (m.priority ? `<div><span class="badge ok">${esc(m.priority_mark || \'\')} ${esc(m.priority_title || \'優先のモデル\')}</span>`',
    "      + ` <b>${esc(m.priority_label || '')}</b>${m.priority_reason ? ` ${esc(m.priority_reason)}` : ''}</div>` : '')",
    '    + (m.recommended ? `<div><span class="badge ok">★推奨のモデル</span>${m.recommended_reason ? ` ${esc(m.recommended_reason)}` : \'\'}`',
]
APP_FILL_PRI = "${m.priority ? `${esc(m.priority_mark || '')}${esc(m.priority_label || '優先')} ` : ''}"   # option の先頭に足した印
APP_INFO_HEAD = "  $('#dr-model-info').innerHTML = "

# tests/test_fix8_docs_final.py: この直しで変えた文(説明の文と 3 つの確かめ)と足した文(この順で、BOOT のすぐあと)
DOCS_FINAL_CHANGED = ['doc', 'def:test_words_match_the_code', 'def:test_restart_not_needed', 'def:test_fix8_readme']
DOCS_FINAL_ADDED = ['set:FIX9_BEFORE', 'set:LATER_FIX9_PRIORITY', 'set:LATER_FIX9_HM', 'set:LATER_FIX9_BOOT', 'set:SELF',
                    'def:_fix9_sums']

# 選ぶ欄を node で作る台(tests/test_drive_priority.py の _fill と同じ作り。app.js の fillDriveModels だけを流す)
STUB = """
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const sel = { value: '', innerHTML: '' };
const $ = (q) => sel;
const state = { drive: { models: %s, st: null, restored: true } };
const pickDriveModel = (ms) => ms.length ? ms[0].name : '';
const renderDriveModel = () => {};
"""
OPT = re.compile(r'<option value="([^"]*)"([^>]*)>(.*?)</option>')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sums():
    return dict(reversed(x.split(None, 1)) for x in (FIX9 / 'SHA256SUMS').read_text().splitlines() if x.strip())


def top(path):
    """上の段の文を名前で: 説明の文 'doc'・関数 'def:名前'・クラス 'class:名前'・代入 'set:名前'・ほか 'stmt:文'。値は AST(ast.dump)。
    順も返す(同じ名前が 2 つあれば落とす = 名前で照らせない)"""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    out, order = {}, []
    for i, n in enumerate(tree.body):
        if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            k = 'doc'
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            k = 'def:' + n.name
        elif isinstance(n, ast.ClassDef):
            k = 'class:' + n.name
        elif isinstance(n, ast.Assign):
            k = 'set:' + ','.join(ast.unparse(t) for t in n.targets)
        elif isinstance(n, ast.AnnAssign):
            k = 'set:' + ast.unparse(n.target)
        else:
            k = 'stmt:' + ast.unparse(n)
        assert k not in out, (str(path), k)
        out[k] = ast.dump(n)
        order.append(k)
    return out, order


def changes(before, now):
    """(足した文・消した文・変えた文・写しの順・今の順)。文は top の名前"""
    b, ob = top(before)
    c, oc = top(now)
    return ([k for k in oc if k not in b], [k for k in ob if k not in c], [k for k in ob if k in c and b[k] != c[k]], ob, oc)


def assert_drive_changed_only_priority(now=DRIVE_PY):
    """drive.py は写しから優先のモデルの所だけ変えた(足した定数 3 つ・関数 3 つ、変えた list_models。場所も決まり)。ほかの試験も呼ぶ
    (今の drive.py がこの直しの版のとき、写しで前の版につなぐ前に)"""
    added, removed, changed, ob, oc = changes(FIX9 / 'drive.py', now)
    assert (added, removed, changed) == (DRIVE_ADDED, [], DRIVE_CHANGED), (added, removed, changed)
    assert [k for k in oc if k not in DRIVE_ADDED] == ob                          # ほかの文の順は同じ
    i, j = oc.index('set:RECOMMENDED_PATH'), oc.index('def:list_models')
    assert oc[i + 1:i + 4] == DRIVE_ADDED[:3] and oc[j - 3:j] == DRIVE_ADDED[3:]


def assert_docs_final_changed_only_fix9(now=DOCS_FINAL):
    """tests/test_fix8_docs_final.py は写し(文書 10 章の表の版)から、説明の文・3 つの確かめと、BOOT のすぐあとに足した定数・関数だけ
    変えた。test_fix8_readme も呼ぶ(表のこの試験の行を写しで照らす前に)"""
    assert sha(FIX9 / 'test_fix8_docs_final.py.orig') == sums()['test_fix8_docs_final.py.orig']
    added, removed, changed, ob, oc = changes(FIX9 / 'test_fix8_docs_final.py.orig', now)
    assert (added, removed, changed) == (DOCS_FINAL_ADDED, [], DOCS_FINAL_CHANGED), (added, removed, changed)
    assert [k for k in oc if k not in DOCS_FINAL_ADDED] == ob
    i = oc.index('set:BOOT')
    assert oc[i + 1:i + 1 + len(DOCS_FINAL_ADDED)] == DOCS_FINAL_ADDED


def app_diff(a_path, b_path):
    a = Path(a_path).read_text(encoding='utf-8').splitlines()
    b = Path(b_path).read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    return [a[i] for o in ops for i in range(o[1], o[2])], [b[j] for o in ops for j in range(o[3], o[4])], ops


def js_func(js, name):
    """app.js の上の段の関数 name(前の改行から終わりの「}」の行まで)"""
    m = re.search(r'\nfunction %s\(\) \{\n.*?\n\}\n' % re.escape(name), js, re.S)
    assert m, name
    return m.group(0)


_BEFORE = {}


def before_drive():
    """写しの drive.py をモジュールにする(sha256 を照らしてから。jetformer の中のほかのモジュールは今のもの)"""
    if 'd' not in _BEFORE:
        assert sha(FIX9 / 'drive.py') == sums()['drive.py'] == BEFORE_SHA['drive.py']
        spec = importlib.util.spec_from_loader('jetformer._drive_before_fix9p', loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'jetformer'
        exec(compile((FIX9 / 'drive.py').read_text(encoding='utf-8'), str(FIX9 / 'drive.py'), 'exec'), mod.__dict__)
        assert not hasattr(mod, 'priority') and not hasattr(mod, 'PRIORITY_NAME')       # 写しは直す前
        _BEFORE['d'] = mod
    return _BEFORE['d']


@pytest.fixture
def both(monkeypatch, tmp_path):
    """写しと今の drive.py を、同じ作り物の学習結果(tests/test_drive_priority.py の 7 つ)・同じ推奨のファイルの場所にする"""
    d0 = before_drive()
    r = tmp_path / 'runs'
    r.mkdir()
    for mod in (drive, d0):
        monkeypatch.setattr(mod, 'RUNS_DIR', r)
        monkeypatch.setattr(mod, 'RECOMMENDED_PATH', tmp_path / 'drive_recommended.json')
    for name, at in DP.NAMES:
        DP._run(r, name, at)
    return d0


def fill_html(app_path, models):
    """app.js(写しか今)の fillDriveModels を node で流し、選ぶ欄の HTML を返す"""
    js = Path(app_path).read_text(encoding='utf-8')
    m = re.search(r'^function fillDriveModels\(\) \{.*?^}\n', js, re.S | re.M)
    assert m, str(app_path)
    code = (STUB % json.dumps(models, ensure_ascii=False)) + m.group(0) + \
        '\nfillDriveModels();\nprocess.stdout.write(JSON.stringify(sel.innerHTML));'
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# ================================================================== 写し・今の版
def test_before_copies_are_the_versions_before_this_fix():
    """写しは前の直しのあとの drive.py(3f2e31a2…)・app.js(e61dfac9…)と、直す前の tests/test_fix8_docs_final.py(a0030643…)。
    main session の控えがあれば同じ"""
    s = sums()
    assert s == BEFORE_SHA
    for name, h in s.items():
        assert sha(FIX9 / name) == h, name
    for name in ('drive.py', 'app.js'):
        b = BAK / (name + '.bak-20261029-priority')
        if b.exists():
            assert sha(b) == s[name], name


def test_now_is_the_priority_version():
    """今の drive.py・app.js はこの直しの版(このあと誰かが直せば落ちて知らせる = 前の直しの試験の LATER_FIX9_PRIORITY も合わせる)"""
    for rel, h in LATER_FIX9_PRIORITY.items():
        assert sha(ROOT / rel) == h, rel


# ================================================================== drive.py
def test_drive_only_priority_parts_changed():
    """drive.py で足したのは定数 3 つと関数 3 つ、変えたのは list_models だけ(上の段の文を名前で照らす。説明の文・既定値・Driver・ほかの
    関数は同じ)。足した定数の値・優先のモデルのファイルの場所(推奨のモデルと同じ data/)"""
    assert_drive_changed_only_priority()
    d0 = before_drive()
    assert drive.DEFAULT_PARAMS == d0.DEFAULT_PARAMS and drive.PARAM_LIMITS == d0.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == d0.RESET_ON_BOOT
    assert (drive.PRIORITY_NAME, drive.PRIORITY_MAX, drive.PRIORITY_MARKS) == ('drive_priority.json', 5, '①②③④⑤')
    assert drive._priority_path() == drive.RECOMMENDED_PATH.with_name('drive_priority.json')
    assert drive.RECOMMENDED_PATH == d0.RECOMMENDED_PATH


def test_list_models_without_priority_file_is_the_before_copy(both):
    """優先のモデルのファイルが無い・壊れている・空なら、今の list_models は写しの list_models とまったく同じ(推奨のモデルがあってもなくても)"""
    d0 = both
    assert not drive._priority_path().exists()
    assert drive.list_models() == d0.list_models() and len(d0.list_models()) == len(DP.NAMES)
    drive.save_recommended(DP.B, '評価で決めた')
    assert drive.list_models() == d0.list_models() and d0.list_models()[0]['name'] == DP.B
    for text in ('{broken', json.dumps({'models': []}), json.dumps({'models': [{'run': ''}, 3]}), json.dumps([1, 2])):
        drive._priority_path().write_text(text)
        assert drive.priority() is None and drive.list_models() == d0.list_models(), text


def test_list_models_with_priority_only_moves_and_marks(both):
    """優先のモデルがあれば、そのモデルを書いた順に先頭へ出して priority の項目(5 つ)を足しただけ。ほかの値と、残りのモデルの順は写しと同じ"""
    d0 = both
    drive.save_recommended('first_trial', '古い推奨')
    drive.save_priority(DP.FIVE, title='10/29 の試験で確かめる 5 つ')
    now, old = drive.list_models(), d0.list_models()
    top5 = [x['run'] for x in DP.FIVE]
    assert [m['name'] for m in now] == top5 + [m['name'] for m in old if m['name'] not in top5]
    olds = {m['name']: m for m in old}
    for m in now:
        assert {k: v for k, v in m.items() if k not in PRIORITY_KEYS} == olds[m['name']], m['name']
        assert set(m) - set(olds[m['name']]) == (PRIORITY_KEYS if m['name'] in top5 else set()), m['name']


# ================================================================== app.js
def test_app_js_only_priority_lines_changed():
    """app.js で消した行は 2 行・足した行は 6 行で、fillDriveModels と renderDriveModel の中だけ。前の行の先頭に優先の印を足しただけ
    (優先のモデルでなければ字は同じ)。足した行に送る物・問い合わせ・タイマー・トースト・確認の文は無い。版は 9 のまま。2 つの関数のほかは
    写しと同じ"""
    removed, added, ops = app_diff(FIX9 / 'app.js', APP)
    assert (removed, added) == (APP_REMOVED, APP_ADDED) and [o[0] for o in ops] == ['replace', 'replace'], (removed, added)
    js, old = APP.read_text(encoding='utf-8'), (FIX9 / 'app.js').read_text(encoding='utf-8')
    fill, render = js_func(js, 'fillDriveModels'), js_func(js, 'renderDriveModel')
    assert all(x in fill.split('\n') for x in added[:3]) and all(x in render.split('\n') for x in added[3:])
    # option: 前の 1 行 = 足した 2 行目から終わりの「`」と印を除き、3 行目の「    + `」のあとをつないだもの
    assert added[1].count(APP_FILL_PRI) == 1 and added[1].endswith(APP_FILL_PRI + '`') and added[2].startswith('    + `')
    assert added[1][:-1].replace(APP_FILL_PRI, '') + added[2][len('    + `'):] == removed[0]
    # 情報の欄: 前の 1 行 = 頭 + 足した 3 行目の「    + 」のあと(優先の行は前に足した 2 行。優先のモデルでなければ '')
    assert removed[1].startswith(APP_INFO_HEAD + '(m.recommended ? ') and added[3].startswith(APP_INFO_HEAD + '(m.priority ? ')
    assert added[5].startswith('    + (m.recommended ? ') and APP_INFO_HEAD + added[5][len('    + '):] == removed[1]
    assert added[4].endswith("</div>` : '')")
    code = [x.split('//')[0] for x in added]
    assert not any(re.search(r"\bapi\(|'POST'|fetch\(|setTimeout|setInterval|toast\(|confirm\(|\bbody\b|client", x) for x in code), added
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert js.replace(fill, '').replace(render, '') == old.replace(js_func(old, 'fillDriveModels'), '').replace(
        js_func(old, 'renderDriveModel'), '')


@needs_node
def test_select_html_same_as_before_copy(both):
    """選ぶ欄の HTML: 優先のモデルが無ければ写しの fillDriveModels とまったく同じ。あれば、優先のモデルの option だけ ①〜⑤ と名前が前に付き、
    ほかの字・値・印(disabled)・順は同じ"""
    drive.save_recommended(DP.B, '評価で決めた')
    ms = drive.list_models()
    assert not any(PRIORITY_KEYS & set(m) for m in ms)
    h1, h0 = fill_html(APP, ms), fill_html(FIX9 / 'app.js', ms)
    assert h1 == h0 and len(OPT.findall(h1)) == len(DP.NAMES)
    drive.save_priority(DP.FIVE, title='10/29 の試験で確かめる 5 つ')
    ms = drive.list_models()
    now, old = OPT.findall(fill_html(APP, ms)), OPT.findall(fill_html(FIX9 / 'app.js', ms))
    assert [o[0] for o in now] == [o[0] for o in old] == [m['name'] for m in ms]
    for (v, a1, t1), (_, a0, t0), m in zip(now, old, ms):
        pre = m['priority_mark'] + m['priority_label'] + ' ' if m.get('priority') else ''
        assert a1 == a0 and t1 == pre + t0, v
    assert sum(1 for m in ms if m.get('priority')) == 5


# ================================================================== 文書の表にある試験
def test_docs_final_test_changed_only_the_fix9_parts():
    """tests/test_fix8_docs_final.py(sha256 が fix8/README.md 10 章の表にある)は、写し(= 表の版 a0030643…)から説明の文・足した定数と
    1 つの関数・3 つの確かめだけ変えた(表のこの行は test_fix8_readme が写しで照らす)。足した定数の版はこの直しの版"""
    assert_docs_final_changed_only_fix9()
    t = DOCS_FINAL.read_text(encoding='utf-8')
    for rel, h in LATER_FIX9_PRIORITY.items():
        assert "'%s': '%s'" % (rel, h) in t, rel
