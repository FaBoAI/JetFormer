"""画面の係の直し 8-2 回目(2026-10-28 fix8。反証 8-2 回目・係 ui の指摘 1 件と、8-1 回目から渡された試験の古い期待 1 件と、
main session があとで足した係 rehearsal の指摘 1 件。~/jetformer_reports/fix_20261026/fix8/fix_ui_r2.md)の試験。

  float-stop-half-width-under-toast(style.css .float-stop・走行中の .toast): 浮いている ■(#btn-float-stop)は .toast の 8-1 の直しの前と
    同じ left: 50%・幅 auto の形で、使える幅が画面の半分になり、スマホ(幅 360〜430)で 3〜4 行・高さ 120〜150 px になった。走行開始の
    トースト(body.driving .toast の下 96 px。ふつうの ▶ の 3 秒「AI走行 No.N を開始しました」・推論の確認の 3 秒・11/3 のブロックの
    6 秒の「試し」の文)が ■ の上の行「■ AI走行を」を 39.5〜59 px 隠した。直し(style.css だけ): .float-stop に width: max-content・
    max-width: 90%(1〜2 行・高さ 60/90 px)、幅 600 px 以下の画面だけ走行中のトーストを 126 px(2 行の ■ の上 16 + 90 + 20)に。
    幅 768 px 以上・止まっているときのトースト(8-1 の直し)は前と同じ。
  test_vbat_ui.py 383 行(JF_UI_FIREFOX=1 のときだけ): 天井の確認の文に固定の「全体 4.5・左旋回中 4.0 m/s」を求め、今の既定
    (ceil_left_mps 4.1)で落ちた。期待の文を偽のサーバーの state['params'](= drive.DEFAULT_PARAMS の写し)から作る(toFixed(1))。
  rehearsal-r2-vbefore-screen-vs-server(app.js vbatSourceText): ▶・自動合わせのスイッチの確認の文の「電圧の源」は、基板の電圧が無いとき
    説明欄の「V_before= のあとが数字で始まるか」だけで「説明欄の V_before(走行のあいだ一定)」と出したので、V_before=831・8,31・0.83・9.5・
    「8.31,」でも「説明欄」と出て、サーバー(speed_auto.v_before_of。6.0〜9.0 V の数だけ)は使わず電圧の源「なし」で走った(全角の ８.３１ は
    逆)。直し(表示だけ): 読み方をサーバーと同じにした(Python の str.split() と同じ空白・V_before= のあと Vb_before=・末尾の V を除く・
    Python の float() と同じ決まり(Unicode の 10 進の字・下線・指数・inf/nan)・6.0〜9.0 V)。文の言葉は前と同じ。

直す前の写し tests/fixtures/fix8_ui_r2_before/(style.css dbc8b79a… = 8-1 回目のあとの版・test_vbat_ui.py 2508f48b…・
app.js aa275bb5… = 8-1 回目のあとの版。SHA256SUMS)。
index.html・server.py・bench.py・speed_auto.py・drive.py は変えていない(送る物・断る決まりは同じ。再起動は要らない)。app.js は
vbatSourceText の中だけ(版 9 のまま。サーバーは app.js の更新時刻で古い画面の ▶ を断る = 前の係と同じ)。
重なりは JF_UI_FIREFOX=1 のときだけ headless の Firefox で(file:// の頁に本物の index.html と style.css。app.js は読まない = 通信なし。
走行中は body.driving と ■ の表示を、本物の app.js の renderDrive と同じ形で付ける)。車・jetformer-web には触れない。
本物の ▶ の押し方で直す前と後を測った台本は experiments/20261026_fix/fix8/fix_ui_r2/measure/(報告 2 章)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed --basetemp=<消せる場所> tests/test_ui_fix8_r2.py
    # Firefox の 1 本(--basetemp は /tmp・隠しフォルダの外 = snap の Firefox が読める場所):
    JF_UI_FIREFOX=1 /home/jetson/bin/jf-heavy --lock big --mem 3000M -- ... -m pytest ... tests/test_ui_fix8_r2.py -k firefox

JF_UI82_CSS・JF_UI82_APP=<写し> でその写しを「今の style.css・app.js」として流す(わざと壊す台本 experiments/20261026_fix/fix8/fix_ui_r2/mut/)。
"""
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from jetformer import drive

import test_vbat_ui as V

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
FIXT = ROOT / 'tests' / 'fixtures'
BEFORE = FIXT / 'fix8_ui_r2_before'                     # 直す前の style.css・test_vbat_ui.py・app.js(SHA256SUMS)
CSS = Path(os.environ.get('JF_UI82_CSS') or STATIC / 'style.css')
APP = Path(os.environ.get('JF_UI82_APP') or STATIC / 'app.js')
VBAT = ROOT / 'tests' / 'test_vbat_ui.py'
NODE = shutil.which('node')
FIREFOX = os.environ.get('JF_UI_FIREFOX') == '1'
GECKO = V.GECKO
MEDIA = '@media (max-width: 600px) { body.driving .toast { bottom: 126px; } }'
ADDED_DECL = ' width: max-content; max-width: 90%; }'
# 走行開始で出るトースト(app.js の ▶ の後: toast(dry ? … : `AI走行 No.${r.ai_no} を開始しました`)・plan_diffs の「試し」の文)
TRIAL1 = 'ブロックの計画と違うので「試し」(cand=EXP)として記録します: throttle_gain 0.9(基準 1.0)'
TRIAL3 = TRIAL1 + '、speed_ceiling 1(基準 0)、ceil_left_mps 4.2(基準 4.1)'
RUN_TOASTS = {'start': ('AI走行 No.12 を開始しました', False), 'dry': ('推論の確認を開始しました(出力は中立)', False),
              'trial1': (TRIAL1, True), 'trial3': (TRIAL3, True), 'switch': ('走行中は ON にできません(OFF にはできます)', True)}
PHONES = [(w, h) for w in (360, 375, 390, 412, 430) for h in (660, 700, 760, 800, 844, 900)]
WIDE = [(768, 900), (1280, 800), (844, 390)]            # 見え方が前と同じであること
MID = [(600, 800), (700, 800)]                          # 前は 2 行(使える幅が画面の半分)→ 今は 1 行


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sums(d):
    return dict(reversed(x.split()) for x in (d / 'SHA256SUMS').read_text().splitlines() if x.strip())


def _diff(a_path, b_path):
    a = Path(a_path).read_text(encoding='utf-8').splitlines()
    b = Path(b_path).read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    return [a[i] for o in ops for i in range(o[1], o[2])], [b[j] for o in ops for j in range(o[3], o[4])], ops


def _rules(css):
    """一番外側の規則 {選択子: [宣言]}(@media の中は '@media …' の下に {選択子: [宣言]})。注は除く"""
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    out, i = [], 0
    while True:
        m = re.compile(r'\s*([^{}]+?)\s*\{').match(css, i)
        if not m:
            break
        sel, j = m.group(1), m.end()
        if sel.startswith('@'):                                  # @media・@keyframes は中を規則として読む
            depth, k = 1, j
            while depth:
                depth += {'{': 1, '}': -1}.get(css[k], 0)
                k += 1
            out.append((sel, _rules(css[j:k - 1])))
            i = k
        else:
            k = css.index('}', j)
            out.append((sel, [x.strip() for x in css[j:k].split(';') if x.strip()]))
            i = k + 1
    return out


# ================================================================== 形(直す前の写し・変えた所・変えていないファイル)
def test_before_copies_are_the_versions_before_this_fix():
    """直す前の写しは 8-1 回目のあとの style.css(dbc8b79a…。.toast の幅を直した版)・test_vbat_ui.py(2508f48b…)・
    app.js(aa275bb5…。8-1 回目のあとの版 = 10/28 04:54)"""
    sums = _sums(BEFORE)
    assert set(sums) == {'style.css', 'test_vbat_ui.py', 'app.js'}
    for name, h in sums.items():
        assert _sha(BEFORE / name) == h, name
    assert sums['style.css'].startswith('dbc8b79ad935') and sums['test_vbat_ui.py'].startswith('2508f48bc342')
    assert sums['app.js'].startswith('aa275bb5da0b')


def test_style_css_only_the_float_stop_rule_and_one_media_rule():
    """style.css で変えたのは .float-stop の 2 行目の終わりに「width: max-content; max-width: 90%;」を足した 1 行と、注の行と、
    「body.driving main { padding-bottom: 96px; }」のすぐ下の @media の 1 行だけ。.toast の行(8-1 の直し)・body.driving .toast の
    96 px の行・ほかの規則は同じ。index.html・server.py・bench.py は変えていない(app.js は 3 件目。test_app_js_only_vbat_source_text)"""
    removed, added, ops = _diff(BEFORE / 'style.css', CSS)
    assert len(removed) == 1 and removed[0].startswith('  font-weight: 700; background: var(--danger);') and removed[0].endswith(' }')
    new = [x for x in added if not (x.startswith('/*') or x.startswith('   '))]
    assert new == [removed[0][:-2] + ADDED_DECL, MEDIA], new
    assert [o[0] for o in ops] == ['insert', 'replace', 'insert'] or [o[0] for o in ops] == ['replace', 'insert'], ops
    lines = CSS.read_text(encoding='utf-8').splitlines()
    assert lines[lines.index(MEDIA) - 1] == 'body.driving main { padding-bottom: 96px; }'
    i = lines.index(removed[0][:-2] + ADDED_DECL)
    assert lines[i - 1].startswith('.float-stop { position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%); z-index: 40;')
    for keep in ('body.driving .toast { bottom: 96px; }', '.toast.error { background: var(--danger); }'):
        assert keep in lines
    old_toast = [x for x in (BEFORE / 'style.css').read_text(encoding='utf-8').splitlines() if x.startswith('.toast {')]
    assert old_toast == [x for x in lines if x.startswith('.toast {')] and len(old_toast) == 1
    assert _sha(STATIC / 'index.html').startswith('ef81bb6017d6')
    assert _sha(ROOT / 'jetformer' / 'web' / 'server.py').startswith('ad7a7d05f7a9')
    assert _sha(ROOT / 'jetformer' / 'web' / 'bench.py').startswith('a633bec341a3')


def test_float_stop_takes_its_text_width_and_driving_toast_goes_above_it_on_phones():
    """.float-stop は前の宣言のまま(左 50%・下 16 px・translateX(-50%)・z 40・14px 40px・20 px)に width: max-content と
    max-width: 90% を足しただけ。幅 600 px 以下の @media は走行中のトーストの bottom 126 px だけ(.float-stop・止まっているときの
    .toast には触れない)。■ とトーストを扱う規則はほかに無い。トーストは ■ より上(z 50 > 40)・押しても通る(pointer-events: none)"""
    now, old = _rules(CSS.read_text(encoding='utf-8')), _rules((BEFORE / 'style.css').read_text(encoding='utf-8'))
    top = {s: d for s, d in now if not s.startswith('@')}
    otop = {s: d for s, d in old if not s.startswith('@')}
    assert top['.float-stop'] == otop['.float-stop'] + ['width: max-content', 'max-width: 90%']
    assert {'position: fixed', 'left: 50%', 'bottom: 16px', 'transform: translateX(-50%)', 'z-index: 40', 'padding: 14px 40px',
            'font-size: 20px'} <= set(top['.float-stop'])
    # ほかの規則(一番外側・@media 980px・@keyframes)は順も中身も同じ。足した @media は 600px の 1 つだけで、中は走行中のトーストだけ
    assert [x for x in now if x[0] not in ('.float-stop', '@media (max-width: 600px)')] == [x for x in old if x[0] != '.float-stop']
    assert [r for s, r in now if s == '@media (max-width: 600px)'] == [[('body.driving .toast', ['bottom: 126px'])]]
    assert not [s for s, r in old if s == '@media (max-width: 600px)']
    assert [s for s, d in now if 'float-stop' in s or 'btn-float' in s] == ['.float-stop']
    assert 'z-index: 50' in top['.toast'] and 'pointer-events: none' in top['.toast'] and top['body.driving .toast'] == ['bottom: 96px']
    # 126 = ■ の下 16 + 2 行の ■ の高さ 90(上下の padding 14 × 2 + 枠 2 + 行 30 px × 2。行の高さは Firefox で測った)+ 間 20
    # (広い画面の 1 行の ■(高さ 60)と 96 px の間と同じ)
    fl = dict(x.split(': ', 1) for x in top['.float-stop'])
    pad_v, f_bottom = int(fl['padding'].split()[0][:-2]), int(fl['bottom'][:-2])
    t_phone = int(dict(x.split(': ', 1) for x in [r for s, r in now if s == '@media (max-width: 600px)'][0][0][1])['bottom'][:-2])
    t_wide = int(top['body.driving .toast'][0].split(': ')[1][:-2])
    assert t_phone - (f_bottom + 2 * pad_v + 2 + 2 * 30) == 20 == t_wide - (f_bottom + 2 * pad_v + 2 + 30)


def test_vbat_ui_only_the_ceiling_expectation_changed():
    """test_vbat_ui.py で変えたのは天井の確認の文の 1 行(固定の「全体 4.5・左旋回中 4.0 m/s」)を、偽のサーバーの state['params']
    (= drive.DEFAULT_PARAMS の写し)から作る形にしたことと、JavaScript の toFixed(1) と同じ字を作る関数 _to_fixed1 を足したことだけ"""
    removed, added, ops = _diff(BEFORE / 'test_vbat_ui.py', VBAT)
    assert removed == ["        assert spec['ceiling']['rule_text'] in msg and spec['ceiling']['note'] in msg and '全体 4.5・左旋回中 4.0 m/s' in msg"]
    code = [x.split('#')[0].rstrip() for x in added if x.split('#')[0].strip()]
    assert code == [
        'def _to_fixed1(x):',
        '    """JavaScript の Number.prototype.toFixed(1) と同じ文字(2 進の値そのものを丸める・ちょうど半分なら大きい側)"""',
        '    from decimal import ROUND_HALF_UP, Decimal',
        "    return str(Decimal(float(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))",
        "        ceil_keys = ('ceil_all_mps', 'ceil_left_mps')",
        "        assert {k: state['params'][k] for k in ceil_keys} == {k: drive.DEFAULT_PARAMS[k] for k in ceil_keys}",
        "        assert (spec['ceiling']['all_mps'], spec['ceiling']['left_mps']) == tuple(state['params'][k] for k in ceil_keys)",
        "        ceil_text = '全体 %s・左旋回中 %s m/s' % tuple(_to_fixed1(state['params'][k]) for k in ceil_keys)",
        "        assert spec['ceiling']['rule_text'] in msg and spec['ceiling']['note'] in msg and ceil_text in msg"]
    assert not any(re.search(r'\d\.\d', x) for x in code[4:])                          # 固定の速さの数は無い
    assert len(ops) == 2


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_to_fixed1_is_javascript_to_fixed():
    """_to_fixed1 は node の (x).toFixed(1) と同じ字(ちょうど半分の 4.25 → 4.3 も。Python の '%.1f' は 4.2 になる)"""
    xs = [4.5, 4.1, 4.0, 4.2, 4.25, 4.75, 4.15, 4.05, 3.95, 0.05, 0.25, 1.45, 4.35, 4.45, 5.0, 4.149999, 4.150001]
    xs += [round(2.5 + i * 0.01, 2) for i in range(300)] + [drive.DEFAULT_PARAMS['ceil_all_mps'], drive.DEFAULT_PARAMS['ceil_left_mps']]
    r = subprocess.run([NODE, '-e', 'console.log(JSON.stringify(JSON.parse(process.argv[1]).map((x) => x.toFixed(1))))', json.dumps(xs)],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert [V._to_fixed1(x) for x in xs] == json.loads(r.stdout)
    assert V._to_fixed1(4.25) == '4.3' and '%.1f' % 4.25 == '4.2'


@pytest.mark.skipif(NODE is None, reason='node がありません')
@pytest.mark.parametrize('left', [None, 4.2, 3.9, 4.25])
def test_vbat_ui_expectation_is_what_app_js_writes(left, monkeypatch):
    """test_vbat_ui の Firefox の試験と同じ作り方の期待の文(state['params'] = drive.DEFAULT_PARAMS の写しから _to_fixed1)が、
    本物の app.js の ceilConfirmText(speedSpec(…)) の文に入る。既定のまま(4.1)と、既定の左を 4.2・3.9・4.25(ちょうど半分 → 4.3)に変えた日も
    (期待が既定に付いてくる)。固定の「左旋回中 4.0」はどれにも入らない(前の期待の形)"""
    if left is not None:
        monkeypatch.setitem(drive.DEFAULT_PARAMS, 'ceil_left_mps', left)
    params = dict(drive.DEFAULT_PARAMS)                      # 偽のサーバーの state['params'] と同じ作り方
    spec = drive.speed_ctrl_spec(params)
    st = {'speed_ctrl_spec': spec, 'params': params, 'vbat': {'supported': True}}
    got = V._node(V.SPEC_FNS, 'const st = %s; process.stdout.write(JSON.stringify(ceilConfirmText(speedSpec(st))));' % json.dumps(st))
    keys = ('ceil_all_mps', 'ceil_left_mps')
    text = '全体 %s・左旋回中 %s m/s' % tuple(V._to_fixed1(params[k]) for k in keys)
    assert (spec['ceiling']['all_mps'], spec['ceiling']['left_mps']) == tuple(params[k] for k in keys)
    assert text in got and spec['ceiling']['rule_text'] in got and spec['ceiling']['note'] in got
    assert '全体 4.5・左旋回中 4.0 m/s' not in got


# ================================================================== 重なり(headless の Firefox)
MEASURE = r'''
const [txt, err, driving] = arguments;
document.documentElement.style.scrollbarWidth = 'none';     // スマホの重ねるスクロールバー(幅を取らない)と同じに
document.querySelectorAll('.tab-page').forEach((p) => p.classList.toggle('active', p.id === 'tab-drive'));
const f = document.querySelector('#btn-float-stop'), t = document.querySelector('#toast');
document.body.classList.toggle('driving', driving);        // app.js の renderDrive と同じ(走行中 = body.driving・■ を出す)
f.classList.toggle('hidden', !driving);
t.textContent = txt; t.className = 'toast' + (err ? ' error' : '');
const R = (b) => [b.left, b.top, b.width, b.height].map((v) => Math.round(v * 10) / 10);
const vw = document.documentElement.clientWidth, lines = [], tn = f.firstChild;
let out = '';
for (let i = 0; i < tn.textContent.length; i++) {
  const rg = document.createRange(); rg.setStart(tn, i); rg.setEnd(tn, i + 1);
  const rs = rg.getClientRects(); if (!rs.length) continue;
  const top = Math.round(rs[0].top);
  let L = lines.find((l) => Math.abs(l.top - top) < 4);
  if (!L) { L = {top, text: ''}; lines.push(L); }
  L.text += tn.textContent[i];
  if (rs[0].left < 0 || rs[0].right > vw + 0.5) out += tn.textContent[i];
}
return {vw, vh: innerHeight, f: R(f.getBoundingClientRect()), t: R(t.getBoundingClientRect()), lines: lines.map((l) => l.text), out,
        fz: getComputedStyle(f).zIndex, tz: getComputedStyle(t).zIndex};
'''


def _vov(a, b):
    """縦の重なり [px](横にも重なるときだけ)"""
    if min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]) <= 0:
        return 0.0
    return max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def _refusal():
    from jetformer.web import server as srv           # 断りの文だけ(import だけ。起動しない)
    import time
    os.environ['TZ'] = 'Asia/Tokyo'
    time.tzset()
    return '走行は始まっていません(停止中)。' + srv._boot_refusal({'boot_at': 1793116404.711}, 1793116404.711 + 3600)


def _measure_all(tmp_path, css_by_tag, sizes, texts):
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    html = re.sub(r'<script src="/static/app.js"></script>', '', html)
    assert html.count('href="/static/style.css"') == 1
    pages = {}
    for tag, css in css_by_tag.items():
        cp = tmp_path / ('%s.css' % tag)
        cp.write_bytes(Path(css).read_bytes())
        (tmp_path / ('%s.html' % tag)).write_text(html.replace('href="/static/style.css"', 'href="%s"' % cp.name), encoding='utf-8')
        for (w, h) in sizes:
            p = tmp_path / ('%s_%dx%d.html' % (tag, w, h))
            p.write_text('<!doctype html><meta charset="utf-8"><body style="margin:0"><iframe src="%s.html" style="width:%dpx;height:%dpx;'
                         'border:0;display:block"></iframe>' % (tag, w, h), encoding='utf-8')
            pages[(tag, w, h)] = p
    port = V._free_port()
    gd = subprocess.Popen([GECKO, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wd = V._WD(port)
    snap_common = os.path.expanduser('~/snap/firefox/common')
    profile = tempfile.mkdtemp(prefix='ui-fix8-r2-profile-', dir=snap_common if os.path.isdir(snap_common) else str(tmp_path))
    res = {}
    try:
        assert V._wait(lambda: wd.req('GET', '/status', timeout=3))
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless', '-profile', profile]}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        wd.req('POST', '/session/%s/window/rect' % wd.sid, {'width': 1400, 'height': 1000})
        for (tag, w, h), p in pages.items():
            wd.req('POST', '/session/%s/url' % wd.sid, {'url': 'file://%s' % p})
            wd.req('POST', '/session/%s/frame' % wd.sid, {'id': 0})
            for name, (txt, err, driving) in texts.items():
                res[(tag, w, h, name)] = wd.req('POST', '/session/%s/execute/sync' % wd.sid, {'script': MEASURE, 'args': [txt, err, driving]})
            wd.req('POST', '/session/%s/frame' % wd.sid, {'id': None})
    finally:
        try:
            wd.req('DELETE', '/session/%s' % wd.sid)
        except Exception:      # noqa: BLE001
            pass
        gd.terminate()
        gd.wait(timeout=30)
        shutil.rmtree(profile, ignore_errors=True)
    return res


@pytest.mark.skipif(not (FIREFOX and GECKO), reason='JF_UI_FIREFOX=1 と geckodriver のときだけ(jf-heavy --lock big --mem 3000M)')
def test_firefox_float_stop_and_driving_toast_on_phone(tmp_path):
    """headless の Firefox(本物の index.html に style.css を file:// で。画面 = 枠なしの iframe。スクロールバーは幅を取らない)。
    走行中の 5 つのトースト(ふつうの ▶・推論の確認・11/3 の「試し」1 つ/3 つ・走行中のスイッチの断り)と、止まっているときの
    2 つ(▶ の断りの出し直し = 本物の server._boot_refusal の文・「停止しました」)を、幅 360・375・390・412・430 × 高さ 660〜900 と
    600×800・700×800・768×900・1280×800・844×390 で測る。
    直す前: スマホの全部で ■ の幅 = 画面の半分・3 行以上、走行中のトーストがどれも ■ に 39 px 以上重なる(指摘)。
    今: スマホの全部で ■ は 2 行まで・画面の中(字も)・幅は画面の 90% まで、トーストは ■ に重ならず 10 px 以上上・画面の中。
    600×800・700×800 は前の 2 行(重なる)が 1 行(重ならない)。768・1280・844×390 は走行中も止まっているときも ■ とトーストの箱が前と同じ。止まっているときのトーストは
    どの大きさでも前と同じ(8-1 の直しのまま)"""
    if str(tmp_path).startswith('/tmp') or '/.' in str(tmp_path):
        pytest.skip('snap の Firefox が読めない場所(--basetemp を /tmp・隠しフォルダの外に)')
    texts = {k: (txt, err, True) for k, (txt, err) in RUN_TOASTS.items()}
    texts.update({'idle_refusal': (_refusal(), True, False), 'idle_stopped': ('停止しました', False, False)})
    sizes = PHONES + MID + WIDE
    res = _measure_all(tmp_path, {'before': BEFORE / 'style.css', 'now': CSS}, sizes, texts)
    run = list(RUN_TOASTS)
    for (w, h) in PHONES:
        for name in run:
            b, n = res[('before', w, h, name)], res[('now', w, h, name)]
            assert b['vw'] == n['vw'] == w and n['vh'] == h
            # 直す前(指摘): ■ は画面の半分の幅・3 行以上、トーストが上の行に重なる
            assert abs(b['f'][2] - w / 2) <= 1 and len(b['lines']) >= 3 and _vov(b['t'], b['f']) >= 39, (w, h, name, b)
            # 今: 2 行まで・画面の中・90% まで、重ならない(10 px 以上上)・トーストも画面の中
            f, t = n['f'], n['t']
            assert 1 <= len(n['lines']) <= 2 and n['out'] == '' and f[0] >= 0 and f[0] + f[2] <= w + 0.5, (w, h, name, n)
            assert f[2] <= 0.9 * w + 0.5 and (len(n['lines']) == 1 or f[2] >= 0.9 * w - 1), (w, h, name, n)
            assert _vov(t, f) == 0 and f[1] - (t[1] + t[3]) >= 10 and t[1] >= 0 and t[0] >= 0 and t[0] + t[2] <= w + 0.5, (w, h, name, n)
            assert (n['fz'], n['tz']) == ('40', '50')
        if w >= 412:
            assert len(res[('now', w, h, 'start')]['lines']) == 1, (w, h)
    for (w, h) in MID:
        for name in run:
            b, n = res[('before', w, h, name)], res[('now', w, h, name)]
            assert len(b['lines']) == 2 and _vov(b['t'], b['f']) > 0, (w, h, name, b)                  # 前: 2 行・重なる
            assert len(n['lines']) == 1 and _vov(n['t'], n['f']) == 0 and n['f'][1] - (n['t'][1] + n['t'][3]) >= 10, (w, h, name, n)
    for (w, h) in WIDE:
        for name in texts:
            b, n = res[('before', w, h, name)], res[('now', w, h, name)]
            assert (b['f'], b['t'], b['lines']) == (n['f'], n['t'], n['lines']), (w, h, name, b, n)
            if texts[name][2]:
                assert len(n['lines']) == 1 and _vov(n['t'], n['f']) == 0
    for (w, h) in sizes:
        for name in ('idle_refusal', 'idle_stopped'):
            b, n = res[('before', w, h, name)], res[('now', w, h, name)]
            assert b['t'] == n['t'] and n['f'][2] == 0, (w, h, name, b, n)        # 止まっているとき: トーストは前と同じ・■ は出ない


# ================================================================== rehearsal-r2-vbefore-screen-vs-server(app.js vbatSourceText)
VSRC = {'説明欄': '説明欄の V_before(走行のあいだ一定)', 'なし': 'なし(1.00 倍。周ごとの直しだけ)', '基板': '基板の電圧(走り始めから)'}
IDEO, NBSP, FSEP, BOM = chr(0x3000), chr(0xa0), chr(0x1c), chr(0xfeff)


def _fw(s):
    """数字を全角に(８.３１)"""
    return ''.join(chr(ord(c) + 0xfee0) if c.isdigit() else c for c in s)


def _ar(s):
    """数字をアラビア・インドの数字に(Python の float() は読む)"""
    return ''.join(chr(0x660 + int(c)) if c.isdigit() else c for c in s)


# 入力の表: 指摘(831・8,31・0.83)・票 A(9.5・「8.31,」・全角)・票 B(4.15)の形と、決めの境目(範囲の端・末尾の V・2 つ目の V_before・
# Vb_before・空白の字・全角の点・下線・指数・符号・inf/nan・大文字小文字・= のまわりの空白)
VSRC_TABLE = [
    'battery=#A V_before=8.31 check=合格 memo=', 'battery=#A V_before=831 check=合格', 'battery=#A V_before=8,31 check=合格',
    'battery=#A V_before=0.83 check=合格', 'battery=#A V_before=8.31V', 'battery=#A V_before=8.31v', 'battery=#A V_before=8.31VV',
    'battery=#A V_before=__ V_after=__ check=__ memo=', 'battery=#A V_before=' + _fw('8.31'), 'battery=#A V_before=' + _fw('8') + chr(0xff0e) + _fw('31'),
    'battery=#A V_before=9.5', 'battery=#A V_before=8.31, check=合格', 'battery=#A V_before=4.15', 'V_before=6.0', 'V_before=9.0',
    'V_before=5.999', 'V_before=9.001', 'V_before=6', 'V_before=8.', 'V_before=.831e1', 'V_before=8.31e0', 'V_before=8_31',
    'V_before=8.3_1', 'V_before=8__3', 'V_before=+8.31', 'V_before=-8.31', 'V_before=nan V_before=8.31', 'V_before=x V_before=8.31',
    'V_before=8.31 V_before=9.9', 'V_before=inf', 'V_before=Infinity', 'Vb_before=8.27', 'V_before=__ Vb_before=8.27',
    'V_before=9.5 Vb_before=8.27', 'V_before=8.31 Vb_before=8.27', 'v_before=8.31', 'V_before =8.31', 'V_before= 8.31',
    'V_before=8.31' + IDEO + 'check=合格', 'V_before=8.31' + NBSP + 'check=合格', 'V_before=8.31' + FSEP + 'check',
    'V_before=8.31' + BOM, BOM + 'V_before=8.31', 'memo=V_before=8.31', 'xV_before=8.31', 'V_before=8.31memo', 'V_before=' + _ar('8.31'),
    'V_before=8.31 V', '', 'V_before=', 'battery=#A V_before=8.31\ncheck=合格', 'V_before=0x8', 'V_before=8.31e', 'V_before=1e0_0',
    'V_before=8.31\tcheck', 'ai_no=3 V_before=7.95 cand=B memo=計画と違うので試し(元 cand=B): throttle_gain 0.9',
]


def _server_src(desc):
    """サーバーの電圧の源(基板の電圧なし): 本物の drive._SpeedAuto に、走行開始でサーバーが頭に ai_no= を付けた説明欄を渡す(計算だけ)"""
    from jetformer import speed_auto
    from jetformer.web import bench as bench_web
    sa = drive._SpeedAuto(dict(drive.DEFAULT_PARAMS, speed_ceiling=1, speed_auto=1), bench_web.with_ai_no(desc, 7))
    sa.gain(1_000_000_000, 1.00, 0.0, [])                         # 止まっている・基板の電圧のサンプルなし
    name = {speed_auto.SRC_NONE: 'なし', speed_auto.SRC_DESC: '説明欄', speed_auto.SRC_BOARD: '基板'}[sa.src]
    assert (name == '説明欄') == (speed_auto.v_before_of(desc) == speed_auto.v_before_of(desc))       # v_before_of が数のときだけ説明欄
    return name


def _vsrc_parts(js):
    vst = re.search(r'function vbatSourceText\(st\) \{.*?\n\}', js, re.S).group(0)
    pyf = re.search(r'\n  (const pyFloat = \(s\) => \{.*?\n  \};)', js, re.S)
    sp = re.search(r'\n  (const PY_SPACE = /.*?/;)', js)
    return vst, (pyf.group(1) if pyf else None), (sp.group(1) if sp else None)


VSRC_HARNESS = r'''
const fs = require('fs');
const IN = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));
let CUR = '';
const $ = (q) => ({ value: CUR });
const enc = (x) => (x === null ? null : Number.isNaN(x) ? 'nan' : x === Infinity ? 'inf' : x === -Infinity ? '-inf' : x);
const out = { src: IN.descs.map((d) => { CUR = d; return vbatSourceText({ vbat: { supported: false } }); }),
              board: IN.descs.map((d) => { CUR = d; return vbatSourceText({ vbat: { supported: true } }); }) };
if (IN.floats) out.flt = IN.floats.map((s) => enc(pyFloat(s)));
if (IN.allcp) {                       // 1 字の文で pyFloat が数を返す字・PY_SPACE に入る字(サロゲートを除く全部の字)
  out.cpnum = []; out.cpspace = [];
  for (let c = 0; c < 0x110000; c++) {
    if (c >= 0xd800 && c <= 0xdfff) continue;
    const ch = String.fromCodePoint(c);
    const x = pyFloat(ch);
    if (x !== null) out.cpnum.push([c, enc(x)]);
    if (PY_SPACE.test(ch)) out.cpspace.push(c);
  }
}
process.stdout.write(JSON.stringify(out));
'''


def _node_vsrc(tmp_path, js_path, descs, floats=None, allcp=False):
    vst, pyf, sp = _vsrc_parts(Path(js_path).read_text(encoding='utf-8'))
    extra = '\n'.join(x for x in (pyf, sp) if x and (floats is not None or allcp))
    p, q = tmp_path / 'vsrc.js', tmp_path / 'vsrc_in.json'
    p.write_text(vst + '\n' + extra + '\n' + VSRC_HARNESS, encoding='utf-8')
    q.write_text(json.dumps({'descs': list(descs), 'floats': floats, 'allcp': allcp}, ensure_ascii=False), encoding='utf-8')
    r = subprocess.run([NODE, str(p), str(q)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def _short(text):
    return next(k for k, v in VSRC.items() if v == text)


def _py_float(s):
    try:
        x = float(s)
    except ValueError:
        return None
    return 'nan' if x != x else ('inf' if x == float('inf') else ('-inf' if x == float('-inf') else x))


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_vbefore_screen_matches_server_table(tmp_path):
    """入力の表(56 行)の全部で、画面の電圧の源(本物の app.js の vbatSourceText。基板の電圧なし)= サーバーの電圧の源(本物の
    drive._SpeedAuto・speed_auto.v_before_of)。基板の電圧があれば前と同じ「基板の電圧(走り始めから)」。
    直す前の app.js(写し)では指摘の形が出る: 831・8,31・0.83・9.5・「8.31,」・4.15 で画面「説明欄」・サーバー「なし」、全角は逆"""
    server = [_server_src(d) for d in VSRC_TABLE]
    now = _node_vsrc(tmp_path, APP, VSRC_TABLE)
    assert [_short(x) for x in now['src']] == server
    assert set(now['board']) == {VSRC['基板']}
    assert server.count('説明欄') >= 20 and server.count('なし') >= 20
    old = _node_vsrc(tmp_path, BEFORE / 'app.js', VSRC_TABLE)
    bad = {d: (_short(a), b) for d, a, b in zip(VSRC_TABLE, old['src'], server) if _short(a) != b}
    for d in ('battery=#A V_before=831 check=合格', 'battery=#A V_before=8,31 check=合格', 'battery=#A V_before=0.83 check=合格',
              'battery=#A V_before=9.5', 'battery=#A V_before=8.31, check=合格', 'battery=#A V_before=4.15'):
        assert bad[d] == ('説明欄', 'なし'), d
    assert bad['battery=#A V_before=' + _fw('8.31')] == ('なし', '説明欄')
    assert len(bad) == 23, bad


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_vbefore_random_descriptions_match_server(tmp_path):
    """でたらめな説明欄 3000 本(種 82)で、画面の電圧の源 = サーバーの speed_auto.v_before_of が数か(基板の電圧なし)。1 本は 1〜4 個の欄
    (V_before=・Vb_before=・v_before=・「V_before =」などの鍵 + 5.5〜9.5 V あたりの数に、下線・コンマ・V・e・点・符号・空白・全角・
    アラビアの数字・全角の点を 0〜2 個まぜた値、か battery=・check=・memo= などのかけら)を、いろいろな空白の字でつないだもの"""
    import random
    from jetformer import speed_auto
    rnd = random.Random(82)
    marks = ['_', ',', 'V', 'v', 'e', '0', '.', '+', '-', ' ', _fw('8'), _ar('3'), chr(0xff0e), BOM]

    def num():
        x = rnd.choice(['%.2f' % rnd.uniform(5.5, 9.5), '%.1f' % rnd.uniform(5.5, 9.5), '%d' % rnd.randint(5, 10), '%.3f' % rnd.uniform(0, 12),
                        'nan', 'inf', '__', ''])
        for _ in range(rnd.choice([0, 0, 1, 2])):
            i = rnd.randrange(len(x) + 1)
            x = x[:i] + rnd.choice(marks) + x[i:]
        if rnd.random() < 0.1:
            x = _fw(x)
        return x + rnd.choice(['', '', '', 'V', 'v', 'VV'])

    def field():
        if rnd.random() < 0.3:
            return rnd.choice(['battery=#A', 'check=合格', 'memo=', 'V_after=8.05', 'cand=B', '__', 'ai_no=3', 'memo=V_before=8.31'])
        return rnd.choice(['V_before=', 'V_before=', 'V_before=', 'Vb_before=', 'v_before=', 'V_before =', 'xV_before=', 'Vb_before=']) + num()

    seps = [' ', ' ', ' ', IDEO, NBSP, FSEP, '\t', '\n', BOM + ' ', '  ']
    descs = [''.join(field() + rnd.choice(seps) for _ in range(rnd.randint(1, 4))) for _ in range(3000)]
    got = _node_vsrc(tmp_path, APP, descs)
    want = ['説明欄' if speed_auto.v_before_of(d) == speed_auto.v_before_of(d) else 'なし' for d in descs]
    bad = [(d, a, b) for d, a, b in zip(descs, (_short(x) for x in got['src']), want) if a != b]
    assert not bad, bad[:5]
    assert want.count('説明欄') >= 400 and want.count('なし') >= 1500           # 種 82 では 499 本と 2501 本


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_pyfloat_is_python_float_and_py_space_is_python_isspace(tmp_path):
    """app.js の pyFloat(s) は Python の float(s) と同じ値(読めない = null ⇔ ValueError)。手で書いた形と、でたらめな 5000 本(種 83)と、
    1 字の文の全部の字(サロゲートを除く。数になるのは Unicode の 10 進の字だけ。違ってよいのは Python 3.13 の Unicode 15.1 に無い新しい
    数字だけ)。PY_SPACE に入る字 = Python の str.isspace() の字(全部の字)"""
    import random
    import sys
    import unicodedata
    rnd = random.Random(83)
    alpha = list('0123456789._eE+-VvnaifINFty, x#') + [_fw('8'), chr(0xff0e), _ar('3'), IDEO, chr(0x2212)]
    floats = ['8.31', '831', '8,31', '0.83', '.5', '5.', '.', '', '_', '8_3', '8__3', '_8', '8_', '8._3', '8_.3', '1e5', '1e', 'e5', '1e+5',
              '1E-5', '1e_5', '1e5_0', '+8', '-8', '+-8', 'inf', '-Infinity', 'INF', 'nan', '+NaN', 'infinit', 'nann', '0x8', '1_000.000_1',
              _fw('8.31'), _ar('8.31'), _fw('8') + chr(0xff0e) + _fw('3'), '8.31' + chr(0x2212), '00008.3100', '8.31e0', '9' * 400]
    floats += [''.join(rnd.choice(alpha) for _ in range(rnd.randint(1, 8))) for _ in range(5000)]
    got = _node_vsrc(tmp_path, APP, [], floats=floats, allcp=True)
    assert got['flt'] == [_py_float(s) for s in floats]
    py_num = {}
    for c in range(0x110000):
        if 0xd800 <= c <= 0xdfff:
            continue
        x = _py_float(chr(c))
        if x is not None:
            py_num[c] = x
    js_num = {c: x for c, x in got['cpnum']}
    diff = {c for c in set(py_num) | set(js_num) if py_num.get(c) != js_num.get(c)}
    assert all(unicodedata.category(chr(c)) == 'Cn' for c in diff), sorted(diff)[:10]       # Python の Unicode に無い字だけ
    assert len(py_num) >= 600 and all(js_num[c] == py_num[c] for c in py_num)
    assert set(got['cpspace']) == {c for c in range(0x110000) if not 0xd800 <= c <= 0xdfff and chr(c).isspace()}
    assert sys.version_info[:2] == (3, 13)              # jetformer-web と同じ python(scripts/run_web.sh = miniconda の jetracer の env)


def test_vdesc_range_is_the_server_range():
    """app.js の VDESC(6.0〜9.0)は speed_auto.AutoParams の vdesc_lo・vdesc_hi(サーバーが説明欄の電圧に使う範囲)"""
    from jetformer import speed_auto
    m = re.findall(r'^  const VDESC = \[([\d.]+), ([\d.]+)\];', APP.read_text(encoding='utf-8'), re.M)
    p = speed_auto.AutoParams()
    assert [tuple(map(float, x)) for x in m] == [(p.vdesc_lo, p.vdesc_hi)] == [(6.0, 9.0)]


@pytest.mark.skipif(NODE is None, reason='node がありません')
@pytest.mark.parametrize('desc', ['battery=#A V_before=8.31 check=合格 memo=', 'battery=#A V_before=831 check=合格',
                                  'battery=#A V_before=8,31', 'battery=#A V_before=' + _fw('8.31'), 'battery=#A V_before=__ memo='])
def test_start_and_switch_confirm_texts_say_what_the_server_uses(desc):
    """▶ の確認の文(speedStartLines の「速さのゲインの自動合わせ: ON(… 電圧の源 X)」)と、自動合わせのスイッチの確認の文
    (autoConfirmText の「電圧の源: X」)の X は、同じ説明欄でサーバーが使う電圧の源(基板の電圧なし)。本物の app.js の関数を node で"""
    js = APP.read_text(encoding='utf-8')
    names = ('SPEED_FALLBACK', 'speedSpec', 'ceilSpecText', 'autoLapText', 'vbatSourceText', 'autoConfirmText', 'speedStartLines')
    params = dict(drive.DEFAULT_PARAMS, speed_ceiling=1, speed_auto=1)
    st = {'speed_ctrl_spec': drive.speed_ctrl_spec(params), 'params': params, 'vbat': {'supported': False}}
    code = (V.STUBS + ''.join(V._extract(js, n) for n in names)
            + "$('#dr-desc').value = %s; $('#dr-speed-ceiling').checked = true; $('#dr-speed-auto').checked = true;\n"
              "const st = %s; process.stdout.write(JSON.stringify([speedStartLines(st), autoConfirmText(speedSpec(st), st)]));"
            % (json.dumps(desc, ensure_ascii=False), json.dumps(st, ensure_ascii=False)))
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    start, sw = json.loads(r.stdout)
    want = VSRC[_server_src(desc)]
    assert '・電圧の源 %s)' % want in start and '電圧の源: %s\n' % want in sw
    other = [v for k, v in VSRC.items() if v != want]
    assert not any(o in start or o in sw for o in other)


def test_app_js_only_vbat_source_text_changed():
    """app.js で変えたのは vbatSourceText の中だけ: 消した行は前の 1 行(V_before= のあとが数字で始まるか)、足した行は全部この関数の中。
    出す文は前と同じ 3 つ。足した行に送る物・問い合わせ・タイマー・トースト・確認の文の呼び出しは無い。版は 9 のまま。
    index.html・server.py・bench.py・speed_auto.py・drive.py は変えていない"""
    old = (BEFORE / 'app.js').read_text(encoding='utf-8')
    removed, added, ops = _diff(BEFORE / 'app.js', APP)
    assert removed == ["  return /(^|\\s)V_before=\\d/.test($('#dr-desc').value || '') ? '説明欄の V_before(走行のあいだ一定)' : "
                       "'なし(1.00 倍。周ごとの直しだけ)';"]
    js = APP.read_text(encoding='utf-8')
    fn = re.search(r'function vbatSourceText\(st\) \{.*?\n\}', js, re.S).group(0)
    assert all(x in fn.split('\n') for x in added) and len(ops) == 1
    code = [x.split('//')[0] for x in added]
    assert not any(re.search(r"\bapi\(|'POST'|fetch\(|setTimeout|toast\(|confirm\(|body|client", x) for x in code), added
    texts = re.findall(r"'([^']*)'", fn.split('// ')[0]) + list(re.findall(r"return ok \? '([^']*)' : '([^']*)';", fn)[0])
    assert texts == [VSRC['基板'], VSRC['説明欄'], VSRC['なし']]
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert js.replace(fn, '') == old.replace(re.search(r'function vbatSourceText\(st\) \{.*?\n\}', old, re.S).group(0), '')
    for rel, h in (('jetformer/web/server.py', 'ad7a7d05f7a9'), ('jetformer/web/static/index.html', 'ef81bb6017d6'),
                   ('jetformer/web/bench.py', 'a633bec341a3'), ('jetformer/drive.py', '3f2e31a22ecb')):
        assert _sha(ROOT / rel).startswith(h), rel
