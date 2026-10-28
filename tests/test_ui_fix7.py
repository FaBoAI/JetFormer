"""画面の係の直し 7-1 回目(2026-10-27 fix7。反証の指摘 w1-ui-r7-ui-1・w1-ui-r7-ui-2。~/jetformer_reports/fix_20261026/fix7/fix_ui_r1.md)の試験。

  w1-ui-r7-ui-1: 起き直しの知らせ(noteServerBoot のトーストと走行タブの文)が、そのときのサーバーの値を見ずに、トーストは「朝に入れた
    設定(…)は既定に戻っています」、走行タブの文は試験の選択肢を全部「…は OFF」と書き、天井の左・発進の前進の強さは今の値を「起動の
    たびに既定に戻るもの」として出した。ライブのタブにいる間に起き直し、走行タブを開く前に curl・別の画面で朝の設定を入れ直すと、
    ON を OFF と書き、4.2・200 を既定と読ませた(既定は 4.1・250)。▶ の断りの文の出し直し(fix6)も同じ固定の文を後ろに付けた。
  直し(app.js。表示だけ): 起動のたびに既定に戻る項目(drive.RESET_ON_BOOT)が全部既定なら前とまったく同じ文。既定と違う項目が
    あれば、トーストは今の値を挙げる(「起動のあとに入れ直した設定があります(今は …)。朝の設定はスイッチと走行開始の確認の文で読む」)。
    走行タブの文は ON の選択肢を「ON(起動のあとに入れ直し済み)」、天井の左・発進の前進の強さを「今 X(既定 Y)」、ほかの既定と違う
    項目を「(ほかに今は既定と違うもの: …)」と書き、表示のたびに今の st.params から作り直す(知らせのあとに入れ直したときもスイッチと
    同じ)。▶ の出し直しはトーストと同じ文。送る物・断る決まり・スイッチ・確認の文は前と同じ。
  w1-ui-r7-ui-2: 試験の台本(tests/fixtures/app_sim_fix6_r2.js ほか)の 'click'・'slide' が、短いタイマーを鳴らすときに長いタイマー
    (表示の問い合わせの輪など)を全部捨て、それ以降は輪が回らなかった(本物のブラウザと違う)。直し: 新しい写し
    tests/fixtures/app_sim_fix7.js(もとは変えない)で、鳴らさないタイマーを一覧に残し、鳴らす前に消されたタイマーは鳴らさない。
    この試験の動きは全部この写しで流す。6-1・6-2 回目の結論(通るときの流れ・走行中・起き直しの場面)がこの写しでも変わらないことも見る。

本物の app.js(今の版と、直す前の写し tests/fixtures/fix7_ui_before/app.js = 直す 6-2 回目のあとの版 908934c6…。sha256 を照らす)を
node の vm で。409 の文は本物の server._boot_refusal で作る(import だけ。起動しない)。車・ブラウザ・jetformer-web には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix7.py

JF_UI7_APP=<app.js の写し> でその写しを「今の版」として、JF_UI7_SIM=<台本の写し> でその台本を流す(わざと壊す台本
experiments/20261026_fix/fix7/ui_r1/mut/ が使う)。
そのあと直す 7-2 回目(w2-ui-w2-ui-r72-ui-1・-2。tests/test_ui_fix7_r2.py)が app.js を直した。動きの試験は今の app.js で流す(7-2 回目の
直しは ▶ の catch の同期が通らないときと、走行タブを開いたときのモデルの一覧の GET が通らないときだけ効くので、ここの場面では前と同じ)。
「違いは 3 か所だけ」は 7-1 回目のあとの版の写し(tests/fixtures/fix7_ui_r2_before/)で見る。
"""
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive
from jetformer.web import server as srv

from test_ui_fix3 import _flows
import test_ui_fix6 as F6
import test_ui_fix6_r2 as F62

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
FIXT = ROOT / 'tests' / 'fixtures'
APP = Path(os.environ.get('JF_UI7_APP') or STATIC / 'app.js')
SIM = Path(os.environ.get('JF_UI7_SIM') or FIXT / 'app_sim_fix7.js')
OLD_SIM = FIXT / 'app_sim_fix6_r2.js'                  # 直す前の台本(長いタイマーを捨てる。変えていない)
BEFORE = FIXT / 'fix7_ui_before'                       # 直す前の app.js(6-2 回目のあとの版)
# 2026-10-28 fix8 tools-decided のあとの版の sha256(このあとの直しはその係がここに足す)
LATER_FIX8_TOOLS_DECIDED = {'experiments/20261026_fix/plan/runs_1103.py': '33bc7fca99216cc2d4704e8fa5de9741626231fd4a0d165cb981183801acba00'}
# 2026-10-28 fix8 画面の係 直す 8-1 回目(tests/test_ui_fix8_r1.py)のあとの版の sha256。その版なら直す前の写し tests/fixtures/fix8_ui_r1_before/
# (= 上の版)へたどる(中身の違いは tests/test_ui_fix8_r1.py)
FIX8_UI_R1_BEFORE = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r1_before'
LATER_FIX8_UI_R1 = {'experiments/20261026_fix/plan/runs_1103.py': 'c2d616c2f72b00935868691236f94e4733c3185a24d200c524b6049d615fd9fc',
                    'jetformer/web/static/style.css': 'dbc8b79ad935ed914c7b8a2cc02f8c4ddaf0c453af33ca3fb85fa0ea1103df56',
                    'jetformer/web/static/app.js': 'aa275bb5da0bbd740f38c217dd0ae9ee19240f061dafd8ca70f6916448837285'}
# 2026-10-28 fix8 画面の係 直す 8-2 回目(tests/test_ui_fix8_r2.py。浮いている ■ の幅と走行中のトーストの位置)のあとの style.css の sha256。
# その版なら直す前の写し tests/fixtures/fix8_ui_r2_before/(= 上の LATER_FIX8_UI_R1 の版)へたどる(中身の違いは tests/test_ui_fix8_r2.py)
FIX8_UI_R2_BEFORE = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r2_before'
LATER_FIX8_UI_R2 = {'jetformer/web/static/style.css': 'b1e900656e2463c3c2292fba885bb415298a286aa131db25f481f5ecbce1924a'}
R61 = FIXT / 'fix6_ui_r2_before'                       # 6-1 回目のあとの版(6-2 回目の直す前)
# この組(7-1 回目)のあとの版の写し(3cf098d9…)。そのあと 7-2 回目(tests/test_ui_fix7_r2.py)が app.js を直したので、「違いは 3 か所
# だけ」はこの写しで見る(今の app.js との違いは test_ui_fix7_r2.py が見る)。JF_UI7_APP を付けたときはその写し
R1_AFTER = FIXT / 'fix7_ui_r2_before'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
ST = '/api/drive/status'
PREFIX = F6.PREFIX
BOOT_TOAST = F6.BOOT_TOAST                              # 全部既定のときのトースト(前と同じ文)
TAIL_NOTE = '朝に入れた値は、止まっているときに入れ直し、スイッチと走行開始の確認の文で読む(手順書 7 章)。画面を読み直すとこの文は消えます'
HEAD = '管理画面のサーバーが 09:33 に起動し直しました。'
GUARD, KEEP, NOKICK = 'リカバリーの後ろの見張り', '中立止めのあともリカバリーを残す', '発進の前進を出さない区間は発進の補助も出さない'
DEFAULT_NOTE = (HEAD + f'起動のたびに既定に戻るもの: {GUARD}・{KEEP}・{NOKICK}は OFF、天井の左旋回中 4.1 m/s、発進の前進の強さ 250 µs など。'
                + TAIL_NOTE)
REENTER = {'rec_back_guard': 1, 'rec_hold_keep': 1, 'ceil_left_mps': 4.2, 'launch_us': 200.0}     # 手順書 3 章の朝の設定(11/3 の形)
REENTERED_NOTE = (HEAD + f'起動のたびに既定に戻るもの: {GUARD}・{KEEP}は ON(起動のあとに入れ直し済み)、{NOKICK}は OFF、'
                  '天井の左旋回中 今 4.2 m/s(既定 4.1)、発進の前進の強さ 今 200 µs(既定 250) など。' + TAIL_NOTE)
REENTERED_TOAST = (HEAD + f'起動のあとに入れ直した設定があります(今は {GUARD} ON・{KEEP} ON・天井の左旋回中 4.2 m/s・発進の前進の強さ 200 µs)。'
                   '朝の設定はスイッチと走行開始の確認の文で読む')
OLD_BUG_NOTE = (HEAD + f'起動のたびに既定に戻るもの: {GUARD}・{KEEP}・{NOKICK}は OFF、天井の左旋回中 4.2 m/s、発進の前進の強さ 200 µs など。'
                + TAIL_NOTE)
MORNING = F6.MORNING                                    # 見張り・残す ON・左 4.2・launch_us 200・速度 0.9(boot_at 1000)
REAL = dict(F6.REAL, **{'2000->3000': srv._boot_refusal({'boot_at': 2000.0}, 3000.0), 'null->3000': srv._boot_refusal({}, 3000.0)})

OLD_LINES = [
    "  const opts = OPT_SWITCHES.filter((o) => p[o.key] !== undefined).map((o) => o.name);",
    "  state.drive.bootNote = `管理画面のサーバーが ${hm} に起動し直しました。起動のたびに既定に戻るもの: `",
    "    + (opts.length ? `${opts.join('・')}は OFF、` : '')",
    "    + `天井の左旋回中 ${p.ceil_left_mps ?? '-'} m/s、発進の前進の強さ ${p.launch_us ?? '-'} µs など。`",
    "    + '朝に入れた値は、止まっているときに入れ直し、スイッチと走行開始の確認の文で読む(手順書 7 章)。画面を読み直すとこの文は消えます';",
    "  const msg = `管理画面のサーバーが ${hm} に起動し直しました。朝に入れた設定(後ろの見張り・左の値・発進の前進の強さなど)は既定に戻っています`;",
    "  state.drive.bootToast = msg;                        // 走行開始の断りの文を出し直すときに使う(▶ の catch。2026-10-27 fix6)",
    "    + (state.drive.bootNote ? `<div class=\"warn-text\">${esc(state.drive.bootNote)}</div>` : '');",
]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sums(d):
    return dict(reversed(x.split()) for x in (d / 'SHA256SUMS').read_text().splitlines() if x.strip())


def test_before_copy_and_only_the_boot_note_changed():
    """直す前の写しは 6-2 回目のあとの版(908934c6…)。この組のあとの版(3cf098d9…。写し tests/fixtures/fix7_ui_r2_before/。7-2 回目の
    直す前)との違いは noteServerBoot のまわり(文を作る関数と既定の表を足し、
    トーストと走行タブの文をそこから作る)と、走行タブの文を出す 1 行だけ。送る物(api・body・client)・版(9)は変えていない。
    server.py・index.html・style.css・bench.py・runs_1103.py は変えていない"""
    assert _sums(BEFORE) == {'app.js': _sha(BEFORE / 'app.js')} and _sha(BEFORE / 'app.js').startswith('908934c6297b')
    after = APP if os.environ.get('JF_UI7_APP') else R1_AFTER / 'app.js'       # 7-2 回目の直す前の写し(7-1 回目のあとの版)
    if not os.environ.get('JF_UI7_APP'):
        assert _sums(R1_AFTER) == {'app.js': _sha(after)} and _sha(after).startswith('3cf098d93386')
    a = (BEFORE / 'app.js').read_text(encoding='utf-8').splitlines()
    b = after.read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    removed = [a[i] for o in ops for i in range(o[1], o[2])]
    added = [b[j] for o in ops for j in range(o[3], o[4])]
    assert removed == OLD_LINES, removed
    assert len(ops) == 3, ops                       # 足した関数と表・noteServerBoot の中・走行タブの文の 1 行
    code = [x.split('//')[0] for x in added]
    assert not any(re.search(r"\bapi\(|'POST'|body\.|const body|client:|fetch\(|setTimeout|clearTimeout|pollDrive|drLost|confirm\(", x)
                   for x in code), added                                       # 送る物・問い合わせ・確認の文は足していない
    assert sum('toast(' in x for x in code) == 0                                # トーストを出すのは前からの 1 か所だけ
    js = after.read_text(encoding='utf-8')
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert js.count('toast(msg, true);') == 1 and js.count('state.drive.bootToast = msg;') == 1
    assert _sha(ROOT / 'jetformer' / 'web' / 'server.py').startswith('ad7a7d05f7a9')
    assert _sha(STATIC / 'index.html').startswith('ef81bb6017d6')
    css = STATIC / 'style.css'                    # fix8 ui r1(.toast の幅)の版なら、その直す前の写しを見る
    if _sha(css) == LATER_FIX8_UI_R2['jetformer/web/static/style.css']:     # fix8 ui r2(■ の幅)の版 → その直す前の写し(dbc8b79a…)
        css = FIX8_UI_R2_BEFORE / 'style.css'
    if _sha(css) == LATER_FIX8_UI_R1['jetformer/web/static/style.css']:
        css = FIX8_UI_R1_BEFORE / 'style.css'
    assert _sha(css).startswith('5842642f6e85')
    assert _sha(ROOT / 'jetformer' / 'web' / 'bench.py').startswith('a633bec341a3')
    # 2026-10-28 fix8 tools-decided が runs_1103.py を変えた(試験の選択肢の印の後ろの文と説明だけ)。その版なら直す前の写し
    # (tests/fixtures/fix8_tools_decided_before/runs_1103.py.orig)がこの組の版と同じかを見る(中身の違いは tests/test_fix8_tools_decided.py)
    rp = ROOT / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'
    if _sha(rp) == LATER_FIX8_UI_R1['experiments/20261026_fix/plan/runs_1103.py']:      # fix8 ui r1 の版 → その直す前の写し(33bc7fca…)
        rp = FIX8_UI_R1_BEFORE / 'runs_1103.py'
    if _sha(rp) == LATER_FIX8_TOOLS_DECIDED['experiments/20261026_fix/plan/runs_1103.py']:
        rp = ROOT / 'tests' / 'fixtures' / 'fix8_tools_decided_before' / 'runs_1103.py.orig'
    assert _sha(rp).startswith('302296586af9')


def _js_table(js, name):
    body = re.search(r'^const %s = \{(.*?)\};' % name, js, re.M | re.S).group(1)
    return {k: float(v) for k, v in re.findall(r'(\w+): ([-\d.]+)', body)}


def test_boot_defaults_are_drive_defaults():
    """画面の既定の表(BOOT_DEFAULTS)は drive.DEFAULT_PARAMS の RESET_ON_BOOT の分とまったく同じ(項目も値も)。ON/OFF の項目
    (BOOT_FLAGS)は PARAM_LIMITS が (0, 1) の項目。既定の値・RESET_ON_BOOT・PARAM_LIMITS は変えていない(前の値と照らす)"""
    js = APP.read_text(encoding='utf-8')
    tbl = _js_table(js, 'BOOT_DEFAULTS')
    assert tbl == {k: float(drive.DEFAULT_PARAMS[k]) for k in drive.RESET_ON_BOOT}
    flags = re.search(r"^const BOOT_FLAGS = \[(.*?)\];", js, re.M).group(1)
    assert set(re.findall(r"'(\w+)'", flags)) == {k for k in drive.RESET_ON_BOOT if tuple(drive.PARAM_LIMITS[k]) == (0, 1)}
    units = re.search(r"^const BOOT_UNITS = \{(.*?)\};", js, re.M).group(1)
    assert set(re.findall(r'(\w+): ', units)) == {'ceil_left_mps', 'launch_us', 'ceil_all_mps', 'auto_target_mps', 'launch_mps', 'launch_max_s'}
    assert drive.RESET_ON_BOOT == ('tcp_mode', 'speed_scale', 'zone_caps', 'speed_ceiling', 'ceil_all_mps', 'ceil_left_mps', 'speed_auto',
                                   'auto_target_mps', 'launch_boost', 'launch_us', 'launch_mps', 'launch_max_s', 'rec_back_guard',
                                   'rec_hold_keep', 'launch_no_kick')
    assert {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT} == {
        'tcp_mode': 0, 'speed_scale': 1.0, 'zone_caps': 0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
        'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'rec_back_guard': 0,
        'rec_hold_keep': 0, 'launch_no_kick': 0}


# ------------------------------------------------------------------ 画面を node で流す
cands = F6.cands
blk = F6.blk
server_params = F6.server_params


def run_sim(tmp_path, c, steps, app=None, sim=None, name='input.json', server=None, **extra):
    inp = tmp_path / name
    d = {'bench': c, 'server_params': server if server is not None else MORNING, 'steps': steps, 'server_client_version': 9,
         'confirm': True, 'boot_at': 1000.0, 'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True, 'boot_refusal_texts': REAL}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(sim or SIM), str(app or APP), str(STATIC / 'index.html'), str(inp)], capture_output=True, text=True,
                       timeout=120, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def both(tmp_path, c, steps, **kw):
    """今の版と直す前の写しで同じ操作を流す(どちらも直した台本 app_sim_fix7.js)"""
    return (run_sim(tmp_path, c, steps, name='n.json', **kw), run_sim(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json', **kw))


def vals(res, op):
    return [x.get('value') for x in res if x['op'] == op]


def tagged(res):
    return [x['value'] for x in res if x['op'] in ('el', 'eval')]


def starts(res):
    return [x for x in res if x['op'] == 'start']


def note_of(html):
    """走行タブの文(#dr-message の最後の warn-text)。起き直しの文が無ければ None"""
    parts = (html or '').split('<div class="warn-text">')
    last = parts[-1].replace('</div>', '') if len(parts) > 1 else ''
    return last if last.startswith('管理画面のサーバーが ') else None


def boot_toasts(toasts):
    return [t for t in toasts if re.match(r'^管理画面のサーバーが \d\d:\d\d に起動し直しました。', t)]


def same_sent(new, old):
    assert vals(new, 'posts') == vals(old, 'posts')
    assert [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(new)] == \
           [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(old)]


LOOK = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-rec-back-guard', 'checked'],
        ['el', '#dr-rec-hold-keep', 'checked'], ['el', '#dr-launch-no-kick', 'checked'], ['el', '#dr-ceil-spec', 'text'],
        ['el', '#dr-launch-spec', 'text'], ['read', 'server']]


@needs_node
def test_reentered_before_opening_the_drive_tab(tmp_path, cands):
    """指摘の場面: 朝の設定(見張り・残す ON・左 4.2・launch_us 200)→ ライブのタブ → 起き直す(既定に戻る)→ 走行タブを開く前に
    手順書 3・5 章の curl で朝の設定を入れ直す → 走行タブ。直す前の写しはトースト「…既定に戻っています」・走行タブの文「…は OFF、
    天井の左旋回中 4.2 m/s、発進の前進の強さ 200 µs」(ON を OFF、4.2・200 を既定のように)。今の版はトーストが今の値を挙げ、走行タブの
    文は「…は ON(起動のあとに入れ直し済み)」「今 4.2 m/s(既定 4.1)」「今 200 µs(既定 250)」。スイッチ・天井と発進の前進の横の文・
    サーバーはどちらも ON・ON・4.2・200。送った物(curl・そのあとの ▶ の body と返事)・確認の文は同じ"""
    c, di = cands
    steps = ([['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, RESET], ['api', REENTER], ['tab', 'drive']]
             + LOOK + [['tick'], ['el', '#dr-message', 'html'], ['pick', di, blk(c, di, 10)], ['start'], ['posts']])
    (new, tn), (old, to) = both(tmp_path, c, steps)
    n, o = tagged(new), tagged(old)
    assert o[0] == BOOT_TOAST and note_of(o[1]) == OLD_BUG_NOTE                     # 直す前: 指摘が起きる
    assert n[0] == REENTERED_TOAST and note_of(n[1]) == REENTERED_NOTE
    assert note_of(n[7]) == REENTERED_NOTE and note_of(o[7]) == OLD_BUG_NOTE         # 次の問い合わせのあとも同じ
    for x in (n, o):
        assert x[2:7] == [True, True, False, '(全体 4.5・左旋回中 4.2 m/s)', '(前進 200 µs 以上・2.5 m/s まで・最大 1.2 秒)']
    assert vals(new, 'read') == vals(old, 'read') and {k: vals(new, 'read')[0][k] for k in REENTER} == {
        'rec_back_guard': 1, 'rec_hold_keep': 1, 'ceil_left_mps': 4.2, 'launch_us': 200}
    same_sent(new, old)
    assert starts(new)[0]['status'] == 200 and starts(new)[0]['boot_at'] == 2000.0
    assert boot_toasts(tn) == [REENTERED_TOAST] and boot_toasts(to) == [BOOT_TOAST]
    assert [t for t in tn if t not in boot_toasts(tn)] == [t for t in to if t not in boot_toasts(to)]
    assert '既定に戻っています' not in ''.join(tn) and f'{KEEP}・{NOKICK}は OFF' not in n[1] and '天井の左旋回中 4.2 m/s、' not in n[1]


@needs_node
@pytest.mark.parametrize('where', ['drive_tab', 'live_tab'])
def test_no_reentry_texts_exactly_as_before(tmp_path, cands, where):
    """起き直したあと入れ直していない(全部既定): トーストも走行タブの文も前とまったく同じ文。通信・タイマー・送った物も同じ"""
    c, di = cands
    pre = [['tab', 'drive'], ['tick']] + ([['tab', 'live'], ['tick']] if where == 'live_tab' else [])
    steps = pre + [['reboot', 2000.0, RESET]] + ([['tab', 'drive']] if where == 'live_tab' else [['tick']]) + LOOK + [
        ['tick'], ['el', '#dr-message', 'html'], ['start'], ['calls'], ['ntimers'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert tagged(new) == tagged(old) and tn == to
    assert tagged(new)[0] == BOOT_TOAST and note_of(tagged(new)[1]) == DEFAULT_NOTE == note_of(tagged(new)[7])
    assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers')
    same_sent(new, old)


@needs_node
def test_note_follows_reentry_after_the_notice(tmp_path, cands):
    """知らせが出たあとに入れ直す(手順書 7 章の形: スイッチで見張り → 残す、curl で左・launch_us)。直す前の写しは読み直すまで
    「…は OFF、4.1、250」のまま(スイッチと違う)。今の版はスイッチと同じ(ON・今 4.2・今 200)。見張りを OFF にすると(サーバーが
    残すも OFF)文も OFF に戻る。トースト・送った物・スイッチ・確認の文は前と同じ"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['reboot', 2000.0, RESET], ['tick'], ['el', '#dr-message', 'html'],
             ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True],
             ['api', {'ceil_left_mps': 4.2, 'launch_us': 200.0}], ['tick'], ['el', '#dr-message', 'html'],
             ['el', '#dr-rec-back-guard', 'checked'], ['el', '#dr-rec-hold-keep', 'checked'],
             ['check', '#dr-rec-back-guard', False], ['tick'], ['el', '#dr-message', 'html'], ['el', '#dr-rec-hold-keep', 'checked'],
             ['start'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    n, o = tagged(new), tagged(old)
    assert [note_of(x) for x in (o[0], o[1], o[4])] == [DEFAULT_NOTE] * 3          # 直す前: 入れ直しても OFF・4.1・250 のまま
    assert note_of(n[0]) == DEFAULT_NOTE and note_of(n[1]) == REENTERED_NOTE
    assert note_of(n[4]) == (HEAD + f'起動のたびに既定に戻るもの: {GUARD}・{KEEP}・{NOKICK}は OFF、天井の左旋回中 今 4.2 m/s(既定 4.1)、'
                             '発進の前進の強さ 今 200 µs(既定 250) など。' + TAIL_NOTE)
    assert n[2:4] == o[2:4] == [True, True] and n[5] is o[5] is False
    assert tn == to and boot_toasts(tn) == [BOOT_TOAST]
    same_sent(new, old)
    assert [p[1] for x in vals(new, 'posts') for p in x if p[0] == '/api/drive/params'] == [
        {'rec_back_guard': 1}, {'rec_hold_keep': 1}, {'ceil_left_mps': 4.2, 'launch_us': 200.0}, {'rec_back_guard': 0}]


OTHER = {'rec_back_guard': 1, 'speed_ceiling': 1, 'ceil_all_mps': 4.4, 'tcp_mode': 1, 'speed_scale': 1.2, 'launch_max_s': 1.0}


@needs_node
def test_partial_reentry_and_other_reset_items(tmp_path, cands):
    """一部だけ入れ直した・知らせの文に名前の無い項目(天井・天井の全体・TCP の出し方・倍率・発進の前進の時間)を入れ直した:
    トーストは既定と違う項目だけを今の値で挙げ、走行タブの文は ON と OFF を分け、ほかの項目を「(ほかに今は既定と違うもの: …)」"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, RESET], ['api', OTHER], ['tab', 'drive'],
             ['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-speed-ceiling', 'checked'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    n, o = tagged(new), tagged(old)
    items = f'{GUARD} ON・速さの天井 ON・天井の全体 4.4 m/s・発進の前進の時間 1 秒・TCP の出し方「ハイブリッド」・目標速度の倍率 1.2'
    assert n[0] == HEAD + f'起動のあとに入れ直した設定があります(今は {items})。朝の設定はスイッチと走行開始の確認の文で読む'
    assert note_of(n[1]) == (HEAD + f'起動のたびに既定に戻るもの: {GUARD}は ON(起動のあとに入れ直し済み)、{KEEP}・{NOKICK}は OFF、'
                             '天井の左旋回中 4.1 m/s、発進の前進の強さ 250 µs など(ほかに今は既定と違うもの: 速さの天井 ON・天井の全体 4.4 m/s・'
                             '発進の前進の時間 1 秒・TCP の出し方「ハイブリッド」・目標速度の倍率 1.2)。' + TAIL_NOTE)
    assert o[0] == BOOT_TOAST and note_of(o[1]) == DEFAULT_NOTE                     # 直す前: 見張り ON も「OFF」
    assert n[2] is o[2] is True
    assert vals(new, 'posts') == vals(old, 'posts')


@needs_node
@pytest.mark.parametrize('kind', ['net_fail', 'old_page', 'boot_409'])
def test_refused_start_appends_the_same_text_as_the_toast(tmp_path, cands, kind):
    """▶ の断りの文の出し直し(fix6): 確認の文を読んでいる間に起き直し、curl で朝の設定を入れ直していた。▶ が通信の途切れ・古い画面で
    断られ、同期で起き直しを知ったときは、後ろに付ける文がトーストと同じ今の値の文(直す前は「…既定に戻っています」)。起き直しの 409
    のときは前と同じ(断った文だけ)。送った物は同じ"""
    c, di = cands
    pre = [['tab', 'drive'], ['tick'], ['reboot', 2000.0, RESET], ['api', REENTER]]
    extra = {}
    if kind == 'net_fail':
        pre.append(['start_fail'])
    if kind == 'old_page':
        extra = {'server_client_version': 10}
    steps = pre + [['start'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps, **extra)
    fn, fo = tagged(new)[0], tagged(old)[0]
    assert boot_toasts(tn) == [REENTERED_TOAST] and boot_toasts(to) == [BOOT_TOAST]
    assert [t.replace(REENTERED_TOAST, BOOT_TOAST) for t in tn] == to                 # 違うのは知らせの文だけ(同じ順・同じ数)
    if kind == 'boot_409':
        assert fn == fo == PREFIX + REAL['1000']
    else:
        why = 'Failed to fetch (sim6: 走行開始が届かない)。' if kind == 'net_fail' else '管理画面が古いままです。'
        assert fn == PREFIX + why + REENTERED_TOAST and fo == PREFIX + why + BOOT_TOAST
        assert fn.endswith(boot_toasts(tn)[0])                                        # トーストと同じ文
        assert note_of(tagged(new)[1]) == REENTERED_NOTE
    same_sent(new, old)
    assert all(s['refused'] for s in starts(new))


def test_refusal_pattern_does_not_match_the_new_toasts():
    """▶ の catch の bootRefusal の形(起き直しの 409 の文)は、新しいトーストの文には当たらない(当たると出し直しの形が変わる)"""
    js = APP.read_text(encoding='utf-8')
    pat = re.compile(re.search(r'bootRefusal = /(.+?)/\.test\(m\)', js).group(1))
    for t in (BOOT_TOAST, REENTERED_TOAST, DEFAULT_NOTE, REENTERED_NOTE):
        assert not pat.search(t), t
    assert pat.search(REAL['1000']) and pat.search(REAL['2000->3000'])


@needs_node
def test_second_reboot_uses_the_new_time_and_values(tmp_path, cands):
    """2 回起き直す(2000 で入れ直し → 3000 で既定): 2 回目の知らせは新しい時刻と今の値(全部既定 = 前と同じ文)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['reboot', 2000.0, RESET], ['api', REENTER], ['tick'], ['el', '#dr-message', 'html'],
             ['reboot', 3000.0, RESET], ['tick'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html']]
    (new, tn), _ = both(tmp_path, c, steps)
    n = tagged(new)
    assert note_of(n[0]) == REENTERED_NOTE and boot_toasts(tn)[0] == REENTERED_TOAST
    hm2 = '管理画面のサーバーが 09:50 に起動し直しました。'
    assert n[1] == BOOT_TOAST.replace(HEAD, hm2) and note_of(n[2]) == DEFAULT_NOTE.replace(HEAD, hm2) and len(boot_toasts(tn)) == 2


# ------------------------------------------------------------------ 前からの流れは前とまったく同じ(送る物・通信・文)
def _all_flows(c, di):
    """10/26〜27 の画面の流れ・6-1 回目の起き直しの場面・6-2 回目の通るときの流れと走行中の流れ(どれも直した台本で)"""
    flows = {}
    for k, (st, sp) in dict(_flows(c, di)).items():
        flows[('flow', k)] = ([['poll']] + st + [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['posts']], {}, server_params(**sp))
    for k, v in F6.SCENES.items():
        flows[('scene', k)] = (F6.setup_steps(c, di) + v[0] + F6.TAIL, dict({'server_client_version': 8}, **v[1]), F6.MORNING)
    tail = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
    for k, st in F62.NORMAL.items():
        st = [s if s[0] != 'pick' else ['pick', di, blk(c, di, 10)] for s in st]
        flows[('normal', k)] = (st + tail, {}, F62.PRE_A)
    for k, st in F62.RUNNING.items():
        flows[('running', k)] = (st + tail, {}, F62.PRE_B)
    return flows


@needs_node
def test_all_earlier_flows_exactly_as_before(tmp_path, cands):
    """起き直しのあとが全部既定の流れ(10/26〜27 の流れ・6-1 回目の場面 A〜I・6-2 回目の流れ): 送った物(POST の body と返事)・
    走行開始の params・確認の文・boot_at・通信の全部・タイマー・トースト・文が、直す前の写しとまったく同じ(直した台本で)"""
    c, di = cands
    for i, (key, (st, extra, sp)) in enumerate(_all_flows(c, di).items()):
        st = st + [['calls'], ['ntimers']]
        new, tn = run_sim(tmp_path, c, st, name='n%d.json' % i, server=sp, **extra)
        old, to = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='o%d.json' % i, server=sp, **extra)
        assert tn == to and tagged(new) == tagged(old), key
        assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers'), key
        same_sent(new, old)


# ------------------------------------------------------------------ w1-ui-r7-ui-2 試験の台本
CLICK_STEPS = {
    'click': ['click', '#btn-dr-tg-up'],
    'slide': ['slide', '#dr-trim', 10],
}


@needs_node
@pytest.mark.parametrize('op', list(CLICK_STEPS))
def test_fixed_harness_keeps_the_loop_after_click_and_slide(tmp_path, cands, op):
    """直した台本: 'click'・'slide' のあとも表示の問い合わせの輪(pollDrive)が残り、curl の左 4.2 を 2 回の tick で読む(本物の
    ブラウザと同じ)。もとの台本(app_sim_fix6_r2.js)では輪が消え、4.1 のまま・status の GET 0 本(指摘の再現)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['timer_names'], CLICK_STEPS[op], ['timer_names'], ['eval', 'driveTimer !== null'], ['calls'],
             ['api', {'ceil_left_mps': 4.2}], ['tick'], ['tick'], ['calls'], ['read', 'st_params'], ['el', '#dr-ceil-spec', 'text']]
    res, _ = run_sim(tmp_path, c, steps, server=server_params())
    tn = vals(res, 'timer_names')
    assert ['pollDrive', 1000] in tn[0] and ['pollDrive', 1000] in tn[1]
    assert tn[1].count(['pollDrive', 1000]) == 1                                   # 1 つだけ(二重に掛けない)
    gets = [x for x in vals(res, 'calls')[-1] if x[1] == ST]
    assert len(gets) == 2 and all(x[2] == 200 for x in gets)
    assert vals(res, 'read')[-1]['ceil_left_mps'] == 4.2 and tagged(res)[-1] == '(全体 4.5・左旋回中 4.2 m/s)'
    old_steps = [s if s[0] != 'timer_names' else ['ntimers'] for s in steps]
    old, _ = run_sim(tmp_path, c, old_steps, sim=OLD_SIM, server=server_params(), name='old.json')
    assert vals(old, 'ntimers')[1] == [] and tagged(old)[0] is True             # もと: 輪のタイマーが消えた(driveTimer の番号は残る)
    assert not [x for x in vals(old, 'calls')[-1] if x[1] == ST] and vals(old, 'read')[-1]['ceil_left_mps'] == 4.1


CLEAR = ("globalThis.__x = []; setTimeout(function a() { __x.push('a'); clearTimeout(globalThis.__b); clearTimeout(globalThis.__L); }, 50); "
         "globalThis.__b = setTimeout(function b() { __x.push('b'); }, 100); "
         "globalThis.__L = setTimeout(function L() { __x.push('L'); }, 1000); "
         "globalThis.__L2 = setTimeout(function L2() { __x.push('L2'); }, 1000); 1")
CLEAR_TICK = ("globalThis.__y = []; setTimeout(function c() { __y.push('c'); clearTimeout(globalThis.__d); }, 1000); "
              "globalThis.__d = setTimeout(function d() { __y.push('d'); }, 1000); 1")


@needs_node
@pytest.mark.parametrize('how', ['timers', 'click', 'tick'])
def test_fixed_harness_respects_clear_before_fire(tmp_path, cands, how):
    """直した台本: 先に鳴ったタイマーが消したタイマー(短いもの・長いもの)は鳴らない・一覧に戻らない(ブラウザと同じ)。消していない
    長いタイマーは残って次の tick で鳴る。tick で同時に鳴るタイマーでも、先に鳴ったものが消したものは鳴らない。もとの台本は消した
    ものも鳴らし(['timers']・['tick'])、消した長いタイマーを戻した"""
    c, di = cands
    if how == 'tick':
        steps = [['tab', 'drive'], ['tick'], ['eval', CLEAR_TICK], ['tick'], ['eval', '__y.slice()']]
        want, want_old = [['c']], [['c', 'd']]
    else:
        fire = ['timers'] if how == 'timers' else ['click', '#btn-dr-tg-down']
        steps = [['tab', 'drive'], ['tick'], ['eval', CLEAR], fire, ['eval', '__x.slice()'], ['tick'], ['eval', '__x.slice()']]
        want, want_old = [['a'], ['a', 'L2']], [['a', 'b'], ['a', 'b', 'L', 'L2']]
    res, _ = run_sim(tmp_path, c, steps, server=server_params())
    assert vals(res, 'eval')[1:] == want
    if how != 'click':                                  # もとの台本の 'click' は長いタイマーを捨てるので比べない(上の試験)
        old, _ = run_sim(tmp_path, c, steps, sim=OLD_SIM, server=server_params(), name='old.json')
        assert vals(old, 'eval')[1:] == want_old


@needs_node
def test_fix6_conclusions_hold_on_the_fixed_harness(tmp_path, cands):
    """6-2 回目の「通るときの流れ・走行中の流れは 6-1 回目の版とまったく同じ」(test_ui_fix6_r2 の NORMAL・RUNNING。click・slide の
    あとの輪と問い合わせを含む)と、6-1 回目の起き直しの場面の最後の文(test_ui_fix6 の SCENES)が、直した台本でも成り立つ"""
    c, di = cands
    tail = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
    for i, (k, st) in enumerate(list(F62.NORMAL.items()) + list(F62.RUNNING.items())):
        st = [s if s[0] != 'pick' else ['pick', di, blk(c, di, 10)] for s in st] + tail
        sp = F62.PRE_A if k in F62.NORMAL else F62.PRE_B
        a, ta = run_sim(tmp_path, c, st, app=R61 / 'app.js', name='a%d.json' % i, server=sp)
        b, tb = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='b%d.json' % i, server=sp)
        assert vals(a, 'calls') == vals(b, 'calls') and ta == tb and tagged(a) == tagged(b) and vals(a, 'posts') == vals(b, 'posts'), k
        if not k.startswith('一覧の GET が止まる'):
            assert vals(a, 'ntimers') == vals(b, 'ntimers'), k
        else:
            assert vals(a, 'ntimers')[-1] == vals(b, 'ntimers')[-1], k
    for k, (steps, extra, want_new, _) in F6.SCENES.items():
        st = F6.setup_steps(c, di) + steps + F6.TAIL
        res, _ = run_sim(tmp_path, c, st, name='s.json', server=F6.MORNING, **dict({'server_client_version': 8}, **extra))
        assert tagged(res)[-3] == want_new, k


@needs_node
def test_fix6_r2_look_scenes_on_the_fixed_harness(tmp_path, cands):
    """6-2 回目の指摘の場面 B(一覧の GET だけ失敗 → curl で左 4.2)を直した台本で: 6-1 回目の版は輪が無く 4.1、6-2 回目のあとの版と
    今の版は輪(1 秒)があり 4.2。輪のほかのタイマー(上の状態・ライブの値の輪)は 3 つの版で同じ"""
    c, di = cands
    steps = F62.BASE + [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['api', {'ceil_left_mps': 4.2}],
                        ['tick'], ['tick'], ['dtimers'], ['timer_names'], ['read', 'st_params'], ['el', '#dr-ceil-spec', 'text']]
    out = {}
    for name, app in (('r61', R61 / 'app.js'), ('before', BEFORE / 'app.js'), ('now', APP)):
        res, _ = run_sim(tmp_path, c, steps, app=app, name='%s.json' % name, server=F62.PRE_B)
        out[name] = (vals(res, 'dtimers')[0], vals(res, 'read')[-1]['ceil_left_mps'], tagged(res)[-1],
                     sorted(x for x in vals(res, 'timer_names')[0] if x[0] in ('pollStatus', 'pollValues')))
    assert out['r61'][:3] == ([], 4.1, '(全体 4.5・左旋回中 4.1 m/s)')
    assert out['before'][:3] == out['now'][:3] == ([1000], 4.2, '(全体 4.5・左旋回中 4.2 m/s)')
    assert out['r61'][3] == out['before'][3] == out['now'][3] == [['pollStatus', 1000], ['pollValues', 300]]
