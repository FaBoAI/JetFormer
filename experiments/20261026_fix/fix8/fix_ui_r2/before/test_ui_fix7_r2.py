"""画面の係の直し 7-2 回目(2026-10-27 fix7。反証の指摘 w2-ui-w2-ui-r72-ui-1・w2-ui-w2-ui-r72-ui-2。
~/jetformer_reports/fix_20261026/fix7/fix_ui_r2.md)の試験。

  w2-ui-w2-ui-r72-ui-1: ▶ の断りの文の出し直し(fix6 w1-ui-ui61-2)は、▶ の catch の同期(pollDrive)が通るときだけ効いた。確認の文を
    読んでいる間にサーバーが落ち、起き上がる前に OK を押すと(POST も同期も届かない)、あとで 1 秒の輪が起き直しを見つけて出す知らせ
    「…既定に戻っています」だけが最後に残り、「走行は始まっていません(停止中)」も ▶ が届かなかったことも出なかった。起き直しの
    409 のあと catch の同期だけ途切れたときも、次の輪の知らせが出し直した文を消した。
  直し(app.js。表示だけ): catch の同期が通らず知らせも出ていなければ、断った文を覚える(state.drive.startRefused = {seq, m,
    bootRefusal})。次に同期が通ったとき noteServerBoot が捨て、起き直しの知らせを出したときだけ、まだ次の ▶ を押していなければ
    知らせのあとに同期が通る場合と同じ文(startRefusalText)を出し直す。起き直していなければ(Wi-Fi の途切れだけ)前と同じ。
  w2-ui-w2-ui-r72-ui-2: 走行タブを開いたときにモデルの一覧の GET だけ失敗すると、輪は回って「同期済み」になるのに一覧は
    タブを替えるまで読み直さず、初めて開いたときは「基準の設定」が「モデル … が走行の一覧にありません(TensorRT エンジンが必要)」と
    まちがった理由を出した。
  直し(app.js。表示と問い合わせだけ): 一覧の GET が失敗・5 秒返らなかったら印(modelsMiss)を付け、止まっていて走行タブに
    いれば、輪の 2 回目の同期で一覧を 1 回だけ読み直す(retryDriveModels → reloadDriveModels。pollDrive は呼ばない)。読めていない
    間は、止まっていれば「基準の設定」・ブロック・▶ の文を DR_MODELS_MISS にし、空の選ぶ欄に短い文を出す。
  どちらも送る物(POST の body・走行開始の params・確認の文)・断る決まり・版(9)は前と同じ。走行中・token のある画面・一覧が通る
  ときは問い合わせも前とまったく同じ。

本物の app.js(今の版と、直す前の写し tests/fixtures/fix7_ui_r2_before/app.js = 直す 7-1 回目のあとの版 3cf098d9…。sha256 を照らす)を
node の vm で(tests/fixtures/app_sim_fix7_r2.js = 反証 7-2 回目の台本 sim_r2.js の写し = app_sim_fix7.js + 確認の文の間に落ちる・
status の GET だけ途切れる・返事だけ届かない POST など)。409 の文は本物の server._boot_refusal で作る(import だけ。起動しない)。
車・ブラウザ・jetformer-web には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix7_r2.py

JF_UI72_APP=<app.js の写し> でその写しを「今の版」として流す(わざと壊す台本 experiments/20261026_fix/fix7/ui_r2/mut/ が使う)。
"""
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive

import test_ui_fix6 as F6
import test_ui_fix6_r2 as F62
import test_ui_fix7 as T7

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
FIXT = ROOT / 'tests' / 'fixtures'
APP = Path(os.environ.get('JF_UI72_APP') or STATIC / 'app.js')
SIM = FIXT / 'app_sim_fix7_r2.js'
REVIEW_SIM = ROOT / 'experiments' / '20261026_fix' / 'fix7' / 'review_ui_r2' / 'sim_r2.js'
BEFORE = FIXT / 'fix7_ui_r2_before'                    # 直す前の app.js(7-1 回目のあとの版 3cf098d9…)
# 2026-10-28 fix8 tools-decided のあとの版の sha256(このあとの直しはその係がここに足す)
LATER_FIX8_TOOLS_DECIDED = {'experiments/20261026_fix/plan/runs_1103.py': '33bc7fca99216cc2d4704e8fa5de9741626231fd4a0d165cb981183801acba00'}
# 2026-10-28 fix8 画面の係 直す 8-1 回目(tests/test_ui_fix8_r1.py)のあとの版の sha256。その版なら直す前の写し tests/fixtures/fix8_ui_r1_before/
# (= 上の版)へたどる(中身の違いは tests/test_ui_fix8_r1.py)
FIX8_UI_R1_BEFORE = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r1_before'
LATER_FIX8_UI_R1 = {'experiments/20261026_fix/plan/runs_1103.py': 'c2d616c2f72b00935868691236f94e4733c3185a24d200c524b6049d615fd9fc',
                    'jetformer/web/static/style.css': 'dbc8b79ad935ed914c7b8a2cc02f8c4ddaf0c453af33ca3fb85fa0ea1103df56',
                    'jetformer/web/static/app.js': 'aa275bb5da0bbd740f38c217dd0ae9ee19240f061dafd8ca70f6916448837285'}
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
ST, MD, BC = '/api/drive/status', '/api/drive/models', '/api/bench/candidates'
PREFIX = F6.PREFIX                                     # 走行は始まっていません(停止中)。
BOOT_TOAST = F6.BOOT_TOAST                             # 起き直しの知らせ(全部既定)
REAL = T7.REAL                                         # 本物の 409 の文(画面 1000 → 今 2000 など)
MORNING = T7.MORNING                                   # 見張り・残す ON・左 4.2・launch_us 200・速度 0.9(boot_at 1000)
DOWN = 'Failed to fetch (sim6r2: サーバーが落ちている。GET も全部)'
NET = 'Failed to fetch (r2: 表示の問い合わせだけ途切れ)'
MFAIL = 'Failed to fetch (sim6r2: モデルの一覧だけ通らない)'
MISS = ('モデルの一覧を読めていません(一覧の問い合わせが通らないか遅れています。TensorRT エンジンが無いという意味ではありません)。'
        '少し待っても「モデル」の欄が空なら、ほかのタブ → 走行タブで開き直すか、画面を読み直してください')
PLACEHOLDER = '— モデルの一覧を読めていません(通信の途切れ)。少し待つか、ほかのタブ → 走行タブ —'
WRONG = 'が走行の一覧にありません(TensorRT エンジンが必要)'
BASE_OK = '基準の設定を画面に入れました(走行は始めていません)'

# 直す前の写しから消えた行(ほかは足しただけ)
OLD_LINES = [
    "  const lost = () => { if (drIdle()) { drLost(); pollDrive(); } };",
    "  try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) { clearTimeout(late); toast(e.message, true); lost(); return; }",
    "  pollDrive();",
    "  if (!m) { toast('モデルを選んでください', true); return; }",
    "    try { await pollDrive(); } finally {",
    "        const st = state.drive.st;",
    "        toast((st && !st.running ? '走行は始まっていません(停止中)。' : '')",
    "          + (bootRefusal ? m : `${m.replace(/。$/, '')}。${state.drive.bootToast}`), true);",
    "  if (!m) { toast(`モデル ${run} が走行の一覧にありません(TensorRT エンジンが必要)`, true); return false; }",
]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sums(d):
    return dict(reversed(x.split()) for x in (d / 'SHA256SUMS').read_text().splitlines() if x.strip())


def _diff():
    # 2026-10-28 fix8 画面の係 直す 8-1 回目(tests/test_ui_fix8_r1.py)が app.js に行を足したので、「消えた行は 9 行だけ・足した行」は
    # この組(7-2 回目)のあとの版の写し(tests/fixtures/fix8_ui_r1_before/app.js = 2b211259…。sha256 を照らす)で見る(今の app.js との
    # 違いは test_ui_fix8_r1.py が見る)。JF_UI72_APP を付けたとき(わざと壊す台本)はその写し。動きの試験は今の app.js で流す
    a = (BEFORE / 'app.js').read_text(encoding='utf-8').splitlines()
    after = APP if os.environ.get('JF_UI72_APP') else FIX8_UI_R1_BEFORE / 'app.js'
    if not os.environ.get('JF_UI72_APP'):
        assert _sums(FIX8_UI_R1_BEFORE)['app.js'] == _sha(after) and _sha(after).startswith('2b211259db71')
    b = after.read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    return a, b, ops


def test_before_copy_and_only_these_places_changed():
    """直す前の写しは 7-1 回目のあとの版(3cf098d9…)。消えた行は上の 9 行だけ(ほかは足した)。足した行に送る物(POST・body・client)・
    タイマー・確認の文は無い。足した問い合わせは GET /api/drive/models の 1 本(読み直し)だけ。版は 9 のまま。server.py・index.html・
    style.css・bench.py・runs_1103.py は変えていない"""
    assert _sums(BEFORE) == {'app.js': _sha(BEFORE / 'app.js')} and _sha(BEFORE / 'app.js').startswith('3cf098d93386')
    a, b, ops = _diff()
    removed = [a[i] for o in ops for i in range(o[1], o[2])]
    added = [b[j] for o in ops for j in range(o[3], o[4])]
    assert removed == OLD_LINES, removed
    code = [x.split('//')[0] for x in added]
    assert not any(re.search(r"'POST'|body\.|const body|client:|fetch\(|setTimeout|confirm\(", x) for x in code), added
    assert [x.strip() for x in code if 'api(' in x] == [
        "try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) {",
        "try { models = await api('GET', '/api/drive/models'); } catch (e) { return; } finally { state.drive.modelsBusy--; }"]
    assert sum('clearTimeout(late)' in x for x in code) == 1 and sum('drLost()' in x for x in code) == 1      # 前からの catch・lost の形
    assert [x.strip() for x in code if re.search(r'\bpollDrive\(\)', x)] == [         # どれも前からの 3 か所(形だけ変えた)
        'const lost = () => { state.drive.modelsMiss = true; state.drive.modelsRetry = 2; if (drIdle()) { drLost(); pollDrive(); } };',
        'pollDrive();', 'try { synced = await pollDrive(); } finally {']
    js = APP.read_text(encoding='utf-8')
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert js.count('toast(msg, true);') == 1 and js.count('bootRefusal = /^管理画面のサーバーが .*に起動し直しました\\(/.test(m)') == 1
    assert _sha(ROOT / 'jetformer' / 'web' / 'server.py').startswith('ad7a7d05f7a9')
    assert _sha(STATIC / 'index.html').startswith('ef81bb6017d6')
    css = STATIC / 'style.css'                    # fix8 ui r1(.toast の幅)の版なら、その直す前の写しを見る
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


def test_fill_is_the_old_loaddrive_body_and_texts_match():
    """fillDriveModels の中身は、直す前の loadDrive の「選ぶ欄に入れる」行とまったく同じ(関数に分けただけ)。loadDrive は前と同じ順に
    fillDriveModels → pollDrive → loadBench。試験の文(MISS・選ぶ欄の短い文)は app.js の文と同じ"""
    a = (BEFORE / 'app.js').read_text(encoding='utf-8')
    js = APP.read_text(encoding='utf-8')
    old = re.search(r"\n(  const sel = \$\('#dr-model'\), cur = sel\.value;\n.*?\n  state\.drive\.restored = true;\n)  pollDrive\(\);\n  loadBench\(\);\n\}",
                    a, re.S).group(1)
    new = re.search(r'\nfunction fillDriveModels\(\) \{\n(.*?)\}\n', js, re.S).group(1)
    assert new == old
    assert "  fillDriveModels();\n  pollDrive();\n  loadBench();\n}\n" in js
    body = re.search(r"const DR_MODELS_MISS = ('.*?')\n  \+ ('.*?');", js, re.S)
    assert eval(body.group(1)) + eval(body.group(2)) == MISS                       # noqa: S307(app.js の文字の定数だけ)
    assert f'<option value="">{PLACEHOLDER}</option>' in js


def test_sim_copy_is_the_review_harness():
    """台本 app_sim_fix7_r2.js は反証 7-2 回目の sim_r2.js に 2 行の頭書きを足しただけ"""
    body = SIM.read_text(encoding='utf-8').split('\n', 2)[2]
    if REVIEW_SIM.exists():
        assert body == REVIEW_SIM.read_text(encoding='utf-8')
    assert hashlib.sha256(body.encode('utf-8')).hexdigest()[:12] in SIM.read_text(encoding='utf-8').split('\n', 2)[1]


# ------------------------------------------------------------------ 画面を node で流す
cands = F6.cands
blk = F6.blk
server_params = F6.server_params


def run_sim(tmp_path, c, steps, app=None, name='input.json', server=None, **extra):
    inp = tmp_path / name
    d = {'bench': c, 'server_params': server if server is not None else MORNING, 'steps': steps, 'server_client_version': 9,
         'confirm': True, 'boot_at': 1000.0, 'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True, 'boot_refusal_texts': REAL}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app or APP), str(STATIC / 'index.html'), str(inp)], capture_output=True, text=True,
                       timeout=120, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def both(tmp_path, c, steps, **kw):
    """今の版と直す前の写しで同じ操作を流す"""
    return (run_sim(tmp_path, c, steps, name='n.json', **kw), run_sim(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json', **kw))


def vals(res, op):
    return [x.get('value') for x in res if x['op'] == op]


def tagged(res):
    return [x['value'] for x in res if x['op'] in ('el', 'eval')]


def starts(res):
    return [x for x in res if x['op'] == 'start']


def sent(res):
    return [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(res)]


def boot_toasts(toasts):
    return [t for t in toasts if re.match(r'^管理画面のサーバーが \d\d:\d\d に起動し直しました。', t)]


def gets(calls, path):
    return [x for c in calls for x in c if x[0] == 'GET' and x[1].split('?')[0] == path]


def is_reshow(prev, t):
    """起き直しの知らせ(prev)のすぐあとに出し直した文か(「走行は始まっていません」+ 断った文、または断った文 + 知らせ)"""
    return bool(boot_toasts([prev])) and (t.startswith(PREFIX) or t.endswith('。' + prev))


def old_plus_reshows(tn, to):
    """今の版のトーストの列 = 直す前の列に「知らせのすぐあとの出し直し」を足しただけか。足した数を返す(違えば None)"""
    i = j = extra = 0
    while i < len(tn):
        if j < len(to) and tn[i] == to[j]:
            i += 1
            j += 1
        elif i and is_reshow(tn[i - 1], tn[i]):
            i += 1
            extra += 1
        else:
            return None
    return extra if j == len(to) else None


# ------------------------------------------------------------------ w2-ui-w2-ui-r72-ui-1
LOOK1 = [['el', '#toast', 'text'], ['el', '#dr-phase', 'text'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-message', 'html']]
UP = [['reboot', 2000.0, RESET], ['down_all', False], ['tick']]                # systemd が起こし直した → 1 秒の輪が知る
SCENES1 = {
    # 指摘の S1: 確認の文の間に落ちる → OK(届かない)→ catch の同期も落ちたまま → 起き上がる → 輪の問い合わせで知らせ
    'S1': ([['down_on_confirm'], ['start'], ['tick']] + LOOK1 + UP, PREFIX + DOWN + '。' + BOOT_TOAST),
    # 落ちている間に輪が 4 回失敗してから起き上がる(10/27 17:16 の再起動は約 8 秒)
    'S1_long': ([['down_on_confirm'], ['start'], ['tick'], ['tick'], ['tick'], ['tick']] + LOOK1 + UP, PREFIX + DOWN + '。' + BOOT_TOAST),
    # 起き上がったあと curl で朝の設定を入れ直してから輪が知る(7-1 回目の「今は …」の知らせ)
    'S1_reentered': ([['down_on_confirm'], ['start'], ['tick'], ['reboot', 2000.0, RESET], ['down_all', False], ['api', T7.REENTER],
                      ['tick']], PREFIX + DOWN + '。' + T7.REENTERED_TOAST),
    # 指摘の S1c: 起き上がったあとに OK(409)、catch の同期だけ通信の途切れ → 次の輪で知らせ
    'S1c': ([['reboot_on_confirm', 2000.0, RESET], ['status_fail_next', 1], ['start']] + LOOK1 + [['tick']], PREFIX + REAL['1000']),
    # 同期が 3 回続けて途切れてから通る
    'S1c_3': ([['reboot_on_confirm', 2000.0, RESET], ['status_fail_next', 3], ['start'], ['tick'], ['tick'], ['tick']], PREFIX + REAL['1000']),
}


@needs_node
@pytest.mark.parametrize('name', list(SCENES1))
def test_refusal_reshown_after_the_boot_notice_when_sync_failed(tmp_path, cands, name):
    """▶ の catch の同期が通らなかった: 次に同期が通って起き直しの知らせが出たところで、断った文をもう一度出す(最後に見える文は
    同期が通る場合と同じ形 = 「走行は始まっていません(停止中)。」+ 断った文(+ 知らせ))。直す前の写しは知らせだけが残った。
    落ちている間の見え方(トースト・▶ を押せない・「つながっていません」・バッジ)・送った物・確認の文は直す前と同じ。トーストは
    出し直した 1 つのほかは直す前と同じ順・同じ数"""
    c, di = cands
    steps, want = SCENES1[name]
    steps = [['tab', 'drive'], ['tick']] + steps + LOOK1 + [['posts'], ['calls']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    clock = lambda xs: [re.sub(r'\d\d:\d\d:\d\d に失敗', 'HH:MM:SS に失敗', x) if isinstance(x, str) else x for x in xs]   # 失敗の時刻は実時間
    n, o = clock(tagged(new)), clock(tagged(old))
    assert n[-4] == want and o[-4] == (T7.REENTERED_TOAST if name == 'S1_reentered' else BOOT_TOAST)      # 直す前: 指摘が起きる
    assert n[-3] == o[-3] == '停止中' and n[-2] is o[-2] is False and n[-1] == o[-1]
    if name in ('S1', 'S1_long'):
        assert n[:4] == o[:4] and n[0] == DOWN and n[2] is True and 'サーバーにつながっていません' in n[3]    # 落ちている間は前と同じ
    assert old_plus_reshows(tn, to) == 1 and tn[-1] == want
    assert vals(new, 'posts') == vals(old, 'posts') and sent(new) == sent(old) and vals(new, 'calls') == vals(old, 'calls')
    s = starts(new)[0]
    assert s['sent'] is not None and s.get('status') in (None, 409) and s['confirm'].startswith('AI走行を開始します。')


@needs_node
def test_same_last_text_whether_or_not_the_sync_passes(tmp_path, cands):
    """どの返事の順でも最後に残る文が同じ(fix6 の約束): 同期が通る場合(fix6 r1 の場面 F・起き直しの 409)と通らない場合(S1・S1c)で、
    最後のトーストが同じ文"""
    c, di = cands
    pre = [['tab', 'drive'], ['tick']]
    pairs = {
        'net': ([['reboot_on_confirm', 2000.0, RESET], ['start_fail'], ['start'], ['tick']],
                [['reboot_on_confirm', 2000.0, RESET], ['start_fail'], ['status_fail_next', 1], ['start'], ['tick']]),
        '409': ([['reboot_on_confirm', 2000.0, RESET], ['start'], ['tick']],
                [['reboot_on_confirm', 2000.0, RESET], ['status_fail_next', 1], ['start'], ['tick']]),
    }
    for k, (passes, fails) in pairs.items():
        a, ta = run_sim(tmp_path, c, pre + passes + [['el', '#toast', 'text']], name='a.json')
        b, tb = run_sim(tmp_path, c, pre + fails + [['el', '#toast', 'text']], name='b.json')
        assert tagged(a)[-1] == tagged(b)[-1] == ta[-1] == tb[-1], k
        assert tagged(a)[-1].startswith(PREFIX)
        assert tagged(a)[-1] == (PREFIX + 'Failed to fetch (sim6: 走行開始が届かない)。' + BOOT_TOAST if k == 'net' else PREFIX + REAL['1000'])


# 起き直していない・出し直さない場面(前とまったく同じ)
SAME1 = {
    # Wi-Fi の途切れだけ(boot_at は同じ): 断った文がそのまま残る
    'wifi_only': [['down_on_confirm'], ['start'], ['tick'], ['down_all', False], ['tick'], ['tick']],
    # 途切れのあと同期が通った(知らせなし)→ だいぶあとで起き直した: 古い ▶ の文は出し直さない(覚えた文は最初の同期で捨てた)
    'wifi_then_later_reboot': [['down_on_confirm'], ['start'], ['tick'], ['down_all', False], ['tick'], ['reboot', 2000.0, RESET], ['tick']],
    # 走行開始の返事だけ届かず(サーバーは始めた)、catch の同期も途切れた: 次の同期は走行中(起き直していない)→ 何も出し直さない
    'start_lost_running': [['start_lost'], ['status_fail_next', 1], ['start'], ['tick'], ['tick']],
    # 落ちている間にもう一度 ▶(つながっていないので断る。送らない)→ 起き上がる: 知らせのあとの出し直しは前の ▶ の文(seq は同じ)
    # → 直す前と違うので、ここではなく下の試験で見る
    # catch の同期は通ったが起き直しなし(古い画面で断られた): 前と同じ(覚えない)
    'old_page_synced': [['start'], ['tick'], ['reboot', 2000.0, RESET], ['tick']],
}


@needs_node
@pytest.mark.parametrize('name', list(SAME1))
def test_no_reshow_without_a_restart_exactly_as_before(tmp_path, cands, name):
    """起き直していない(Wi-Fi の途切れだけ)・途切れのあと知らせの無い同期が通った・走行開始の返事だけ届かなかった・同期が通った:
    トースト・見え方・通信の全部・タイマー・送った物が直す前とまったく同じ"""
    c, di = cands
    extra = {'server_client_version': 10} if name == 'old_page_synced' else {}
    steps = [['tab', 'drive'], ['tick']] + SAME1[name] + LOOK1 + [['posts'], ['calls'], ['ntimers']]
    (new, tn), (old, to) = both(tmp_path, c, steps, **extra)
    assert tn == to and tagged(new) == tagged(old)
    assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers') and sent(new) == sent(old)
    if name.startswith('wifi'):
        assert not any(t.startswith(PREFIX) for t in tn)
        assert tn[-1] == (DOWN if name == 'wifi_only' else BOOT_TOAST)


@needs_node
def test_blocked_second_press_while_down_keeps_the_first_refusal(tmp_path, cands):
    """落ちている間にもう一度 ▶ を押した(つながっていないので確認の文も出さずに断る。番号も進まない)→ 起き上がる: 知らせのあとに
    最初の ▶ の断りの文を出し直す。送った走行開始は 1 本だけ(どちらの版も)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['down_on_confirm'], ['start'], ['tick'], ['start']] + UP + [['el', '#toast', 'text'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert tagged(new)[-1] == PREFIX + DOWN + '。' + BOOT_TOAST and tagged(old)[-1] == BOOT_TOAST
    assert old_plus_reshows(tn, to) == 1 and vals(new, 'posts') == vals(old, 'posts')
    assert sum(p[0] == '/api/drive/start' for x in vals(new, 'posts') for p in x) == 1 and starts(new)[1]['sent'] is None


@needs_node
def test_newer_press_is_not_overwritten(tmp_path, cands):
    """覚えた文は、そのあと ▶ の番号が進んでいたら出し直さない(次の ▶ の文・走り出した知らせを消さない)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['down_on_confirm'], ['start'], ['tick'], ['eval', 'state.drive.startSeq += 1']] + UP + [
        ['el', '#toast', 'text']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert tn == to and tagged(new)[-1] == BOOT_TOAST


@needs_node
def test_no_prefix_when_the_synced_status_is_running(tmp_path, cands):
    """起き上がったサーバーで、ほかの画面がもう走らせていた: 出し直す文に「走行は始まっていません」を付けない(断った文 + 知らせ)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['down_on_confirm'], ['start'], ['tick'], ['reboot', 2000.0, RESET], ['down_all', False],
             ['running', True], ['el', '#toast', 'text']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert tagged(new)[-1] == DOWN + '。' + BOOT_TOAST and not any(t.startswith(PREFIX) for t in tn)
    assert tn == to + [DOWN + '。' + BOOT_TOAST] and tagged(old)[-1] == BOOT_TOAST


# ------------------------------------------------------------------ w2-ui-w2-ui-r72-ui-2
LOOK2 = [['el', '#toast', 'text'], ['el', '#dr-model', 'value'], ['eval', 'state.drive.models.length'], ['eval', 'state.drive.paramsLoaded'],
         ['eval', 'JSON.stringify(state.bench.data && state.bench.data.today)'],
         ['eval', 'document.querySelector("#dr-model").options.map((o) => o.textContent).join("|").slice(0, 80)']]
OPEN_FAIL = [['models_fail', True], ['tab', 'drive'], ['models_fail', False]]       # 読み直したあと初めて走行タブ(その瞬間だけ途切れる)


def look2(res, k):
    """LOOK2 の k 番目の結果"""
    t = tagged(res)
    return t[k * len(LOOK2):(k + 1) * len(LOOK2)]


@needs_node
def test_first_open_models_fail_reloads_on_the_next_loop_sync(tmp_path, cands):
    """指摘の場面: 初めて走行タブを開いたときに一覧の GET だけ失敗 → 待つ(手順書 7 章)→ 基準の設定。今の版は失敗のすぐあと
    選ぶ欄に短い文を出し、輪の 2 回目の同期(1 秒後)で一覧を 1 回読み直して(選ぶ欄・ベンチマーク)、基準の設定が通る。直す前の
    写しは一覧が 0 本のまま「…一覧にありません(TensorRT エンジンが必要)」(エンジンはある)"""
    c, di = cands
    steps = (OPEN_FAIL + LOOK2 + [['calls'], ['tick']] + LOOK2 + [['calls'], ['tick']] + LOOK2 + [['calls'], ['dtimers'], ['baseline'],
             ['el', '#toast', 'text'], ['calls'], ['posts']])
    (new, tn), (old, to) = both(tmp_path, c, steps, no_init=True)
    n0, n1, n2 = look2(new, 0), look2(new, 1), look2(new, 2)
    assert n0[:5] == [MFAIL, '', 0, True, 'null'] and n0[5] == PLACEHOLDER                  # 失敗のすぐあと: 選ぶ欄に短い文
    assert n1[2] == n2[2] == 6 and n1[4] == '"2026-11-03"' and n1[3] is True                # 1 秒後: 読み直した
    assert n1[1] == c['baseline']['run'] and PLACEHOLDER not in n1[5]                       # 推奨のモデル(基準 = B)
    o2 = look2(old, 2)
    assert o2[:5] == [MFAIL, '', 0, True, 'null']
    assert tagged(new)[-1] == BASE_OK and tagged(old)[-1].endswith(WRONG)                     # 直す前: まちがった理由
    calls = vals(new, 'calls')
    assert len(gets(calls, MD)) == 2 and len(gets(calls, BC)) == 1                           # 一覧は 1 回だけ読み直した
    assert len(gets(vals(old, 'calls'), MD)) == 1 and not gets(vals(old, 'calls'), BC)
    # 通信の全部: 失敗 → すぐの同期(1 回目)→ 1 秒後の輪の同期(2 回目)のあとに一覧・ベンチマーク → 次の輪は status だけ(輪は 1 つ)
    assert [[(m, u) for m, u, _ in x] for x in calls[:3]] == [
        [('GET', MD), ('GET', ST)], [('GET', '/api/status'), ('GET', ST), ('GET', MD), ('GET', BC)], [('GET', '/api/status'), ('GET', ST)]]
    assert vals(new, 'dtimers') == vals(old, 'dtimers') == [[1000]]
    assert [[(m, u) for m, u, _ in x] for x in vals(old, 'calls')[:3]] == [
        [('GET', MD), ('GET', ST)], [('GET', '/api/status'), ('GET', ST)], [('GET', '/api/status'), ('GET', ST)]]
    assert vals(new, 'posts') == vals(old, 'posts') == [[]]                                   # 何も送っていない
    assert MFAIL in tn and not any(WRONG in t for t in tn) and not any(t == MISS for t in tn)


@needs_node
def test_after_reload_sends_exactly_what_a_clean_open_sends(tmp_path, cands):
    """一覧を読み直したあとの朝の流れ(基準の設定 → 見張り → 残す → ブロック → ▶)で送る物(POST の body と返事・走行開始の params・
    確認の文・boot_at)は、一覧が初めから通った画面(直す前の写し)とまったく同じ"""
    c, di = cands
    flow = [['baseline'], ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['pick', di, blk(c, di, 1)],
            ['tick'], ['start'], ['posts']]
    srv = server_params(throttle_gain=0.9)
    new, tn = run_sim(tmp_path, c, OPEN_FAIL + [['tick'], ['tick']] + flow, server=srv, no_init=True, name='n.json')
    old, to = run_sim(tmp_path, c, [['tab', 'drive'], ['tick'], ['tick']] + flow, server=srv, no_init=True, app=BEFORE / 'app.js',
                      name='o.json')
    assert vals(new, 'posts') == vals(old, 'posts') and sent(new) == sent(old)
    assert starts(new)[0]['status'] == 200 and '後ろの見張り: ON' in starts(new)[0]['confirm']
    assert [t for t in tn if t != MFAIL] == to


@needs_node
def test_reload_fails_too_then_the_message_and_no_more_gets(tmp_path, cands):
    """読み直しも通らない(途切れが続く): 一覧の GET は最初の 1 本と読み直しの 1 本だけ(毎秒は掛けない)。基準の設定・▶ は
    「モデルの一覧を読めていません…」(直す前は「…TensorRT エンジンが必要」「モデルを選んでください」)。ブロックのカードは出ない。
    ほかのタブ → 走行タブで開き直すと読める(手順書 7 章に足す文)"""
    c, di = cands
    steps = ([['models_fail', True], ['tab', 'drive'], ['tick'], ['tick'], ['tick'], ['tick'], ['baseline'], ['el', '#toast', 'text'],
              ['start'], ['el', '#toast', 'text'], ['tapday', di, blk(c, di, 1)], ['el', '#toast', 'text'], ['calls'],
              ['el', '#dr-model', 'value'], ['el', '#bench-card', 'hidden'],
              ['models_fail', False], ['tab', 'live'], ['tab', 'drive'], ['baseline'], ['el', '#toast', 'text'], ['calls'], ['posts']])
    (new, tn), (old, to) = both(tmp_path, c, steps, no_init=True)
    n, o = tagged(new), tagged(old)
    assert n[0] == MISS and n[1] == MISS and n[2] == MISS                                   # 基準の設定・▶(カードは押しても何もしない)
    assert o[0].endswith(WRONG) and o[1] == 'モデルを選んでください'
    assert n[3] == o[3] == '' and n[4] == o[4]
    assert len(gets(vals(new, 'calls')[:1], MD)) == 2 and len(gets(vals(old, 'calls')[:1], MD)) == 1
    assert n[5] == o[5] == BASE_OK                                                         # 開き直すと読める
    assert vals(new, 'posts') == vals(old, 'posts') and all(s['sent'] is None for s in starts(new))


@needs_node
def test_slow_list_gets_the_message_and_no_second_get(tmp_path, cands):
    """一覧の GET が遅い(5 秒を過ぎても返らない): 待っている間は読み直さない(GET は 1 本)。その間の基準の設定は「読めていません」
    (直す前は「…TensorRT エンジンが必要」)。返事が来たら前と同じに続き、基準の設定が通る"""
    c, di = cands
    steps = [['hold', MD, True], ['tab', 'drive'], ['tick'], ['tick'], ['tick'], ['baseline'], ['el', '#toast', 'text'], ['calls'],
             ['hold', MD, False], ['release', MD], ['tick'], ['baseline'], ['el', '#toast', 'text'], ['calls'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps, no_init=True)
    n, o = tagged(new), tagged(old)
    assert n[0] == MISS and o[0].endswith(WRONG)
    assert n[1] == o[1] == BASE_OK
    assert len(gets(vals(new, 'calls'), MD)) == len(gets(vals(old, 'calls'), MD)) == 1
    assert vals(new, 'calls')[1] == vals(old, 'calls')[1] and vals(new, 'posts') == vals(old, 'posts')


@needs_node
def test_no_reload_off_the_drive_tab_or_while_running(tmp_path, cands):
    """読み直すのは止まっていて走行タブにいるときだけ: (a) 失敗のあとすぐライブのタブへ → 何秒たっても一覧の GET は 1 本。
    (b) ほかの画面が走行中 → 走っている間は読み直さず、止まってから 2 回目の同期で 1 回(台本の上の状態の問い合わせは 404 なので、
    本物の画面で renderStatus が始める輪を realpoll で始める)"""
    c, di = cands
    a = [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tab', 'live'], ['tick'], ['tick'], ['tick'], ['calls']]
    res, _ = run_sim(tmp_path, c, a, no_init=True, name='a.json')
    assert len(gets(vals(res, 'calls'), MD)) == 1
    b = [['running', True], ['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['realpoll'], ['tick'], ['tick'], ['calls'],
         ['start'], ['el', '#toast', 'text'], ['running', False], ['tick'], ['calls'], ['tick'], ['calls'], ['eval', 'state.drive.models.length']]
    res, _ = run_sim(tmp_path, c, b, no_init=True, name='b.json')
    cs = vals(res, 'calls')
    assert [len(gets([x], MD)) for x in cs] == [1, 0, 1] and tagged(res)[-1] == 6
    assert tagged(res)[0] == 'モデルを選んでください'                                         # 走行中は前と同じ文(drIdle のときだけ変える)
    old, _ = run_sim(tmp_path, c, b, no_init=True, name='bo.json', app=BEFORE / 'app.js')
    assert tagged(old)[0] == 'モデルを選んでください' and [len(gets([x], MD)) for x in vals(old, 'calls')] == [1, 0, 0]


@needs_node
def test_reopen_failure_keeps_the_old_list_and_sends_the_same(tmp_path, cands):
    """一覧を前に読んだ画面で、開き直したときだけ失敗: 前の一覧のまま基準の設定・ブロック・▶ が通る(前と同じ)。選ぶ欄に短い文は
    出さない。1 秒後に一覧を 1 回読み直す(GET が 1 本ふえるだけ)。送った物・確認の文は直す前とまったく同じ"""
    c, di = cands
    steps = (F62.BASE + [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['el', '#dr-model', 'value'], ['baseline'],
                         ['el', '#toast', 'text'], ['tick'], ['tick'], ['pick', di, blk(c, di, 10)], ['check', '#dr-speed-ceiling', True],
                         ['start'], ['calls'], ['posts']])
    (new, tn), (old, to) = both(tmp_path, c, steps, server=F62.PRE_A)
    n, o = tagged(new), tagged(old)
    assert n == o and n[0] and n[1] == BASE_OK
    assert vals(new, 'posts') == vals(old, 'posts') and sent(new) == sent(old) and starts(new)[0]['status'] == 200
    assert len(gets(vals(new, 'calls')[-1:], MD)) == len(gets(vals(old, 'calls')[-1:], MD)) + 1
    assert tn == to


@needs_node
@pytest.mark.parametrize('how', ['clean', 'reloaded', 'reopened'])
def test_model_really_missing_keeps_the_engine_text(tmp_path, cands, how):
    """一覧は通ったが基準のモデルが一覧に無い(本当にエンジンが無い): 文は前と同じ「…一覧にありません(TensorRT エンジンが必要)」。
    一覧がいったん失敗し、読み直し(reloaded)・開き直し(reopened)で通ったあとも同じ(読めていない印は一覧が通ったら消す)"""
    c, di = cands
    others = sorted({cd['run'] for cd in c['candidates'] if cd.get('run')} - {c['baseline']['run']})
    pre = {'clean': [['tab', 'drive'], ['tick']],
           'reloaded': OPEN_FAIL + [['tick'], ['tick']],
           'reopened': OPEN_FAIL + [['tab', 'live'], ['tab', 'drive'], ['tick']]}[how]
    steps = pre + [['baseline'], ['el', '#toast', 'text'], ['calls']]
    (new, tn), (old, to) = both(tmp_path, c, steps, models=others, no_init=True)
    assert tagged(new)[0].endswith(WRONG) and tagged(old)[0].endswith(WRONG)
    if how == 'clean':
        assert tagged(new) == tagged(old) and tn == to and vals(new, 'calls') == vals(old, 'calls')


# ------------------------------------------------------------------ 前からの流れは前とまったく同じ(送る物・通信・文)
def _earlier_flows(c, di):
    """10/26〜27 の画面の流れ・6-1 回目の起き直しの場面・6-2 回目の流れ(test_ui_fix7 の 27 通り)と、7-1 回目の場面"""
    flows = T7._all_flows(c, di)
    look = T7.LOOK + [['tick'], ['el', '#dr-message', 'html'], ['start'], ['posts']]
    flows[('fix7r1', 'reentered')] = ([['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, RESET], ['api', T7.REENTER],
                                       ['tab', 'drive']] + look, {}, MORNING)
    flows[('fix7r1', 'after_notice')] = ([['tab', 'drive'], ['tick'], ['reboot', 2000.0, RESET], ['tick'], ['check', '#dr-rec-back-guard', True],
                                          ['check', '#dr-rec-hold-keep', True], ['api', {'ceil_left_mps': 4.2, 'launch_us': 200.0}],
                                          ['tick']] + look, {}, MORNING)
    flows[('fix7r1', 'net_fail')] = ([['tab', 'drive'], ['tick'], ['reboot', 2000.0, RESET], ['api', T7.REENTER], ['start_fail']] + look, {},
                                     MORNING)
    return flows


@needs_node
def test_all_earlier_flows_exactly_as_before(tmp_path, cands):
    """一覧が通り、▶ の catch の同期も通る流れ(30 通り): 送った POST(body と返事)・走行開始の params・確認の文・boot_at・通信の全部・
    タイマー・トースト・文が直す前の写しとまったく同じ = スイッチが全部 OFF のとき(と ON のときも)送る物はビット単位で同じ"""
    c, di = cands
    for i, (key, (st, extra, sp)) in enumerate(_earlier_flows(c, di).items()):
        st = st + [['calls'], ['ntimers']]
        new, tn = run_sim(tmp_path, c, st, name='n%d.json' % i, server=sp, **extra)
        old, to = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='o%d.json' % i, server=sp, **extra)
        assert tn == to and tagged(new) == tagged(old), key
        assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers'), key
        assert vals(new, 'posts') == vals(old, 'posts') and sent(new) == sent(old), key


# でたらめな朝の流れ(反証 7-2 回目の fuzz_morning と同じ混ぜ方。種を固定)。どの流れでも送る物は直す前とまったく同じ
GUARD, KEEP = '#dr-rec-back-guard', '#dr-rec-hold-keep'
DIST = [                                                 # (前に足す操作, 次の操作のあとに足す操作)
    lambda b: ([['reboot', b, RESET]], []),
    lambda b: ([['reboot_on_confirm', b, RESET]], []),
    lambda b: ([['down_on_confirm']], []),
    lambda b: ([['post_lost', 1]], []),
    lambda b: ([['status_fail_next', 1]], []),
    lambda b: ([['status_fail_next', 3]], []),
    lambda b: ([['down_all', True], ['tick'], ['down_all', False]], []),
    lambda b: ([['down_all', True], ['tick'], ['reboot', b, RESET], ['down_all', False]], []),
    lambda b: ([['tab', 'live'], ['models_fail', True], ['tab', 'drive'], ['models_fail', False]], []),
    lambda b: ([['start_lost'], ['status_fail_next', 1]], []),
    # 指摘の S1(確認の文の間に落ち、輪が 1 回失敗してから起き上がる)と S1c(起き直しの 409 のあと同期だけ途切れる)
    lambda b: ([['down_on_confirm']], [['tick'], ['reboot', b, RESET], ['down_all', False], ['tick']]),
    lambda b: ([['reboot_on_confirm', b, RESET], ['status_fail_next', 1]], [['tick']]),
]
REBOOTS = {0, 1, 7, 10, 11}                                # 起き直す(boot_at が進む)もの


def _fuzz_flow(rng, c, di):
    blocks = [blk(c, di, n) for n in (1, 5, 10, 15)]
    boot = 1000.0
    steps = [['tab', 'drive'], ['tick']]
    flow = []
    for _ in range(2):
        flow += [['baseline'], ['press_if_off', GUARD], ['press_if_off', KEEP], ['pick', di, rng.choice(blocks)], ['click', '#btn-dr-tg-up'],
                 ['press_if_off', '#dr-speed-ceiling'], ['press_if_off', '#dr-launch-boost'], ['start']]
    for s in flow:
        post = []
        if rng.random() < 0.3:
            k = rng.randrange(len(DIST))
            pre, post = DIST[k](boot + 1000.0)
            if k in REBOOTS:
                boot += 1000.0
            steps += pre
        steps.append(s)
        steps += post
        if s[0] == 'start':
            steps += [['tick'], ['el', '#toast', 'text'], ['running', False], ['tick'], ['press_if_off', GUARD], ['tick'],
                      ['press_if_off', KEEP], ['tick']]
        elif rng.random() < 0.7:
            steps += [['tick']]
    return steps + [['start'], ['tick'], ['posts']]


@needs_node
def test_random_mornings_send_exactly_the_same(tmp_path, cands):
    """でたらめな朝の流れ 30 本(起き直し・落ちる・途切れ・返事だけ届かない・一覧の失敗・指摘の S1・S1c を混ぜる。一覧は初めに読んである): 送った
    POST(body と返事)・走行開始の params・確認の文・boot_at・スイッチを押したか が直す前の写しとまったく同じ。違うのはトーストだけで、
    違いは知らせのあとの出し直しだけ"""
    c, di = cands
    rng = random.Random(72)
    reshown = 0
    for i in range(30):
        st = _fuzz_flow(rng, c, di)
        new, tn = run_sim(tmp_path, c, st, server=server_params(throttle_gain=0.9), name='fn.json')
        old, to = run_sim(tmp_path, c, st, server=server_params(throttle_gain=0.9), app=BEFORE / 'app.js', name='fo.json')
        assert vals(new, 'posts') == vals(old, 'posts') and sent(new) == sent(old), i
        assert [x for x in new if x['op'] == 'press_if_off'] == [x for x in old if x['op'] == 'press_if_off'], i
        k = old_plus_reshows(tn, to)
        assert k is not None, (i, tn, to)
        reshown += k
    assert reshown >= 1                                  # 出し直しの場面を含む(種 72 では 1 つ)
