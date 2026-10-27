"""画面の係の直し(2026-10-27 fix6。反証 6 回目の指摘 w1-ui-ui61-1・w1-ui-ui61-2。~/jetformer_reports/fix_20261026/fix6/fix_ui_r1.md)の試験。

  w1-ui-ui61-2: ▶ がサーバーの起き直しで断られた(409「管理画面のサーバーが HH:MM:SS に起動し直しました(…)…始めません…」)とき、
    app.js の catch がすぐ同期し(pollDrive)、その返事で起動し直しの知らせのトースト(noteServerBoot「…既定に戻っています」)が出て、
    409 の文が数十 ms で消えた。見える文に「走行を始めていない」が無かった。直し(app.js。表示だけ): catch は同期を待ち、起き直しで
    断った文か、同期のあいだに知らせが出たときは、断った文をもう一度出す(頭に、同期した status で止まっていれば「走行は始まって
    いません(停止中)。」。断った文に起動し直しが無ければ知らせを後ろに)。同期を待つあいだに次の ▶ を押したときは
    出し直さない。走行開始で送る body・断る決まり・同期の回数は前と同じ(この試験が直す前の写しと比べる)。
  w1-ui-ui61-1: ユーザーが見る文書の「画面の版 8」(画面は版 9)。文書はこの係の担当の外なので、直し案(experiments/20261026_fix/
    fix6/ui/docs_ban9.py。main session が入れる)を、今の文書と 10/27 09:4x の写し(tests/fixtures/fix6_ui_docs_before/)に当てて見る。

本物の app.js(今の版と直す前の写し tests/fixtures/fix6_ui_before/app.js = fix5 のあとの版。sha256 を照らす)を node の vm で
(tests/fixtures/app_sim_fix6.js = app_sim_fix5.js + 本物の 409 の文 + 走行開始だけの通信の途切れ)。409 の文は本物の
server._boot_refusal で作る(import だけ。起動しない)。車・ブラウザ・jetformer-web には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix6.py

JF_UI6_APP=<app.js の写し> を付けると、その写しを「今の版」として流す(わざと壊す台本 experiments/20261026_fix/fix6/ui/mut/ が使う)。
そのあと直す 6-2 回目(w2-ui-w1-ui-r62-1。tests/test_ui_fix6_r2.py)が app.js を直した。動きの試験は今の app.js で流す(6-2 回目の直しは
一覧の GET が通らないときだけ効くので、ここの場面では前と同じ)。「違いは 3 か所だけ」は 6-1 回目のあとの版の写し
(tests/fixtures/fix6_ui_r2_before/)で見る。
"""
import difflib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException

from jetformer import drive
from jetformer.web import bench
from jetformer.web import server as srv

from test_ui_fix3 import _flows
from test_vbat_desc import fake_start, tracker, feed      # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
APP = Path(os.environ.get('JF_UI6_APP') or STATIC / 'app.js')
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix6_ui_before'
# この組(6-1 回目)のあとの版の写し(e41c2f3e…)。そのあと 6-2 回目(w2-ui-w1-ui-r62-1。tests/test_ui_fix6_r2.py)が app.js を直したので、
# 「違いは 3 か所だけ」はこの写しで見る(今の app.js との違いは test_ui_fix6_r2.py が見る)。JF_UI6_APP を付けたときはその写し
R1_AFTER = ROOT / 'tests' / 'fixtures' / 'fix6_ui_r2_before'
DOCS_BEFORE = ROOT / 'tests' / 'fixtures' / 'fix6_ui_docs_before'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix6.js'
DOCS_TOOL = ROOT / 'experiments' / '20261026_fix' / 'fix6' / 'ui' / 'docs_ban9.py'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
PREFIX = '走行は始まっていません(停止中)。'
BOOT_TOAST = '管理画面のサーバーが 09:33 に起動し直しました。朝に入れた設定(後ろの見張り・左の値・発進の前進の強さなど)は既定に戻っています'
OLD_CATCH = "  } catch (e) { toast(e.message, true); pollDrive(); }     // 断られた・届かなかった: すぐ同期して、表示と確認の文を今のサーバーに(t1)"
OLD_TOAST = ("  toast(`管理画面のサーバーが ${hm} に起動し直しました。朝に入れた設定(後ろの見張り・左の値・発進の前進の強さなど)は既定に"
             "戻っています`, true);")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_before_copy_and_only_three_places_changed():
    """直す前の写しは fix5 のあとの版(fcc18443…)。この組のあとの版(e41c2f3e…。写し tests/fixtures/fix6_ui_r2_before/。6-2 回目の
    直す前)との違いは ▶ の番号の 1 行・▶ の catch・noteServerBoot のトーストの 3 か所だけ。版(DRIVE_CLIENT・client)・server.py・
    index.html は変えていない"""
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert set(sums) == {'app.js'} and _sha(BEFORE / 'app.js') == sums['app.js']
    assert sums['app.js'].startswith('fcc18443ce4b')
    after = Path(os.environ['JF_UI6_APP']) if os.environ.get('JF_UI6_APP') else R1_AFTER / 'app.js'
    if not os.environ.get('JF_UI6_APP'):
        r1 = dict(reversed(x.split()) for x in (R1_AFTER / 'SHA256SUMS').read_text().splitlines() if x.strip())
        assert r1 == {'app.js': _sha(after)} and r1['app.js'].startswith('e41c2f3ea796')
    a = (BEFORE / 'app.js').read_text(encoding='utf-8').splitlines()
    b = after.read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    removed = [a[i] for o in ops for i in range(o[1], o[2])]
    added = [b[j] for o in ops for j in range(o[3], o[4])]
    assert removed == [OLD_CATCH, OLD_TOAST], removed
    assert len(ops) == 3, ops                       # ▶ の番号の 1 行・catch・noteServerBoot
    assert added[0].startswith('  const seq = state.drive.startSeq = ')
    assert sum('pollDrive' in x for x in added) == 1 and [x.strip() for x in added].count('try { await pollDrive(); } finally {') == 1
    assert not any(re.search(r'\bapi\(|body\.|fetch\(', x) for x in added)     # 送る物は足していない
    js = after.read_text(encoding='utf-8')
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert _sha(ROOT / 'jetformer' / 'web' / 'server.py').startswith('ad7a7d05f7a9')          # fix5 のあとの版のまま
    assert _sha(STATIC / 'index.html').startswith('ef81bb6017d6')


# ------------------------------------------------------------------ 画面を node で流す
@pytest.fixture(scope='module')
def cands():
    c = bench.load_candidates()
    days = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    if not days:
        pytest.skip('data/bench_candidates.json に 2026-11-03 のブロックがありません')
    return dict(c, today='2026-11-03'), days[0]


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def server_params(**over):
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(over)
    return sp


# 起き直しで断る本物の文(画面の boot_at 1000 = 09:16:40、今のサーバー 2000 = 09:33:20。無い = まだ読んでいない)
REAL = {'1000': srv._boot_refusal({'boot_at': 1000.0}, 2000.0), 'null': srv._boot_refusal({}, 2000.0)}
MORNING = server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)


def run_sim(tmp_path, c, steps, app=None, name='input.json', server=None, **extra):
    inp = tmp_path / name
    d = {'bench': c, 'server_params': server if server is not None else MORNING, 'steps': steps, 'server_client_version': 8, 'confirm': True, 'boot_at': 1000.0,
         'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}), 'script_src': 'http://x/static/app.js?v=1793667600',
         'server_boot_check': True, 'boot_refusal_texts': REAL}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app or APP), str(STATIC / 'index.html'), str(inp)], capture_output=True, text=True,
                       timeout=120, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def els(res):
    return [x['value'] for x in res if x['op'] == 'el']


def posts(res):
    return [x['value'] for x in res if x['op'] == 'posts']


HOLD = ("globalThis.__held = []; globalThis.__f0 = fetch; globalThis.__holdN = 1; fetch = function (u, o) { if (o && o.method === 'POST' "
        "&& String(u).startsWith('/api/drive/start') && __holdN > 0) { __holdN--; return new Promise((r, j) => __held.push(() => "
        "__f0(u, o).then(r, j))); } return __f0(u, o); }; 1")
CLICK = "document.querySelector('#btn-dr-start').fire('click'); globalThis.__held.length"     # await しない(POST は止めてある)
RELEASE = "const f = __held.shift(); if (f) f(); 'released'"
# GET /api/drive/status を、出した時の返事のまま止める(遅れて届く古い返事)
HOLD_GET = ("globalThis.__heldG = []; globalThis.__f1 = fetch; fetch = function (u, o) { if (String(u).startsWith('/api/drive/status')) "
            "{ const p = __f1(u, o); return new Promise((r, j) => __heldG.push(() => p.then(r, j))); } return __f1(u, o); }; 1")
RELEASE_GETS = "const g = __heldG.splice(0); for (const f of g) f(); g.length"


def setup_steps(c, di):
    return [['pick', di, blk(c, di, 15)], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True],
            ['check', '#dr-launch-boost', True], ['poll'], ['posts']]


TAIL = [['timers'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
R1000 = PREFIX + REAL['1000']
# 場面: (操作, 入力の足し, 今の版の最後のトースト, 直す前の写しの最後のトースト)
SCENES = {
    'A_no_sync': ([['reboot', 2000.0, RESET], ['start']], {}, R1000, BOOT_TOAST),
    'B_dialog_only': ([['reboot_on_confirm', 2000.0, RESET], ['start']], {}, R1000, BOOT_TOAST),
    'C_overdue_get_first': ([['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['realpoll'], ['eval', RELEASE]],
                            {}, R1000, REAL['1000']),
    'D_overdue_post_first': ([['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['gstart'], ['eval', RELEASE],
                              ['timers'], ['gend']], {}, R1000, BOOT_TOAST),
    'E_get_inflight': ([['eval', HOLD], ['gstart'], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['gend'], ['eval', RELEASE]],
                       {}, R1000, BOOT_TOAST),
    'F_net_fail_then_up': ([['reboot_on_confirm', 2000.0, RESET], ['start_fail'], ['start']], {},
                           PREFIX + 'Failed to fetch (sim6: 走行開始が届かない)。' + BOOT_TOAST, BOOT_TOAST),
    'G_old_page_and_reboot': ([['reboot_on_confirm', 2000.0, RESET], ['start']], {'server_client_version': 10},
                              PREFIX + '管理画面が古いままです。' + BOOT_TOAST, BOOT_TOAST),
    'I_never_read_boot': ([['reboot', 2000.0, RESET], ['start']], {'boot_at': None}, PREFIX + REAL['null'], REAL['null']),
}


@needs_node
@pytest.mark.parametrize('name', list(SCENES))
def test_refused_start_keeps_the_refusal_visible(tmp_path, cands, name):
    """起き直しで断られた ▶ のあと、どの返事の順でも最後に見える文は「走行は始まっていません(停止中)。」+ 断った文(直す前の写しは
    A・B・D・E・F・G で「…既定に戻っています」だけが残った)。送った POST(body と返事)・走行タブの文・バッジは直す前と同じ。
    トーストは、出し直した 1 つのほかは直す前と同じ順"""
    c, di = cands
    steps, extra, want_new, want_old = SCENES[name]
    st = setup_steps(c, di) + steps + TAIL
    new, tn = run_sim(tmp_path, c, st, name='n.json', **extra)
    old, to = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='o.json', **extra)
    (fn, mn, pn), (fo, mo, po) = els(new)[-3:], els(old)[-3:]
    assert fn == want_new, fn
    assert fo == want_old, fo                                       # 直す前の写しで指摘が起きる(C・I は前から 409 が残る)
    assert pn == po == '停止中' and mn == mo
    assert posts(new) == posts(old)                                 # 送った物は同じ
    starts = [p for x in posts(new) for p in x if p[0] == '/api/drive/start']
    assert len(starts) == 1 and starts[0][2] in (409, None)     # 1 本だけ送り、断られた(409)・届かなかった(None)
    assert [t for t in tn if not t.startswith(PREFIX)] == to and sum(t.startswith(PREFIX) for t in tn) == 1


@needs_node
def test_no_prefix_when_synced_status_is_running(tmp_path, cands):
    """断られた ▶ のあと同期した status が走行中(ほかの画面が始めた)なら「走行は始まっていません」を付けない(断った文だけ)"""
    c, di = cands
    st = setup_steps(c, di) + [['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['running', True],
                               ['eval', RELEASE]] + TAIL
    new, tn = run_sim(tmp_path, c, st, name='n.json')
    old, to = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='o.json')
    assert els(new)[-3] == REAL['1000'] == els(old)[-3] and not any(t.startswith(PREFIX) for t in tn)
    assert tn == to + [REAL['1000']] and posts(new) == posts(old)


@needs_node
def test_next_start_while_waiting_is_not_overwritten(tmp_path, cands):
    """断られた ▶ の同期の返事が遅れ、そのあいだに表示が合って次の ▶ で走り出した: 遅れて届いた古い返事(止まっている)のあとも、
    前の ▶ の文(「走行は始まっていません」)で走り出した知らせを消さない"""
    c, di = cands
    st = setup_steps(c, di) + [['eval', HOLD], ['reboot', 2000.0, RESET], ['eval', CLICK], ['eval', HOLD_GET], ['eval', RELEASE],
                               ['poll'], ['start'], ['eval', RELEASE_GETS]] + TAIL
    new, tn = run_sim(tmp_path, c, st, name='n.json')
    old, to = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='o.json')
    starts = [p for x in posts(new) for p in x if p[0] == '/api/drive/start']
    assert [p[2] for p in starts] == [409, 200] and posts(new) == posts(old)
    assert not any(t.startswith(PREFIX) for t in tn) and tn == to
    assert els(new)[-3] == els(old)[-3] and 'を開始しました' in ''.join(tn[-2:])


@needs_node
def test_other_refusals_and_flows_same_as_before(tmp_path, cands):
    """起き直しの無い断り(版)と、10/26〜27 の画面の流れ(test_ui_fix3 の _flows)は、送った物・確認の文・トースト・走行タブの文が
    直す前の写しとまったく同じ"""
    c, di = cands
    flows = dict(_flows(c, di))
    flows['版で断られる'] = ([['start'], ['start']], {})
    for i, (name, (steps, sp)) in enumerate(flows.items()):
        st = [['poll']] + steps + [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['posts']]
        kw = {'server_client_version': 10} if name == '版で断られる' else {}
        new, tn = run_sim(tmp_path, c, st, name='n%d.json' % i, server=server_params(**sp), **kw)
        old, to = run_sim(tmp_path, c, st, app=BEFORE / 'app.js', name='o%d.json' % i, server=server_params(**sp), **kw)
        assert tn == to and els(new) == els(old) and posts(new) == posts(old), name
        sn = [x for x in new if x['op'] == 'start']
        so = [x for x in old if x['op'] == 'start']
        assert [(a['sent'], a['confirm'], a['boot_at'], a['status']) for a in sn] == \
               [(b['sent'], b['confirm'], b['boot_at'], b['status']) for b in so], name


def test_refusal_pattern_matches_only_the_boot_refusal(monkeypatch, tracker, fake_start):
    """app.js の bootRefusal の形(/^管理画面のサーバーが .*に起動し直しました\\(/)は本物の _boot_refusal の文(どの boot_at でも)に
    当たり、ほかの 409(版・app.js の時刻・学習中)と起動し直しの知らせのトーストには当たらない"""
    js = APP.read_text(encoding='utf-8')
    pat = re.compile(re.search(r'bootRefusal = /(.+?)/\.test\(m\)', js).group(1))
    for seen in (None, 1000.0, float('nan'), 'x', 1e30, -5.0):
        assert pat.search(srv._boot_refusal({'boot_at': seen}, 2000.0)), seen
    assert pat.search(srv._boot_refusal({'boot_at': 1000.0}, srv.driver.boot_at))
    assert not pat.search(BOOT_TOAST) and not pat.search(PREFIX + REAL['1000'])
    feed(tracker, 97.0, 100.0, 8.3)
    body = {'client': 9, 'boot_at': srv.driver.boot_at, 'run': bench.B_RUN, 'mode_id': 5, 'description': 'AI走行 B', 'dry_run': True,
            'record': False, 'params': {}}
    others = []
    for bad in ({'client': 8}, {'build': 1}):
        with pytest.raises(HTTPException) as e:
            srv.api_drive_start(dict(body, **bad))
        others.append(e.value.detail)
    monkeypatch.setattr(srv.trainer.__class__, 'running', property(lambda self: True))
    with pytest.raises(HTTPException) as e:
        srv.api_drive_start(body)
    others.append(e.value.detail)
    assert len(others) == 3 and '古いまま' in others[0] and '古いまま' in others[1] and '学習中' in others[2]
    assert not any(pat.search(t) for t in others) and fake_start == []



# ------------------------------------------------------------------ w1-ui-ui61-1 文書の版(直し案)
def _docs_tool():
    spec = importlib.util.spec_from_file_location('docs_ban9_fix6', DOCS_TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_docs_proposal_fixes_every_old_version_in_the_copies():
    """10/27 09:4x の文書の写し(手順書・todo_user_1026.md・README.md・fix3/approvals.md)に直し案を当てると、今の版(9)と違う
    「画面の版 N」「版 N 以下」(N 以下は 8)が 1 つも残らない。どの直しも写しの中でちょうど 1 回当たる。1 つでも外すと残る
    (手順書の付録の版の歴史の 1 か所だけは、書き足すだけで残りを直さない)"""
    m = _docs_tool()
    sums = dict(reversed(x.split(None, 1)) for x in (DOCS_BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert set(sums) == set(m.FILES)
    for f, h in sums.items():
        assert _sha(DOCS_BEFORE / f) == h, f
    assert m.versions() == (9, 9, 9)
    texts = m.read_all(DOCS_BEFORE)
    before = m.stale(texts, 9)
    assert len(before) == 17 and {f for f, _, _ in before} == set(m.FILES)
    new, rep = m.apply_text(texts)
    assert [s for _, _, s in rep] == ['apply'] * len(m.EDITS)
    assert m.stale(new, 9) == []
    for i, (f, old, _) in enumerate(m.EDITS):
        less, _ = m.apply_text(texts, m.EDITS[:i] + m.EDITS[i + 1:])
        if old.startswith('(`DRIVE_CLIENT_VERSION`'):
            assert m.stale(less, 9) == []
        else:
            assert m.stale(less, 9), (f, old)
    again, rep2 = m.apply_text(new)                                  # 入れたあとにもう一度: 何もしない
    assert again == new and [s for _, _, s in rep2] == ['done'] * len(m.EDITS)


def test_docs_proposal_still_applies_to_the_live_documents():
    """今の文書(main session が入れる前でも後でも): どの直しも「入れる」か「済」で、入れたあとに古い版の数が残らない"""
    m = _docs_tool()
    if not all((m.REPORTS / f).exists() for f in m.FILES):
        pytest.skip('~/jetformer_reports の文書がありません')
    new, rep = m.apply_text(m.read_all())
    assert all(s in ('apply', 'done') for _, _, s in rep), [(f, o[:30], s) for f, o, s in rep if s not in ('apply', 'done')]
    assert m.stale(new, m.versions()[0]) == []
