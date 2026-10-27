"""fix6 画面の係(反証 6 回目 w1-ui-ui61-2)の確かめ: ▶ が起き直しで断られた(409)とき、最後に見える文はどれか。

本物の app.js(引数 before = tests/fixtures/fix6_ui_before/app.js = 直す前、after = 今の jetformer/web/static/app.js)と index.html を
tests/fixtures/app_sim_fix6.js(偽のサーバー)で動かす。409 の文は本物の server._boot_refusal で作る(import だけ。起動しない)。
装置・jetformer-web には触れない。出力: 標準出力に JSON(場面ごとの最後のトースト・送った走行開始・バッジ)。

場面(反証の r1_order.py の A〜E と同じ順の作り方 + F・G):
  A no_sync: 起き直し(画面は問い合わせていない)→ ▶ → 409
  B dialog_only: ▶ の確認の文の間に起き直し → 409
  C overdue_get_first: 確認の文の間に 1 秒の問い合わせのタイマーが期限切れ。その GET(B2)の返事が POST(409)より先
  D overdue_post_first: C と同じだが POST(409)の返事が先、期限切れのタイマーの GET の返事は後
  E get_inflight: ▶ を押したとき前の問い合わせ(B1)が届く途中。その返事 → 409 → catch の GET(B2)
  F net_fail_then_up: 確認の文の間に起き直し、POST は通信の途切れ(届かない)、catch の GET は通る(B2)
  G old_page_and_reboot: 版の照らしで断られる(サーバーの版 10 を真似る)+ 確認の文の間に起き直し
  H no_reboot_refused: 起き直していない。版で断られる(前と同じトーストだけのはず)
  I never_read_boot: 画面は boot_at の無いサーバー(再起動の前の 07:41 の版)を読んだまま。再起動のあとの ▶ → 409「まだ読んでいません」
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive               # noqa: E402
from jetformer.web import bench           # noqa: E402
from jetformer.web import server as srv   # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else 'after'
APP = {'before': JF / 'tests' / 'fixtures' / 'fix6_ui_before' / 'app.js', 'after': JF / 'jetformer' / 'web' / 'static' / 'app.js'}[which]
SIM = JF / 'tests' / 'fixtures' / 'app_sim_fix6.js'
STATIC = JF / 'jetformer' / 'web' / 'static'
NODE = shutil.which('node')
TMP = Path(tempfile.mkdtemp(prefix='ui6_', dir=str(Path(__file__).resolve().parent)))

c = dict(bench.load_candidates(), today='2026-11-03')
di = next(i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03')
b15 = next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == 15)
pre = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=0.9, lookahead=3, start_kick=1,
           rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
REAL = {'1000': srv._boot_refusal({'boot_at': 1000.0}, 2000.0), 'null': srv._boot_refusal({}, 2000.0)}
setup = [['pick', di, b15], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True],
         ['poll'], ['posts']]
HOLD = ("globalThis.__held = []; globalThis.__f0 = fetch; fetch = function (u, o) { if (o && o.method === 'POST' && "
        "String(u).startsWith('/api/drive/start')) return new Promise((r, j) => __held.push(() => __f0(u, o).then(r, j))); "
        "return __f0(u, o); }; 1")
CLICK = "document.querySelector('#btn-dr-start').fire('click'); globalThis.__held.length"
RELEASE = "const f = __held.shift(); if (f) f(); 'released'"
TAIL = [['timers'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
SC = {
    'A_no_sync': (setup + [['reboot', 2000.0, RESET], ['start']] + TAIL, {}),
    'B_dialog_only': (setup + [['reboot_on_confirm', 2000.0, RESET], ['start']] + TAIL, {}),
    'C_overdue_get_first': (setup + [['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['realpoll'],
                                     ['eval', RELEASE]] + TAIL, {}),
    'D_overdue_post_first': (setup + [['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['gstart'],
                                      ['eval', RELEASE], ['timers'], ['gend']] + TAIL, {}),
    'E_get_inflight': (setup + [['eval', HOLD], ['gstart'], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK], ['gend'],
                                ['eval', RELEASE]] + TAIL, {}),
    'F_net_fail_then_up': (setup + [['reboot_on_confirm', 2000.0, RESET], ['start_fail'], ['start']] + TAIL, {}),
    'G_old_page_and_reboot': (setup + [['reboot_on_confirm', 2000.0, RESET], ['start']] + TAIL, {'server_client_version': 10}),
    'H_no_reboot_refused': (setup + [['start']] + TAIL, {'server_client_version': 10}),
    'I_never_read_boot': (setup + [['reboot', 2000.0, RESET], ['start']] + TAIL, {'boot_at': None}),
}
out = {}
for i, (name, (steps, extra)) in enumerate(SC.items()):
    inp = TMP / ('in_%d.json' % i)
    d = {'bench': c, 'server_params': pre, 'steps': steps, 'server_client_version': 8, 'confirm': True, 'boot_at': 1000.0,
         'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}), 'script_src': 'http://x/static/app.js?v=1793667600',
         'server_boot_check': True, 'boot_refusal_texts': REAL}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(APP), str(STATIC / 'index.html'), str(inp)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-3000:])
    o = json.loads(r.stdout.strip().splitlines()[-1])
    res, toasts = o['steps'], o['toasts']
    els = [x['value'] for x in res if x['op'] == 'el']
    final, msg, phase = els[-3:]
    st = [p for x in res if x['op'] == 'posts' for p in x['value'] if p[0] == '/api/drive/start']
    out[name] = {'start_posts': [{'boot_at': p[1].get('boot_at'), 'status': p[2] if len(p) > 2 else None} for p in st],
                 'final_toast': final, 'final_has_not_started': any(w in final for w in ('始めません', '始めていません', '始まっていません')),
                 'phase': phase, 'boot_note_in_tab': '起動のたびに既定に戻るもの' in msg, 'toasts_after_setup': toasts[-4:]}
print(json.dumps({'app': str(APP), 'scenes': out}, ensure_ascii=False, indent=1))
shutil.rmtree(TMP, ignore_errors=True)
