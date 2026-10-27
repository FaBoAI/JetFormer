"""反証 w1-ui-ui61-2: 起き直しで断られた 409 の文が「起動し直しました」のトーストに上書きされるか。

本物の app.js・index.html を sim6.js(ui61/probe の写し)で動かす。サーバーは偽物(sim6.js の中)。装置・jetformer-web には触れない。
場面:
  A no_sync         : 起き直し(画面は問い合わせていない)→ ▶ → 409(ui61 の P1 と同じ)
  B dialog_only     : ▶ の確認の文の間に起き直し → 409(偽のサーバーの返事は即座。catch の pollDrive の GET だけ)
  C dialog_overdue_get_first : 本物のブラウザの順の 1 つ。確認の文の間に 1 秒の問い合わせのタイマーが期限切れ →
                      OK のあと POST を出した直後にそのタイマーの GET が出る。GET(B2)の返事が POST(409)より先に届く
  D dialog_overdue_post_first: C と同じだが POST(409)の返事が先に届き、期限切れのタイマーの GET の返事は後
  E dialog_get_inflight: ▶ を押したとき前の問い合わせ(B1)が届く途中。その返事 → POST 409 の順
"""
import json
import common6 as C

c = C.bench_today()
di = C.day_index(c)
b15 = C.blk(c, di, 15)
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
setup = [['pick', di, b15], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True],
         ['poll'], ['posts']]
HOLD = ("globalThis.__held = []; globalThis.__f0 = fetch; fetch = function (u, o) { if (o && o.method === 'POST' && "
        "String(u).startsWith('/api/drive/start')) return new Promise((r, j) => __held.push(() => __f0(u, o).then(r, j))); "
        "return __f0(u, o); }; 1")
CLICK = "document.querySelector('#btn-dr-start').fire('click'); globalThis.__held.length"   # await しない(POST は止めてある)
RELEASE = "const f = __held.shift(); if (f) f(); 'released'"
TAIL = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['eval', '__T.state.drive.bootAt'],
        ['posts']]
SC = {
    'A_no_sync': setup + [['reboot', 2000.0, RESET], ['start']] + TAIL,
    'B_dialog_only': setup + [['reboot_on_confirm', 2000.0, RESET], ['start']] + TAIL,
    'C_dialog_overdue_get_first': setup + [['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK],
                                           ['realpoll'], ['el', '#toast', 'text'], ['eval', RELEASE], ['timers']] + TAIL,
    'D_dialog_overdue_post_first': setup + [['eval', HOLD], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK],
                                            ['gstart'], ['eval', RELEASE], ['timers'], ['el', '#toast', 'text'], ['gend']] + TAIL,
    'E_dialog_get_inflight': setup + [['eval', HOLD], ['gstart'], ['reboot_on_confirm', 2000.0, RESET], ['eval', CLICK],
                                      ['gend'], ['eval', RELEASE], ['timers']] + TAIL,
}
out = {}
for name, steps in SC.items():
    n_setup = len(setup)
    res, toasts = C.sim(c, steps, server=pre, server_boot_check=True)
    els = [x for x in res if x['op'] == 'el']
    final_toast, msg, phase = els[-3]['value'], els[-2]['value'], els[-1]['value']
    starts = [p for x in res if x['op'] == 'posts' for p in x['value'] if p[0] == '/api/drive/start']
    mid = [x['value'] for x in els[:-3]]
    out[name] = {
        'start_posts': [{'boot_at': p[1].get('boot_at'), 'status': p[2]} for p in starts],
        'toast_mid': mid,
        'final_toast': final_toast,
        'final_toast_is_409': final_toast.startswith('管理画面のサーバーが起動し直しました(sim5'),
        'final_toast_says_not_started': ('始めません' in final_toast) or ('始めていません' in final_toast),
        'msg_says_not_started': ('始めません' in msg) or ('始めていません' in msg) or ('始まって' in msg),
        'phase': phase,
        'bootAt_after': [x['value'] for x in res if x['op'] == 'eval'][-1],
        'toasts_tail': toasts[-4:],
        'msg_head': msg[:400],
    }
print(json.dumps(out, ensure_ascii=False, indent=1))
C.cleanup()
