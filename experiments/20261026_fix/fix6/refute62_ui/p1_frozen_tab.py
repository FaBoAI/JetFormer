"""反証 6-2 回目・ui の P1: 走行タブに戻ったときに /api/drive/models が通らないと、表示の問い合わせ(pollDrive)が始まらない。

本物の app.js(今の版 9。e41c2f3e…)と index.html を node の vm で(sim6.js = tests/fixtures/app_sim_fix6.js の写し + down_all・tick)。
loadDrive は models の GET が失敗すると toast だけで return し、pollDrive を呼ばない。ほかのタブでは pollDrive の輪は止まる
(state.tab が drive でないと次のタイマーを掛けない)。→ 走行タブは前の表示のまま固まり、drLost も呼ばれない(paramsLoaded は true のまま)。

A: 起き直し(RESET_ON_BOOT)。ライブのタブ → サーバーが落ちる → 走行タブ(models が通らない)→ サーバーが起き直して戻る。
B: 起き直さない(通信の途切れで models だけ通らない)。そのあと 2-5 の curl で左 4.2(手順書 5 章)。
どちらも: 固まったままか・▶ が押せるか・ブロックが入るか・天井のスイッチの確認の文・▶ の確認の文・返事。本物の Driver に POST を入れる。
"""
import json
import common as C

c = C.bench_today()
di = C.day_index(c)
b10, b15 = C.blk(c, di, 10), C.blk(c, di, 15)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
out = {}


def look():
    return [['tick'], ['tick'], ['tick'], ['ntimers'], ['eval', 'String(driveTimer)'], ['eval', 'state.drive.paramsLoaded'],
            ['eval', 'state.drive.bootAt'], ['read', 'st_params'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-message', 'html']]


def lines(t, words):
    return [l for l in t.split('\n') if any(w in l for w in words)]


W = ('見張り', '残す', '天井', '発進の前進:', '速度ゲイン')

# ---- A: 起き直し
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
steps = [['realpoll'], ['eval', "showTab('live')"], ['tick'], ['tick'], ['down_all', True], ['eval', "showTab('drive')"], ['tick'],
         ['reboot', 2000.0, RESET], ['down_all', False]] + look() + [
         ['pick', di, b15], ['read', 'note'], ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True],
         ['start'], ['read', 'toasts'], ['start'], ['posts']]
res, toasts = C.sim(c, steps, server=pre, server_boot_check=True, server_client_version=9)
ev = [x.get('value') for x in res if x['op'] == 'eval']
els = [x['value'] for x in res if x['op'] == 'el']
chk = [x for x in res if x['op'] == 'check']
st = C.starts(res)
out['A_restart'] = {
    'timers_after_back_ms': [x['value'] for x in res if x['op'] == 'ntimers'][0], 'driveTimer': ev[2], 'paramsLoaded': ev[3],
    'page_bootAt': ev[4], 'st_params_shown': C.pick(C.reads(res, 'st_params')[0]),
    'start_button_disabled': els[0], 'dr_message_has_lost_text': 'つながっていません' in (els[1] or ''),
    'block_note_head': C.reads(res, 'note')[0][:120],
    'ceiling_switch_confirm': lines(chk[0]['confirm'], ('左旋回中',)), 'launch_switch_confirm': lines(chk[1]['confirm'], ('前進',))[:1],
    'starts': [{'status': s['status'], 'boot_at': s['boot_at'], 'confirm': lines(s['confirm'], W)} for s in st],
    'server_after': C.pick(C.reads(res, 'server')[0]) if C.reads(res, 'server') else None,
    'toasts_tail': toasts[-3:]}

# ---- B: 起き直さない(models だけ通らない)→ 2-5 の curl で左 4.2
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.1)
steps = [['realpoll'], ['eval', "showTab('live')"], ['tick'], ['tick'], ['down_all', True], ['eval', "showTab('drive')"], ['tick'],
         ['down_all', False], ['api', {'ceil_left_mps': 4.2}]] + look() + [
         ['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server'], ['posts']]
res, toasts = C.sim(c, steps, server=pre, server_boot_check=True, server_client_version=9)
ev = [x.get('value') for x in res if x['op'] == 'eval']
els = [x['value'] for x in res if x['op'] == 'el']
chk = [x for x in res if x['op'] == 'check']
st = C.starts(res)
plist = C.posts(res)[-1]
d = C.driver(params=pre)
real = C.replay(d, plist)
out['B_no_restart_curl_left'] = {
    'timers_after_back_ms': [x['value'] for x in res if x['op'] == 'ntimers'][0], 'driveTimer': ev[2], 'paramsLoaded': ev[3],
    'st_params_shown_left': C.reads(res, 'st_params')[0].get('ceil_left_mps'),
    'start_button_disabled': els[0], 'dr_message_has_lost_text': 'つながっていません' in (els[1] or ''),
    'ceiling_switch_confirm': lines(chk[0]['confirm'], ('左旋回中',)),
    'start': [{'status': s['status'], 'confirm': lines(s['confirm'], ('天井',))} for s in st],
    'server_left_at_start': C.reads(res, 'server')[0].get('ceil_left_mps'),
    'real_driver': {'replay': real, 'ceil_left_mps': d.params['ceil_left_mps'], 'speed_ceiling': d.params['speed_ceiling']}}
print(json.dumps(out, ensure_ascii=False, indent=1))
(C.HERE / 'out' / 'p1_frozen_tab.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
C.cleanup()
