"""w1-ui-ui61-2 の打ち消し: 起き直しで 409 のあと、画面に最後に残るもの(トースト・走行タブの文・バッジ・スイッチ)。
no_sync(起き直し → 同期前に ▶)と dialog_only(確認の文を読んでいる間に起き直し)。本物の app.js・index.html を node の vm で。"""
import json
import common6 as C

c = C.bench_today()
di = C.day_index(c)
b15 = C.blk(c, di, 15)
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
setup = [['pick', di, b15], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True], ['poll'], ['posts']]
tail = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['el', '#dr-after', 'hidden'],
        ['read', 'st_params']]
out = {}
for name, pre_start in (('no_sync', [['reboot', 2000.0, RESET]]), ('dialog_only', [['reboot_on_confirm', 2000.0, RESET]])):
    res, toasts = C.sim(c, setup + pre_start + [['start'], ['poll']] + tail, server=pre, server_boot_check=True)
    st = C.starts(res)
    els = [x['value'] for x in res if x['op'] == 'el']
    rd = [x['value'] for x in res if x['op'] == 'read']
    out[name] = {'start_status': st[0]['status'], 'start_boot_at': st[0]['boot_at'],
                 'toasts_after_start': toasts[-3:], 'final_toast': els[0], 'dr_message': els[1][:600], 'dr_phase': els[2],
                 'dr_after_hidden': els[3],
                 'server_params_seen': {k: rd[0].get(k) for k in ('rec_back_guard', 'rec_hold_keep', 'ceil_left_mps', 'launch_us')} if rd and isinstance(rd[0], dict) else rd}
print(json.dumps(out, ensure_ascii=False, indent=1))
C.cleanup()
