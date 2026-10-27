"""P1: 起き直しで 409 になったあと、最後に見えるトースト(#toast の文字)。P4: 試験の選択肢の確認の文を読んでいる間に起き直し → ▶"""
import json
import common6 as C

c = C.bench_today()
di = C.day_index(c)
b15 = C.blk(c, di, 15)
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
setup = [['pick', di, b15], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True], ['poll'], ['posts']]
out = {}
# P1: 同期しないまま ▶ → 409 → 画面は pollDrive。最後のトースト・状態
res, toasts = C.sim(c, setup + [['reboot', 2000.0, RESET], ['start'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html'],
                               ['el', '#dr-phase', 'text'], ['posts']], server=pre, server_boot_check=True)
st = C.starts(res)
out['P1'] = {'start': {k: st[0][k] for k in ('status', 'boot_at')}, 'final_toast': [x['value'] for x in res if x['op'] == 'el'][0],
             'toasts_after_start': toasts[-3:], 'msg': [x['value'] for x in res if x['op'] == 'el'][1][:300]}
# P4: 見張りを OFF→ON にするときの確認の文の間に起き直す(見張りは pre で ON なので、先に OFF にしてから)
res, toasts = C.sim(c, setup + [['check', '#dr-rec-hold-keep', False], ['check', '#dr-rec-back-guard', False], ['poll'],
                               ['reboot_on_confirm', 2000.0, RESET], ['check', '#dr-rec-back-guard', True],
                               ['read', 'st_params'], ['start'], ['posts'], ['start'], ['posts']], server=pre, server_boot_check=True)
st = C.starts(res)
out['P4'] = {'st_params_guard_after_switch': [x['value']['rec_back_guard'] for x in res if x['op'] == 'read'],
             'starts': [{'status': s['status'], 'boot_at': s['boot_at'],
                         'confirm_lines': [l for l in s['confirm'].split('\n') if any(w in l for w in ('見張り', '天井', '発進の前進:'))]} for s in st],
             'toasts': toasts[-4:]}
print(json.dumps(out, ensure_ascii=False, indent=1))
C.cleanup()
