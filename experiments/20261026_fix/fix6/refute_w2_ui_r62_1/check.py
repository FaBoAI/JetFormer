"""反証 w2-ui-w1-ui-r62-1 を自分で確かめる(直さない)。本物の app.js(読むだけ)を node の vm で。

ctrl: 走行タブに戻るときに models が通る → curl の 4.2 を 1 秒の問い合わせで読む(固まらない)
B   : 戻るときに models だけ通らない → 輪が止まる・curl の 4.2 を読まない・▶ の確認の文 4.1・本物の Driver は 4.2 で走る
B2  : B のあと ▶ をキャンセル → 輪は始まらない。ライブ → 走行タブ(models が通る)で直る
A   : 戻るときにサーバーが落ちていて、その後起き直す → ▶ は 409(boot_at)→ 同期 → 2 回目の ▶ は新しい値の文
patched: loadDrive の catch に drLost(); pollDrive(); を足した一時の写し(本物は変えない)で B → 固まらない
"""
import json
import common as C

c = C.bench_today()
di = C.day_index(c)
b10 = C.blk(c, di, 10)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
out = {}


def look():
    return [['tick'], ['tick'], ['ntimers'], ['eval', 'String(driveTimer)'], ['eval', 'state.drive.paramsLoaded'],
            ['read', 'st_params'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-ceil-spec', 'text'], ['el', '#dr-message', 'html']]


def summary(res, toasts):
    ev = [x.get('value') for x in res if x['op'] == 'eval']
    els = [x['value'] for x in res if x['op'] == 'el']
    chk = [x for x in res if x['op'] == 'check']
    return {
        'timers': [x['value'] for x in res if x['op'] == 'ntimers'],
        'driveTimer': ev[0] if ev else None, 'paramsLoaded': ev[1] if len(ev) > 1 else None,
        'st_left': [v.get('ceil_left_mps') for v in C.reads(res, 'st_params')],
        'start_disabled': els[0] if els else None, 'ceil_spec_text': els[1] if len(els) > 1 else None,
        'lost_text': ('つながっていません' in (els[2] or '')) if len(els) > 2 else None,
        'ceil_switch_confirm': [l for x in chk for l in x['confirm'].split('\n') if '左旋回中' in l][:1],
        'starts': [{'status': s['status'], 'boot_at': s['boot_at'],
                    'confirm_ceil': [l for l in s['confirm'].split('\n') if '天井' in l or '前進:' in l]} for s in C.starts(res)],
        'server': [C.pick(v, ('ceil_left_mps', 'speed_ceiling', 'launch_us')) for v in C.reads(res, 'server')],
        'toasts_tail': toasts[-3:],
    }


base = [['realpoll'], ['eval', "showTab('live')"], ['tick'], ['tick']]
pre = C.server_params(rec_back_guard=1, ceil_left_mps=4.1)

# ctrl
steps = base + [['eval', "showTab('drive')"], ['api', {'ceil_left_mps': 4.2}]] + look() + [
    ['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server']]
res, t = C.sim(c, steps, server=dict(pre), server_boot_check=True, server_client_version=9)
out['ctrl_models_ok'] = summary(res, t)

# B
steps = base + [['models_fail', True], ['eval', "showTab('drive')"], ['models_fail', False], ['api', {'ceil_left_mps': 4.2}]] + look() + [
    ['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server'], ['posts']]
res, t = C.sim(c, steps, server=dict(pre), server_boot_check=True, server_client_version=9)
o = summary(res, t)
d = C.driver(params=dict(pre))
o['real_driver'] = {'replay': C.replay(d, C.posts(res)[-1]), 'ceil_left_mps': d.params['ceil_left_mps'], 'speed_ceiling': d.params['speed_ceiling']}
out['B_models_fail'] = o

# B2: cancel → still frozen; then live → drive (models OK) → recovers
steps = base + [['models_fail', True], ['eval', "showTab('drive')"], ['models_fail', False], ['api', {'ceil_left_mps': 4.2}],
                ['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['confirm', False], ['start'], ['tick'], ['tick'],
                ['ntimers'], ['read', 'st_params'],
                ['eval', "showTab('live')"], ['eval', "showTab('drive')"], ['tick'], ['tick'], ['ntimers'], ['read', 'st_params'],
                ['el', '#dr-ceil-spec', 'text'], ['confirm', True], ['start']]
res, t = C.sim(c, steps, server=dict(pre), server_boot_check=True, server_client_version=9)
out['B2_cancel_then_retab'] = {
    'timers': [x['value'] for x in res if x['op'] == 'ntimers'],
    'st_left': [v.get('ceil_left_mps') for v in C.reads(res, 'st_params')],
    'ceil_spec_after_retab': [x['value'] for x in res if x['op'] == 'el'],
    'starts': [{'status': s['status'], 'confirm_ceil': [l for l in s['confirm'].split('\n') if '天井' in l]} for s in C.starts(res)]}

# A: server down at tab click, then reboots
preA = C.server_params(rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
steps = base + [['down_all', True], ['eval', "showTab('drive')"], ['tick'], ['reboot', 2000.0, RESET], ['down_all', False]] + look() + [
    ['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['start'], ['start'], ['read', 'server']]
res, t = C.sim(c, steps, server=dict(preA), server_boot_check=True, server_client_version=9)
out['A_reboot'] = summary(res, t)

# patched copy (only in TMP)
src = (C.STATIC / 'app.js').read_text(encoding='utf-8')
old = "try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) { toast(e.message, true); return; }"
assert src.count(old) == 1
pa = C.TMP / 'app_patched.js'
pa.write_text(src.replace(old, "try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) { toast(e.message, true); drLost(); pollDrive(); return; }"), encoding='utf-8')
steps = base + [['models_fail', True], ['eval', "showTab('drive')"], ['models_fail', False], ['api', {'ceil_left_mps': 4.2}]] + look() + [
    ['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server']]
res, t = C.sim(c, steps, server=dict(pre), server_boot_check=True, server_client_version=9, app=str(pa))
out['patched_B'] = summary(res, t)

print(json.dumps(out, ensure_ascii=False, indent=1))
C.cleanup()
