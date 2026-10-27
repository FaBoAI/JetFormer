"""S1: jetformer-web が止まっている車の間に起こし直し、画面が最初の同期をする前に ▶ を押す(手順書 7 章「起こし直した」)。

朝: 見張り・残す・補助なしを ON(スイッチ)・左 4.2(API)・launch_us 200(API。回 3 の空転)。回 3 の ON の本(ブロック 15 = 3-4)を
選び、ゲイン 0.92・天井 ON・発進の前進 ON。そこでサーバーが起こし直す(RESET_ON_BOOT)。画面は問い合わせがまだ通っていない。
→ ▶ の確認の文・送る値・本物の drive.Driver(起こし直したあと = 新しく作った Driver)の走る値を比べる。
比べ: 同じ操作で、起こし直したあと 1 回同期してから ▶。
"""
import json
import common as C

c = C.bench_today()
di = C.day_index(c)
b15 = C.blk(c, di, 15)
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, launch_no_kick=1, ceil_left_mps=4.2, launch_us=200.0)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
setup = [['pick', di, b15], ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'],
         ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True], ['poll'],
         ['confirm', False], ['start'], ['confirm', True], ['posts']]
out = {}
for name, tail in (('no_sync', [['reboot', 2000.0, RESET], ['start'], ['posts'], ['read', 'toasts']]),
                   ('dialog_spans_restart', [['down', True], ['realpoll'], ['realpoll'], ['el', '#btn-dr-start', 'disabled'],
                                             ['reboot_on_confirm', 2000.0, RESET], ['start'], ['posts'], ['read', 'toasts']]),
                   ('synced', [['reboot', 2000.0, RESET], ['poll'], ['start'], ['posts'], ['read', 'toasts']])):
    res, toasts = C.sim(c, setup + tail, server=pre)
    st = C.starts(res)
    ps = C.posts(res)
    # 本物の Driver: 起こし直す前の Driver に朝の POST を入れ、設定ファイルを書かせる → 新しい Driver(起こし直し)に ▶ の POST
    d1 = C.driver(params=pre, settings={'params': pre, 'run': C.BRUN, 'mode_id': 5})
    d1._save_settings()
    r1 = C.replay(d1, ps[0])
    d2 = C.driver()                                     # 起こし直したサーバー(同じ設定ファイルから。RESET_ON_BOOT)
    r2 = C.replay(d2, ps[1])
    out[name] = {
        'confirm_before_reboot(cancel)': st[0]['confirm'],
        'starts_after_reboot': [{'confirm': s['confirm'], 'sent': C.pick(s['sent'] or {}), 'refused': s['refused']} for s in st[1:]],
        'real_driver_after_start': C.pick(d2.params), 'real_driver_status': r2,
        'toasts_tail': toasts[-4:],
        'start_button_disabled_while_down': [x['value'] for x in res if x['op'] == 'el'],
    }
print(json.dumps(out, ensure_ascii=False, indent=1))
(C.HERE / 'out').mkdir(exist_ok=True)
(C.HERE / 'out' / 's1_reboot_stale.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
C.cleanup()
