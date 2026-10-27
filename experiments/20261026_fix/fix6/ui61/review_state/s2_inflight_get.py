"""S2: スイッチ(postDriveParams)の送信の返事より先に、送る前に出ていた表示の問い合わせ(古い値)の返事が来る。

2-4(天井 + 自動合わせ ON)のあと、2-5/2-6(天井だけ)のブロックを選ぶ → 天井のスイッチ ON(確認の文に答える)。
クリックしたとき表示の問い合わせが 1 本飛んでいた(1 秒ごと。返事は確認の文を閉じたあとに処理される)。
→ 自動合わせのスイッチと、すぐ ▶ したときに送る値。比べ: 問い合わせが飛んでいない(ふつう)。スライダーの ±0.01 は同じ場面で守られる。
"""
import json
import common as C

c = C.bench_today()
di = C.day_index(c)
b11 = C.blk(c, di, 11)
pre = C.server_params(throttle_gain=0.95, speed_ceiling=1, speed_auto=1)
base = [['pick', di, b11], ['read', 'pending']]
scen = {
    'get_in_flight_at_click': base + [['gstart'], ['defer', True], ['checkn', '#dr-speed-ceiling', True], ['gend'],
                                      ['read', 'auto'], ['resolve'], ['read', 'auto'], ['read', 'server'], ['defer', False],
                                      ['start'], ['posts']],
    'get_processed_before_post': base + [['defer', True], ['checkn', '#dr-speed-ceiling', True], ['gstart'], ['gend'],
                                         ['read', 'auto'], ['resolve'], ['read', 'auto'], ['read', 'server'], ['defer', False],
                                         ['start'], ['posts']],
    'normal': base + [['check', '#dr-speed-ceiling', True], ['read', 'auto'], ['read', 'auto'], ['read', 'server'], ['start'], ['posts']],
    'self_corrects_after_next_poll': base + [['gstart'], ['defer', True], ['checkn', '#dr-speed-ceiling', True], ['gend'],
                                             ['resolve'], ['defer', False], ['poll'], ['read', 'auto'], ['start'], ['posts']],
}
out = {}
for name, steps in scen.items():
    aor = name == 'get_processed_before_post'
    res, toasts = C.sim(c, steps, server=pre, apply_on_resolve=aor)
    st = C.starts(res)
    d = C.driver(params=pre)
    r = C.replay(d, [p for ps in C.posts(res) for p in ps])
    out[name] = {'pending_after_pick': C.reads(res, 'pending')[0], 'auto_checkbox': C.reads(res, 'auto'),
                 'server_before_start': C.pick(C.reads(res, 'server')[0]) if C.reads(res, 'server') else None,
                 'sent': C.pick(st[0]['sent']), 'auto_line_in_confirm': [l for l in st[0]['confirm'].split('\n') if '自動合わせ' in l],
                 'real_driver_after_start': C.pick(d.params), 'real': r}
print(json.dumps(out, ensure_ascii=False, indent=1))
(C.HERE / 'out' / 's2_inflight_get.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
C.cleanup()
