"""P3: 07:41 の古いサーバー(status に boot_at・発進の前進・試験の選択肢が無い。版の照らしは 6)+ 今の app.js(版 9)"""
import json
import common6 as C

c = C.bench_today()
di = C.day_index(c)
old = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'steer_trim_us': 0.0, 'lookahead': 3, 'osc_percent': 0.0, 'esc_reverse': 1, 'recovery': 1,
       'rev_us': 300.0, 'rev_s': 0.4, 'tof_guard': 1, 'speed_cap': 0.0, 'speed_scale': 1.0, 'tcp_mode': 0, 'ensemble_j': 0, 'ensemble_m': 0.5,
       'zone_caps': 0, 'start_kick': 1, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'auto_vref_v': 8.35}
steps = [['baseline'], ['poll'], ['pick', di, C.blk(c, di, 4)], ['poll'], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True],
         ['poll'], ['start'], ['running', False],
         ['down', True], ['realpoll'], ['el', '#btn-dr-start', 'disabled'], ['baseline'], ['start'], ['down', False],
         ['realpoll'], ['el', '#btn-dr-start', 'disabled'], ['pick', di, C.blk(c, di, 6)], ['poll'], ['check', '#dr-speed-ceiling', True], ['poll'], ['start'],
         ['el', '#dr-client-ver', 'text'], ['posts']]
res, toasts = C.sim(c, steps, server=old, boot_at=None, server_client_version=6, server_boot_check=True)
st = C.starts(res)
print(json.dumps({'starts': [{'status': s['status'], 'boot_at': s['boot_at'], 'client': s['client'], 'sent_ceil': (s['sent'] or {}).get('speed_ceiling'),
                              'gain': (s['sent'] or {}).get('throttle_gain'), 'confirm_shown': bool(s['confirm'])} for s in st],
                  'el': [x['value'] for x in res if x['op'] == 'el'], 'toasts_tail': toasts[-6:]}, ensure_ascii=False, indent=1))
C.cleanup()
