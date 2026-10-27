"""試験の台本 tests/fixtures/app_sim_fix6_r2.js(と前の app_sim_*.js)の 'click'・'slide' は、短いタイマー(120 ms の送信)を鳴らすときに
長いタイマー(表示の問い合わせの輪 1000/250 ms・打ち切り 5000 ms・トースト)を全部捨てる。本物のブラウザでは輪は続く。
同じ操作を、もとの台本と、その 1 行だけ直した写し sim7.js で流して比べる(読むだけ)。  python3 p_harness_click.py <一時の場所>"""
import json
import sys
from pathlib import Path

import common
from common import cands, run_sim, server_params

tmp = sys.argv[1]
c, di = cands()
steps = [['tab', 'drive'], ['tick'], ['ntimers'], ['eval', 'driveTimer !== null'],
         ['click', '#btn-dr-tg-up'], ['ntimers'], ['eval', 'driveTimer !== null'],
         ['api', {'ceil_left_mps': 4.2}], ['tick'], ['tick'], ['ntimers'], ['read', 'st_params']]
for label, sim in (('もとの app_sim_fix6_r2.js', common.ROOT / 'tests' / 'fixtures' / 'app_sim_fix6_r2.js'), ('sim7.js(長いタイマーを捨てない)', Path(common.__file__).parent / 'sim7.js')):
    common.SIM = sim
    res, toasts = run_sim(tmp, c, steps, server_params(), name='hc.json')
    nt = [x['value'] for x in res if x['op'] == 'ntimers']
    ev = [x['value'] for x in res if x['op'] == 'eval']
    left = [x['value'] for x in res if x['op'] == 'read'][-1]['ceil_left_mps']
    print(json.dumps({'台本': label, 'click の前のタイマー': nt[0], 'click の後のタイマー': nt[1], '輪のタイマー(前・後)': ev,
                      'curl 4.2 → 2 回 tick のあとのタイマー': nt[2], '画面が読んだ左': left}, ensure_ascii=False))
