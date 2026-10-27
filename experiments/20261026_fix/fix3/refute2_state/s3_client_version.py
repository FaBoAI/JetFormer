#!/usr/bin/env python3
"""DRIVE_CLIENT_VERSION: fix3 係 ui(20:25)で 7 にしたあと、直す 1 回目(r1-state3-1・3-2・rs1-2・state3-4)で app.js を 2 回
変えたが版は 7 のまま。20:25〜22:4x に開いた画面(tests/fixtures/fix3_r1_before/app.js)も、再起動のあとのサーバーは断らない。
その画面で ±0.01 を 2 回押し、1 回目の返事 → 同期 → 2 回目のタイマー の順(r1-state3-1 の指摘の場面)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_sim import cands, blk, run, server_params, ROOT   # noqa: E402
c, di = cands()
UP = '#btn-dr-tg-up'
steps = [['pick', di, blk(c, di, 1)], ['tap', UP], ['timers'], ['tap', UP], ['resolve1'], ['poll'], ['timers'], ['resolve'], ['poll'], ['start']]
for name, app in (('20:25 の画面(fix3_r1_before)', ROOT / 'tests/fixtures/fix3_r1_before/app.js'), ('今の画面', ROOT / 'jetformer/web/static/app.js')):
    res = run(steps, server_params(), app=app, extra={'defer_posts': True})
    st = [x for x in res['steps'] if x['op'] == 'start'][0]
    print('%s: client=%s 断られた=%s 送った速度ゲイン=%s 確認の文=%s' % (name, st['client'], st['refused'], st['sent'] and st['sent']['throttle_gain'],
          [l for l in st['confirm'].splitlines() if '速度ゲイン' in l]))
