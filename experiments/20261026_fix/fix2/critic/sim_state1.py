"""critic(2026-10-26 夜): 反証 1 回目の state-1(天井を ON にすると、ブロックで OFF にした自動合わせが黙って ON に戻る)が
今の app.js でも起きるか。本物の app.js を critic/app_sim_critic.js(tests/fixtures/app_sim_round2.js の写しに speed_ctrl_spec を足したもの。node の vm。DOM・サーバーは偽物)で流す。
読むだけ。本物のサーバー・車には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix2/critic/sim_state1.py <作業場所> [<app.js。既定は本物の jetformer/web/static/app.js>]
    (patch を当てた写しを渡すと、直したあとの動きを見られる。結果: 直す前は自動合わせ ON を送る・直したあとは OFF)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
c = bench.load_candidates()
day = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03'][0]
c = dict(c, today='2026-11-03')
blocks = {b['block']: b for b in c['days'][day]['blocks']}
print('ブロック 10:', blocks[10].get('run_id'), ' 11:', blocks[11].get('run_id'))
sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
sp.update(speed_ceiling=1, speed_auto=1)            # 前の本(2-4)が天井 + 自動合わせ ON のまま
steps = [['poll'], ['pick', day, 11], ['read', 'pending'], ['read', 'ceil'],
         ['read', 'auto'], ['ceil_on'], ['read', 'ceil'], ['read', 'auto'], ['read', 'pending'], ['poll'], ['read', 'ceil'], ['read', 'auto'],
         ['poll'], ['read', 'auto'], ['read', 'server'], ['start']]
inp = work / 'input.json'
inp.write_text(json.dumps({'bench': c, 'server_params': sp, 'steps': steps}, ensure_ascii=False, default=str), encoding='utf-8')
r = subprocess.run([shutil.which('node'), str(Path(__file__).resolve().parent / 'app_sim_critic.js'), str(Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'jetformer/web/static/app.js'),
                    str(inp)], capture_output=True, text=True, timeout=60)
if r.returncode:
    print(r.stderr[-3000:])
    sys.exit(1)
res = json.loads(r.stdout.strip().splitlines()[-1])['steps']
for x in res:
    if x['op'] == 'start':
        s = x['sent'] or {}
        print('start: 送った speed_ceiling=%s speed_auto=%s' % (s.get('speed_ceiling'), s.get('speed_auto')))
        print('確認の文(天井・自動の行):', [ln for ln in x['confirm'].splitlines() if '天井' in ln or '自動' in ln])
    elif x['op'] == 'read':
        v = x['value']
        if x['what'] == 'server':
            v = {k: v.get(k) for k in ('speed_ceiling', 'speed_auto')}
        print('read', x['what'], v)
    else:
        print(x['op'], {k: v for k, v in x.items() if k != 'op'})
