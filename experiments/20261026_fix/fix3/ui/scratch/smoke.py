import json, subprocess, sys, shutil
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import drive
from jetformer.web import bench
c = bench.load_candidates()
di = [i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03'][0]
c = dict(c, today='2026-11-03')
sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1, speed_ceiling=1, speed_auto=1)
steps = [['pick', di, 10], ['read', 'pending'], ['check', '#dr-speed-ceiling', True], ['posts'], ['poll'], ['poll'], ['read', 'auto'], ['start'],
         ['el', '#dr-opt-line-rec_back_guard', 'hidden'], ['click', '#btn-dr-tg-up'], ['posts'], ['read', 'gain_text']]
inp = '/home/jetson/jetformer/experiments/20261026_fix/fix3/ui/scratch/in.json'
json.dump({'bench': c, 'server_params': sp, 'steps': steps}, open(inp, 'w'), ensure_ascii=False, default=str)
for app, html in [('jetformer/web/static/app.js', 'jetformer/web/static/index.html'), ('tests/fixtures/ui_fix3_before/app.js', 'tests/fixtures/ui_fix3_before/index.html')]:
    r = subprocess.run([shutil.which('node'), 'tests/fixtures/app_sim_fix3.js', app, html, inp], capture_output=True, text=True, timeout=60)
    print(app, r.returncode, r.stderr[-2000:])
    if r.returncode == 0:
        out = json.loads(r.stdout.strip().splitlines()[-1])
        for x in out['steps']:
            if x['op'] == 'start':
                print('  start', {k: x['sent'].get(k) for k in ('speed_ceiling', 'speed_auto', 'throttle_gain')}, x['client'], x['refused'], [l for l in x['confirm'].splitlines() if '天井' in l or '自動' in l])
            else:
                print(' ', x)
        print('  toasts', out['toasts'])
