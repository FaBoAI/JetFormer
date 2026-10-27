#!/usr/bin/env python3
"""反証 1 回目(state、fix3): 手順書 11/3 の流れを本物の app.js(node の vm・tests/fixtures/app_sim_fix3.js)で流す。
サーバーは偽物(再起動のあとの drive.DEFAULT_PARAMS)。車・jetformer-web には触れない。
  python3 flow_1103.py <作業場所> [steps.json]
"""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import drive
from jetformer.web import bench

ROOT = Path('/home/jetson/jetformer')
SIM = Path(__file__).with_name('sim.js') if Path(__file__).with_name('sim.js').exists() else ROOT / 'tests/fixtures/app_sim_fix3.js'
APP = ROOT / 'jetformer/web/static/app.js'
HTML = ROOT / 'jetformer/web/static/index.html'


def cands():
    c = bench.load_candidates()
    di = [i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03'][0]
    return dict(c, today='2026-11-03'), di


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def server_params(**over):
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(over)
    return sp


def run(work, steps, server, sim=SIM, app=APP, html=HTML, extra=None):
    c, di = cands()
    inp = Path(work) / 'in.json'
    d = {'bench': c, 'server_params': server, 'steps': steps, 'server_client_version': 7, 'confirm': True}
    d.update(extra or {})
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run(['node', str(sim), str(app), str(html), str(inp)], capture_output=True, text=True, timeout=120)
    if r.returncode:
        print(r.stderr[-3000:]); raise SystemExit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])
