#!/usr/bin/env python3
"""反証 2 回目(state、fix3): 本物の app.js を node の vm(sim2.js = tests/fixtures/app_sim_fix3_r1_2.js の写し)で流す共通の台本。
サーバーは偽物(再起動のあとの drive.DEFAULT_PARAMS + 基準の値)。車・jetformer-web には触れない。"""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import drive          # noqa: E402
from jetformer.web import bench      # noqa: E402

HERE = Path(__file__).resolve().parent
SIM = HERE / 'sim2.js'
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


def run(steps, server, app=APP, extra=None):
    c, _ = cands()
    with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
        inp = Path(td) / 'in.json'
        d = {'bench': c, 'server_params': server, 'steps': steps, 'server_client_version': 7, 'confirm': True, 'boot_at': 1000.0}
        d.update(extra or {})
        inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
        r = subprocess.run(['node', str(SIM), str(app), str(HTML), str(inp)], capture_output=True, text=True, timeout=120)
    if r.returncode:
        print(r.stderr[-3000:]); raise SystemExit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])
