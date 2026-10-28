"""反証 8-2 回目・係 ui の台本の共通部分(8-1 の係の run.py の写しに少し足した)(読むだけ。本物の app.js・index.html を node の vm で動かす。サーバー・車には触れない)。

- 台本 sim.js = tests/fixtures/app_sim_fix7_r2.js の写し(変えていない)。sim2.js = その写しに「2 つの画面」を足したもの(この係)。
- ベンチマークの候補・モデルの一覧・サーバーの params は、本物の jetformer-web の GET の答え(bench_live.json・models_live.json・
  status_live.json。10/28 に GET だけで取った)。今日は 2026-11-03 にする。
- 起き直しの 409 の文は server.py の _boot_refusal を ast で取り出して(import しない = サーバーを作らない)作る。
"""
import ast
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
APP = JF / 'jetformer' / 'web' / 'static' / 'app.js'
HTML = JF / 'jetformer' / 'web' / 'static' / 'index.html'
SIM = HERE / 'sim_r2.js'

BENCH = json.loads((HERE / 'bench_live.json').read_text(encoding='utf-8'))
BENCH['today'] = '2026-11-03'
MODELS = [m['name'] for m in json.loads((HERE / 'models_live.json').read_text(encoding='utf-8'))]
LIVE = json.loads((HERE / 'status_live.json').read_text(encoding='utf-8'))
DEFAULTS = dict(LIVE['params'])                  # 本物のサーバーの今の値(起動のときの既定)
RESET_KEYS = ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'ceil_left_mps', 'launch_us', 'speed_ceiling', 'ceil_all_mps',
              'speed_auto', 'auto_target_mps', 'launch_boost', 'launch_mps', 'launch_max_s', 'zone_caps', 'tcp_mode', 'speed_scale')
RESET = {k: DEFAULTS[k] for k in RESET_KEYS}
DI = next(i for i, d in enumerate(BENCH['days']) if d['date'] == '2026-11-03')


def blk(no):
    return next(i for i, b in enumerate(BENCH['days'][DI]['blocks']) if int(b['block']) == no)


def _boot_refusal_fn():
    src = (JF / 'jetformer' / 'web' / 'server.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_boot_refusal')
    ns = {'time': time}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'server.py', 'exec'), ns)
    return ns['_boot_refusal']


os.environ['TZ'] = 'Asia/Tokyo'
time.tzset()
BR = _boot_refusal_fn()


def real_texts(pairs):
    """{鍵: 本物の 409 の文}。pairs = [(画面の boot_at か None, 今の boot_at)]"""
    out = {}
    for seen, now in pairs:
        t = BR({} if seen is None else {'boot_at': seen}, now)
        out['%s->%s' % (_k(seen), _k(now))] = t
        out.setdefault(_k(seen), t)
    return out


def _k(x):
    if x is None:
        return 'null'
    s = str(float(x))
    return s[:-2] if s.endswith('.0') else s


def run(steps, server=None, app=APP, sim=SIM, name='in.json', **extra):
    d = {'bench': BENCH, 'server_params': server if server is not None else dict(DEFAULTS), 'steps': steps,
         'server_client_version': 9, 'confirm': True, 'boot_at': 1000.0, 'models': MODELS,
         'script_src': 'http://x/static/app.js?v=%d' % int(APP.stat().st_mtime), 'server_boot_check': True,
         'boot_refusal_texts': real_texts([(1000.0, 2000.0), (None, 2000.0), (2000.0, 3000.0), (1000.0, 3000.0)])}
    d.update(extra)
    p = HERE / 'tmp' / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')
    r = subprocess.run(['node', str(sim), str(app), str(HTML), str(p)], capture_output=True, text=True, timeout=180,
                       env=dict(os.environ, TZ='Asia/Tokyo'))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def steps_of(out, op):
    return [s for s in out['steps'] if s.get('op') == op]


CONF_HEADS = ('リカバリーの後ろの見張り', '中立止めのあとも', '速さの天井', '発進の前進', 'モデル', 'ジャンル')


def confirm_lines(c):
    return [x for x in (c or '').split('\n') if x.startswith(CONF_HEADS)]
