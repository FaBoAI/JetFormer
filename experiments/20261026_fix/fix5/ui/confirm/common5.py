"""fix5 画面の係の確かめ: review_state/common.py の写し(SIM だけ tests/fixtures/app_sim_fix5.js に)。"""
"""fix4 反証 4 回目(見方: 状態と設定の受け渡し)の共通の台本。直さない・装置を開かない・jetformer-web に触れない。

- bench(): 本物の data/bench_candidates.json を本物の web/bench.load_candidates で読み、today を 2026-11-03 にしたもの(読むだけ)
- sim(steps, server=..., **extra): 本物の app.js・index.html を node の vm で(sim4.js = tests/fixtures/app_sim_fix3_r3_1.js の写し + down)
- driver(params, settings): 本物の drive.Driver(hub=None・recorder=None・設定ファイルは一時の場所・走行ループは動かさない)
- replay(d, res): sim の POST を本物の Driver に同じ順で入れる
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
STATIC = JF / 'jetformer' / 'web' / 'static'
SIM = JF / 'tests' / 'fixtures' / 'app_sim_fix5.js'      # fix5: sim4.js + 偽のサーバーの起動の照らし(server_boot_check)
NODE = shutil.which('node')
TMP = Path(tempfile.mkdtemp(prefix='fix5_ui_', dir=str(HERE)))

from jetformer import drive          # noqa: E402
from jetformer.web import bench      # noqa: E402

BRUN = bench.B_RUN


def bench_today(today='2026-11-03'):
    c = bench.load_candidates()
    return dict(c, today=today)


def day_index(c, date='2026-11-03'):
    return next(i for i, d in enumerate(c['days']) if d['date'] == date)


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def server_params(**over):
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(over)
    return sp


_n = [0]


def sim(c, steps, server=None, app=None, **extra):
    _n[0] += 1
    inp = TMP / ('in_%d.json' % _n[0])
    models = sorted({cd['run'] for cd in c['candidates'] if cd.get('run')})
    d = {'bench': c, 'server_params': server if server is not None else server_params(), 'steps': steps,
         'server_client_version': 8, 'confirm': True, 'boot_at': 1000.0, 'models': models,
         'script_src': 'http://x/static/app.js?v=1793667600'}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app or STATIC / 'app.js'), str(STATIC / 'index.html'), str(inp)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def starts(res):
    return [x for x in res if x['op'] == 'start']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


def posts(res):
    return [x['value'] for x in res if x['op'] == 'posts']


class _Rec:
    recording = False

    def add_event(self, *a, **k):
        return True


def driver(params=None, settings=None):
    """本物の drive.Driver。settings = 一時の drive_settings.json の中身(None なら書かない)。params があれば上書き"""
    sp = TMP / 'drive_settings.json'
    drive.SETTINGS_PATH = sp
    drive.model_info = lambda name: {'trained_modes': [5], 'drivable': True, 'postprocess': None, 'name': name}
    if settings is not None:
        sp.write_text(json.dumps(settings, ensure_ascii=False), encoding='utf-8')
    d = drive.Driver(hub=None, recorder=_Rec())
    d._run = lambda *a, **k: None                       # 走行ループは動かさない(装置に触れない)
    if params is not None:
        d.params = dict(params)
    return d


def replay(d, plist):
    """sim の ['posts'] の列(path, body, status)を本物の Driver に入れる。戻り値 [(path, 本物の status, sim の status, 本物の詳細)]"""
    out = []
    for path, body, status in plist:
        if path == '/api/drive/params':
            try:
                d.set_params(body)
                got, why = 200, ''
            except ValueError as e:
                got, why = 400, str(e)
        elif path == '/api/drive/start':
            try:
                d.start(body['run'], body['mode_id'], params=body['params'], description=body.get('description', ''),
                        dry_run=body.get('dry_run', False))
                if d.thread is not None:
                    d.thread.join(1)
                got, why = 200, ''
            except (ValueError, RuntimeError) as e:
                got, why = 400, str(e)
        else:
            continue
        out.append((path, got, status, why))
    return out


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)


KEYS = ('throttle_gain', 'speed_ceiling', 'ceil_left_mps', 'speed_auto', 'launch_boost', 'launch_us', 'rec_back_guard',
        'rec_hold_keep', 'launch_no_kick', 'start_kick', 'zone_caps', 'steer_trim_us')


def pick(p, keys=KEYS):
    return {k: p.get(k) for k in keys}
