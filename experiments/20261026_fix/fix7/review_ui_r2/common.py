"""反証 7-2 回目・ui の台本の共通(読むだけ)。本物の app.js・index.html を、この場所の sim_r2.js(tests/fixtures/app_sim_fix7.js の
写しに「返事だけ届かない」などを足したもの)で node に読み込む。409 の文は本物の server._boot_refusal(import だけ。起動しない)。
車・jetformer-web には触れない。本物のファイルは変えない。"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402
from jetformer.web import server as srv  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC = ROOT / 'jetformer' / 'web' / 'static'
APP = Path(os.environ.get('JF_R72_APP') or STATIC / 'app.js')
SIM = Path(os.environ.get('JF_R72_SIM') or HERE / 'sim_r2.js')
FIXT = ROOT / 'tests' / 'fixtures'
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
REAL = {'1000': srv._boot_refusal({'boot_at': 1000.0}, 2000.0), 'null': srv._boot_refusal({}, 2000.0)}
for _a in range(1000, 6001, 1000):
    REAL['null->%d' % _a] = srv._boot_refusal({}, float(_a))
    for _b in range(1000, 6001, 1000):
        if _a != _b:
            REAL['%d->%d' % (_a, _b)] = srv._boot_refusal({'boot_at': float(_a)}, float(_b))


def cands():
    c = bench.load_candidates()
    di = next(i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03' and d['blocks'])
    return dict(c, today='2026-11-03'), di


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def server_params(**over):
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(over)
    return sp


MORNING = server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)


def run_sim(tmp, c, steps, server=None, app=None, sim=None, name='in.json', **extra):
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    d = {'bench': c, 'server_params': server if server is not None else MORNING, 'steps': steps, 'server_client_version': 9,
         'confirm': True, 'boot_at': 1000.0, 'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True, 'boot_refusal_texts': REAL}
    d.update(extra)
    p = tmp / name
    p.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run(['node', str(sim or SIM), str(app or APP), str(STATIC / 'index.html'), str(p)], capture_output=True, text=True,
                       timeout=120, env=dict(os.environ, TZ='Asia/Tokyo'))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def ops(res, op):
    return [x for x in res if x['op'] == op]


def vals(res, op):
    return [x.get('value') for x in res if x['op'] == op]


def note_of(html):
    parts = (html or '').split('<div class="warn-text">')
    last = parts[-1].replace('</div>', '') if len(parts) > 1 else ''
    return last if last.startswith('管理画面のサーバーが ') else None


def confirm_vs_server(conf, srvp):
    """走行開始が通ったとき: ▶ の確認の文の行と、サーバーが実際に走る値の違い(無ければ [])"""
    bad = []
    lines = conf.split('\n')
    for word, key in (('後ろの見張り: ON', 'rec_back_guard'), ('残す: ON', 'rec_hold_keep'), ('補助も出さない: ON', 'launch_no_kick')):
        if any(word in x for x in lines) != (srvp[key] == 1):
            bad.append('%s 文%s 実%s' % (key, any(word in x for x in lines), srvp[key]))
    m = re.search(r'速さの天井: ON\(全体 ([\d.]+)・左旋回中 ([\d.]+)', conf)
    if bool(m) != (srvp['speed_ceiling'] == 1) or (m and abs(float(m.group(2)) - srvp['ceil_left_mps']) > 1e-9):
        bad.append('天井 文%s 実%s/%s' % (m.group(2) if m else None, srvp['speed_ceiling'], srvp['ceil_left_mps']))
    if any('自動合わせ: ON' in x for x in lines) != (srvp['speed_auto'] == 1):
        bad.append('自動合わせ 文%s 実%s' % (any('自動合わせ: ON' in x for x in lines), srvp['speed_auto']))
    m = re.search(r'発進の前進: ON\(前進 (\d+) µs', conf)
    if bool(m) != (srvp['launch_boost'] == 1) or (m and abs(float(m.group(1)) - srvp['launch_us']) > 1e-9):
        bad.append('発進 文%s 実%s/%s' % (m.group(1) if m else None, srvp['launch_boost'], srvp['launch_us']))
    m = re.search(r'速度ゲイン ([\d.]+)', conf)
    if m and abs(float(m.group(1)) - srvp['throttle_gain']) > 1e-9:
        bad.append('ゲイン 文%s 実%s' % (m.group(1), srvp['throttle_gain']))
    return bad
