"""反証 r1-state3-4 の確かめ: jetformer-web が起こし直されると、朝に入れた試験の選択肢が黙って OFF になるか。
A: 本物の drive.Driver(装置なし・設定ファイルは tmp)で set_params → 保存 → 新しい Driver(= 起動)
B: 本物の app.js(tests/fixtures/app_sim_fix3.js)で、サーバーの値が 1 → 0 に変わったときの画面・確認の文・トースト
"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import drive
from jetformer.web import bench

tmp = Path(tempfile.mkdtemp(prefix='r1s34_', dir=str(Path(__file__).parent)))
try:
    # ---- A
    drive.SETTINGS_PATH = tmp / 'drive_settings.json'
    d = drive.Driver(hub=None, recorder=None)
    d.set_params({'rec_back_guard': 1})
    d.set_params({'rec_hold_keep': 1, 'launch_no_kick': 1, 'ceil_left_mps': 4.2, 'launch_us': 200})
    keys = ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'ceil_left_mps', 'launch_us', 'steer_trim_us', 'throttle_gain')
    d.set_params({'steer_trim_us': 15, 'throttle_gain': 0.95})
    before = {k: d.params[k] for k in keys}
    saved = json.loads(drive.SETTINGS_PATH.read_text()) if drive.SETTINGS_PATH.exists() else None
    d2 = drive.Driver(hub=None, recorder=None)          # 起こし直し
    after = {k: d2.params[k] for k in keys}
    print('A before ', before)
    print('A saved  ', {k: (saved or {}).get('params', {}).get(k) for k in keys})
    print('A reboot ', after)

    # ---- B
    c = bench.load_candidates()
    di = [i for i, x in enumerate(c['days']) if x.get('date') == '2026-11-03'][0]
    c = dict(c, today='2026-11-03')
    blk = lambda no: next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1,
              rec_back_guard=1, rec_hold_keep=1, launch_no_kick=1)
    steps = [['poll'], ['pick', di, blk(20)], ['el', '#dr-rec-back-guard', 'checked'], ['el', '#dr-rec-hold-keep', 'checked'],
             ['read', 'note'], ['start'], ['running', False],
             # 起こし直し: RESET_ON_BOOT で 3 つが 0 に
             ['server_set', {'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}], ['poll'], ['poll'],
             ['el', '#dr-rec-back-guard', 'checked'], ['el', '#dr-rec-hold-keep', 'checked'],
             ['pick', di, blk(21)], ['read', 'note'], ['start']]
    inp = tmp / 'in.json'
    inp.write_text(json.dumps({'bench': c, 'server_params': sp, 'steps': steps, 'server_client_version': 7, 'confirm': True,
                               'status_fail_first': 0, 'missing_ids': []}, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run(['node', str(ROOT / 'tests/fixtures/app_sim_fix3.js'), str(ROOT / 'jetformer/web/static/app.js'),
                        str(ROOT / 'jetformer/web/static/index.html'), str(inp)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    toasts_before = None
    for s in out['steps']:
        if s['op'] == 'start':
            print('B start confirm lines with OPT:', [l for l in s['confirm'].split('\n') if '見張り' in l or '中立止め' in l or '発進の補助' in l])
            print('B start sent has opt?', [k for k in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick') if k in s['sent']])
        elif s['op'] == 'el':
            print('B el', s['sel'], s['prop'], s['value'])
        elif s['op'] == 'read':
            print('B note:', s['value'])
    print('B toasts:', out['toasts'])
finally:
    shutil.rmtree(tmp, ignore_errors=True)
