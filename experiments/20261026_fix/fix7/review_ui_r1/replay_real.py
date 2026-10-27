"""fuzz_confirm.py と同じでたらめな操作(起き直しは reboot だけ)を本物の app.js に流し、送った POST を本物の drive.Driver と
server._boot_refusal に同じ順で入れて、(1) 返事(200・400・409)が偽のサーバーと同じか (2) 走行開始のときの本物の params が
▶ の確認の文と合うかを見る(読むだけ。設定ファイルは一時の場所。車・jetformer-web には触れない)。
    python3 replay_real.py <一時の場所> [回数] [種]"""
import json
import random
import sys
from pathlib import Path

from common import RESET, blk, cands, check, drive, ops, run_sim, server_params, srv

tmp, N, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30, int(sys.argv[3]) if len(sys.argv) > 3 else 1
Path(tmp).mkdir(parents=True, exist_ok=True)
drive.SETTINGS_PATH = Path(tmp) / 'drive_settings_replay.json'          # 本物の data/drive_settings.json は読まない・書かない
drive.model_info = lambda name: {'trained_modes': [5], 'drivable': True, 'postprocess': None, 'name': name}
c, di = cands()
blocks = [int(b['block']) for b in c['days'][di]['blocks'] if b.get('drivable')]
rng = random.Random(seed)


class _Rec:
    recording = False

    def add_event(self, *a, **k):
        return True


def new_driver(params, boot):
    if drive.SETTINGS_PATH.exists():
        drive.SETTINGS_PATH.unlink()
    d = drive.Driver(hub=None, recorder=_Rec())
    d._run = lambda *a, **k: None
    d.params = dict(params)
    d.boot_at = boot
    return d


def gen():
    steps, boot = [['tab', 'drive'], ['tick']], 1000.0
    for _ in range(rng.randint(8, 30)):
        r = rng.random()
        if r < 0.10:
            steps += [['baseline']]
        elif r < 0.30:
            steps += [['check', rng.choice(['#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-speed-ceiling', '#dr-speed-auto',
                                            '#dr-launch-boost']), rng.random() < 0.75]]
        elif r < 0.40:
            steps += [['pick', di, blk(c, di, rng.choice(blocks))]]
        elif r < 0.48:
            steps += [['click', rng.choice(['#btn-dr-tg-up', '#btn-dr-tg-down'])]]
        elif r < 0.53:
            steps += [['api', rng.choice([{'ceil_left_mps': 4.2}, {'ceil_left_mps': 4.1}, {'launch_us': 200.0}, {'rec_hold_keep': 1},
                                          {'rec_back_guard': 1, 'rec_hold_keep': 1}])], ['tick']]
        elif r < 0.58:
            boot += 1000.0
            steps += [['posts'], ['eval', "'REBOOT %d'" % boot], ['reboot', boot, RESET]]
        elif r < 0.64:
            steps += [['down_all', True], ['tick']] + ([['tick']] if rng.random() < 0.5 else []) + [['down_all', False]]
        elif r < 0.68:
            steps += [['tab', 'live'], ['tick'], ['models_fail', rng.random() < 0.5], ['tab', 'drive'], ['models_fail', False]]
        elif r < 0.85:
            steps += [['tick']]
        else:
            steps += [['start'], ['read', 'server'], ['running', False], ['tick']]
    return steps + [['tick'], ['start'], ['read', 'server'], ['posts']]


bad, nstart = [], 0
for t in range(N):
    steps = gen()
    sp = server_params(throttle_gain=0.9)
    res, toasts = run_sim(tmp, c, steps, sp, name='r%d.json' % t)
    d = new_driver(sp, 1000.0)
    sr = [(s, r) for s, r in zip(ops(res, 'start'), [x['value'] for x in res if x['op'] == 'read']) if s.get('status') is not None]
    starts, reads = iter([a for a, _ in sr]), iter([b for _, b in sr])      # POST まで行った ▶ だけ(画面で断った ▶ は POST が無い)
    for x in res:
        if x['op'] == 'eval' and isinstance(x.get('value'), str) and x['value'].startswith('REBOOT'):
            keep = {k: v for k, v in d.params.items() if k not in drive.RESET_ON_BOOT}
            d = new_driver(dict(drive.DEFAULT_PARAMS, **keep, **RESET), float(x['value'].split()[1]))
            continue
        if x['op'] != 'posts':
            continue
        for path, body, status in x['value']:
            if path == '/api/drive/params':
                try:
                    d.set_params(body)
                    got = 200
                except ValueError:
                    got = 400
            elif path == '/api/drive/start':
                nstart += 1
                s, fake = next(starts), next(reads)
                if srv._boot_refusal(body, d.boot_at):
                    got = 409
                else:
                    d.start(body['run'], body['mode_id'], params=body['params'], description=body.get('description', ''),
                            dry_run=body.get('dry_run', False))
                    d.thread.join(1)
                    got = 200
                    m = check(s['confirm'], d.params, s['sent'])
                    diff = {k: (fake[k], d.params[k]) for k in d.params if k in fake and abs(float(fake[k]) - float(d.params[k])) > 1e-9}
                    if m or diff:
                        bad.append({'t': t, 'mismatch': m, 'fake_vs_real': diff})
            else:
                continue
            if status is not None and got != status:
                bad.append({'t': t, 'path': path, 'body': body, 'fake': status, 'real': got})
print(json.dumps({'runs': N, 'seed': seed, 'starts': nstart, 'bad': bad[:20]}, ensure_ascii=False)[:6000])
