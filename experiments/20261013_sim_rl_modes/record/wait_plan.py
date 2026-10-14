"""記録の段: プランが「済み N シーン以上」か「取り込みが終わった」か「止まった」になるまで(最長 --timeout 秒)待って、様子を 1 行出す。
    python -u record/wait_plan.py <プランの ID> [--done N] [--import-key fastest] [--timeout 570]
"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('pid')
ap.add_argument('--done', type=int, default=10 ** 6)
ap.add_argument('--import-key', default=None)
ap.add_argument('--timeout', type=float, default=570)
a = ap.parse_args()
t0 = time.time()
while True:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:8080/api/sim/rl/plans/{a.pid}', timeout=20) as r:
            p = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        p = None
        print('読めない', e, flush=True)
    if p:
        prog = p['progress']
        done = sum(1 for x in prog.values() if x.get('status') == 'done')
        failed = [sid for sid, x in prog.items() if x.get('status') == 'failed']
        imps = {k: v.get('state') for k, v in (p.get('imports') or {}).items()}
        hit = (done >= a.done or (a.import_key and imps.get(a.import_key) in ('done', 'error')) or p['state'] in ('error', 'stopped', 'finished'))
        if hit or time.time() - t0 > a.timeout:
            print(time.strftime('%H:%M:%S'), p['state'], f'済み {done}/{len(p["order"])}', '外れ', failed, '取り込み', imps,
                  'hit' if hit else 'timeout', flush=True)
            break
    time.sleep(15)
