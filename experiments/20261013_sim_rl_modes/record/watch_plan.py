"""記録の段: プランの進み具合を見張り、変わったときだけ 1 行出す(Monitor 用)。外れ・取り込み・状態の変化と、10 シーンごとの区切り。
    python -u record/watch_plan.py <プランの ID>
"""
import json
import sys
import time
import urllib.request

pid = sys.argv[1]
last = None
last_done = -1
while True:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:8080/api/sim/rl/plans/{pid}', timeout=20) as r:
            p = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        print('読めない', e, flush=True)
        time.sleep(60)
        continue
    prog = p['progress']
    done = sum(1 for x in prog.values() if x.get('status') == 'done')
    failed = [(sid, x.get('error') or x.get('reason')) for sid, x in prog.items() if x.get('status') == 'failed']
    imps = {k: (v.get('state'), v.get('dataset'), (v.get('error') or '')[:120]) for k, v in (p.get('imports') or {}).items()}
    key = json.dumps([p['state'], failed, imps], ensure_ascii=False)
    if key != last or done // 10 != last_done // 10:
        print(time.strftime('%H:%M:%S'), p['state'], f'済み {done}/{len(p["order"])}', '外れ', failed, '取り込み', imps, (p.get('message') or '')[:120], flush=True)
        last, last_done = key, done
    fin = p['state'] in ('finished', 'stopped', 'error')
    if fin and all((p['imports'].get(k) or {}).get('state') in ('done', 'error') for k in p['modes']):
        print('プランが終わりました', p['state'], flush=True)
        break
    if p['state'] == 'error':
        time.sleep(120)
    time.sleep(30)
