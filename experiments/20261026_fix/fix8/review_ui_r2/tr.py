"""p2_fuzz の 1 本を、操作と結果を並べて出す(読むだけ)。  python3 tr.py 種 wait|raw 番号 [app.js]"""
import json
import random
import sys
from pathlib import Path

seed, mode, idx = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
app = sys.argv[4] if len(sys.argv) > 4 else '-'
sys.argv = ['p2_fuzz.py', '1', str(seed), app, mode]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_fuzz  # noqa: E402
from run import run  # noqa: E402

rng = random.Random(seed)
for i in range(idx + 1):
    pages, steps = p2_fuzz.gen(rng, i % 2 == 1)
out = run(steps, pages=pages, name='tr.json', **({'app': Path(app)} if app != '-' else {}))
for st, o in zip(steps, out['steps']):
    oo = {k: v for k, v in o.items() if k not in ('sent', 'confirms', 'server_at_start', 'value', 'desc')}
    if o.get('op') == 'start':
        sv = o.get('server_at_start') or {}
        oo['confirm'] = [x[:30] for x in (o.get('confirm') or '').split('\n') if x.startswith(('リカバリー', '中立', '速さの天井'))]
        oo['srv'] = (sv.get('rec_back_guard'), sv.get('rec_hold_keep'), sv.get('ceil_left_mps'))
    if o.get('op') == 'check':
        oo['confirm'] = (o.get('confirm') or '')[:20]
    if o.get('op') == 'dstate':
        oo['value'] = o['value']
    if o.get('op') == 'server':
        oo['value'] = {k: o['value'].get(k) for k in ('rec_back_guard', 'rec_hold_keep', 'ceil_left_mps')}
    print(json.dumps(st, ensure_ascii=False)[:70].ljust(70), '=>', json.dumps(oo, ensure_ascii=False)[:260])
print(json.dumps(out['toasts'], ensure_ascii=False)[:3000])
