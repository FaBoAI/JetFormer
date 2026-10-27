"""指摘 state2-1 の確かめの入力を作る(読むだけ)。bench.load_candidates() と drive の設定(既定 + drive_settings.json)を JSON に。"""
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

out = Path(sys.argv[1])
cands = bench.load_candidates()
p = dict(drive.DEFAULT_PARAMS)
ds = JF / 'data' / 'drive_settings.json'
try:
    saved = json.loads(ds.read_text(encoding='utf-8'))
    p.update(saved.get('params', saved) if isinstance(saved, dict) else {})
except FileNotFoundError:
    saved = None
out.write_text(json.dumps({'bench': cands, 'server_params': {k: v for k, v in p.items() if not isinstance(v, (dict, list))},
                           'saved_keys': sorted(saved.keys()) if isinstance(saved, dict) else None}, ensure_ascii=False, default=str))
print('days', [(d['date'], len(d['blocks'])) for d in cands['days']], 'today', cands['today'])
print('baseline trim', cands['baseline']['params'].get('steer_trim_us'), 'run', cands['baseline'].get('run'))
print('server trim', p.get('steer_trim_us'), 'saved keys', sorted(saved.keys()) if isinstance(saved, dict) else None)
