"""段 Decide: jetformer/web/bench.py の load_candidates で data/bench_candidates.json を読む(読むだけ。走行・保存の API は呼ばない)"""
import json
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.web import bench  # noqa: E402

r = bench.load_candidates()
d0 = [d for d in r['days'] if d['date'] == '2026-10-23'][0]
res = {'exists': r['exists'], 'ok': r['ok'], 'error': r['error'], 'warnings': r['warnings'], 'n_warnings': len(r['warnings']),
       'baseline_run': r['baseline']['run'], 'baseline_mode_id': r['baseline']['mode_id'],
       'candidates': [{k: c[k] for k in ('id', 'label', 'run', 'risk', 'drivable', 'why_not', 'policy_title')} for c in r['candidates']],
       'day1': [{k: b[k] for k in ('block', 'id', 'run', 'label', 'drivable', 'why_not', 'description')} for b in d0['blocks']],
       'all_drivable': all(c['drivable'] for c in r['candidates']) and all(b['drivable'] for b in d0['blocks']),
       'torch_imported': 'torch' in sys.modules, 'tensorrt_imported': 'tensorrt' in sys.modules}
print(json.dumps(res, ensure_ascii=False, indent=1))
sys.exit(0 if (res['ok'] and res['n_warnings'] == 0 and res['all_drivable']) else 1)
