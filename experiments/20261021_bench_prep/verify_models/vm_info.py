#!/usr/bin/env python3
"""V-models: drive.model_info と model_config の確かめ(torch を読まない)。読むだけ"""
import hashlib, json, sys, os
from pathlib import Path
REPO = Path('/home/jetson/jetformer'); sys.path.insert(0, str(REPO))
from jetformer import drive, modes as M
C = json.load(open(REPO / 'data/bench_candidates.json'))
extra = {'MSF': 'clean_driving_20261011-msfusion-s0-nocmd-stride3-e30', 'TCP': 'clean_driving_20261011-tcp-s0-nocmd-stride3-e30',
         'SIM': 'simreal_mix_20261013-cotrain-ft-s0'}
rows = [(c['id'], c['run']) for c in C['candidates']] + list(extra.items())
out = {}
lm = {m['name']: m for m in drive.list_models()}
rec = drive.recommended()
for cid, run in rows:
    d = REPO / 'data/runs' / run
    info = drive.model_info(run)
    cfg = json.load(open(d / 'model_config.json'))
    eng = d / drive.ENGINE_FILE
    st = eng.stat()
    ec = json.load(open(d / 'engine_check.json')) if (d / 'engine_check.json').exists() else None
    h = hashlib.sha256(open(eng, 'rb').read()).hexdigest()
    e = (ec or {}).get('engine') or {}
    r = {'run': run, 'drivable': info['drivable'], 'why_not': info['why_not'], 'in_list_models': run in lm,
         'policy': info['policy'], 'num_modes': info['num_modes'], 'modes': info['modes'], 'trained_modes': info['trained_modes'],
         'command_input': info['command_input'], 'inputs': info['inputs'], 'outputs': info['outputs'],
         'postprocess': info['postprocess'], 'steer_center_us': info['steer_center_us'], 'throttle_neutral_us': info['throttle_neutral_us'],
         'future_steps': info['future_steps'], 'engine_precision': info['engine_precision'],
         'engine_check_seen_by_drive': info['engine_check'],
         'engine_size': st.st_size, 'engine_mtime_ns': st.st_mtime_ns, 'engine_sha256': h,
         'ec_engine': e, 'ec_current': bool(e) and int(e.get('size', -1)) == st.st_size and int(e.get('mtime_ns', -1)) == st.st_mtime_ns,
         'ec_accuracy_ok': (ec or {}).get('accuracy_ok'), 'ec_actions_max_us': (ec or {}).get('actions_max_us'),
         'ec_gate_us': (ec or {}).get('gate_us'), 'ec_ok': (ec or {}).get('ok'), 'ec_keys': sorted((ec or {}).keys()),
         'ec_checked_at': (ec or {}).get('checked_at'), 'ec_ckpt': (ec or {}).get('ckpt') or (ec or {}).get('checkpoint'),
         'files': sorted(p.name for p in d.iterdir()),
         'mode5_name': (info['modes'][5] if len(info['modes']) > 5 else None),
         }
    out[cid] = r
out['_recommended'] = rec
out['_modes_module'] = {k: getattr(M, k) for k in dir(M) if k.isupper() and isinstance(getattr(M, k), (int, str, list, tuple, dict))}
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)
for cid, r in out.items():
    if cid.startswith('_'): print(cid, json.dumps(r, ensure_ascii=False, default=str)[:600]); continue
    print(cid, r['drivable'], r['why_not'], r['policy'], 'nm', r['num_modes'], 'mode5', r['mode5_name'], 'trained', r['trained_modes'],
          'cmd', r['command_input'], 'cur', r['ec_current'], r['ec_accuracy_ok'], r['ec_actions_max_us'], r['postprocess'],
          'ref', r['steer_center_us'], r['throttle_neutral_us'], 'list', r['in_list_models'])
