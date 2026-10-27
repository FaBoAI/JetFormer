#!/usr/bin/env python3
"""係 tools(fix3、2026-10-26 夜): course_fit.LogData を本物の記録で読み、配列の sha256 を出す(直す前と後で同じかを見る)。読むだけ。
    python3 -I course_fit_hash.py <tools の場所> <出力 JSON> [記録 …](既定: 10/10・10/23・10/24 の AI 走行で AI_CMD のあるもの)"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(sys.argv[1])
sys.path.insert(0, str(TOOLS))
import course_fit as cf  # noqa: E402

RAW = Path('/home/jetson/jetformer/data/raw')
logs = sys.argv[3:] or sorted(p.name for p in RAW.iterdir() if p.is_dir() and len(p.name) == 15
                              and p.name[:8] in ('20261010', '20261023', '20261024') and list(p.glob('*/AI_CMD.npz')))


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()[:16]


out = {}
for lg in logs:
    try:
        d = cf.LogData(lg, str(RAW))
    except Exception as e:      # noqa: BLE001
        out[lg] = {'error': '%s: %s' % (type(e).__name__, e)}
        continue
    r = {'scenes': d.scenes, 'has_ai': bool(d.has_ai)}
    for k in ('cam_us', 'ct', 'cam_scene', 'it', 'acc', 'gyr', 'vt', 'pwm', 'erpm', 'vsel', 'hall', 'board', 'at', 'a_select', 'g', 'hg'):
        r[k] = h(getattr(d, k))
    if d.has_ai:
        r['a_out'] = h(d.a_out)
    grid = np.arange(0.0, float(d.ct[-1]) if len(d.ct) else 0.0, 0.05)
    r['ai_at_grid'] = h(d.ai_at(grid)) if len(grid) else None
    r['ai_at_n'] = int(np.sum(d.ai_at(grid))) if len(grid) else 0
    out[lg] = r
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('logs', len(out), 'errors', sum(1 for v in out.values() if 'error' in v))
