"""反証 7-2 回目・係 tools(読むだけ): 本物の 10/10・10/23・10/24 のラベルで、今の standing_3lap.abort_resumes が印を付ける AI の区間を
数える(fix7 r1 の報告 7 章「10/10 の 20261010-181504@37.70 の 1 本だけ」と同じかを、別の台本で確かめる)。出力 ../out/scan_real_abort_marks.json"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402

RAW, LAB = JF / 'data' / 'raw', JF / 'data' / 'labels'
out, n_logs, n_sec, n_giveup_logs = [], 0, 0, 0
for p in sorted(RAW.iterdir()):
    if p.name[:8] not in ('20261010', '20261023', '20261024') or len(p.name) != 15:
        continue
    lz, lj = LAB / (p.name + '.npz'), LAB / (p.name + '.json')
    if not (lz.exists() and lj.exists()):
        continue
    z = dict(np.load(lz))
    t = z['t_s'].astype(np.float64)
    ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
    if not ai.any():
        continue
    n_logs += 1
    n_sec += len(le.runs(ai))
    if (z['assist'] == 3).any():
        n_giveup_logs += 1
    lab = json.loads(lj.read_text(encoding='utf-8'))
    V = le.load_log(p.name, RAW)['VEHICLE']
    vt = (V['timestamps_us'].astype(np.int64) - int(lab['t0_us'])) / 1e6
    _, Df = le.cumulative_distance(vt, V['erpm'].astype(np.float64))
    ab = s3.abort_resumes(t, ai, z['assist'], z['hold'], z['ai_cmd_select'], vt, Df)
    for i0, v in ab.items():
        out.append({'log': p.name, 't_ai': round(float(t[i0]), 3), **v})
res = {'logs': n_logs, 'ai_sections': n_sec, 'logs_with_assist3': n_giveup_logs, 'marked': out}
(Path(__file__).resolve().parent.parent / 'out' / 'scan_real_abort_marks.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(res)
