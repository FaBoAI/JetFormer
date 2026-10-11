#!/usr/bin/env python3
"""オンラインで使える「ジャイロの向き ψ による区間判定」(sections.py と同じ規則)が、
夜間にコース図へ当てはめた軌跡(overnight_20261011/course/tracks_*.npz の section。後処理・未来も使う)と
どれだけ一致するかを数える(読み取りのみ、CPU)。  nice -n 15 python phase_check.py > ../phase_check.json"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from sections import analyse  # noqa: E402
from common import load  # noqa: E402
TR = '/home/jetson/jetformer_reports/overnight_20261011/course/tracks_%s.npz'
res = {}
for log, name in (('20261010-172910', 'ai_run_006'), ('20261010-173445', 'ai_run_007')):
    o, s = analyse(log, name)
    _, d = load(log, ('IMU',))
    t_abs = d['IMU']['timestamps_us']
    z = np.load(TR % log, allow_pickle=True)
    names = [str(x) for x in z['section_names']]
    ok = z['valid']
    k = np.searchsorted(z['t_us'], t_abs)
    k = np.clip(k, 0, len(z['t_us']) - 1)
    near = np.abs(z['t_us'][k] - t_abs) < 50_000
    lab_map = np.array([names[i] if i >= 0 else '' for i in z['section']])[k]
    m = (s['sec'] != '') & near & ok[k] & s['sel']
    conf = {}
    for a, b in zip(s['sec'][m], lab_map[m]):
        conf.setdefault(a, {}).setdefault(b, 0)
        conf[a][b] += 1
    res[name] = {'samples': int(m.sum()), 'confusion_psi_vs_map': conf, 'map_section_names': names}
json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
