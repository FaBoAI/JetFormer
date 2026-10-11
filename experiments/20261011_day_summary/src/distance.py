#!/usr/bin/env python3
"""走行距離と速度の目安(|eRPM| x 0.5e-3 m/s、65mmタイヤ、推定±6%)。AI走行のログは AI 側だけ → distance.json

    nice -n 15 python distance.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs_table import cat, load_log  # noqa: E402

LOGS = ['20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505', '20261010-172910',
        '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208', '20261010-204605', '20261010-135449',
        '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707', '20261010-193638', '20261010-200916',
        '20261010-201626', '20261010-202633']
out = {}
for log in LOGS:
    meta, ok, broken, missing, sj, data = load_log(log)
    vt = cat(data['VEHICLE'], 'timestamps_us') / 1e6
    e = cat(data['VEHICLE'], 'erpm').astype(float)
    s = cat(data['VEHICLE'], 'select').astype(bool)
    dt = np.diff(vt)
    use = (dt > 0) & (dt <= 0.2)
    v = np.abs(e[1:]) * 0.5e-3
    m = use & (s[1:] if meta.get('drive') else True)
    mv = v[m][v[m] > 0.1]
    out[log] = {'distance_m': round(float(np.sum(v[m] * dt[m])), 1),
                'v_p50_moving_ms': round(float(np.median(mv)), 2) if len(mv) else None,
                'v_p99_ms': round(float(np.percentile(v[m], 99)), 2) if m.any() else 0}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distance.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
