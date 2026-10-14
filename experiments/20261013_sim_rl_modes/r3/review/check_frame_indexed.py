"""第 3〜4 回の審査: 昇格の段の dataset.py の差分(_frame_indexed)が、実機のシーンでは働かない(今までどおり時刻で引く)ことを data/raw の全シーンで確かめる(読むだけ)
    python check_frame_indexed.py > 結果.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import config, dataset as jds  # noqa: E402

out = {'real': {'scenes': 0, 'frame_indexed': 0, 'no_vehicle': 0}, 'expanded_sim': {'scenes': 0, 'frame_indexed': 0, 'no_vehicle': 0}, 'examples': [],
       'unreadable_scene_json': []}
for log in sorted(p for p in config.RAW_DIR.iterdir() if p.is_dir()):
    for sd in sorted(p for p in log.iterdir() if p.is_dir()):
        if not (sd / 'scene.json').exists():
            continue
        try:
            meta = json.load(open(sd / 'scene.json'))
        except ValueError:
            out['unreadable_scene_json'].append(str(sd))   # 0 バイトなど(この作業の前からあるもの)
            continue
        kind = 'expanded_sim' if meta.get('imported_from') else 'real'
        out[kind]['scenes'] += 1
        if not (sd / 'VEHICLE.npz').exists() or not (sd / 'CAM_FRONT.npz').exists():
            out[kind]['no_vehicle'] += 1
            continue
        veh = dict(np.load(sd / 'VEHICLE.npz', allow_pickle=True))
        cam_t = np.load(sd / 'CAM_FRONT.npz', allow_pickle=True)['timestamps_us']
        fi = bool(jds._frame_indexed(veh, cam_t))
        out[kind]['frame_indexed'] += int(fi)
        if (kind == 'real') == fi:
            out['examples'].append({'scene': str(sd), 'kind': kind, 'frame_indexed': fi})
print(json.dumps(out, ensure_ascii=False, indent=1))
