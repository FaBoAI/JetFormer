"""離脱復活 r4(2026-10-14): 確認 r3 の ck_real_scene_equal.py の写し(中身は同じ。差分 v2 で走らせ直すため)。
離脱復活 r3 の確認: 昇格の段の差分(dataset_frame_index.patch)が、今ある本物のシーン(data/raw)の学習用の配列をどう変えるか(2026-10-14)。

元の写しの jetformer をそのまま読み、差分を当てた dataset.py だけを jetformer.dataset_patched として同じパッケージに読み込む
(config・modes・tof_format は同じもの)。data/raw の全シーンで _aligned_arrays を両方で計算し、配列を 1 つずつ比べる。
読むだけ(data/raw には書かない)。
  python -I ck_real_scene_equal.py <元の写しの根> <差分を当てた dataset.py> <out.json>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT, PATCHED, OUT = sys.argv[1:4]
sys.path.insert(0, ROOT)
import jetformer  # noqa: E402
from jetformer import dataset as D0  # noqa: E402

assert Path(D0.__file__).resolve().is_relative_to(Path(ROOT).resolve()), D0.__file__
spec = importlib.util.spec_from_file_location('jetformer.dataset_patched', PATCHED)
D1 = importlib.util.module_from_spec(spec)
sys.modules['jetformer.dataset_patched'] = D1
spec.loader.exec_module(D1)

RAW = Path('/home/jetson/jetformer/data/raw')


def load(p, s):
    f = p / ('%s.npz' % s)
    return dict(np.load(f, allow_pickle=False)) if f.exists() else None


res = {'root': ROOT, 'patched': PATCHED, 'scenes': 0, 'identical': 0, 'changed': [], 'skipped': 0, 'by_quality': {}}
for p in sorted(RAW.glob('*/*')):
    if not p.is_dir() or not (p / 'CAM_FRONT.npz').exists():
        continue
    try:
        cam, tof, imu, veh = load(p, 'CAM_FRONT'), load(p, 'TOF_FRONT'), load(p, 'IMU'), load(p, 'VEHICLE')
    except Exception as e:  # 壊れた・空の npz(読めないシーン)は数えるだけ
        res.setdefault('unreadable', []).append({'scene': str(p.relative_to(RAW)), 'error': repr(e)[:120]})
        continue
    cam_t = cam['timestamps_us'].astype(np.int64)
    if veh is None or len(cam_t) == 0:
        res['skipped'] += 1
        continue
    q = str(veh['quality']) if 'quality' in veh else None
    a0 = D0._aligned_arrays(cam_t, tof, imu, veh)
    a1 = D1._aligned_arrays(cam_t, tof, imu, veh)
    res['scenes'] += 1
    keys = sorted(set(a0) | set(a1))
    diff = [k for k in keys if k not in a0 or k not in a1 or a0[k].shape != a1[k].shape or not np.array_equal(a0[k], a1[k])]
    bq = res['by_quality'].setdefault(str(q), {'scenes': 0, 'identical': 0, 'frame_indexed': 0})
    bq['scenes'] += 1
    fi = bool(D1._frame_indexed(veh, cam_t))
    bq['frame_indexed'] += fi
    if not diff:
        res['identical'] += 1
        bq['identical'] += 1
    else:
        fa0, fa1 = a0.get('future_actions'), a1.get('future_actions')
        nchg = int(np.any(fa0 != fa1, axis=(1, 2)).sum()) if fa0 is not None and fa1 is not None else None
        res['changed'].append({'scene': str(p.relative_to(RAW)), 'quality': q, 'frame_indexed': fi, 'keys': diff, 'frames': int(len(cam_t)),
                               'obs_with_changed_labels': nchg,
                               'future_valid_true': [int(a0['future_valid'].sum()), int(a1['future_valid'].sum())]})
res['changed_count'] = len(res['changed'])
res['changed_not_frame_indexed'] = sum(1 for c in res['changed'] if not c['frame_indexed'])
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'changed'}, ensure_ascii=False, indent=1))
print('changed (first 5):', json.dumps(res['changed'][:5], ensure_ascii=False))
