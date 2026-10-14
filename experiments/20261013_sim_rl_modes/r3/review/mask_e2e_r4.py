#!/usr/bin/env python3
"""第 3〜4 回の審査(2026-10-15): 昇格の段が記録・取り込んだデータセットで、乱した操作(info.jf.l = 0)が学習に使われないこと、
正解(未来の操作)が k+1〜k+10 のフレームの操作であることを、JetFormer の今のコードの道で確かめる(書き込みなし・CPU)。
review/mask_e2e.py(2026-10-14)の写しで、B を昇格の段の dataset.py(展開したシーンはフレームの番号で引く)に合わせた。

A. 取り込んだデータセット: tools/train.py が使う jetformer.train_data.JetFormerDataset(train・val)の index の観測 k について、
   入力の操作 k と正解 k+1〜k+10 に l = 0 が無いこと、正解の値が frames.jsonl の applied_pwm_us の k+1〜k+10 と同じこと。
B. シーンタブに展開したシーン(data/raw/ds-<名前>/)から作り直す道(jetformer.dataset._aligned_arrays・_frame_modes。build_dataset と同じ):
   学習に使える観測(future_valid & vehicle_valid & mode != 0)で、正解 = applied_pwm の k+1〜k+10(値で照合)・l = 0 が入らない。
C. B + シーンタブでジャンルを付け直す(dataset._segments_keeping_mask(meta, 8)。update_scene と同じ)。

    python review/mask_e2e_r4.py <データセット名> [...] > 結果.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import config, dataset as jds  # noqa: E402
from jetformer.train_data import JetFormerDataset  # noqa: E402

FUT = 10


def frames_l(path):
    L, pwm = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            jf = ((r.get('action_info') or {}).get('jf') or {})
            v = jf.get('l', 0)
            L.append(int(v) if isinstance(v, (int, float)) else 0)
            pwm.append(r['applied_pwm_us'])
    return np.array(L), np.array(pwm, dtype=np.float64)


def check_dataset(name):
    root = config.DATASET_DIR / name
    splits = json.load(open(root / 'splits.json'))
    out = {'dataset': name, 'A_loader': {}, 'B_rebuild': {}, 'C_rebuild_relabel': {}}
    lmap, pmap = {}, {}
    for sc in splits['train'] + splits['val']:
        lmap[sc], pmap[sc] = frames_l(root / 'simulation' / 'scenes' / sc / 'frames.jsonl')
    for split in ('train', 'val'):
        ds = JetFormerDataset(str(root), split)
        names = splits[split]
        bad_in = bad_tgt = n = mism = 0
        for k_sc, i in ds.index:
            sc = names[k_sc]
            L, P = lmap[sc], pmap[sc]
            a = ds.scenes[k_sc]
            n += 1
            bad_in += int(L[i] == 0)
            bad_tgt += int((L[i + 1:i + 1 + FUT] == 0).any())
            if not np.allclose(a['future_actions'][i], P[i + 1:i + 1 + FUT, :2], atol=1e-3):
                mism += 1
        out['A_loader'][split] = {'scenes': len(names), 'indexed_obs': n, 'l0_actions_in_split': sum(int((lmap[s] == 0).sum()) for s in names),
                                  'obs_with_l0_input': bad_in, 'obs_with_l0_in_targets': bad_tgt, 'targets_not_k+1..k+10': mism}
    log = 'ds-' + name
    raw = config.RAW_DIR / log
    out['scene_tab_dir'] = str(raw)
    for key, relabel in (('B_rebuild', False), ('C_rebuild_relabel', True)):
        tot = {'scenes': 0, 'learnable_obs': 0, 'obs_with_l0_input': 0, 'obs_with_l0_in_targets': 0, 'targets_not_k+1..k+10': 0, 'targets_k..k+9_would_match': 0,
               'frame_indexed_scenes': 0, 'mode_ids': {}}
        if not raw.is_dir():
            out[key] = {'error': 'シーンタブのログが無い'}
            continue
        for sd in sorted(raw.iterdir()):
            if not (sd / 'scene.json').exists():
                continue
            meta = json.load(open(sd / 'scene.json'))
            sc = meta['imported_from']['scene']
            L, P = lmap[sc], pmap[sc]
            tot['mode_ids'][str(meta.get('mode_id'))] = tot['mode_ids'].get(str(meta.get('mode_id')), 0) + 1
            if relabel:
                meta = dict(meta)
                meta['mode_segments'] = jds._segments_keeping_mask(meta, 8)
                meta['mode_id'] = 8
            cam = dict(np.load(sd / 'CAM_FRONT.npz', allow_pickle=True))
            veh = dict(np.load(sd / 'VEHICLE.npz', allow_pickle=True))
            cam_t = cam['timestamps_us']
            tot['frame_indexed_scenes'] += int(jds._frame_indexed(veh, cam_t))
            arr = jds._aligned_arrays(cam_t, None, None, veh)
            mode = jds._frame_modes(meta, cam_t)
            ok = arr['future_valid'] & arr['vehicle_valid'] & (mode != 0)
            tot['scenes'] += 1
            for i in np.nonzero(ok)[0]:
                tot['learnable_obs'] += 1
                tot['obs_with_l0_input'] += int(L[i] == 0)
                tot['obs_with_l0_in_targets'] += int((L[i + 1:i + 1 + FUT] == 0).any())
                # 展開したシーンの VEHICLE.npz は PWM を整数(µs)で持つので、丸めの差 0.501 µs までを同じとみなす(float32 の x.50002 が偶数丸めで下がる分)
                if i + 1 + FUT > len(P) or np.abs(arr['future_actions'][i] - P[i + 1:i + 1 + FUT, :2]).max() > 0.501:
                    tot['targets_not_k+1..k+10'] += 1
                if np.abs(arr['future_actions'][i] - P[i:i + FUT, :2]).max() <= 0.501:
                    tot['targets_k..k+9_would_match'] += 1
        out[key] = tot
    return out


if __name__ == '__main__':
    print(json.dumps([check_dataset(n) for n in sys.argv[1:]], ensure_ascii=False, indent=1))
