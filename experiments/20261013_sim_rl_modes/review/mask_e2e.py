#!/usr/bin/env python3
"""審査の段(2026-10-14): 乱した操作(info.jf.l = 0)が学習に使われないことを、JetFormer の本物の読み込みの道で確かめる(書き込みなし・CPU)。

A. 取り込んだデータセット(data/datasets/sim-rl-<mode>_…): tools/train.py が使う jetformer.train_data.JetFormerDataset(train・val)の
   index に入る観測 k について、入力の操作 k と正解の操作 k+1〜k+10(future_actions、frames.jsonl の applied_pwm と値で照合)に l = 0 が無いこと。
B. シーンタブに展開したシーンからデータセットを作り直す道(jetformer.dataset._aligned_arrays・_frame_modes、build_dataset と同じ計算)で、
   学習に使える観測(future_valid & vehicle_valid & mode != 0)の正解が取る VEHICLE のサンプル(= フレーム)と入力のサンプルに l = 0 が無いこと。
   ジャンルを付け直した場合(dataset._segments_keeping_mask(meta, 8)、update_scene と同じ)も同じ。
    python review/mask_e2e.py <データセット名> [...] > 結果.json
"""
import json
import os
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
            L.append(int(jf.get('l', 0)) if isinstance(jf.get('l', 0), (int, float)) else 0)
            pwm.append(r['applied_pwm_us'])
    return np.array(L), np.array(pwm, dtype=np.float64)


def check_dataset(name):
    root = config.DATASET_DIR / name
    splits = json.load(open(root / 'splits.json'))
    out = {'dataset': name, 'A_loader': {}, 'B_rebuild': {}, 'B_rebuild_relabel': {}}
    lmap, pmap = {}, {}
    for sc in splits['train'] + splits['val']:
        lmap[sc], pmap[sc] = frames_l(root / 'simulation' / 'scenes' / sc / 'frames.jsonl')
    # A: 本物の読み込み
    for split in ('train', 'val'):
        ds = JetFormerDataset(str(root), split)
        names = splits[split]
        bad_in = bad_tgt = n = 0
        tgt_mismatch = 0
        for k_sc, i in ds.index:
            sc = names[k_sc]
            L, P = lmap[sc], pmap[sc]
            a = ds.scenes[k_sc]
            n += 1
            if L[i] == 0:
                bad_in += 1
            if (L[i + 1:i + 1 + FUT] == 0).any():
                bad_tgt += 1
            # 正解の値が frames.jsonl の applied_pwm の k+1〜k+10 と同じか(どのフレームの操作かの確かめ)
            if not np.allclose(a['future_actions'][i], P[i + 1:i + 1 + FUT, :2], atol=1e-3):
                tgt_mismatch += 1
        l0 = sum(int((lmap[s] == 0).sum()) for s in names)
        out['A_loader'][split] = {'scenes': len(names), 'indexed_obs': n, 'l0_actions_in_split': l0, 'obs_with_l0_input': bad_in,
                                  'obs_with_l0_in_targets': bad_tgt, 'targets_not_matching_k+1..k+10': tgt_mismatch}
    # B: 展開したシーンから作り直す道
    log = 'ds-' + name
    for key, relabel in (('B_rebuild', False), ('B_rebuild_relabel', True)):
        tot = {'scenes': 0, 'learnable_obs': 0, 'obs_with_l0_input': 0, 'obs_with_l0_in_targets': 0, 'target_frame_shift_counts': {}}
        for sd in sorted((config.RAW_DIR / log).iterdir()):
            if not (sd / 'scene.json').exists():
                continue
            meta = json.load(open(sd / 'scene.json'))
            sc = meta['imported_from']['scene']
            L = lmap[sc]
            if relabel:
                meta = dict(meta)
                meta['mode_segments'] = jds._segments_keeping_mask(meta, 8)
                meta['mode_id'] = 8
            cam = dict(np.load(sd / 'CAM_FRONT.npz', allow_pickle=True))
            veh = dict(np.load(sd / 'VEHICLE.npz', allow_pickle=True))
            cam_t = cam['timestamps_us']
            arr = jds._aligned_arrays(cam_t, None, None, veh)
            mode = jds._frame_modes(meta, cam_t)
            ok = arr['future_valid'] & arr['vehicle_valid'] & (mode != 0)
            # 正解の各段が取る VEHICLE のサンプル(build_dataset と同じ計算)。VEHICLE のサンプル = フレームの操作(600 個)
            sel = veh['valid'][:, 0] & veh['valid'][:, 1] & ~(veh['select'] if 'select' in veh else np.zeros(len(veh['timestamps_us']), bool))
            vt = veh['timestamps_us'][sel]
            vidx = np.nonzero(sel)[0]
            assert len(veh['timestamps_us']) == len(L), (sc, len(veh['timestamps_us']), len(L))
            frame_us = 1e6 / config.CAMERA['fps']
            j0 = vidx[np.clip(jds._latest_index(vt, cam_t), 0, len(vt) - 1)]
            jks = [vidx[np.clip(jds._latest_index(vt, cam_t + int((k + 1) * frame_us)), 0, len(vt) - 1)] for k in range(FUT)]
            J = np.stack(jks, 1)
            tot['scenes'] += 1
            for i in np.nonzero(ok)[0]:
                tot['learnable_obs'] += 1
                if L[j0[i]] == 0:
                    tot['obs_with_l0_input'] += 1
                if (L[J[i]] == 0).any():
                    tot['obs_with_l0_in_targets'] += 1
                sh = int(J[i][0] - i)
                tot['target_frame_shift_counts'][sh] = tot['target_frame_shift_counts'].get(sh, 0) + 1
        out[key] = tot
    return out


if __name__ == '__main__':
    res = [check_dataset(n) for n in sys.argv[1:]]
    print(json.dumps(res, ensure_ascii=False, indent=1))
