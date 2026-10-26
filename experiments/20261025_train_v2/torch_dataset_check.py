#!/usr/bin/env python3
"""学習が読むサンプルを、train.py と同じ作り方の JetFormerDataset で確かめる(2026-10-25。小さな確かめ。学習しない)。

torch を読むので jf-heavy --lock big で動かす(CPU だけ。GPU に何も載せない)。各腕の学習・検証のデータセットを
train.py と同じ引数で作り、サンプル数・ジャンル別の数・重み・action_reference を出し、サンプルを少し読む。
check_v2_data.py(numpy の数え直し)と数が同じかを比べる。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big --mem 3G --need-mb 2000 -- nice -n 10 \
      /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/torch_dataset_check.py
出力: /home/jetson/jetformer_reports/train_v2_20261025/torch_dataset_check.json
"""
import json
import os
import random
import resource
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch  # noqa: E402
from jetformer import modes as jmodes  # noqa: E402
from jetformer.train_data import JetFormerDataset, load_frame_table  # noqa: E402

NAME = 'dataset_v2_20261025'
DATA = str(ROOT / 'data/datasets' / NAME)
TBL = ROOT / 'data/derived' / NAME
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
OUT = REP / 'torch_dataset_check.json'
ARMS = {
    'v2h': {'drop_mask': TBL / 'drop_v2h.npz'},
    'v2m': {'drop_mask': TBL / 'drop_v2m.npz', 'sample_weights_npz': TBL / 'weights_v2m.npz',
            'mode_remap_npz': TBL / 'remap_v2m.npz'},
    'bfast': {'drop_mask': TBL / 'drop_bfast.npz', 'sample_weights_npz': TBL / 'weights_bfast.npz'},
}


def rl_of(spec):
    """tools/train.py の rl の作り方と同じ"""
    rl = {}
    if spec.get('drop_mask'):
        with np.load(spec['drop_mask']) as z:
            has_keep = 'keep_impact' in z.files
        rl['drop'] = load_frame_table(str(spec['drop_mask']), 'drop')
        if has_keep:
            rl['keep_impact'] = load_frame_table(str(spec['drop_mask']), 'keep_impact')
    if spec.get('sample_weights_npz'):
        rl['weights'] = load_frame_table(str(spec['sample_weights_npz']), 'weight')
    if spec.get('mode_remap_npz'):
        rl['mode_remap'] = load_frame_table(str(spec['mode_remap_npz']), 'mode')
        ids = sorted({int(v) for _, val in rl['mode_remap'].values() for v in np.unique(val) if v >= 0})
        extra = []
        for i in ids:
            m = jmodes.get(i)
            if m is None:
                raise SystemExit('ジャンル %d が data/modes.json にありません' % i)
            extra.append(m)
        rl['extra_modes'] = extra
    return rl


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    res = {'dataset': NAME, 'torch': torch.__version__, 'cuda_used': False, 'arms': {}}
    for arm, spec in ARMS.items():
        rl = rl_of(spec)
        tr = JetFormerDataset(DATA, 'train', color_jitter=True, stride=3, use_command=False, mask_impacts=True,
                              impact_g=2.5, remap_dropout=0.0, return_weight='weights' in rl, **rl)
        va = JetFormerDataset(DATA, 'val', stride=3, use_command=False, mask_impacts=True, impact_g=2.5,
                              return_weight='weights' in rl, **rl)
        names = {m['id']: m['name'] for m in tr.modes}
        w = tr.sample_weights() if 'weights' in rl else np.ones(len(tr))
        src = np.array([int(tr.scenes[k]['source'][i]) for k, i in tr.index])
        # サンプルを読む: AI のサンプルは全部の中から 64、ほかから 64(形・値の範囲)
        idx_ai = [j for j in range(len(tr)) if src[j] == 1]
        idx_hu = [j for j in range(len(tr)) if src[j] != 1]
        pick = random.sample(idx_ai, min(64, len(idx_ai))) + random.sample(idx_hu, 64)
        shapes = set()
        bad = 0
        tgt_rng = [np.inf, -np.inf]
        for j in pick:
            item = tr[j]
            shapes.add(tuple(tuple(x.shape) for x in item[:6]))
            for x in item[:6]:
                if not torch.isfinite(x).all():
                    bad += 1
            t = item[5].numpy() * 500.0 + 1500.0
            tgt_rng = [min(tgt_rng[0], float(t.min())), max(tgt_rng[1], float(t.max()))]
            if 'weights' in rl:
                assert len(item) == 7
        vitem = va[0]
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        n_frames_loaded = int(sum(len(a['timestamps_us']) for a in tr.scenes) + sum(len(a['timestamps_us']) for a in va.scenes))
        res['arms'][arm] = {
            'train': len(tr), 'val': len(va), 'num_modes': tr.num_modes, 'val_num_modes': va.num_modes,
            'mode_counts': {names.get(m, m): c for m, c in sorted(tr.mode_counts().items())},
            'impact_masked': [tr.impact_masked, va.impact_masked], 'dropped': [tr.dropped, va.dropped],
            'source_counts': {str(k): int((src == k).sum()) for k in np.unique(src)},
            'weight': {'mean': round(float(w.mean()), 4), 'min': float(w.min()), 'max': float(w.max()),
                       'eff_samples': round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
                       'ai_weighted_share': round(float(w[src == 1].sum() / w.sum()), 4)},
            'action_reference': tr.action_reference(),
            'items_read': len(pick), 'item_shapes': [list(map(list, s)) for s in shapes], 'nonfinite_items': bad,
            'target_us_range': [round(tgt_rng[0], 1), round(tgt_rng[1], 1)],
            'val_item_shapes': [list(x.shape) for x in vitem[:6]],
            'max_rss_mb_so_far': round(rss_mb, 0), 'frames_loaded_in_memory': n_frames_loaded,
        }
        print(arm, json.dumps(res['arms'][arm], ensure_ascii=False), flush=True)
        del tr, va
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)


if __name__ == '__main__':
    main()
