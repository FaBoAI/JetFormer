# -*- coding: utf-8 -*-
"""学習データの読み込み(JPEG の読み込み・色の揺らぎ・正規化)だけの速さ [枚/秒](CPU だけ。GPU の計算が速い方策で律速になるかを見る)

    CUDA_VISIBLE_DEVICES= python loader_rate.py [--workers 4]   → results/loader_rate.json
"""
import os
import sys
import time

from torch.utils.data import DataLoader

from common import DATASET, RESULTS, save_json
from jetformer.train_data import JetFormerDataset


def main():
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    ds = JetFormerDataset(DATASET, 'train', color_jitter=True, stride=3, use_command=False, mask_impacts=True, impact_g=2.5)
    dl = DataLoader(ds, 64, shuffle=True, num_workers=workers, drop_last=True)
    n, t0, t_first = 0, time.time(), None
    for k, batch in enumerate(dl):
        if k == 2:
            t_first, n = time.time(), 0
        n += batch[0].shape[0]
        if k >= 32:
            break
    rate = n / (time.time() - t_first)
    res = {'workers': workers, 'batch': 64, 'samples_per_s': round(rate, 1), 'train_samples': len(ds),
           'startup_s': round(t_first - t0, 1)}
    save_json(os.path.join(RESULTS, 'loader_rate_w%d.json' % workers), res)
    print(res)


if __name__ == '__main__':
    main()
