#!/usr/bin/env python3
"""天井の見積もり(2026-10-15)の下ごしらえ(CPU だけ): 保留の表(10,985 フレーム)の正解・今の値・場所と、
今までの全モデルの予測(best・last)を 1 つの npz にまとめる。学習はしない。

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_real_accuracy/ideas_ceiling/build_cache.py
"""
import glob
import os
import sys

import numpy as np

REPO = '/home/jetson/jetformer'
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, REPO)
import rl_evaluate as ev  # noqa: E402
import policy_zoo as pz  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
PRED_DIRS = ['policy_zoo_20261012', 'sim_cotrain_20261013', 'sim_diverse_train_20261015']


def main():
    os.makedirs(OUT, exist_ok=True)
    tab = ev.pred_table()
    held = ev._heldout()
    assert tab['key'] == held['pred_table_key']
    cols, f = ev._val_arrays(tab)
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    print('human_clean', int(clean.sum()))
    frow = cols['frow']
    fr = np.where(frow >= 0, frow, 0)
    extra = {}
    for k in ('timestamps_us', 'log', 'loc_code', 'progress_m', 'erpm', 'gz', 'label_index', 'split', 'mode', 'source'):
        v = f[k][fr]
        extra['f_' + k] = v
    preds, names = [], []
    for d in PRED_DIRS:
        for p in sorted(glob.glob(os.path.join(REPO, 'data', 'zoo', d, 'pred', '*.npz'))):
            with np.load(p) as z:
                if str(z['key']) != tab['key']:
                    print('skip(別の表)', p)
                    continue
                preds.append(z['pred'].astype(np.float32))
                names.append(d + '/' + os.path.basename(p)[:-4])
    print(len(names), 'models')
    np.savez_compressed(os.path.join(OUT, 'heldout_cache.npz'), key=tab['key'], part=tab['part'], scene=tab['scene'],
                        frame=tab['frame'], y=cols['y'].astype(np.float32), cur=cols['cur'].astype(np.float32),
                        fv=cols['fv'], impact=cols['impact'], mode=cols['mode'], frow=frow, human=human, clean=clean,
                        names=np.array(names), preds=np.stack(preds), **extra)
    # 確かめ: A 種 0 の主指標(10/12・10/13 と同じ 49.146)
    i = names.index('policy_zoo_20261012/jetformer_v0_s0')
    print('A_s0 L1 = %.3f' % pz.l1_rows(preds[i], cols['y'])[clean].mean())


if __name__ == '__main__':
    main()
