#!/usr/bin/env python3
"""案 D の表(この台本の out/ablation/ に作る。numpy だけ): 交代の直後 3 秒の学習のサンプルの「スロットルの正解」だけを、
B の予測(G1・PyTorch FP32・ジャンル 5・行 0〜9)に差し替える。ステアの正解は人のまま(立て直しのステアは学ぶ)。
tools/train.py の --label-override-npz(scene, frame, future_throttle_us [M,10])の形。"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
TK_SUBS = ('交代の直後3秒', '走り+交代の直後')


def main():
    D = np.load(HERE / 'out' / 'd1_samples.npz')
    g = np.load(HERE / 'out' / ('g1_%s.npz' % B_RUN))
    tr = g['part'] == 0
    pB = g['pred'][tr].astype(np.float32)
    if len(pB) != len(D['scene']):
        raise SystemExit('G1 の学習のサンプルの数が d1 と合いません')
    new = D['group'] != 'B41(10/10 人)'
    tk = new & np.isin(D['sub'], TK_SUBS)
    out = HERE / 'out' / 'ablation' / 'override_tk_thr_B.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, scene=D['scene'][tk].astype('<U20'), frame=D['frame'][tk].astype(np.int64),
             future_throttle_us=pB[tk][:, :, 1])
    info = {'rows': int(tk.sum()), 'label_thr_row2_mean': float(D['thr2'][tk].mean()),
            'B_thr_row2_mean': float(pB[tk][:, 2, 1].mean())}
    (out.with_suffix('.json')).write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(info)


if __name__ == '__main__':
    sys.exit(main())
