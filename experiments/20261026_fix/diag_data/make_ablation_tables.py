#!/usr/bin/env python3
"""原因 A の確かめの学習(短い・同じ種の 3 本)の表を、この台本の場所の out/ablation/ に作る(numpy だけ)。
data/derived には書かない(v3 の表は段 data の担当)。どれも drop_v2h.npz に「外す」を足しただけ:
  drop_v2h_same.npz      drop_v2h と同じ(比べの土台。sha256 が変わらないことを確かめる)
  drop_v2h_noTK.npz      + 交代の直後 3 秒の学習のサンプル(d1 の sub が 交代の直後3秒・走り+交代の直後)を外す
  drop_v2h_noTK_noLB.npz + さらに v < 1.0 m/s で正解(行 0)がブレーキ側(> 1598)の学習のサンプルを外す
reason: 0〜4 は drop_v2h と同じ・5 = 交代の直後(この診断)・6 = 低速のブレーキの正解(この診断)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'ablation'
SRC = Path('/home/jetson/jetformer/data/derived/dataset_v2_20261025/drop_v2h.npz')
TK_SUBS = ('交代の直後3秒', '走り+交代の直後')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    D = np.load(HERE / 'out' / 'd1_samples.npz')
    z = np.load(SRC)
    scene, frame, drop, reason = z['scene'], z['frame'], z['drop'].copy(), z['reason'].copy()
    key = {(str(s), int(f)): i for i, (s, f) in enumerate(zip(scene, frame))}
    new = D['group'] != 'B41(10/10 人)'
    tk = new & np.isin(D['sub'], TK_SUBS)
    lb = (D['v'] < 1.0) & (D['thr0'] > 1598)
    rows_tk = np.array([key[(str(s), int(f))] for s, f in zip(D['scene'][tk], D['frame'][tk])], np.int64)
    rows_lb = np.array([key[(str(s), int(f))] for s, f in zip(D['scene'][lb & ~tk], D['frame'][lb & ~tk])], np.int64)
    if drop[rows_tk].any() or drop[rows_lb].any():
        raise SystemExit('外すつもりのフレームがもう drop になっています(d1 と表が合わない)')
    OUT.mkdir(parents=True, exist_ok=True)
    info = {'source': str(SRC), 'source_sha256': sha(SRC), 'n_takeover_samples': int(len(rows_tk)),
            'n_lowspeed_brake_samples_extra': int(len(rows_lb)), 'files': {}}
    variants = {'drop_v2h_same.npz': ([], []), 'drop_v2h_noTK.npz': ([rows_tk], [5]),
                'drop_v2h_noTK_noLB.npz': ([rows_tk, rows_lb], [5, 6])}
    for name, (rows, codes) in variants.items():
        d, r = drop.copy(), reason.copy()
        for rr, c in zip(rows, codes):
            d[rr] = True
            r[rr] = c
        np.savez(OUT / name, scene=scene, frame=frame, drop=d, reason=r.astype(np.int8))
        info['files'][name] = {'sha256': sha(OUT / name), 'dropped_total': int(d.sum()), 'added': int(d.sum() - drop.sum())}
    (OUT / 'tables.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
