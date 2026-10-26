#!/usr/bin/env python3
"""原因 A の診断 G1(GPU・読むだけ): 学習済みのモデルで、v2-H の学習のサンプル(6,506)と机上の集合
(直線 2,043・止まった場面 1,115・10/10 の AI 走行の低速 2,326)の予測と、ヘッドの手前の埋め込み(256 次元)を出す。
学習はしない。モデル・データ・既定値は変えない。

    cd /home/jetson/jetformer
    /home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 \
        env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 \
        /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261026_fix/diag_data/g1_embed.py

書くのは同じ場所の out/g1_<学習>.npz だけ(pred [N,10,2] µs・emb [N,256] float16・part)。
part: 0 = 学習のサンプル(d1_samples.npz の順)・1 = 直線・2 = 止まった場面・3 = 10/10 の AI 走行の低速(m2 と同じ並び)
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
DATA = REPO / 'data'
DS = DATA / 'datasets'
RUNS = DATA / 'runs'
SETS = Path('/home/jetson/jetformer_reports/train_v2_20261025/desk/sets_v2.npz')
GENRE = 5
RUN_LIST = ['clean_driving_20261011-nocmd-s3-20261011-024626',             # B
            'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',     # 10/10 だけの別の種(基準)
            'v2_20261025-v2h-e30-s0',                                       # 直線 +48
            'v2_20261025-v2h-e30-s1',                                       # 直線 +45・ブレーキ 0.019(少ない)
            'v2_20261025-v2h-e60-s2',                                       # 直線 +52・ブレーキ 0.124
            'v2_20261025-v2h-aug2-e60-s1',                                  # 最終の腕・H1 の最良
            'bfast_20261025-e15-s0']


def items_for(ds, scenes, frames):
    items, cur, a = [], None, None
    for s, i in zip(scenes, frames):
        s = str(s)
        if s != cur:
            with np.load(DS / ds / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                a = {k: z[k] for k in ('cam_files', 'tof', 'imu', 'vehicle')}
            cur = s
        i = int(i)
        items.append((str(DS / ds / str(a['cam_files'][i])), a['tof'][i].copy(), a['imu'][i].copy(), a['vehicle'][i].copy()))
    return items


def ai1010_low():
    """m2_brake_frames.ai_frames と同じ並びの 10/10 の AI 走行のうち v < 1.0 m/s"""
    import m2_brake_frames as M2
    rows = M2.ai_frames()
    v = np.array([-r[8] * M2.SPEED_PER_ERPM for r in rows])
    sel = np.nonzero(v < 1.0)[0]
    return [rows[i][0] for i in sel], [rows[i][1] for i in sel], sel


class Frames(object):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        import cv2
        import torch
        from jetformer.train_data import normalize_inputs
        path, tof, imu, veh = self.items[n]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError('画像が読めません: %s' % path)
        img, t, im, st = normalize_inputs(img, tof, imu, veh, use_command=False)
        return torch.from_numpy(img), torch.from_numpy(t), torch.from_numpy(im), torch.from_numpy(st)


def main():
    import torch
    from torch.utils.data import DataLoader
    from jetformer.policies import runtime
    sys.path.insert(0, str(HERE))
    D = np.load(OUT / 'd1_samples.npz')
    S = np.load(SETS)
    st_rows = np.nonzero(S['straight'])[0]
    sd_rows = np.nonzero(S['stand'])[0]
    a_sc, a_fr, a_sel = ai1010_low()
    items = items_for('dataset_v2_20261025', D['scene'], D['frame'])
    part = [0] * len(items)
    items += items_for('dataset_v2_20261025', S['scene'][st_rows], S['frame'][st_rows])
    part += [1] * len(st_rows)
    items += items_for('dataset_v2_20261025', S['scene'][sd_rows], S['frame'][sd_rows])
    part += [2] * len(sd_rows)
    items += items_for('ai_mix_20261011', a_sc, a_fr)
    part += [3] * len(a_sel)
    part = np.asarray(part, np.int8)
    print('フレーム %d(学習 %d・直線 %d・止まった場面 %d・10/10 低速 %d)' % (len(items), (part == 0).sum(), (part == 1).sum(),
                                                               (part == 2).sum(), (part == 3).sum()), flush=True)
    for name in RUN_LIST:
        dst = OUT / ('g1_%s.npz' % name)
        if dst.exists():
            print('%s: あり(飛ばす)' % name, flush=True)
            continue
        t0 = time.time()
        model, cfg, spec, pcfg = runtime.load_run(RUNS / name, 'best', device='cuda')
        if not hasattr(model, 'trunk'):
            print('%s: trunk が無い(飛ばす)' % name)
            continue
        trained = [int(x) for x in cfg['trained_modes']]
        g = GENRE if GENRE in trained else trained[0]
        emb = []
        h = model.trunk.register_forward_hook(lambda m, inp, out: emb.append(inp[0].detach().float().cpu().numpy()))
        dl = DataLoader(Frames(items), batch_size=64, shuffle=False, num_workers=2)
        preds = []
        with torch.no_grad():
            for image, tof, imu, state in dl:
                onehot = torch.nn.functional.one_hot(torch.full((len(image),), g, dtype=torch.int64), cfg['num_modes']).float()
                x = dict(image=image.cuda(), tof=tof.cuda(), imu=imu.cuda(), state=state.cuda(), mode=onehot.cuda())
                o = runtime.forward(model, x)
                preds.append(o['actions'].float().cpu().numpy())
        h.remove()
        pred = np.concatenate(preds) * 500.0 + 1500.0
        E = np.concatenate(emb)
        if pred.shape != (len(items), 10, 2) or E.shape[0] != len(items) or not np.isfinite(pred).all():
            raise SystemExit('%s: 形か値がおかしい %s %s' % (name, pred.shape, E.shape))
        np.savez_compressed(dst, pred=pred.astype(np.float32), emb=E.astype(np.float16), part=part,
                            st_rows=st_rows, sd_rows=sd_rows, a1010_sel=a_sel, genre=np.array(g),
                            ref=np.array([cfg['steer_center_us'], cfg['throttle_neutral_us']], np.float64))
        print('%s: %.0f 秒' % (name, time.time() - t0), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
