#!/usr/bin/env python3
"""モデルB(操作値なし)のジャンルを入れ替えたときの予測(CPU だけ、GPU・CUDA は使わない)

    nice -n 19 python genre_swap.py lap_quality.json > genre_swap.json

人の走行(clean_driving_20261011 のシーン)の同じフレームに、ジャンル 4(高速)と 5(低速)を入れて、
3フレーム先(先読み3 で使う k=2)のスロットル・ステアの予測を比べる。場所は lap_quality.json の R1 の起点と
緑マット出口 tg から決める:
  R1手前 = [起点−1.0, 起点)、DE = [tg+0.3, tg+1.5)、残り(SW〜HS〜LH の前半) = [tg+2.0, 次の起点−1.5)
フレームは 3枚に1枚。前進量 = 中立 1496µs − スロットル(+ が前進)。人の値は future_actions[k=2]。
モデルB は操作値を入力しないので、この比較は AI走行の条件(自己入力)と同じになる(ただし状態は人の走行のもの)。
学習に使ったシーン(train)と検証シーン(val)を分けて出す。
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.model import PWM_CENTER, PWM_SCALE, JetFormer  # noqa: E402
from jetformer.train_data import normalize_inputs  # noqa: E402

torch.set_num_threads(2)
RUN = '/home/jetson/jetformer/data/runs/clean_driving_20261011-nocmd-s3-20261011-024626'
DS = '/home/jetson/jetformer/data/datasets/clean_driving_20261011'
NEUTRAL = 1496.0
STEP = 3

cfg = json.load(open(os.path.join(RUN, 'model_config.json'), encoding='utf-8'))
assert cfg.get('command_input', True) is False
model = JetFormer(num_modes=cfg['num_modes']).eval()
model.load_state_dict(torch.load(os.path.join(RUN, 'best.pth'), map_location='cpu', weights_only=True))
splits = json.load(open(os.path.join(DS, 'splits.json')))
lq = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'lap_quality.json'))

rows = []
for log, v in lq['logs'].items():
    if v['kind'] != 'human':
        continue
    t0 = v['t0_us'] / 1e6
    ps = v['passes']
    zones = []
    for k, P in enumerate(ps):
        a = P['a'] + t0
        zones.append(('R1', a - 1.0, a, P))
        if P['tg'] is not None:
            tg = P['tg'] + t0
            zones.append(('DE', tg + 0.3, tg + 1.5, P))
            if k + 1 < len(ps):
                b = ps[k + 1]['a'] + t0
                if b - a < 14.0 and b - 1.5 > tg + 2.0:
                    zones.append(('REST', tg + 2.0, b - 1.5, P))
    for scene in sorted(f[:-4] for f in os.listdir(os.path.join(DS, 'aligned')) if f.startswith(log)):
        split = 'val' if scene in splits['val'] else 'train' if scene in splits['train'] else None
        if split is None:
            continue
        al = np.load(os.path.join(DS, 'aligned', scene + '.npz'), allow_pickle=True)
        ts = al['timestamps_us'] / 1e6
        for zname, x, y, P in zones:
            idx = np.nonzero((ts >= x) & (ts < y) & al['future_valid'] & al['vehicle_valid'])[0][::STEP]
            if not len(idx):
                continue
            ims, tofs, imus, sts = [], [], [], []
            for i in idx:
                img = cv2.imread(os.path.join(DS, str(al['cam_files'][i])))
                im, tof, imu, st = normalize_inputs(img, al['tof'][i], al['imu'][i], al['vehicle'][i], False)
                ims.append(im), tofs.append(tof), imus.append(imu), sts.append(st)
            n = len(idx)
            preds = {}
            with torch.no_grad():
                for g in (4, 5):
                    mode = torch.zeros(n, cfg['num_modes'])
                    mode[:, g] = 1
                    out = model(torch.from_numpy(np.stack(ims)), torch.from_numpy(np.stack(tofs)),
                                torch.from_numpy(np.stack(imus)), torch.from_numpy(np.stack(sts)), mode)
                    preds[g] = out.numpy() * PWM_SCALE + PWM_CENTER
            hum = al['future_actions'][idx]
            for j in range(n):
                rows.append(dict(log=log[-6:], scene=scene[-4:], split=split, zone=zname, genre=int(al['mode'][idx[j]]),
                                 v_in=round(P['v_in']) if P['v_in'] else None,
                                 fwd_h=float(NEUTRAL - hum[j, 2, 1]), fwd_4=float(NEUTRAL - preds[4][j, 2, 1]),
                                 fwd_5=float(NEUTRAL - preds[5][j, 2, 1]), st_h=float(hum[j, 2, 0] - 1443),
                                 st_4=float(preds[4][j, 2, 0] - 1443), st_5=float(preds[5][j, 2, 0] - 1443)))
    print(log, len(rows), file=sys.stderr)

summ = {}
for split in ('train', 'val', 'all'):
    for z in ('R1', 'DE', 'REST'):
        for g in (4, 5):
            s = [r for r in rows if r['zone'] == z and r['genre'] == g and (split == 'all' or r['split'] == split)]
            if not s:
                continue
            summ['%s/%s/genre%d' % (split, z, g)] = {
                'n': len(s), 'fwd_human': round(float(np.mean([r['fwd_h'] for r in s])), 1),
                'fwd_pred_g4': round(float(np.mean([r['fwd_4'] for r in s])), 1),
                'fwd_pred_g5': round(float(np.mean([r['fwd_5'] for r in s])), 1),
                'steer_human': round(float(np.mean([r['st_h'] for r in s])), 1),
                'steer_pred_g4': round(float(np.mean([r['st_4'] for r in s])), 1),
                'steer_pred_g5': round(float(np.mean([r['st_5'] for r in s])), 1)}
# R1 手前: 人が速く入った通過(≥6200)と遅く入った通過(<5500)で分ける
for lab, lo, hi in (('fast_in', 6200, 1e9), ('slow_in', 0, 5500)):
    s = [r for r in rows if r['zone'] == 'R1' and r['v_in'] is not None and lo <= r['v_in'] < hi]
    if s:
        summ['R1_%s' % lab] = {'n': len(s), 'fwd_human': round(float(np.mean([r['fwd_h'] for r in s])), 1),
                               'fwd_pred_g4': round(float(np.mean([r['fwd_4'] for r in s])), 1),
                               'fwd_pred_g5': round(float(np.mean([r['fwd_5'] for r in s])), 1)}
json.dump({'summary': summ, 'rows': rows}, sys.stdout, ensure_ascii=False)
