#!/usr/bin/env python3
"""オフライン評価(protocol.json offline_eval / design.md 6.3 節): 学習した S-T・S-R の各種を、RL の val 12 シーンと
教師の val 12 シーンの全フレーム(future_valid・間引きなし・衝突 2.5G の前後は学習と同じく除く)で比べる(GPU、1 プロセス)

    cd ~/jetformer && jf-heavy --mem 2500M --lock big -- python -u experiments/20261013_sim_rl_bestlap/train/offline_eval.py

指標: L1 [µs]、舵・スロットル別、1 ステップ目(シミュレーターの走行で使う先読み 1)と 10 ステップの平均、ステップごと(1〜10)。
フレーム番号を 3 で割った余りごとの平均も出す(学習の検証は間引き 3 = 余り 0 のフレームだけを使う)。
モデルは runtime.load_run(best)。FP32(AMP なし)。1 つのデータの読み込みで全モデルを回す。合否には使わない(参考)。
結果 → /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/train/offline_l1.json
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
OUT = REPORT / 'train' / 'offline_l1.json'
STATE = REPORT / 'train' / 'state.json'
TEACHER = 'jetformer-simulator-id8-20min_20261012-150956_b55616cf'
PWM_SCALE = 500.0


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    import torch
    from torch.utils.data import DataLoader
    from jetformer.train_data import JetFormerDataset
    from jetformer.policies import runtime
    state = json.load(open(STATE, encoding='utf-8'))
    rl = state['rl_dataset']
    runs = [(k, v['run']) for k, v in state['runs'].items() if v.get('done')]
    if not runs:
        raise SystemExit('済んだ学習結果がありません')
    dev = torch.device('cuda')
    models = {}
    for key, name in runs:
        m, cfg, _, _ = runtime.load_run(REPO / 'data' / 'runs' / name, 'best', device=dev)
        m.eval()
        models[key] = (name, m)
    res = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'device': torch.cuda.get_device_name(0),
           'precision': 'fp32', 'models': {k: n for k, (n, _) in models.items()}, 'sets': {}}
    for set_name, ds_name in (('RL_val', rl), ('teacher_val', TEACHER)):
        ds = JetFormerDataset(str(REPO / 'data' / 'datasets' / ds_name), 'val', stride=1, use_command=False,
                              mask_impacts=True, impact_g=2.5)
        dl = DataLoader(ds, 64, shuffle=False, num_workers=2)
        err = {k: [] for k in models}
        labels = []
        t = time.time()
        with torch.no_grad():
            for batch in dl:
                inputs = {n: b.to(dev, non_blocking=True) for n, b in zip(('image', 'tof', 'imu', 'state', 'mode'), batch[:5])}
                target = batch[5].numpy()
                labels.append(target)
                for k, (_, m) in models.items():
                    out = runtime.forward(m, inputs)
                    act = out['actions'].float().cpu().numpy()
                    err[k].append(np.abs(act - target) * PWM_SCALE)
        lab = np.concatenate(labels) * PWM_SCALE + 1500.0
        residue = np.array([i % 3 for _, i in ds.index])          # 学習の検証(間引き 3)は residue 0 のフレームだけ
        entry = {'dataset': ds_name, 'frames': len(ds), 'seconds': round(time.time() - t, 1),
                 'label_steer_at_limit_frac': round(float(((lab[:, 0, 0] >= 1990) | (lab[:, 0, 0] <= 1010)).mean()), 4),
                 'models': {}}
        for k, chunks in err.items():
            e = np.concatenate(chunks)          # [N, 10, 2]
            entry['models'][k] = {
                'run': models[k][0],
                'steer_step1_us': round(float(e[:, 0, 0].mean()), 2), 'throttle_step1_us': round(float(e[:, 0, 1].mean()), 2),
                'steer_mean10_us': round(float(e[:, :, 0].mean()), 2), 'throttle_mean10_us': round(float(e[:, :, 1].mean()), 2),
                'both_mean10_us': round(float(e.mean()), 2),
                'steer_per_step_us': [round(float(x), 2) for x in e[:, :, 0].mean(0)],
                'throttle_per_step_us': [round(float(x), 2) for x in e[:, :, 1].mean(0)],
                'both_mean10_by_frame_mod3_us': [round(float(e[residue == r].mean()), 2) for r in range(3)],
                'steer_step1_p90_us': round(float(np.percentile(e[:, 0, 0], 90)), 1),
                'throttle_step1_p90_us': round(float(np.percentile(e[:, 0, 1], 90)), 1)}
        res['sets'][set_name] = entry
        print(set_name, json.dumps({k: (v['steer_step1_us'], v['throttle_step1_us'], v['both_mean10_us'])
                                    for k, v in entry['models'].items()}, ensure_ascii=False), flush=True)
    write_json(OUT, res)
    return 0


if __name__ == '__main__':
    sys.exit(main())
