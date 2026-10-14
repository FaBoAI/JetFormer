#!/usr/bin/env python3
"""オフライン評価(参考。合否には使わない): A0(S-R 種 0)・A1・A2 の best を、3 つのデータの val の全フレームで比べる(GPU、1 プロセス)

    cd ~/jetformer && jf-heavy --mem 2500M --lock big -- python -u experiments/20261014_sim_dart_margin/train/offline_eval.py

ベストラップの offline_eval.py の写しを、この作業のデータに合わせて直したもの。
  A0_val  最速の線のきれいなデータ(A0)の val 12 シーン
  D1_val  DART のデータ(D1)の val 6 シーン。正解は先生の操作に付け替え済み。さらに
            changed   = 付け替えで 1 ステップ目の正解が 1 µs より大きく変わったフレーム(線から外れた状態・乱しの最中)
            unchanged = それ以外
          に分け、changed では「付け替える前の正解(記録した乱し入りの操作)」との L1 も出す(どちらに近いか)
  D2_val  余裕を持たせた線(D2)の val 6 シーン
フレームは future_valid・vehicle_valid・衝突 2.5G の前後を除く(学習と同じ)。間引きなし。FP32(AMP なし)。
指標: L1 [µs]、舵・スロットル別、1 ステップ目(シミュレーターの走行で使う先読み 1)と 10 ステップの平均。
結果 → /home/jetson/jetformer_reports/sim_dart_20261014/train/offline_l1.json
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
OUT = REPORT / 'train' / 'offline_l1.json'
STATE = REPORT / 'train' / 'state.json'
READY = REPORT / 'recording' / 'READY.json'
BACKUP = REPO / 'experiments' / '20261014_sim_dart_margin' / 'relabel' / 'backup'
PWM_SCALE = 500.0
CHANGED_US = 1.0
MODELS = [('A0/s0', 'simrl_bestlap_20261013-SR-s0')]       # A0 = ベストラップの S-R 種 0(作り直さない)


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def stats(e):
    """e: [N, 10, 2] の |予測 − 正解| [µs]"""
    if len(e) == 0:
        return None
    return {'frames': int(len(e)),
            'steer_step1_us': round(float(e[:, 0, 0].mean()), 2), 'throttle_step1_us': round(float(e[:, 0, 1].mean()), 2),
            'steer_mean10_us': round(float(e[:, :, 0].mean()), 2), 'throttle_mean10_us': round(float(e[:, :, 1].mean()), 2),
            'both_mean10_us': round(float(e.mean()), 2),
            'steer_step1_p90_us': round(float(np.percentile(e[:, 0, 0], 90)), 1),
            'steer_per_step_us': [round(float(x), 2) for x in e[:, :, 0].mean(0)],
            'throttle_per_step_us': [round(float(x), 2) for x in e[:, :, 1].mean(0)]}


def main():
    import torch
    from torch.utils.data import DataLoader
    from jetformer.train_data import JetFormerDataset
    from jetformer.policies import runtime
    state = json.load(open(STATE, encoding='utf-8'))
    ready = json.load(open(READY, encoding='utf-8'))
    runs = list(MODELS) + [(k, v['run']) for k, v in state['runs'].items()
                           if (REPO / 'data' / 'runs' / v['run'] / 'result.json').exists()
                           and json.load(open(REPO / 'data' / 'runs' / v['run'] / 'result.json')).get('state') == 'done']
    dev = torch.device('cuda')
    models = {}
    for key, name in runs:
        m, cfg, _, _ = runtime.load_run(REPO / 'data' / 'runs' / name, 'best', device=dev)
        m.eval()
        models[key] = (name, m)
    res = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'device': torch.cuda.get_device_name(0), 'precision': 'fp32',
           'models': {k: n for k, (n, _) in models.items()}, 'changed_threshold_us': CHANGED_US, 'sets': {}}
    sets = (('A0_val', ready['A0']['name']), ('D1_val', ready['D1']['name']), ('D2_val', ready['D2']['name']))
    for set_name, ds_name in sets:
        root = REPO / 'data' / 'datasets' / ds_name
        ds = JetFormerDataset(str(root), 'val', stride=1, use_command=False, mask_impacts=True, impact_g=2.5)
        val_names = json.load(open(root / 'splits.json', encoding='utf-8'))['val']
        if len(ds.scenes) != len(val_names):
            raise SystemExit('%s: 読み込んだシーンの数が splits と違います' % ds_name)
        old = None
        if set_name == 'D1_val':           # 付け替える前の正解(記録した乱し入りの操作から収集器が作った値)
            old = [np.load(BACKUP / ds_name / (s + '.npz'))['future_actions'] for s in val_names]
        dl = DataLoader(ds, 64, shuffle=False, num_workers=1)
        preds = {k: [] for k in models}
        labels = []
        t = time.time()
        with torch.no_grad():
            for batch in dl:
                inputs = {n: b.to(dev, non_blocking=True) for n, b in zip(('image', 'tof', 'imu', 'state', 'mode'), batch[:5])}
                labels.append(batch[5].numpy())
                for k, (_, m) in models.items():
                    preds[k].append(runtime.forward(m, inputs)['actions'].float().cpu().numpy())
        lab = np.concatenate(labels)                                   # 正規化([µs − 1500] / 500)
        entry = {'dataset': ds_name, 'frames': len(ds), 'scenes': len(val_names), 'seconds': None, 'models': {}}
        sel = {'all': np.ones(len(ds), bool)}
        lab_old = None
        if old is not None:
            lab_old = np.stack([(old[k][i] - 1500.0) / PWM_SCALE for k, i in ds.index]).astype(np.float32)
            d1 = np.abs(lab[:, 0, :] - lab_old[:, 0, :]).max(1) * PWM_SCALE
            sel['changed'] = d1 > CHANGED_US
            sel['unchanged'] = ~sel['changed']
            entry['changed_frames'] = int(sel['changed'].sum())
            entry['label_change_step1_steer_us_mean_on_changed'] = round(
                float(np.abs(lab[sel['changed'], 0, 0] - lab_old[sel['changed'], 0, 0]).mean() * PWM_SCALE), 2)
        for k in models:
            p = np.concatenate(preds[k])
            e = np.abs(p - lab) * PWM_SCALE
            row = {'run': models[k][0]}
            for sname, mask in sel.items():
                row[sname] = stats(e[mask])
            if lab_old is not None:
                ch = sel['changed']
                e_old = np.abs(p[ch] - lab_old[ch]) * PWM_SCALE
                row['changed_vs_old_label'] = stats(e_old)
                # 予測の 1 ステップ目の舵が「付け替えた正解」と「付け替える前の正解」のどちらに近いか
                closer_new = np.abs(p[ch, 0, 0] - lab[ch, 0, 0]) < np.abs(p[ch, 0, 0] - lab_old[ch, 0, 0])
                row['changed_steer_step1_closer_to_teacher_frac'] = round(float(closer_new.mean()), 4)
            entry['models'][k] = row
        entry['seconds'] = round(time.time() - t, 1)
        res['sets'][set_name] = entry
        print(set_name, json.dumps({k: (v['all']['steer_step1_us'], v['all']['both_mean10_us'],
                                        (v.get('changed') or {}).get('steer_step1_us'))
                                    for k, v in entry['models'].items()}, ensure_ascii=False), flush=True)
        write_json(OUT, res)           # データごとに保存(途中で止まっても前の結果は残る)
    res['done'] = True
    write_json(OUT, res)
    return 0


if __name__ == '__main__':
    sys.exit(main())
