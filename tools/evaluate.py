#!/usr/bin/env python3
"""データセットと学習結果の評価(レポートを data/reports/<データセット>/ に作る)

    python tools/evaluate.py --dataset lowspeed_20261010                        # データの品質だけ
    python tools/evaluate.py --dataset lowspeed_20261010 --run <学習結果の名前>  # + モデルの予測の評価

データの品質(シーンごと): フレームの欠け・時刻のずれ・センサーのレート・操作の分布・ステアリングとヨーレートの
向きの関係(チャネルの取り違えがないか)・衝撃。
モデルの評価(検証用シーン): 1〜10フレーム先の [ステアリング, スロットル] の誤差を、
  ・教師あり(現在の操作値に人の値を入れる。学習と同じ条件)
  ・自己入力(現在の操作値に前フレームの予測を入れる。AI走行でCH3をAI側にしたときと同じ条件)
で測り、「今の操作値をそのまま続ける」「学習データの平均」と比べる。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import config, tof_format  # noqa: E402

FPS = 30.0


def _font():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    return plt


# ------------------------------------------------------------ データの品質
def scene_quality(root, scene):
    a = dict(np.load(root / 'aligned' / (scene + '.npz')))
    out = {'scene': scene, 'frames': int(len(a['timestamps_us']))}
    t = a['timestamps_us'] / 1e6
    gaps = np.diff(t) * 1000
    out['duration_s'] = round(float(t[-1] - t[0]), 2)
    out['camera_fps'] = round(float((len(t) - 1) / (t[-1] - t[0])), 2)
    out['camera_dropped'] = int(np.sum(np.round(gaps / (1000 / FPS)) - 1))
    out['camera_max_gap_ms'] = round(float(gaps.max()), 1)
    out['tof_age_ms'] = {'mean': round(float(a['tof_age_ms'].mean()), 1), 'max': round(float(a['tof_age_ms'].max()), 1)}
    if 'vehicle_age_ms' in a:
        out['vehicle_age_ms'] = {'mean': round(float(a['vehicle_age_ms'].mean()), 1), 'max': round(float(a['vehicle_age_ms'].max()), 1)}
    usable = a['vehicle_valid'] & a['future_valid']
    out['usable_frames'] = int(usable.sum())
    v = a['vehicle']
    steer, thr, erpm = v[:, 0], v[:, 1], v[:, 3]
    out['steering_us'] = {k: round(float(np.percentile(steer, p)), 0) for k, p in (('p5', 5), ('p50', 50), ('p95', 95))}
    out['throttle_us'] = {k: round(float(np.percentile(thr, p)), 0) for k, p in (('p5', 5), ('p50', 50), ('p95', 95))}
    out['moving_ratio'] = round(float(np.mean(np.abs(erpm) > 0)), 3)
    out['erpm'] = {'min': int(erpm.min()), 'max': int(erpm.max()), 'median_moving': int(np.median(erpm[np.abs(erpm) > 0])) if (np.abs(erpm) > 0).any() else 0}
    d = tof_format.distance_mm(a['tof'])
    valid = d > 0
    out['tof_invalid_ratio'] = round(float(1 - valid.mean()), 3)
    out['tof_median_mm'] = int(np.median(d[valid])) if valid.any() else 0
    out['tof_near_ratio'] = round(float(np.mean(d[valid] < 300)), 3) if valid.any() else 0.0   # 30cm未満の割合
    # ステアリングとヨーレートの向き(動いているとき): 符号の関係がシーン間でそろっていれば取り違えはない
    gz = a['imu'][:, -1, 5]                             # 直近のIMUの z 角速度 [rad/s]
    m = (np.abs(erpm) > 300) & a['vehicle_valid']
    if m.sum() > 30:
        c = np.corrcoef(steer[m] - np.median(steer), gz[m] * np.sign(erpm[m]))[0, 1]
        out['steer_yaw_corr'] = round(float(c), 2)
    acc = np.linalg.norm(a['imu'][:, -1, :3], axis=1)
    out['accel_peak_ms2'] = round(float(np.abs(acc - 9.80665).max()), 1)
    out['impacts'] = int(np.sum(np.abs(acc - 9.80665) > 20))           # 2g超の衝撃(フレーム数)
    return out, a


def data_report(name, out_dir):
    root = config.DATASET_DIR / name
    splits = json.load(open(root / 'splits.json'))
    rows, arrays = [], {}
    for split in ('train', 'val'):
        for s in splits.get(split, []):
            q, a = scene_quality(root, s)
            q['split'] = split
            rows.append(q)
            arrays[s] = a
    plt = _font()
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), squeeze=False)
    for ax, q in zip(axes[:, 0], rows):
        a = arrays[q['scene']]
        t = (a['timestamps_us'] - a['timestamps_us'][0]) / 1e6
        ax.plot(t, a['vehicle'][:, 0], color='#2563eb', lw=1.2, label='ステアリング µs')
        ax.plot(t, a['vehicle'][:, 1], color='#dc2626', lw=1.2, label='スロットル µs')
        ax.set_ylim(900, 2100)
        ax2 = ax.twinx()
        ax2.plot(t, a['vehicle'][:, 3], color='#16a34a', lw=0.8, alpha=.6, label='eRPM')
        ax2.set_ylabel('eRPM', color='#16a34a')
        ax.set_title('%s(%s)フレーム %d・欠け %d・衝撃 %d' % (q['scene'], '学習' if q['split'] == 'train' else '検証',
                                                            q['frames'], q['camera_dropped'], q['impacts']), fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
    axes[-1, 0].set_xlabel('秒')
    fig.tight_layout()
    fig.savefig(out_dir / 'data_timeline.png', dpi=90)
    plt.close(fig)
    return rows, arrays


# ------------------------------------------------------------ モデルの評価
def model_report(name, run, arrays, out_dir):
    import torch
    from jetformer.model import JetFormer, PWM_CENTER, PWM_SCALE
    from jetformer.train_data import normalize_inputs
    rdir = config.DATA_DIR / 'runs' / run
    cfg = json.load(open(rdir / 'model_config.json'))
    model = JetFormer(num_modes=cfg['num_modes']).cuda().eval()
    model.load_state_dict(torch.load(rdir / 'best.pth', map_location='cuda', weights_only=True))
    use_command = cfg.get('command_input', True)
    root = config.DATASET_DIR / name
    splits = json.load(open(root / 'splits.json'))
    train_actions = np.concatenate([arrays[s]['future_actions'][arrays[s]['vehicle_valid'] & arrays[s]['future_valid']]
                                    for s in splits['train']])
    mean_action = train_actions.reshape(-1, 2).mean(axis=0)
    import cv2
    res = {}
    for scene in splits['val']:
        a = arrays[scene]
        n = len(a['timestamps_us'])
        ok = a['vehicle_valid'] & a['future_valid']
        preds_tf, preds_sf = np.full((n, 10, 2), np.nan), np.full((n, 10, 2), np.nan)
        prev = None
        with torch.no_grad():
            for i in range(n):
                img = cv2.imread(str(root / str(a['cam_files'][i])))
                mode = torch.zeros(1, cfg['num_modes'])
                mode[0, int(a['mode'][i])] = 1
                for kind in ('tf', 'sf'):
                    veh = a['vehicle'][i].copy()
                    if kind == 'sf' and prev is not None:          # 自己入力: 前フレームの予測(次の時刻に当たる値)
                        veh[0], veh[1] = prev
                    im, tof, imu, st = normalize_inputs(img, a['tof'][i], a['imu'][i], veh, use_command)
                    out = model(torch.from_numpy(im)[None].cuda(), torch.from_numpy(tof)[None].cuda(),
                                torch.from_numpy(imu)[None].cuda(), torch.from_numpy(st)[None].cuda(), mode.cuda())
                    p = out[0].float().cpu().numpy() * PWM_SCALE + PWM_CENTER
                    (preds_tf if kind == 'tf' else preds_sf)[i] = p
                    if kind == 'sf':
                        prev = (float(p[0, 0]), float(p[0, 1]))
        y = a['future_actions'][ok]
        cur = np.repeat(a['vehicle'][ok][:, None, :2], 10, axis=1)          # 今の操作値を続ける
        mean = np.broadcast_to(mean_action, y.shape)

        def mae(p):
            return np.abs(p - y).mean(axis=0)                              # [10,2]
        m = {'model_teacher': mae(preds_tf[ok]), 'model_self': mae(preds_sf[ok]), 'hold_current': mae(cur), 'train_mean': mae(mean)}
        res[scene] = {k: {'steer_us': v[:, 0].round(1).tolist(), 'throttle_us': v[:, 1].round(1).tolist()} for k, v in m.items()}
        res[scene]['frames'] = int(ok.sum())
        # グラフ: 予測と実際(1フレーム先・10フレーム先)
        plt = _font()
        t = (a['timestamps_us'] - a['timestamps_us'][0]) / 1e6
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        for ax, j, label in ((axes[0], 0, 'ステアリング'), (axes[1], 1, 'スロットル')):
            ax.plot(t, a['future_actions'][:, 0, j], color='#111827', lw=1.6, label='実際(人の操作、1フレーム先)')
            ax.plot(t, preds_tf[:, 0, j], color='#2563eb', lw=1, label='予測 1フレーム先(教師あり)')
            ax.plot(t, preds_sf[:, 0, j], color='#dc2626', lw=1, alpha=.8, label='予測 1フレーム先(自己入力=AI走行と同じ)')
            ax.plot(t, preds_tf[:, 9, j], color='#16a34a', lw=1, alpha=.7, label='予測 10フレーム先(教師あり)')
            ax.set_ylabel('%s [µs]' % label)
            ax.legend(fontsize=8, loc='upper right')
        k = np.arange(1, 11)
        for key, color, label in (('model_teacher', '#2563eb', 'モデル(教師あり)'), ('model_self', '#dc2626', 'モデル(自己入力)'),
                                  ('hold_current', '#6b7280', '今の操作を続ける'), ('train_mean', '#d97706', '学習データの平均')):
            axes[2].plot(k, m[key][:, 0], color=color, marker='o', label=label + ' ステア')
            axes[2].plot(k, m[key][:, 1], color=color, marker='x', ls='--', label=label + ' スロットル')
        axes[2].set_xlabel('何フレーム先か(1フレーム=33ms)')
        axes[2].set_ylabel('平均絶対誤差 [µs]')
        axes[2].legend(fontsize=7, ncol=2)
        axes[0].set_title('検証シーン %s: 予測と実際' % scene)
        fig.tight_layout()
        fig.savefig(out_dir / ('model_%s.png' % scene), dpi=90)
        plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--run', default=None, help='評価する学習結果(data/runs/ の名前)')
    args = ap.parse_args()
    out_dir = config.DATA_DIR / 'reports' / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, arrays = data_report(args.dataset, out_dir)
    report = {'dataset': args.dataset, 'scenes': rows}
    if args.run:
        report['run'] = args.run
        report['model'] = model_report(args.dataset, args.run, arrays, out_dir)
    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print('レポート: %s' % out_dir)


if __name__ == '__main__':
    main()
