#!/usr/bin/env python3
"""Jetson で収集した1シーン(元の RC 教師・差し替え確認用)と、取り込み済みの20分データの同じシーンを比べる

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/compare_probe_scene.py <収集したデータセットのフォルダー> [元のシーン名]

  - PWM(vehicle)・未来の操作(future_actions)・IMU・時刻の差 = 物理と教師が同じか
  - RGB(JPEG)・ToF の差 = 描画(Jetson の Mesa llvmpipe と、作者の Mac の GPU)の違い
  結果は /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/webgl_probe/compare_<シーン>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ORIG = Path('/home/jetson/jetformer/data/datasets/jetformer-simulator-id8-20min_20261012-150956_b55616cf')
OUT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/webgl_probe')


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    new_root = Path(sys.argv[1])
    scene0 = sys.argv[2] if len(sys.argv) > 2 else '20261012-145130-469770cc-0001'
    new_npz = sorted((new_root / 'aligned').glob('*.npz'))
    assert len(new_npz) == 1, new_npz
    a, b = np.load(new_npz[0]), np.load(ORIG / 'aligned' / f'{scene0}.npz')
    res = {'new': str(new_npz[0]), 'orig_scene': scene0, 'keys_equal': sorted(a.files) == sorted(b.files)}
    for k in ('vehicle', 'future_actions', 'imu', 'future_valid', 'vehicle_valid', 'mode', 'human_control', 'synthetic'):
        x, y = a[k], b[k]
        res[k] = {'shape': list(x.shape), 'dtype': str(x.dtype), 'same_shape': x.shape == y.shape,
                  'max_abs_diff': float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.shape == y.shape else None}
    rel = lambda t: (t - t[0]).astype(np.int64)  # noqa: E731
    res['timestamps_rel_equal'] = bool(np.array_equal(rel(a['timestamps_us']), rel(b['timestamps_us'])))
    tof_a, tof_b = a['tof'].astype(np.int32), b['tof'].astype(np.int32)
    da, db = tof_a & 0x1FFF, tof_b & 0x1FFF
    both = (da > 0) & (db > 0) & (da < 8191) & (db < 8191)
    res['tof'] = {'confidence_bits_equal_frac': float(np.mean((tof_a >> 13) == (tof_b >> 13))),
                  'valid_both_frac': float(both.mean()),
                  'distance_abs_diff_mm_median': float(np.median(np.abs(da - db)[both])),
                  'distance_abs_diff_mm_p99': float(np.percentile(np.abs(da - db)[both], 99)),
                  'invalid_mismatch_frac': float(np.mean((da == 0) != (db == 0)))}
    ps = []
    for i in range(0, len(a['cam_files']), 20):
        ia = cv2.imread(str(new_root / str(a['cam_files'][i])))
        ib = cv2.imread(str(ORIG / str(b['cam_files'][i])))
        ps.append(psnr(ia, ib))
        if i in (0, 300):
            cv2.imwrite(str(OUT / f'compare_frame{i:03d}.png'), np.concatenate([ia, ib], axis=1))
    res['rgb_psnr_db'] = {'frames': len(ps), 'min': min(ps), 'median': float(np.median(ps)), 'max': max(ps)}
    # 学習の正解(PWM)の範囲: 中心 1441µs / 中立 1496µs、最大操舵 0.52 rad、最大速度 7.24 m/s(デモの収集は校正なし)
    v = a['vehicle']
    res['pwm_range'] = {'steer_min': float(v[:, 0].min()), 'steer_max': float(v[:, 0].max()),
                        'throttle_min': float(v[:, 1].min()), 'throttle_max': float(v[:, 1].max()),
                        'ch3': sorted(set(v[:, 2].tolist())), 'erpm_min': float(v[:, 3].min())}
    sess = json.loads((new_root / 'simulation/session.json').read_text())
    res['session'] = {'controller': sess['controller'], 'label_mode': sess['training_label_mode'],
                      'calibration': {k: sess['calibration'][k] for k in ('max_speed_mps', 'max_steering_rad', 'pwm_min_us', 'pwm_max_us', 'steering_sign')},
                      'model_config': sess['model_config'], 'teacher_id': sess['context'].get('teacher', {}).get('id')}
    out = OUT / f'compare_{new_root.name}.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
