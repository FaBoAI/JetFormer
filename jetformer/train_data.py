"""学習用データローダー(エクスポートしたデータセットの aligned/*.npz と画像を読む)

    ds = JetFormerDataset('data/datasets/indoor_v1', split='train')
    image, tof, imu, state, mode, target = ds[0]      # mode は走行ジャンルのone-hot
"""

import json
import os

import cv2
import numpy as np
import torch

from . import tof_format
from .model import ERPM_SCALE, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, PWM_CENTER, PWM_SCALE, TOF_MAX_MM

UNSET_MODE = 0

GRAVITY = 9.80665


def normalize_inputs(image_bgr, tof_raw, imu, vehicle):
    """推論時(Jetson)と学習時で共通の正規化。numpy配列 → モデル入力(float32)

    tof_raw: ToFの生の値 [42,54] / vehicle: [steering_us, throttle_us, mode_us, erpm]"""
    img = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    tof = np.stack([np.clip(tof_format.distance_mm(tof_raw) / TOF_MAX_MM, 0.0, 1.0),
                    tof_format.confidence(tof_raw).astype(np.float32) / tof_format.CONF_MAX])
    imu_n = imu.astype(np.float32).copy()
    imu_n[:, :3] /= GRAVITY
    state = np.array([(vehicle[0] - PWM_CENTER) / PWM_SCALE, (vehicle[1] - PWM_CENTER) / PWM_SCALE,
                      np.clip(vehicle[3] / ERPM_SCALE, -3.0, 3.0)], dtype=np.float32)
    return img.astype(np.float32), tof, imu_n, state


class JetFormerDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', color_jitter=False, include_unset=False, stride=1):
        self.root = root
        self.color_jitter = color_jitter
        with open(os.path.join(root, 'dataset_card.json'), encoding='utf-8') as f:
            card = json.load(f)
        self.modes = card.get('modes', [{'id': 0, 'name': '未設定'}])
        self.num_modes = max(m['id'] for m in self.modes) + 1
        with open(os.path.join(root, 'splits.json')) as f:
            scenes = json.load(f)[split]
        self.scenes = []
        self.index = []
        for s in scenes:
            a = dict(np.load(os.path.join(root, 'aligned', s + '.npz')))
            if 'vehicle' not in a or 'tof' not in a or 'imu' not in a:
                continue
            ok = a['future_valid'] & a['vehicle_valid']
            if 'mode' not in a:
                a['mode'] = np.zeros(len(ok), dtype=np.int16)
            if not include_unset:
                ok &= a['mode'] != UNSET_MODE          # ジャンル未設定のフレームは学習に使わない
            if stride > 1:                              # 間引き: 30fpsの連続フレームは似ているので N枚に1枚
                ok &= (np.arange(len(ok)) % stride) == 0
            k = len(self.scenes)
            self.scenes.append(a)
            self.index.extend((k, int(i)) for i in np.nonzero(ok)[0])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        k, i = self.index[n]
        a = self.scenes[k]
        image = cv2.imread(os.path.join(self.root, str(a['cam_files'][i])))
        if self.color_jitter:
            gain = np.random.uniform(0.8, 1.2, size=3)
            image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        img, tof, imu, state = normalize_inputs(image, a['tof'][i], a['imu'][i], a['vehicle'][i])
        target = (a['future_actions'][i] - PWM_CENTER) / PWM_SCALE
        mode = np.zeros(self.num_modes, dtype=np.float32)
        mode[int(a['mode'][i])] = 1.0
        return (torch.from_numpy(img), torch.from_numpy(tof), torch.from_numpy(imu), torch.from_numpy(state),
                torch.from_numpy(mode), torch.from_numpy(target.astype(np.float32)))

    def mode_counts(self):
        counts = {}
        for k, i in self.index:
            m = int(self.scenes[k]['mode'][i])
            counts[m] = counts.get(m, 0) + 1
        return counts

    def action_reference(self):
        """(ステアリング中心, スロットル中立) [µs]: 止まっている間(eRPM=0)の人の操作値の中央値。
        走行時のゲインはこの値を中心に掛ける。止まっているフレームが少なければ PWM_CENTER"""
        rows = [a['vehicle'][a['vehicle_valid'] & a.get('erpm_valid', True) & (np.abs(a['vehicle'][:, 3]) < 1)]
                for a in self.scenes]           # 回転数が無いデータ(旧ファーム)は止まっているか分からないので除く
        rows = np.concatenate(rows) if rows else np.zeros((0, 4), np.float32)
        if len(rows) < 10:
            return float(PWM_CENTER), float(PWM_CENTER)
        return round(float(np.median(rows[:, 0])), 1), round(float(np.median(rows[:, 1])), 1)
