#!/usr/bin/env python3
"""ens_gate.py の平均のモジュールのテスト(CPU・小さな作り物のモデル)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q experiments/20261015_real_accuracy/ens/test_ens_gate.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ens_gate as G  # noqa: E402


class Fake(torch.nn.Module):
    def __init__(self, k, as_dict):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((1,), float(k)))
        self.as_dict = as_dict

    def forward(self, image, tof, imu, state, mode):
        a = (image.mean(dim=(1, 2, 3)) * self.w + mode.sum(1)).view(-1, 1, 1).expand(-1, 10, 2) + tof.mean() * 0
        return {'actions': a, 'waypoints': a * 2} if self.as_dict else a


def test_average_module_is_mean_of_members():
    ms = [Fake(1.0, True), Fake(3.0, False)]
    avg = G.make_average_module(ms)
    x = dict(image=torch.randn(4, 3, 8, 8), tof=torch.rand(4, 2, 4, 5), imu=torch.zeros(4, 10, 6), state=torch.zeros(4, 3),
             mode=torch.eye(6)[:4])
    with torch.no_grad():
        out = avg(**x).numpy()
        a = ms[0](**x)['actions'].numpy()
        b = ms[1](**x).numpy()
    assert out.shape == (4, 10, 2)
    assert np.allclose(out, (a + b) / 2)


def test_gate_thresholds_match_protocol():
    import json
    p = json.load(open('/home/jetson/jetformer_reports/real_accuracy_20261015/protocol.json'))['arms']['ENS_GATE']['gates']
    assert G.GATES['actions_max_us'] == p['accuracy_max_us']
    assert G.GATES['run_numpy_median_ms'] == p['run_numpy_median_ms']
    assert G.GATES['run_numpy_p99_ms'] == p['run_numpy_p99_ms']
    assert G.PAIR == tuple(json.load(open('/home/jetson/jetformer_reports/real_accuracy_20261015/protocol.json'))
                           ['arms']['ENS']['pairs_fixed_before_training']['P2'])
