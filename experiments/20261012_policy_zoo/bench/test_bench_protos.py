# -*- coding: utf-8 -*-
"""試作(protos.py・controller_proto.py)の CPU だけのテスト(30秒以内)

    cd bench && CUDA_VISIBLE_DEVICES= python -m pytest -q test_bench_protos.py
"""

import math
import os

import numpy as np
import pytest
import torch

import protos
from controller_proto import Calib, TCPController, TemporalEnsembler, pure_pursuit_curvature


def test_gru_cell_export_matches_torch():
    torch.manual_seed(0)
    ref = torch.nn.GRUCell(8, 16)
    mine = protos.GRUCellExport(8, 16)
    mine.ih.weight.data.copy_(ref.weight_ih)
    mine.ih.bias.data.copy_(ref.bias_ih)
    mine.hh.weight.data.copy_(ref.weight_hh)
    mine.hh.bias.data.copy_(ref.bias_hh)
    x, h = torch.randn(4, 8), torch.randn(4, 16)
    assert torch.allclose(ref(x, h), mine(x, h), atol=1e-6)


def test_select_mode_picks_one_genre():
    all_modes = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    mode = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    out = protos.select_mode(all_modes, mode)
    assert torch.equal(out[0], all_modes[0, 1]) and torch.equal(out[1], all_modes[1, 2])


@pytest.mark.parametrize('name', ['act', 'tcp', 'tcp_dec'])
def test_small_policies_output_contract(name):
    torch.manual_seed(0)
    m = protos.build(name).eval()
    with torch.no_grad():
        y = protos.outputs_of(m(*protos.example_inputs(batch=2)))
    assert tuple(y['actions'].shape) == (2, 10, 2)
    if name == 'tcp':
        assert tuple(y['waypoints'].shape) == (2, 10, 2) and tuple(y['target_speed'].shape) == (2, 1)
        assert (y['target_speed'] >= 0).all()
    if name == 'tcp_dec':
        assert tuple(y['waypoints'].shape) == (2, 10, 2) and tuple(y['speed_logits'].shape) == (2, 10)


def test_tcp_dec_cumsum_forms_agree():
    torch.manual_seed(0)
    a = protos.build('tcp_dec').eval()
    b = protos.build('tcp_dec', cumsum='matmul').eval()
    b.load_state_dict(a.state_dict())
    x = protos.example_inputs(batch=2)
    with torch.no_grad():
        assert torch.allclose(a(*x)['waypoints'], b(*x)['waypoints'], atol=1e-5)


def test_act_cvae_inference_uses_zero_latent():
    torch.manual_seed(0)
    m = protos.build('act_cvae').eval()
    x = protos.example_inputs(batch=2)
    with torch.no_grad():
        a = m(*x)
        b = m(*x, actions=torch.randn(2, 10, 2))       # 推論では正解の行動を渡しても z=0
    assert torch.allclose(a, b)


def test_msfusion_stage_grids_divide():
    for s in range(4):
        hi, wi = protos.IMG_STAGE_HW[s]
        ht, wt = protos.TOF_STAGE_HW[s]
        assert hi % 7 == 0 and wi % 7 == 0 and ht % 6 == 0 and wt % 7 == 0


@pytest.mark.skipif(not os.path.exists(protos.DINO_WEIGHTS), reason='DINOv2 の重みが無い')
def test_dinov2_loads_official_weights_and_runs():
    vit = protos.DinoViTS14(224).load_official()
    assert tuple(vit.pos_embed.shape) == (1, 257, 384)
    sd = torch.load(protos.DINO_WEIGHTS, map_location='cpu', weights_only=True)
    same = protos.DinoViTS14.interpolate_pos(sd['pos_embed'], 37)
    assert same is sd['pos_embed']                               # 37x37(518px)のときは補間しない
    assert torch.equal(vit.blocks[11].mlp.fc2.weight, sd['blocks.11.mlp.fc2.weight'])
    with torch.no_grad():
        out = vit(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 257, 384) and torch.isfinite(out).all()


def test_pure_pursuit_on_circle():
    r = 2.0                                                       # 半径 2m の左カーブ → κ = 0.5
    th = np.linspace(0.1, 1.0, 10)
    wp = np.stack([r * np.sin(th), r * (1 - np.cos(th))], 1)
    assert abs(pure_pursuit_curvature(wp, speed=2.0, t_look=0.5) - 1 / r) < 0.02
    straight = np.stack([np.linspace(0.2, 2.0, 10), np.zeros(10)], 1)
    assert pure_pursuit_curvature(straight, 2.0) == 0.0


def test_controller_conventions_and_speed_cap():
    calib = Calib(steer_center_us=1441, curv_per_us=0.004, curv_bias=0.0, throttle_neutral_us=1496, ff_a0=40, ff_a1=30)
    th = np.linspace(0.1, 1.0, 10)
    left = np.stack([2 * np.sin(th), 2 * (1 - np.cos(th))], 1)
    ctl = TCPController(calib)
    out, info = ctl.step(left, target_speed=3.0, erpm=-4000)       # eRPM 負 = 前進 2m/s
    assert out.shape == (10, 2) and abs(info['speed'] - 2.0) < 1e-6
    assert out[0, 0] > 1441                                       # 左 = 中心より大きい µs
    assert out[0, 1] < 1496                                       # 前進 = 中立より小さい µs
    capped = TCPController(calib)
    out_c, info_c = capped.step(left, target_speed=3.0, erpm=-4000, speed_cap=1.0)
    assert info_c['target_speed'] == 1.0 and out_c[0, 1] >= out[0, 1]
    stop = TCPController(calib)
    out_s, _ = stop.step(left, target_speed=0.0, erpm=-6000)       # 止まれ: ブレーキ・後退は出さず中立まで
    assert out_s[0, 1] == pytest.approx(1496)


def test_temporal_ensembler():
    te = TemporalEnsembler(m=0.0)
    p = np.tile(np.arange(10, dtype=np.float32)[:, None], (1, 2)) + 1500   # 時刻 0 の予測: ステップ s の値 1500+s
    out0 = te.push(0.0, p)
    assert np.allclose(out0, p)
    out1 = te.push(33333.0, p)              # 1ステップ後: 古い予測は 1 つずれて (1501+s)、新しい予測 (1500+s) と平均
    assert np.allclose(out1[:9, 0], (p[1:, 0] + p[:9, 0]) / 2)
    assert np.allclose(out1[9], p[9])
    out_gap = TemporalEnsembler(m=0.0)
    out_gap.push(0.0, p)
    o = out_gap.push(3 * 33333.0, p)        # フレームが2つ抜けても時刻で数える
    assert np.allclose(o[0, 0], (1503 + 1500) / 2)
    assert math.isfinite(float(o.sum()))
