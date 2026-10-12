"""新しい方策が共有する部品(torch)

act・tcp は v0 と同じトークン化を使い、state_dict のキーも v0 と同じにする(将来、v0 の重みから部分的に始められるように)。
そのため、トークン化は「子のモジュール」ではなく、呼んだモジュールに v0 と同じ名前の層を直接足す関数にする:

    class MyPolicy(nn.Module):
        def __init__(self, num_modes, dim=256, pretrained=False):
            super().__init__()
            blocks.add_sensor_tokens(self, dim, num_modes, pretrained)     # image_backbone・image_proj・tof_encoder・...
        def forward(self, image, tof, imu, state, mode):
            tok = blocks.sensor_tokens(self, image, tof, imu, state, mode)  # [B,94,dim]
"""

import torch
import torch.nn as nn
import torchvision

from ..model import IMU_STEPS, ToFEncoder

N_SENSOR_TOKENS = 49 + 42 + 3          # 画像 49(7x7)・ToF 42(6x7)・IMU 1・操作値 1・ジャンル 1


def select_mode(all_modes, mode):
    """ジャンル別の出力 [B,M,...] から one-hot の mode [B,M] で1つ選ぶ → [B,...](掛け算と和だけ。TensorRT で特別な処理なし)"""
    shape = list(mode.shape) + [1] * (all_modes.dim() - 2)
    return (all_modes * mode.view(shape)).sum(dim=1)


def mlp2(cin, dim):
    """Linear → ReLU → Linear(v0 の imu_mlp・state_mlp と同じ)"""
    return nn.Sequential(nn.Linear(cin, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))


def encoder(dim, heads, ffn, layers, dropout=0.1):
    """前置き LayerNorm・batch_first の Encoder(norm_first では nested tensor を使えないので切る)"""
    layer = nn.TransformerEncoderLayer(dim, heads, ffn, dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def decoder(dim, heads, ffn, layers, dropout=0.1):
    """前置き LayerNorm・batch_first の Decoder(自己注意 + memory への交差注意)"""
    layer = nn.TransformerDecoderLayer(dim, heads, ffn, dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerDecoder(layer, layers)


def add_sensor_tokens(module, dim, num_modes, pretrained):
    """module に v0 と同じ名前・同じ初期化(同じ順)で image_backbone(ResNet-18 の children()[:-2])・image_proj
    (Conv 1x1 512→dim)・tof_encoder(model.ToFEncoder)・imu_mlp・state_mlp・mode_embed(Linear(num_modes, dim, bias=False)、
    normal std 0.02)を足す。pretrained=True のときだけ ImageNet の重み(torchvision のキャッシュ)を読む"""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = torchvision.models.resnet18(weights=weights)
    module.image_backbone = nn.Sequential(*list(resnet.children())[:-2])      # [B,512,7,7]
    module.image_proj = nn.Conv2d(512, dim, 1)
    module.tof_encoder = ToFEncoder(dim)
    module.imu_mlp = mlp2(IMU_STEPS * 6, dim)
    module.state_mlp = mlp2(3, dim)
    module.num_modes = num_modes
    module.mode_embed = nn.Linear(num_modes, dim, bias=False)               # one-hot → ジャンル埋め込み
    nn.init.normal_(module.mode_embed.weight, std=0.02)


def sensor_tokens(module, image, tof, imu, state, mode):
    """v0 の forward と同じ計算 → [B,94,dim](画像 49・ToF 42・IMU 1・操作値 1・ジャンル 1。位置の埋め込みは呼ぶ側で)"""
    img = module.image_proj(module.image_backbone(image)).flatten(2).transpose(1, 2)    # [B,49,D]
    tof_t = module.tof_encoder(tof).flatten(2).transpose(1, 2)                         # [B,42,D]
    imu_t = module.imu_mlp(imu.flatten(1)).unsqueeze(1)
    st_t = module.state_mlp(state).unsqueeze(1)
    mode_t = module.mode_embed(mode).unsqueeze(1)
    return torch.cat([img, tof_t, imu_t, st_t, mode_t], dim=1)
