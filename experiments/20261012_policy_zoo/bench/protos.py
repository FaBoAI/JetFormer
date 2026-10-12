# -*- coding: utf-8 -*-
"""方策の試作(TensorRT に載るか・速さ・メモリを測るためだけのもの。リポジトリには入れない)

入力は今の走行と同じ5つ(バッチ1・固定の形で書き出す):
    image [B,3,224,224] ImageNet 正規化 / tof [B,2,42,54] / imu [B,10,6] / state [B,3] / mode [B,num_modes] one-hot
出力: actions [B,10,2](正規化した操作 (µs-1500)/500)。tcp だけ waypoints [B,10,2](m)・target_speed [B,1](m/s)も出す。

重みは乱数(計測の bench_infer.py では ResNet-18 だけ ImageNet の重みにし、BatchNorm の統計を実データで測り直す)。
DINOv2 の画像の部分は公式の重み data/pretrained/dinov2_vits14_pretrain.pth。
    v0        JetFormer v0(リポジトリの jetformer.model.JetFormer そのもの。基準)
    act       ACT 型: v0 と同じトークン → Encoder → 10個の行動クエリの Decoder → クエリごとのジャンル別ヘッド
    tcp       TCP / TransFuser++ 型: v0 と同じ Encoder + 経路(10点・0.1秒おき・GRU)+ 目標速度 + 直接の操作の枝
    msfusion  TransFuser 型の多段融合: ResNet-18 の4段と ToF の CNN の4段を、各段で小さな Transformer で融合
    dinov2    DINOv2 ViT-S/14(凍結)の画像トークン + v0 の ToF・IMU・操作値・ジャンルのトークン → v0 型の Encoder
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

REPO = '/home/jetson/jetformer'
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from jetformer.model import FUTURE_STEPS, IMU_STEPS, TOF_CHANNELS, TOF_SHAPE, JetFormer, ToFEncoder  # noqa: E402

DINO_WEIGHTS = os.path.join(REPO, 'data', 'pretrained', 'dinov2_vits14_pretrain.pth')


# ---------------------------------------------------------------- 共通の部品

def select_mode(all_modes, mode):
    """ジャンル別の出力 [B,M,...] から one-hot の mode [B,M] で1つ選ぶ(TensorRT でも掛け算と和だけ)"""
    shape = list(mode.shape) + [1] * (all_modes.dim() - 2)
    return (all_modes * mode.view(shape)).sum(dim=1)


def mlp2(cin, dim):
    return nn.Sequential(nn.Linear(cin, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))


def resnet18(pretrained=False):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=weights)


def encoder(dim, heads, ffn, layers, dropout=0.1):
    """v0 と同じ前置き LayerNorm の Encoder(enable_nested_tensor は norm_first では使えないので切る)"""
    layer = nn.TransformerEncoderLayer(dim, heads, ffn, dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class SensorTokens(nn.Module):
    """v0 と同じトークン化: 画像 49(ResNet-18 → 7x7)・ToF 42(6x7)・IMU 1・操作値 1・ジャンル 1 → [B,94,D]"""

    def __init__(self, dim, num_modes, pretrained=False):
        super().__init__()
        r = resnet18(pretrained)
        self.image_backbone = nn.Sequential(*list(r.children())[:-2])
        self.image_proj = nn.Conv2d(512, dim, 1)
        self.tof_encoder = ToFEncoder(dim)
        self.imu_mlp = mlp2(IMU_STEPS * 6, dim)
        self.state_mlp = mlp2(3, dim)
        self.mode_embed = nn.Linear(num_modes, dim, bias=False)
        nn.init.normal_(self.mode_embed.weight, std=0.02)

    def forward(self, image, tof, imu, state, mode):
        img = self.image_proj(self.image_backbone(image)).flatten(2).transpose(1, 2)
        tof_t = self.tof_encoder(tof).flatten(2).transpose(1, 2)
        rest = torch.stack([self.imu_mlp(imu.flatten(1)), self.state_mlp(state), self.mode_embed(mode)], dim=1)
        return torch.cat([img, tof_t, rest], dim=1)            # [B,94,D]


# ---------------------------------------------------------------- v0(基準)

def build_v0(num_modes=6, pretrained=False, **_):
    return JetFormer(num_modes=num_modes, pretrained=pretrained)


# ---------------------------------------------------------------- ACT 型

class ACTProto(nn.Module):
    """ACT(Zhao ほか 2023)型。センサーのトークン → Encoder → 10個の行動クエリ(未来の1ステップに1つ)が
    Encoder の出力を参照する Decoder → クエリごとにジャンル別ヘッド [steer, throttle]。

    cvae=True: 学習時だけ、正解の行動列と操作値から潜在変数 z(32次元)を作る CVAE の Encoder を足す。
    推論は z=0(潜在トークン = latent_proj のバイアスだけの定数)なので、書き出すグラフは cvae=False と同じ大きさ。
    元の ACT の実装は decoder 7層の設定だが、実際には1層目の出力しか使っていない(LeRobot も既定 1層)ので既定は 1。"""

    def __init__(self, num_modes=6, dim=256, enc_layers=4, dec_layers=1, heads=8, ffn=512, cvae=False,
                 latent_dim=32, pretrained=False):
        super().__init__()
        self.num_modes, self.cvae, self.latent_dim = num_modes, cvae, latent_dim
        self.tokens = SensorTokens(dim, num_modes, pretrained)
        n = 94 + (1 if cvae else 0)
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        self.encoder = encoder(dim, heads, ffn, enc_layers)
        self.enc_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.zeros(1, FUTURE_STEPS, dim))
        dec = nn.TransformerDecoderLayer(dim, heads, ffn, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.dec_norm = nn.LayerNorm(dim)
        self.heads = nn.Linear(dim, num_modes * 2)                 # クエリごと・ジャンル別 [steer, throttle]
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)
        if cvae:
            self.cvae_cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.cvae_action = nn.Linear(2, dim)
            self.cvae_state = nn.Linear(3, dim)
            self.cvae_pos = nn.Parameter(torch.zeros(1, 2 + FUTURE_STEPS, dim))
            self.cvae_encoder = encoder(dim, heads, ffn, 4)
            self.cvae_out = nn.Linear(dim, latent_dim * 2)
            self.latent_proj = nn.Linear(latent_dim, dim)
            nn.init.trunc_normal_(self.cvae_pos, std=0.02)
        self.last_kl = None

    def forward(self, image, tof, imu, state, mode, actions=None):
        b = image.shape[0]
        tok = self.tokens(image, tof, imu, state, mode)
        if self.cvae:
            if self.training and actions is not None:
                c = torch.cat([self.cvae_cls.expand(b, -1, -1), self.cvae_state(state).unsqueeze(1),
                               self.cvae_action(actions)], dim=1) + self.cvae_pos
                mu, logvar = self.cvae_out(self.cvae_encoder(c)[:, 0]).chunk(2, dim=-1)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                self.last_kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(-1).mean()
            else:
                z = torch.zeros(b, self.latent_dim, device=image.device, dtype=image.dtype)
            tok = torch.cat([tok, self.latent_proj(z).unsqueeze(1)], dim=1)
        memory = self.enc_norm(self.encoder(tok + self.pos))
        hs = self.dec_norm(self.decoder(self.queries.expand(b, -1, -1), memory))     # [B,10,D]
        all_modes = self.heads(hs).view(b, FUTURE_STEPS, self.num_modes, 2).transpose(1, 2)
        return select_mode(all_modes, mode)                                       # [B,10,2]


# ---------------------------------------------------------------- TCP / TransFuser++ 型

class GRUCellExport(nn.Module):
    """nn.GRUCell と同じ計算を Linear だけで書いたもの(重みの並びも同じ: r, z, n)。
    ONNX の書き出しで aten::gru_cell が通らない場合の代わり"""

    def __init__(self, cin, hidden):
        super().__init__()
        self.ih = nn.Linear(cin, 3 * hidden)
        self.hh = nn.Linear(hidden, 3 * hidden)

    def forward(self, x, h):
        ir, iz, in_ = self.ih(x).chunk(3, dim=-1)
        hr, hz, hn = self.hh(h).chunk(3, dim=-1)
        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(in_ + r * hn)
        return (1 - z) * n + z * h


class TCPProto(nn.Module):
    """TCP(Wu ほか 2022)・TransFuser++(Jaeger ほか 2023)型。

    v0 と同じトークン + 出力トークン2つ(経路用・操作用)→ Encoder 4層。
      経路の枝: GRU で 10点(0.1秒おき・1秒先まで)の通過点を自車の座標(x 前・y 左、m)で順に出す + 目標速度(m/s)
      操作の枝: v0 と同じジャンル別ヘッド → actions [B,10,2](TCP の「直接の操作」の枝)
    経路と目標速度から操作を作る制御器(純追従 + 速度の FF/PI)は Python(numpy)側に置く想定(design_bench.md)。
    gru='torch' は nn.GRUCell、'linear' は GRUCellExport(書き出しの確認用)"""

    def __init__(self, num_modes=6, dim=256, layers=4, heads=8, ffn=512, n_wp=10, hidden=128, gru='torch',
                 pretrained=False):
        super().__init__()
        self.num_modes, self.n_wp = num_modes, n_wp
        self.tokens = SensorTokens(dim, num_modes, pretrained)
        self.queries = nn.Parameter(torch.zeros(1, 2, dim))
        self.pos = nn.Parameter(torch.zeros(1, 96, dim))
        self.fusion = encoder(dim, heads, ffn, layers)
        self.norm = nn.LayerNorm(dim)
        self.wp_init = nn.Linear(dim, hidden)
        cin = 2 + num_modes
        self.wp_gru = nn.GRUCell(cin, hidden) if gru == 'torch' else GRUCellExport(cin, hidden)
        self.wp_out = nn.Linear(hidden, num_modes * 2)              # 1ステップの移動 [dx, dy](ジャンル別)
        self.speed_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, num_modes))
        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.heads = nn.Linear(dim, num_modes * FUTURE_STEPS * 2)     # v0 と同じ並び(--init-mode-copy が使える)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, image, tof, imu, state, mode):
        b = image.shape[0]
        tok = torch.cat([self.tokens(image, tof, imu, state, mode), self.queries.expand(b, -1, -1)], dim=1)
        out = self.norm(self.fusion(tok + self.pos))
        q_traj, q_ctrl = out[:, -2], out[:, -1]
        h = torch.tanh(self.wp_init(q_traj))
        x = torch.zeros(b, 2, device=image.device, dtype=out.dtype)
        wps = []
        for _ in range(self.n_wp):
            h = self.wp_gru(torch.cat([x, mode.to(x.dtype)], dim=1), h)
            x = x + select_mode(self.wp_out(h).view(b, self.num_modes, 2), mode)
            wps.append(x)
        waypoints = torch.stack(wps, dim=1)                                        # [B,10,2] m
        target_speed = F.softplus(select_mode(self.speed_head(q_traj).unsqueeze(-1), mode))   # [B,1] m/s
        actions = select_mode(self.heads(self.trunk(q_ctrl)).view(b, self.num_modes, FUTURE_STEPS, 2), mode)
        return {'actions': actions, 'waypoints': waypoints, 'target_speed': target_speed}


class TCPDecProto(nn.Module):
    """design_research.md §4.4 の既定の形: v0 と同じトークン → Encoder 4層 → Decoder 2層。
    クエリ 21個 = 通過点 10・速度 1・操作 10(GRU を使わない = TensorRT でループを作らない。ACT と同じ Decoder のコード)。
      通過点: Linear(256, M×2) の Δ(x, y) → 累積和 → waypoints [B,10,2](m)
      速度  : Linear(256, M×10) → speed_logits [B,10](0, 0.5, …, 4.5 m/s の two-hot。期待値は Python 側)
      操作  : Linear(256, M×2) → actions [B,10,2](操作の枝)
    cumsum='matmul' は累積和を下三角の定数行列の掛け算で書く(CumSum が TensorRT に通らない場合の代わり)"""

    def __init__(self, num_modes=6, dim=256, enc_layers=4, dec_layers=2, heads=8, ffn=512, n_bins=10, cumsum='op',
                 pretrained=False):
        super().__init__()
        self.num_modes, self.n_bins, self.cumsum = num_modes, n_bins, cumsum
        self.tokens = SensorTokens(dim, num_modes, pretrained)
        self.pos = nn.Parameter(torch.zeros(1, 94, dim))
        self.encoder = encoder(dim, heads, ffn, enc_layers)
        self.enc_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.zeros(1, 2 * FUTURE_STEPS + 1, dim))
        dec = nn.TransformerDecoderLayer(dim, heads, ffn, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.dec_norm = nn.LayerNorm(dim)
        self.heads_wp = nn.Linear(dim, num_modes * 2)
        self.heads_v = nn.Linear(dim, num_modes * n_bins)
        self.heads = nn.Linear(dim, num_modes * 2)
        self.register_buffer('tril', torch.tril(torch.ones(FUTURE_STEPS, FUTURE_STEPS)), persistent=False)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, image, tof, imu, state, mode):
        b, m, n = image.shape[0], self.num_modes, FUTURE_STEPS
        memory = self.enc_norm(self.encoder(self.tokens(image, tof, imu, state, mode) + self.pos))
        hs = self.dec_norm(self.decoder(self.queries.expand(b, -1, -1), memory))
        delta = select_mode(self.heads_wp(hs[:, :n]).view(b, n, m, 2).transpose(1, 2), mode)        # [B,10,2]
        waypoints = delta.cumsum(dim=1) if self.cumsum == 'op' else torch.matmul(self.tril, delta)
        speed_logits = select_mode(self.heads_v(hs[:, n]).view(b, m, self.n_bins), mode)               # [B,10]
        actions = select_mode(self.heads(hs[:, n + 1:]).view(b, n, m, 2).transpose(1, 2), mode)
        return {'actions': actions, 'waypoints': waypoints, 'speed_logits': speed_logits}


# ---------------------------------------------------------------- TransFuser 型の多段融合

IMG_STAGE_HW = ((56, 56), (28, 28), (14, 14), (7, 7))     # 224 の入力での ResNet-18 の layer1〜4
TOF_STAGE_HW = ((48, 56), (24, 28), (12, 14), (6, 7))     # ToF 42x54 を 48x56 に詰めてから 1, 1/2, 1/4, 1/8
STAGE_CH = (64, 128, 256, 512)
IMG_GRID, TOF_GRID = (7, 7), (6, 7)                        # 各段で融合するトークンの格子(49 + 42 = 91 トークン)


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ToFStages(nn.Module):
    """ToF(距離・信頼度・有効マスク)の CNN。ResNet-18 の各段と同じチャンネル数(64/128/256/512)の4段。
    42x54 を 48x56 に 0 で詰める(上下3・左右1)と、各段が 7x7・6x7 の格子でちょうど割り切れる
    (adaptive_avg_pool2d の割り切れない大きさは ONNX に書き出せないため)"""

    def __init__(self, chans=STAGE_CH):
        super().__init__()
        c1, c2, c3, c4 = chans
        self.stem = conv_bn(TOF_CHANNELS + 1, 32)
        self.stages = nn.ModuleList([
            nn.Sequential(conv_bn(32, c1), conv_bn(c1, c1)),
            nn.Sequential(conv_bn(c1, c2, 2), conv_bn(c2, c2)),
            nn.Sequential(conv_bn(c2, c3, 2), conv_bn(c3, c3)),
            nn.Sequential(conv_bn(c3, c4, 2), conv_bn(c4, c4))])

    def prepare(self, tof):
        valid = (tof[:, :1] > 0).to(tof.dtype)
        x = torch.cat([tof, valid], dim=1)
        return self.stem(F.pad(x, (1, 1, 3, 3)))                               # [B,32,48,56]


class StageFusion(nn.Module):
    """1段分の融合(TransFuser の GPT): 画像と ToF を平均プールで小さな格子のトークンにし、まとめて Transformer に通し、
    双線形で元の大きさに戻して各枝に足す"""

    def __init__(self, ch, img_hw, tof_hw, layers=1, heads=4, ffn_mult=2):
        super().__init__()
        self.ki = (img_hw[0] // IMG_GRID[0], img_hw[1] // IMG_GRID[1])
        self.kt = (tof_hw[0] // TOF_GRID[0], tof_hw[1] // TOF_GRID[1])
        assert self.ki[0] * IMG_GRID[0] == img_hw[0] and self.kt[1] * TOF_GRID[1] == tof_hw[1]
        self.ch = ch
        self.n_img = IMG_GRID[0] * IMG_GRID[1]
        self.pos = nn.Parameter(torch.zeros(1, self.n_img + TOF_GRID[0] * TOF_GRID[1], ch))
        self.tf = encoder(ch, heads, ch * ffn_mult, layers)
        self.norm = nn.LayerNorm(ch)
        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _pool(x, k):
        return x if k == (1, 1) else F.avg_pool2d(x, k, k)

    @staticmethod
    def _up(x, k):
        return x if k == (1, 1) else F.interpolate(x, scale_factor=k, mode='bilinear', align_corners=False)

    def forward(self, fi, ft):
        pi, pt = self._pool(fi, self.ki), self._pool(ft, self.kt)
        tok = torch.cat([pi.flatten(2), pt.flatten(2)], dim=2).transpose(1, 2) + self.pos
        tok = self.norm(self.tf(tok)).transpose(1, 2)
        ti = tok[:, :, :self.n_img].reshape(-1, self.ch, *IMG_GRID)
        tt = tok[:, :, self.n_img:].reshape(-1, self.ch, *TOF_GRID)
        return fi + self._up(ti, self.ki), ft + self._up(tt, self.kt)


class MSFusionProto(nn.Module):
    """TransFuser(Chitta ほか PAMI 2022)型の多段融合。

    画像: ResNet-18 の layer1〜4(56→7)/ ToF: ToFStages(48x56→6x7)。fuse_stages の各段で StageFusion。
    最後: 両方の4段目を平均プール → 和(TransFuser と同じ)→ 256 + IMU・操作値・ジャンル
      head='mlp': 連結 → MLP → ジャンル別ヘッド
      head='tf' : 4段目の融合トークン 91 を 256 に写し、IMU・操作値・ジャンル・出力トークンと 2層の Encoder → ヘッド"""

    def __init__(self, num_modes=6, dim=256, fuse_stages=(0, 1, 2, 3), layers=1, heads=4, ffn_mult=2, head='mlp',
                 pretrained=False):
        super().__init__()
        self.num_modes, self.head_kind = num_modes, head
        r = resnet18(pretrained)
        self.img_stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.img_stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.tof = ToFStages()
        self.fuse = nn.ModuleDict({str(s): StageFusion(STAGE_CH[s], IMG_STAGE_HW[s], TOF_STAGE_HW[s], layers,
                                                       heads if STAGE_CH[s] >= 128 else 2, ffn_mult)
                                   for s in fuse_stages})
        self.imu_mlp = mlp2(IMU_STEPS * 6, dim)
        self.state_mlp = mlp2(3, dim)
        self.mode_embed = nn.Linear(num_modes, dim, bias=False)
        nn.init.normal_(self.mode_embed.weight, std=0.02)
        if head == 'mlp':
            self.global_proj = nn.Linear(STAGE_CH[3], dim)
            self.trunk = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim),
                                       nn.ReLU(inplace=True))
        else:
            self.token_proj = nn.Linear(STAGE_CH[3], dim)
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, 91 + 4, dim))
            self.head_tf = encoder(dim, 8, dim * 2, 2)
            self.norm = nn.LayerNorm(dim)
            self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.query, std=0.02)
        self.heads = nn.Linear(dim, num_modes * FUTURE_STEPS * 2)

    def forward(self, image, tof, imu, state, mode):
        b = image.shape[0]
        fi = self.img_stem(image)
        ft = self.tof.prepare(tof)
        for s in range(4):
            fi = self.img_stages[s](fi)
            ft = self.tof.stages[s](ft)
            if str(s) in self.fuse:
                fi, ft = self.fuse[str(s)](fi, ft)
        imu_t, st_t, mode_t = self.imu_mlp(imu.flatten(1)), self.state_mlp(state), self.mode_embed(mode)
        if self.head_kind == 'mlp':
            g = self.global_proj(fi.mean(dim=(2, 3)) + ft.mean(dim=(2, 3)))
            h = self.trunk(torch.cat([g, imu_t, st_t, mode_t], dim=1))
        else:
            sp = torch.cat([fi.flatten(2), ft.flatten(2)], dim=2).transpose(1, 2)    # [B,91,512]
            tok = torch.cat([self.token_proj(sp), torch.stack([imu_t, st_t, mode_t], 1),
                             self.query.expand(b, -1, -1)], dim=1) + self.pos
            h = self.trunk(self.norm(self.head_tf(tok))[:, -1])
        return select_mode(self.heads(h).view(b, self.num_modes, FUTURE_STEPS, 2), mode)


class ToFStagesRD(nn.Module):
    """design_research.md §5.2 の ToF の枝: 48x56 に詰めて stem 32ch → 64ch 24x28 → 128ch 12x14 → 256ch 6x7 → (stride 1)256 → 1x1 で 512ch 6x7"""

    def __init__(self):
        super().__init__()
        self.stem = conv_bn(TOF_CHANNELS + 1, 32)
        self.stages = nn.ModuleList([
            nn.Sequential(conv_bn(32, 64, 2), conv_bn(64, 64)),
            nn.Sequential(conv_bn(64, 128, 2), conv_bn(128, 128)),
            nn.Sequential(conv_bn(128, 256, 2), conv_bn(256, 256)),
            nn.Sequential(conv_bn(256, 256), nn.Conv2d(256, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))])

    def prepare(self, tof):
        valid = (tof[:, :1] > 0).to(tof.dtype)
        return self.stem(F.pad(torch.cat([tof, valid], dim=1), (1, 1, 3, 3)))


class MSFusionRDProto(nn.Module):
    """design_research.md §5.2 の既定の形の多段融合。
    交換: 段1〜3 は 2層・4ヘッド・FFN×4、段4 は 1層・FFN×2(画像 7x7 + ToF 6x7 = 91 トークン)。
    後: 最終段のトークン(画像 49・ToF 42 を各 512→256)+ IMU・操作値・ジャンル・出力用トークン → v0 型の融合 2層 → ジャンル別ヘッド"""

    TOF_HW = ((24, 28), (12, 14), (6, 7), (6, 7))

    def __init__(self, num_modes=6, dim=256, pretrained=False):
        super().__init__()
        self.num_modes = num_modes
        r = resnet18(pretrained)
        self.img_stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.img_stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.tof = ToFStagesRD()
        cfg = ((2, 4), (2, 4), (2, 4), (1, 2))
        self.fuse = nn.ModuleList([StageFusion(STAGE_CH[s], IMG_STAGE_HW[s], self.TOF_HW[s], cfg[s][0], 4, cfg[s][1])
                                   for s in range(4)])
        self.img_proj = nn.Linear(512, dim)
        self.tof_proj = nn.Linear(512, dim)
        self.imu_mlp = mlp2(IMU_STEPS * 6, dim)
        self.state_mlp = mlp2(3, dim)
        self.mode_embed = nn.Linear(num_modes, dim, bias=False)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 91 + 4, dim))
        self.fusion = encoder(dim, 8, dim * 2, 2)
        self.norm = nn.LayerNorm(dim)
        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.heads = nn.Linear(dim, num_modes * FUTURE_STEPS * 2)
        nn.init.normal_(self.mode_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, image, tof, imu, state, mode):
        b = image.shape[0]
        fi, ft = self.img_stem(image), self.tof.prepare(tof)
        for s in range(4):
            fi, ft = self.img_stages[s](fi), self.tof.stages[s](ft)
            fi, ft = self.fuse[s](fi, ft)
        rest = torch.stack([self.imu_mlp(imu.flatten(1)), self.state_mlp(state), self.mode_embed(mode)], dim=1)
        tok = torch.cat([self.img_proj(fi.flatten(2).transpose(1, 2)), self.tof_proj(ft.flatten(2).transpose(1, 2)),
                         rest, self.query.expand(b, -1, -1)], dim=1) + self.pos
        h = self.trunk(self.norm(self.fusion(tok))[:, -1])
        return select_mode(self.heads(h).view(b, self.num_modes, FUTURE_STEPS, 2), mode)


# ---------------------------------------------------------------- DINOv2 ViT-S/14(自前の実装・公式の重みを読む)

class _PatchEmbed(nn.Module):
    def __init__(self, dim=384, patch=14):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoViTS14(nn.Module):
    """DINOv2 ViT-S/14(dinov2_vits14、レジスターなし)を torch.hub のコードなしで組んだもの。
    state_dict のキーは公式と同じ。pos_embed は公式(518px → 37x37)を、読み込むときに一度だけ
    公式と同じ方法(bicubic・antialias なし・scale_factor = (格子+0.1)/37)で img_size//14 の格子に補間して持つ
    (グラフに補間を残さない = ONNX・TensorRT で固定の定数)。出力は最後の LayerNorm の後 [B,1+N,384](先頭が CLS)"""

    def __init__(self, img_size=224, dim=384, depth=12, heads=6):
        super().__init__()
        self.grid = img_size // 14
        self.patch_embed = _PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    @staticmethod
    def interpolate_pos(pos, grid, offset=0.1):
        n = pos.shape[1] - 1
        m = int(math.sqrt(n))
        if m == grid:
            return pos
        patch = pos[:, 1:].float().reshape(1, m, m, -1).permute(0, 3, 1, 2)
        s = (grid + offset) / m
        patch = F.interpolate(patch, scale_factor=(s, s), mode='bicubic', antialias=False)
        assert patch.shape[-2:] == (grid, grid)
        return torch.cat([pos[:, :1].float(), patch.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)], dim=1)

    def load_official(self, path=DINO_WEIGHTS):
        sd = torch.load(path, map_location='cpu', weights_only=True)
        sd = {k: v for k, v in sd.items() if k != 'mask_token'}
        sd['pos_embed'] = self.interpolate_pos(sd['pos_embed'], self.grid)
        self.load_state_dict(sd, strict=True)
        return self

    def forward(self, x):
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), self.patch_embed(x)], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class DinoProto(nn.Module):
    """DINOv2 の画像の特徴 + v0 の残りのトークン → v0 型の Encoder(d256・4層)→ 出力トークン → ジャンル別ヘッド。

    vit_res: ViT に入れる大きさ(224 = 16x16 パッチ。168 なら 224 の入力をグラフの中で双線形で縮める = 12x12)
    pool   : パッチの格子を k x k の平均で減らす(1 = 減らさない。2 で 16x16 → 8x8 = 64 トークン)
    finetune_blocks: 後ろから何ブロックを学習するか(0 = 全部凍結。凍結した部分は学習でも no_grad・eval)"""

    def __init__(self, num_modes=6, dim=256, layers=4, heads=8, vit_res=224, pool=2, finetune_blocks=0,
                 load_weights=True, pretrained=False):
        super().__init__()
        self.num_modes, self.vit_res, self.pool = num_modes, vit_res, pool
        self.vit = DinoViTS14(img_size=vit_res)
        if load_weights:
            self.vit.load_official()
        self.finetune_blocks = finetune_blocks
        n_frozen = len(self.vit.blocks) - finetune_blocks
        for name, p in self.vit.named_parameters():
            trainable = name.startswith('norm.') or any(name.startswith('blocks.%d.' % i)
                                                        for i in range(n_frozen, len(self.vit.blocks)))
            p.requires_grad_(finetune_blocks > 0 and trainable)
        self.n_frozen = n_frozen
        g = (vit_res // 14) // pool
        self.n_img = g * g + 1
        self.image_proj = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, dim))
        self.tof_encoder = ToFEncoder(dim)
        self.imu_mlp = mlp2(IMU_STEPS * 6, dim)
        self.state_mlp = mlp2(3, dim)
        self.mode_embed = nn.Linear(num_modes, dim, bias=False)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n_img + 42 + 4, dim))
        self.fusion = encoder(dim, heads, dim * 2, layers)
        self.norm = nn.LayerNorm(dim)
        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.heads = nn.Linear(dim, num_modes * FUTURE_STEPS * 2)
        nn.init.normal_(self.mode_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()            # 凍結した ViT は学習中も eval(DINOv2 は dropout なしなので実際の差は無い)
        return self

    def image_tokens(self, image):
        x = image
        if self.vit_res != image.shape[-1]:
            x = F.interpolate(x, size=(self.vit_res, self.vit_res), mode='bilinear', align_corners=False)
        v = self.vit
        h = torch.cat([v.cls_token.expand(x.shape[0], -1, -1), v.patch_embed(x)], dim=1) + v.pos_embed
        if self.n_frozen > 0 and self.training:
            with torch.no_grad():
                for blk in v.blocks[:self.n_frozen]:
                    h = blk(h)
            h = h.detach()
        else:
            for blk in v.blocks[:self.n_frozen]:
                h = blk(h)
        for blk in v.blocks[self.n_frozen:]:
            h = blk(h)
        h = v.norm(h)
        cls, patches = h[:, :1], h[:, 1:]
        if self.pool > 1:
            g = v.grid
            p = patches.transpose(1, 2).reshape(-1, 384, g, g)
            patches = F.avg_pool2d(p, self.pool, self.pool).flatten(2).transpose(1, 2)
        return torch.cat([cls, patches], dim=1)

    def forward(self, image, tof, imu, state, mode):
        b = image.shape[0]
        img = self.image_proj(self.image_tokens(image))
        tof_t = self.tof_encoder(tof).flatten(2).transpose(1, 2)
        rest = torch.stack([self.imu_mlp(imu.flatten(1)), self.state_mlp(state), self.mode_embed(mode)], dim=1)
        tok = torch.cat([img, tof_t, rest, self.query.expand(b, -1, -1)], dim=1) + self.pos
        out = self.norm(self.fusion(tok))[:, -1]
        return select_mode(self.heads(self.trunk(out)).view(b, self.num_modes, FUTURE_STEPS, 2), mode)


# ---------------------------------------------------------------- 試す形の一覧

VARIANTS = {
    'v0':            (build_v0, {}),
    'act':           (ACTProto, {}),
    'act_dec2':      (ACTProto, {'dec_layers': 2}),                  # design_research.md の既定
    'act_dec4':      (ACTProto, {'dec_layers': 4}),
    'act_cvae':      (ACTProto, {'cvae': True}),
    'act_d512':      (ACTProto, {'dim': 512, 'ffn': 3200}),          # 元の ACT の大きさ(d512・FFN 3200・Encoder 4・Decoder 1)
    'tcp':           (TCPProto, {}),
    'tcp_dec':       (TCPDecProto, {}),                              # design_research.md の既定(Decoder 2層・21 クエリ)
    'msfusion':      (MSFusionProto, {}),
    'msfusion_l2':   (MSFusionProto, {'layers': 2}),
    'msfusion_tf':   (MSFusionProto, {'head': 'tf'}),
    'msfusion_rd':   (MSFusionRDProto, {}),                          # design_research.md の既定
    'msfusion_l4':   (MSFusionProto, {'layers': 4, 'ffn_mult': 4}),  # TransFuser の元に近い深さ(各段 4層・FFN 4倍)
    'dinov2_p1':     (DinoProto, {'pool': 1}),
    'dinov2_p2':     (DinoProto, {'pool': 2}),
    'dinov2_p2_ft2': (DinoProto, {'pool': 2, 'finetune_blocks': 2}),
    'dinov2_r168_p2': (DinoProto, {'vit_res': 168, 'pool': 2}),
}


def build(name, num_modes=6, **over):
    fn, kw = VARIANTS[name]
    kw = dict(kw, **over)
    return fn(num_modes=num_modes, **kw)


def example_inputs(num_modes=6, batch=1, device='cpu', seed=0):
    g = torch.Generator().manual_seed(seed)
    mode = torch.zeros(batch, num_modes)
    mode[:, 5 if num_modes > 5 else 0] = 1.0
    x = (torch.randn(batch, 3, 224, 224, generator=g), torch.rand(batch, TOF_CHANNELS, *TOF_SHAPE, generator=g),
         torch.randn(batch, IMU_STEPS, 6, generator=g) * 0.1, torch.randn(batch, 3, generator=g) * 0.3, mode)
    return tuple(t.to(device) for t in x)


def outputs_of(y):
    """forward の戻り値 → {名前: Tensor}(Tensor だけなら actions)"""
    return dict(y) if isinstance(y, dict) else {'actions': y}
