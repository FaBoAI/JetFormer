"""msfusion — 本家 TransFuser に近い多段融合(spec.md §3.3、Chitta ほか IEEE TPAMI 2022 / Prakash ほか CVPR 2021)

画像(ResNet-18 の4段)と ToF(同じチャンネル数の4段の CNN)を、各段で小さな Transformer に通して情報を交換し
(TransFuser の GPT と同じ「平均で小さな格子のトークンにする → 注意 → 双線形で元の大きさに戻して各枝に足す」)、
最後に4段目のトークンを v0 型の融合(Encoder 2層)にかけて、ジャンル別のヘッドで 10 フレーム先までの操作を出す。

    image [B,3,224,224] → img_stem(conv1・bn1・relu・maxpool)→ 64×56×56
    tof   [B,2,42,54]  + 有効の印 → F.pad(左右 1・上下 3)で 3×48×56(詰めた所は無効 = 0)→ tof.stem 32×48×56

    段 | 画像(img_stages)        | ToF(tof.stages)                     | 交換(fuse['段'])
    1  | layer1 64×56×56  → 平均 8×8 | conv_bn s2・conv_bn  64×24×28 → 平均 4×4 | d64・4ヘッド・2層・FFN×4
    2  | layer2 128×28×28 → 平均 4×4 | conv_bn s2・conv_bn 128×12×14 → 平均 2×2 | d128・4ヘッド・2層・FFN×4
    3  | layer3 256×14×14 → 平均 2×2 | conv_bn s2・conv_bn 256×6×7              | d256・4ヘッド・2層・FFN×4
    4  | layer4 512×7×7              | conv_bn s1・1×1 conv+BN+ReLU 512×6×7      | d512・4ヘッド・1層・FFN×2
       (どの段も画像 7×7 = 49・ToF 6×7 = 42 の 91 トークン + 段ごとの位置の埋め込み pos [1,91,ch])

    head='tokens'(既定): 4段目の画像 49 → img_proj、ToF 42 → tof_proj(512→256)、imu_mlp・state_mlp・mode_embed の 3、
        query 1 → + pos [1,95,256] → fusion(Encoder 256・8ヘッド・FFN 512・late_layers 層)→ norm → 最後のトークン
        → trunk(Linear+ReLU)→ heads Linear(256, M×20)(v0 と同じ並び)→ one-hot でジャンルを選ぶ → actions [B,10,2]
    head='pooled'(TransFuser どおり): global_proj(画像と ToF の4段目の平均の和、512→256)・IMU・操作値・ジャンルをつなぐ
        (4×256)→ trunk(MLP 2層)→ heads

本家との違い(spec.md §3.3・design_research.md §5.2):
- ToF は LiDAR の鳥瞰図ではなく、カメラとほぼ同じ向きの距離画像(42×54)。
- 交換の格子は TransFuser の 8×8 ではなく、224 の入力で割り切れる画像 7×7・ToF 6×7(adaptive_avg_pool2d の割り切れない
  大きさと antialias の縮小は opset 17 に書き出せないので、固定の F.avg_pool2d と antialias なしの双線形だけを使う)。
- 交換の深さは段ごとに 2/2/2/1 層(本家は 4〜8 層。学習の時間とメモリの予算)。補助タスク(深度など)は入れない(将来の aux_depth)。
- 最後は既定でトークンの融合(v0 の一番強い部分を残し、v0 との違いを「早い段の交換」だけにする)。

出力は actions だけ(正規化 (µs − 1500)/500。人の操作と同じ符号: ステアは中心より大きい = 左、スロットルは中立より小さい = 前進)。
損失は既定の L1、ラベルも v0 と同じ(この方策は正解の作り方を変えない)。
state_dict のジャンルの層は一番上の mode_embed.weight・heads(1ジャンル 20 行、登録簿の genre_keys)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..model import FUTURE_STEPS, IMU_STEPS, TOF_CHANNELS, TOF_SHAPE
from .blocks import encoder, mlp2, select_mode

STAGE_CH = (64, 128, 256, 512)                              # ResNet-18 の layer1〜4 のチャンネル数(ToF の枝も同じ)
IMG_STAGE_HW = ((56, 56), (28, 28), (14, 14), (7, 7))      # 224 の入力での layer1〜4 の大きさ
TOF_PAD = (1, 1, 3, 3)                                     # F.pad の (左, 右, 上, 下): 42×54 → 48×56
TOF_PADDED_HW = (TOF_SHAPE[0] + TOF_PAD[2] + TOF_PAD[3], TOF_SHAPE[1] + TOF_PAD[0] + TOF_PAD[1])   # (48, 56)
TOF_STAGE_HW = ((24, 28), (12, 14), (6, 7), (6, 7))        # ToF の枝の段1〜4(stem は 48×56、段4 は stride 1)
IMG_GRID, TOF_GRID = (7, 7), (6, 7)                        # 各段で交換するトークンの格子
N_IMG, N_TOF = IMG_GRID[0] * IMG_GRID[1], TOF_GRID[0] * TOF_GRID[1]     # 49・42
N_EXTRA = 4                                                # IMU・操作値・ジャンル・出力トークン
STAGES = (1, 2, 3, 4)
HEAD_KINDS = ('tokens', 'pooled')
TOF_STEM_CH = 32


def conv_bn(cin, cout, stride=1):
    """3×3 Conv(バイアスなし)→ BatchNorm → ReLU(v0 の ToFEncoder と同じ部品)"""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def window(hw, grid):
    """段の大きさ hw を格子 grid にする平均の窓(割り切れなければ ValueError。書き出せる形だけを作る)"""
    k = (hw[0] // grid[0], hw[1] // grid[1])
    if k[0] < 1 or k[1] < 1 or k[0] * grid[0] != hw[0] or k[1] * grid[1] != hw[1]:
        raise ValueError('段の大きさ %s が格子 %s で割り切れません' % (tuple(hw), tuple(grid)))
    return k


class ToFStages(nn.Module):
    """ToF の枝(BatchNorm の CNN)。入力は [距離, 信頼度, 有効の印] を 48×56 に 0 で詰めたもの。
    段1〜4 の出力は 64×24×28・128×12×14・256×6×7・512×6×7(ResNet-18 の各段と同じチャンネル数)"""

    def __init__(self):
        super().__init__()
        c1, c2, c3, c4 = STAGE_CH
        self.stem = conv_bn(TOF_CHANNELS + 1, TOF_STEM_CH)
        self.stages = nn.ModuleList([
            nn.Sequential(conv_bn(TOF_STEM_CH, c1, 2), conv_bn(c1, c1)),
            nn.Sequential(conv_bn(c1, c2, 2), conv_bn(c2, c2)),
            nn.Sequential(conv_bn(c2, c3, 2), conv_bn(c3, c3)),
            nn.Sequential(conv_bn(c3, c3),
                          nn.Conv2d(c3, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))])

    @staticmethod
    def padded_input(tof):
        """tof [B,2,42,54] → [B,3,48,56]。距離 0 = 無効(v0 の ToFEncoder と同じ印)。上下 3 行・左右 1 列を 0(無効)で詰める
        (向きは変えない: 元の [r, c] は [r + 3, c + 1] に入る)"""
        valid = (tof[:, :1] > 0).to(tof.dtype)
        return F.pad(torch.cat([tof, valid], dim=1), TOF_PAD)

    def prepare(self, tof):
        return self.stem(self.padded_input(tof))              # [B,32,48,56]


class StageFusion(nn.Module):
    """1段分の交換(TransFuser の GPT): 画像と ToF を固定の平均で 7×7・6×7 の格子のトークンにし(91 個)、位置の埋め込みを足して
    Encoder → LayerNorm → 元の格子に戻し、双線形(align_corners=False)で元の大きさに拡大して各枝に足す(残差)"""

    def __init__(self, ch, img_hw, tof_hw, layers, heads, ffn_mult, dropout=0.1):
        super().__init__()
        if ch % heads:
            raise ValueError('交換のヘッドの数 %d がチャンネル数 %d を割り切りません' % (heads, ch))
        self.ch = ch
        self.ki = window(img_hw, IMG_GRID)
        self.kt = window(tof_hw, TOF_GRID)
        self.pos = nn.Parameter(torch.zeros(1, N_IMG + N_TOF, ch))
        self.tf = encoder(ch, heads, ch * ffn_mult, layers, dropout)
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
        tok = torch.cat([pi.flatten(2), pt.flatten(2)], dim=2).transpose(1, 2) + self.pos       # [B,91,ch]
        tok = self.norm(self.tf(tok)).transpose(1, 2)                                            # [B,ch,91]
        ti = tok[:, :, :N_IMG].reshape(-1, self.ch, IMG_GRID[0], IMG_GRID[1])
        tt = tok[:, :, N_IMG:].reshape(-1, self.ch, TOF_GRID[0], TOF_GRID[1])
        return fi + self._up(ti, self.ki), ft + self._up(tt, self.kt)


def _int_list(name, value, n=4, lo=1):
    v = [int(x) for x in value]
    if len(v) != n or any(x < lo for x in v):
        raise ValueError('%s は %d 以上の整数を %d 個(段1〜4)にしてください: %s' % (name, lo, n, list(value)))
    return v


def check_cfg(dim, scales, ex_layers, ex_heads, ex_ffn, late_layers, late_heads, head, dropout):
    """設定を確かめて (scales, ex_layers, ex_ffn) を整える。だめなら ValueError(学習を始める前に分かるように)"""
    sc = [int(s) for s in scales]
    if any(s not in STAGES for s in sc) or len(set(sc)) != len(sc):
        raise ValueError('scales は 1〜4 の段の番号(重複なし)にしてください: %s' % list(scales))
    ex_layers = _int_list('ex_layers', ex_layers)
    ex_ffn = _int_list('ex_ffn', ex_ffn)
    if int(ex_heads) < 1 or any(STAGE_CH[s - 1] % int(ex_heads) for s in sc):
        raise ValueError('ex_heads %s が交換する段のチャンネル数 %s を割り切りません' % (ex_heads, [STAGE_CH[s - 1] for s in sc]))
    if head not in HEAD_KINDS:
        raise ValueError('head は %s のどれか(指定 %r)' % (' / '.join(HEAD_KINDS), head))
    if int(dim) < 1 or int(late_layers) < 1 or int(late_heads) < 1 or (head == 'tokens' and int(dim) % int(late_heads)):
        raise ValueError('dim %s・late_layers %s・late_heads %s が使えません(dim は late_heads で割り切れること)' % (
            dim, late_layers, late_heads))
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError('dropout は 0 以上 1 未満(指定 %s)' % dropout)
    return sorted(sc), ex_layers, ex_ffn


class MSFusion(nn.Module):
    """多段融合の方策(モジュールの説明を参照)。forward(image, tof, imu, state, mode) → actions [B,10,2]"""

    def __init__(self, num_modes, dim=256, scales=STAGES, ex_layers=(2, 2, 2, 1), ex_heads=4, ex_ffn=(4, 4, 4, 2),
                 late_layers=2, late_heads=8, head='tokens', dropout=0.1, pretrained=False):
        super().__init__()
        scales, ex_layers, ex_ffn = check_cfg(dim, scales, ex_layers, ex_heads, ex_ffn, late_layers, late_heads, head,
                                              dropout)
        dim = int(dim)
        self.num_modes, self.dim, self.head_kind, self.scales = int(num_modes), dim, head, tuple(scales)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)        # pretrained=False ではファイル・ネットに触れない
        self.img_stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.img_stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.tof = ToFStages()
        self.fuse = nn.ModuleDict({
            str(s): StageFusion(STAGE_CH[s - 1], IMG_STAGE_HW[s - 1], TOF_STAGE_HW[s - 1], ex_layers[s - 1],
                                int(ex_heads), ex_ffn[s - 1], dropout)
            for s in scales})
        self.imu_mlp = mlp2(IMU_STEPS * 6, dim)
        self.state_mlp = mlp2(3, dim)
        self.mode_embed = nn.Linear(self.num_modes, dim, bias=False)       # one-hot → ジャンル埋め込み
        c4 = STAGE_CH[3]
        if head == 'tokens':
            self.img_proj = nn.Linear(c4, dim)
            self.tof_proj = nn.Linear(c4, dim)
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, N_IMG + N_TOF + N_EXTRA, dim))
            self.fusion = encoder(dim, int(late_heads), dim * 2, int(late_layers), dropout)
            self.norm = nn.LayerNorm(dim)
            self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.query, std=0.02)
        else:
            self.global_proj = nn.Linear(c4, dim)
            self.trunk = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim),
                                       nn.ReLU(inplace=True))
        self.heads = nn.Linear(dim, self.num_modes * FUTURE_STEPS * 2)       # ジャンル別ヘッド(v0 と同じ並び)
        nn.init.normal_(self.mode_embed.weight, std=0.02)

    def stage_features(self, image, tof):
        """画像と ToF の4段(各段の後で交換)→ 4段目の特徴 (画像 [B,512,7,7], ToF [B,512,6,7])"""
        fi, ft = self.img_stem(image), self.tof.prepare(tof)
        for s in STAGES:
            fi, ft = self.img_stages[s - 1](fi), self.tof.stages[s - 1](ft)
            key = str(s)
            if key in self.fuse:
                fi, ft = self.fuse[key](fi, ft)
        return fi, ft

    def _trunk_input(self, image, tof, imu, state, mode):
        fi, ft = self.stage_features(image, tof)
        imu_t, st_t, mode_t = self.imu_mlp(imu.flatten(1)), self.state_mlp(state), self.mode_embed(mode)
        if self.head_kind == 'tokens':
            b = image.shape[0]
            tok = torch.cat([self.img_proj(fi.flatten(2).transpose(1, 2)),           # 画像 49
                             self.tof_proj(ft.flatten(2).transpose(1, 2)),           # ToF 42
                             torch.stack([imu_t, st_t, mode_t], dim=1),              # IMU・操作値・ジャンル
                             self.query.expand(b, -1, -1)], dim=1) + self.pos        # 出力トークン → [B,95,dim]
            return self.norm(self.fusion(tok))[:, -1]                                # [B,dim]
        g = self.global_proj(fi.mean(dim=(2, 3)) + ft.mean(dim=(2, 3)))            # 両枝の平均の和(TransFuser)
        return torch.cat([g, imu_t, st_t, mode_t], dim=1)                            # [B,4·dim]

    def embed(self, image, tof, imu, state, mode):
        """tools/offline_rl.py features 用の特徴 [B,dim]。tokens: 融合の後の出力トークン(v0 の model.norm の出力と同じ位置)、
        pooled: trunk の出力"""
        h = self._trunk_input(image, tof, imu, state, mode)
        return h if self.head_kind == 'tokens' else self.trunk(h)

    def forward(self, image, tof, imu, state, mode):
        """入力・出力は v0 と同じ(jetformer/model.py の JetFormer.forward を参照)→ actions [B,10,2]"""
        b = image.shape[0]
        h = self.trunk(self._trunk_input(image, tof, imu, state, mode))
        return select_mode(self.heads(h).view(b, self.num_modes, FUTURE_STEPS, 2), mode)


def build(num_modes, pretrained=False, dim=256, scales=STAGES, ex_layers=(2, 2, 2, 1), ex_heads=4, ex_ffn=(4, 4, 4, 2),
          late_layers=2, late_heads=8, head='tokens', dropout=0.1):
    """登録簿から呼ばれる(spec.md §2.2)。pretrained=True のときだけ ResNet-18 の ImageNet の重み(torchvision のキャッシュ)を読む"""
    return MSFusion(num_modes, dim=dim, scales=scales, ex_layers=ex_layers, ex_heads=ex_heads, ex_ffn=ex_ffn,
                    late_layers=late_layers, late_heads=late_heads, head=head, dropout=dropout, pretrained=pretrained)
