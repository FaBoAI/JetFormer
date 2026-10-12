"""DINOv2 ViT-S/14 の最小の実装(torch.hub・xformers のコードを使わずに、公式の重みをそのまま読む)

公式(https://github.com/facebookresearch/dinov2、Apache-2.0)の hub の `dinov2_vits14` と同じ形:
    パッチ 14・幅 384・12 ブロック・6 ヘッド・MLP 4 倍(GELU は erf 版)・qkv と proj にバイアス・LayerScale(gamma を掛ける)・
    LayerNorm の eps 1e-6・レジスタなし。最後の LayerNorm の後の [B, 1 + N, 384](先頭が CLS)を返す(公式の forward_features の
    x_norm_clstoken・x_norm_patchtokens をつないだもの)。
state_dict のキーは公式と同じ(cls_token・pos_embed・patch_embed.proj・blocks.<i>.{norm1,attn.qkv,attn.proj,ls1.gamma,norm2,
mlp.fc1,mlp.fc2,ls2.gamma}・norm)。公式の重みの mask_token(学習のときだけ使う)は読み込むときに捨てる。

位置の埋め込み: 公式の重みは 518 px 用(37×37)。読み込むときに1回だけ、公式と同じ方法(bicubic・antialias なし・
scale_factor = (格子 + 0.1) / 37)で img_size // 14 の格子に補間して **パラメータとして持つ**。だからグラフに補間の演算が残らず
(ONNX・TensorRT では定数)、best.pth にも補間した後の値が入る(走行・評価では補間しない)。

注意は F.scaled_dot_product_attention(ONNX の書き出しで MatMul + Softmax に分かれる。TensorRT 10.16 で通ることを bench/ で確かめた)。
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 14
VITS14 = {'dim': 384, 'depth': 12, 'heads': 6, 'mlp_ratio': 4}
OFFICIAL_IMG_SIZE = 518                        # 公式の位置の埋め込みの大きさ(37×37 のパッチ)
INTERPOLATE_OFFSET = 0.1                       # 公式の通常版の補間の設定(レジスタ版は 0.0・antialias あり。この実装は通常版だけ)
LN_EPS = 1e-6


class _PatchEmbed(nn.Module):
    """画像 → パッチのトークン(Conv 14×14・stride 14)。[B,3,H,W] → [B,(H/14)·(W/14),dim](行の順 = 上から、各行は左から)"""

    def __init__(self, dim=384, patch=PATCH):
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
        # 公式と同じ分け方: [B,N,3,ヘッド,ch] → (q, k, v) 各 [B,ヘッド,N,ch]。スケールは 1/sqrt(ch)(SDPA の既定 = 公式の self.scale)
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))        # 公式の hub は init_values = 1.0

    def forward(self, x):
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)   # GELU は erf 版(既定)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    """前置き LayerNorm の Transformer のブロック(公式の Block。推論では drop_path なし)"""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp = _Mlp(dim, dim * mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoViTS14(nn.Module):
    """DINOv2 ViT-S/14(dinov2_vits14、レジスタなし)。forward(x [B,3,img,img] ImageNet 正規化)→ 最後の LayerNorm の後
    [B, 1 + grid², dim](先頭が CLS、残りはパッチを行の順に)。

    depth はテストで小さい ViT を作るためだけ(公式の重みは 12)。img_size は 14 の倍数(224 → 16×16 = 256 パッチ)"""

    def __init__(self, img_size=224, dim=VITS14['dim'], depth=VITS14['depth'], heads=VITS14['heads'],
                 mlp_ratio=VITS14['mlp_ratio']):
        super().__init__()
        if img_size % PATCH:
            raise ValueError('DINOv2 の入力の大きさは %d の倍数にしてください(%d)' % (PATCH, img_size))
        if depth < 0:
            raise ValueError('ViT のブロックの数は 0 以上(%d)' % depth)
        self.img_size, self.dim = img_size, dim
        self.grid = img_size // PATCH
        self.patch_embed = _PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=LN_EPS)
        # 乱数の初期値(公式と同じ。公式の重みを読まないとき = 評価で best.pth を読む前・テストだけで使う)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    # ---------------------------------------------------------------- 位置の埋め込みの補間(読み込みのときだけ)
    @staticmethod
    def interpolate_pos(pos, grid, offset=INTERPOLATE_OFFSET):
        """公式の位置の埋め込み [1, 1 + M², C](M×M の格子)→ [1, 1 + grid², C]。公式の interpolate_pos_encoding と同じ計算
        (float32・bicubic・antialias なし・scale_factor = (grid + offset) / M。scale_factor は出力の座標の計算にも使われるので、
        size = grid を渡すのとはわずかに違う)。M == grid ならそのまま"""
        n = pos.shape[1] - 1
        m = int(round(math.sqrt(n)))
        if m * m != n:
            raise ValueError('位置の埋め込みのパッチの数 %d が正方形の格子ではありません' % n)
        if m == grid:
            return pos.float()
        patch = pos[:, 1:].float().reshape(1, m, m, -1).permute(0, 3, 1, 2)
        s = float(grid + offset) / m
        patch = F.interpolate(patch, scale_factor=(s, s), mode='bicubic', antialias=False)
        if tuple(patch.shape[-2:]) != (grid, grid):
            raise RuntimeError('位置の埋め込みの補間の結果 %s が %d×%d ではありません' % (tuple(patch.shape[-2:]), grid, grid))
        return torch.cat([pos[:, :1].float(), patch.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)], dim=1)

    def convert_official(self, sd):
        """公式の state_dict → この実装の state_dict(mask_token を捨て、pos_embed をこの格子へ補間)"""
        if not isinstance(sd, dict) or 'pos_embed' not in sd:
            raise ValueError('DINOv2 の公式の重み(state_dict)ではありません(pos_embed がありません)')
        if 'register_tokens' in sd:
            raise ValueError('レジスタ版(vits14_reg4)の重みです。この実装は通常版(vits14)だけ読めます')
        sd = {k: v for k, v in sd.items() if k != 'mask_token'}
        sd['pos_embed'] = self.interpolate_pos(sd['pos_embed'], self.grid)
        return sd

    def load_official(self, path):
        """公式の重みのファイル(dinov2_vits14_pretrain.pth)を読む。キーは過不足なく一致すること(strict=True)。self を返す"""
        sd = torch.load(str(Path(path)), map_location='cpu', weights_only=True)
        self.load_state_dict(self.convert_official(sd), strict=True)
        return self

    # ---------------------------------------------------------------- 順伝播
    def prepare_tokens(self, x):
        """[B,3,H,W] → CLS + パッチ + 位置の埋め込み [B, 1 + grid², dim](ブロックの前)"""
        return torch.cat([self.cls_token.expand(x.shape[0], -1, -1), self.patch_embed(x)], dim=1) + self.pos_embed

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
