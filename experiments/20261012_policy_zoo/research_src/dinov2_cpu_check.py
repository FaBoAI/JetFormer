#!/usr/bin/env python3
"""DINOv2 ViT-S/14 を、torch.hub のコードを使わない最小の実装で公式の重みから読み、CPU で次を確かめる
(GPU・TensorRT は使わない。1スレッド・nice で動かす)。

    nice -n 19 python dinov2_cpu_check.py <ckpt.pth> [<ckpt_reg4.pth>] > dinov2_cpu_check.json

  1. 重みのキーが最小の実装と過不足なく合うか(strict=True で読めるか)
  2. 224×224(16×16 パッチ)へ位置埋め込みを補間したときの形
  3. FP16 の余裕: 実際の車の画像(clean_driving_20261011 の val から 16 枚)で、各ブロックの
     注意のロジット(q·kᵀ·scale)・残差の値・MLP の中間の最大絶対値。FP16 の最大 65504 と比べる
  4. ONNX(opset 17・dynamo=False)へ書き出せるか、演算の種類とノード数(TensorRT の確認は別。GPU が要る)
"""
import io
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = '/home/jetson/jetformer/data/datasets/clean_driving_20261011'
MEAN, STD = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.stats = None

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        logits = (q * self.scale) @ k.transpose(-2, -1)          # 明示的な注意(ONNX で MatMul + Softmax になる)
        if self.stats is not None:
            self.stats['logit_absmax'] = max(self.stats.get('logit_absmax', 0.0), float(logits.abs().max()))
            self.stats['qk_absmax'] = max(self.stats.get('qk_absmax', 0.0), float(q.abs().max()), float(k.abs().max()))
        return self.proj((logits.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.stats = None

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        if self.stats is not None:
            self.stats['mlp_hidden_absmax'] = max(self.stats.get('mlp_hidden_absmax', 0.0), float(h.abs().max()))
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.attn, self.ls1 = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads), LayerScale(dim)
        self.norm2, self.mlp, self.ls2 = nn.LayerNorm(dim, eps=1e-6), Mlp(dim, dim * 4), LayerScale(dim)
        self.stats = None

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        if self.stats is not None:
            self.stats['resid_absmax'] = max(self.stats.get('resid_absmax', 0.0), float(x.abs().max()))
            self.stats['resid_token_norm_max'] = max(self.stats.get('resid_token_norm_max', 0.0), float(x.norm(dim=-1).max()))
        return x


class ViTS14(nn.Module):
    """DINOv2 ViT-S/14 の最小の実装(キーは公式の重みと同じ)。224px 固定: 位置埋め込みは読み込みのときに 16×16 へ補間して持つ"""
    def __init__(self, registers=0, grid=16, dim=384, depth=12, heads=6):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, 14, 14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim)) if registers else None
        self.mask_token = nn.Parameter(torch.zeros(1, dim))           # 使わない(重みを strict で読むためだけ)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.grid = grid

    def load_official(self, sd, antialias, offset):
        pe = sd['pos_embed']
        m = int(math.sqrt(pe.shape[1] - 1))
        patch = pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2)
        kw = dict(scale_factor=((self.grid + offset) / m,) * 2) if offset else dict(size=(self.grid, self.grid))
        patch = F.interpolate(patch.float(), mode='bicubic', antialias=antialias, **kw)
        sd = dict(sd)
        sd['pos_embed'] = torch.cat([pe[:, :1], patch.permute(0, 2, 3, 1).reshape(1, self.grid * self.grid, -1)], 1)
        return self.load_state_dict(sd, strict=True), m

    def forward(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        if self.register_tokens is not None:
            x = torch.cat([x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def images(n=16):
    sp = json.load(open(os.path.join(ROOT, 'splits.json')))['val']
    out = []
    rng = np.random.default_rng(0)
    for s in sp:
        a = np.load(os.path.join(ROOT, 'aligned', s + '.npz'))
        for i in rng.choice(len(a['cam_files']), 2, replace=False):
            im = cv2.cvtColor(cv2.imread(os.path.join(ROOT, str(a['cam_files'][i]))), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA) if im.shape[:2] != (224, 224) else im
            out.append(((im / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))
    return torch.from_numpy(np.stack(out[:n]))


def check(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    reg = int(sd['register_tokens'].shape[1]) if 'register_tokens' in sd else 0
    model = ViTS14(registers=reg).eval()
    # 公式の hub: 通常版は antialias=False・offset=0.1、レジスタ版は antialias=True・offset=0.0
    res, m = model.load_official(sd, antialias=bool(reg), offset=0.0 if reg else 0.1)
    info = {'file': os.path.basename(path), 'registers': reg, 'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            'pos_embed_src_grid': m, 'load_strict': str(res), 'ls_gamma_absmax': round(max(float(b.ls1.gamma.abs().max()) for b in model.blocks), 3)}
    for i, b in enumerate(model.blocks):
        b.stats = {}; b.attn.stats = b.stats; b.mlp.stats = b.stats
    x = images()
    with torch.no_grad():
        y = model(x)
    info['blocks'] = [{k: round(v, 1) for k, v in b.stats.items()} for b in model.blocks]
    info['max_over_blocks'] = {k: round(max(b.stats[k] for b in model.blocks), 1) for k in model.blocks[0].stats}
    info['fp16_max'] = 65504.0
    # 高いノルムのパッチ(レジスタ論文の「人工物」)の目安: 最終層の前の残差でノルムが中央値の 4倍を超えるパッチの割合
    with torch.no_grad():
        h = model.patch_embed.proj(x).flatten(2).transpose(1, 2)
        h = torch.cat([model.cls_token.expand(h.shape[0], -1, -1), h], 1) + model.pos_embed
        if reg:
            h = torch.cat([h[:, :1], model.register_tokens.expand(h.shape[0], -1, -1), h[:, 1:]], 1)
        for b in model.blocks[:-1]:
            b.stats = None; b.attn.stats = None; b.mlp.stats = None
            h = b(h)
        nrm = h[:, 1 + reg:].norm(dim=-1)
        info['high_norm_patch_frac(>4x median)'] = round(float((nrm > 4 * nrm.median()).float().mean()), 4)
    # FP16 で CPU 推論(TensorRT の代わりではないが、演算の範囲の目安)と FP32 の差
    try:
        with torch.no_grad():
            y16 = model.half()(x.half()).float()
        model.float()
        info['cpu_fp16_vs_fp32_cls_cos'] = round(float(F.cosine_similarity(y16[:, 0], y[:, 0], dim=-1).min()), 5)
        info['cpu_fp16_nan'] = bool(torch.isnan(y16).any())
    except Exception as e:                                   # CPU の half が使えない版もある
        model.float()
        info['cpu_fp16'] = 'skip: %s' % e
    # ONNX
    for b in model.blocks:
        b.stats = None; b.attn.stats = None; b.mlp.stats = None
    buf = io.BytesIO()
    torch.onnx.export(model, (x[:1],), buf, input_names=['image'], output_names=['tokens'], opset_version=17, dynamo=False)
    import onnx
    g = onnx.load_from_string(buf.getvalue()).graph
    ops = {}
    for nd in g.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    info['onnx'] = {'nodes': len(g.node), 'ops': dict(sorted(ops.items(), key=lambda kv: -kv[1])),
                    'output_shape': [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]}
    return info


if __name__ == '__main__':
    json.dump([check(p) for p in sys.argv[1:]], sys.stdout, ensure_ascii=False, indent=1)
