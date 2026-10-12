"""dinov2 — DINOv2 ViT-S/14(凍結)の画像特徴 + v0 の ToF・IMU・操作値・ジャンルのトークン(spec.md §3.4)

    image [B,3,224,224](今の ImageNet 正規化のまま。DINOv2 と同じ平均・標準偏差)
      → vit(DINOv2 ViT-S/14、凍結)→ 最後の LayerNorm → CLS [B,1,384] + パッチ [B,256,384]
      → パッチを 16×16 に並べて pool×pool の平均(既定 2 → 8×8 = 64。pool 1 なら 256)
      → image_proj(LayerNorm(384) → Linear(384, dim))→ 65 トークン(use_cls false なら 64)
    tof_encoder(v0 の ToFEncoder)42・imu_mlp・state_mlp・mode_embed 各1・query 1 → + pos [1,111,dim]
      → fusion = Encoder(dim, heads, 2·dim, layers)→ norm → 最後のトークン → trunk → heads Linear(dim, M×20) → ジャンルを選ぶ
    → actions [B,10,2](v0 と同じ意味・同じ正規化)

層の名前: vit・image_proj・tof_encoder・imu_mlp・state_mlp・mode_embed・query・pos・fusion・norm・trunk・heads
(ジャンルの層は v0 と同じ mode_embed.weight・heads = genre_keys)。

凍結: finetune_last = 0(既定)で ViT の全パラメータが requires_grad=False。学習のときも ViT は eval で、凍結した部分は no_grad
(train() を上書き。DINOv2 に dropout は無いので、eval にしても数値は同じ)。finetune_last = k で最後の k ブロックと vit.norm を学習し、
param_groups(lr) で学習率 lr × lr_mult の群にする(LP-FT の2段目: 凍結で学習した best.pth から --init で始める。
finetune_last・lr_mult は train_only_keys なので --init の構造の確認を通る)。

重み: build(pretrained=True) のときだけ ROOT / weights(公式の dinov2_vits14_pretrain.pth)を読む。**ネットから自動では取らない**。
build(pretrained=False)(runtime.load_run・評価の道具・テスト)は重みのファイルに触れない(best.pth に ViT の重みも入る。約98 MB)。
取得の手順(1回だけ。電源断で半端なファイルを残さないよう .part に書いてから名前を変える):

    cd ~/jetformer && mkdir -p data/pretrained && \
    curl -fL -o data/pretrained/dinov2_vits14_pretrain.pth.part \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth && \
    sha256sum data/pretrained/dinov2_vits14_pretrain.pth.part   # b938bf1b…aef60cd9(WEIGHTS_SHA256)と同じか確かめる
    mv data/pretrained/dinov2_vits14_pretrain.pth.part data/pretrained/dinov2_vits14_pretrain.pth

(data/ は .gitignore に入っているので git には入らない。DINOv2 のコードと重みは Apache-2.0)

DINOv2 の特徴はキャッシュしない(学習の色の揺らぎと両立しないため。spec.md §8)。
"""

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import FUTURE_STEPS, IMU_STEPS, ToFEncoder
from . import ROOT
from .blocks import encoder, mlp2, select_mode
from .vit import PATCH, VITS14, DinoViTS14

VARIANTS = ('vits14',)                                   # 通常版だけ(レジスタ版は data/pretrained/ に無い。spec.md §0.3)
DEFAULT_WEIGHTS = 'data/pretrained/dinov2_vits14_pretrain.pth'
WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
WEIGHTS_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
WEIGHTS_BYTES = 88283115
OFFICIAL_DEPTH = VITS14['depth']
IMAGE_SIZE = 224
N_TOF_TOKENS = 42                                        # ToFEncoder の 6×7
N_OTHER_TOKENS = 4                                       # IMU・操作値・ジャンル・出力用(query)


def weights_path(weights=DEFAULT_WEIGHTS):
    """設定の weights(リポジトリからの相対パス、または絶対パス)→ Path"""
    p = Path(os.path.expanduser(str(weights)))
    return p if p.is_absolute() else ROOT / p


def missing_message(path):
    return 'DINOv2 の重みがありません: %s(取得の手順は README・jetformer/policies/dinov2.py の先頭。公式 %s)' % (path, WEIGHTS_URL)


def image_token_count(pool=2, use_cls=True, img_size=IMAGE_SIZE):
    """画像のトークンの数(CLS を含む)。224 px・pool 2 → 65、pool 1 → 257"""
    g = img_size // PATCH
    return (g // pool) ** 2 + (1 if use_cls else 0)


def pool_patches(patches, grid, pool):
    """パッチのトークン [B, grid², C](行の順 = 上から、各行は左から)→ pool×pool の平均 [B, (grid/pool)², C](同じ並び)。
    固定の F.avg_pool2d(ONNX の AveragePool)。adaptive_avg_pool2d は使わない(spec.md §8)"""
    if pool == 1:
        return patches
    c = patches.shape[-1]
    p = patches.transpose(1, 2).reshape(-1, c, grid, grid)
    return F.avg_pool2d(p, pool, pool).flatten(2).transpose(1, 2)


class DinoV2Policy(nn.Module):
    """DINOv2 ViT-S/14 の画像トークン + v0 の残りのトークン → v0 型の Encoder → 出力トークン → ジャンル別ヘッド(spec.md §3.4)"""

    EXPORT_OUTPUTS = ('actions',)

    def __init__(self, num_modes, pool=2, use_cls=True, finetune_last=0, lr_mult=0.1, dim=256, layers=4, heads=8,
                 vit_depth=OFFICIAL_DEPTH):
        super().__init__()
        grid = IMAGE_SIZE // PATCH
        if pool < 1 or grid % pool:
            raise ValueError('pool は %d を割り切る正の整数(指定 %s)' % (grid, pool))
        if not 0 <= finetune_last <= vit_depth:
            raise ValueError('finetune_last は 0〜%d(ViT のブロックの数)の整数(指定 %s)' % (vit_depth, finetune_last))
        if not lr_mult > 0:
            raise ValueError('lr_mult は正の数(指定 %s)' % lr_mult)
        self.num_modes = num_modes
        self.pool, self.use_cls = int(pool), bool(use_cls)
        self.finetune_last, self.lr_mult = int(finetune_last), float(lr_mult)
        self.vit = DinoViTS14(img_size=IMAGE_SIZE, depth=vit_depth)
        vdim = self.vit.dim
        self.n_frozen = vit_depth - self.finetune_last          # 前から何ブロックを凍結するか
        self.n_img = image_token_count(self.pool, self.use_cls)
        self.n_tokens = self.n_img + N_TOF_TOKENS + N_OTHER_TOKENS   # pool 2 → 111、pool 1 → 303
        self.image_proj = nn.Sequential(nn.LayerNorm(vdim), nn.Linear(vdim, dim))
        self.tof_encoder = ToFEncoder(dim)
        self.imu_mlp = mlp2(IMU_STEPS * 6, dim)
        self.state_mlp = mlp2(3, dim)
        self.mode_embed = nn.Linear(num_modes, dim, bias=False)       # one-hot → ジャンル埋め込み
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        self.fusion = encoder(dim, heads, dim * 2, layers)           # v0 と同じ(FFN 2×dim・dropout 0.1・前置き LayerNorm)
        self.norm = nn.LayerNorm(dim)
        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.heads = nn.Linear(dim, num_modes * FUTURE_STEPS * 2)   # ジャンル別ヘッド(v0 と同じ並び = 20 行/ジャンル)
        nn.init.normal_(self.mode_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)
        self._set_trainable()

    # ---------------------------------------------------------------- 凍結
    def _vit_trainable(self, name):
        """ViT のパラメータ名(self.vit からの相対)を学習するか。最後の finetune_last ブロックと norm だけ"""
        if self.finetune_last <= 0:
            return False
        if name.startswith('norm.'):
            return True
        if name.startswith('blocks.'):
            return int(name.split('.')[1]) >= self.n_frozen
        return False

    def _set_trainable(self):
        for name, p in self.vit.named_parameters():
            p.requires_grad_(self._vit_trainable(name))

    def train(self, mode=True):
        """ViT は学習中も eval のまま(凍結した部分の統計・挙動を変えない)"""
        super().train(mode)
        self.vit.eval()
        return self

    def param_groups(self, lr):
        """AdamW の群: [ViT 以外の学習するパラメータ(lr)] + 微調整のときだけ [ViT の学習するブロック・norm(lr × lr_mult)]。
        凍結したパラメータは入れない。並びは毎回同じ(--resume で optimizer の状態を読めるように)"""
        vit_ids = {id(p) for p in self.vit.parameters()}
        groups = [{'params': [p for p in self.parameters() if p.requires_grad and id(p) not in vit_ids]}]
        tuned = [p for p in self.vit.parameters() if p.requires_grad]
        if tuned:
            groups.append({'params': tuned, 'lr': lr * self.lr_mult})
        return groups

    # ---------------------------------------------------------------- 順伝播
    def image_tokens(self, image):
        """image [B,3,224,224] → DINOv2 の特徴 [B, n_img, 384](CLS + パッチの平均。最後の LayerNorm の後)"""
        v = self.vit
        h = v.prepare_tokens(image)
        if self.training:
            # 学習: 凍結した部分は勾配なし(活性を保存しない = メモリ・時間が小さい)。微調整するブロックだけ勾配あり
            with torch.no_grad():
                for blk in v.blocks[:self.n_frozen]:
                    h = blk(h)
                if self.finetune_last <= 0:
                    h = v.norm(h)
            h = h.detach()
            for blk in v.blocks[self.n_frozen:]:
                h = blk(h)
            if self.finetune_last > 0:
                h = v.norm(h)
        else:
            for blk in v.blocks:
                h = blk(h)
            h = v.norm(h)
        patches = pool_patches(h[:, 1:], v.grid, self.pool)
        return torch.cat([h[:, :1], patches], dim=1) if self.use_cls else patches

    def fused(self, image, tof, imu, state, mode):
        """融合の出力トークン(最後の LayerNorm の後、trunk の前)[B, dim]"""
        b = image.shape[0]
        img = self.image_proj(self.image_tokens(image))                               # [B,65,D]
        tof_t = self.tof_encoder(tof).flatten(2).transpose(1, 2)                      # [B,42,D]
        imu_t = self.imu_mlp(imu.flatten(1)).unsqueeze(1)
        st_t = self.state_mlp(state).unsqueeze(1)
        mode_t = self.mode_embed(mode).unsqueeze(1)
        tokens = torch.cat([img, tof_t, imu_t, st_t, mode_t, self.query.expand(b, -1, -1)], dim=1) + self.pos
        return self.norm(self.fusion(tokens))[:, -1]

    def forward(self, image, tof, imu, state, mode):
        """戻り値: actions [B,10,2](正規化した未来の [ステア, スロットル]。v0 と同じ)"""
        out = self.fused(image, tof, imu, state, mode)
        all_modes = self.heads(self.trunk(out)).view(out.shape[0], self.num_modes, FUTURE_STEPS, 2)
        return select_mode(all_modes, mode)

    def embed(self, image, tof, imu, state, mode):
        """tools/offline_rl.py features 用の特徴(v0 の model.norm のフックと同じ位置: 出力トークン 256 次元、trunk の前)"""
        return self.fused(image, tof, imu, state, mode)


def build(num_modes, pretrained=False, variant='vits14', weights=DEFAULT_WEIGHTS, pool=2, use_cls=True, finetune_last=0,
          lr_mult=0.1, dim=256, layers=4, heads=8, vit_depth=OFFICIAL_DEPTH):
    """登録簿から呼ばれる(spec.md §2.2)。pretrained=True のときだけ公式の DINOv2 の重みを読む(無ければ FileNotFoundError)。
    pretrained=False は重みのファイルに触れない(この後 best.pth を読む前提)"""
    if variant not in VARIANTS:
        raise ValueError('DINOv2 の variant は %s だけです(指定 %s)' % (', '.join(VARIANTS), variant))
    if pretrained and vit_depth != OFFICIAL_DEPTH:
        raise ValueError('vit_depth はテスト用の設定です。公式の重みを読むときは %d(指定 %s)' % (OFFICIAL_DEPTH, vit_depth))
    model = DinoV2Policy(num_modes, pool=pool, use_cls=use_cls, finetune_last=finetune_last, lr_mult=lr_mult, dim=dim,
                         layers=layers, heads=heads, vit_depth=vit_depth)
    if pretrained:
        path = weights_path(weights)
        if not path.is_file():
            raise FileNotFoundError(missing_message(path))
        model.vit.load_official(path)
    return model


# ---------------------------------------------------------------- 学習の前の確認(CPU。tools/train.py が1回呼ぶ)
_CHECKED = {}


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def prepare(data_root, cfg):
    """重みのファイルがあるか(無ければ SystemExit)と sha256(違えば警告して続ける)。設定(variant・vit_depth)も確かめる。
    何度呼んでもよい(同じファイル・同じ大きさ・同じ更新時刻なら sha256 を計算し直さない)。戻り値 {'weights', 'sha256', 'ok'}"""
    cfg = dict(cfg or {})
    variant = cfg.get('variant', 'vits14')
    if variant not in VARIANTS:
        raise SystemExit('DINOv2 の variant は %s だけです(指定 %s)' % (', '.join(VARIANTS), variant))
    if int(cfg.get('vit_depth', OFFICIAL_DEPTH)) != OFFICIAL_DEPTH:
        raise SystemExit('vit_depth はテスト用の設定です。学習では %d にしてください' % OFFICIAL_DEPTH)
    path = weights_path(cfg.get('weights', DEFAULT_WEIGHTS))
    if not path.is_file():
        raise SystemExit(missing_message(path))
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _CHECKED:
        digest = file_sha256(path)
        _CHECKED[key] = digest
        if digest != WEIGHTS_SHA256:
            print('警告: DINOv2 の重み %s の sha256 が公式と違います(%s、公式 %s)。このまま続けます' % (
                path, digest, WEIGHTS_SHA256), flush=True)
    digest = _CHECKED[key]
    return {'weights': str(path), 'sha256': digest, 'ok': digest == WEIGHTS_SHA256}
