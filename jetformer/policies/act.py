"""act — ACT 型(Action Chunking with Transformers、Zhao ほか 2023)の方策(spec.md §3.1)

v0 と同じセンサーのトークン(画像 49・ToF 42・IMU・操作値・ジャンル)を Encoder で融合し、学習できる 10 個の
行動クエリ(1 クエリ = 1 フレーム先)が Decoder で Encoder の出力(memory)を参照して、クエリごとに [ステア, スロットル] を出す:

    blocks.sensor_tokens → [B,94,D](+ CVAE のときだけ潜在トークン 1)+ pos [1,94(+1),D]
      → encoder(D, heads, ffn, enc_layers) → enc_norm = memory
    queries [1,10,D] → decoder(D, heads, ffn, dec_layers)(クエリどうしの自己注意 + memory への交差注意)→ dec_norm → hs [B,10,D]
      → heads: Linear(D, M×2)(10 クエリで共有)→ view(B,10,M,2).transpose(1,2) → select_mode → actions [B,10,2]

v0 との違いは「1 つの出力トークン → trunk → 20 個の数」が「10 個のクエリ → 各 2 個」になることだけ(トークン化・層の名前は同じ)。
出力の意味も v0 と同じ: actions [B,10,2] = 33 ms 刻みの 1〜10 フレーム先の [ステア, スロットル]((µs − 1500)/500)。
ステア > 中心 = 左、スロットルは中立より下 = 前進(学習の正解をそのまま写すので、符号の決まりは v0 と同じ)。

CVAE(既定オフ、比較の本体には入れない参考): cvae=True のとき学習中だけ、[cvae_cls・ジャンル(mode_embed)・正解の行動 10 個]
を小さな Encoder(cvae_encoder)に通して潜在変数 z(latent_dim 次元)を作り、latent_proj(z) を Encoder の 1 トークンにする。
損失は L1 + kl_weight·KL。推論・検証・書き出しは z = 0(潜在トークン = latent_proj のバイアスの定数)なので、
書き出すグラフに CVAE の符号化器は入らない。

予測の重ね合わせ(temporal ensembling。ACT の元の論文の手法)は方策に依存しない走行側の処理として
jetformer/policies/ensemble.py にある(既定オフ、比較の判定には使わない)。このモジュールは毎フレーム 10 ステップを出すだけ。
"""

import torch
import torch.nn as nn

from . import FUTURE_STEPS, blocks

LOGVAR_CLIP = 10.0          # CVAE の logσ² を [−10, 10] に切る(exp のあふれ・つぶれを防ぐ)


class ACTPolicy(nn.Module):
    """ACT 型(CVAE なし)。forward(image, tof, imu, state, mode) → actions [B,10,2]

    層の名前: image_backbone・image_proj・tof_encoder・imu_mlp・state_mlp・mode_embed(v0 と同じ)、
    pos・encoder・enc_norm・queries・decoder・dec_norm・heads。heads の行 = ジャンル×2 + [ステア, スロットル]
    (登録簿の genre_keys は {'heads': 2})"""

    EXPORT_OUTPUTS = ('actions',)
    USES_TARGET = False
    N_LATENT = 0                # Encoder に足す潜在トークンの数(CVAE で 1)

    def __init__(self, num_modes, dim=256, enc_layers=4, dec_layers=2, heads=8, ffn=512, dropout=0.1, pretrained=False):
        super().__init__()
        if dim % heads:
            raise ValueError('ACT 型: dim(%d)は heads(%d)で割り切れる数にしてください' % (dim, heads))
        if min(enc_layers, dec_layers) < 1:
            raise ValueError('ACT 型: enc_layers・dec_layers は 1 以上にしてください')
        self.dim = dim
        blocks.add_sensor_tokens(self, dim, num_modes, pretrained)      # v0 と同じ名前・同じ初期化(num_modes も置く)
        self.pos = nn.Parameter(torch.zeros(1, blocks.N_SENSOR_TOKENS + self.N_LATENT, dim))
        self.encoder = blocks.encoder(dim, heads, ffn, enc_layers, dropout)
        self.enc_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.zeros(1, FUTURE_STEPS, dim))    # 1 クエリ = 1 フレーム先
        self.decoder = blocks.decoder(dim, heads, ffn, dec_layers, dropout)
        self.dec_norm = nn.LayerNorm(dim)
        self.heads = nn.Linear(dim, num_modes * 2)                         # クエリで共有・ジャンル別 [ステア, スロットル]
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

    # ---------------------------------------------------------------- 計算
    def latent_token(self, b, mode, target):
        """Encoder に足す潜在トークン [B,1,D](CVAE だけ)。CVAE なしは None"""
        return None

    def decode(self, image, tof, imu, state, mode, target=None):
        """→ hs [B,10,D](dec_norm の後。クエリ k = k+1 フレーム先)"""
        b = image.shape[0]
        tok = blocks.sensor_tokens(self, image, tof, imu, state, mode)        # [B,94,D]
        lat = self.latent_token(b, mode, target)
        if lat is not None:
            tok = torch.cat([tok, lat.to(tok.dtype)], dim=1)                   # [B,95,D]
        memory = self.enc_norm(self.encoder(tok + self.pos))
        return self.dec_norm(self.decoder(self.queries.expand(b, -1, -1), memory))

    def forward(self, image, tof, imu, state, mode, target=None):
        hs = self.decode(image, tof, imu, state, mode, target)
        b = hs.shape[0]
        all_modes = self.heads(hs).view(b, FUTURE_STEPS, self.num_modes, 2).transpose(1, 2)   # [B,M,10,2]
        return blocks.select_mode(all_modes, mode)                                           # [B,10,2]

    def embed(self, image, tof, imu, state, mode):
        """tools/offline_rl.py features 用の特徴 [B,D](10 個の行動クエリの Decoder の出力の平均。z = 0)"""
        return self.decode(image, tof, imu, state, mode).mean(dim=1)


class ACTCVAEPolicy(ACTPolicy):
    """ACT 型 + CVAE(学習のときだけの潜在変数。推論は z = 0)。

    学習中に target [B,10,2] があれば: [cvae_cls・mode_embed(mode)・cvae_action(target) × 10] + cvae_pos [1,12,D]
    → cvae_encoder(cvae_layers 層)→ 先頭 → cvae_out → μ・logσ²(logσ² は [−10, 10] に切る)→ z = μ + ε·exp(logσ²/2)。
    サンプルごとの KL [B] を覚えておき、policy_loss で L1 + kl_weight·KL にする(loss_parts = {'kl': 平均})。
    それ以外(検証・推論・書き出し・target なし)は z = 0 → 潜在トークン = latent_proj のバイアス"""

    USES_TARGET = True
    N_LATENT = 1

    def __init__(self, num_modes, dim=256, enc_layers=4, dec_layers=2, heads=8, ffn=512, dropout=0.1, latent_dim=32,
                 kl_weight=10.0, cvae_layers=2, pretrained=False):
        super().__init__(num_modes, dim, enc_layers, dec_layers, heads, ffn, dropout, pretrained)
        if latent_dim < 1 or cvae_layers < 1:
            raise ValueError('ACT 型の CVAE: latent_dim・cvae_layers は 1 以上にしてください')
        self.latent_dim = latent_dim
        self.kl_weight = float(kl_weight)
        self.cvae_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.cvae_action = nn.Linear(2, dim)                                  # 正解の行動 1 ステップ → 1 トークン
        self.cvae_pos = nn.Parameter(torch.zeros(1, 2 + FUTURE_STEPS, dim))   # [CLS]・ジャンル・行動 10
        self.cvae_encoder = blocks.encoder(dim, heads, ffn, cvae_layers, dropout)
        self.cvae_out = nn.Linear(dim, 2 * latent_dim)                        # → μ・logσ²
        self.latent_proj = nn.Linear(latent_dim, dim)                         # z → Encoder の 1 トークン
        nn.init.trunc_normal_(self.cvae_cls, std=0.02)
        nn.init.trunc_normal_(self.cvae_pos, std=0.02)
        self._kl = None             # 直前の学習の forward のサンプルごとの KL [B](float32)
        self.loss_parts = {}

    def latent_token(self, b, mode, target):
        self._kl = None
        if not (self.training and target is not None):
            # z = 0: latent_proj(0) = バイアス。書き出しでは定数になる(CVAE の符号化器はグラフに入らない)
            return self.latent_proj.bias.view(1, 1, -1).expand(b, -1, -1)
        c = torch.cat([self.cvae_cls.expand(b, -1, -1), self.mode_embed(mode).unsqueeze(1), self.cvae_action(target)],
                      dim=1) + self.cvae_pos                                  # [B,12,D]
        mu, logvar = self.cvae_out(self.cvae_encoder(c)[:, 0]).float().chunk(2, dim=-1)
        logvar = logvar.clamp(-LOGVAR_CLIP, LOGVAR_CLIP)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        self._kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)     # [B](float32)
        return self.latent_proj(z).unsqueeze(1)

    def policy_loss(self, out, target, extras):
        """サンプルごとの損失 [B] = L1 + kl_weight·KL(KL は学習中に target を渡した forward のときだけ。ほかは L1)"""
        l1 = (out['actions'].float() - target.float()).abs().mean(dim=(1, 2))
        kl, self._kl = self._kl, None
        if kl is None:
            self.loss_parts = {}
            return l1
        self.loss_parts = {'kl': kl.detach().mean()}
        return l1 + self.kl_weight * kl


def build(num_modes, pretrained=False, dim=256, enc_layers=4, dec_layers=2, heads=8, ffn=512, dropout=0.1, cvae=False,
          latent_dim=32, kl_weight=10.0, cvae_layers=2):
    """登録簿から呼ばれる(設定は jetformer/policies/__init__.py の act の defaults の全部のキー)。
    pretrained=True のときだけ ResNet-18 の ImageNet の重み(torchvision のキャッシュ)を読む"""
    common = dict(dim=int(dim), enc_layers=int(enc_layers), dec_layers=int(dec_layers), heads=int(heads), ffn=int(ffn),
                  dropout=float(dropout), pretrained=bool(pretrained))
    if cvae:
        return ACTCVAEPolicy(int(num_modes), latent_dim=int(latent_dim), kl_weight=float(kl_weight),
                             cvae_layers=int(cvae_layers), **common)
    return ACTPolicy(int(num_modes), **common)
