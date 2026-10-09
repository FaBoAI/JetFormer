"""JetFormer v0: カメラ・ToF・IMU・操作値をTransformerで融合し、未来の操作列を出力する

TransFuserと同じく、各センサーを別々にトークン化してTransformerで相互参照させる。
Orin Nano(30fps)向けに、マルチスケール融合ではなく最終段のトークン融合1回に絞った軽量版。

    画像   3x224x224 → ResNet-18 → 512x7x7 → 1x1conv → 49トークン
    ToF    1x42x54   → 小CNN(stride 8)   → 6x7       → 42トークン
    IMU    10x6(100ms履歴)               → MLP       → 1トークン
    操作値 [steer, throttle, motor]       → MLP       → 1トークン
    + 出力用トークン(学習可能) 1
    → Transformer Encoder ×4 (d=256) → 出力トークン → MLP → [FUTURE_STEPS, 2]
"""

import torch
import torch.nn as nn
import torchvision

IMAGE_SIZE = 224
TOF_SHAPE = (42, 54)
IMU_STEPS = 10
FUTURE_STEPS = 10

# 正規化(PWMはµs、ToFはmm)
PWM_CENTER, PWM_SCALE = 1500.0, 500.0
TOF_MAX_MM = 4000.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ToFEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.net = nn.Sequential(block(2, 32, 1), block(32, 64, 2), block(64, 128, 2), block(128, dim, 2))

    def forward(self, depth):                # depth: [B,1,42,54] 正規化済み(0〜1)
        valid = (depth > 0).to(depth.dtype)   # 0 = 無効ゾーン
        return self.net(torch.cat([depth, valid], dim=1))   # [B,dim,6,7]


class JetFormer(nn.Module):
    def __init__(self, dim=256, layers=4, heads=8, pretrained=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-2])      # [B,512,7,7]
        self.image_proj = nn.Conv2d(512, dim, 1)
        self.tof_encoder = ToFEncoder(dim)
        self.imu_mlp = nn.Sequential(nn.Linear(IMU_STEPS * 6, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.state_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        n_tokens = 49 + 42 + 3
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.modality = nn.Parameter(torch.zeros(5, dim))        # 画像/ToF/IMU/操作値/出力
        self.register_buffer('modality_index', torch.tensor([0] * 49 + [1] * 42 + [2, 3, 4]), persistent=False)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, FUTURE_STEPS * 2))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.modality, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, image, tof, imu, state):
        """
        image: [B,3,224,224] ImageNet正規化済みRGB
        tof:   [B,1,42,54]   距離/TOF_MAX_MM(0=無効)
        imu:   [B,10,6]      accel(m/s^2)/9.8, gyro(rad/s)
        state: [B,3]         (steer_us-1500)/500, (throttle_us-1500)/500, motor_pct/100
        戻り値: [B,FUTURE_STEPS,2] 正規化した未来の [steering, throttle]
        """
        b = image.shape[0]
        img = self.image_proj(self.image_backbone(image)).flatten(2).transpose(1, 2)     # [B,49,D]
        tof_t = self.tof_encoder(tof).flatten(2).transpose(1, 2)                         # [B,42,D]
        imu_t = self.imu_mlp(imu.flatten(1)).unsqueeze(1)
        st_t = self.state_mlp(state).unsqueeze(1)
        tokens = torch.cat([img, tof_t, imu_t, st_t, self.query.expand(b, -1, -1)], dim=1)
        tokens = tokens + self.pos + self.modality[self.modality_index]
        out = self.norm(self.fusion(tokens))[:, -1]
        return self.head(out).view(b, FUTURE_STEPS, 2)
