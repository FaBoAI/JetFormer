"""ens_average: 学習済みの方策の actions の平均(アンサンブル。2026-10-21)

    model = policies.build_policy('ens_average', 6, {'members': [{'run': ..., 'policy': 'jetformer_v0', 'policy_cfg': {...}},
                                                                 {'run': ..., 'policy': 'msfusion', 'policy_cfg': {...}}]})
    model.load_state_dict(torch.load('<学習結果>/best.pth'))       # キーは ms.0.*・ms.1.*(メンバーの best.pth をまとめたもの)

forward(image, tof, imu, state, mode)(キーワード)→ actions [B,10,2] = メンバーの actions の平均(正規化のまま)。
メンバーの追加の出力(TCP の通過点など)・後処理は使わない。experiments/20261015_real_accuracy/ens/ens_gate.py の
make_average_module と同じ形・同じキー(10/15 に作ったエンジン ens_p2_fp16.engine はこの形から書き出した)。
学習はしない(登録簿の trainable False)。重みは experiments/20261021_bench_prep/ens/place_ens.py がメンバーの best.pth から作る。
"""

import torch
import torch.nn as nn

from . import get
from .runtime import as_outputs


class EnsAverage(nn.Module):
    EXPORT_OUTPUTS = ('actions',)

    def __init__(self, members):
        super().__init__()
        self.ms = nn.ModuleList(members)

    def forward(self, image, tof, imu, state, mode):
        acts = [as_outputs(m(image=image, tof=tof, imu=imu, state=state, mode=mode))['actions'] for m in self.ms]
        return torch.stack(acts, 0).mean(0)


def build(num_modes, pretrained=False, members=(), member_runs=()):
    """members: [{'run', 'policy', 'policy_cfg'}](2 本以上)。メンバーの構造だけを作る(重みは読まない・pretrained も使わない)"""
    members = list(members or [])
    if len(members) < 2:
        raise ValueError('ens_average のメンバーは 2 本以上(policy_cfg の members)にしてください(%d 本)' % len(members))
    ms = []
    for m in members:
        if not isinstance(m, dict) or 'policy' not in m:
            raise ValueError('ens_average のメンバーは {"run", "policy", "policy_cfg"} の形にしてください: %r' % (m,))
        if m['policy'] == 'ens_average':
            raise ValueError('アンサンブルの入れ子は作りません')
        ms.append(get(m['policy']).build(int(num_modes), m.get('policy_cfg'), pretrained=False))
    return EnsAverage(ms)


def from_members(models):
    """読み込み済みのメンバーのモデル → EnsAverage(重みはそのまま。place_ens.py が best.pth を作るとき)"""
    return EnsAverage(list(models))
