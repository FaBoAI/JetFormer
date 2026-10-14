"""収集時の PWM 換算だけをブリッジに渡す「走らないポリシー」(2026-10-13、強化学習のベストラップ)

    python -I run_bridge.py --port 8791 --no-browser --policy calibration_only_policy.py

  - 元の20分データ(ID8)は、モデルをブリッジに接続した状態で教師(デモ)を収集したため、
    PWM の換算に「そのモデルの校正・設定」(操舵中心 1442 µs・スロットル中立 1498 µs・最大 0.52 rad・7.24 m/s)が使われている。
    接続なしで収集すると収集器の既定値(1441 / 1496 µs)になり、PWM が 1〜2 µs ずれる。
  - ブリッジの公式の拡張点(--policy の POLICY_INFO)で同じ校正・設定を渡し、アプリの「接続を確認」で読み込ませる。
    走行は教師(コントローラー=経路追従デモ)が行い、この predict は呼ばれない(呼ばれたらエラー)。
"""
import json
from pathlib import Path

_SRC = json.loads((Path(__file__).resolve().parent / 'id8_original_calibration.json').read_text(encoding='utf-8'))
POLICY_NAME = 'ID8元データと同じPWM換算(走行しない)'
CAPABILITIES = {'camera': False, 'ranges': False, 'depth': False, 'imu': False}
POLICY_INFO = {'model_name': 'calibration-only (no driving)', 'mode_id': 4, 'mode_name': '高速走行',
               'trained_modes': _SRC['model_config']['trained_modes'], 'command_input': False,
               'calibration': _SRC['calibration'], 'model_config': _SRC['model_config'],
               'calibration_source': _SRC['source'], 'calibration_source_sha256': _SRC['source_sha256']}


def predict(observation):
    raise RuntimeError('このポリシーは収集時の PWM 換算を渡すためだけのもので、走行には使いません')
