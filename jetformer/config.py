"""JetFormerの設定(環境変数で上書き可能)"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('JETFORMER_DATA', ROOT / 'data'))
RAW_DIR = DATA_DIR / 'raw'            # 収集した生データ(ログ/シーン)
DATASET_DIR = DATA_DIR / 'datasets'   # nuScenes形式に組み上げたデータセット
EXPORT_DIR = DATA_DIR / 'exports'     # クラウド学習用アーカイブ

WEB_HOST = os.environ.get('JETFORMER_HOST', '0.0.0.0')
WEB_PORT = int(os.environ.get('JETFORMER_PORT', '8080'))

SCENE_SECONDS = float(os.environ.get('JETFORMER_SCENE_SECONDS', '20'))
VEHICLE_NAME = os.environ.get('JETFORMER_VEHICLE', 'jetformer-01')

# ---- センサー ----
CAMERA = {
    'device': 0,
    'width': 224, 'height': 224,   # 推論(30fps)と同じ解像度で保存する
    'fps': 30,
    'sensor_mode': 2,              # IMX219 1920x1080@30 (JetRacerの録画・走行と同じ画角)
    'jpeg_quality': 92,
}
TOF = {
    'port': os.environ.get('JETFORMER_TOF_PORT', '/dev/ttyACM0'),
    'baudrate': 115200,            # USB CDCなので実際の速度には影響しない
    'unit': 'mm',                  # 要確認(L9FRの距離値の単位)
}
IMU = {
    'bus': 7, 'address': 0x68,     # ICM-20948
    'rate_hz': 100,
    'accel_range_g': 8,            # 衝突も記録できるよう±8g
    'gyro_range_dps': 1000,
    'mag': False,                  # AK09916(バイパス経由 0x0C)。現状バス上に現れないため無効
}
VEHICLE = {
    'bus': 7, 'address': 0x28,     # JetFormerコントロール基板(ESP32 I2Cスレーブ)
    'rate_hz': 50,
    # 'usb'   : 推奨。ToFと同じUSBストリームの 'JFVH' パケット(修正版ファーム)
    # 'frame' : I2Cでスナップショットフレームを読む(修正版ファーム)
    # 'legacy': 現行ファーム。レジスタ毎にwrite→read(チャネルの取り違えが頻発し学習には使えない)
    'protocol': os.environ.get('JETFORMER_VEHICLE_PROTOCOL', 'legacy'),
    'channels': {'steering': 0x00, 'throttle': 0x01, 'aux': 0x02, 'motor_pct': 0x10},
}

# 取り付け位置(車体座標: x前方, y左, z上 [m])。カメラとToFは5cm以内に近接設置。
# 実測値で更新すること。回転はクォータニオン[w, x, y, z]。
CALIBRATION = {
    'CAM_FRONT': {'translation': [0.10, 0.0, 0.12], 'rotation': [1.0, 0.0, 0.0, 0.0]},
    'TOF_FRONT': {'translation': [0.10, -0.03, 0.12], 'rotation': [1.0, 0.0, 0.0, 0.0]},
    'IMU': {'translation': [0.0, 0.0, 0.08], 'rotation': [1.0, 0.0, 0.0, 0.0]},
}
