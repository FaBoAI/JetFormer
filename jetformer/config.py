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
    'unit': 'mm',                  # 下位13bit=距離mm、上位3bit=信頼度(tof_format.py)
}
IMU = {
    'bus': 7, 'address': 0x68,     # ICM-20948
    'rate_hz': 100,
    'accel_range_g': 8,            # 衝突も記録できるよう±8g
    'gyro_range_dps': 1000,
    'mag': False,                  # AK09916(バイパス経由 0x0C)。現状バス上に現れないため無効
}
AUDIO = {
    # I2S MEMSマイク(40ピンヘッダのI2S2、pin38=DIN)。2026-10-10の確認では信号が来ていないため既定は無効。
    # マイクをつないだら JETFORMER_MIC=1 で有効にする。ピンの有効化は scripts/i2s_mic/enable.sh(要再起動)
    'enabled': os.environ.get('JETFORMER_MIC', '0') == '1',
    'card': 'APE', 'device': 'hw:APE,0',          # ADMAIF1
    # Jetsonがマスター、48kHz・32bit×2ch(BCLK 3.072MHz: I2S MEMSマイクの多くは2〜4MHzが必要)
    'mixer': [('ADMAIF1 Mux', 'I2S2'), ('I2S2 codec master mode', 'cbs-cfs'), ('I2S2 codec frame mode', 'i2s'),
              ('I2S2 Sample Rate', '48000'), ('I2S2 Sample Channels', '2'), ('I2S2 Sample Bits', '32'),
              ('I2S2 Loopback', 'off')],
    'capture_rate': 48000, 'rate': 16000,        # 48kHzで受けて16kHzに間引いて保存・配信
    'chunk_ms': 20,
    'channel': os.environ.get('JETFORMER_MIC_CHANNEL', 'auto'),   # 'auto' / 'left' / 'right'(マイクのL/R選択ピン)
}
SPEAKER = {
    # JetFormer基板のI2Sスピーカーアンプ(40ピンヘッダのI2S2、pin40=DOUT)。ピンの有効化は scripts/i2s_mic/enable.sh
    'card': 'APE', 'device': 'hw:APE,1',          # ADMAIF2 → I2S2
    'mixer': [('I2S2 Mux', 'ADMAIF2'), ('I2S2 codec master mode', 'cbs-cfs'), ('I2S2 codec frame mode', 'i2s'),
              ('I2S2 Sample Rate', '48000'), ('I2S2 Sample Channels', '2'), ('I2S2 Sample Bits', '32'),
              ('I2S2 Loopback', 'off')],
}
VEHICLE = {
    'bus': 7, 'address': 0x28,     # コントローラ基板(XIAO ESP32-S3)。CH1=ステアリング, CH2=スロットル, CH3=モード切替
    'rate_hz': 100,
    # 'snapshot': ファーム leracer_ctrl v2(レジスタ0x20のスナップショット、CRC-8付き)
    # 'legacy'  : 量産ファーム v1(チャネルの取り違えが起き、学習には使えない)
    'protocol': os.environ.get('JETFORMER_VEHICLE_PROTOCOL', 'snapshot'),
}

# ---- AI走行の出力 ----
# PCA9685 → コントローラ基板のセレクタ → サーボ/ESC。プロポのCH3がAI側(1500〜2500µs、LED紫)のときだけ
# PCA9685の出力が車に届く(それ以外はプロポの信号がそのまま届く)。JetRacerと同じ CH0=ステアリング、CH1=スロットル。
PCA9685 = {
    'bus': 7, 'address': 0x40,
    'freq_hz': 60,
    # PCA9685の内部発振器は個体差が数%ある。出力のµsがずれる(中立で進む・まっすぐ走らない)ときは
    # 管理画面の「時計補正」で合わせる。ここは補正0%のときの値
    'osc_hz': float(os.environ.get('JETFORMER_PCA9685_OSC', '25000000')),
    'steering_ch': 0, 'throttle_ch': 1,
}
DRIVE = {
    'min_us': 1000, 'max_us': 2000,   # 出力の上下限
    'max_sensor_age_ms': 150,         # カメラ・ToF・車両の値がこれより古ければ止める(中立を出す)
    'imu_max_age_ms': 100,
    'heartbeat_s': 3.0,               # 管理画面からの問い合わせがこれだけ途絶えたら走行を止める
}

# 取り付け位置(車体座標: x前方, y左, z上 [m])。カメラとToFは5cm以内に近接設置。
# 実測値で更新すること。回転はクォータニオン[w, x, y, z]。
CALIBRATION = {
    'CAM_FRONT': {'translation': [0.10, 0.0, 0.12], 'rotation': [1.0, 0.0, 0.0, 0.0]},
    'TOF_FRONT': {'translation': [0.10, -0.03, 0.12], 'rotation': [1.0, 0.0, 0.0, 0.0]},
    'IMU': {'translation': [0.0, 0.0, 0.08], 'rotation': [1.0, 0.0, 0.0, 0.0]},
}
