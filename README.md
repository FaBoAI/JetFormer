# JetFormer

RCカー(Jetson Orin Nano 8GB / Super mode)で、カメラ・ToF(VL53L9)・IMU・操作値を
同期収集し、nuScenes形式のデータセットとしてクラウド学習へ渡すための基盤です。
推論は30fpsで、TransFuserのようにTransformerでセンサーフュージョンします。

## 管理画面

```bash
~/jetformer/scripts/run_web.sh          # http://<jetsonのIP>:8080/
```

- **ライブ・収集**:カメラ|ToFのライブ映像、センサー状態、操作値・IMUのグラフ、収集の開始/停止
- **シーン**:20秒単位のシーンを一覧・絞り込み・選択、詳細(スクラブ再生・時系列・説明/タグ編集・削除)
- **データセット**:選んだシーンからnuScenes形式を作成 → tarでエクスポート(SHA256付き)→ ダウンロード

JupyterLab(8888番)とは別ポートです。カメラは同時に1つのプロセスしか使えないので、
JetRacerのノートブックでカメラを使っている間は収集できません。

## センサー

| チャネル | デバイス | 接続 | レート | 備考 |
|---|---|---|---|---|
| CAM_FRONT | IMX219 | CSI(V4L2直接、Argus不使用) | 30Hz | 224x224、撮像時刻はV4L2のハードウェア時刻 |
| TOF_FRONT | VL53L9(54x42ゾーン) | JetFormer基板 USB CDC(`L9FR`) | 30Hz | 単位は mm と仮定(要確認) |
| IMU | ICM-20948 | I2C bus7 0x68 | 100Hz | ±8g / ±1000dps。地磁気は現在無効 |
| VEHICLE | JetFormer基板(ESP32-S3) | I2C 0x28 / USB `JFVH` | 50〜100Hz | **現行ファームでは値が不正確**(docs/firmware_protocol.md) |

全センサーの時刻はCLOCK_MONOTONICに揃え、保存時にUNIXエポックのµs(nuScenesと同じ)に変換します。

## データ

```
data/raw/<log>/log.json                       収集セッション(車両・センサー設定・取付位置)
data/raw/<log>/<scene>/scene.json             20秒シーン(統計・品質情報)
                       CAM_FRONT/<ts>.jpg     カメラ
                       TOF_FRONT.npz / IMU.npz / VEHICLE.npz
data/datasets/<name>/v1.0-<name>/*.json       nuScenes互換の表
                     samples/  can_bus/  aligned/<scene>.npz  splits.json  dataset_card.json
data/exports/<name>.tar(.sha256)              クラウド学習用
```

`aligned/<scene>.npz` は学習用に、カメラ各フレームへToF・IMU履歴(100ms)・現在の操作値・
1〜10フレーム先の操作値を揃えた配列です(内容は dataset_card.json に記載)。

## モデル(TransFuser風センサーフュージョン)

`jetformer/model.py`:カメラ49+ToF 42+IMU+操作値のトークンをTransformerで融合し、
約0.33秒先までの操作値を予測。**Orin Nano(MAXN_SUPER)のTensorRT FP16で約1.9ms**。
学習(`tools/train.py`、クラウド)→ ONNX → Jetsonで `tools/build_engine.py`。詳細は docs/model_design.md。

## 設定

`jetformer/config.py`(環境変数で上書き可):`JETFORMER_PORT`、`JETFORMER_DATA`、
`JETFORMER_SCENE_SECONDS`、`JETFORMER_VEHICLE_PROTOCOL`(`legacy` / `frame` / `usb`)など。

## 未解決・要確認

- JetFormer基板ファームの修正(PWM・回転数を確実に読むため)→ docs/firmware_protocol.md
- ToF距離の単位、PWMチャネルの割り当て(ステアリング/スロットル)
- カメラ・ToF・IMUの取付位置(config.CALIBRATION)の実測
- ICM-20948の地磁気(AK09916)がバス上に現れない
