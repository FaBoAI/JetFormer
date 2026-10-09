# JetFormer v0 モデル設計(TransFuser風センサーフュージョン)

## 方針

TransFuserと同じく、センサーごとに特徴を抽出してトークン化し、Transformerの自己注意で
相互参照させてから、未来の行動を出力する。Orin Nano(30fps)で確実に動くよう、
TransFuserの多段(マルチスケール)融合ではなく、**最終段のトークン融合1回**に絞った。

| 入力 | 前処理 | エンコーダ | トークン |
|---|---|---|---|
| カメラ 224×224 RGB | ImageNet正規化 | ResNet-18(事前学習)→ 1×1conv | 7×7 = 49 |
| ToF 54×42 距離 | mm / 4000、0=無効をマスクとして追加入力 | 小CNN(stride 8) | 6×7 = 42 |
| IMU 直前10サンプル(100ms) | accel/9.8、gyro rad/s | MLP | 1 |
| 現在の操作値 | (µs−1500)/500、motor/100 | MLP | 1 |
| 出力用トークン | 学習可能 | — | 1 |

各トークンに位置埋め込み+モダリティ埋め込みを加え、Transformer Encoder(4層・256次元・8ヘッド)で融合。
出力トークン → MLP → **1〜10フレーム先(約0.33秒)の [ステアリング, スロットル]**(正規化値)。
TransFuserのウェイポイントの代わりに「行動の時系列」を予測する形(自己位置が無いため)。

## Orin Nano 8GB(MAXN_SUPER)での推論速度(実測 2026-10-10)

| 実行方式 | 平均 | p99 |
|---|---|---|
| PyTorch FP16 | 12.5〜13.2 ms | 14.0 ms |
| **TensorRT FP16**(Python版TensorRT 10.16) | **1.8〜1.9 ms** | 2.1 ms |

TensorRTとPyTorch(FP32)の出力差は最大0.0007。30fps(33ms)に対し推論は約2msなので、
画像側を大きくする(ResNet-34/EfficientNet)、ToF・画像の中間層での融合を足す、などの余地がある。

## 学習(クラウド)

```bash
tar xf indoor_v1.tar                 # JetFormerの管理画面からエクスポートしたもの
python tools/train.py --data datasets/indoor_v1 --epochs 30 --out runs/indoor_v1
# → runs/indoor_v1/best.pth と jetformer.onnx
```

- 入力・正規化は `jetformer/train_data.py` の `normalize_inputs()` を学習と推論で共用する
- 損失は未来10ステップの行動のL1。学習/検証はシーン単位で分割済み(splits.json)
- `aligned/<scene>.npz` の `vehicle_valid` / `future_valid` が偽のフレームは使わない
  (信号断・読み取り失敗の操作値を学習に混ぜない。100ms以上古い値も無効)

## Jetsonへのデプロイ

```bash
python tools/build_engine.py runs/indoor_v1/jetformer.onnx data/models/indoor_v1_fp16.engine
```

推論は `jetformer.trt_runtime.TRTRunner`。エンジンは必ずJetson上のPython版TensorRTで作る
(JetPack同梱のtrtexecはバージョンが異なり、Pythonから読み込めない)。

## 次の検討事項

- ファーム修正後のデータで、行動予測の精度(µs誤差)を評価する
- 走行ループ(カメラ・ToF・IMU・操作値 → 推論 → PCA9685)を30fpsで回す実装
- カメラ・ToFの外部パラメータを使った幾何的な融合(ToF点群を画像座標へ投影)
- TransFuser同様の多段融合や、補助タスク(ToFからの深度推定など)
