# JetFormer

RCカー(Jetson Orin Nano 8GB / Super mode)で、カメラ・ToF(VL53L9)・IMU・操作値を
同期収集し、nuScenes形式のデータセットとしてクラウド学習へ渡すための基盤です。
推論は30fpsで、TransFuserのようにTransformerでセンサーフュージョンします。

## 管理画面

```bash
sudo systemctl status jetformer-web     # 起動時に自動で立ち上がる(scripts/jetformer-web.service)
sudo systemctl restart jetformer-web    # コードを更新したとき
~/jetformer/scripts/run_web.sh          # サービスを使わずに手動で起動する場合 → http://<jetsonのIP>:8080/
```

- **ライブ・収集**:カメラ|ToFのライブ映像、センサー状態、操作値・IMUのグラフ、収集の開始/停止
- **シーン**:20秒単位のシーンを一覧・絞り込み・選択、詳細(スクラブ再生・時系列・説明/タグ編集・削除)
- **ジャンル**:走行ジャンル表の確認・追加・名前変更
- **データセット**:選んだシーンからnuScenes形式を作成 → tarでエクスポート(SHA256付き)→ ダウンロード / Hugging Face へアップロード(下記)
- **学習**:データセットと間引き率を選んで、このJetsonのGPUで学習(下記)
- **走行**:学習結果のモデルで車を走らせる。ステアリング・速度のゲインを走行中に調整、走行中の収集(評価用)も選べる(下記)

JupyterLab(8888番)とは別ポートです。カメラは同時に1つのプロセスしか使えないので、
JetRacerのノートブックでカメラを使っている間は収集できません。

## 評価

```bash
python tools/evaluate.py --dataset <データセット> --run <学習結果>   # → data/reports/<データセット>/
```

データの品質(フレームの欠け・時刻のずれ・操作の分布・ステアリングとヨーレートの向き・衝撃)と、検証シーンでの
予測の誤差(1〜10フレーム先、教師あり/AI走行と同じ自己入力)を、「今の操作を続ける」「学習データの平均」と比べる。

## データセットを PC にダウンロード(データセットタブ)

データセットごとの「⬇ PCにダウンロード」で、ブラウザにそのまま `<名前>.tar` を保存します(Hugging Face は使わない)。
Jetson がその場で tar にしながら送るので、Jetson にファイルは作らず、空き容量も使いません(`GET /api/datasets/<名前>/download`)。
tar の中身は「エクスポート」と同じ nuScenes 形式で、別の Jetson の `data/datasets/` に展開すればそのまま学習に使えます。
「エクスポート(tar作成)」は、`data/exports/` に tar と sha256 を残したいときに使います。

## Hugging Face へのアップロード(データセットタブ)

データセットごとの「Hugging Face へ」から、Hub のデータセットリポジトリに上げます。ブラウザの **Dataset Viewer** で
1フレームずつ(カメラ画像・ToFの色画像・操作値・ジャンル)見られる Parquet と、学習用の nuScenes 形式(tar)、
データセットカード(README)を上げます(`jetformer/hf_export.py`)。目安: 走行1分あたり約72MB、作成に約50秒。

- 最初に1回、Jetsonでログイン(Write権限のトークン): `/home/jetson/miniconda/envs/jetracer/bin/hf auth login`
- 既定は**非公開**。非公開のデータセットは Hugging Face の PRO / Enterprise でないと Dataset Viewer が表示されない
- 読み込み: `load_dataset("ユーザー名/jetformer-xxx", split="train")`

## 別の環境(シミュレーターなど)で作ったデータセットを取り込む(データセットタブ)

データセットタブの「📥 PC から取り込む」に、別の環境で作ったデータセットのファイルをドロップします(複数可、1つずつ順に送る)。
1ファイル = 1データセットの `.tar`・`.tar.gz`・`.tgz`・`.zip` で、中身はエクスポートと同じ形です(先頭のフォルダ = データセット名、
中に `dataset_card.json`・`splits.json`・`aligned/<シーン>.npz`・`samples/` の画像)。先頭のフォルダが無く `dataset_card.json` が
いちばん上にあるものは、カードの `name` で取り込みます。Mac の Finder や tar が入れる `__MACOSX/`・`.DS_Store`・`._*` は無視します。

- 既定では「🎮 シミュレーターのデータ」として取り込み、`simulator` のタグを付けます(外すと実機のデータとして)
- 取り込む前に、学習が読むものがそろっているかを確かめます: `splits.json` の各シーンの `aligned/*.npz` に
  `timestamps_us`・`cam_files`・`tof [N,42,54]`・`imu [N,10,6]`・`vehicle [N,4]`・`vehicle_valid`・`future_actions [N,10,2]`・
  `future_valid`・`mode`(0 = 未設定は学習に使われない)があり、画像があること。足りなければ取り込みません(半端なデータセットは残らない)
- 同じ名前のデータセットがあるときは取り込みません。取り込んだ記録は `import_source.json`
- シミュレーターの PC からコマンドでまとめて送る(Python 3 の標準ライブラリだけ。フォルダはその場で tar にして送る):
  `tools/upload_datasets.py`(管理画面の `/api/tools/upload_datasets.py` からも取れる)

```bash
python3 upload_datasets.py --host http://192.168.55.1:8080 sim_run_01/ sim_run_02/ more.tar
```

- API: `PUT /api/datasets/import?filename=<名前>.tar&simulator=1`(本文はファイルそのもの。`curl -T x.tar ...` でも送れる)・
  `POST /api/datasets/<名前>/tags`

## Hugging Face からのダウンロード(データセットタブ)

データセットタブの「🤗 Hugging Face からダウンロード」で、Hub に上げた JetFormer のデータセットをこの Jetson に取り込みます。
「探す」で `jetformer` タグの付いたデータセットを一覧し(ユーザー名・組織で絞り込める)、「確認」で学習用の tar と大きさを見て、
「ダウンロード」で `data/datasets/<名前>/` に展開します。取り込んだデータセットは一覧(取り込み元つき)と学習タブに出ます。

- 取り込むのは、アップロードで上げた学習用の nuScenes 形式 `nuscenes/<名前>.tar`。Dataset Viewer 用の Parquet だけのリポジトリは取り込めない
- 公開リポジトリはログインなしで取れる。非公開は、上と同じ `hf auth login` が必要
- Hub のハッシュ(SHA-256)で確かめてから、途中のフォルダに展開して置き換える(止まっても半端なデータセットは残らない)。
  同じ名前のデータセットがあるときは取り込まない(先に削除する)
- API: `GET /api/hf/hub_datasets?author=` ・ `GET /api/hf/hub_files?repo_id=` ・ `POST /api/hf/download {"repo_id": ...}`

## 学習(管理画面の「学習」タブ)

1. 学習するデータセットを選ぶ(ジャンルごとのフレーム数が出る)
2. 間引き率を選ぶ。30fpsの連続フレームはよく似ているので **1/3(10fps相当)がおすすめ**。1/1は全フレーム
3. エポック数(既定30)・バッチ・学習率を確認し「学習を開始」。所要時間の見積もりが下に出る
4. 進捗バー・残り時間・学習/検証誤差のグラフ・ジャンル別の検証誤差が更新される。「停止」でも、そこまでの最良モデルは保存される
5. 終了後、走行用のTensorRTエンジン(FP16)を自動で作る(約1分)。結果は `data/runs/<データセット>-s<間引き>-<日時>/` に残り、画面からダウンロードできる
   (`best.pth` / `jetformer.onnx` / `jetformer_fp16.engine` / `model_config.json` / `result.json` / `train.log`)

学習中は収集・走行できません(逆も同じ)。目安:Orin Nanoで学習 約200枚/秒、
100シーン(6万フレーム)を1/3間引き・30エポックで約40分、間引きなしで約2時間。
クラウドで学習する場合は、データセットをエクスポートして `tools/train.py` を使う(docs/model_design.md)。

**メモリに注意**:Orin Nano 8GBにはスワップが無く、学習は約3GB使います。メモリが尽きると本体ごと止まって
再起動します(2026-10-10に1回発生)。そのため学習は空きが3.5GB未満なら始めず、TensorRTエンジンは別プロセスで作ります。
ほかに大きなプログラム(JupyterのJetRacerのカーネルなど)が動いているときは止めてから学習してください。

### モデルの種類(方策)

学習タブの「モデルの種類」で、構造の違うモデルを選べます(`jetformer/policies/` の登録簿。一覧は `GET /api/policies`)。
既定は **JetFormer v0(今の標準)** で、選ばなければ今までとまったく同じ学習・同じファイルになります。

| 名前 | 中身 |
|---|---|
| `jetformer_v0` | **TransFuser軽量型**: 今までのモデル(下の「モデル」)。TransFuser を参考に、融合を最後に1回だけにした軽量版 |
| `act` | **ACT型**: v0 と同じトークン → Encoder → 10 個の行動クエリの Decoder |
| `tcp` | **TCP型**(TCP・TransFuser++ を参考): 通過点 + 目標速度 + 操作の枝。走行タブで目標速度 → 制御器・速度の上限を選べる(既定は操作の枝のまま。出し方 1〜3 は試験的で、起動のたび・モデルを替えたときに操作の枝に戻る。速度の上限は速度ゲインに関係なく効く) |
| `msfusion` | **TransFuser多段型**: 本家 TransFuser と同じ多段融合(ResNet-18 と ToF の CNN を4段で融合) |
| `dinov2` | **DINOv2型**: DINOv2 ViT-S/14(凍結)の画像特徴。重み `data/pretrained/dinov2_vits14_pretrain.pth` が要る(自動では取らない) |

どの種類も出力 `actions`(10 フレーム先までの [ステア, スロットル])を持つので、走行の安全の仕組み(ゲイン・トリム・先読み・ToF ガード・
リカバリー・ラップ・CH3 の交代・センサーの途切れで中立)はそのまま効きます。まだ作っていない種類は「使えません」と出て選べません。
「詳細」の「衝突とみなす加速度」(2.5G = モデル B と同じ)・「乱数の種」は、モデルどうしを比べるときに条件をそろえるためのものです。
CLI では `tools/train.py --policy act [--policy-cfg '{"dec_layers": 3}']`。`model_config.json` に `policy`・`policy_cfg`・
`inputs`(書き出した ONNX から読み直した名前)などが入り、`policy` の無い古い学習結果は `jetformer_v0` として読みます。
エンジンの精度は `model_config.json` の `engine.precision`(既定 fp16)で、`tools/build_engine.py` は作った後に入出力を確かめます。
学習結果のフォルダーに `engine_check.json`(実フレームで TensorRT と PyTorch FP32 の差・速さを確かめた結果)があれば、走行タブに出します。今のエンジンが精度の確認(差 最大 2.5 µs 以内)に落ちていれば、走行タブで選べなくし、理由を出します(走行も始めません。エンジンを作り直せば、古い確認は当てません)。速さだけ落ちたときは警告だけです。

走行タブは、評価で決めた**推奨のモデル**(`data/drive_recommended.json`)を一覧の先頭に「★推奨」と出し、ページを開いたときに選びます(前回使ったモデルより先。無ければ今まで通り前回のモデル → 一番新しいモデル)。方策の比較の評価の推奨は `python tools/policy_zoo.py recommend` で書きます(手で決めるときは `--run <学習結果> --reason <理由>`)。

**DINOv2 の重みの取得**(`dinov2` を使うときだけ・1回だけ。ネットから自動では取りません)。電源断で半端なファイルを残さないよう、
`.part` に書いて sha256 を確かめてから名前を変えます:

```bash
cd ~/jetformer && mkdir -p data/pretrained
curl -fL -o data/pretrained/dinov2_vits14_pretrain.pth.part \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
sha256sum data/pretrained/dinov2_vits14_pretrain.pth.part
# → b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 と同じなら
mv data/pretrained/dinov2_vits14_pretrain.pth.part data/pretrained/dinov2_vits14_pretrain.pth
```

(公式の通常版 ViT-S/14、約 88 MB。`data/` は git に入りません。DINOv2 のコードと重みは Apache-2.0)

## 学習済みモデルを Jetson の外で使う(エミュレーター用の書き出し)

学習タブの各モデルの「⬇ エミュレーター用(zip)」で、Mac などのエミュレーターで動かすための一式を zip で取れます
(「＋重み(.pth)」は PyTorch の重みも入れる)。TensorRT のエンジンは Jetson 専用なので入れません。

| zip の中身 | 内容 |
|---|---|
| `jetformer.onnx` | 推論用(バッチ 1 固定)。onnxruntime(CPU / CoreML)で動く |
| `model_config.json` / `result.json` | 学習の設定・正規化の定数 / 学習の結果 |
| `io_spec.json` | 入力の作り方(カメラ・ToF・IMU・操作値・ジャンル → 入力)と出力の読み方(µs・左右と前後の向き) |
| `README.md` | 使い方 |
| `run_onnx_example.py` | onnxruntime で動かす例(`tools/emulator_run_onnx.py`)。`JetFormerOnnx(フォルダ).predict(...)` をエミュレーターから呼べる |

- 入力の作り方は Jetson の学習・走行(`jetformer/train_data.py` の `normalize_inputs`)と同じ。記録した実フレームで、入力は 2e-7 以内、
  出力(onnxruntime と PyTorch)は 0.001µs 未満で一致することを確かめた
- API: `GET /api/models`(一覧)・`GET /api/models/<名前>/io_spec`・`GET /api/models/<名前>/bundle.zip[?pth=true]`

## AI走行(管理画面の「走行」タブ)

```
カメラ・ToF・IMU・車両 → 推論(TensorRT, 30fps) → ゲイン → PCA9685(I2C 0x40) ─┐
プロポ受信機 ───────────────────────────────────────────────┤ コントローラ基板のセレクタ → サーボ・ESC
                                    プロポのCH3(1500〜2500µs=AI側、LED紫)で切り替え ┘
```

モデルはプロポと同じ「パルス幅µs」で操作値を出すので、PCA9685でも同じµsを作ります
(JetRacerのような左右・前後の向きの設定は不要)。CH0=ステアリング、CH1=スロットル、60Hz(`config.PCA9685`)。

1. 学習タブで「学習後にTensorRTエンジンを作る」にして学習する(エンジンのある学習結果だけが選べる)
2. モデルと走行ジャンルを選ぶ(そのモデルが学習したジャンルだけ選べる)
3. はじめは「モーターに出力しない」で開始し、予測が妥当か(ゲージ・グラフ)を確認する。この間PCA9685は中立を出し続ける
4. 速度ゲインを小さめ(既定0.5。最初は0.3〜0.5)にして「走行開始」→ プロポのCH3をAI側にすると車がAIで動く
5. 止めるとき:**プロポのCH3を戻す(常に最優先)**/停止ボタン/画面下の赤いボタン(どのタブでも出る)/スペースキー・Escキー

**走行を始めた画面は開いたままにする**:その画面からの問い合わせが5秒途絶える(画面を閉じた・スマホがスリープした・
通信断)と走行を止めます。画面のファイルが古いまま(ブラウザのキャッシュ)だと走行を始められないので、ページを再読み込みしてください。ほかの端末で同じ管理画面を開いていても、
走行を始めた画面の代わりにはなりません(表示と停止はできます)。

- **ゲイン**:ステアリング・速度とも、学習データで車が止まっているときの操作値(中心・中立)を基準に、そこからの量を何倍にするか。
  走行中に動かしてもすぐ反映。0で直進・停止、1で学習どおり
- **詳細設定**:ステアリングのトリム、先読み(何フレーム先の予測を使うか。遅れて曲がるときは大きく)、PCA9685の時計補正
  (PCA9685は個体差でパルス幅が数%ずれる。停止中にCH3をAI側にして、モーターが回り出す・ハンドルが動くなら合わせる)
- **走行中に収集する(評価用)**:20秒シーンとして保存し、各フレームのAIの予測・出力・ゲインを `AI_CMD.npz` に残す。
  シーンの詳細で映像と一緒に見返せる。CH3を人側にして介入した区間は人の操作として、そのまま学習データにも使える
- 安全のための動作(中立を出す):センサーの値が途切れた・古い(150ms以上)/推論が異常/
  **CH3がAI側のままプロポの信号が届かない**(送信機の電源断・圏外。基板はSELECTを最後の値のまま保つので、
  人が取り戻せない)/PCA9685への書き込み失敗(続けて5回失敗したら走行を止める)
- 走行を止める:停止操作/走行を始めた画面が5秒途絶えた/サーバーの終了(`systemctl stop`・`restart` でもすぐ止めて中立を出す)。
  サーバー起動時は、前のプロセスが残したPCA9685の出力をパルスなしにする
- **受信機のフェイルセーフ**:送信機の電源断・圏外のとき、CH3が人側(1500µs未満)になるか、パルスが止まる設定にしておく。
  CH3の最後の値を保つ設定だと、基板もJetsonも信号断に気づけない
- サーバーが強制終了(停電・SIGKILL)した場合はPCA9685が最後の出力を出し続けるので、CH3を人側に戻す
- 実測(Orin Nano Super):30fps、撮像から出力まで平均約28ms(最大約35ms)、推論(TensorRT FP16、転送込み)約5.7ms

テスト(実機不要。センサー・PCA9685・TensorRTは偽物で、走行ループの安全動作・収集・学習の見積もりなどを確認):

```bash
cd ~/jetformer && /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests
```

## 走行ジャンル(モデルの走り方の切り替え)

駐車(1)〜(3)・高速走行・低速走行などの**走行ジャンルID**をデータに付け、モデルは入力されたIDで
走り方を切り替えます(条件付き模倣学習)。

- ジャンル表は `data/modes.json`(画面の「ジャンル」タブで追加・名前変更)。**IDはモデルの出力ヘッド番号なので変えない**。0=未設定は学習に使わない
- 収集時にジャンルと「走行の説明」を選ぶ。**収集中に切り替えるとその瞬間から**新しいジャンルとして記録(シーン内の区間として保存)
- シーン一覧ではジャンルで絞り込み・まとめて設定ができる。データセットには、フレームごとのジャンル(`aligned/*.npz` の `mode`)とジャンル表のスナップショットが入る
- 旧ファームv1で収集したシーン(「PWM:v1(学習不可)」)は、選んでもデータセットに入らない(`dataset_card.json` の `skipped_legacy_scenes`)

## センサー

| チャネル | デバイス | 接続 | レート | 備考 |
|---|---|---|---|---|
| CAM_FRONT | IMX219 | CSI(V4L2直接、Argus不使用) | 30Hz | 224x224、撮像時刻はV4L2のハードウェア時刻 |
| TOF_FRONT | VL53L9(54x42ゾーン) | ESP32-P4基板 USB CDC `/dev/ttyACM0`(`L9FR`/`L9FE`) | 30Hz | 生の値を保存。下位13bit=距離mm、上位3bit=信頼度 |
| IMU | ICM-20948 | I2C bus7 0x68 | 100Hz | ±8g / ±1000dps。地磁気は現在無効 |
| VEHICLE | コントローラ基板(XIAO ESP32-S3、ファームv2) | I2C bus7 0x28 スナップショット | 100Hz | CH1=ステアリング、CH2=スロットル、CH3=モード切替、eRPM、受信中フラグ、SELECT |
| AUDIO | I2S MEMSマイク(既定は無効) | 40ピンのI2S2(pin12/35/38) | 50Hz(20ms) | 16kHz・モノラル・24bit精度。下記参照 |

全センサーの時刻はCLOCK_MONOTONICに揃え、保存時にUNIXエポックのµs(nuScenesと同じ)に変換します。
通信の詳細は docs/firmware_protocol.md。学習の教師には、**CH1・CH2を受信中で、かつ人が操縦している
(SELECT=人側)**フレームだけを使います(AI走行中・プロポ途絶のフレームは除外)。

## スピーカー(音声で知らせる)とマイク:40ピンのI2S

JetFormer基板のI2Sスピーカーアンプ(とマイク)は、40ピンヘッダのI2S2につながっています。
pin12=BCLK、pin35=LRCLK(Jetsonがクロックを出す)、pin40=DOUT(スピーカーへ)、pin38=DIN(マイクから)。

**最初に1回、40ピンのI2Sを有効にする**(標準ではGPIOのままで、音が出ません):

```bash
sudo ~/jetformer/scripts/i2s_mic/enable.sh     # 起動設定に「I2S2有効」のエントリを追加(追加済みならデバイスツリーを更新)
sudo reboot
```

今の起動エントリはそのまま残り、`/boot/extlinux/extlinux.conf` はバックアップされます。元に戻すときは
`sudo ~/jetformer/scripts/i2s_mic/disable.sh` → 再起動。(デバイスツリーのオーバーレイは
`scripts/i2s_mic/jetformer-i2s2-mic.dts`。NVIDIA純正のI2Sオーディオ用オーバーレイと同じピン設定です。)

### 音声案内(ライブ・収集タブの「サウンド」)

車のスピーカーから、チャイムと音声(Open JTalk)で知らせます。出来事ごとにON/OFF、音量、テスト、任意の文を話させることもできます。

| 出来事 | 例 |
|---|---|
| 起動 | 「JetFormer を起動しました」 |
| 収集の開始・終了 | 「データ収集を開始します。高速走行」/「データ収集を終了しました。3 シーンを保存しました」 |
| ジャンルの切り替え | 「ジャンル、低速走行」 |
| シーン保存の合図(既定OFF) | 20秒ごとに短い「ピッ」 |
| AI走行の開始・終了 | 「AI走行を開始します」/「AI走行を終了しました」/「画面との通信が途絶えたため、AI走行を止めました」 |
| 操縦の切り替え(CH3) | 「AIが操縦します」/「人の操縦に戻りました」(走行中、CH3を切り替えた瞬間) |
| 警告 | 「プロポの信号が届きません。停止します」(AI側のまま送信機の信号が途絶えたとき) |
| 学習の終了 | 「学習が完了しました」 |

音声は `data/sounds` にキャッシュし(初回のみ合成)、再生は ADMAIF2 → I2S2(48kHz・32bit・2ch)。
設定は `data/sound_settings.json`。走行ループなどは再生を待たない(キューに入れて別スレッドで鳴らす)。

### マイク(既定は無効)

2026-10-10の確認では pin38 にマイクの信号が来ていない(全て0)ため、既定では使いません。マイクをつないだら
`JETFORMER_MIC=1` で有効にすると、ライブ画面の「マイク」(音量メーター・🔊 聞く)とシーンへの録音(`AUDIO.npz`、
16kHz)が使えます。チャネルを固定したいときは `JETFORMER_MIC_CHANNEL=left` / `right`。

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

`jetformer/model.py`:カメラ49+ToF 42+IMU+操作値+走行ジャンルのトークンをTransformerで融合し、
ジャンル別の出力ヘッドで約0.33秒先までの操作値を予測。**Orin Nano(MAXN_SUPER)のTensorRT FP16で約1.9ms**。
学習(管理画面の「学習」タブ、またはクラウドで `tools/train.py`)→ ONNX → TensorRTエンジン。詳細は docs/model_design.md。

## 設定

`jetformer/config.py`(環境変数で上書き可):`JETFORMER_PORT`、`JETFORMER_DATA`、`JETFORMER_SCENE_SECONDS`、
`JETFORMER_TOF_PORT`、`JETFORMER_VEHICLE_PROTOCOL`(`snapshot`=ファームv2〔既定〕/ `legacy`=v1)など。

## 検証中のトライアル

実車走行・学習・解析の検証の記録(レポート・解析スクリプト・図)は [`experiments/`](experiments/README.md) にある。
10/10 の AI走行の解析、衝突のないデータでの学習と精度比較、「ぶつからずに速く走る車への道筋」、オフライン強化学習のチャレンジ(10/11)など。

## 未解決・要確認

- 実車でのAI走行の確認(PCA9685の時計補正、ゲインの目安、先読みの最適値)
- カメラ・ToF・IMUの取付位置(config.CALIBRATION)の実測
- ICM-20948の地磁気(AK09916)がバス上に現れない
