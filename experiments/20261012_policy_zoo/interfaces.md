# 方策(ポリシー)を差し替えるための差し込み口と、最小の「方策の登録簿」

作成 2026-10-12 ・ 対象 `/home/jetson/jetformer`(main、2b9774e)・ 設計のメモ(リポジトリのファイルは変更していない。GPU は使っていない)

ユーザーの目的: 「TransFuser 系の policy として、ほかにも可能性のあるモデルを policy として追加し、学習して走行できるようにしたい」。
JetFormer v0 を登録簿の1つにして、ほかの構造を **CLI と管理画面の学習タブ** で選んで学習し、ONNX から TensorRT にして、
**走行タブ** で同じ安全の仕組み(ゲイン・補助/リカバリー・ToF ガード(記録だけ)・ラップの計測)のまま走らせる。

---

## 0. 要約

1. **新しい構造が通る道は1本**: 学習データ(`jetformer/train_data.py`)→ 学習(`tools/train.py`)→ `best.pth`・`model_config.json`・
   `jetformer.onnx` → エンジン(`tools/build_engine.py`。管理画面からは `jetformer/training.py` が別のプロセスで呼ぶ)→
   走行(`jetformer/drive.py`)→ 評価(`tools/compare_models.py`・`tools/rl_evaluate.py`・`tools/benchmark_model.py` ほか)。
   クラスの名前 `JetFormer(num_modes=...)` を直接呼んでいる所は、リポジトリの中に **9か所**(実験のスクリプトを除く。§2.10)。
2. **走行側の安全の仕組みは、モデルの出力 `actions` [1,10,2](正規化した µs)だけに依存している**。ゲイン(`apply_gains`)・
   先読み・前の予測の戻し(`predicted_now`)・補助とリカバリー(`assist.Assist`)・ToF ガード・ラップは、どれも構造を見ない。
   → **出力の約束(名前 `actions`、形 [1,10,2]、(µs−1500)/500)を守れば、方策を替えてもそのまま効く**。
3. 一番強いハードコードの前提(§3): 入力が5つ(名前・順・形が固定)、出力が [B,10,2]、ジャンル別ヘッド(`num_modes`)、
   画像 224×224、1フレームだけ、バッチ1・固定の形・FP32 の入出力・FP16 のエンジン、ファイル名が固定。
4. **確かめた事実**(§4): ONNX 書き出し(torch 2.10、`dynamo=False`)は **使っていない入力をグラフから消す**。
   `TRTRunner.run_numpy` はエンジンの入力の名前だけを読む。→ 走行は「知っている入力をすべて(上位集合で)渡す」、
   `model_config.json` の `inputs` は **書き出した ONNX から読み直す** のが安全。
5. **提案**(§5): torch を読み込まない登録簿 `jetformer/policies/__init__.py`(`PolicySpec`・`build_policy(name, num_modes, cfg)`)、
   torch 側の共通処理(読み込み・書き出し・損失)、numpy だけの入力の組み立て(`feed.py`。フレームの履歴など)。
   `model_config.json` に `policy`・`policy_cfg`・`input_specs` を足し、**無ければ `jetformer_v0`**。
   v0 は `JetFormer` のインスタンスをそのまま返す(ラッパーで包まない)ので、**今の `best.pth`・`last.pth`・エンジンはそのまま使える**。

---

## 1. 今のデータの流れ(どこで形が決まっているか)

```
data/datasets/<ds>/aligned/<scene>.npz  (30fps のフレームごと: cam_files, tof[42,54], imu[10,6], vehicle[4], future_actions[10,2], mode)
      │  JetFormerDataset.__getitem__ → normalize_inputs(学習と走行で共通)
      ▼
(image[3,224,224], tof[2,42,54], imu[10,6], state[3], mode[num_modes], target[10,2] [, weight])
      │  tools/train.py run_epoch: model(image, tof, imu, state, mode) → L1
      ▼
data/runs/<run>/ best.pth(state_dict)・last.pth(再開用)・model_config.json・result.json・run.json(管理画面から)
      │  export_onnx(固定の形・バッチ1・opset 17)
      ▼
jetformer.onnx ── tools/build_engine.py(FP16、作業メモリ 1GB)──▶ jetformer_fp16.engine
      │
      ▼  drive.model_info(エンジンと model_config.json があれば一覧に出る)
Driver._run → TRTRunner → _loop: assemble_inputs → JPEG を通す → normalize_inputs → run_numpy(5つの名前)
      → res['actions'][0] [10,2] → 先読み k → apply_gains → ToF ガード → 補助/リカバリー → PCA9685
```

---

## 2. 差し込み口の一覧(file:line)

行番号は main(2b9774e)のもの。「変える」は登録簿を入れるときに手を入れる所、「確認」は前提を確かめるだけの所。

### 2.1 モデルの構築 — `jetformer/model.py`

| 行 | 中身 | 新しい方策との関係 |
|---|---|---|
| 24-27 | `IMAGE_SIZE=224`・`TOF_SHAPE=(42,54)`・`IMU_STEPS=10`・`FUTURE_STEPS=10` | 入力・出力の形の定数。`train_data.py`・`train.py`・`benchmark_model.py` が import する。**登録簿の基本の入力の形の元**としてここに残す |
| 30-35 | `PWM_CENTER/PWM_SCALE`・`TOF_MAX_MM`・`TOF_CHANNELS=2`・`ERPM_SCALE`・ImageNet の平均と標準偏差 | 正規化の定数。方策ごとに変えない(変えると `normalize_inputs` と走行を分けることになる) |
| 38-48 | `ToFEncoder`(距離・信頼度 + 有効マスク → stride 8 → [B,dim,6,7]) | 新しい方策が部品として再利用できる |
| 51-52 | `JetFormer.__init__(num_modes=8, dim=256, layers=4, heads=8, pretrained=False)` | v0 の構築。**`dim/layers/heads` の既定値は v0 の `policy_cfg` の既定値として凍結する**(古い学習結果は `num_modes` だけで作り直しているため) |
| 54-56 | `torchvision.models.resnet18(weights=IMAGENET1K_V1)` | `pretrained` の意味(ImageNet)。キャッシュにあるのは resnet18 の重みだけ(§4) |
| 62, 72 | `mode_embed = Linear(num_modes, dim)`・`heads = Linear(dim, num_modes*FUTURE_STEPS*2)` | ジャンル別ヘッド。`tools/train.py` の `expand_mode_state` がこのキー名に依存(§2.3) |
| 64-67 | `n_tokens = 49 + 42 + 4`・`modality_index` | トークン数が 224×224(→7×7)と ToF 42×54(→6×7)に結びついている。v0 の中だけの前提 |
| 78 | `forward(self, image, tof, imu, state, mode)` | **5つの入力の約束**(名前と順) |
| 95 | `self.norm(...)[:, -1]` | 融合後の出力トークン。`tools/offline_rl.py:537` がここにフックを掛ける(v0 専用) |
| 96-97 | `[B,num_modes,10,2]` を one-hot で選ぶ → `[B,10,2]` | **出力の約束** |
| 100-103 | `mode_onehot(ids, num_modes)` | 共通の道具(登録簿の外に残す) |

### 2.2 学習データ — `jetformer/train_data.py`

| 行 | 中身 | 新しい方策との関係 |
|---|---|---|
| 22 | `model.py` から定数を import | 確認 |
| 44-63 | `normalize_inputs(image_bgr, tof_raw, imu, vehicle, use_command)` → `(img[3,224,224], tof[2,42,54], imu[10,6], state[3])` | **学習と走行で共有する正規化**。50-54 の画像部分を `normalize_image()` に切り出すと、履歴のフレームにも同じ処理を使える |
| 51-52 | 224 以外なら `cv2.resize` | 画像の大きさを方策ごとに変えるなら、ここと走行(`drive.py:416`)とカメラの設定(`config.py:21`)が関わる。**v1 では 224 固定** |
| 61-62 | `use_command=False` で `state[:2]=0` | 方策に依存しない。`model_config.json` の `command_input` で走行と揃う |
| 94-97 | `JetFormerDataset.__init__(..., drop, keep_impact, weights, mode_remap, remap_dropout, extra_modes, return_weight)` | RL のオプションはデータの使い方だけで、方策に依存しない。**`extra_inputs=None` を足す**(§5.6) |
| 174 | `cv2.imread(cam_files[i])` — **1フレームだけ** | フレームの履歴を入れるならここ。aligned には 30fps の全フレームの `cam_files`・`timestamps_us` があるので、前のフレームは読める(§4) |
| 175-177 | 色の揺らぎ(サンプルごとに RGB のゲイン) | 履歴のフレームにも **同じゲイン** を掛けること |
| 179 | `target = (future_actions[i] − 1500)/500` → [10,2] | 出力の約束の元。aligned が [N,10,2] で作られているので、**10ステップは v1 で固定** |
| 182-187 | 戻り値 `(image, tof, imu, state, mode, target[, weight])` | 6個(+重み)のタプル。`rl_policies.py:506, 593`・`rl_evaluate.py` が位置で取り出しているので **崩さない**。追加の入力は最後に dict で足す(§5.6) |
| 202-210 | `action_reference()`(ゲインの中心・中立) | 方策に依存しない |

### 2.3 学習 — `tools/train.py`(RL のオプションを含む)

| 行 | 中身 | 新しい方策との関係 |
|---|---|---|
| 36-38 | `from jetformer.model import ... JetFormer` | 変える: 登録簿から作る |
| 48-54 | `RL_ARGS` と `rl_args_used`(既定値のままなら記録しない) | **同じ方式で `policy`・`policy_cfg` を足す**と、前からの `last.pth` で `--resume` できる |
| 62-103 | `run_epoch` | 変える(§5.6) |
| 76 | `image, tof, imu, state, mode, target = batch[:6]` | 位置で取り出し |
| 77 | `weight = batch[6] if len(batch) > 6` | **追加の入力を足すと、ここが重みと取り違える**。dict を最後に置き、`isinstance` で見分ける |
| 78-79 | `torch.autocast(float16)` | 方策が FP16 で不安定なら、方策の側で無効にできるようにする(ヒント) |
| 80 | `pred = model(image, tof, imu, state, mode)` | **5つの位置の引数で呼ぶ** |
| 81-82 | 損失 = 10ステップ×2 の L1(重みつき) | 方策ごとの損失(補助タスク・速度の分類など)を入れるならここ。**検証の µs はどの方策でも同じ L1 のままにする**(比べられるように) |
| 95 | ジャンル別の誤差 `mode.argmax` | 方策に依存しない |
| 106-137 | `expand_mode_state`(`mode_embed.weight`・`heads.weight/bias` のキー名、`rows = FUTURE_STEPS*2`) | **v0 の構造に依存**。`--init-mode-copy`(RECAP 型のジャンル 6・7)は、同じキー名を持つ方策だけで使える(§5.3 `genre_keys`) |
| 163-170 | `export_onnx`: ダミー入力の形を直書き(`(1,3,224,224)`…`mode_onehot([1], num_modes)`)、`input_names=['image','tof','imu','state','mode']`、`output_names=['actions']`、`opset_version=17`、`dynamo=False`、`dynamic_axes` なし(バッチ1固定)、165 `mha fastpath` を切る | 変える: 登録簿の入力の表から作る(§5.7) |
| 173-201 | 引数 | `--policy`・`--policy-cfg` を足す |
| 235-239 | データセットを作る | `extra_inputs=` を渡す |
| 270-271 | `model = JetFormer(num_modes=num_modes, pretrained=not args.no_pretrained)` | 変える: `build_policy(...)` |
| 272-273 | AdamW(wd 1e-4)+ cosine | 方策が「事前学習の部分だけ学習率を下げる」などを望むなら、任意のフック(`param_groups`)で |
| 278-280 | `same_args`(`--resume` で比べる引数) | `policy` は既定値以外のときだけ足す |
| 284-292 | `--resume`: `load_state_dict(ck['model'])` | 方策が同じなら変更なし |
| 296-301 | `--init`: `torch.load` → `expand_mode_state` → `load_state_dict` | **`--init` の元の学習結果の `model_config.json` を読み、方策と `policy_cfg` が同じか確かめる**(違えば止める) |
| 335-343 | `result.json` | `policy`・`policy_cfg` を足す |
| 344-352 | `rl_options`(使ったときだけ) | 変更なし |
| 353-355 | 最良の重みを読み直して `export_onnx(model, 'jetformer.onnx', num_modes)` | 変える |
| 356-365 | `model_config.json`(`inputs` は 358 で直書き、`future_steps`・`tof_channels`・正規化の定数・`trained_modes`・`steer_center_us`・`throttle_neutral_us`・`command_input`・`mask_impacts`) | `policy`・`policy_cfg`・`contract`・`input_specs`・`outputs`・`engine` を足す。`inputs` は ONNX から読み直す(§5.5) |

### 2.4 チェックポイント・設定ファイル

| ファイル | 書く所 | 読む所 | 中身 |
|---|---|---|---|
| `best.pth` | `train.py:314` | `train.py:354`、`compare_models.py:37`、`evaluate.py:118`、`rl_evaluate.py:520`(`load_state_dict` 136-142 は `last.pth` の形も読む)、`rl_policies.py:502, 590`、`offline_rl.py:535`、`benchmark_model.py:52`(`--weights`) | state_dict だけ(方策の名前は入っていない) |
| `last.pth` | `train.py:321-325` | `train.py:284-292`、`rl_evaluate.py:136-142` | `{'model', 'optimizer', 'scheduler', 'epoch', 'best', ..., 'args': same_args}` |
| `model_config.json` | `train.py:356-365` | `drive.py:92, 97-108`、`compare_models.py:35, 39, 47`、`evaluate.py:116`、`rl_evaluate.py:511, 615`、`rl_policies.py:500, 588, 663`、`offline_rl.py:530`、テスト(`test_drive.py:344`、`test_rl_review.py:100` は `{"num_modes": 6, "trained_modes": [4, 5]}` だけの最小の形) | **最小の形(`num_modes` と `trained_modes` だけ)でも読めること** が後方互換の条件 |
| `run.json` | `training.py:66-71`、`rl_policies.py:430-439` | `training.py:213, 219-221`、`drive.py:96` | 管理画面の学習の一覧の引数 |
| `result.json` | `train.py:370` | `training.py:131, 216`、`drive.py:95`、`compare_models.py:38`、`rl_policies.py:381` | |

### 2.5 ONNX の書き出し

- `tools/train.py:163-170`(本番)と `tools/benchmark_model.py:69-72`(計測用の複製)の2か所。**共通の関数にまとめる**。
- 約束: 入力の名前 `image`・`tof`・`imu`・`state`・`mode`、出力の名前 `actions`、opset 17、TorchScript の書き出し(`dynamo=False`)、
  固定の形でバッチ1、`torch.backends.mha.set_fastpath_enabled(False)`(ONNX に無い融合カーネルを避ける)。
- モデル B の `jetformer.onnx` を読んで確かめた形(§4): `image [1,3,224,224]`、`tof [1,2,42,54]`、`imu [1,10,6]`、`state [1,3]`、
  `mode [1,6]`(すべて FLOAT)→ `actions [1,10,2]`、427 ノード。

### 2.6 エンジンの作成 — `tools/build_engine.py`・`jetformer/trt_runtime.py`・`jetformer/training.py`

| 行 | 中身 | 新しい方策との関係 |
|---|---|---|
| `build_engine.py:15-20` | 引数は ONNX とエンジンの2つだけ。いつも FP16 | `model_config.json`(ONNX と同じフォルダー)の `engine` 欄(`fp16`・`workspace_gb`)を読む。作った後に入出力を確かめる(§5.7) |
| `trt_runtime.py:13-30` | `build_engine(onnx, engine, fp16=True, workspace_gb=1.0)` | 変更なし(引数で渡す) |
| `trt_runtime.py:45-55` | エンジンの入出力の形で `torch.empty(shape, float32)` を確保 | **形が固定(-1 があると壊れる)・すべて float32** が前提。`self.shapes` を公開すると、走行の空回しの入力をエンジンから作れる |
| `trt_runtime.py:57-65`(`__call__`)・`67-80`(`run_numpy`) | **エンジンの入力の名前だけを** `arrays[name]` で読む。出力はすべてホストへ写す | 余分な入力を渡しても動く(上位集合で良い)。足りないと `KeyError`(走行ループの中で止まる)→ 走る前に確かめる |
| `training.py:21-25, 34` | `RUNS_DIR`・`TRAIN_SCRIPT`・`ENGINE_SCRIPT`・`MIN_FREE_MB=3500`・`ARTIFACTS`(ファイル名) | `MIN_FREE_MB` は方策の目安(`hints.min_free_mb`)との大きい方 |
| `training.py:48-49` | `start(dataset, stride, epochs, batch, lr, pretrained, build_engine, command_input, mask_impacts)` | `policy='jetformer_v0', policy_cfg=None` を足す |
| `training.py:63` | 学習結果の名前 `'%s-s%d-%s' % (dataset, stride, 日時)` | v0 以外は方策の短い名前を入れる(`<ds>-<slug>-s3-<日時>`)。v0 は今のまま |
| `training.py:66-71` | `run.json` の引数 | `policy`・`policy_cfg` を足す |
| `training.py:72-80` | `train.py` のコマンド | 既定以外のときだけ `--policy`・`--policy-cfg` を足す(v0 は今と同じコマンド) |
| `training.py:139-154` | 学習の後、別のプロセスで `build_engine.py jetformer.onnx jetformer_fp16.engine`(20分で打ち切り) | 変更なし(ファイル名は方策に関係なく同じにする) |
| `training.py:208-228` | `list_runs`(219-221 で引数のキーを列挙、`command_input` だけ既定値 True) | `policy` を足す(無ければ `jetformer_v0`) |

ファイル名 `jetformer.onnx`・`jetformer_fp16.engine` は、`drive.py:37, 90`、`training.py:34, 139, 144`、`rl_policies.py:69-70, 459-468`、
`rl_review.py:151-154`(走らせないエンジンの名前を変える)、テスト(`test_rl_review.py:81-104`)が使っている。
**方策が変わってもファイル名は変えない**(変えると、走らせない仕組みと一覧が全部ずれる)。

### 2.7 走行 — `jetformer/drive.py`

| 行 | 中身 | 新しい方策との関係 |
|---|---|---|
| 36-39 | `ENGINE_FILE`、`IMU_STEPS = 10`(「torch を読み込まずに使うため複製」)、`PWM_CENTER/SCALE` | **drive.py は torch を読み込まずに一覧を作る**。登録簿のメタデータも torch なしで読めること |
| 41-52 | `PARAM_LIMITS`(`lookahead` 1〜10) | 出力が10ステップの前提 |
| 87-109 | `model_info`: エンジンと `model_config.json` が無ければ None、`num_modes`・`trained_modes`・`command_input` など | `policy`・`policy_title`・`input_specs`・`drivable`・`why_not` を足す(§5.8) |
| 112-117 | `list_models` | 変更なし |
| 129-150 | `assemble_inputs`(撮像時刻以前の最新の ToF、直近10個の IMU、最新の車両の値。データセットと同じ規則) | 方策に依存しない。IMU を長くするならここと `imu_buf`(423、maxlen 60 = 0.6秒)とデータセットの作り直し |
| 153-164 | `vehicle_state`(今の操作値: 人 / 前の予測 / 中立) | 方策に依存しない |
| 167-171 | `predicted_now`(`len(prev_pred)` を使う) | 出力 [10,2] |
| 174-180 | `apply_gains` | 方策に依存しない |
| 409-414 | `TRTRunner(RUNS_DIR/run/ENGINE_FILE)` | ここで **エンジンの入力が走行で作れるものか** を確かめる(§5.8) |
| 415-420 | 空回しの入力を **直書き**: `image (1,3,224,224)`・`tof (1,2,42,54)`・`imu (1,10,6)`・`state (1,3)`・`mode (1,num_modes)` | エンジンの形(`runner.shapes`)から作る |
| 422-424 | `tof_buf` 10・`imu_buf` 60・`veh_buf` 50 | 履歴の入力が増えるときは、別の履歴(正規化した画像)を持つ |
| 444-445 | 収集に残す `drive_info`(run・dataset・params・中心・中立) | `policy` を足す(走行のログからどの構造か分かるように) |
| 451-452 | `self._loop(st, runner, normalize_inputs, pca, out, ref, num_modes, osc_pct, cam_slot, cam_evt, tof_buf, imu_buf, veh_buf, dry_run)` | テスト(`test_drive.py:85-86`)が位置で呼ぶので、**足す引数は最後にキーワードで**(`feeder=None`) |
| 541 | `use_command = model.command_input` | 方策に依存しない |
| 560 | 中立で止めた行の `pred = np.full((10, 2), nan)` | 出力10ステップの前提(`recorder.py:471` が [N,10,2] に積む) |
| 620-622 | JPEG(q92)を一度通してから `normalize_inputs` | 履歴の画像も、この正規化済みの配列を貯める |
| 623-626 | ジャンルの one-hot のキャッシュ | 方策に依存しない |
| 628-629 | `runner.run_numpy(image=, tof=, imu=, state=, mode=)` — **5つの名前を直書き** | 辞書で組み立て、追加の入力(履歴など)を足す |
| 630 | `pred = res['actions'][0] * 500 + 1500` | 出力の名前と形の約束 |
| 632-635 | 数値でなければ中立 | 方策に依存しない(安全) |
| 636-707 | 先読み → ゲイン → ToF ガード(記録だけ)→ 補助/リカバリー → 書き込み → 前の予測を戻す → 記録 | **方策に依存しない**(ここが「同じ安全の仕組み」の正体) |

### 2.8 管理画面 — `jetformer/web/server.py`・`static/app.js`・`static/index.html`

| 場所 | 中身 | 変える内容 |
|---|---|---|
| `server.py:21-24` | import | `policies`(torch なし)を足す |
| `server.py:609-625` | `POST /api/train/start`: `dataset, stride, epochs, batch, lr, pretrained, build_engine, command_input, mask_impacts` | `policy`・`policy_cfg` を受け取る(無ければ v0。古い画面のキャッシュからでも今と同じ) |
| `server.py:642-644` | `GET /api/runs` → `trainer.list_runs()` | 変更なし(`policy` が増えるだけ) |
| (新規) | — | `GET /api/policies` → 登録簿の一覧(名前・表示名・説明・状態・オプションの表・目安の速度) |
| `server.py:669-671` | `GET /api/drive/models` → `drive.list_models()` | 変更なし(`policy`・`drivable` が増えるだけ) |
| `server.py:684-701` | `DRIVE_CLIENT_VERSION = 6`、`POST /api/drive/start` | 走行の開始の中身は変わらないので、**版は上げなくて良い**(表示だけの変更) |
| `index.html:203-229` | 学習の入力欄(データセット 209、間引き、エポック、バッチ、学習率、224-227 のチェック) | 209 の後に「モデルの種類」の選択 `#tr-policy`・説明 `#tr-policy-info`・オプション `#tr-policy-opts` |
| `index.html:257-259` | 走行のモデルの選択・情報・ジャンル | 情報に方策を出す |
| `index.html:311-312` | 先読みのスライダー `max=10` | 出力10ステップの前提 |
| `app.js:765` | `TRAIN_RATE=200, VAL_RATE=650`(v0 の実測の枚/秒) | 方策の目安(`hints`)から |
| `app.js:769-776` | `loadTrain` | `/api/policies` を読んで選択肢を作る |
| `app.js:777-792` | `updateEstimate` | 方策の目安の速度で見積もる |
| `app.js:795-803` | 学習の開始の body | `policy`・`policy_cfg` を足す |
| `app.js:846-870` | `loadRuns`(858-862 に引数・「今の操作値なし」の印) | 方策の印を足す |
| `app.js:889-905` | `loadDrive`(894-896 で選択肢の文字) | 方策の表示名、`drivable` が false なら選べなくする |
| `app.js:906-926` | `renderDriveModel`(911-913 に情報) | 「モデルの種類・入力(履歴 K フレームなど)」を足す |

**比べるための注意**: 管理画面からの学習は `--impact-g` を渡さない(`training.py:72-80`)ので **4.0G**(`train.py:188` の既定)で外す。
モデル B は CLI から **2.5G** で学習している(`data/runs/clean_driving_20261011-nocmd-s3-20261011-024626/run.json` の `impact_g: 2.5`)。
管理画面で新しい方策を学習してモデル B と比べると、構造の差と衝突の除外の差が混ざる。**学習タブに「衝突とみなす加速度(2.5/4.0)」と
「乱数の種」を足す** か、比べるための学習は CLI(後述の一括の道具)で行う。管理画面の文言(`index.html:227` の「水平4G以上」)もこれに合わせる。

### 2.9 評価の道具

| 道具 | 行 | 前提 | 変える内容 |
|---|---|---|---|
| `tools/compare_models.py` | 28 `H=10`、31-40 `load_model`(`JetFormer(num_modes)`、`best.pth`)、43-70 `predict_scene`(フレームを順に。57-58 one-hot、63 `normalize_inputs`、64-65 5つの位置の引数、66 `out[0]`) | 1フレーム・5入力 | 共通の `load_run` で作る。順に回しているので、履歴は `Feeder` に1フレームずつ入れれば走行と同じになる |
| `tools/evaluate.py` | 111-148 `model_report`(113-118 構築、142-145 呼び出し) | 同上 | 同上 |
| `tools/rl_evaluate.py` | 96-104 `CONFIGS`(7通りの固定)、113-114 `POSTHOC_CONFIGS`、136-142 `load_state_dict`、158-167 `drive_outputs`(apply_gains と同じ式)、458-482 `_FrameSet`(1フレーム・4テンソル + ジャンル)、492-536 `cmd_predict`(519-520 構築、528 呼び出し)、661-683 `evaluate_all`(668 `names = list(CONFIGS)`)、988-1067 `cmd_engine_check`(1024-1026 空回しの入力を直書き、1043-1045 5つの名前) | R1・DE・直線・L1 の判定は **予測 [N,10,2] だけから計算**(方策に依存しない)。ただし比べる相手が `CONFIGS` に固定 | `_FrameSet` に追加の入力(履歴は aligned の `cam_files`・`timestamps_us` から)。`predict --extra 名前=run` で `pred_<名前>.npz` を作り、**別の小コマンド(または新しい道具)で「各方策 − モデル B」を計算する**。`heldout.json` の規則と `evaluation.json`(ゲートC)には足さない(事後の比較として別ファイル) |
| `tools/rl_policies.py` | 62-64 `GPU_PATTERNS`、360-377 `train_command`、459-479 エンジン、484-523 `check-engines`(501-503 構築、506 `ds[n]` の6個、510-513 5つの名前)、562-599 `common-val`(589-590 構築、593 6個、598 呼び出し) | v0 | `load_run` に置き換え。**`GPU_PATTERNS` に `benchmark_model.py` が無い**(新しい評価の道具と一緒に足す) |
| `tools/offline_rl.py` | 507-560 `cmd_features`(534-537 構築と `model.norm` へのフック、571 呼び出し) | v0 の融合トークン | 方策が任意で `embed()` を持つときだけ使えるようにする(無ければ止める) |
| `tools/benchmark_model.py` | 16-17 import、19 `NUM_MODES = 8`(今の学習結果は 6)、22-28 ダミー入力、50 構築、69-72 ONNX(複製)、81 5つの名前 | v0 | `--policy`・`--policy-cfg`・`--num-modes 6`・`--json`。書き出しは共通の関数を使う |

### 2.10 `JetFormer(...)` を直接呼んでいる所(実験のスクリプトを除く)

`tools/train.py:271`、`tools/compare_models.py:36`、`tools/evaluate.py:117`、`tools/rl_evaluate.py:519`、`tools/rl_policies.py:501`、
`tools/rl_policies.py:589`、`tools/offline_rl.py:534`、`tools/benchmark_model.py:50`、テスト `tests/test_rl_policy_training.py:159-160`。
(`experiments/20261011_day_summary/src/explore_data/genre_swap.py:34` は記録として残すので触らない。)

### 2.11 テスト

| テスト | 行 | 依存 |
|---|---|---|
| `tests/test_drive.py` | 23-30 `FakeRunner`(`run_numpy(**kw)` → `actions`)、75-108 `run_loop`(85-86 で `_loop` を位置の引数で呼ぶ)、262-293 収録シーンを走行の組み立てに流すと学習の入力と一致、339-347 `model_info` の `command_input`(古い設定は True) | `_loop` の引数の順・`model_info` の後方互換 |
| `tests/test_rl_policy_training.py` | 157-173 `expand_mode_state`(v0 のキー)、176-193 `_Fixed.forward(image, tof, imu, state, mode)` と6個のバッチ、196-208 `run_epoch`(重みあり/なし)、211-220 `--resume` の引数の比較、276-290 `train_command` | `run_epoch` のバッチの形・`forward` の引数 |
| `tests/test_rl_review.py` | 81-104 エンジンの名前を変えると走行の一覧から消える(最小の `model_config.json`) | ファイル名・最小の設定 |
| `tests/test_ai_dataset.py` | 265-267, 332-334 `JetFormerDataset` で読める | 6個のタプル |

---

## 3. ハードコードされている前提

| # | 前提 | 場所 | 新しい方策への影響 | v1(最小)での扱い |
|---|---|---|---|---|
| H1 | 入力は5つ(`image, tof, imu, state, mode`)、この名前と順 | `model.py:78`、`train.py:80, 169, 358`、`drive.py:416-418, 628-629`、`compare_models.py:64-65`、`evaluate.py:143-144`、`rl_evaluate.py:528, 1024-1026, 1045`、`rl_policies.py:510-513, 598`、`offline_rl.py:571`、`benchmark_model.py:71, 81` | 入力を減らす(例: カメラだけ)と ONNX から消える(§4)。増やすと全部の呼び出しで足りない | 名前で渡す辞書にする。走行は上位集合を渡す。基本の5つの **形は固定**、追加は登録した「種類」だけ |
| H2 | 出力は `actions` [B,10,2]、(µs−1500)/500 | `model.py:27, 72, 96-97`、`train.py:81, 170`、`train_data.py:179`、`drive.py:45, 560, 630, 636`、`recorder.py:471`、`compare_models.py:28, 48`、`rl_evaluate.py:158-167`、`index.html:312`、aligned の `future_actions [N,10,2]` | 経路+目標速度(TransFuser++・TCP 型)などは、**グラフの中で [10,2] に直す** 必要がある | **固定**。補助の出力は名前を変えて足して良い(走行は読まない) |
| H3 | ジャンル別ヘッド(`num_modes`、one-hot) | `model.py:62, 72, 96-97`、`train.py:106-137, 270-271`、`drive.py:100, 415, 623-626`、`benchmark_model.py:19` | どの方策も `mode` [1,num_modes] を受け取る。`--init-mode-copy` は v0 のキー名に依存 | 入力として必須(使わない方策では ONNX から消えても良い)。キー名は `genre_keys` で宣言 |
| H4 | 画像 224×224 | `model.py:24, 64`、`train_data.py:51-52`、`config.py:21`(カメラの保存も 224)、`drive.py:416`、`train.py:166` | 高解像度の方策は、収集から作り直しになる | **固定** |
| H5 | ToF 2×42×54 | `model.py:25, 32, 64`、`train_data.py:55-56`、`drive.py:416` | ToF の鳥瞰グリッドなどは、グラフの中で作るか、追加の入力の種類にする | 固定(派生はグラフの中で作るのを勧める) |
| H6 | IMU 10×6(100ms) | `model.py:26, 59`、`drive.py:38, 140, 149, 423`、aligned の `imu [N,10,6]` | 0.5〜1秒の IMU は、データセットに無い(フレームごとに10個だけ) | v1 では固定。長い IMU はデータセットの作り直し(段階3) |
| H7 | state 3(steer, throttle, eRPM)、`command_input=False` で前2つが0 | `train_data.py:59-62`、`model.py:60`、`drive.py:153-164, 541` | 方策に依存しない | そのまま |
| H8 | **1フレームだけ** | `train_data.py:174`、`drive.py:620-622`、`compare_models.py:53`、`rl_evaluate.py:470, 481` | 時間の情報(前のフレーム・動き)が入れられない | 追加の入力の種類 `frame_history`(段階2、§5.4) |
| H9 | バッチ1・固定の形・float32 の入出力 | `train.py:166-170`(`dynamic_axes` なし)、`trt_runtime.py:45-55` | 可変長(トークン数が変わる)・int の入力は不可 | **固定**。方策は固定の形で書き出す |
| H10 | エンジンは FP16、作業メモリ 1GB | `trt_runtime.py:13, 22-24`、`build_engine.py:15-20` | FP16 で精度が落ちる方策 | `model_config.json` の `engine` 欄で FP32 も選べるように |
| H11 | ファイル名が固定 | §2.6 の最後 | — | **変えない** |
| H12 | 構築は `JetFormer(num_modes=...)`(大きさは既定値) | §2.10 | 古い学習結果は `dim/layers/heads` を記録していない | `policy_cfg` を必ず書く。無ければ v0 の凍結した既定値 |
| H13 | 損失は L1 だけ | `train.py:81-82` | 速度の分類・補助タスク(ToF から深度など)が入れられない | 任意のフック `policy_loss`(サンプルごとの損失 [B])。検証の µs は L1 のまま |
| H14 | 最適化は AdamW(wd 1e-4)+ cosine で全体が同じ学習率 | `train.py:272-273` | 事前学習の部分を遅く学習したいとき | 任意のフック `param_groups`(段階3) |
| H15 | `pretrained` = ImageNet の ResNet-18 | `model.py:54-55`、`train.py:182, 271`、`index.html:224` | 別の背骨の重みはネットから取る必要がある(キャッシュは resnet18 だけ、§4) | 方策の説明に「事前学習の重み」を書き、無ければ学習の前に知らせる |
| H16 | 学習中は FP16 の autocast | `train.py:78-79` | 不安定な方策 | ヒント `amp: false` |
| H17 | `model.norm` へのフック | `offline_rl.py:537` | v0 の構造だけ | 任意の `embed()` |
| H18 | 学習の速さ・メモリの目安が v0 の実測 | `app.js:765`、`training.py:25` | 重い方策で見積もり・空きメモリの判定が外れる | `hints`(`train_rate`・`val_rate`・`min_free_mb`) |
| H19 | 学習結果の名前と一覧のキー | `training.py:63, 219-221` | 方策が名前で分からない | 名前に短い名前、一覧に `policy` |
| H20 | 走行は torch を読まずに一覧を作る | `drive.py:38`、`server.py:21` | 登録簿が torch を読むと、サーバーの起動が重くなる | 登録簿の `__init__` は torch なし。構築のときだけ遅延 import |
| H21 | 走行の入力がエンジンに足りないと、ループの中で `KeyError` | `trt_runtime.py:70-71`、`drive.py:628` | 中立で止まる(安全)が、原因が分かりにくい | 走る前に確かめる(§5.8) |
| H22 | 評価の比べる相手が固定(ゲートC) | `rl_evaluate.py:96-104, 668` | 新しい方策を R1・DE で比べられない | 事後の比較として別に計算 |

---

## 4. このメモのために確かめたこと(CPU だけ。GPU・サービス・車は触っていない)

1. **ONNX の書き出しは使っていない入力を消す**(torch 2.10.0+cu130、`torch.onnx.export(..., dynamo=False, opset_version=17)`)。
   `forward(image, tof, imu, state, mode)` のうち `state`・`mode` しか使わない小さなモデルを、名前つきの引数に直すラッパーで書き出すと、
   グラフの入力は `state [1,3]`・`mode [1,6]` の2つだけになった(名前の対応は正しい)。
   → (a) 走行は5つを全部渡す(上位集合)。`TRTRunner.run_numpy` はエンジンの入力だけを読むのでこれで動く。
   (b) `model_config.json` の `inputs` は宣言ではなく **書き出した ONNX から読み直した名前** にする。
   (c) カメラだけ・ToF だけの方策(対照として有用)も、同じ仕組みで走れる。
2. モデル B(`clean_driving_20261011-nocmd-s3-20261011-024626/jetformer.onnx`)の入出力: opset 17、
   `image [1,3,224,224]`・`tof [1,2,42,54]`・`imu [1,10,6]`・`state [1,3]`・`mode [1,6]`(FLOAT)→ `actions [1,10,2]`。
   → v0 を登録簿から書き出したとき、**この形と名前に一致すること** を回帰テストにする。
3. 版: torch 2.10.0+cu130、torchvision 0.25.0、onnx 1.23.2、TensorRT 10.16.1.11。**onnxruntime・timm は入っていない**
   (ONNX の数値の確認は TensorRT とで行うことになる)。事前学習の重みのキャッシュは `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` だけ。
4. aligned のシーン(`clean_driving_20261011`): 1シーン 約600フレーム(20秒、30fps、`timestamps_us` の間隔 約33.3ms)、
   キーは `timestamps_us, tof, tof_age_ms, imu, vehicle, vehicle_age_ms, vehicle_valid, future_actions, future_valid, erpm_valid, cam_files, mode`。
   **全フレームの `cam_files` があるので、間引き(stride 3)で選んだフレームの前のフレームも読める**。IMU は各フレームの直前10個だけ。
   `ai_mix_20261011` は 109シーン(232〜601フレーム)、`intervention_val_20261011` は 29窓(147〜241フレーム)。
   収集は20秒ごとにシーンが切れる(`config.SCENE_SECONDS`)ので、**履歴はシーンの始めで足りなくなる**(K=3・間隔3フレームなら約0.3秒 = 各シーンの1.5%)。
5. `drive.py` の `_loop` はテストから位置の引数で呼ばれる(`test_drive.py:85-86`)。追加の引数は最後にキーワードで足す必要がある。
6. `test_rl_review.py:100` は `{"num_modes": 6, "trained_modes": [4, 5]}` だけの `model_config.json` で `model_info` が None でないことを確かめている
   → 新しい欄は全部「無ければ既定値」で読む必要がある。

---

## 5. 提案: 方策の登録簿(最小の形)

### 5.1 ファイルの構成

```
jetformer/policies/
  __init__.py      torch を読まない: 登録簿・PolicySpec・既定値・入力の表・model_config の解釈(drive.py・server.py が使う)
  feed.py          numpy だけ: 履歴の選び方(学習・走行・評価で共通)と、走行用の Feeder
  runtime.py       torch: build / load_run / 書き出し / 損失 / ジャンルを増やす(train.py と評価の道具が使う)
  jetformer_v0.py  build() が jetformer.model.JetFormer をそのまま返す(ラッパーなし → state_dict のキーが今と同じ)
  <新しい方策>.py  build() と nn.Module
```

`jetformer/model.py` はそのまま残す(定数の元、v0 のクラス)。

> 名前の注意: `tools/rl_policies.py` の「方策」はオフライン RL の **データの使い方の違い**(4本)で、構造は v0 のまま。
> 管理画面の表示は「モデルの種類」とし、コードの `policy` = 構造、と書き分ける。

### 5.2 `jetformer/policies/__init__.py`(torch を読まない)

```python
"""方策(モデルの構造)の登録簿。drive.py・web/server.py からも読むので、ここでは torch を import しない"""
import importlib
import json

CONTRACT = 1                       # 入出力の約束の版(model_config.json の 'contract')
DEFAULT = 'jetformer_v0'           # 'policy' が無い学習結果 = 今までのもの
FUTURE_STEPS = 10
BASE_SHAPES = {'image': [1, 3, 224, 224], 'tof': [1, 2, 42, 54], 'imu': [1, 10, 6], 'state': [1, 3]}   # + mode [1, num_modes]
FEED_KINDS = ('image', 'tof', 'imu', 'state', 'mode', 'frame_history')   # drive.py が作れる入力の種類(段階2で frame_history)


class PolicySpec(object):
    def __init__(self, name, title, module, defaults=None, options=(), description='', status='experimental',
                 extra_inputs=None, hints=None, genre_keys=None, slug=None):
        self.name, self.title, self.module = name, title, module      # module = 'jetformer.policies.jetformer_v0'
        self.slug = slug or name                                         # 学習結果の名前に入れる短い名前
        self.defaults = dict(defaults or {})                            # build() の引数の既定値(凍結する)
        self.options = list(options)        # 管理画面で変えられる項目 [{'key','label','type':'int|float|choice|bool','min','max','choices'}]
        self.description, self.status = description, status            # status: 'stable' | 'experimental'
        self._extra = extra_inputs          # cfg → [{'name','kind','shape',...}](torch なし)
        self.hints = dict({'train_rate': 200, 'val_rate': 650, 'min_free_mb': 3500, 'amp': True}, **(hints or {}))
        self.genre_keys = genre_keys        # {'embed': 'mode_embed.weight', 'heads': 'heads'}。無ければ --init-mode-copy は使えない

    def resolve(self, cfg=None):
        """既定値 + 指定。知らないキー・範囲外・options に無い項目の変更は ValueError(管理画面は 400 で返す)"""
        ...

    def input_specs(self, cfg, num_modes):
        """入力の表(書き出しの順)。基本の5つ + 方策が足すもの"""
        specs = [dict(name=k, kind=k, shape=list(v)) for k, v in BASE_SHAPES.items()]
        specs.append(dict(name='mode', kind='mode', shape=[1, int(num_modes)]))
        return specs + (self._extra(self.resolve(cfg)) if self._extra else [])

    def build(self, num_modes, cfg=None, pretrained=False):
        mod = importlib.import_module(self.module)          # ここで初めて torch を読む
        return mod.build(num_modes=int(num_modes), pretrained=bool(pretrained), **self.resolve(cfg))

    def describe(self):
        """GET /api/policies 用"""
        return {'name': self.name, 'title': self.title, 'description': self.description, 'status': self.status,
                'defaults': self.defaults, 'options': self.options, 'hints': self.hints,
                'genre_expand': self.genre_keys is not None}


REGISTRY = {}


def register(spec):
    REGISTRY[spec.name] = spec
    return spec


def get(name):
    if name not in REGISTRY:
        raise KeyError('知らないモデルの種類です: %s(登録されているもの: %s)' % (name, ', '.join(REGISTRY)))
    return REGISTRY[name]


def list_specs():
    return [s.describe() for s in REGISTRY.values()]


def from_config(cfg):
    """model_config.json(dict)→ (spec, policy_cfg)。'policy' が無い = 今までの学習結果 = jetformer_v0 の既定の大きさ"""
    spec = get(cfg.get('policy') or DEFAULT)
    return spec, spec.resolve(cfg.get('policy_cfg') or {})


def input_specs_of(cfg):
    """走行・評価用: model_config.json の input_specs。無ければ基本の5つ(古い学習結果)"""
    if cfg.get('input_specs'):
        return cfg['input_specs']
    specs = [dict(name=k, kind=k, shape=list(v)) for k, v in BASE_SHAPES.items()]
    return specs + [dict(name='mode', kind='mode', shape=[1, int(cfg['num_modes'])])]


def build_policy(name, num_modes, cfg=None, pretrained=False):
    return get(name).build(num_modes, cfg, pretrained)


register(PolicySpec(
    'jetformer_v0', 'JetFormer v0(今の標準)', 'jetformer.policies.jetformer_v0',
    defaults={'dim': 256, 'layers': 4, 'heads': 8}, options=(),       # v0 の大きさは管理画面では変えない(比べる基準)
    description='カメラ 49・ToF 42・IMU・操作値・ジャンルのトークンを Transformer 4層で1回融合し、ジャンル別ヘッドで 10ステップ先まで',
    status='stable', genre_keys={'embed': 'mode_embed.weight', 'heads': 'heads'}, slug='v0'))
# 新しい方策はここに register(...) を足す(モジュールは build() を持つ)
```

`jetformer/policies/jetformer_v0.py`:

```python
from ..model import JetFormer


def build(num_modes, pretrained=False, dim=256, layers=4, heads=8):
    return JetFormer(num_modes=num_modes, dim=dim, layers=layers, heads=heads, pretrained=pretrained)   # 包まない
```

### 5.3 方策のモジュールの約束(torch 側)

| 項目 | 必須 | 内容 |
|---|---|---|
| `build(num_modes, pretrained, **cfg)` | 必須 | `nn.Module` を返す |
| `forward(image, tof, imu, state, mode, **extra)` | 必須 | 基本の5つは名前で受け取る(使わなくて良い)。`extra` は `input_specs` で足したもの(`image_hist` など) |
| 戻り値 | 必須 | `Tensor [B,10,2]`(正規化)、または `{'actions': [B,10,2], <補助>: ...}`。経路+速度などに分けるときも、**`actions` はグラフの中で作る** |
| `policy_loss(out, target, batch)` | 任意 | サンプルごとの損失 `[B]` を返す(重みつきの学習でもそのまま掛けられる)。無ければ L1。`batch` には入力も入るので、補助タスク(例: ToF を正解にした深度)を作れる |
| `EXPORT_OUTPUTS` | 任意 | 書き出しに残す補助の出力の名前(既定は `('actions',)` だけ)。走行は `actions` しか読まない |
| `embed(...)` | 任意 | `tools/offline_rl.py features` 用の特徴 [B,D]。無ければその道具は使えない |
| `param_groups(lr)` | 任意(段階3) | 学習率を部分ごとに変えたいとき |
| ジャンルの埋め込み・ヘッドのキー名 | 任意 | `PolicySpec.genre_keys` で宣言すると `--init-mode-copy` が使える(v0 と同じ `mode_embed`・`heads` の形を勧める) |

`jetformer/policies/runtime.py`(torch)が持つ共通処理:

```python
def outputs(model, inputs):            # inputs: dict(名前 → Tensor)→ (actions [B,10,2], out)。dict でも Tensor でも
def sample_loss(model, out, target, inputs):    # model.policy_loss があればそれ、無ければ L1 の [B]
def example_inputs(specs, seed=0):     # 入力の表 → 書き出し用の固定の形のダミー(mode は one-hot)
class ExportWrapper(nn.Module):        # forward(*xs) → model(**dict(zip(names, xs))) → actions(+ EXPORT_OUTPUTS)
def export_onnx(model, path, specs):   # opset 17・dynamo=False・mha fastpath を切る・バッチ1 → onnx.load で読み直し、
                                       # 実際の入力の名前の並び(消えた入力を除く)と出力の形を返す。actions が [1,10,2] でなければ止める
def load_run(run_dir, ckpt='best', device='cuda'):
    """model_config.json → from_config → build → best.pth / last.pth(両方の形)を読む → (model.eval(), cfg, spec, specs)"""
def expand_modes(spec, state, model_state, copy_from):   # genre_keys に従う。train.expand_mode_state は v0 のキーでこれを呼ぶ
```

### 5.4 入力の種類

| 種類(`kind`) | 形 | 学習(`train_data.py`) | 走行(`drive.py`) | 段階 |
|---|---|---|---|---|
| `image` | [1,3,224,224] | 今のまま | 今のまま | 1 |
| `tof` | [1,2,42,54] | 今のまま | 今のまま | 1 |
| `imu` | [1,10,6] | 今のまま | 今のまま | 1 |
| `state` | [1,3] | 今のまま(`command_input`) | 今のまま | 1 |
| `mode` | [1,num_modes] | 今のまま | 今のまま | 1 |
| `frame_history` | [1,K,3,224,224](古い順。今のフレームは含めない)。引数 `frames=K`・`gap=g`(30fps のフレーム数) | 同じシーンの aligned から `history_index` で選んで読み、今のフレームと **同じ色の揺らぎ** を掛け、`normalize_image` | `Feeder` が JPEG を通して正規化した画像を時刻つきで貯め、同じ `history_index` の規則で選ぶ | 2 |
| `carry`(前の推論の出力を次の入力へ。例: フレームごとの特徴 [1,K,D]) | 方策が宣言 | 学習では過去のフレームから同じネットで作る | `Feeder` が出力を入力へ回す | 3 |
| `imu_long`(0.5〜1秒) | [1,M,6] | aligned に無い → データセットの作り直し | `imu_buf` を長く | 3 |

`feed.py`(numpy だけ)の履歴の規則(学習・走行・評価で同じ関数を使い、`test_replay_matches_training_inputs` と同じ形のテストで一致を確かめる):

```python
def history_index(ts_ns, i, frames, gap, fps=30):
    """フレーム i の過去 K 枚(古い順)。目標時刻 t_i − k·gap/fps(k = K..1)に対し、時刻が 目標 + 半フレーム 以下で一番新しいフレーム。
    無ければ(シーンの始め)いちばん古いフレーム。フレームの取りこぼしがあっても時刻で選ぶので、走行とずれない"""


class Feeder(object):
    """走行用。入力の表から必要な追加の入力を作る(基本の5つは drive.py が今のまま作る)"""
    def __init__(self, input_specs, fps=30): ...
    names = ...                              # 追加の入力の名前
    def push(self, t_ns, image_norm): ...    # 推論したフレームの正規化済みの画像(JPEG を通したもの)を貯める
    def extras(self, t_ns): ...              # {名前: [1,...] float32}
```

履歴の注意: (1) 学習ではシーンの始め(20秒ごと)で履歴が足りず、いちばん古いフレームで埋める。走行ではいつも本物の履歴がある。
同じログの前のシーンがデータセットにあればつなぐ、という改良は後回し。(2) 走行では、中立で止めた(`hold`)フレームは正規化の前に飛ばしているので
履歴に入らない。時刻で選ぶので大きくはずれないが、長く止めた後の最初の数フレームは古い画像が混ざる → 走り出しは `Feeder.reset()` で
「今の画像で埋める」に揃える(学習のシーンの始めと同じ)。(3) 走行の負荷: K=3 で正規化済みの画像 約1.8MB をホストからデバイスへ
(ピン留め)、背骨は K+1 回。v0 の推論 1.9ms に対し、履歴つきは計測してから決める(`benchmark_model.py`)。

### 5.5 `model_config.json`(足す欄。無ければ右の既定値)

| 欄 | 内容 | 無いとき(古い学習結果) |
|---|---|---|
| `policy` | 登録簿の名前 | `jetformer_v0` |
| `policy_cfg` | `build()` の引数(既定値も含めて全部書く) | v0 の凍結した既定値 `{dim:256, layers:4, heads:8}` |
| `contract` | 入出力の約束の版 | 1 |
| `inputs` | **書き出した ONNX から読み直した** 入力の名前(今もある欄。中身の決め方を変える) | `['image','tof','imu','state','mode']` |
| `input_specs` | `[{name, kind, shape, ...}]`(`inputs` にあるものだけ) | 基本の5つ(`mode` は `num_modes`) |
| `outputs` | `['actions', ...]` | `['actions']` |
| `engine` | `{'fp16': true, 'workspace_gb': 1.0}` | FP16・1GB |

今の欄(`num_modes`・`modes`・`future_steps`・`tof_channels`・正規化の定数・`dataset`・`stride`・`trained_modes`・`train_mode_counts`・
`steer_center_us`・`throttle_neutral_us`・`command_input`・`mask_impacts`)はそのまま書く。`run.json`・`result.json` にも `policy`・`policy_cfg` を足す。

### 5.6 学習(`tools/train.py`・`jetformer/train_data.py`)

- 引数: `--policy NAME`(既定 `jetformer_v0`、選択肢は登録簿)、`--policy-cfg '{"layers": 2}'`(JSON)。
  `POLICY_ARGS = {'policy': 'jetformer_v0', 'policy_cfg': None}` を `RL_ARGS` と同じ方式で `same_args` に入れる(既定値なら記録しない)
  → **前からの `last.pth` で `--resume` できる**、`tools/rl_policies.py` の `train_command` も変えずに同じ学習になる。
- データセット: `JetFormerDataset(..., extra_inputs=None)`。追加の入力があるときだけ、戻り値の **最後に dict** を足す:
  `(image, tof, imu, state, mode, target[, weight][, extras])`。既定では今と同じ6個(+重み)。
- `run_epoch`(62-103):

```python
image, tof, imu, state, mode, target = (t.to(device, non_blocking=True) for t in batch[:6])
extras = {k: v.to(device, non_blocking=True) for k, v in batch[-1].items()} if isinstance(batch[-1], dict) else {}
weight = batch[6].to(device).float() if len(batch) > 6 and not isinstance(batch[6], dict) else None
inputs = dict(image=image, tof=tof, imu=imu, state=state, mode=mode, **extras)
with ...autocast(enabled=amp and spec_amp):
    actions, out = runtime.outputs(model, inputs)
    l1 = (actions.float() - target).abs().mean(dim=(1, 2))          # 報告する誤差: どの方策でも同じ L1(µs で比べられる)
    per_sample = runtime.sample_loss(model, out, target, inputs) if train else l1   # 学習の損失(既定は l1 と同じ)
    loss = per_sample.mean() if weight is None else (per_sample * weight).mean() / weight_scale
```

  (`tests/test_rl_policy_training.py` の `_Fixed` は `forward(image, tof, imu, state, mode)` を位置で受け取るが、名前で渡しても動く。)
- 構築(271): `model = policies.build_policy(args.policy, num_modes, cfg, pretrained=not args.no_pretrained)`。
  学習の前に `example_inputs` で1回通し、`actions` が `[B,10,2]` でなければ止める(30分の学習の後で書き出しに失敗しないように)。
- `--init`(296-301): `Path(args.init).parent / 'model_config.json'` があれば読み、**`policy` と `policy_cfg` が同じでなければ止める**。
  `--init-mode-copy` は `spec.genre_keys` が無ければ止める。別の方策の重みの一部(背骨など)を使う「部分的な初期値」は段階3
  (`strict=False` と、合わなかったキーの一覧を `result.json` に残す)。
- RL のオプション(`--drop-mask`・`--sample-weights-npz`・`--mode-remap-npz`・`--mode-remap-dropout`・`--seed`)は **データの使い方なので、どの方策でも使える**。
- 書き出し(353-366): `actual = runtime.export_onnx(model, out/'jetformer.onnx', specs)` → `model_config.json` に §5.5 の欄。

### 5.7 ONNX とエンジン

- 書き出しは `runtime.export_onnx` の1つにまとめ、`train.py:163-170` と `benchmark_model.py:69-72` から呼ぶ。
- `tools/build_engine.py`: ONNX と同じフォルダーの `model_config.json` があれば `engine` 欄を読む(無ければ今と同じ FP16・1GB)。
  作った後にエンジンを読み直し、**入力がすべて `FEED_KINDS` の名前で形が `input_specs` と同じ、出力 `actions` が [1,10,2]** でなければ
  終了コード 2(管理画面は「エンジンの作成に失敗」と表示し、走行の一覧に壊れたモデルが出ないよう、エンジンのファイルを残さない)。
- `trt_runtime.TRTRunner`: `self.shapes = {名前: 形}` を公開する(走行の空回し・確認に使う)。動きは変えない。
- `jetformer/training.py`: 呼び方は今のまま(ファイル名も同じ)。

### 5.8 走行(`jetformer/drive.py`)

- `model_info`(87-109)に足す: `policy`(無ければ `jetformer_v0`)、`policy_title`(登録簿に無ければ名前のまま)、
  `input_specs`(`policies.input_specs_of(cfg)`)、`drivable`(`input_specs` の種類がすべて `FEED_KINDS` にあるか)、`why_not`。
  エンジンが無ければ今まで通り None(`test_rl_review.py:94-104`)。`drivable=False` の学習結果は一覧に出すが選べなくし、`Driver.start` でも断る。
  ※ 走行に必要なのは **エンジンと入力の種類だけ** で、登録簿にその方策が無くても(別の枝で学習したものでも)走れる。
- `_run`(409-420): `TRTRunner` を作った直後に、`runner.inputs` がすべて「作れる名前」で、`runner.shapes` が `input_specs` と同じか、
  `runner.outputs['actions']` が (1,10,2) かを確かめ、違えば `RuntimeError`(phase=error、中立のまま)。
  空回しの入力は `runner.shapes` から作る(416-418 の直書きをやめる)。追加の入力があれば `Feeder` を作る。
- `_loop`: 引数の最後に `feeder=None`。622 の後で `feeder.push(cam.t_ns, img)`、628-629 は

```python
feed = {'image': img[None], 'tof': tof[None], 'imu': imu_n[None], 'state': state[None], 'mode': mode_cache[mode_id]}
if feeder is not None:
    feed.update(feeder.extras(cam.t_ns))
res = runner.run_numpy(**feed)        # エンジンが使わない入力は読まれない(上位集合で良い)
```

- 収集の `drive_info`(444-445)に `policy`・`policy_cfg` を足す。
- **変えない所**: `PARAM_LIMITS`、`apply_gains`、`predicted_now`、`vehicle_state`、ToF ガード(記録だけ)、`assist`、`LapCounter`、
  心拍・CH3 の確認・中立の出し方。→ どの方策でも同じ安全の仕組みで走る。

### 5.9 管理画面

- `GET /api/policies` → `policies.list_specs()`(torch なし)。
- `POST /api/train/start` に `policy`・`policy_cfg`(無ければ v0)。`TrainingManager.start(..., policy=DEFAULT, policy_cfg=None)` で
  `spec.resolve()` が通るか確かめる(だめなら 400)。空きメモリは `max(MIN_FREE_MB, spec.hints['min_free_mb'])`。
  学習結果の名前に `spec.slug`(v0 は今の名前のまま)。コマンドは既定以外のときだけ `--policy`・`--policy-cfg`。
- 学習タブ: データセットの下に「モデルの種類」の選択、説明、`status='experimental'` なら「試験的」の印、`options` から作る入力欄。
  見積もり(`updateEstimate`)は方策の `train_rate`・`val_rate` で。学習結果の一覧に方策の印。
  あわせて「衝突とみなす加速度(2.5 / 4.0)」と「乱数の種」を詳細の欄に足す(§2.8 の注意)。
- 走行タブ: モデルの選択肢に方策の表示名、`drivable=false` は選べない(理由を表示)、情報欄に「モデルの種類・入力(履歴 K フレームなど)」。
  走行の開始の body は変わらないので `DRIVE_CLIENT_VERSION` は上げない。

### 5.10 評価の道具

- 9か所の `JetFormer(num_modes=...)` + `torch.load(best.pth)` を `policies.runtime.load_run(run_dir, ckpt)` に置き換える
  (`rl_evaluate.load_state_dict` の「best/last 両方の形を読む」をここへ移す)。
- 順に予測する道具(`compare_models.predict_scene`・`evaluate.model_report`)は、`Feeder` にフレームを1枚ずつ入れれば走行と同じ入力になる。
- ばらばらに予測する道具(`rl_evaluate._FrameSet`・`rl_policies` の `JetFormerDataset` を使う所)は、`history_index` で aligned から履歴を読む。
- **R1・DE の比較**: `rl_evaluate.py` の判定(R1 の進入のスロットル・進入 eRPM の見積もり、DE のステア、直線の前進の強さ、人のきれいな走行の L1、
  前進の偏り)は予測 [N,10,2] と `drive_outputs`(ゲートD の固定設定: ステア 1.3・速度 1.0・先読み 3)だけで計算できる。
  `CONFIGS`(96-104)と `heldout.json` の規則は固定なので、そこには足さず、`predict --extra 名前=run` で `pred_<名前>.npz` を作り、
  新しい小コマンド(例 `rl_evaluate.py zoo --names ...`)か新しい道具で「各方策 − モデル B」を同じ関数(`heldout_passes`・`_pass_values`・
  `entry_estimate`・`_erpm_map`・`cluster_bootstrap`)で計算して `eval/zoo_<名前>.json` に書く。`evaluation.json` は触らない。
  `clean_driving_20261011` で学習した方策にとって、保留のフレーム(ai_mix の val = 人の 11シーン + AI のログ2本、intervention_val)は全部学習に使っていない。
- `engine-check`(1024-1026, 1043-1045): 空回しは `runner.shapes` から、追加の入力は `Feeder` で。
- `benchmark_model.py`: `--policy`・`--policy-cfg`・`--num-modes`(既定を 8 → 6 に)・`--json`。パラメータ数、PyTorch FP16、TensorRT FP16 の平均と p99、
  TensorRT と PyTorch(FP32)の差の最大、エンジンの大きさを出す。`rl_policies.GPU_PATTERNS` に `benchmark_model.py` と新しい道具を足す。
- 学習結果の差は学習のたびのばらつきで約10µs あるので(`fast_and_safe_plan.md`)、方策の比べ方は「同じデータ・同じ設定(モデル B と同じ:
  stride 3・今の操作値なし・2.5G・30エポック・lr 3e-4・バッチ 64)で、乱数の種を2〜3通り」を基本にし、L1 より R1・DE の動きを重く見る。

### 5.11 テスト(足すもの)

| テスト | 中身 | GPU |
|---|---|---|
| 登録簿は torch を読まない | `import jetformer.policies` の後で `'torch' not in sys.modules`(新しいプロセスで) | 不要 |
| 古い設定は v0 | `from_config({'num_modes': 6})` → `jetformer_v0`・`{dim:256, layers:4, heads:8}`。`input_specs_of` が基本の5つ | 不要 |
| v0 のキーが今と同じ | `build_policy('jetformer_v0', 6).state_dict().keys() == JetFormer(num_modes=6).state_dict().keys()` | 不要(CPU) |
| v0 の書き出しが今と同じ形 | `export_onnx` の入力が `image/tof/imu/state/mode`、形がモデル B の ONNX と同じ、出力 `actions [1,10,2]`(tmp に書く) | 不要(CPU) |
| 登録した全方策 | 小さい cfg で CPU の forward が [2,10,2]、ONNX に書き出せる、入力が `FEED_KINDS` の中、`resolve` が知らないキーを断る | 不要(CPU) |
| 消えた入力 | `mode` を使わない方策 → `model_config.inputs` に `mode` が無く、走行の上位集合の入力で `FakeRunner` が動く | 不要 |
| 履歴の規則が学習と走行で同じ | 収録シーンを `Feeder` に流した履歴と、`history_index` で aligned から作った履歴が一致(`test_replay_matches_training_inputs` と同じ作り) | 不要 |
| 走行の確認 | エンジンの入力に知らない名前 → `_run` が走らずに error・中立。`_loop` を今の位置の引数のまま呼べる | 不要(偽物) |
| 学習のバッチ | `run_epoch` が「6個」「6個+重み」「6個+重み+dict」「6個+dict」で正しく重みを取り出す | 不要(CPU) |
| `--resume` | 既定の方策では `same_args` が今と同じ(前からの `last.pth` で再開できる) | 不要 |
| `--init` の確認 | 方策が違う学習結果の `best.pth` を `--init` に渡すと止まる | 不要 |

---

## 6. 後方互換の規則(まとめ)

1. `model_config.json` に `policy` が無い → `jetformer_v0`・`policy_cfg` は v0 の凍結した既定値(`JetFormer` の既定値と同じなので、今の学習結果は同じ形で作り直せる)。
2. v0 は `JetFormer` をそのまま返す → `best.pth`・`last.pth`・`best_epoch25.pth` などのキーは変わらない。
3. ONNX・エンジンのファイル名は変えない → 走行の一覧・`rl_review.py disable-engines`・`rl_policies.engine_disabled` はそのまま。
4. 走行は5つの基本の入力を必ず作り、エンジンが使うものだけが読まれる → 今のエンジン(5入力)はそのまま走る。
5. `same_args` に方策を足すのは既定以外のときだけ → 前からの `last.pth` で再開できる。`rl_policies.train_command` も同じコマンドのまま。
6. データセットの戻り値は、追加の入力が無ければ今と同じ6個(+重み)→ `rl_policies`・`rl_evaluate`・テストの位置の取り出しはそのまま。
7. 管理画面の古いキャッシュからの学習の開始(`policy` 無し)は v0。走行の開始の body は変わらない(`DRIVE_CLIENT_VERSION` は 6 のまま)。
8. `evaluation.json`(ゲートC)と `heldout.json` は触らない。新しい方策の比較は別のファイル。

---

## 7. 実装の順番(小さく、各段階で走行できる状態を保つ)

| 段階 | 中身 | 終わりの確認 |
|---|---|---|
| 0(動きは変えない) | 登録簿・`jetformer_v0`・`runtime.load_run/export_onnx`。9か所の構築を置き換え。`benchmark_model.py` の書き出しを共通化 | §5.11 の v0 のテスト、既存のテストが全部通る。モデル B を `load_run` で読んだ予測が今の `pred_B.npz` と同じ(`rl_evaluate crosscheck` と同じ考え方) |
| 1(基本の5入力の方策) | `--policy`・`--policy-cfg`、`model_config.json` の欄、`build_engine.py` の入出力の確認、`drive.model_info`・`_run` の確認、管理画面の選択・表示、`/api/policies`、`benchmark_model.py --policy` | 新しい方策を1つ(5入力のもの)CLI と管理画面で学習 → エンジン → 走行タブの一覧に出て、推論の確認(dry run)で中立のまま推論できる |
| 2(フレームの履歴) | `feed.py`(`history_index`・`Feeder`)、データセットの `extra_inputs`、`_loop` の `feeder`、評価の道具の履歴 | 収録シーンでの一致のテスト、`benchmark_model.py` で推論の時間 |
| 3(任意) | `policy_loss`・補助の出力、`carry`、`imu_long`(データセットの作り直し)、部分的な初期値、`param_groups` | 方策ごと |

---

## 8. 候補の構造に課す制約(候補を考える側へ)

- **入力**: `image [1,3,224,224]`(ImageNet 正規化の RGB)・`tof [1,2,42,54]`(距離/4000(0=無効)・信頼度/7)・`imu [1,10,6]`(加速度/9.8・角速度 rad/s)・
  `state [1,3]`(モデル B と同じく今の操作値を入れないなら前2つは0、eRPM/20000)・`mode [1,num_modes]` のうち使うもの。
  段階2から `image_hist [1,K,3,224,224]`。ToF の鳥瞰グリッドなどの派生は **グラフの中で作る**(走行の入力を増やさない)。
- **出力**: `actions [1,10,2]`(33ms 刻みの 1〜10フレーム先、(µs−1500)/500)。経路+目標速度に分ける構造も、グラフの中でここへ直す。
  走行はこの後にゲイン・先読み・ToF ガード・補助を掛ける。
- **書き出し**: opset 17、TorchScript の書き出し(`dynamo=False`)、固定の形・バッチ1・float32 の入出力。TensorRT 10.16 の FP16 で動く演算だけ
  (独自の演算・データで変わる分岐・可変長は不可。RNN のループは避け、注意機構と畳み込みを勧める)。
- **時間**: v0 は TensorRT FP16 で 1.9ms、走行のループ 5.6ms、カメラ → PWM 約28ms(30fps)。新しい方策の推論は **6ms 以下を目安**
  (カメラ → PWM の増分を数ms に抑えるため。これより重い場合は計測して判断)。
- **学習**: Orin Nano 8GB・スワップなしで、学習の最大使用量は v0 で約3GB(`MIN_FREE_MB = 3500`)。バッチ 64 で同じ程度に収まること。
  データは約6千サンプル(52シーン、stride 3)で、v0 も過学習している(学習 13µs・検証 36µs)→ パラメータを増やすほど不利。
- **事前学習の重み**: キャッシュは resnet18 だけ。ほかの torchvision の重みは、最初の学習でネットにつながっている必要がある。timm は無い。
- **ジャンル**: `mode` の one-hot を受け取り、できれば v0 と同じ `mode_embed`・`heads` の形(`--init-mode-copy` と RECAP 型のジャンル 6・7 が使える)。
- **比べ方**: 検証の誤差はどの方策でも L1 [µs](同じ定義)。差は学習のばらつき(約10µs)と比べ、R1・DE の動き(§5.10)で判断する。

---

## 9. 未解決・リスク

- 履歴の入力は、シーンの区切り(20秒)で学習と走行の条件が少し違う(§5.4)。
- 管理画面からの学習は 4.0G・乱数の種なしなので、そのままではモデル B と比べられない(§2.8)。
- `onnxruntime` が無いので、書き出しの数値の確認は TensorRT(GPU)で行う必要がある(`benchmark_model.py`・`rl_policies.py check-engines` の手順)。
- 重い方策は学習の時間・メモリの見積もり(`hints`)を実測で入れるまで、管理画面の見積もりが外れる。
- `tools/offline_rl.py features` は v0 の融合トークン専用(新しい方策は `embed()` を実装しない限り使えない)。
