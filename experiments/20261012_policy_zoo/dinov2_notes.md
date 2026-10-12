# 方策 `dinov2`(DINOv2 の画像特徴)の実装メモ

作成 2026-10-12 ・ 担当 P-dino ・ 対象 `/home/jetson/jetformer`(main 7239fea + 枠組みの作業ツリー。commit はしていない)・ **GPU は使っていない(CPU だけ)**

spec.md §3.4 のとおりに作った。`dinov2.py` の `POLICY_STUB` の行は消したので、登録簿の `ready()` は重みがあれば True(管理画面・`tools/train.py` で選べる)。

## 1. 作ったもの

| ファイル | 中身 |
|---|---|
| `jetformer/policies/vit.py` | DINOv2 ViT-S/14 の最小の実装(torch.hub・xformers に依存しない)。`DinoViTS14`(キーは公式と同じ)・`load_official(path)`(`mask_token` を捨て、`pos_embed` 37×37 → 16×16 を読み込むときに1回だけ公式と同じ bicubic・antialias なし・`scale_factor=(16+0.1)/37` で補間、`strict=True`)・`interpolate_pos`・`convert_official`(レジスタ版は分かる例外) |
| `jetformer/policies/dinov2.py` | `DinoV2Policy`・`build`・`prepare`・`pool_patches`・`image_token_count`・`weights_path`。`train()` の上書き(ViT は常に eval)、`param_groups(lr)`(凍結した ViT は入れない。`finetune_last > 0` なら ViT の群を lr × `lr_mult`)、`embed()`(`offline_rl features` 用、v0 と同じ「最後の LayerNorm の後の出力トークン」)。重みの取得の手順は先頭の docstring |
| `tests/test_policy_dinov2.py` | 20 件・CPU 約16 秒(下の §3) |

形(既定): ViT(凍結)→ CLS + 2×2 平均の 64 パッチ → `image_proj`(LayerNorm → Linear 384→256)→ + ToF 42・IMU・操作値・ジャンル・query = **111 トークン** + `pos` →
`fusion`(Encoder 4層・d256・FFN 512・dropout 0.1)→ `norm` → 最後のトークン → `trunk` → `heads`(ジャンル×20 行、v0 と同じ並び)。
層の名前は bench の `DinoProto` と同じ(`vit`・`image_proj`・`tof_encoder`・`imu_mlp`・`state_mlp`・`mode_embed`・`query`・`pos`・`fusion`・`norm`・`trunk`・`heads`)。

## 2. 確かめたこと(すべて CPU)

| 項目 | 結果 |
|---|---|
| 公式のソースとの一致 | 公式の `facebookresearch/dinov2`(commit 7764ea0)の生のファイルを取得し(スクラッチ、リポジトリには入れていない)、hub の `dinov2_vits14` と同じ設定で同じ重みを読んで、実フレーム 8 枚 + 乱数 2 枚の `forward_features` と比べた: **最大の差 0.0(ビット単位で一致)**、補間した位置の埋め込みの差 0.0。スクリプトと結果は `dinov2_check/`。テストには公式の出力の指紋(決めた入力の CLS・パッチの値)を入れた(ネットなしで退行を検出できる) |
| 重みのファイル | `data/pretrained/dinov2_vits14_pretrain.pth` の sha256 = 公式(`b938bf1b…aef60cd9`)。取り直していない |
| bench の試作との一致 | bench の `dinov2_p2.pth`(計測に使った重み)を `strict=True` で読めて、`DinoProto` と出力の差 **0.0**・state_dict のキーの並びも同じ |
| 大きさ | 24.50M(学習 **2.87M**)。`finetune_last 2` で学習 6.42M。spec の 24.5M・2.9M ± 0.2M に入る |
| ONNX | 実データの CPU の試し学習(下)で書き出した ONNX は 1,054 ノード。bench の `dinov2_p2.onnx`(1,096)との違いは、何もしない `Identity` 41 個と `Concat` 1 個だけ(bench は小さい3トークンを `stack` した)。補間(`Resize`)・ループ・`If` は無い。`onnx.reference`(numpy の参照の実行)で PyTorch FP32 との差 **8.9e-7(0.0004 µs)** |
| 試し学習 | `CUDA_VISIBLE_DEVICES= nice python tools/train.py --policy dinov2 --data data/datasets/clean_driving_20261011 --out <スクラッチ> --epochs 1 --max-steps 2 --batch 4 --workers 0 --stride 30 --no-command-input --impact-g 2.5` が 19 秒で終了。`best.pth` 98 MB・`jetformer.onnx` 98 MB・`last.pth` 121 MB・`model_config.json`(`policy`・`policy_cfg` 全キー・`inputs` 基本の5つ・`outputs` actions・`engine` fp16・`params_m` 24.5/2.87)。`drivable` と `check_io` は合格 |
| テスト | `tests/test_policy_dinov2.py` 20 passed(約16 秒)。関係するもの(`test_policies`・`test_policy_ui`・`test_policy_post`・`test_policy_eval`)72 passed。全部(他の担当の新しいテストを含む)319 passed・2 skipped |

## 3. テストの中身(`tests/test_policy_dinov2.py`)

登録簿の項目と仮のモジュールでないこと / 既定の大きさとトークン数 111 / 公式と同じ形の合成の重み(`mask_token`・`pos_embed [1,1370,384]`)を `strict=True` で読み、`pos_embed` が [1,257,384]・キーの不足とレジスタ版は例外 /
補間が公式の式(`scale_factor`、`size` とは違う)・定数は定数のまま / 実際の重みがあれば sha256 と公式の出力の指紋 / `pretrained=False` は重みのファイルに触れない(`torch.load` などを monkeypatch)/
重みが無い・`vit_depth` ≠ 12・知らない variant・範囲外の設定の例外、12 ブロックの合成の重みを `pretrained=True` で読む / `prepare`(無い → SystemExit、sha256 違い → 警告して続ける、2回目は計算し直さない)/
凍結(`requires_grad` False・`param_groups` に入らない・逆伝播で ViT に勾配なし・凍結の特徴は計算グラフなし)/ `finetune_last=2` で2群・学習率 ×0.1(`lr_mult` を変えると変わる)・前のブロックは勾配なし /
`train()` の後も ViT は eval / `--init` の確認が LP-FT(`finetune_last`・`lr_mult`)を通し、`pool` の違いは止める / トークン数 111・303・110(CLS なし)と forward [2,10,2]・`embed` [2,dim] /
**画像の左右・上下とトークンの格子の対応**(ViT 0 ブロックで、左上・右上・左下・中の 28×28 px を変えると、対応する1つのトークンだけが変わる = 左右の反転なし)/
ジャンルのヘッドが 20 行/ジャンル / 小さなバッチで損失が半分未満(凍結の ViT は変わらない)/ ONNX の入出力が約束どおり・`AveragePool` あり・`Resize`・ループ無し / `load_run` の往復で予測が同じ・`drivable`。

この方策にはラベル・制御器が無い(出力は v0 と同じ `actions` だけ)ので、符号の確かめは「画像の左右がトークンの並びで保たれる」ことと、ジャンルのヘッドの並びにした。

## 4. 速さ・エンジン(GPU で確かめていない。bench の実測が当てはまる理由)

ONNX のグラフが bench の `dinov2_p2` と同じ(違いは何もしないノードだけ)なので、bench の TensorRT 10.16 FP16 の実測がそのまま当てはまる見込み:
エンジンだけ 中央 3.86 ms / p99 4.10 ms、`run_numpy` 5.0 ms(p99 5.3 ms)、エンジン 49 MB・作成 105 秒(作成のプロセスのピーク 3.9 GB)、学習 1ステップ 270 ms・0.5 GB。

**注意(ゲートに掛かるかもしれない)**: bench の FP16 エンジンと PyTorch FP32 の `actions` の差は、実フレーム 16 個で最大 **4.15 µs**(平均 0.64 µs)だった
(ViT は公式の重み・頭は乱数)。spec §5.1 のゲートは最大 2.5 µs なので、学習した重みでも超えるかもしれない。超えたら spec §3.4 のとおり、
その学習結果の `model_config.json` の `engine.precision` を `bf16` → `fp32` の順に変えて作り直す(M)。FP32 で `run_numpy` が 6 ms を超えたら走行の候補から外す。
もう1つの手(試していない・`build_engine.py` の変更が要る): LayerNorm・Softmax の層だけ FP32 に固定する(`OBEY_PRECISION_CONSTRAINTS`)。

## 5. 学習の使い方

- 本体(spec §4.1): `--policy dinov2`(`--policy-cfg` なし)。`prepare` が重みの有無と sha256 を確かめる(違えば警告して続ける)。
  `--no-pretrained` は止まる(枠組みが確かめる)。重みは学習のときだけ読む。評価・`load_run` は `best.pth`(ViT の重みも入る)だけで作る。
- LP-FT(spec §4.3 の O3): `--policy dinov2 --policy-cfg '{"finetune_last": 2}' --init data/runs/<dinov2 種0>/best.pth`。
  最後の2ブロックと `vit.norm` を学習率 3e-5(lr × `lr_mult` 0.1)で学習する。`finetune_last`・`lr_mult` は `train_only_keys` なので `--init` の確認を通る。
  `finetune_last` は登録簿の `choices` が [0, 2] なので、CLI でも 0 か 2 だけ(1・3 などは `resolve` が断る)。
- `use_cls false`(CLS を使わない、110 トークン)・`pool 1`(303 トークン、bench で `run_numpy` 5.5 ms)も作れる。

## 6. 共有のファイルで必要な変更(M が決める)

1. **README**: 重みの取得の手順の段落が無い(例外の文言と登録簿の `ready()` の理由が「取得の手順は README」と言っている)。`dinov2.py` の先頭の docstring の手順
   (公式の URL を `.part` に `curl -fL` → `sha256sum` を確かめる → `mv`)を「モデルの種類(方策)」の表の下に1段落で足してほしい。
2. 任意: FP16 のゲートに落ちたときの中間の手として、`build_engine.py` に「LayerNorm・Softmax だけ FP32」の精度の選択肢(§4)。今は bf16・fp32 だけ。

## 7. 正直な見込み

- 机上の L1 の見込みは −10〜+15 µs で、4つの中で一番読めない(`design_research.md` §7)。凍結の特徴は、床のテープなど細い物の細部を 14 px のパッチ +
  2×2 の平均で粗くするので、v0(ResNet-18 を全部学習)に負ける可能性もある。差が出るとすれば「ずれへの強さ」(副指標)かもしれない(仮説)。
- 別々に学習した結果は、種・チェックポイントだけで最大 約10 µs 違う。2本の種の範囲と δ で判定すること(spec §5.2)。
- `best.pth`・`jetformer.onnx` が各 98 MB、`last.pth` 121 MB(学習結果1本で エンジン込み 約370 MB)。ディスクの空きは 33 GB あった。
