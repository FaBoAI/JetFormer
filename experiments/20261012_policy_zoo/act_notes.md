# 方策の比較: `act`(ACT 型)の実装メモ

作成 2026-10-12 ・ 担当 P-act ・ 対象 `/home/jetson/jetformer` の作業ツリー(commit はしていない)・ **GPU は使っていない**(CPU だけ)

## 1. 作ったもの

| ファイル | 中身 |
|---|---|
| `jetformer/policies/act.py` | `ACTPolicy`(CVAE なし)・`ACTCVAEPolicy`(CVAE つき)・`build(num_modes, pretrained=False, **cfg)`。`POLICY_STUB` の行は消したので、登録簿の `ready()` は `(True, '')` になり、管理画面と `tools/train.py` で選べる |
| `tests/test_policy_act.py` | 15 本。CPU だけで 23 秒(負荷が低いとき。ほかの作業と同時だと 40〜60 秒になる) |

構造(spec.md §3.1 のとおり):

```
blocks.sensor_tokens → [B,94,256](CVAE なら + 潜在トークン 1)+ pos → encoder 4層 → enc_norm = memory
queries [1,10,256](1 クエリ = 1 フレーム先)→ decoder 2層(自己注意 + memory への交差注意)→ dec_norm
  → heads Linear(256, M×2)(10 クエリで共有)→ ジャンルを one-hot で選ぶ → actions [B,10,2]
```

- 層の名前は v0 と同じ(`image_backbone`・`image_proj`・`tof_encoder`・`imu_mlp`・`state_mlp`・`mode_embed`)+ `pos`・`encoder`・`enc_norm`・`queries`・`decoder`・`dec_norm`・`heads`。
  センサーの層のキーと形が v0 と同じことはテストで確かめた(将来、v0 の重みから部分的に始められる)。
- `heads` の行 = ジャンル×2 + [ステア, スロットル](`genre_keys.heads = {'heads': 2}`)。`runtime.expand_modes` でジャンルを増やすと、写した元と同じ出力になる(テスト)。
- 出力の意味は v0 と同じ(正規化 (µs − 1500)/500、33 ms 刻みの 1〜10 フレーム先)。正解をそのまま写すので符号の決まりも同じ:
  ステア > 1443 µs = 左、スロットル < 1500 µs = 前進。小さなバッチを覚えさせるテストで、左・右・前進・ステップの並び(だんだん左へ切る列)が正解どおりに出ることを確かめた。
- `embed(...)`(任意のフック)を足した: 10 個のクエリの Decoder の出力の平均 [B,256]。`tools/offline_rl.py features` で使える。

### CVAE(既定オフ、比較の本体には入れない)

- `cvae=True` のときだけ `ACTCVAEPolicy`(`USES_TARGET = True`・`policy_loss` あり)。CVAE なしの `ACTPolicy` には `policy_loss` が無いので、
  学習の損失・`history` は v0 と同じ扱い(L1 だけ、`train_loss` の欄なし)。
- 学習中に target があるとき: [`cvae_cls`・`mode_embed(mode)`・`cvae_action(target)` × 10] + `cvae_pos` → `cvae_encoder`(2層)→ 先頭 → `cvae_out` → μ・logσ²
  (logσ² は [−10, 10] に切る)→ z。KL はサンプルごと [B]・float32(AMP でも)。損失 = L1 + `kl_weight`·KL、`loss_parts = {'kl': 平均}`。
- 検証・推論・書き出し・target なしは z = 0 → 潜在トークン = `latent_proj` のバイアス。書き出したグラフに CVAE の符号化器は入らない
  (LayerNorm の数が CVAE なしと同じ・乱数の演算なし、をテストで確かめた)。eval では target の有無・符号化器の重みで出力が変わらない(テスト)。
- 実データの試し学習(2 ステップ)で `train_parts.kl = 2.93`、`train_loss = 30.2`(β = 10 で KL が損失の大半)。KL が 0 に張り付くか(z が使われない =
  普通の ACT と同じ)は、30 エポックの学習の最後の `train_parts.kl` で見る(spec §3.1。< 0.01 なら報告に書く)。小さな合成バッチでは 100 ステップで 1e-5 程度まで下がった。

### 予測の重ね合わせ(temporal ensembling)

ACT の論文の手法だが、方策に依存しない走行側の処理として枠組み(F)の `jetformer/policies/ensemble.py`・`post.Stage` にある(走行タブの `ensemble_j`・`ensemble_m`、
**既定オフ、比較の判定には使わない**)。`act` のモジュールは毎フレーム 10 ステップを出すだけ。`act` の予測が `post.replay` を通り、
`ensemble_j = 0` で予測そのもの・`ensemble_j = 1` で前のフレームの予測を 1 行ずらして新しいほど重く混ぜることをテストで確かめた。
チャンクを開ループで実行する(毎フレーム問い合わせない)形は作っていない(30 fps の車では反応が遅れる。design_research.md §3.3)。

## 2. 大きさ・書き出し(CPU で確かめた数字)

| 形 | パラメータ | bench の試作 | 備考 |
|---|---|---|---|
| 既定(Decoder 2層) | **15.57M** | 15.6M(`act_dec2`) | spec §3.1 の 15.6M ± 0.3M |
| Decoder 1層 | 14.78M | 14.8M(`act`) | |
| Decoder 4層 | 17.15M | 17.1M(`act_dec4`) | |
| CVAE つき | 16.65M(+1.08M) | — | +約1.1M は学習のときだけ使う(推論のグラフは同じ大きさ) |

- ONNX(opset 17・`dynamo=False`・バッチ1・固定の形): 入力 `image, tof, imu, state, mode`、出力 `actions [1,10,2]`、`check_io` 合格。
  実データで学習した重みの書き出しは 718 ノードで、演算の種類と数は bench の `act_dec2.onnx`(TensorRT 10.16 FP16 でエンジンになり 1.64 ms)と同じ
  (違いは Concat 12 / 13 と、乱数の重みのときだけ出る Identity)。演算はモデル B の ONNX と同じ組 + Identity だけ。
- **TensorRT のエンジンはまだ作っていない**(GPU を使わない作業のため)。段階4の最初に `tools/build_engine.py` で確かめること。

## 3. 試し学習(spec §7 の最後の確認、CPU・`CUDA_VISIBLE_DEVICES=`)

```
CUDA_VISIBLE_DEVICES= nice python tools/train.py --policy act --data data/datasets/clean_driving_20261011 \
  --out <スクラッチ> --epochs 1 --max-steps 2 --batch 4 --workers 0 --stride 30 --no-command-input --impact-g 2.5
```

- 既定の形: 通った(17 秒)。`jetformer.onnx`・`model_config.json`(`policy: act`・`policy_cfg` の全部のキー・`contract 1`・`inputs` 5つ・
  `output_shapes {'actions': [1,10,2]}`・`engine fp16`・`params_m 15.57`、`postprocess` なし)・`result.json` ができた。
- `--policy-cfg '{"cvae": true}' --seed 0` も通った(`history` に `train_loss`・`train_parts.kl`)。
- どちらも `runtime.load_run` で読み直せ、numpy と Tensor の入力で予測が同じ(差 0)。出力はスクラッチのフォルダーだけ(`data/runs/` には書いていない)。

## 4. テスト(`tests/test_policy_act.py`、CPU)

登録簿の準備・既定の大きさと層の名前(v0 と同じキー)・CVAE の大きさとフック・設定の拒否(dim と heads、管理画面の範囲)・forward の形(学習・eval・`check_forward`)・
ヘッドの並び(ジャンル×2 + [ステア, スロットル])・選んだジャンルの行だけが効く・`expand_modes`(行 2)・CVAE の KL と損失と内訳と勾配・eval で z = 0・
小さなバッチを覚える(CVAE なし・あり。損失が 1/4 未満・誤差 30 µs 未満・左右と前進と並び)・ONNX の約束と演算(CVAE つきでも符号化器・乱数が入らない)・
走行側の重ね合わせ・`tools/train.py` の通し(CVAE つき、合成データ)と `--init` の一致の確認。

学習のテストは速くするため、画像の ResNet-18 と ToF の CNN を同じ形のトークンを出す小さな凍結の層に替え、1 スレッドで回す(トークンより後ろはそのまま学習する)。

ほかに通したもの: `test_policies.py`・`test_policy_ui.py`・`test_policy_post.py`・`test_policy_eval.py`(52 passed)、`test_drive.py`・`test_model_export.py`・
`test_rl_policy_training.py`(59 passed, 2 skipped)。`act` が「準備できた」になっても既存のテストは変わらず通る。

## 5. 見込みと注意(比較の前に)

- 伸びしろは小さい見込み(design_research.md §3.4): v0 との違いは出力の作り方だけ。L1 の差は乱数の種のばらつき(約 10 µs)の中に入る可能性が高い。
  期待できるとすれば遠いステップ(k10)。
- 約 6 千サンプルで Decoder を足すので、過学習が少し増えうる(dropout 0.1・30 エポックで v0 と同じ条件)。
- CVAE は β = 10 で KL が損失の大半を占める。z = 0 の推論は「一番ありそうな型」を出すだけなので、走行での利点は限られる(参考の O2 だけ)。
