# `msfusion`(多段融合・TransFuser 型)の実装メモ(担当 P-msf)

作成 2026-10-12 ・ CPU だけ(GPU は使っていない)・ commit はしていない(M がレビュー)

## 作ったもの

| ファイル | 中身 |
|---|---|
| `jetformer/policies/msfusion.py` | 方策の本体(`POLICY_STUB` の行は消した → 登録簿の `ready()` が True、学習タブ・`tools/train.py` で選べる) |
| `tests/test_policy_msfusion.py` | CPU のテスト 24 件(GPU・実データ・ネットなし。ImageNet の重みも読まない) |

共有のファイル(登録簿・runtime・train.py など)は変えていない。登録簿の項目(既定値・選択肢・hints)は F が書いたまま使った。

## 形(spec.md §3.3 のとおり)

- 画像: ResNet-18 の `img_stem`(conv1・bn1・relu・maxpool)と `img_stages`(layer1〜4、`pretrained` で ImageNet)。
- ToF: [距離, 信頼度, 有効の印] を `F.pad(左右 1・上下 3)` で 48×56 に 0(無効)で詰める → `tof.stem` 32ch → 段1〜4
  (64×24×28・128×12×14・256×6×7・512×6×7。段4 は stride 1 + 1×1 で 512ch)。
- 交換 `fuse['1'〜'4']`(`scales` にある段だけ): 固定の `F.avg_pool2d` で画像 7×7・ToF 6×7 の 91 トークン + 段ごとの `pos` →
  Encoder(4 ヘッド。段1〜3 は 2層・FFN×4、段4 は 1層・FFN×2)→ LayerNorm → 元の格子 → 双線形(`align_corners=False`)で拡大して各枝に足す。
- 最後(`head='tokens'`、既定): 4段目の画像 49・ToF 42 を `img_proj`・`tof_proj`(512→256)、IMU・操作値・ジャンル・出力トークン → `pos [1,95,256]`
  → `fusion`(Encoder 2層・8 ヘッド・FFN 512)→ `norm` → 出力トークン → `trunk` → `heads`(1ジャンル 20 行、v0 と同じ並び)。
- `head='pooled'`(選択肢、TransFuser どおり): `global_proj`(両枝の4段目の平均の和 512→256)と IMU・操作値・ジャンルをつなぐ → `trunk`(MLP 2層)→ `heads`。
- 損失は既定の L1、正解も v0 と同じ(この方策はラベルを作らない)。出力は `actions [B,10,2]` だけ。`USES_TARGET`・`policy_loss`・`prepare`・`finalize` なし。
- 任意の `embed(...)` を足した(`tools/offline_rl.py features` 用。tokens は融合の後の出力トークン [B,256] = v0 の `model.norm` の出力と同じ位置、pooled は trunk の出力)。

## 確かめたこと(CPU)

| 項目 | 結果 |
|---|---|
| 大きさ(ジャンル 6) | **18,919,320**(18.92M、全部学習する)。spec の 18.9M ± 0.5M の中 |
| 試作との一致 | `bench/protos.py` の `MSFusionRDProto`(速さを実測した既定の形)と state_dict の形が全部同じ(`fuse` の番号だけ 0〜3 → '1'〜'4')。同じ重みを入れると出力の差 **0.0** |
| テスト | `tests/test_policy_msfusion.py` 24 件合格。**21.7 秒**(CPU が空いているとき。ほかの作業と重なったときは 33〜43 秒) |
| 関係するテスト | `test_policies.py`・`test_policy_ui.py`・`test_policy_eval.py` 41 件合格(msfusion が実装済みになっても通る) |
| CPU の試し学習 | spec §7 のコマンド(`--max-steps 2 --batch 4 --workers 0 --stride 30 --no-command-input --impact-g 2.5`、出力はスクラッチ)が 14〜16 秒で終了。`jetformer.onnx`(入力 5 つ・`actions [1,10,2]`、opset 17、1,710 ノード)・`model_config.json`(§2.5 の欄: policy・policy_cfg・contract・inputs・input_shapes・outputs・output_shapes・engine fp16・params_m 18.92)・`result.json`。`runtime.load_run` で読み直せて、`policies.drivable` と `check_io` も合格。確かめた後に消した(`data/runs/` には書いていない) |
| ONNX の中身(小さい設定) | `AveragePool` 5 個(画像 8・4・2、ToF 4・2、窓 = 歩幅、詰め物なし)、`Resize` 5 個(linear・half_pixel・antialias なし)、`Adaptive*`・`ATen`・`GlobalAveragePool`・`Loop`・`If`・`GRU` なし。pooled は `ReduceMean` |
| 混合精度 | モデル全体を CPU の bfloat16 の autocast で順伝播・逆伝播(一度だけ。8 秒かかるのでテストからは外し、交換の1段だけをテストに残した)→ 型の混ざり(Conv の低い精度 + LayerNorm の FP32 の足し戻し・連結)で止まらない |

テストの中身: 登録簿の項目と ready、既定の大きさ・層の名前・各段の層数/ヘッド/FFN、`check_forward`、詰めた後の各段の格子が割り切れる(画像 7×7・ToF 6×7)、
ToF の詰め物が向きを変えない(元の [r, c] → [r+3, c+1]、左の無効ゾーンは左のまま)、交換の出力が元の格子の同じ場所に戻る(左右・上下の反転、画像と ToF の取り違えなし)、
`scales` を減らすと交換の層が減る、不正な設定は ResNet を作る前に ValueError、`head='pooled'`・`embed`、ジャンルの行の選択と `expand_modes`(6 → 8、`--init-mode-copy 6:5`)、
1ステップで勾配が全部の層に届く、小さなバッチで損失が下がり符号の決まり(ステア > 中心 1443 = 左、スロットル < 中立 = 前進)を覚える、ONNX の書き出し(tokens・pooled)、
`load_run` の往復、`pretrained=False` は重みを読まない・`pretrained=True` は ImageNet の重みを正しい層に入れる(読み込みは偽物に差し替え)。

## spec との違い・補足

1. **ONNX のノード数**: spec §3.3 の「実測 951〜1619」は既定以外の試作の値。既定の形の試作 `msfusion_rd` は **1,786**(`bench/logs/infer_msfusion_rd.log`)、
   この実装は **1,710**(計算は同じ。上の「試作との一致」)。v0(488)の約 3.5 倍。エンジンの作成の目安は試作の実測 210 秒のまま。
2. **`scales` の選択肢は CLI でも守られる**: 登録簿の規則(§2.1.1 の 4)で、`options` のキーは `--policy-cfg` でも `choices`([1,2,3,4] / [2,3,4] / [3,4])の外を断る。
   `[1,3]` や `[]`(交換なし = 遅い融合だけ)は、モジュールの `build` を直接呼ぶときだけ作れる(テストで使う)。比較の本体は既定なので影響なし。
3. `embed` は spec では任意。足しただけで、学習・書き出し・走行には関係しない。

## まだ確かめていないこと(GPU が要る。段階3・4 の M)

- TensorRT のエンジンの作成・FP16 の誤差(ゲート: 実フレーム 64 個で ≤ 0.005 = 2.5 µs)・`run_numpy` の速さ。参考は試作の実測(乱数の重み):
  エンジンだけ 2.18 / 2.37 ms(中央 / p99)、`run_numpy` 3.12 ms、学習 1 ステップ(64)463 ms、学習のピーク 1.8 GB、30 エポック 約26 分、エンジンの作成 約210 秒。
  段4 の交換(d512)の残差で FP16 があふれる心配は、学習した重みで確かめ直すこと(design_research.md §5.3)。
  design_research.md §5.3 の「エンジンが 4 ms を超えたら段4 の交換を外す(`scales: [1,2,3]`)」は、今の登録簿の選択肢に無い
  (使うなら M が `__init__.py` の choices に足す。モジュールは [1,2,3] も作れる)。
- GPU の AMP(FP16)での実際の学習(CPU の bfloat16 では通った)。

## 気づいたこと(比較の本体では変えない)

- 交換の出力は LayerNorm の後(1 くらいの大きさ)で、ImageNet で学習した ResNet の各段の特徴にそのまま足される。学習の始めは事前学習の特徴が少し乱れる
  (本家 TransFuser・試作と同じ作り)。将来の候補: 交換の LayerNorm の重みを 0 で始める(足し戻しが 0 から始まる)。結果を見てから変えない(§8)ので、この作業では入れていない。
- ToF は鳥瞰図ではなく、カメラとほぼ同じ向きの距離画像なので、早い段の交換で新しく分かることは少ないかもしれない(spec §9)。L1 で v0 と差が出なくても想定どおり。
