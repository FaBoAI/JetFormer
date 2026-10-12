# 4つの新しい方策は Orin Nano に載るか: 試作の ONNX → TensorRT FP16 の計測(速さ・メモリ・精度・学習の時間)

作成 2026-10-12(09:25〜11:20)・ 対象 `/home/jetson/jetformer`(リポジトリのファイルは変更していない)・ 置き場所 `bench/`

**この文書の範囲**: 「ACT 型・TCP/TransFuser++ 型・本家 TransFuser に近い多段融合・DINOv2 の画像特徴」の方策を、**今の入力の形のまま**試作し、
ONNX(opset 17・バッチ1・固定の形)→ TensorRT FP16 のエンジンにして、Orin Nano の上で測った。

- 走らせる前の「載るか・どのくらい重いか」の判断の材料です。**どの方策が良く走るか(ベストか)は、この計測では分かりません。**
  重みは乱数です(下の §1)。良さは、学習して検証の L1・R1・DE で比べる別の作業が決めます。
- 設計の既定の値は `design_research.md`(同じフォルダー)に合わせた。その既定の形(`act_dec2`・`tcp_dec`・`msfusion_rd`・`dinov2_p2`)も、
  ここで直接測っている。
- 車は動かしていない。PCA9685・車両の基板には触れていない。`jetformer-web` は再起動していない。GPU は1つのプロセスずつ使った
  (計測の前に、CUDA を開いている別のプロセスが無いことを確かめた。`bench/common.py` の `gpu_users`)。

---

## 0. 結論(先に要点)

1. **4種類とも、今の道具(`torch.onnx.export(dynamo=False, opset 17)` → `jetformer/trt_runtime.build_engine`、FP16・作業メモリ 1GB)で
   エンジンになり、Orin Nano で 30fps に十分間に合う。** TensorRT の警告は全部の作成で 0 件。入力は今と同じ5つ、出力 `actions [1,10,2]` も同じ。
2. 走行ループと同じ呼び方(`TRTRunner.run_numpy`: 転送・推論・回収)の時間の中央値。v0(モデル B と同じ形)は **2.33 ms**:

   | 方策(設計の既定の形) | エンジンだけ | `run_numpy` | v0 との差 | 学習 30エポック(モデル B の条件) |
   |---|---|---|---|---|
   | `jetformer_v0`(基準) | 1.47 ms | 2.33 ms | — | 17 分(実測 34 秒/エポック) |
   | `act`(Decoder 2層) | 1.64 ms | 2.48 ms | +0.15 ms | 18 分 |
   | `tcp`(Decoder 2層・21 クエリ) | 1.69 ms | 2.50 ms | +0.2 ms | 18 分 |
   | `msfusion`(4段の交換 + トークンの融合) | 2.18 ms | 3.12 ms | +0.8 ms | 26 分 |
   | `dinov2`(ViT-S/14 凍結・2×2 で 64 トークン) | 3.88 ms | 5.06 ms | **+2.7 ms** | 16 分 |

   一番重い `dinov2` でも、撮像から PWM まで(今 約28 ms)に +3 ms(3 m/s で 約9 mm 進む分)。予算の 10 ms の半分。
3. **大きさの選び方**: 今の大きさなら、どの方策も推論は余る。大きくしたときの目安(エンジンだけ / `run_numpy`):
   元の ACT の大きさ(d512・FFN 3200)2.32 / 3.15 ms、TransFuser に近い深さ(各段 4層)3.07 / 3.71 ms、DINOv2 のパッチを減らさない(256 トークン)4.30 / 5.51 ms。
   **どれも 10 ms の予算に入る。** 推論より先に、学習の時間とメモリが効く(`msfusion` の各段 4層は 37 分・2.8 GB)。
4. **学習の1ステップ**(バッチ 64・AMP FP16・AdamW、GPU だけ): v0 310 ms(GPU で 206 枚/秒)。読み込みは 4 プロセスで 約530 枚/秒なので、
   **どの方策も GPU が律速**。`act`・`tcp` は v0 とほぼ同じ、`msfusion` は 1.5 倍、`dinov2`(凍結)は **v0 より速い**(270 ms。ViT は勾配なしの順伝播だけ)。
   ピークのメモリはどれも 2.8 GB 以下(既定の形は 0.5〜1.8 GB)。今の `MIN_FREE_MB = 3500` のままで良い。
5. **FP16 の精度**: TensorRT FP16 と PyTorch FP32(同じ重み)の差は、`actions` で最大 1〜7 µs(乱数の頭)。学習済みのモデル B では最大 0.6 µs。
   DINOv2 の特徴(公式の重み)は、実画像 16 枚・257 トークンでコサイン類似度 **0.9979 以上**(CLS は 0.9999 以上)。FP16 のあふれ・NaN は無い。
   → **合格の判定(例: ≤ 2.5 µs)は、学習した重みでやり直す必要がある**(乱数の頭は出力の大きさが違うので、そのまま比べられない)。
6. **書き出しの落とし穴**(§3。小さなモジュールで確かめた): MHA の高速経路、割り切れない adaptive pooling、antialias の縮小、使っていない入力が消える、
   の4つは **書き出しが失敗する・入力が変わる**。GRU は `nn.GRUCell` なら普通の演算に分かれる(ONNX の GRU は避ける)。
   `CumSum`・`Erf`(GELU)・`LayerNormalization`・SDPA の分解は TensorRT 10.16 で問題なく通った。
7. **TCP の制御器は Python(numpy)に置く**(§2.2)。1回 約30 µs(10ステップ分なら 約0.2 ms)。予測の重ね合わせ(temporal ensembling)も numpy で 1フレーム 約70 µs。
   どちらも推論に比べて小さい。
8. 正直に: 測るたびの揺れは、同じプロセスの中では多くが ±0.02 ms(ときどき 1回だけ 0.1〜0.2 ms 跳ねる)、**別のプロセス・作り直したエンジンでは ±0.2 ms**
   (同じ形の v0 で 1.47〜1.65 ms、モデル B 1.50〜1.52 ms)。`run_numpy` は転送の分さらに揺れる(同じエンジンで最大 0.6 ms)。0.2 ms 以下の差は意味がない。

---

## 1. 計測の条件と方法

| 項目 | 内容 |
|---|---|
| 機械 | Jetson Orin Nano 8GB(スワップなし)、`nvpmodel` MAXN_SUPER、GPU の周波数は計測の前後とも 1020 MHz(最大)、L4T R39.2 |
| 版 | torch 2.10.0+cu130、torchvision 0.25.0、onnx 1.23.2、TensorRT 10.16.1.11(Python 版。trtexec は使っていない) |
| 書き出し | `torch.onnx.export(..., opset_version=17, dynamo=False)`、入力名 `image, tof, imu, state, mode`、出力名 `actions`(+ TCP の追加の出力)、バッチ1・固定の形・`dynamic_axes` なし、`torch.backends.mha.set_fastpath_enabled(False)` |
| エンジン | `jetformer/trt_runtime.build_engine` と同じ設定(FP16・作業メモリ 1GB)。警告を集めるため Logger だけ自前(`bench_infer.py`) |
| 重み | 乱数。ただし **ResNet-18 は ImageNet の重み、BatchNorm の統計は実データ 32 フレームで測り直した**(乱数の BN のままだと活性が層ごとに大きくなり、FP16 の比較にならない)。DINOv2 は公式の重み。同じ重みを `.pth` に保存して、書き出しと比較の両方で使った |
| 精度 | 検証の分割の実フレーム 16 個(`normalize_inputs`、今の操作値なし)で、TensorRT と PyTorch FP32 の出力の差 |
| 速さ | (a) エンジンだけ: 入力を GPU に置いたまま `execute_async_v3` + 同期、(b) `TRTRunner.__call__`(`tools/benchmark_model.py` と同じ。今まで「1.9 ms」と書いてきた数字の測り方)、(c) `run_numpy`(走行ループと同じ)。各 600 回(先に 100 回の空回し)。表の中央値は、全エンジンを1つのプロセスに読み込み、順番を入れ替えて3回測った中央値(`latency_repeat.py`) |
| メモリ | 推論: エンジンの大きさ(≒ 重み)と活性の作業メモリ(`device_memory_size_v2`)。学習: `torch.cuda.max_memory_allocated`(上限 `set_per_process_memory_fraction(0.55)` ≒ 4 GB をかけて測った) |
| 学習 | `tools/train.py` の `run_epoch` と同じ: `autocast(float16)`・GradScaler なし・`clip_grad_norm 1.0`・AdamW(3e-4, wd 1e-4)。データは GPU 上の乱数(読み込みは別に測った)。3回の空回しの後 15 回の中央値。検証は勾配なしの順伝播 |
| 1エポックの見込み | モデル B の実測 34 秒/エポック(学習 5,912・検証 1,559 サンプル、間引き 3)を、GPU の計算時間(学習 93 ステップ + 検証 25 バッチ)の比で伸び縮みさせた。読み込みは律速でない(§1.1) |

### 1.1 読み込みの速さ(CPU だけ、`loader_rate.py`)

`JetFormerDataset`(学習の分割・間引き 3・色の揺らぎ・衝突 2.5G の除外)を 4 プロセス・バッチ 64 で読むと **約530 枚/秒**。
GPU の学習が一番速い `dinov2_r168_p2` でも 374 枚/秒なので、どの方策も GPU が律速。v0 は GPU で 206 枚/秒、実測の学習は 約190 枚/秒で合っている。

---

## 2. 結果

### 2.1 全部の試作(`bench/results/summary_table.md` の要約)

「エンジンだけ」「`run_numpy`」は3回の中央値、p99 はエンジンだけの p99 の最大。誤差は TensorRT FP16 と PyTorch FP32 の `actions` の最大(µs)。

| 名前 | 中身 | パラメータ | エンジン | 作成 | エンジンだけ 中央 / p99 | `run_numpy` | 誤差 最大 | 学習 /64 | ピーク | 30エポック |
|---|---|---|---|---|---|---|---|---|---|---|
| modelB_engine | 今のモデル B(学習済み、既存のエンジンを読むだけ) | 14.1M | 29 MB | — | 1.50 / 1.71 ms | 2.35 ms | **0.6 µs** | — | — | — |
| v0 | JetFormer v0(同じ形・乱数) | 14.1M | 26 MB | 69 秒 | 1.47 / 1.66 | 2.33 | 3.1 | 310 ms | 1.3 GB | 17 分 |
| act | ACT 型・Decoder 1層(元の実装の実質) | 14.8M | 28 MB | 83 秒 | 1.52 / 1.74 | 2.37 | 4.6 | 315 | 1.4 | 18 |
| **act_dec2** | ACT 型・Decoder 2層(**既定**) | 15.6M | 28 MB | 96 秒 | 1.64 / 1.82 | 2.48 | 3.7 | 323 | 1.4 | 18 |
| act_dec4 | ACT 型・Decoder 4層 | 17.1M | 29 MB | 99 秒 | 1.99 / 2.23 | 2.65 | 4.0 | 341 | 1.5 | 18 |
| act_d512 | 元の ACT の大きさ(d512・FFN 3200・Dec 1) | 35.5M | 46 MB | 107 秒 | 2.32 / 2.56 | 3.15 | 4.5 | 442 | 2.2 | 24 |
| act_cvae | ACT 型 + CVAE(学習だけ。推論のグラフは act と同じ) | 16.9M | — | — | — | — | — | 331 | 1.4 | 18 |
| tcp | TCP 型・GRU(`nn.GRUCell` 10 回)+ 目標速度の回帰 + 操作の枝 | 14.2M | 27 MB | 79 秒 | 1.65 / 2.10 | 2.40 | 3.6 | 313 | 1.3 | 17 |
| **tcp_dec** | TCP 型・Decoder 2層・クエリ 21(通過点 10・速度 1・操作 10)(**既定**) | 15.6M | 28 MB | 95 秒 | 1.69 / 2.13 | 2.50 | 5.7 | 334 | 1.4 | 18 |
| msfusion | 多段融合・ToF の段 48x56〜6x7・各段 1層・最後は平均 + MLP(TransFuser どおり) | 19.4M | 40 MB | 153 秒 | 2.00 / 2.19 | 2.63 | 1.1 | 488 | 1.7 | 26 |
| msfusion_l2 | 同じ・各段 2層 | 22.2M | 41 MB | 195 秒 | 2.32 / 2.77 | 2.93 | 2.0 | 537 | 2.0 | 30 |
| msfusion_l4 | 同じ・各段 4層・FFN×4(TransFuser に近い深さ) | 33.4M | 44 MB | 231 秒 | 3.07 / 3.30 | 3.71 | 4.9 | 680 | 2.8 | 37 |
| msfusion_tf | 同じ・最後をトークンの融合 2層 | 20.2M | 41 MB | 170 秒 | 2.13 / 2.34 | 3.07 | 5.2 | 514 | 1.9 | 28 |
| **msfusion_rd** | `design_research.md` §5.2 の形(**既定**) | 18.9M | 36 MB | 210 秒 | 2.18 / 2.37 | 3.12 | 7.2 | 463 | 1.8 | 26 |
| dinov2_vit | DINOv2 ViT-S/14 だけ(公式の重み、出力 [1,257,384]) | 21.6M | 46 MB | 79 秒 | 3.32 / 3.48 | 3.99 | cos ≥ 0.9979 | — | — | — |
| **dinov2_p2** | 凍結 ViT + 2×2 平均で 64 + CLS → 融合 4層(**既定**) | 24.5M(学習 2.9M) | 49 MB | 105 秒 | 3.88 / 4.13 | 5.06 | 4.1 | 270 | 0.5 | 16 |
| dinov2_p1 | 同じ・パッチを減らさない(256 + CLS) | 24.6M(2.9M) | 49 MB | 94 秒 | 4.30 / 4.52 | 5.51 | 2.9 | 376 | 1.0 | 22 |
| dinov2_r168_p2 | 同じ・グラフの中で 224 → 168 に縮めて ViT(12x12)→ 36 + CLS | 24.4M(2.9M) | 49 MB | 106 秒 | 2.73 / 2.91 | 3.93 | 3.6 | 171 | 0.4 | 10 |
| dinov2_p2_ft2 | `dinov2_p2` + 最後の2ブロックと最後の LayerNorm を学習 | 24.5M(6.4M) | — | — | (`dinov2_p2` と同じ) | — | — | 350 | 1.1 | 20 |

補足:

- 推論の活性の作業メモリはどれも 1.5〜3.1 MB。GPU のメモリはほぼエンジンの大きさ(26〜49 MB)+ CUDA の文脈(どの方策も同じ)。
- エンジンの作成は 69〜231 秒。管理画面の作成の打ち切り(`training.py`、20 分)に余裕で入る。ただし作成のプロセスの常駐メモリは最大 3.0〜4.0 GB
  (この計測では PyTorch のモデルと書き出しを同じプロセスで行った分を含む上限)。**8GB・スワップなしなので、学習と作成を同時に動かさない**
  (今の `training.py` は学習の後に別のプロセスで作るので守られている)。
- `__call__`(今までの「1.9 ms」の測り方)は、エンジンだけより 0.3〜0.6 ms 長い。`run_numpy` は画像 600 KB の転送(ピン留めメモリへの写しを含む)の分、さらに 0.1〜0.8 ms 長い。
- `act_cvae` の推論のグラフは `act` と同じ(z = 0 の潜在トークンは `latent_proj` のバイアスだけの定数になる)。CVAE で学習は 5% 長くなるだけ。
- PyTorch(FP16、そのまま)だと 13〜34 ms(v0 15 ms)。**TensorRT にしないと 30fps に間に合わない方策がある**ので、走行は必ずエンジンで。

### 2.2 方策ごとのメモ(配備に関わること)

#### `act`(ACT 型)

- 形: v0 と同じ 94 トークン → Encoder 4層(d256・8ヘッド・FFN 512・前置き LayerNorm)→ 10 個の行動クエリの Decoder → クエリごとに
  `Linear(256, num_modes×2)` → one-hot でジャンルを選ぶ。`nn.TransformerDecoder` はそのまま書き出せる(fastpath を切ること。§3)。
- 推論の差は Decoder の層の数だけ: 1層 +0.05(揺れの中)・2層 +0.15・4層 +0.5 ms。元の ACT の大きさ(d512)でも 2.3 ms。
- 元の ACT の実装は Decoder 7層の設定だが、1層目の出力しか使っていない(LeRobot も既定 1層)。`design_research.md` の既定 2層は +0.15 ms で問題ない。
- `--init-mode-copy`(`expand_mode_state`)は、ヘッドが「1ステップ = 1クエリ」なので行の数が 2(v0 は 20)。`genre_keys` に `rows_per_mode: 2` を持たせる。
  試作では `mode_embed` を `tokens.mode_embed` の下に置いたので、**本実装ではキーの名前を v0 と同じ一番上の `mode_embed`・`heads` にする**。
- 予測の重ね合わせ(temporal ensembling)は方策に依存しない走行側の処理にする(§2.3)。

#### `tcp`(TCP / TransFuser++ 型)

- 既定の形(`tcp_dec`)の出力は `actions [1,10,2]`(操作の枝)・`waypoints [1,10,2]`・`speed_logits [1,10]`。TensorRT は3つとも出し、
  `TRTRunner` はすべての出力をホストへ写すので、走行は追加の出力をそのまま読める。今の走行は `actions` だけを読むので、**`tcp` を知らない走行でも操作の枝で走れる**。
- 通過点の累積和は ONNX の `CumSum` になり、TensorRT 10.16 で通った(誤差は通過点の大きさの 0.7%)。通らない環境のために、下三角の定数行列の掛け算の形
  (`cumsum='matmul'`、結果は同じ。テストあり)も用意した。GRU 版(`tcp`)は `nn.GRUCell` を 10 回呼ぶ形なら普通の演算(Gemm・Sigmoid・Tanh)に分かれて通る。
  `nn.GRU`(層)は ONNX の `GRU` 演算になるので避ける(TensorRT ではループになる。試していない)。
- **制御器は ONNX の外、Python(numpy、torch を読まない)に置く**(`design_research.md` §4.5 と同じ結論)。理由:
  1. 速度の上限・倍率を走行中に変えたい。グラフに入れると入力(`speed_cap`)が増え、エンジンを作り直さずに較正を変えられない。
  2. 走行(`drive.py`)と評価の道具が **同じ関数** を呼べる(torch と numpy の二重実装を作らない)。
  3. 時間は小さい: 試作 `controller_proto.TCPController.step` は 1回 約32 µs、pure pursuit 1回 約23 µs(10ステップ分を別々に計算しても 約0.2 ms、ベクトル化すれば 0.05 ms 以下の見込み)。
  4. 積分(PI の I)は状態を持つので、**既定は使わない**(開ループの評価で積分が飽和し、走行と評価が食い違う)。比例 + 前向き(FF)だけなら、評価をフレームごとにできる。
- 学習のデータでの下調べ(`tcp_oracle.py`、CPU、`results/tcp_oracle.json`):
  - ステア → 曲率: κ = −0.0186 + 0.00343 × (µs − 1441)[1/m]、遅れ 2 フレーム(67 ms)で R² 0.87(ヨーレート ÷ eRPM の速さ、0.8 m/s 以上、学習の分割 18,720 フレーム)。
  - スロットルの定常: 前進の量 = −20 + 77 × 速さ [µs、m/s](R² 0.73、遅れ 3 フレーム)。
  - 推測航法の通過点(1秒先): 前へ 中央値 1.94 m、横 |y| の 90% 点 1.95 m(急なカーブが多い)。シーンの最後の 1秒(約5%)はラベルが作れない。
  - **正解の通過点と正解の 0.5 秒先の速さを、この単純な制御器(10 ステップとも同じ値)に入れても、人の操作との L1 は 57 µs**
    (ステア 62・スロットル 52 µs。検証の分割、衝突 2.5G を除く 1,503 フレーム)。比べる値: 今の人の操作をそのまま 10 ステップ 51 µs、モデル B 49 µs。
    `design_research.md` §4.6 の、ステップごとに道の上の位置から追う形(ステア 34 µs)の方が良い。**スロットルは両方とも 約52 µs で、
    速度の制御から作るスロットルは L1 では不利に見える**。→ TCP の良さは L1 ではなく、速度の誤差・R1/DE・走行で見る(既定は hybrid: ステアは操作の枝)。

#### `msfusion`(本家 TransFuser に近い多段融合)

- 画像は ResNet-18 の各段(56・28・14・7)、ToF は同じチャンネル数の CNN。**ToF を 48×56 に 0 で詰める**と、どの段も画像 7×7・ToF 6×7 の格子で割り切れ、
  固定の `AvgPool2d` と双線形の拡大(`Resize`)だけで書ける。TransFuser の 8×8 の格子を 224 の入力に当てると、28→8・14→8・7→8 が割り切れず **書き出しが失敗する**(§3)。
- 交換の Transformer を足しても推論は軽い(各段 1層で +0.5 ms、既定の形 +0.7 ms)。重いのは **学習**(ResNet と ToF の CNN の高い解像度の活性に、拡大して足す分):
  v0 の 1.5 倍(26 分 / 30エポック)、各段 4層だと 2.2 倍(37 分)・2.8 GB。
- ノードの数が v0 の 2〜6 倍(923〜2,927)。エンジンの作成も 2.5〜4 分と長い。
- 誤差は小さい(1〜7 µs、乱数の頭)。段4(d512)の残差の値が大きくなる心配があったが、FP16 のあふれは見えなかった。学習した重みで確かめ直すこと。

#### `dinov2`(DINOv2 の画像特徴)

- 重み: `data/pretrained/dinov2_vits14_pretrain.pth`(88,283,115 バイト、sha256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`)。
  公式の `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` から 09:26 に取得(`.part` に書いてから名前を変えた)。
  `data/` は `.gitignore` に入っているので git には入らない。コードと重みは Apache-2.0。
- 中身はただの state_dict(175 個、`torch.load(weights_only=True)` で読める)。`pos_embed` は 518 px 用の [1,1370,384](37×37)、`mask_token` は推論で使わない。
- **自前の最小の ViT**(`bench/protos.py` の `DinoViTS14` と部品、約100行)で読める: キーの名前は公式と同じ。`mask_token` を除き、`pos_embed` を読み込むときに一度だけ
  16×16 に補間して、`strict=True` で全キーが一致した。torch.hub のコードには依存しない。公式のソース(`vision_transformer.py`・`block.py`・`attention.py`・
  `layer_scale.py`・`patch_embed.py`・`hub/backbones.py`)を **読んで**(実行はしていない)次を合わせた: LayerNorm の eps 1e-6、qkv にバイアスあり、
  LayerScale(`gamma` を掛ける)、GELU は erf 版、MLP は 4 倍、レジスタなし、位置埋め込みの補間は bicubic・antialias なし・`scale_factor = (16 + 0.1) / 37`。
- 補間をグラフに残さない(読み込みのときに定数にする)ので、ONNX には補間の演算が入らない。ViT だけで 657 ノード(`MatMul` 72・`LayerNormalization` 25・`Softmax` 12・`Erf` 12)。
  注意は `F.scaled_dot_product_attention` で書いたが、書き出しで MatMul + Softmax に分かれ、TensorRT はそのまま通した(明示的な `q·kᵀ` でも同じ形になる)。
- **ViT だけで 3.3 ms**(224 px・257 トークン)。文献の ViT-S/16 2.36 ms(TRT 10.3・224 px。パッチ 16 なので 196 トークン)より 1 ms 重い(トークンが 1.3 倍)。
  パッチを 2×2 で減らした既定は全体で 3.9 ms、減らさないと 4.3 ms(融合が 303 トークンになる分 +0.4 ms)。
  さらに軽くしたいなら、グラフの中で 168 px に縮める(`dinov2_r168_p2`、2.7 ms)。ただし細い物(テープ・レールの端)が粗くなる。縮小は antialias なしの双線形(§3)。
- FP16: 特徴のコサイン類似度 ≥ 0.9979(トークン単位の最悪)、平均 0.99992、CLS ≥ 0.9999。最大の差は 0.85(値の最大 18.3 に対して 4.7%)で、
  PyTorch の autocast(0.17)より大きい(TensorRT は LayerNorm・Softmax も FP16 で動かすためと思われる。層ごとの精度は調べていない)。運転に効く大きさではない見込みだが、
  気になるなら LayerNorm の層だけ FP32 に固定する(`OBEY_PRECISION_CONSTRAINTS`。試していない)。BF16・FP32 のエンジンは試していない。
- 学習: 凍結の ViT は勾配なしの順伝播だけなので、**学習の1ステップは v0 より速く(270 対 310 ms)、メモリは 0.5 GB**。検証(勾配なしの順伝播)は逆に
  v0 の 2 倍(221 対 103 ms/バッチ)。特徴のキャッシュは要らない(色の揺らぎもそのまま使える)。最後の2ブロックを学習すると 350 ms・1.1 GB。
- `best.pth` に ViT の 21.6M(FP32 で 約87 MB)が毎回入る。**評価の道具や作り直し(`load_run`)では、DINOv2 の重みのファイルを読まずに作れるようにする**
  (`pretrained=False` のときは読まない)。走行はエンジンだけで動くので、重みのファイルは要らない。

### 2.3 走行側の追加の処理の時間(CPU、`controller_proto.py`)

| 処理 | 1回の時間 | 置き場所 |
|---|---|---|
| TCP の制御器(純追従 + FF/P、10ステップとも同じ値) | 約32 µs | `drive.py` の `pred = res['actions'][0] ...` の直後、先読みの前 |
| 同じ、ステップごとに道の上から追う形(pure pursuit 10 回) | 約0.23 ms(未ベクトル化) | 同上 |
| 予測の重ね合わせ(`TemporalEnsembler.push`、過去 10 フレーム) | 約68 µs | 同上(どの方策にも使える。既定オフ) |

予測の重ね合わせは、出力を [10,2] のまま返すので、先読み・ゲイン・ToF ガード・補助/リカバリーはそのまま効く。フレームが抜けても撮像時刻で数える。
重みは w = exp(−m·j)(j = 0 が一番新しい予測 = 新しいほど重い。`design_research.md` §2.2 の J = 3・m = 0.5 と同じ向き)。元の ACT は古い予測ほど重く(m = 0.01 でほぼ平均)、30fps の車では遅れが増えるので、そのままは勧めない。

---

## 3. 書き出し(ONNX → TensorRT)の落とし穴

小さなモジュールで確かめた(`export_pitfalls.py` → `results/export_pitfalls.json`、torch 2.10・opset 17・`dynamo=False`)。

| 項目 | 結果 | 対策 |
|---|---|---|
| `nn.TransformerEncoderLayer` / `nn.MultiheadAttention` を、高速経路(fastpath)を切らずに eval・`no_grad` で書き出す | **失敗**(`aten::_transformer_encoder_layer_fwd`・`aten::_native_multi_head_attention` は opset 17 に無い) | 書き出しの前に `torch.backends.mha.set_fastpath_enabled(False)`(今の `export_onnx` は済み)。`TransformerDecoder` の中の MHA も同じ |
| `F.adaptive_avg_pool2d` で割り切れない大きさ(28→8、7→8) | **失敗**(`output size that are not factor of input size`) | 段の大きさは固定なので、窓の大きさを計算して `F.avg_pool2d`。割り切れるよう入力を詰める(ToF 42×54 → 48×56) |
| `F.interpolate(..., antialias=True)` の縮小 | **失敗**(`aten::_upsample_bilinear2d_aa` は opset 17 に無い。opset 18 から) | antialias なしの双線形(`Resize`)。位置埋め込みの補間は読み込みのときに済ませる |
| 使っていない入力 | **グラフから消える**(`dinov2_vit` は入力が `image` だけになった) | 走行は知っている入力を全部渡す(`TRTRunner` はエンジンの入力だけを読む)。`model_config.json` の `inputs` は ONNX から読み直す(`interfaces.md` §4) |
| `nn.GRUCell` | 通る(Gemm・Sigmoid・Tanh・Split に分かれる) | — |
| `nn.GRU`(層) | ONNX の `GRU` 演算になる | 避ける(GRUCell を回すか、Decoder のクエリにする) |
| GELU(erf 版)・LayerNorm・SDPA | `Erf`・`LayerNormalization`(opset 17 の正式な演算)・MatMul + Softmax に分かれる。TensorRT の警告なし | — |
| `CumSum` | 通る(TensorRT 10.16) | 心配なら下三角の定数行列の掛け算 |
| 辞書の出力 | 名前は `output_names` の順で付く | ラッパーで決まった順のタプルにして `output_names` を渡す(`bench_infer.py`) |
| 形 | バッチ1・固定の形。`x.shape[0]` は Shape・Gather の演算として残るが、固定の形なら問題ない。`TRTRunner` は `-1` の形を扱えない | `dynamic_axes` を使わない。CVAE の z のように推論で変わらないものは定数にする |
| 書き出しの警告 | `TracerWarning`(テンソルを bool にする・テンソルを回す)は `nn.Transformer*` の中の確認と、ViT の q・k・v の分け方から。固定の形なので無害 | — |
| 書き出しの方式 | torch 2.10 の既定は `dynamo=True`。今の道具は `dynamo=False`(古い方式、DeprecationWarning が出る) | 今は `dynamo=False` のまま。torch を上げるときに移行を検討(試していない) |
| 計測の落とし穴(自分が踏んだ) | 書き出しと比較で別々にモデルを作ると、乱数の重みが違って「誤差が大きい」と見える。BN の統計が乱数のままだと活性が大きくなる | 重みを `.pth` に保存して両方で使う。ImageNet の ResNet + 実データで BN を測り直す |
| 既存のエンジンを読むときの警告 | モデル B のエンジンで `Using an engine plan file across different models of devices` が出た(自分で作ったエンジンでは出ない) | 速さ・出力は正常(誤差 0.6 µs)。気になるならこの機械で作り直す |

---

## 4. 実装への申し送り(登録簿の `hints` の目安)

`interfaces.md` §5 の `PolicySpec.hints`(管理画面の学習の時間の見込み `TRAIN_RATE`・`VAL_RATE`、空きメモリ `min_free_mb`)に入れる値。
v0 の実測(学習 200・検証 650 枚/秒)を、GPU の計算時間の比で直したもの。

| 方策(既定の形) | `train_rate` | `val_rate` | `min_free_mb` | 推論(`run_numpy`)の目安 |
|---|---|---|---|---|
| `jetformer_v0` | 200 | 650 | 3500 | 2.3 ms |
| `act` | 190 | 620 | 3500 | 2.5 ms |
| `tcp` | 185 | 600 | 3500 | 2.5 ms(+ 制御器 0.2 ms) |
| `msfusion` | 135 | 430 | 3500 | 3.1 ms |
| `dinov2` | 230 | 300 | 3500 | 5.1 ms |

- どれも学習のピークは 2 GB 以下(既定の形)なので、`MIN_FREE_MB = 3500` のままで足りる(`design_research.md` の `msfusion` 4000 は安全側)。
- 比較の学習(モデル B と同じ条件・30エポック)の GPU の時間の目安: 1本あたり v0 17・act 18・tcp 18・msfusion 26・dinov2 16 分。
  5種類 × 3 つの種なら 約4.7 時間(GPU は1つずつ)。電源が落ちても `--resume` で続きから。
- 本実装で確かめること:
  1. 学習した重みで、TensorRT FP16 と PyTorch FP32 の `actions` の差(合格 ≤ 2.5 µs など)。乱数の頭の 1〜7 µs は目安にならない。
  2. `run_numpy` の時間を走行の記録(`infer_ms`)で確かめる(走行ループ全体は今 5.6 ms。`dinov2` で 約8.3 ms の見込み)。
  3. `tcp` の制御器は、ステップごとに道の上から追う形で、学習した通過点(正解ではなく)で `T_la`・`L_min` を決める。

---

## 5. ファイル(`/home/jetson/jetformer_reports/policy_zoo_20261012/bench/`)

| ファイル | 中身 |
|---|---|
| `protos.py` | 試作の方策(`VARIANTS` の 17 種)。`DinoViTS14`(公式の重みを読む自前の ViT)・`GRUCellExport`・`StageFusion` |
| `bench_infer.py` | 1つの試作: 書き出し(CPU)→ エンジン → 精度 → 速さ → `results/<名前>_infer.json`(段ごとに書き、済んだ段は飛ばす) |
| `bench_train.py` | 学習の1ステップの時間とメモリ → `results/<名前>_train.json` |
| `run_all.sh` | 全部を1つずつ順に(GPU は1プロセスずつ・使えるメモリが 3 GB 未満なら待つ)。何度でも続きから実行できる |
| `latency_repeat.py` | 全エンジンを1つのプロセスで順番を入れ替えて3回 → `results/latency_repeat.json` |
| `export_pitfalls.py` | 書き出しの落とし穴の確認(CPU)→ `results/export_pitfalls.json` |
| `controller_proto.py` | TCP の制御器(純追従 + FF/P、速度の上限)と予測の重ね合わせの試作(numpy だけ) |
| `tcp_oracle.py` | 較正の当てはめ・推測航法の通過点・正解を入れたときの制御器の L1・CPU の時間 → `results/tcp_oracle.json` |
| `loader_rate.py` | 学習データの読み込みの速さ → `results/loader_rate_w4.json` |
| `summarize.py` | 表 → `results/summary.json`・`results/summary_table.md` |
| `common.py` | 結果の JSON(一時ファイル → 置き換え)・GPU を使っている別のプロセスの確認・実データの入力 |
| `test_bench_protos.py` | CPU だけのテスト 12 件(11.5 秒): GRUCell の一致、ジャンルの選択、出力の約束、CVAE の z=0、格子の割り切れ、DINOv2 の読み込み、純追従、制御器の符号の約束と速度の上限、重ね合わせ、累積和の2つの形 |
| `work/` | ONNX・`.pth`・エンジン(合計 約2.4 GB。消してよい。`run_all.sh` で作り直せる) |
| `logs/` | 各計測のログ |

再現: `cd bench && bash run_all.sh && python latency_repeat.py && python summarize.py`(GPU を使う。ほかの GPU のプロセスが無いときに)。
テスト: `cd bench && CUDA_VISIBLE_DEVICES= python -m pytest -q test_bench_protos.py`。
