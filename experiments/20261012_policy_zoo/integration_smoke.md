# 方策の比較: 段階3(結合・GPU の試し)の記録

作成 2026-10-12 ・ 対象 `/home/jetson/jetformer`(main 7239fea の作業ツリー。**commit はしていない**)・ jetformer-web は再起動していない

やったこと: 実装者の依頼(`shared_changes_needed`)の取り込み → 全テスト → 5つの方策を **1本ずつ・1プロセスずつ** GPU で試した
(試し学習 → ONNX → TensorRT FP16 → TensorRT と PyTorch FP32 の差 → 走行と同じ手順での読み込みと推論 → 速さ)。
車は動かしていない。PCA9685・車両の基板には触れていない(走行のループは偽の PCA9685 で回した)。

試しの重みは **5 ステップだけ学習したもの**。精度の数字(差・L1)は配線の確認で、学習した重みの値ではない。速さは重みに依存しない。

## 1. 実装者の依頼の取り込み

| 依頼 | 扱い |
|---|---|
| P-msf: `scales` の選択肢に `[1, 2, 3]`(4段目の交換なし。design_research §5.3 の「4 ms を超えたら」の代わり) | **入れた**(`jetformer/policies/__init__.py`。テスト `test_fewer_scales_mean_fewer_exchange_layers` に足した) |
| P-dino: README に DINOv2 の重みの取得の手順 | **入れた**(README「モデルの種類(方策)」の下。`.part` → sha256 → mv) |
| P-tcp: spec §3.2.4 の pure pursuit を「頂点の向きの補間」に、§3.2.3 のスロットルの係数の例を直す、端の区切り・fallback の扱い | **spec.md を直した**(§0.4 に変更の表。正解の通過点で 56 / 52 µs 対 弦の向き 70 / 65 µs で、実装の方が良い) |
| P-msf: spec §3.3 の ONNX のノード数 | **spec.md を直した**(既定 1,710) |
| F: GPU で `build_engine.py` を一度動かす(TensorRT での `engine_io`・走行の `check_io`) | **済み**(下の §3。5つとも終了コード 0) |
| P-act: act のエンジンを一度作る | **済み**(FP16、差 1.32 µs) |
| P-tcp・P-dino: FP16 のゲートを確かめ、落ちたら bf16 → fp32 | **確かめた**(試しの重みでは全部合格)。ただし **bf16 は代わりにならない** と分かった(§5) |
| P-tcp・P-dino(任意): `build_engine.py` に「一部の層だけ FP32」 | **入れていない**。実験だけした(§5)。試しの重みでは改善が小さく、必要になったら足す |
| P-tcp: jetformer-web の再起動(学習タブに tcp・走行タブの TCP の欄) | **していない**(この作業では再起動しない約束)。M がレビューの後に |
| F・P-*: commit(`tests/fixtures/v0_reference.npz`・`vit.py` など新しいファイルを含む) | **していない**(M) |
| P-act が見た `test_policy_tcp.py` の失敗 | P-tcp が直していた(全テストで通る) |

## 2. テスト(CPU)

- 全体: **359 passed, 2 skipped**(取り込みの前も後も同じ。約 3 分、CPU)。
- 方策のテストの1ファイルの時間(静かな CPU、pytest の表示): msfusion 23.9 秒・act 23.7 秒・tcp 16.9 秒・dinov2 10.7 秒・policies 16.2 秒。
  どれも 30 秒未満だが、msfusion・act は余裕が少ない(ほかの作業と CPU を分け合うと 30 秒を超える)。

## 3. GPU の試し(5つの方策、1本ずつ)

手順(`smoke/run_smoke.py`。出来たものは飛ばすので途中で止まっても続きから):
`tools/train.py --policy P --epochs 1 --max-steps 5 --batch 64 --workers 4 --stride 3 --no-command-input --impact-g 2.5 --seed 0`
(clean_driving_20261011、AMP FP16)→ `tools/build_engine.py`(別のプロセス)→ `smoke/engine_check.py`(別のプロセス)。
学習結果は `data/runs/_smoke/` に置き(走行・学習の一覧に出ない)、**終わってから全部消した**(`data/runs/` は前と同じ 18 本)。

| | jetformer_v0 | act | tcp | msfusion | dinov2 |
|---|---|---|---|---|---|
| パラメータ(全体 / 学習) | 14.08M | 15.57M | 15.59M | 18.92M | 24.50M / 2.87M |
| 学習のピークのメモリ(バッチ 64・AMP) | 1,274 MB | 1,332 MB | 1,347 MB | 1,741 MB | 475 MB |
| ONNX のノード / 大きさ | 427 / 54 MB | 718 / 60 MB | 746 / 60 MB | 1,710 / 72 MB | 1,054 / 94 MB |
| エンジンの作成(build_engine の表示)/ 大きさ | 72 秒 / 28 MB | 98 秒 / 31 MB | 102 秒 / 31 MB | 207 秒 / 38 MB | 109 秒 / 50 MB |
| `check_io`(TensorRT で読み直し・走行の手順) | 合格 | 合格 | 合格(出力3つ) | 合格 | 合格 |
| TensorRT FP16 と PyTorch FP32 の差 actions 最大(平均)[µs] | 0.68(0.15) | 1.32(0.29) | 1.28(0.31) | 1.37(0.24) | 1.43(0.23) |
| ゲート ≤ 2.5 µs | 合格 | 合格 | 合格 | 合格 | 合格 |
| **エンジンだけ 中央 / p99 [ms]** | **1.48 / 1.67** | **1.67 / 1.86** | **1.72 / 2.01** | **2.17 / 2.33** | **3.95 / 4.15** |
| **`run_numpy`(走行の呼び方)中央 / p99 [ms]** | **2.38 / 2.61** | **2.60 / 2.90** | **2.61 / 2.95** | **3.11 / 3.38** | **5.12 / 5.46** |
| 速さのゲート(中央 ≤ 6.0・p99 ≤ 8.0) | 合格 | 合格 | 合格 | 合格 | 合格 |
| 走行のループ(45 フレーム × 場合の数) | 2 場合 全部走行 | 2 場合 全部走行 | 7 場合 全部走行 | 2 場合 全部走行 | 2 場合 全部走行 |

- 差は検証のシーンの実フレーム 64 個(等間隔、2.5G の衝突の前後を除く)。PyTorch FP32 は CPU(TF32 が入らないように)。
- tcp の追加の出力: `waypoints` の差 最大 0.0116 m(ゲート 0.02 m、合格)、目標速度の期待値の差 最大 0.0005 m/s(ゲート 0.05 m/s、合格)。
  ただし試しの重みでは通過点の大きさが最大 5.8 m(学習前のでたらめな値)なので、学習した重みでもう一度確かめること。
- 速さは 100 回の空回しの後 600 回。同じ v0 のエンジンを2回測ると エンジンだけ 1.63 → 1.48 ms・`run_numpy` 2.46 → 2.38 ms
  (0.15 ms 程度の揺れ)。bench の値(1.47 / 1.64 / 1.69 / 2.18 / 3.88 ms)とほぼ同じ。

### 3.1 走行と同じ手順の確かめ(`smoke/engine_check.py` §2・§5)

`drive.model_info`(`policy`・`drivable`・`outputs`・`postprocess` の kind)→ `TRTRunner` → `policies.check_io` → `runner.shapes` から作った
入力で空回し → **本物の `Driver._loop`** を本物のエンジンで回した。記録したシーン(検証の `20261010-135449-0003`、ジャンル 5、
前進中)の連続 45 フレームの JPEG・ToF の生の値・IMU の履歴・車両の値を 30fps で流し、PCA9685 と記録は偽物、音声の通知先なし、
走行の設定ファイルは一時ファイル。

- 5つとも、どの場合も **45 フレーム全部で推論・出力(中立で止めた回数 0、例外なし、`model_pred` は有限)**。
- ループが記録した `model_pred` と、同じフレームを同じ手順(JPEG の再圧縮・`vehicle_state`・`normalize_inputs`)で作って
  `run_numpy` した値の差: **全部 0.0 µs**(10 フレームずつ)。
- 後処理のない方策・`ensemble_j = 0`: `pred` = `model_pred`(ビット単位で同じ)。`ensemble_j = 2`: 重ね合わせが効く(`model_pred` との差 最大 15〜60 µs。試しの重みの予測が揺れるため)。
- tcp(`postprocess.kind = tcp_controller`、走行タブの注記「後処理: TCP の制御器」):
  - `tcp_mode 0`・上限なし: `pred` = `actions` とビット単位で同じ(`branch = control`)。
  - `tcp_mode 1 / 2 / 3`: `branch` が `hybrid` / `fusion:ctrl` / `trajectory`。
  - `speed_cap 2.0`(v_now 2.64 m/s): `v_cmd` 2.0、スロットル 1646 µs = 中立 1496 + 150(spec のブレーキの規則。1 m/s を切ると止める)。
  - `v_hat` の中央値 2.51 m/s。試しの重みなので通過点を追うステア(mode 3)は 2000 µs に張り付く(学習前のでたらめな通過点のため)。
- 注意: このループの中の `infer_ms`(約 7 ms)・`latency_ms`(約 55 ms)・fps(約 14)は、同じプロセスで流す側のスレッドと取り合うため
  実車の値ではない。前の版の `drive.py`(HEAD)を同じ道具で回しても同じ値(fps 14.0・latency 58 ms)だったので、今回の変更による
  遅れではない。速さはループの外の `run_numpy` の数字(上の表)で見る。

## 4. 追加の試し(GPU、1本ずつ)

| 試し | 結果 |
|---|---|
| `act` + CVAE(`{"cvae": true}`、AMP、5 ステップ)→ エンジン → 走行の手順 | 学習・書き出し・エンジン・ループとも合格。16.65M(CVAE の符号化器は学習だけ)、ピーク 1,366 MB。`train_parts.kl` 1.93・`train_loss` 20.2(KL の重み 10 が初めは大きい。P-act の見込みどおり)。差 1.21 µs、エンジン 1.66 / 1.88 ms、`run_numpy` 2.73 / 2.96 ms |
| `dinov2` の LP-FT(`{"finetune_last": 2}`、`--init` = dinov2 の試しの best.pth) | 合格。`--init` の構造の確認を通り、学習するのは 6.42M、ピーク 1,057 MB |
| `tcp` の電源断からの再開(`--epochs 2 --max-steps 3`、1 エポック目の後に `kill -9` → 同じ引数 + `--resume`) | 合格。「1 エポック目まで済み … 2 エポック目から」→ 2 エポック目を学習して書き出し(`postprocess` つき) |

## 5. エンジンの精度の代わり(dinov2 の試しの重みで、1本ずつ)

spec §3.4 は「FP16 で差のゲートに落ちたら `engine.precision` を bf16 → fp32」としていた。同じ重みで作り直して測った:

| 精度 | 差 actions 最大(平均)[µs] | エンジンだけ 中央 / p99 [ms] | `run_numpy` 中央 / p99 [ms] | ゲート |
|---|---|---|---|---|
| fp16(既定) | 1.43(0.23) | 3.95 / 4.15 | 5.12 / 5.46 | 合格 |
| **bf16** | **4.31(0.79)** | 3.95 / 4.17 | 5.10 / 5.28 | **差で不合格** |
| fp32 | 0.18(0.04) | **8.54 / 8.93** | **9.46 / 9.77** | **速さで不合格** |
| fp16 + LayerNorm・Softmax だけ fp32(実験。repo には入れていない) | 1.32(0.21) | 4.08 / 4.35 | 5.05 / 5.30 | 合格 |

- **BF16 は FP16 より仮数が短い(7 ビット対 10 ビット)ので、差のゲートの代わりにならない**(差が 3 倍になった)。BF16 が役に立つのは
  FP16 であふれる(NaN・Inf)ときだけ。fp32 は正確だが dinov2 では速さのゲート(6 ms)に落ちる。
- つまり学習した dinov2 が FP16 の差のゲートに落ちると、spec の手順のままでは走行の候補に残る道がない。候補は「一部の層だけ FP32」
  (`smoke/build_mixed_experiment.py`: FP16 + `OBEY_PRECISION_CONSTRAINTS` で正規化・Softmax の 51 層を FP32。+0.13 ms)。
  この重みは FP16 でも合格なので改善は 1.43 → 1.32 µs と小さく、落ちる重みで効くかは確かめていない。必要になったら
  `build_engine.py` の精度の選択肢に足す(spec.md §3.4 に注記した)。
- bf16 の作成は 103 秒、fp32 は 25 秒。

## 6. 気づいたこと(段階4へ)

1. **`tools/policy_zoo.py`(E)と `tests/test_policy_zoo.py` はまだ無い**。`zoo_rules.json` も未作成。段階4の比較の道具はこれから。
2. tcp の学習の最初は通過点の損失が大きい(試しで `wp_m` 4.4〜5.6 m → 6 ステップで 1.9 m。`0.5·wp` が L1 の 3〜4 倍)。
   勾配のクリップ(1.0)は全体に掛かるので、最初のうち操作の枝の学習が遅れるかもしれない。重みは先に固定した値(spec §3.2.1)なので変えない。
   30 エポックの学習曲線で、操作の枝の L1 が v0 に遅れないか見ること。
3. 学習の時間の見込み: 5 ステップの試しでは測れない(起動・データの読み込み 15〜20 秒が主)。最初の v0 種 0 の 1 本で spec §4.2 の見込みと比べる。
4. 試しのエンジンの作成は bench の見込みどおり(msfusion が一番長く 207 秒)。作成のプロセスのメモリの上限は今回は問題なし(空き 4.3〜4.7 GB で開始)。
5. 走行タブの TCP の欄・学習タブの方策の選択は、jetformer-web を再起動するまで画面に出ない。

## 7. ファイル

| ファイル | 中身 |
|---|---|
| `smoke/run_smoke.py` | 5つの方策の試しの手順(1本ずつ・出来たものは飛ばす。GPU を使う別のプロセスがあれば待つ) |
| `smoke/engine_check.py` | 差・速さ・走行の手順の確かめ(1プロセス) |
| `smoke/extra_smoke.sh`・`smoke/precision_smoke.sh`・`smoke/build_mixed_experiment.py` | §4・§5 の試し |
| `smoke/summary.json`・`smoke/check_<方策>.json`・`smoke/check_act_cvae.json`・`smoke/check_dinov2_{bf16,fp32,mixed_ln_softmax_fp32}.json` | 結果 |
| `smoke/logs/` | 各段のログ |

repo で変えたもの(この段階): `jetformer/policies/__init__.py`(msfusion の `scales` の選択肢)・`tests/test_policy_msfusion.py`・`README.md`(DINOv2 の重み)。
このフォルダーの `spec.md`(§0.4・§3.2.3・§3.2.4・§3.2.5・§3.3・§3.4)。
