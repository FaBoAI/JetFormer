# 準備の段 — 腕の学習データ・確かめ・学習の道具(sim_diverse_train_20261015)

- 2026-10-15 09:18〜09:40(CPU だけ。GPU・Firefox は使っていない)。決まりは `protocol.json`(sha256 `f78fade8…`、書き換えていない)。
- この段で行ったこと: 3 つの腕(S-DIV・S-MOD・C-DIV)の学習データのフォルダーを作り、protocol.json の数と同じかを確かめた。
  学習の道具(10/14 の `dart_train.py` の写しを直したもの)と一式の道具を写して直し、3 つの腕で `tools/train.py` を CPU で短く動かした。
- 実車・PCA9685・I2C には触れていない。実車を走らせる試験は、この作業の範囲の外です。
- 確かめの結果: **全部合格**(`prepare_check.json` の `all_ok: true`)。手順と違えたことはありません(`deviations.md`)。

---

## 1. 要点

| 腕 | 中身 | シーン(学習 / 検証) | サンプル(間引き 3、学習 / 検証) | protocol の見込み | ジャンル | 確かめ |
|---|---|---|---|---|---|---|
| **S-DIV** | 5 モードの見本 75 + DART(D1)30 + 余裕の線(D2)30 | 108 / 27 | **20,847 / 5,203** | 20,847 / 5,203 | 8 だけ(ジャンル数 9) | 合格 |
| **S-MOD** | 5 モードの見本 75 | 60 / 15 | **11,391 / 2,839** | 11,391 / 2,839 | 8 だけ(ジャンル数 9) | 合格 |
| **C-DIV** | 実機 52 + 5 モードの学習の 12 シーンずつ(60) | 101 / 11 | **17,303 / 1,559** | 17,303 / 1,559 | 4・5・8(ジャンル数 9) | 合格 |

- 3 つとも、読み込んだサンプルの (シーン, フレーム) の表が、元のデータセットを 1 つずつ読んだ表をつなげたものとまったく同じでした。
- **離脱復活の乱した操作のフレームは、どの腕にも 1 つも入っていません**。
  - 外したフレームを frames.jsonl から数え直すと 1,623 で、取り込みの印と同じでした。
  - 学習に入る全サンプルの正解の窓(次の 10 フレーム)にも、乱した操作は入っていません。
- **C-DIV の実機の検証 1,559 サンプルは、A・C の検証とまったく同じ (シーン, フレーム) です。**
  - 実機の学習 5,912 サンプルも、A と同じです。衝突の前後として外したフレーム(学習 3,522・検証 1,083)も同じでした。
- PWM の約束の違い(5 モード 1441 / 1496 µs、ほか 1442 / 1498 µs)は、舵 ≤ 1.0 µs・スロットル ≤ 2.0 µs でした。protocol の範囲の中なので、付け替えはしていません。
- CPU の動作確認は、3 つの腕とも合格しました。S-DIV・C-DIV では電源断のまね(kill -9)をし、`--resume` で続きから学習できることも確かめました。
- 画像は 1 枚も写していません。3 つのフォルダーはシンボリックリンクと JSON だけで、合わせて 4.3 MB です。ディスクの空きは 9.07 GB です。

---

## 2. 腕のフォルダー

場所は `/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train/data/<腕>/` です。data/datasets には新しいデータセットを作っていません。

- 中身:
  - `aligned/<シーン>.npz` と `samples/{CAM_FRONT,TOF_FRONT}/<シーン>`: 元のデータセットへのシンボリックリンク(絶対パス)
  - `splits.json`: 元の分けを、引数の順につなげたもの
  - `dataset_card.json`: ジャンルの表と、元の一覧
  - `combined.json`: 作ったときの記録。元の splits.json・dataset_card.json の sha256 も入っている
- 作り方: `build_arms.sh` → `make_combined.py`。10/14 の make_combined.py の写しで、次の 5 つを変えた。
  1. ジャンルの表は、id 0〜8 がそろっていることだけを確かめる(名前の書き方が「駐車1」と「駐車(1)」で違うため。protocol のとおり)。
  2. 元ごとに入れる分けを選べる(C-DIV のシミュレーターは学習の 12 シーンだけ)。
  3. `--modes-from` で使うジャンルの表を指定できる(C-DIV は simreal_mix_20261013 と同じ 0〜8 の表)。
  4. `<出力>.tmp` に作ってから名前を替える(電源断で半端なフォルダーを残さない)。
  5. カードのフレーム数は、入れたシーンから数え直す。

### 2.1 S-DIV(Q1 の主の腕)

| 元 | 学習 | 検証 |
|---|---:|---:|
| sim-rl-fastest_20261014-1125(最速) | 2,364 | 591 |
| sim-rl-speed_20261014-1125(速さいろいろ) | 2,364 | 591 |
| sim-rl-no_wall_20261015-0302(絶対壁にぶつからない) | 2,364 | 591 |
| sim-rl-recover_20261015-0302(離脱復活) | 1,935 | 475 |
| sim-rl-line_20261015-0302(経路いろいろ) | 2,364 | 591 |
| D1 jetformer-simulator-rl-dart-fast-10min_…_4446c73d(DART、付け替え済み) | 4,728 | 1,182 |
| D2 jetformer-simulator-rl-margin-line-10min_…_48d04e6b(余裕の線) | 4,728 | 1,182 |
| **合計** | **20,847** | **5,203** |

- 学習サンプルの割合: 5 モード 0.546・D1 0.227・D2 0.227(protocol と同じ)。重みは付けていない。
- 全部で 135 シーン・81,000 フレーム。ジャンル 0 のフレーム 1,623(離脱復活の外したフレーム)は学習に入らない。
- ジャンルの表は 1 つ目(最速)のもの。D1・D2 の表とは id 1〜3 の名前の書き方だけが違う(`combined.json` の mode_name_differences)。

### 2.2 S-MOD(Q1 の 2 つ目の腕)

- 5 モードの 75 シーン(学習 60 / 検証 15)。学習 11,391・検証 2,839 サンプル。モードごとの学習は [2,364, 2,364, 2,364, 1,935, 2,364] で、protocol と同じです。

### 2.3 C-DIV(Q2 の腕)

| 確かめ | 結果 |
|---|---|
| 実機の検証のサンプル | 1,559。A(clean_driving_20261011 を同じ設定で読んだもの)・C(simreal_mix_20261013 の検証)と (シーン, フレーム) がまったく同じ。A の 2 本・C の 2 本の result.json も 1,559 |
| 実機の学習のサンプル | 5,912。A の学習と (シーン, フレーム) が同じ。ジャンル 4 が 2,228、5 が 3,684 |
| 衝突の前後として外したフレーム | 学習 3,522・検証 1,083(A と同じ。シミュレーターは 0) |
| 実機の aligned | 52 シーンとも、リンク先が clean_driving_20261011 のファイルそのもの(同じ i ノード)。画像のフォルダーも同じ |
| シミュレーター | 5 モードの学習の 12 シーンずつ・モードの順 = 60 シーン・36,000 フレーム。学習サンプル 11,391。検証には 0 |
| 5 モードの検証 15 シーン | C-DIV に入っていない(シミュレーターの保留の参考に使える) |
| ジャンルの表 | simreal_mix_20261013 と同じ(0〜8、ジャンル数 9) |
| action_reference(止まっている間の操作の中央値) | **[1441.0, 1496.0]**(A・C と同じ) |

- C-DIV のフレーム数は 63,502 で、C(simreal_mix_20261013)と同じです(実機 27,502 + シミュレーター 36,000)。
- 学習サンプルは C の 17,732 より 429 少ない 17,303 です。離脱復活の外したフレームのぶんで、protocol の見込みと同じです。

---

## 3. 元のデータセットの確かめ

| データセット | フレーム(カードと同じ) | ジャンル(aligned) | 状態 |
|---|---|---|---|
| 5 モード(各 15 シーン × 600) | 9,000 ずつ | 8(離脱復活だけ 0 が 1,623・8 が 7,377) | 5 つとも rl.state = accepted |
| D1(30 × 600) | 18,000 | 8 | 付け替え jf-sim-dart-relabel/v1 |
| D2(30 × 600) | 18,000 | 8 | — |
| clean_driving_20261011(52 シーン、232〜598 フレーム) | 27,502 | 4(10,181)・5(17,321) | — |

- no_wall・recover の 1025・1125(合格前の方策)は、どの腕にも入れていません。
- **離脱復活の外すフレーム**: 15 シーンの frames.jsonl の `action_info.jf.l` から数え直しました。l = 0 の操作 j について、観測 k ∈ [j − 10, j] を外す決まりです。
  - 数え直した結果は 1,623 フレームで、全シーンで aligned のジャンル 0 のフレームとまったく同じでした。どれも future_valid = False でした。
  - 腕のサンプル(S-DIV・S-MOD の学習 1,935 + 検証 475、C-DIV の学習 1,935)について、観測 k と正解の窓 k + 1〜k + 10 を調べました。乱した操作は 1 つも入っていませんでした。

### 3.1 PWM の約束

各データセットの学習の最初の 1 シーンで調べました。frames.jsonl の操作(舵 rad・速さ m/s)から、PWM を 2 つの約束で作り直しました。換算は区分線形で、中心から両端の 1000 / 2000 µs までです。

| データセット | 自分の約束 | 作り直し − 記録の PWM | 2 つの約束の差(舵 / スロットル) | aligned の正解 = 記録の PWM(1 フレーム先から) |
|---|---|---|---|---|
| 5 モード(5 つ) | 1441 / 1496 | 0.0 µs | 1.0 / 2.0 µs | 一致(≤ 0.0001 µs) |
| D1(DART) | 1442 / 1498 | 0.0 µs | 1.0 / 2.0 µs | 付け替え済みなので対象外 |
| D2(余裕の線) | 1442 / 1498 | 0.0 µs | 1.0 / 2.0 µs | 一致 |

- 差の最大(舵 1.0 µs・スロットル 2.0 µs)は、止まっているとき(速さ 0)に出ます。protocol の範囲(≤ 1 / ≤ 2 µs)の中で、付け替えの目安の 5 µs より小さいので、そのまま使います。
- シミュレーターの腕の action_reference は S-DIV で [1399.0, 1188.5] になります。止まっている間の操作の中央値ですが、シミュレーターでは「止まっている」フレームが走り出しだけなので、こうなります。
  - ブリッジ用の一式では、10/14 と同じく中心を 1442 / 1498 µs に直します(`export_bundle.py`)。学習が出した値は model_config の simulator_bundle に残します。

---

## 4. 学習の道具

場所は `/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train/` です。

| ファイル | 中身 | 元からの変更 |
|---|---|---|
| `diverse_train.py` | 学習の道具(train・status・dryrun・bundle・smoke) | 10/14 の dart_train.py の写し。腕・名前・パス、学習データの確かめ(prepare_check.json と sha256)、workers の決まり、A4 の DAgger(`train/a4_plan.json`)、smoke を足した |
| `nothp.py` | THP を切ってからコマンドに置き換わる | 10/14 のまま |
| `make_combined.py`・`build_arms.sh` | 腕のフォルダーを作る | 2 章 |
| `prepare_check.py`・`finalize_prepare.py` | この段の確かめ → `prepare_check.json` | 新しく作った |
| `export_bundle.py`・`check_bundle.py`・`http_load_check.py` | ブリッジ用の一式と、その確かめ | 10/14 の写し。パス(今日のレポート・今日の `sim/`)・作業名・待ち受けの口(8824・8825)だけを変えた |

**学習のコマンド**(`diverse_train.py dryrun` の出力。protocol の作り方と同じ):

```
/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 choom -n 1000 -- \
  /home/jetson/miniconda/envs/jetracer/bin/python -I .../train/nothp.py /home/jetson/miniconda/envs/jetracer/bin/python -u \
  tools/train.py --policy jetformer_v0 --data experiments/20261015_sim_diverse_train/train/data/S-DIV --epochs 30 --lr 3e-4 \
  --out data/runs/simdiv_20261015-SDIV-s0 --seed 0 --stride 3 --batch 64 --workers 1 --no-command-input --impact-g 2.5
```

- 腕と workers:
  - S-MOD は S-DIV と同じ形(workers 1)。
  - C-DIV は `--data .../C-DIV`・`--workers 2`。種は 0・1。
  - A4(DAgger)は `--epochs 10 --lr 1e-4 --init <元の腕の best.pth>`。
- メモリの工夫(10/14 と同じ):
  - expandable_segments・MALLOC_ARENA_MAX 2・THP なし・choom 1000。
  - 全体の空きメモリの見張り: 200 MB 未満が 8 秒続いたら自分の学習だけ止め、空くのを待って続きから。
  - 自分の上限(3G)で止まったら workers を下げる(Q1 の腕は 1 → 0、C-DIV は 2 → 1)。
  - C-DIV がさらに 0 に下がるときと、続きからの学習を上限 3400M にするときは、protocol の外です。道具が自分で `deviations.md` に書きます。
- 電源断への備え:
  - 毎エポック last.pth を保存する。同じコマンドで `--resume` して続きから学習する。last.pth が壊れていたら退避して最初から。
  - 状態は `train/state.json` に、書き終えてから置き換える。
  - 出来た学習結果(result.json の state = done)は飛ばす。
- 一式(Q1 の腕だけ): 学習のあとに `export_bundle.py` → `check_bundle.py` を動かす。
  - 今日のアプリの写し(`sim/`)がまだ無いときは飛ばし、あとで `diverse_train.py bundle S-DIV/s0` で作る。
  - C-DIV では一式も FP16 のエンジンも作らない(protocol の no_engine)。

### 4.1 使い方(次の段)

```
cd ~/jetformer && nohup /home/jetson/miniconda/envs/jetracer/bin/python -u \
  experiments/20261015_sim_diverse_train/train/diverse_train.py train S-DIV/s0 \
  >> /home/jetson/jetformer_reports/sim_diverse_train_20261015/train/logs/driver.log 2>&1 &
```

- 既定の順番は S-DIV/s0 だけです。時間割では、S-DIV の学習のあとに Firefox の評価(lock big)が入ります。
  - そのため C-DIV は、評価が済んでから `train C-DIV/s0 C-DIV/s1` で始めます。
  - S-MOD は `train S-MOD/s0` で別に始めます。
- 電源断のあとは、同じコマンドをもう一度動かします。

---

## 5. CPU の動作確認(smoke)

本番と同じ `tools/train.py` の引数から、次の 4 つだけを変えて動かしました。

- `--epochs 2`・`--batch 8`・`--workers 0` に替え、`--max-steps 2` を足した。
- CUDA を見えなくして(CPU)、`jf-heavy --mem 1500M --lock none` で動かした。
- 出力は `smoke/runs/`(data/runs には作らない)。終わったら消して、結果を `smoke/<腕>.json` とログに残した。

| 腕 | 試し方 | 結果 |
|---|---|---|
| S-DIV | 1 エポック目の last.pth のあとで kill -9(24 秒)→ `--resume`(26 秒) | 「再開: 1 エポック目まで済み」から 2 エポック目を学習。result.json state done・2 エポック、ONNX 56.5 MB、model_config のジャンル数 9・学習したジャンル [8] |
| S-MOD | そのまま(53 秒) | state done・2 エポック、ジャンル数 9・[8] |
| C-DIV | kill -9(21 秒)→ `--resume`(34 秒) | state done・2 エポック、ジャンル数 9・[4, 5, 8]、衝突の除外は学習 3,522 / 検証 1,083、中心 1441 / 1496 |

- 3 つとも、ImageNet の重み(キャッシュ済み)の読み込み・学習・検証・last.pth・続きからの学習・ONNX の書き出しまで通りました。
- GPU での 1 エポックの秒数とメモリは、本番で測ります。見込みは S-DIV 約 125 秒・C-DIV 約 90 秒・S-MOD 約 68 秒です。

---

## 6. 次の段に残したこと(protocol の build_2。S-DIV の学習の間に CPU で行う)

この段では、学習の道具と一式の道具までを準備しました。次のものは、まだ写していません。

1. **アプリの写し**: `sim_dart_20261014/sim` から runtime/datasets(2.5 GB)を除いて、`sim_diverse_train_20261015/sim` に写す。
   - dist の app.js・sim-core.js・index.html・rc-speed-controller.js・bridge.py が、ユーザーのものと 10/14 の写しの両方と同じか確かめる。
   - 写しがあれば、学習の道具が一式を自動で作ります。無ければ、あとで `diverse_train.py bundle S-DIV/s0` で作ります。
2. **閉ループの評価の道具**: 10/14 の `eval/`(browser_eval.py・run_eval.sh・verify_eval.mjs・analyze_eval.py・teacher_eval.mjs など)と `review/` を写す。
   - 口とプロファイルを替える: ブリッジ 8821・geckodriver 4450・`ff_profile`。
   - 計画の E0〜E11・陽性対照・A1 の E0 を入れる。
   - 写した一式は `http_load_check.py`(口 8825)で確かめる。
3. **Q2 の評価の道具**: `tools/sim_cotrain_eval.py`・`sim_cotrain.py`・`sim_cotrain_figures.py` を `experiments/20261015_sim_diverse_train/q2/` に写す。主な直しは次のとおり。
   - 保存先を data/zoo/sim_diverse_train_20261015 と今日のレポートにする。10/13 の場所に書かないようにする。
   - 名前の表を A・B(10/12)・C(10/13 の npz)・C-DIV にする。C+FT・A+FT は外す。
   - 判定は C-DIV 対 A と、C-DIV 対 C にする。C との比べの閾値は best 0.954 µs・last 1.322 µs。
   - 種 1 つの暫定の判定、エンジンのゲートなし(no_engine)、5 モードの検証 15 シーンの頭 8 の L1、低速の直線 HS の前進の偏りを足す。
   - crosscheck は C の種 0 の best で行う(≤ 0.01 µs)。主指標を数え直す別の短い道具も作る。
4. 10/14 の `offline_eval.py`・`train_report.py`(オフラインの誤差・IMU の位相ごとの誤差・図)を写して、S-DIV・S-MOD を足す(15:15 ごろに使う)。
5. DAgger(Q1b)の道具: 差し込みモジュール・付け替えの写し・D0〜D11 の先生の確かめ。`diverse_train.py` の A4 は `train/a4_plan.json`({ok, data, init})を読みます。
   - A4 の学習の道具は動作確認していません(データが無いため)。

---

## 7. ディスク・プロセス

- ディスクの空きは 9.07 GB です(始める前 9.1 GB)。この段で増えたのは 4.5 MB です。
- 一時のもの(smoke の学習結果・`.tmp`)は消しました。
- 動かしたプロセスは全部終わっています。学習・ブリッジ・Firefox は動かしていません。

## 8. ファイル

- `prepare_check.json`: 確かめの全部(`all_ok`・元のデータセット・PWM・離脱復活・腕ごと・C-DIV・smoke・道具の sha256・学習のコマンド)
- `smoke/<腕>.json`・`smoke/logs/<腕>.smoke.log`: CPU の動作確認
- `train/logs/prepare_check.log`: 確かめのログ
- 腕のフォルダー: `/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train/data/{S-DIV,S-MOD,C-DIV}/`
