# 記録の段 — RL のベストラップの走りを、JetFormer のシミュレーターデータにした(sim_rl_bestlap_20261013)

- 作成 2026-10-14 03:30(JST)。この段の仕事は design.md 4 章の「記録」です。強化学習(RL)で選んだ方策で、ユーザーの CourseLab シミュレーター(WebGL)を
  走らせ、アプリ自身の収集器で 20 分のデータを作り、検証して、JetFormer に取り込みました。
- 依頼(ユーザー、2026-10-13): 「シミュレーターは強化学習でベストラップの走行を実現したいです。それをシミュレーターデータとして学習させたい」
- 実車・PCA9685・I2C には触れていません。GPU も使っていません。ユーザーのシミュレーター(`/home/jetson/sim/extracted/course-simulator`)・`jetformer/web`・
  `jetformer/sim*`・`data/sim*`・`data/modes.json` は変えていません。`jetformer-web` は再起動していません。git にはコミットしていません。

---

## 1. 結論

| 項目 | 結果 |
|---|---|
| データセットの名前 | **`jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0`** |
| 置き場所 | `/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0`(シーンのタブ用は `data/raw/ds-jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0`) |
| ジャンル・タグ | ジャンル **8(シミュレーター)**、num_modes 9。タグ **`simulator` + `rl-bestlap`**。新しいジャンルは作っていません(4 章) |
| 大きさ | 60 シーン(train 48 / val 12、baseline 30 / ToF 25 mm の雑音 30)、**36,000 フレーム = 20.0 分**、学習に使えるフレーム 35,400 |
| 容量 | 集約 ZIP 679,060,327 バイト(647.6 MiB、sha256 `8942243f…`)。取り込んだデータセット 約 0.91 GB + シーンのタブ用 約 0.18 GB |
| 運転した方策 | 選んだ方策(main 種 1・10,010,624 ステップ、policy.json の sha256 `7c43cb53…`)。制御遅延 0・速さの上限 4.5 m/s(手順どおり) |
| 周回時間(60 シーン) | **1 周目(停止から)平均 8.092 秒**(8.040〜8.122)、**2 周目(フライング)平均 7.731 秒**(7.651〜7.849) |
| RL の確認済みのラップとの比べ | 停止から 1 周 8.092 秒(標準の開始)・フライング 7.656 秒。記録の 1 周目の平均はほぼ同じ、2 周目は開始の小さなずれで 7.65〜7.85 秒に広がる(Node の予想の平均 7.737 秒と同じ程度) |
| 教師の 20 分データとの比べ | 同じ 60 の開始で、教師は 1 周目 8.743 秒・2 周目 8.375 秒。**RL は 1 周あたり 0.65 秒・0.64 秒速い** |
| 衝突・周 | 全 60 シーンで **衝突 0**・20 秒走り切り・**2 周ずつ(計 120 周)すべて有効**(33 ゲートを順番に・黒シートの必須横断 1 回) |
| 検証 | ユーザーの検証器と同じやり方の 60 シーン検証に **全部合格**(3 章)。ZIP の形式の検査も合格(PWM の再計算の差 0 µs) |
| 画面 | `screens/` に記録中のシミュレーター(窓・車載カメラ・ToF)6 組と、JetFormer のデータセットタブ・シーンタブ(5 章) |

手順と違ったことは 3 つあります(詳しくは 6 章と `deviations.md` の 2026-10-14 03:20 の項目)。

1. **ブラウザの走りは、Node の予想と 1 本ずつは一致しませんでした。** Firefox と Node の計算の最後の桁(約 1e-14)の違いを、方策と物理の閉ループが大きくするためです。
   1 本ずつの周回時間の差は中央値 0.004 秒・最大 0.14 秒です。その代わり、記録した 1 本ずつについて、記録した操作の再生と方策の再計算が一致することを確かめました。
2. **集約 ZIP を作り直しました。** アプリが自動でまとめたときの名前が、教師データと同じ `jetformer-simulator-id8-20min` になったためです。中身のシーンは同じです。
3. **教師の印 `not_learned_policy` を false にしました。** RL の方策は学習した方策だからです。ユーザーの検証器は、この印の 1 行だけを RL 用に変えた写しで使いました。

---

## 2. やり方

| 項目 | 中身 |
|---|---|
| 差し込み | design.md 4.1 節の方法 A。コピーした dist に `rl.html` を作り、import map で `./rc-speed-controller.js` を `./rl/rl-teacher-bestlap.js` に置き換えた。`app.js`・`sim-core.js`・収集器・集約はユーザーのものそのまま(コピーの dist はユーザーの dist と 33 ファイルともバイト単位で同じことを確かめた) |
| 差し込みモジュール | `experiments/…/web/rl-teacher-bestlap.js`。RL の学習で使った `features.mjs`(79 次元の観測)・`policy_forward.mjs`・重み `policy_weights.mjs` を dist/rl/ に写して読む(`record/install_bestlap.py` が sha256 を確かめ、`bestlap_manifest.mjs` に書く)。同じフレームで 2 回呼ばれたら同じ値を返し、フレームが飛んだら止まる(記録では 36,000 回の計算・飛び 0) |
| 開始・物理の約束 | 元の RC 教師の検査(`assertRcTeacherContract`)をそのまま使用: v3 コース・33 ゲート・RC 応答の物理(時定数 0.40 秒・加速の上限 6・ブレーキ 1.2 m/s²)・30 Hz・雑音 0 / 25 mm・連続周回 20 秒・ゲート 0 で左右 ±3 cm・向き ±3° |
| 60 シーンの計画 | ユーザーの `qa/rc-collection-plan-60.json`(教師の 20 分データの計画)をそのまま写し、名前だけ `rlbest20-mode8-{train,val}-NN` に変えた(`recording/plan_rlbest60.json`)。開始・種・baseline / noise の交互・train / val は教師データと同じ |
| ブラウザ | Firefox 157(snap)を DISPLAY :0 の窓で開く(ヘッドレスでは WebGL 2 が使えないため。WebGL 2 は Mesa llvmpipe)。geckodriver 4445、専用プロファイル `sim/ff_profile_rl` |
| ブリッジ | コピーの bridge.py をポート 8791 で起動。PWM の換算は教師の 20 分データと同じ 1442 / 1498 µs(`calibration_only_policy.py`) |
| 操作 | `browser_collect.py` が「20分・60シーンの収集条件を用意」でジャンル 8・自動ラベルにしてから、計画を差し替えて「連続試験を開始」を押す |
| メモリの約束 | `jf-heavy --mem 2500M --lock big` の中で動かし、snap の Firefox のスコープにも MemoryMax 2500M を付けた(付いたことを確かめた)。始める前にほかの Firefox が無いこと・空きメモリ 2 GB 以上を確かめた。記録中の空きメモリの最小 2,272 MB |
| 速さ | 7.1 フレーム/秒、60 シーンで 84 分(01:17〜02:42)。見込みの 10 フレーム/秒より遅いのは、別の作業の RL の学習・検証と CPU を分け合ったため(負荷平均 約 11) |

記録の前に、同じ 60 条件を Node(元の `sim-core.js`・同じ差し込みモジュール)で走らせて予想を作りました(`recording/prediction.json`)。
予想は RL の段の最終評価(`rl/final/final_eval_main.json` の REC60)と、60 シーンすべてで周回時間が **差 0 秒**、最小すき間も同じでした。

---

## 3. 検証(design.md 4.5 節)

ユーザーの `qa/verify-rc-collection.py`(60 シーンの検証)と同じ組み立てで、RC 教師専用の部分だけを RL 用にした
`record/verify_rl_collection.py` を使いました。結果は `recording/verify/collection-integrity.json`(シーンごとは `recording/verify/<セッション ID>/`)です。

| 確かめたこと | 結果 |
|---|---|
| 集約 ZIP の形式(`check_collection_zip.py`) | 合格。マニフェストの sha256 が 72,377 ファイルすべてで一致、NPZ のキー・型・形が教師の 20 分データと同じ、ジャンル 8・num_modes 9、未来ラベル・最後の 10 フレームの除外・合成の印が正しい。データカードの項目は教師データと同じ(違いは JetFormer の取り込みが足す `scene_log`・`tags` だけ) |
| PWM のラベル | 記録した指令から収集器と同じ式で計算し直した PWM と、保存した値の差 **0 µs**(中心 1442・中立 1498)。舵 1000〜2000 µs、スロットル 1188.5〜1498 µs、CH3 1500 |
| 計画・設定との一致(60 シーン) | 種・場面・開始姿勢・雑音・20 秒・衝突で停止・カメラ / ToF の取り付け・センサーの設定・train / val が計画どおり。物理・センサー・教師の約束は 60 シーンで 1 種類ずつ。開始は 60 通り |
| 記録した操作の再生(元の sim-core.js) | 全フレームで軌跡が一致(最大の差 1.3e-13)。自前の積分でも一致(差 0) |
| 方策の再計算 | ブラウザが記録した状態から方策を計算し直すと、記録の操作と **最大 2.1e-14** で一致(60 シーン・36,000 フレーム)。Node で再生した状態から計算した場合は 6.2 節 |
| 周回・ゲート・黒シート・衝突(自前の数え直し) | 衝突 0、ゲートの外し 0、120 周すべて 33 ゲート・黒シートの有効な横断 1 回。最小すき間(車体の長方形と全障害物)**5.65 cm**、ゲートの幅の余裕の最小 2.9 cm |
| センサー(ユーザーの `verify-rc-teacher-dataset.py` の写し) | 全 60 シーン合格: マニフェスト、画像の時刻と姿勢の対応、JPEG 224×224・ToF 54×42、ToF の PNG と NPZ の完全一致、動きから作り直した 100 Hz の IMU と一致(差 0)、PWM・eRPM・未来ラベル(1〜10 フレーム先) |
| 画像がすべて違うか | 1 シーンの中で RGB 600 / 600 枚・ToF 600 / 600 枚がすべて違う。データ全体でもカメラ 36,000 枚・ToF 36,000 枚の sha256 がすべて違う |
| 画像の点検 | RGB の平均 108.7(教師データ 108.8)。ToF の当たりの割合 95.65%(教師データ 95.70%、±10% の条件を満たす)。最初のシーンと最後のシーンの車載カメラ・ToF を目で見た(`screens/`) |
| 周回時間と予想 | 1 周目の平均 8.0920 秒(予想 8.0910)、2 周目 7.7311 秒(予想 7.7366)。1 本ずつの差は中央値 0.004 秒・最大 0.143 秒(6.1 節) |

---

## 4. ジャンルと取り込み

- **ジャンルは 8(シミュレーター)のまま**にしました(design.md 4.7 節の決定)。新しいジャンルは作っていないので、`data/modes.json` は変えていません
  (取り込みの前後で sha256 `f40f64bc…` が同じ)。理由は 3 つです。
  - シミュレーターの推論アダプターは ID 0〜8(num_modes ≤ 9)しか受け付けません。
  - シミュレーターの記録画面のジャンルの選択肢にも 9 がありません。
  - 9 を足すと、実車の収集・走行画面のジャンルの一覧が変わり、以後のモデルがすべて num_modes 10 になります。
- 走り方の区別は **タグ `rl-bestlap`** で付け、学習ではモデル(学習の腕)を分けます(S-T は教師データだけ、S-R はこのデータだけ)。
- 取り込みは `jetformer.dataset_import.import_archive(zip, simulator=True, to_scenes=True)` のあと、`jetformer.dataset.set_dataset_tags(name, ['simulator', 'rl-bestlap'])`
  (`record/import_dataset.py`、jetracer 環境)。シーンのタブに 60 シーンを展開しました(`ds-jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0-0001`〜`-0060`)。
- 取り込みの注意は 1 件だけで、教師データのときと同じです: 「ジャンルの名前がこの Jetson と違います(ID は同じものとして学習します): 6=未使用(予約), 7=未使用(予約)」。
- JetFormer の画面の API(`/api/datasets`)でも、教師データと並んで `simulator`・`rl-bestlap` のタグ付きで出ることを確かめました。
- 取り込みの記録は `rl_dataset.json` です。次の段の S-R の学習は、ここに書いた名前を使います。

---

## 5. 画面(`screens/`)

| ファイル | 中身 |
|---|---|
| `00_t00030_f227_{window,camera,tof}.png` | 1 本目(train-00)の 7.5 秒目。窓・車載カメラ(224×224)・ToF の表示 |
| `01_t00948_f86_*`〜`05_t04601_f503_*` | 13・24・34・44・55 本目の途中(約 15 分ごと) |
| `dataset_last_scene_f300_rgb_tof.png` | 取り込んだデータの最後のシーン(val-59)の 300 フレーム目。保存した RGB(JPEG)と ToF(距離を色にしたもの。色は自分で付けた) |
| `jetformer_datasets_tab.png` | JetFormer のデータセットタブ。60 シーン(学習 48 / 検証 12)・36,000 フレーム・20.0 分・タグ `rl-bestlap`・シミュレーター |
| `jetformer_scenes_tab.png` | JetFormer のシーンタブ。取り込んだシーン(ジャンル 8: シミュレーター・20 秒) |

窓のスクリーンショットは Firefox の窓が小さい(576×359)ため、ページの上の部分(コースの俯瞰)だけです。図は次の 2 つです。

- `fig/record_laps.png`: シーンごとの周回時間。RL の記録(ブラウザ)・記録前の予想(Node)・教師の 20 分データ(同じ開始)。
- `fig/record_trajectories.png`: 記録した 60 本の軌跡(上から)と、教師の標準の開始の軌跡。

---

## 6. 手順と違ったこと・分かったこと

### 6.1 ブラウザの走りは、Node の予想と 1 本ずつは一致しない(閉ループが小さな差を大きくする)

design.md 4.2 節は「周回の集計が予想と一致」を確かめる予定でした。1 本目の記録で、周回時間が予想と違うことに気づきました
(train-00: 記録 8.0925 / 7.7565 秒、予想 8.0911 / 7.6563 秒)。`record/divergence_probe.mjs` で 60 シーンを調べた結果です(`recording/divergence.json`)。

| 見たこと | 値 |
|---|---|
| 操作の差が 1e-15 を超える最初のフレーム | 9〜(中央値 16.5) |
| 操作の差が 1e-9 / 1e-3 を超える最初のフレーム | 中央値 42 / 159 |
| 20 秒の中の位置の差 | 中央値 0.21 m・最大 0.61 m |
| 周回時間の差(ブラウザ − Node) | 中央値 0.004 秒・最大 0.143 秒 |
| Node の中だけで、開始の x を 1e-12 m・1e-9 m ずらしたとき | 周回時間が中央値 0.015 秒・最大 0.178 秒変わる(120 本すべて衝突 0・2 周) |

- 差は最後の桁から始まります。はじめは約 12 フレームで 1,000 倍ずつ大きくなり(1e-15 → 1e-9 が中央値 26 フレーム)、その後はゆっくりになって、
  中央値 159 フレーム目で 1e-3 を超えます。出発点は Firefox(SpiderMonkey)と Node(V8)の計算の最後の桁の違いです
  (記録した操作を Node で再生しただけでも、状態に 1.3e-13 の差が出ます)。
- Node の中だけでも同じ大きさで変わるので、これは差し込みモジュールの誤りではありません。**この方策と物理の閉ループの性質**です。
  予想は RL の段の評価と完全に一致しています。
- そこで、合否は 1 本ずつの「再生の一致」「方策の再計算の一致」「衝突 0・全周が有効」で決めました。予想とは周回時間の分布で比べました(3 章)。
- データとしての意味: 60 本は、同じ決定的な方策がブラウザで走った正しい走りです。ラベルは方策の出力そのものです。
  ただし、同じ条件で Node で作り直しても、同じ 60 本にはなりません。

### 6.2 方策の再計算は「記録した状態から」で判定した

1 回目の検証(`recording/verify_run1_teacher_from_node_replay/`)は、汎用の `verify_recording.mjs --teacher` で行いました。これは Node で再生した状態から方策を計算します。
58 / 60 シーンは差 1.5e-12 以下でした。2 シーンだけは 1 フレームで、速さの指令が 0.0296 m/s 違いました(train-37 の 318 フレーム目、val-54 の 549 フレーム目)。

- `record/teacher_mismatch_probe.mjs` で調べました(`recording/mismatch/*.json`)。再生した状態とブラウザの状態の差は 2.5e-14・3.4e-14 です。
  その差で、観測の「参照ループに対する向きのずれ」(heading_sin・heading_cos)が 0.036 跳んでいました。参照ループの頂点で、射影する線分が入れ替わったためです。
- ブラウザが記録した状態から計算すると、2 シーンとも全フレームで差 2.7e-15 以下です。60 シーン全体でも 2.1e-14 以下です。
- 2 回目の検証では、合否を「記録した状態から」の値にし、Node で再生した状態からの値は書くだけにしました。
- この跳びは観測(`features.mjs`)の性質で、学習の環境でも同じように起きます。データの誤りではありません。

### 6.3 集約 ZIP の名前

- アプリは 60 条件がすべてきれいに終わると、その場で 1 つの ZIP にまとめます。そのときの名前は、画面の既定値 `jetformer-simulator-id8-20min` でした。
  これは教師の 20 分データと同じ名前です。
- 取り違えを防ぐため、その ZIP を消しました(情報は `recording/misnamed_collection/` に残しました)。
  同じ 60 シーンを、ブリッジの同じ API(`app.js` の buildCollection と同じ要求)で、`jetformer-simulator-rl-bestlap-20min` としてまとめ直しました(`record/rebuild_collection.py`)。
- `browser_collect.py` は、次からは連続試験の前に名前を入れるよう直しました。

### 6.4 教師の印

- 元の RC 教師は `not_learned_policy: true`(台本の追従器)です。RL の方策は学習した方策なので、`not_learned_policy: false`・`learned_policy_kind: reinforcement-learning-ppo`・
  `uses_ground_truth: true` にしました。観測は正解の状態(79 次元)で、カメラ・ToF は使っていません。
- このため、ユーザーの `qa/verify-rc-teacher-dataset.py` は「Teacher marker」の 1 行で止まります。そこで、その 1 行だけを
  「教師の id が `rl-bestlap-ppo-v1` で `uses_ground_truth` が真」に変えた写し(`record/verify_rc_teacher_dataset_rl.py`)で検証しました。ほかの検査はすべて元のままです。
- 集約 ZIP の約束(`collection.json` の contract.teacher)は `rl-bestlap-ppo-v1` です。教師の 20 分データ(`rc-response-teacher-v1`)とは別のものとして記録されています。

---

## 7. 次の段(JetFormer の学習 S-R)への注意

- **ラベルは教師より荒いです**(`recording/label_stats.json`)。
  - 舵が端(1000 / 2000 µs)にある時間: 18.8%(教師 3.7%)。
  - 舵の PWM の 1 フレームの変化の平均: 58 µs(教師 18 µs)。95% 点は 165 µs(教師 56 µs)。
  - スロットルが最大の指令(4.5 m/s)にある時間: 37%(教師 22%)。
  - カメラで走る JetFormer がまねするのは、教師より難しい見込みです。rl.md 12 章の「操作をなめらかにする」は、まだ入っていません。
- このデータだけで学習すると、`model_config` の中心値が教師データのときと同じようにずれる見込みです(design.md 1 章: 1375 / 1333 µs)。
  シミュレーターで走らせる前に、1442 / 1498 に直します(design.md 6.1 節)。
- train / val はシーンで分かれています(train 48 / val 12。開始は 60 通りで、分割をまたいで同じ開始はありません。集約の検査で確認)。
- 6.1 節のとおり、この方策の閉ループは小さな差を大きくします。閉ループで評価するときは、1 本の結果ではなく、複数の開始で見る必要があります。

---

## 8. ユーザーの質問「速く・ぶつからないための有望なアップデート」への補足(この段で分かったこと)

答えの本体は rl.md 12 章です。この段の記録で、次の 3 つが数字で見えました。

1. **今の方策は、小さなずれを大きくします。** 1e-14 の差が、20 秒で 0.2〜0.6 m の走りの違いになりました(6.1 節)。60 本はすべて衝突 0 でしたが、
   実車のずれはこれよりずっと大きいです。そのため、評価はいつも複数の開始・複数の物理で見る必要があります。
   rl.md 12 章の「操作をなめらかにする(舵の変化に上限)」は、ラベルの荒さだけでなく、この感度も下げる見込みがあります(まだ試していません)。
2. **観測に「跳び」があります。** 参照ループの頂点で、向きのずれの観測が 0.036 跳び、速さの指令が 1 フレームだけ 0.03 m/s 変わりました(6.2 節)。
   参照ループの向きを頂点の間でなめらかにつなぐ(線分ごとの向きではなく補間した向きにする)と、この跳びを無くせます。手間の小さい直しです。
   ただし、方策の学習し直しが要ります。
3. **ラベルが荒いです**(7 章: 舵が端 18.8%・舵の変化は教師の 3.3 倍)。カメラで走る JetFormer がまねしやすいよう、なめらかさを強めた方策で
   記録し直す案も有望です。記録のしくみはそのまま使えます(`record/install_bestlap.py --policy-dir <方策のフォルダー>` で教師を差し替え、
   同じ計画で記録できます。所要 約 85 分)。

---

## 9. 作ったもの

コード(`/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/`):

| ファイル | 役目 |
|---|---|
| `web/rl-teacher-bestlap.js` | 差し込みモジュール(RL の方策を RC 教師の差し込み口で走らせる) |
| `record/install_bestlap.py` | dist/rl/ への設置(sha256 の確認、`bestlap_manifest.mjs`)と `rl.html` |
| `record/make_plan.py` | 60 シーンの計画(名前だけ変える) |
| `record/predict_rec60.mjs` | 記録前の予想(Node・元の sim-core.js・同じモジュール) |
| `browser_collect.py`(追加あり) | Firefox で連続試験と収集。置き場所・途中の画面・進み具合・名前を前もって入れる、を追加 |
| `record/rebuild_collection.py` | 集約 ZIP を正しい名前で作り直す |
| `record/verify_rl_collection.py` | 60 シーンの検証(ユーザーの verify-rc-collection.py と同じ組み立て) |
| `record/verify_rc_teacher_dataset_rl.py` | ユーザーの verify-rc-teacher-dataset.py の写し(教師の印の 1 行だけ変更) |
| `record/teacher_mismatch_probe.mjs` | 方策の再計算を、再生した状態と記録した状態の 2 通りで比べる |
| `record/divergence_probe.mjs` | ブラウザと Node の閉ループの差、Node の中の感度 |
| `record/import_dataset.py` | JetFormer への取り込みとタグ |
| `record/ui_screens.py` | JetFormer の画面を撮る(ヘッドレス、タブの切り替えだけ) |
| `record/record_figs.py` | 図 |

結果(`/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/`):

- `rl_dataset.json`: 取り込んだデータセット。
- `recording/summary.json`: この段の数字のまとめ。
- `recording/install.json`・`plan_rlbest60.json`・`prediction.json`。
- `recording/browser/`: `result.json`・`progress.jsonl`・`collect.log`・`after_batch.png`。
- `recording/session_ids.json`・`collection.json`・`check.json`。
- `recording/verify/`: 2 回目の検証(合格)。`recording/verify_run1_teacher_from_node_replay/`: 1 回目。
- `recording/mismatch/`・`divergence.json`・`label_stats.json`・`ui_screens.json`・`import.log`・`bridge_8791.log`・`misnamed_collection/`。
- `screens/`・`fig/record_laps.png`・`fig/record_trajectories.png`。
- `deviations.md` に 2026-10-14 03:20 の項目を足しました。

---

## 10. 後片付け

- Firefox・geckodriver・ブリッジ(8791)はすべて止めました。jf-heavy の `big` のロックは放してあります。ほかに動かしているものはありません。
- ディスク: コピーのシミュレーターの `sim/runtime/datasets/` が 2.6 GB です(60 シーンのセッション・集約 ZIP・作業用の `_frames`)。
  検証の証拠として残しました。空きは 20.1 GiB です。`_frames`(約 0.3 GB)は消しても検証には困りません。
