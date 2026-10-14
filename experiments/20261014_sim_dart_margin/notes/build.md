# 準備の段 — DART の先生・記録の道具・正解の付け替えの道具(sim_dart_20261014)

- 作成 2026-10-14 14:05(JST)。この段の仕事は design.md / protocol.json の「準備」です: ブラウザで動く DART の先生(差し込みモジュール)、
  記録の道具(browser_collect.py の写しを広げたもの)、Node の正解の付け替えの道具、確かめと試験、ブラウザでの小さな通しの試し。
- 依頼(ユーザー、2026-10-14):「1. 離脱復活モードのデータで、ずれたときの戻し方を教えます。強化学習の走り手を先生にして、ずれた状態での正しい操作を教えます
  (DART や DAgger と呼ばれる方法です)。2. 絶対壁にぶつからないモードや、余裕を持たせたラインのデータで学習します。にトライし、その結果を今日のレポートでまとめたいですね」
- 実車・PCA9685・I2C には触れていません。GPU は使っていません。Firefox は `jf-heavy --mem 2500M --lock big` の中で 2 回だけ(13:29〜13:40・13:40〜13:44)。
  ほかの作業のファイルは読むだけ(使うものは `experiments/20261014_sim_dart_margin/` に写した)。jetformer-web は再起動していません。git にはコミットしていません。
  **本番の `data/` には何も書いていません**(試しの取り込みは一時の data フォルダーで行い、消しました)。
- 道具の置き場所: `/home/jetson/jetformer/experiments/20261014_sim_dart_margin/`(以下 `EXP/`)。報告・記録の置き場所: `/home/jetson/jetformer_reports/sim_dart_20261014/`(以下 `R/`)。
  ファイルの sha256 は `R/build_sha256.txt`。

---

## 0. 結論

| 確かめたこと | 結果 |
|---|---|
| 先生が元と同じ計算か | **同じ**。T_fast はベストラップの `rl-teacher-bestlap.js`(A0 を記録したファイル)と、T_margin は modes の `createModeTeacher(mode 'line', c0 = +0.3, c_v = 1.0)` と、操作が **ビット単位で一致**(乱しありも。`tests/teacher_core.test.mjs`) |
| 記録の前の Node の予想(30 シーン × 3) | D1(最速 + N_fast): **30/30 が k = 0 で通る**。D3(余裕の線 + N_margin): **30/30**(k = 0 で 25、k = 1 で 4、k = 2 で 1。外れはゲートの外し)。D2(余裕の線、乱しなし): 30/30 |
| 乱しの大きさ(目標: 乱しごとの最大のずれ 中央値 5〜10 cm・90% 点 ≤ 25 cm) | D1 **中央値 5.8 cm・90% 点 12.0 cm**(5〜25 cm が 60%)、D3 **7.6 cm・12.0 cm**(85%)。乱したフレーム D1 16.3%・D3 22.6%。最初の乱しは 0.8 秒までに必ず入る |
| R1: 乱しなしの A0 の 60 シーンを付け替えると元と同じか | **合格**。学習に使う 35,400 フレームすべてで元の正解との差 **0.0 µs**、外すフレーム 0。記録の操作の再生の差 ≤ 1.3e-13 |
| fork(分かれ)と「基準の作り方」(開始から全部やり直す O(N²)) | 同じ状態(Node の再生)から始めると **差 0**(D1・D2・D3 の試し 各 85 フレーム)。記録の状態から始めた fork と基準の差は ≤ 1.2e-10(rad・m/s。PWM では 0.0001 µs 未満) |
| ブラウザの通しの試し(D1・D2・D3 の train-00 を 1 シーンずつ) | 記録 → 検証 → 集約 ZIP → **一時の** JetFormer の data に取り込み → 付け替え → `validate_dataset` まで **すべて通った**。info.dart が frames.jsonl に入り、jf は入らず、取り込みの学習除外(label_mask)は掛からない |
| 付け替えの中身(試しの D1・D3) | 乱していないフレーム t+1 の最初の正解 = 記録の操作(差 **0.0 µs**)。10 個の窓に乱しが無ければ 10 個とも記録の操作と同じ(差 **0.0 µs**、D1 4,576 個)。乱したフレームの次の正解は記録の操作と平均 97 µs(D1)・176 µs(D3)違う。ほかの配列は変わらない |
| 試験 | node 8/8、pytest 5/5 |

見つけて直した問題が 5 つあります(6.4 節)。いちばん大事なのは **ディスクの空き**です: アプリの「1つのZIPにまとめる」は空き 20 GiB を求めますが、今の空きは 約 15 GB で、
まとめが止まります。取り置きだけを 6 GiB にした同じ処理(`record/build_collection.py`)でまとめるようにしました(deviations.md)。

---

## 1. 作ったもの

| ファイル(`EXP/`) | 中身 |
|---|---|
| `web/teacher_core.mjs` | 先生の中身。`fast`(T_fast: features.mjs の 79 次元・policy_forward.mjs・重み)と `margin`(T_margin: obs_modes.mjs の cond97・線 v2・c0 = +0.3)。ブラウザと Node が同じファイルを使う。`fork()` で内部の状態(進みの追跡 s・仮想ゲート・線の起点)の写しを作る |
| `web/dart_noise.mjs` | 乱しの予定(N_fast・N_margin。protocol.json dart をそのまま)。乱数は `rng.mjs`(splitmix32 系)を `mix32(mix32(種, 塩), k)` で初期化。場所・すき間・ゲートの見張り |
| `web/dart_driver.mjs` | 先生 + 乱し。毎フレーム きれいな操作 → 乱しを足す → 「前回の行動」は適用した操作(乱していない成分は先生自身の値)。`info.dart = {v, f, clean, noise, ev}` を返す(jf は出さない) |
| `web/dart-teacher.js` | 差し込み口(ベストラップの `rl-teacher-bestlap.js` を写して広げたもの)。`createRcTrainingTeacher`・`assertRcTeacherContract`。シーンごとの乱しの記録を `globalThis.__dartLog` に残す。取り直しの k は `globalThis.__dartSetOverride(JSON)` |
| `web/entry-D1.js`・`entry-D2.js`・`entry-D3.js` | データセットごとの入口(先生・乱し・塩・k の表を決める) |
| `web/plan_D1.mjs`・`plan_D2.mjs`・`plan_D3.mjs` | k の表(Node の予想が書く)。`alt` はブラウザの取り直し用に Node で通る次の k 2 つ |
| `web/dart_manifest.mjs` | データセットごとの先生の id・方策の sha256・web/ のファイルの sha256(`install_dart.py` が書く) |
| `web/rt/` | modes の `rl/runtime/` の写し 16 個(sha256 は元と同じ)+ `weights_fast.mjs`(ベストラップの policy_weights.mjs の写し)+ `weights_margin.mjs`(line r2 の policy.json を ESM にしたもの。中身の一致を確かめる) |
| `web/gate_miss.mjs` | modes の `rl/gate_miss.mjs` の写し(import の道だけ変えた) |
| `lib/app_emulation.mjs` | ベストラップの写し(SIM_ROOT を今日のコピーに) |
| `record/install_dart.py` | コピーの dist(`R/sim/dist`、ユーザーの dist と 33 ファイル同じ)に `dart/` と `d1.html`・`d2.html`・`d3.html` を置く。写しの sha256・重みの中身を確かめる |
| `record/make_plans.py` | 計画 `R/record_prep/plan_D{1,2,3}.json`(A0 の計画の 0〜23・48〜53 番、名前だけ変える) |
| `record/predict_dart.mjs` | 記録の前の Node の予想(入口そのものを Node で走らせる)。k を決め、`plan_Dn.mjs` に書く |
| `record/check_margin_e0.mjs` | 余裕の線の単周 E0 の周回時間の確かめ(deviations.md の 3) |
| `browser/browser_collect.py` | 記録の道具。ベストラップの写しに、シーンごとの合格判定(600 フレーム・衝突 0・ゲートの外し 0・全周有効)・横の記録(sidecar)・取り直し(Node で通った次の k、2 回まで)を足した |
| `browser/run_record.sh` | 記録の段で使う: ブリッジ(8811)を起動 → 30 シーンを記録 → まとめ → ブリッジを止める |
| `browser/run_browser_smoke.sh`・`smoke_pipeline.sh` | この段の試し |
| `browser/webgl_probe.py`・`run_bridge.py`・`calibration_only_policy.py` | ベストラップの写し(geckodriver 4449・ブリッジ 8811〜8819・プロファイル `R/ff_profile` に変えた) |
| `record/build_collection.py` | 集約 ZIP をアプリと同じ処理で作る(空きの取り置きだけ 20 → 6 GiB) |
| `record/verify_dart_collection.py` | 記録の後の検証(計画との一致・再生・周の数え直し・先生と乱しの再計算・センサー・画像がすべて違う) |
| `record/verify_recording.mjs`・`verify_rc_teacher_dataset_dart.py` | ベストラップの検証の写し(import の道・教師の id の行だけ変えた) |
| `record/import_dataset.py` | 取り込み(D1・D3 はシーンのタブに展開しない、D2 は展開する。label_mask が掛からないことを確かめる。`--temp-data-dir` で一時の data に) |
| `relabel/relabel_scene.mjs` | 付け替え(1 シーン、Node・元の sim-core.js)。`--no-fork` は検証だけ |
| `relabel/relabel_dataset.py` | 付け替え(データセット)。`--mode check`(読むだけ)/ `write`(元の配列を写してから NPZ の future_actions・future_valid だけを書き換え、dataset_card.json に relabel を足す) |
| `tests/teacher_core.test.mjs`・`entry_modules.test.mjs`・`make_fixture.mjs`・`test_relabel.py` | 試験(5 章) |

---

## 2. DART の先生(ブラウザの差し込み口)

- **やり方はベストラップの記録と同じ「方法 A」**: コピーの dist の `d1.html`・`d2.html`・`d3.html` の import map で、app.js が読む `./rc-speed-controller.js` を
  `./dart/entry-Dn.js` に替える。app.js・sim-core.js・収集器・集約はユーザーのまま。開始・物理の約束は元の RC 教師の検査(`assertRcTeacherContract`)のまま。
- **毎フレームの計算**(Node の付け替えと同じファイル):
  1. 先生のきれいな操作 â_t = π(s_t)(観測の「前回の行動」= フレーム t−1 で実際に適用した操作。乱していない成分は先生自身の値)
  2. 乱しの予定(`dart_noise.mjs`): 待ち時間が過ぎていて、場所(modes の `perturbAllowed`)・すき間(N_fast ≥ 12 cm / N_margin ≥ 10 cm)・ゲート(手前 1.2 m で |横| ≤ 0.30 m)
     がよければ、舵のかたより ±U(…) を U(…) フレーム、25% は速さも × U(…)。すき間(< 6 / 5 cm)かゲート(|横| > 0.45 m)で打ち切る
  3. 適用する操作 ã_t = clamp(â_t の舵 + かたより, ±0.52)・clamp(â_t の速さ × 倍率, 0, 4.5)
  4. `info.dart = {v: 1, f, clean: [δ, v], noise: [dδ, 倍率] | null, ev}`(フレーム 0 だけ `p = {dataset, seed, salt, k, scale, spec, teacher}`)。
     アプリが frames.jsonl の `action_info` に残す(試しで確かめた)。**jf は出さない**ので、JetFormer の取り込みの学習除外(`label_mask.py`)は掛からない(試しで確かめた)
- **教師の印**(context.teacher、全シーンで同じ): id は protocol どおり `rl-bestlap-ppo-dart-v1`(D1)・`rl-line-margin-c03-v1`(D2)・`rl-line-margin-c03-dart-v1`(D3)。
  `not_learned_policy: false`・`uses_ground_truth: true`・方策の sha256・乱しの決まり・web/ のファイルの sha256。
- **元と同じ計算であること**(`tests/teacher_core.test.mjs`): 
  - fast: A0 の計画の 0 番・49 番を 600 フレーム、`rl-teacher-bestlap.js` と操作が全フレームでビット単位で同じ(乱しなし)。さらに入口 D1 を乱しなしで走らせた操作列の
    sha256 が A0 の予想(ベストラップの `recording/prediction.json`)と同じ(最初の 2 シーン)
  - margin: `createModeTeacher(line, c0 = +0.3, c_v = 1.0)` に同じ乱しを外から足し、「前回の行動」を `extra.applied` で渡したものと、きれいな操作・適用した操作が全フレームでビット単位で同じ(乱しなし 1 シーン・乱しあり 2 シーン)
  - fork の写しが元と同じ続きを出す(fast・margin)・乱しは (種, 塩, k) で決まり同じフレームで 2 回呼んでも同じ・決まりが protocol.json と同じ

---

## 3. 記録の前の Node の予想(`R/record_prep/prediction_D{1,2,3}.json`)

入口(`dist/dart/entry-Dn.js`)そのものを Node で走らせました(元の sim-core.js、app.js と同じ順)。合格 = 衝突なし・20 秒走り切り・本当のゲートの外しなし(modes の gate_miss.mjs)・1 周目が有効。

| | D1(最速 + N_fast) | D2(余裕の線) | D3(余裕の線 + N_margin) |
|---|---|---|---|
| 合格 | 30/30(すべて k = 0) | 30/30 | 30/30(k = 0: 25、k = 1: 4、k = 2: 1。外れた 6 回はすべてゲートの外し、衝突 0) |
| 1 周目(停止から)の平均 | 8.134 秒(乱しなし 8.091) | 8.936 秒 | 9.082 秒(乱しなし 8.936) |
| 乱しの回数(打ち切り) | 291(7) | — | 298(23) |
| 乱したフレーム | 16.3% | 0 | 22.6% |
| 乱しごとの最大のずれ 中央値 / 90% 点 | 5.8 / 12.0 cm | — | 7.6 / 12.0 cm |
| 5〜25 cm の割合 | 60% | — | 85% |
| 最初の直線のずれ(中央値) | 3.4 cm | — | 6.6 cm |
| 最小すき間(乱しあり) | 2.9 cm | 11.6 cm(乱しなし) | 11.4 cm |

- 乱しの大きさは目標(中央値 5〜10 cm・90% 点 ≤ 25 cm)に入っている。設計の段の下調べ(6.5 / 11.8 cm、8.0 / 12.0 cm)とほぼ同じ(乱数の作り方を protocol の splitmix32 系に替えたので 1 本ずつは違う)。
- ブラウザの取り直し用に、各シーンで Node で通る次の k を 2 つ(`alt`)探して表に書いた(D1・D3 の 30 シーンすべてで 2 つ見つかった)。
- 乱しなしの確かめ(protocol recording.gate_checks): D1 の乱しなしの最初の 2 シーンの操作列は **A0 の予想と sha256 が同じ**。D2 の単周 E0 は 8.932 秒で、
  設計の段の 8.924 秒と 8 ms 違う。原因は設計の段の道具の「前回の行動」の渡し方で、その渡し方にすると **8.924 秒で一致**した(deviations.md の 3)。

---

## 4. 正解の付け替え(`relabel/relabel_scene.mjs`・`relabel_dataset.py`)

**決まり(protocol.json relabel)**: 記録した全フレーム t(t < N − 10)の正解 `future_actions[t, k]`(k = 0〜9)を、本当の状態 s_{t+1} から先生だけで運転したときの操作
â_{t+1+k} の PWM にする。乱した操作は正解にしない。`future_valid[t]` = 元の future_valid かつ「先生だけの 2 秒(60 フレーム)で衝突なし・本当のゲートの外しなし・すき間 ≥ 2 cm」。

**計算のしかた**:
1. シーンの `session.json`・`evaluation-report.json`・`frames.jsonl` を読み、開始から記録の操作を元の sim-core.js で再生する。記録の状態との差 ≤ 1e-9 でなければ失敗(R2)。
   sim の route_index(目標ゲート)とゲートの数えは再生から取る(記録に無いため)。
2. 先生(ブラウザと同じ `teacher_core.mjs`)を開始から同じ順に進める。**状態は記録の値**(ブラウザが計算した値そのもの。画像はこの状態から描かれた)。
   各フレームで計算し直した先生の操作と記録の `info.dart.clean` を比べる(R2)。乱しの予定も計算し直して `info.dart.noise` と比べる。
3. 各 t で s_{t+1} から分かれる: 物理は sim-core の単周の `reset({initialState: {x, z, yaw, speed, steering, routeIndex: 0}})` を、経路を今の目標ゲートで回した world で
   (単周は周の最後のゲートで止まるので、60 フレームの中で周の終わりに届かないように)。先生は `fork()`(begin / advance に渡した位置の列を新しい先生に流し直す)。
   先生だけで 60 フレーム走らせ、最初の 10 個を正解、60 フレーム全体を立て直せるかの判定に使う。
4. SI → PWM は **収集器そのもの**(コピーの `dataset_recorder.py` の `action_to_pwm`、session.json の calibration・model_config)で行い float32 にする。
5. `--mode write`: 元の `future_actions`・`future_valid` を `EXP/relabel/backup/<データセット>/<シーン>.npz` に写し(元の NPZ の sha256 も)、NPZ のその 2 つだけを書き換え、
   読み直してほかの配列が同じことを確かめる。dataset_card.json に `relabel = {schema "jf-sim-dart-relabel/v1", teacher, policy_sha256, noise, horizon 10, validity_horizon_frames 60, frames, changed_frames, invalid_frames, tool_sha256, …}` を足す。
   2 回目の write は断る。

**R1(道具の同一性、`R/record_prep/r1_A0_check.json`)**: A0 の 60 シーン(乱しなし、ブラウザで記録)を `--mode check` で付け替えた。

| 項目 | 結果 |
|---|---|
| 学習に使うフレーム | 35,400(60 シーン × 590) |
| 元の正解との差 | **最大 0.0 µs**(1 µs を超えるフレーム 0)。変わったフレーム 0 |
| 外すフレーム(先生が立て直せない) | **0** |
| 記録の操作の再生と記録の状態の差 | 最大 1.27e-13 |
| 記録の操作の PWM の作り直し | 差 0.0 µs |
| fork の最初の正解 = 本線の先生の計算 | 35,400 / 35,400 でビット単位で同じ |
| 基準の作り方(O(N²))との比べ(2 シーン、37 フレームおき) | 最大 5.1e-12(記録の状態から始めた fork と、Node の再生の状態から始めた基準の差。ブラウザと Node の最後の桁の差) |

- 1 シーンの付け替えは 約 60〜90 秒(ほかの作業と CPU を分け合っている。3 並列で 30 シーン 約 15 分)。R1 は途中で 1 度止まり(19 シーンで「フレーム 0 の状態が記録と違う」、6.4 節の 4)、
  直してから残りだけを計算し直した(`--reuse-work`。合格済みの 41 シーンはそのまま使った。正解の計算は直す前後で同じ)。

**基準の作り方との一致**(protocol: A0 の 2・D1 の 2・D3 の 1 シーン): A0 の 2 シーン(上)・D1・D2・D3 の試し 各 1 シーン(85 フレーム)。
`--states replay`(fork も基準も Node の再生の状態から)では **差 0**、`--states recorded`(本番)では最大 1.2e-10。D1 の 2 シーン目は記録の段で
`--reference-scenes 2` を付けて確かめる(7 章のコマンドに入れた)。

---

## 5. 試験

| 試験 | 中身 | 結果 |
|---|---|---|
| `tests/teacher_core.test.mjs`(node --test) | 乱しの決まり = protocol / fast = rl-teacher-bestlap.js / margin = createModeTeacher(乱しありも)/ 乱しの決定性 / fork | 5/5 |
| `tests/entry_modules.test.mjs` | 入口(dist/dart/entry-D1〜D3.js)の操作列 = Node の予想(D3 は k = 1 の表のシーンも)・info の形・jf が無い・教師の id | 3/3 |
| `tests/test_relabel.py`(pytest) | ① 乱しなしの本物の記録(A0 の 1 シーン)で正解が元と同じ(差 0 µs)・外す 0 ② 乱しなしの Node の記録(D2、150 フレーム)で書いても正解・future_valid が同じ ③ 乱しありの Node の記録(D1・D3、150 フレーム): 再生が一致、乱していないフレーム t+1 の最初の正解 = 記録の操作(≤ 1e-3 µs)・乱したフレームでは 9 割以上違う、窓に乱しが無ければ 10 個とも記録の操作と同じ・乱しがあれば 6 割以上違う(舵と速さが両方頭打ちのフレームは同じになる)、ほかの配列は同じ、元の配列の写し・dataset_card の relabel、2 回目は断る ④ PWM の式が収集器と同じ | 5/5 |

ログ: `R/record_prep/logs/node_tests.out`・`pytest_relabel.out`。

---

## 6. ブラウザの通しの試し(`R/smoke/`)

### 6.1 記録(Firefox、DISPLAY :0、geckodriver 4449、ブリッジ 8811、`jf-heavy --mem 2500M --lock big`)

| | D1 train-00 | D2 train-00 | D3 train-00 |
|---|---|---|---|
| 結果 | 合格(1 回目) | 合格(1 回目) | 1 回目(k = 0)は **ブラウザだけゲートを外した**(Node の予想は通る)→ 取り直し(k = 1)で合格 |
| 周回(ブラウザ / Node) | 8.203・7.805 / 8.201・7.760 秒 | 8.936・8.624 / 8.931・8.653 秒 | 9.115・8.829 / 9.105 秒 |
| 乱したフレーム・乱しの回数 | 92(Node と同じ)・9 | 0 | 143・10 |
| 1 シーンの時間 | 112 秒(5.4 フレーム/秒) | 123 秒 | 112 + 102 秒 |

- D3 の 1 回目の外れは本物の外し: フレーム 581〜582 で目標ゲートの線を横 0.70 m(半幅 0.65 m の外)で横切り、sim はゲートを数えなかった。
  ブラウザと Node の最後の桁の違いが閉ループで大きくなり、Node では通った乱しの並びがブラウザでは外れた(ベストラップの記録と同じ性質)。
- 記録の後の検証(`verify_dart_collection.py`)は 3 シーンとも合格: 再生の差 ≤ 4e-14、先生の再計算と info.dart.clean の差 ≤ 5.4e-14、乱しの印の不一致 0、
  衝突 0、ゲートの外し 0、全周有効、センサーの検査(ユーザーの検証器の写し)合格、カメラ・ToF の画像 600 枚がすべて違う。
  D3 は lap_audit の数え直しで「目標ゲートの線を外で横切った」が 1 回あった(そのあと曲がりで線の手前に戻り、内側で横切り直して数えられた。sim の周は 33 ゲートで有効)。
  protocol のゲートの外しの決まり(s の決まり)では外しではないので合格にし、数は書くことにした。
  **[記録の段 14:40 の訂正(build_check_r1 の指摘)]** 上の「内側で横切り直して数えられた」は誤り。このシーン(k = 1)はフレーム 584 でゲート 3 を横 0.666 m(半幅 0.65 m の外)で
  横切り、シーンの終わり(フレーム 600)までそのゲートは数えられなかった(**終わりの外し**。2 周は有効なので sim の周の集計には出ない)。s の決まり(数えないまま 2.019 m 進む)は
  最後の約 0.5 秒の外しを見られない。付け替えでは t = 576〜589 の 14 フレームが「先生が立て直せない」として学習から外れるので、正解は影響を受けない。
  記録の段から `verify_dart_collection.py` は「終わりの外し」(外で横切り、終わりまで数えられないゲート)を数えて書く(deviations.md 9)。

### 6.2 まとめ → 一時の取り込み → 付け替え → validate(`smoke_pipeline.sh`)

| | D1 | D2 | D3 |
|---|---|---|---|
| 集約 ZIP | `build_collection.py`(アプリのまとめは空き 20 GiB の検査で止まった) | 同じ | 同じ |
| 取り込み(一時の data) | 合格。タグ `simulator`・`rl-dart`、label_mask なし、本番の modes.json 変わらず | 合格(`rl-margin`) | 合格(`rl-margin`・`rl-dart`) |
| 付け替え | write | check(乱しなし: 正解は元と同じ、差 0.0 µs) | write |
| 変わったフレーム / 590 | 171 | 0 | 230 |
| 乱していないフレーム t+1 の最初の正解 − 記録の操作 | 0.0 µs(498 フレーム) | 0.0 µs | 0.0 µs |
| 窓に乱しが無い正解 − 記録の操作 | 0.0 µs(4,576 個) | 0.0 µs | 0.0 µs |
| 乱したフレームの次の正解 − 記録の操作 | 平均 97 µs(95% が 1 µs 超) | — | 平均 176 µs |
| 外したフレーム(先生が 2 秒で立て直せない) | 0 | 0 | **14(2.4%)**: 最後の 14 フレーム(t = 576〜589)。ゲートの外し(上の外れたゲートと同じ場所) |
| validate_dataset のあと | エラーなし、学習に使えるフレーム 590 | 590 | 576 |
| 基準の作り方との差(--states replay / recorded) | 0 / 3.1e-11 | 0 / 1.2e-10 | 0 / 1.4e-11 |

- 一時の data フォルダーは消した。ブリッジの記録(`R/sim/runtime/datasets/` の 9 個、約 200 MB)も消した(一覧は `R/smoke/runtime_sessions_deleted.txt`)。
  結果の JSON・横の記録(sidecar)・付け替えの報告は `R/smoke/browser/D{1,2,3}/` に残した。

### 6.3 分かったこと

- **D3 はブラウザでの取り直しが要る見込み**: Node の予想でも 30 本中 5 本が k = 0 でゲートを外し、ブラウザでは閉ループの差でさらに外れることがある(試しで 1 本中 1 本)。
  取り直しは 1 シーン 約 2 分。余裕の線(c0 = +0.3)はゲート 3(経路の 4 番目)の近くで外寄りを通り、乱しで横に 0.7 m 近くまで出ることがある。
- **D3 は外すフレームが 2% を超えるシーンがありうる**(試しで 2.4%)。protocol R4 は「超えたら書き、そのまま使う」。
- 記録の速さは 5.4〜6 フレーム/秒(ベストラップの 7.1 より遅い。ほかの作業と CPU を分け合っているため)。**30 シーンで 約 55〜60 分**(+ 取り直し)。

### 6.4 見つけて直した問題

1. **集約 ZIP が作れない**: ユーザーの `qa/combine_scene_datasets.py` が「空き ≥ 20 GiB + 元の 2 倍」を求め、今の空き(約 15 GB)では止まる。
   → `record/build_collection.py`(同じ処理で取り置きだけ 6 GiB。ほかの検査・中身・名前は同じ)。`run_record.sh` はアプリのまとめが失敗したら自動でこれを使う。deviations.md。
2. **取り直しの k が効かない**: WebDriver のスクリプトが `globalThis.__dartOverride` に入れたオブジェクトは、ページのコードから読めなかった(1 回目の試しで k = 0 のまま)。
   → ページの側の関数 `globalThis.__dartSetOverride(JSON の文字列)` で渡し、効いたことを返り値と sidecar の k で確かめる。
3. **取り直しの待ちがすぐ終わる**: 前の回の「連続試験終了」の表示が残っていた。→ 押す前に表示を消す。
4. **フレーム 0 の状態の比べ**: 付け替えがフレーム 0 の向きを記録とビット単位で比べていたが、開始姿勢の正規化(atan2)の最後の桁がブラウザと Node で違うことがある(A0 の 19 シーン)。
   → フレーム 0 は記録の姿勢(ブラウザの値)を使い、比べは再生の検査(≤ 1e-9)に任せる。
5. **ゲートの外しの数え方が 2 つある**: lap_audit(外で横切った回数)と protocol の s の決まり。→ 合否は protocol の決まり、lap_audit の数は書くだけ。
   **[記録の段 14:40 の訂正]** 試しの D3 の 1 本は「横切り直して数えられた」のではなく、終わりの外し(数えられないまま終わった)だった。6.1 の訂正を参照。

---

## 7. 次の段(記録)の手順

```bash
E=/home/jetson/jetformer/experiments/20261014_sim_dart_margin; R=/home/jetson/jetformer_reports/sim_dart_20261014
# 0) k の表は記録の前の予想で決まっている(web/plan_D*.mjs)。dist に写っていることを確かめる(変えたら install し直す)
python3 -I $E/record/install_dart.py
# 1) 記録(1 つずつ。D1 → D2 → D3。各 約 55〜60 分 + 取り直し)。Firefox の前に空きメモリ 2 GB 以上・ほかの Firefox が無いこと(browser_collect が待つ)
/home/jetson/bin/jf-heavy --mem 2500M --lock big -- bash $E/browser/run_record.sh D1 $R/recording/D1
#    → $R/recording/D1/result.json(accepted・rejected・retried)・sidecars/・collection.json(集約 ZIP の場所)
# 2) 検証(CPU)
source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
/home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 python -I $E/record/verify_dart_collection.py --dataset D1 \
    --result $R/recording/D1/result.json --out-dir $R/recording/D1/verify --workers 2
# 3) 取り込み(本番の data/。D1・D3 はシーンのタブに展開しない、D2 は展開する。label_mask が掛かったら止まる)
ZIP=$(python -I -c "import json;print(json.load(open('$R/recording/D1/collection.json'))['zip'])")   # アプリがまとめたときは result.json の collection
/home/jetson/bin/jf-heavy --mem 2G --lock none -- python -I $E/record/import_dataset.py $ZIP --dataset D1 --out $R/recording/D1/import.json
# 4) 付け替え(D1・D3 だけ。まず check で基準との比べ・R4 を見てから write)
DS=/home/jetson/jetformer/data/datasets/$(python -I -c "import json;print(json.load(open('$R/recording/D1/import.json'))['name'])")
/home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 python3 -I $E/relabel/relabel_dataset.py --dataset-dir $DS --teacher fast --noise N_fast \
    --mode write --jobs 3 --reference-scenes 2 --reference-stride 37 --report $R/recording/D1/relabel.json
#    D3: --teacher margin --noise N_margin --reference-scenes 1。D2 は --mode check(正解が変わらないことの確かめ)だけ
```

- `relabel_dataset.py --mode write` はデータセットの NPZ を書き換える(元の配列は `EXP/relabel/backup/<データセット>/`)。取り消すときは backup の 2 つの配列を戻し、dataset_card.json の relabel を消す。
- 学習の腕 A1・A3 の「2 つのデータを合わせる」準備(シンボリックリンクと splits.json)は学習の段で行う(protocol arms.combined_datasets)。

---

## 8. 気をつけること

- **ディスク**: 空き 約 15 GB(ユーザーの集約の取り置き 20 GiB を下回っている)。1 データセットでブリッジの記録 約 0.9 GB + 集約 ZIP 約 0.33 GB + 取り込み 約 0.45 GB ≈ 1.7 GB、
  3 つで 約 5 GB。学習の checkpoint も入る。集約 ZIP を取り込んだあと、ブリッジの記録(`R/sim/runtime/datasets/` のシーンごとの ZIP)を消せば 0.9 GB ずつ空く(消すかは記録の段で決める)。
- **時間**: 記録は 1 データセット 約 1 時間(5.4〜6 フレーム/秒)。protocol の見込み(45 分)より遅い。D1 を 14:10 ごろに始めると D2 は 15:10 ごろ、D3 は 16:10 ごろからで、
  「D3 の記録を 15:30 までに始められない → A3 を削る」(cut_order 3)に当たる見込み。判断は記録の段・メインのセッション。
- **lock big の待ち**: この段では modes の記録の終わり(13:29)まで 約 15 分待った。
- **余裕の線の方策は modes の作業では「出さない」候補**(L2 の 1 本の極端な開始だけ不合格)。この作業の開始には当てはまらない(design.md)。

---

## 9. 手順と違えたこと(deviations.md 2026-10-14 13:25・14:05 の項目)

1. 先生の写しは snapshot/restore を足さず、写したファイルは元と同じまま `fork()`(位置の列の流し直し)で分かれた(基準の作り方と差 0)
2. 差し込みモジュールは同じ部品を `teacher_core.mjs` で組み立てる(元のモジュールとビット単位で同じことを試験で確かめた)
3. 余裕の線の単周 E0 は 8.932 秒(設計の段の渡し方では 8.924 秒で一致)
4. ブラウザの試しは 20 秒のまま 3 シーン
5. 付け替えは 1 シーン 約 60〜90 秒
6. 集約 ZIP の空きの取り置きを 6 GiB にした(`build_collection.py`。14:05 の項目)
7. 記録の後の検証の合否は protocol のゲートの外しの決まりで行い、lap_audit の「外で横切った回数」は書くだけ(14:05 の項目)
8. R1 は道具の誤りで 19 シーンが止まり、直してからその 19 シーンだけ計算し直した(14:05 の項目)

## 10. 時間

- 12:48 開始。設計の確認・写し・先生と乱し(〜13:05)、試験・予想(〜13:15)、付け替えの道具・R1 の開始(〜13:15)、
  ブラウザの試しは lock big を 13:15〜13:29 待ち、13:29〜13:44 に記録、検証・取り込み・付け替えの試し(〜14:00)、R1 の合格(13:58)。
- protocol の準備の段の終わり(13:50)より 約 15 分遅れ(lock の待ちと、試しで見つけた 5 つの問題の直し)。削ったものはなし。
