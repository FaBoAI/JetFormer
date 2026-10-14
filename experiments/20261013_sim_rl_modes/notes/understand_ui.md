# シミュレーターの強化学習モード — ブラウザ側と JetFormer 統合の理解(understand_ui)

- 作成: 2026-10-13(理解の段・UI 側。読むことと小さな確認だけ。リポジトリのコードは変えていません)
- 依頼(ユーザー): 「シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども
  作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意」
- この段の役目: JetFormer が配信するシミュレーターのページに RL モードを足し、モード別に収集・命名・取り込み・学習除外・障害物・状態表示を
  するための、**いちばん小さく壊れにくいやり方**を決める。

読んだもの(すべて読むだけ):

| 対象 | ファイル(sha256 の先頭 12 桁) |
|---|---|
| JetFormer のシミュレーター統合 | `jetformer/sim/web.py`・`courses.py`・`datasets.py`・`paths.py`・`static/sim-shim.js`・`course-scene.js`・`course-world.js`、`jetformer/web/static/{index.html,app.js}` の「シミュレーター」タブ、`jetformer/dataset_import.py`・`scene_import.py`・`dataset.py`・`train_data.py`・`modes.py`・`model_export.py`、`tests/test_sim_web.py`・`test_sim_js.py` |
| ユーザーのシミュレーター(導入済みの版 `20261013-144013-6f610dd5`) | `dist/index.html`(9225f75dd742)・`dist/app.js` 全 576 行(4a43205fbca1)・`dist/rc-speed-controller.js`(4a32ab56369b)・`dist/sim-core.js`(aaeb23798ee4)・`dist/world.js`(4099860131f4)・`dist/scene.js`(878fc934af28)・`dataset_recorder.py`(5c968ca36e3d)・`qa/combine_scene_datasets.py`(83575470e5cb) |
| ベストラップの RL(別のワークフロー。読むだけ) | `rl/features.mjs`(3cf941237cdc)・`rl/reference_loop.mjs`(446d4bbc506d)・`rl/policy_forward.mjs`(f16f9f50a5b9)・`rl/env_core.mjs`(df99dacf25c9)・`web/rl-teacher-selftest.js`(d57e16bfd070)・`install_wrapper.py`・`browser_collect.py`(43555b929f81)・`notes/understand_teacher.md`、レポート `sim_integration_20261013/understand_app.md` |

features.mjs・reference_loop.mjs・policy_forward.mjs はコミット c0a334c から変わっていません(git status で確認)。統合の段で使うときは
この sha256 を記録して `experiments/20261013_sim_rl_modes/` か `jetformer/sim/static/rl/` にスナップショットとして写してください。

---

## 0. 結論(5 つの問いへの答え)

1. **RL モードをページに足す**: JetFormer が返す `/sim/c/<コース>/` の index.html の **import map だけ**を書き足し、アプリが読む
   `./rc-speed-controller.js`(「RC加速教師」= 連続周回の収集に使える唯一の教師の差し込み口)を JetFormer のモジュール
   `./_jf/rl/teacher.js` に差し替える(scopes で `./_jf/` の中からは元のファイルを読む)。差し替えモジュールは元の名前を全部再エクスポートし、
   `createRcTrainingTeacher`・`assertRcTeacherContract` だけを「RL モードが選ばれていれば RL の方策、そうでなければ元の RC 教師」に置き換える。
   方策の重みは JetFormer の API(`/api/sim/rl/modes/<id>/policy.json`)から fetch し、観測は **features.mjs と同じファイル**で計算する。
   モードの選択欄・状態表示は shim と同じ classic script(`/sim/static/sim-rl.js`)がページに足す。ユーザーのファイルは変えない。
   この import map の組み方は、JetFormer と同じ階層・非ループバックのアドレスで Firefox 157 で確かめた(7 章)。
2. **収集プランでモードと分数を選ぶ**: プランは JetFormer(Python)で作る(モードごとに 分 × 3 シーン、20 秒・600 フレーム、train/val、
   開始のずれ・シード・場面・障害物の配置)。ページの「JetFormer 収集プラン」の実行係が **1 シーンずつ** アプリ自身の「連続試験」(1 条件)を
   押して記録する(アプリの検証・記録・保存の流れをそのまま使い、シーンごとにモード・配置を切り替えられる)。
   ジャンルは全部 **8(シミュレーター)のまま**、モードは **データセット名とタグ**(`simulator`・`sim-rl`・`sim-rl-<モード>`)で区別する。
   まとめは **JetFormer 側で複数のシーン ZIP を 1 つのデータセットに取り込む**(モード 1 つ = 1 データセット)のを主にする。ブリッジの
   「1つのZIPにまとめる」は空き 20 GiB + 元の 2 倍を要求し、今の空き 24.05 GiB では 20 分のモードを 2〜3 個まとめると使えなくなる(3.5 節)。
3. **学習しないフレーム**: 収集器と集約は「フレームごとの学習除外」を持てない(`vehicle_valid` はシーン単位、`future_valid` は最後の 10 フレームだけ
   false、集約はそれ以外を拒否)。そこで **教師が返す操作に `info` を付けて**(`{steering, speed, info: {jf: {mode, phase, learn, …}}}`)、
   収集器の `simulation/frames.jsonl` の `action_info` に残す(確認済み: 集約後も `simulation/scenes/<シーン>/frames.jsonl` に残る)。
   JetFormer の取り込みで、学習しない操作(`learn: false`)を正解に含むフレーム **k ∈ [乱れの開始 − 10, 乱れの終わり − 2]** の
   `future_valid` を false に、`mode` を 0(未設定 = 学習しない)にする。これで今の学習(`train_data`)とシーンタブへの展開・
   シーンからのデータセット作り直しの両方が、そのまま除外を守る。
4. **障害物**: 物理は円(`disks`、`post-foot` 以外の kind は全部円)とカプセル(`segments`)で、kind は自由(sim-core は検査しない)。
   ベストラップの観測(レイ 19 本・`d_min`)は高さに関係なく全部の障害物を見る。カメラと ToF はシーンのメッシュを描くので、
   **メッシュを足せば見える**。(a) アップロードのコース(汎用描画)には kind `obstacle` を足す。(b) Default の見た目のまま、シーンごとに
   配置を変えるには、**仮想コース `default-obstacles`**(Default の scene.js・world.js + JetFormer の重ね描き `_jf/scene-overlay.js`)を作る。
   アプリの `world.disks` は scene の `physics.disks` と**同じ配列**なので、重ね描きが配列の中身とメッシュを入れ替えれば、次のリセットで
   物理・センサー・RL の観測の全部に入る(確認済み)。配置の ID はフレームの `info` と報告の world_hash に残る。
5. **シミュレーター タブの状態表示**: 新しいカード「強化学習モード」。登録簿 `data/sim_rl/modes/<id>/{mode.json, policy.json, eval.json, status.json}`
   を `/api/sim/rl/modes` で返し、モードごとに 状態(学習済み・学習中・未学習)・ラップ(停止から/走りながら)・衝突(試験の回数のうち)・
   最小のすき間・完走率・学習日時と、「このモードで開く」「収集プラン(モードごとの分数)」を出す。数字は RL の学習側が**元の sim-core で**
   評価して `eval.json` に書く(JetFormer は表示するだけ)。

---

## 1. 今の仕組み(要点だけ)

### 1.1 JetFormer のシミュレーター統合

- `/sim/c/<cid>/` = ユーザーの `dist/index.html` に 3 行(`./_jf/config.js`・`/sim/static/sim-shim.js`・`sim-shim.css`)を `<meta charset>` の直後に
  差し込んだもの(`web.py` `index_html()`)。import map・`app.js` が 1 つずつであることを確かめ、違えば 500。`/sim/c/<cid>/<path>` は
  `scene.js`・`world.js`・`assets/` はコースのもの、ほかは `dist/` のもの。`_jf/` は `config.js` と `course-data.js` 以外 404(予約済み)。
- shim(classic script、アプリのモジュールより先に動く): ブリッジ宛ての絶対 URL(`http://127.0.0.1:8766/api/...`)だけを `/sim/bridge/api/...` に
  書き換える(相対 URL は触らない → ページの JS は `/api/sim/...` を普通に呼べる)、`crypto.randomUUID` を補う(非ループバックの http で開くと無いため)、
  上の帯(戻る・コース・ブリッジの状態)、`#endpoint` を読み取り専用に、`course-20s-*` の localStorage をコース(world_version)ごとに分ける、
  コースの `features.teachers` に無い教師の選択肢を消す(**Default 以外では `speed-rc` が消える**)、`?model=<run>` で JetFormer のモデルを読み込む。
- コースの種類(`courses.py`): `release`(Default)・`snapshot`・`trusted-js`・`data`(アップロード。JetFormer の汎用描画 `course-scene.js` で描く。
  `validate_world()` は kind を `segments: rail/curtain/gate-foot`・`disks: post-foot/post/gate-post` に限る)。`_features(cid)` は `default` だけ
  全教師・60 シーンの収集条件を許す。
- データセット(`datasets.py`・`web.py`): ブリッジの `runtime/datasets/<sid>/` の完成 ZIP を一覧し、`dataset_import.import_archive(p, simulator=True)` で
  JetFormer に取り込む(ジョブ `import`)。アプリの「JetFormerへ送る」も JetFormer の `PUT /api/datasets/import` → 同じ `import_archive` を通る。
  取り込みはカードの `tags` + `simulator` を付け、シーンタブ用に `data/raw/ds-<名前>/` へ展開する(`scene_import.dataset_to_scenes`)。

### 1.2 アプリ(app.js)の制御・教師・収集

- コントローラー(`#controller`)は `manual`・`demo`・`external` の 3 つだけ。収集器も `controller` をこの 3 つに限る(`dataset_recorder.py:207`)。
  **新しいコントローラーの値は足せない**。RL は `demo` + 教師の差し込み口で入れる。
- 教師(`#teacher-profile`): `slow`(`sim.demoAction`)・`speed`・`speed-fast`・`speed-rc`・`speed-optimized`。**連続周回(`continuous-laps-v1`)の
  `demo` は `speed-rc` しか許さない**(app.js:100・505)。`speed-rc` のときだけ `createRcTrainingTeacher(activeWorld, config, {gate, left_offset_m, left_yaw_deg})`
  が呼ばれる(app.js:119)。→ 差し替える口は `./rc-speed-controller.js` のこの 2 関数。
- 差し込み口の約束(ベストラップの理解の段と同じ。今回あらためて読んで確かめた点を足す):
  - `createRcTrainingTeacher` は 1 シーンで **2 回**呼ばれる(連続試験のリセット時と、`record-start` の中のリセット時。app.js:541・330)。決定的・軽量にする。
  - `assertRcTeacherContract` は連続試験の**計画の検証でも**呼ばれ、そのときの world は**開始ゲートで回していない** world(`route[0]` = ゲート 0)、
    config に `seed` が無い(app.js:507)。リセット時は回した `activeWorld`。両方を通す検査にする。
  - `action(snapshot)` は観測を取った後、同じフレームの操作として `sim.step` に渡される(app.js:377・381)。記録では ±最大舵角・±最高速度に丸め、
    **`{...action, steering, speed}` をそのまま** `/api/datasets/frame` に送る(app.js:387)→ `action.info` も届く。
  - 一時停止や中断のとき、`action()` を呼んだ後に `sim.step` をしないことがある(app.js:379 の return)。教師の内部状態(前の行動・乱れの数え)は
    **`snapshot.state.frame` ごとに決まる形**(同じフレームでもう一度呼ばれても同じ答え)にする。
  - `metadata` は `context.teacher` に展開され、集約の契約に入る(全シーンで同じでないとまとまらない)。`options`・`reference.length`・
    `reference.planned_time_s` も `context.teacher` に入る(app.js:135)。シーンごとに変わる値は入れない。
- 「20分・60シーンの収集条件を用意」(app.js:441): 計画欄に `RC_COLLECTION_PLAN`(60 条件)、`#batch-record`・`#automated-labels` に印、ジャンル 8。
  **`automated-labels` は計画の項目に無い**(app.js:466 の許可リストの外)ので、実行係が印を付ける必要がある(付けないと学習ラベルにならず、集約も拒否)。
- 「連続試験を開始」(app.js:462): 1〜100 条件。条件の項目は `formIds` + `name`・`model-*`・`dataset-mode`・`dataset-split`・`initial-state*` だけ
  (**JetFormer 独自の項目は書けない**。書くと「不明な設定」)。1 条件ごとに: 項目を入れる → リセット → `dataset-name = name` → 収集開始
  (ここでもう一度リセット)→ 走る → 収集終了・シーンの ZIP → 連続周回・600 フレーム・衝突 0 なら `#collection-ids` に足す。
  ちょうど 60 条件で全部きれいなときだけ自動でまとめる(app.js:564)。結果は `#batch-results`(JSON。`dataset.session_id` を含む)に出る。
- 走るループは `execution-speed: fast` なら `setTimeout(0)`、それ以外は `requestAnimationFrame`(app.js:547)。**裏のタブでは止まる・遅くなる**。

### 1.3 収集器(dataset_recorder.py)と集約(combine_scene_datasets.py)

| 項目 | 中身 |
|---|---|
| PWM ラベル | 送られた操作(丸めたもの)を換算した値 = `vehicle[k]`。**`future_actions[k][j] = vehicle[k+1+j]`(j = 0..9)**。観測 k の正解は「次のフレームからの 10 個の操作」 |
| 換算の値 | 記録開始の時点でブリッジにモデルが読み込まれていれば、その `calibration`・`model_config`(操舵の中心・スロットルの中立)。無ければ既定(1441 / 1496 µs、最高速度は画面の値)。**`demo` でも読み込み中のモデルの値になる**(app.js:338) |
| フレームの有効 | `vehicle_valid` はシーン単位(`manual` か「自動ラベル」の印)。`future_valid` は最後の 10 フレームだけ false(:421–427) |
| ジャンル | `mode_id` は 0〜5 か 8 だけ(:210)。8 なら `num_modes` 9 |
| 名前 | 収集名 `name`(1〜100 文字、`/` なし)→ データ名 `<名前を英数字に 40 文字>_<UTC 時刻>_<sid 8 桁>`、シーン `<UTC 時刻>-<sid 8 桁>-0001`(:247–250) |
| フレームの記録 | `simulation/frames.jsonl` に 1 行ずつ: `command_si`・`applied_pwm_us`・**`action_info`(= `action.info`、128 kB まで)**・操作前の姿勢・操作後の状態 |
| シーンの記録 | `simulation/session.json`(開始時の context・config・換算)、`simulation/evaluation-report.json`(終了時の報告。`metrics.lap_events`・`collisions`・`world_hash`) |
| 集約が拒否するもの | `demo` でない・合成ラベルでない・`vehicle_valid` が全部 true でない・**`future_valid` が「最後の 10 だけ false」でない**・フレーム数が指定と違う・連続周回でない・衝突あり(既定)・姿勢が飛ぶ・同じ開始姿勢が train と val にまたがる・**契約(`world_version`・`route`(文字列)・センサー・`context.teacher`・物理・換算・`model_config`・ジャンル表)がシーン間で違う**・5 GiB 超・**空きが 20 GiB + 元の 2 倍より少ない** |
| 集約が保つもの | シーンごとの `simulation/scenes/<シーン>/{session.json, frames.jsonl, evaluation-report.json}`(`/` で始まる文字列と password などの鍵は伏せ字) |

契約に入らないもの(= シーンごとに変えてよいもの): 開始姿勢・シード・場面(`scenario`)・**world の形そのもの**(`world_version` の文字列が同じなら、障害物の配置が違っても集約できる)・
フレームの `action_info`。

### 1.4 JetFormer の学習が持っている「学習しない」の仕組み

| 仕組み | 効く場所 | 今の UI の学習で使えるか |
|---|---|---|
| `future_valid & vehicle_valid`(aligned の npz) | `train_data.JetFormerDataset`(:129) | ○ |
| `mode == 0`(未設定)は使わない | 同上(:134)。シーンでは `scene.json` の `mode_segments` → データセット作成で `mode` | ○ |
| `drop` 表(scene, frame) | `JetFormerDataset(drop=…)` | × `training.py` から渡していない(実験のスクリプトだけ) |
| 衝突の前後(`mask_impacts`) | IMU の衝撃 | ○(シミュレーターの IMU は合成なので関係しない) |
| シーンの VEHICLE `valid`・`select` | `dataset._aligned_arrays`(100 ms 以内の有効な操作で補う) | ○ ただしフレーム単位では正確でない |

---

## 2. (1) RL モードをページに足す

### 2.1 仕組み(import map の差し替え。ユーザーのファイルは変えない)

`index_html()` が import map の JSON を読んで書き足す(文字列の置換ではなく JSON として):

```json
{"imports": {"three": "./vendor/three.module.js", "three/addons/": "./vendor/addons/",
             "./rc-speed-controller.js": "./_jf/rl/teacher.js"},
 "scopes":  {"./_jf/": {"./rc-speed-controller.js": "./rc-speed-controller.js"}}}
```

- キーの `./rc-speed-controller.js` はページの URL(`/sim/c/<cid>/`)を基準に解決されるので、app.js が読む元のファイルの URL が差し替わる。
  `./_jf/` の中(`/sim/c/<cid>/_jf/rl/teacher.js`)から `../../rc-speed-controller.js` を読むと、scopes で元のファイルになる。
- **確かめた**(7.2 節): JetFormer と同じ `/sim/c/default/_jf/rl/` の 2 階層で、非ループバック(`http://192.168.55.1:8801`)・Firefox 157 で、
  差し替え・元の読み込み・`export *` の上書き・トップレベル await(重みの fetch)・mode が `rc` のときに元の教師に戻ることを確認。
- `web.py` で `_jf/rl/<名前>` を `jetformer/sim/static/rl/` の決まったファイル(許可リスト)から配る。`index_html()` のキャッシュの鍵と ETag に
  これらのファイルの mtime を足す。`tests/test_sim_web.py::test_index_injects_three_lines` は import map の中身が変わるので直す。
- 環境変数(例 `JETFORMER_SIM_RL=0`)で差し替えを止められるようにしておくと、アプリの版が変わったときに逃げられる。

### 2.2 置くファイル(JetFormer 側)

| 配信のパス | 元(リポジトリ) | 役目 |
|---|---|---|
| `/sim/c/<cid>/_jf/rl/teacher.js` | `jetformer/sim/static/rl/teacher.js` | ブラウザの糊: 元を再エクスポート、`createRcTrainingTeacher`・`assertRcTeacherContract` を置き換える。登録簿(`/api/sim/rl/modes`)をトップレベル await で読む(3 秒で打ち切り、失敗したら RL なし = 元の教師のまま)。`window.JETFORMER_RL = {modes, mode, load(id), status}` を作る |
| `/sim/c/<cid>/_jf/rl/rl-core.mjs` | 同 `rl-core.mjs` | 純粋な ESM(ブラウザの globals を使わない): `createModeTeacher(world, config, variation, modeSpec, weights)` → `{action, reference, options, metadata}`。乱れの予定・配置の記録・`info` を作る。**Node の再生(`verify_recording.mjs` の形)でもこのファイルを import する** |
| `/sim/c/<cid>/_jf/rl/features.mjs`・`reference_loop.mjs`・`policy_forward.mjs` | ベストラップの写し(sha256 を記録) | 観測(79 次元)と方策の計算。学習(Node の env)と**同じファイル** |
| `/sim/static/sim-rl.js` | `jetformer/sim/static/sim-rl.js` | classic script(4 行目として差し込む): ページに「強化学習モード」の選択欄・状態・収集プランの実行係を足す |

- 重みは `GET /api/sim/rl/modes/<id>/policy.json`(登録簿のファイルを JetFormer が検査して返す)。選んだモードだけを `load(id)` で読む
  (今の JSON 形式は 1 モード約 1.9 MB。float32 の base64 にすると約 470 kB。値は float32 なので float64 に直しても今の JSON と同じ数になる。
  変えるなら Node と比べる試験を付ける)。
- **`crypto.subtle` は非ループバックの http では無い**(`randomUUID` と同じ理由)。ページで sha256 を計算しない。sha256 の確認はサーバーで行い、
  登録簿の値を `metadata` に入れるだけにする。
- `import … with {type: 'json'}`(policy_forward.mjs の `loadWeights`)は Safari 17.2 未満で使えないので、ブラウザでは `fetch().json()` + `createPolicy()` を使う。

### 2.3 差し替えの教師の約束(実装の注意)

- `JETFORMER_RL.mode` が無い・`rc` なら、元の `createRcTrainingTeacher` / `assertRcTeacherContract` をそのまま呼ぶ(今と同じ動き)。
- RL モードのとき:
  - コース: features.mjs の参照ループは Default(v3・33 ゲート)の形で作ってあるので、**RL モードは Default と、その上に障害物を重ねた仮想コースだけ**。
    `world.route` が標準の経路を開始ゲートで回したもの(か、検証時の回していないもの)であることを確かめ、違えば日本語のエラー。
  - 物理・雑音: 元の `assertRcTeacherContract` に「障害物を除いた標準の world・ゲート 0・ずれ 0」を渡して確かめる(selftest と同じやり方)。
    開始の範囲(ゲート 0〜32・左右 ±0.3 m・±25°)と時間はモードの定義で自前に検査する。
  - `action(snapshot)` → `{steering, speed, info: {jf: {mode, phase, learn, layout, clearance_mm}}}`。`info` は小さく(1 フレーム 100 バイト前後)。
    `/` で始まる文字列・`steering_pwm`/`throttle_pwm` の鍵は使わない(集約の伏せ字・収集器の PWM 指定に当たる)。
  - 乱れ(離脱復活)や経路の変化は `config.seed`(リセットの config に入っている)と `frame` から決まる関数にする(Node で同じ操作を作り直せる)。
  - `metadata` = `{schema: 'jetformer-sim-rl-teacher/v1', id: 'jf-rl-<mode>-<版>', mode, policy_sha256, features_version, reference_loop_sha256,
    speed_cap_mps, mode_params(モードの決まった値), obstacle_set(あれば)}`。**シーンごとに変わる値(配置・乱れの時刻)は入れない**。
  - `reference = {length: 参照ループの長さ, planned_time_s: eval.json のラップ}`(表示と記録用)。

### 2.4 ページの見た目(sim-rl.js)

- 「反復試験・教師データ収集」の「教師の走行」の下に **「強化学習モード」**の選択欄: 「使わない(RC加速教師)」+ 登録簿のモード
  (未学習は選べない。学習中は「学習中」と出す)。選ぶと: 重みを読み込む → `controller = demo`・`teacher-profile = speed-rc`・物理とセンサーを
  RL を学習した条件(`RC_COLLECTION_PLAN[0]` の値。teacher.js が元のモジュールから取り出して `JETFORMER_RL.basePlan` に置く)にそろえる →
  `#teacher-profile` に change を送ってアプリにリセットさせる。`speed-rc` の選択肢の文字を「強化学習: <モード名>」に変えて、画面と中身を合わせる。
- 上の帯に「RL: <モード名>(読み込み済み)」。`?rl=<id>` で開いたらそのモードを選んだ状態にする(コースの切り替えでも `rl` を引き継ぐ)。
- 連続試験・収集中は選択欄を使えなくする(`#batch-run` が disabled の間)。

---

## 3. (2) 収集プラン: モードと分数、名前・タグ・ジャンル

### 3.1 プランを作る場所: JetFormer(Python)

`POST /api/sim/rl/plans {course, minutes: {<mode>: 分, …}, scene_s: 20, val_ratio: 0.2, seed, scenarios, import: {merge: true, delete_sources: true}}`
→ `data/sim_rl/plans/<id>.json`。pytest で試せる。1 シーン = アプリの連続試験の 1 条件:

- 決まった値: `RC_COLLECTION_PLAN[0]` と同じセンサー・物理(`maxspeed` 7.24・RC 応答・`execution-speed: fast`・`stop-collision: true`)、
  `controller: demo`・`teacher-profile: speed-rc`・`episode-mode: continuous-laps-v1`・`limit: 20`・`dataset-mode: 8`。
- シーンごとに変える値: `name`(`rl-<mode>-<split>-<番号>`)・`dataset-split`・`seed`・`start-gate`(0〜32)・`spawn-offset`(±0.3)・
  `spawn-yaw`(±25)・`scenario`(`baseline`・`noise`・`dim` を順番に。`low-friction` は RL がその物理で学習したモードだけ)。
  開始姿勢は低食い違い列で**重ならない**ようにする(集約と同じく、同じ開始姿勢が train と val にまたがると拒否するため)。
- JetFormer だけが使う値(アプリには渡さない): `jf_mode`・`jf_layout`(障害物の配置)。
- 分 → シーン数 = 分 × 3。1 回の連続試験は 100 条件までだが、下の実行係は 1 条件ずつ押すので関係しない。

### 3.2 実行係(ページの sim-rl.js)

`/sim/c/<cid>/?rlplan=<id>` で開くと「JetFormer 収集プラン」の枠が出る。利用者が「開始」を押すと:

1. ブリッジの状態を読み、**換算(読み込み中のモデルの `calibration`・`model_config`)を記録**。モデルが読み込まれていて、操舵の中心・スロットルの中立が
   実車の値(1442 / 1498 µs 前後)から外れていれば止める(シミュレーターだけで学習したモデルは 1375 / 1333 µs になっていることがある。
   その値で記録すると PWM ラベルがずれる)。途中で換算が変わったら止める。
2. `#batch-record`・`#automated-labels` に印、`#dataset-mode` = 8。
3. シーンごとに: `JETFORMER_RL.load(mode)` → 配置を適用(5 章)→ 計画欄に**その 1 条件だけ**を入れて `#batch-run` を押す → `#batch-run` が使えるように
   戻るまで待つ → `#batch-results` の JSON から `session_id`・衝突・周回を読む → `POST /api/sim/rl/plans/<id>/progress`(タブの進み具合用)。
   失敗(衝突・時間切れ・保存の失敗)は記録して次へ(同じ条件をもう一度やるかは任意)。
4. モードが終わるたびに、そのモードのシーンを JetFormer に取り込む(3.4)。

1 条件ずつにする理由: アプリの検証・記録・保存の流れをそのまま使いつつ、シーンの合間にモード・配置を切り替えられる(アプリの計画には
JetFormer の項目を書けない、教師は計画の何番目かを知らない)。アプリの自動まとめ(ちょうど 60 条件のとき)も起きない。

### 3.3 名前・タグ・ジャンル

| 何 | 決め方 |
|---|---|
| ジャンル | **8(シミュレーター)のまま**。収集器は 0〜5・8 しか受け付けず、集約も ID8 に 9 枠を要求する。モードごとに別のジャンルにするのは、走るときに「最速」「安全」を選び分けたいときの別の話(取り込み時にジャンルを付け替える。モデルの `num_modes` が増える)なので今回はしない |
| シーンの収集名 | `rl-<mode>-<split>-<番号>`(データ名・`session.json` の `display_name` に残る) |
| データセット名 | `sim-rl-<mode>_<YYYYMMDD-HHMM>`(JetFormer 側でまとめるとき。英数字・`_`・`-`) |
| タグ | `simulator`・`sim-rl`・`sim-rl-<mode>`(+ 障害物ありは `sim-obstacles`、乱れありは `sim-recover`)。取り込み時に、ZIP の中の `context.teacher.schema == 'jetformer-sim-rl-teacher/v1'` を見て自動で付ける(サイドの情報に頼らない)。シーンタブに展開したシーンにも同じタグが付く |
| カード | `rl: {mode, teacher_id, policy_sha256, scenes, minutes, label_mask(4 章), obstacle_layouts, calibration}` を足す |

### 3.4 まとめ方: JetFormer 側の「まとめて取り込み」を主にする

- ブリッジの集約(`/api/datasets/collection`)は **空き ≥ 20 GiB + 元の ZIP の 2 倍** を要求する。今の空きは **24.05 GiB**。
  20 秒のシーン ZIP は約 10 MiB(60 シーンで約 0.6 GiB)、取り込んだ 20 分のデータは約 1.1 GB(今ある 20 分データの実測)。
  まとめ ZIP・取り込み・元の削除を毎回してもまとめられるのは **20 分のモードで 3 つ前後**まで。4 つ目でブリッジが拒否する。
- そこで **JetFormer が複数のシーン ZIP を直接 1 つのデータセットに取り込む**(`POST /api/sim/rl/import {session_ids, name}` → ジョブ `import`):
  各 ZIP の `aligned/<シーン>.npz`・`samples/…`・`can_bus/…` をそのまま、`simulation/*` は `simulation/scenes/<シーン>/` に写し、`splits.json` を合わせ、
  カードは最初のシーンのものに数を足す(集約と同じ形。nuScenes 風の表は token で重ねるか省く。JetFormer の学習とシーン展開は使わない)。
  集約の主な検査(600 フレーム・連続周回・衝突 0・契約が同じ・開始姿勢が split をまたがない)は JetFormer 側でも行う。
  終わったら `validate_dataset` → 学習除外(4 章)→ タグ → シーン展開。元のシーン ZIP は消してよい(プランの `delete_sources`)。
- ブリッジの「1つのZIPにまとめる」は、持ち運べる ZIP が欲しいときの道として残す(空きを確かめてから)。

### 3.5 かかる時間と注意

- 1 シーン(20 秒・600 フレーム)の記録は Jetson の Firefox(llvmpipe)で約 60 秒 + ZIP 約 10 秒(ベストラップの実測)。Mac の GPU ならもっと速い
  (1 フレームの送信の往復 約 14 ms が下限)。20 分のモード 1 つで Jetson なら約 70 分。
- **タブを表に出しておく**: 裏に回ると `setTimeout`/`requestAnimationFrame` が絞られ、ほぼ止まる。
- 衝突したシーンは 600 フレームにならない(`stop-collision`)ので、まとめに入らない。プランの結果に数を出す。
- 1 つのデータセットの中は換算(モデル)が同じでなければならない(1 の確認)。

---

## 4. (3) 学習しないフレームの印

### 4.1 なぜ JetFormer の取り込みで行うか

- ページから収集器にフレームごとの有効・無効を渡す口は無い(`vehicle_valid` はシーン単位)。集約は `future_valid` が「最後の 10 だけ false」で
  ないシーンを拒否する。→ 印は **`action.info` に入れて記録だけし、除外は JetFormer で行う**。
- **確かめた**(7.1 節): 収集器の `DatasetManager` を直接呼び、`action.info = {jf: {mode, phase, learn, layout}}` を付けた 24 フレーム × 2 シーンを記録して
  集約すると、`simulation/scenes/<シーン>/frames.jsonl` の各行の `action_info` にそのまま残った。npz の `future_valid` は最後の 10 フレームだけ false。

### 4.2 除外の決まり

- 操作 j が学習しないもの(`learn: false`。例: 離脱復活の「わざと外す」区間 [ps, pe))なら、それを正解(k+1〜k+10 の操作)に含む観測 k を除く:
  **k ∈ [ps − 10, pe − 2]**。最後の乱れのフレーム pe − 1 の観測は、正解が立て直しの操作だけなので**残す**(立て直しを学ぶいちばん大事な所)。
  乱れの間の観測(車が外れている絵)も、正解が立て直しなら学習に使う。
- 取り込み後のデータセットの npz で `future_valid[k] = False` と `mode[k] = 0` にする。`mode = 0` は JetFormer の「未設定 = 学習しない」なので、
  シーンタブへの展開(`mode_segments`)とシーンからのデータセットの作り直しでも除外が残り、シーンタブで区間として見える。
  `vehicle_valid` は変えない(その時の操作値は本当に出したもの)。
- カードに `label_mask = {schema: 'jetformer-sim-label-mask/v1', source: 'frames.jsonl action_info.jf.learn', horizon: 10, masked_frames, per_scene}`、
  `training_frames`・`mode_frames` を数え直す。単体のシーン ZIP(`simulation/frames.jsonl`)と集約 ZIP(`simulation/scenes/<シーン>/frames.jsonl`)の
  両方を読めるようにする(「JetFormerへ送る」の道も同じ `import_archive` を通るので、そこに入れればどの道でも効く)。
- 同じ仕組みで、ほかに学習させたくないもの(例: 経路の変化の切り替えの瞬間、障害物の真横を通る危ない場面)も `learn: false` で表せる。

---

## 5. (4) 障害物

### 5.1 物理・観測・センサー

| | 中身 |
|---|---|
| sim-core | `disks` は `post-foot`(0.16 × 0.15 の長方形)以外は全部**円**、`segments` は**カプセル**。kind は文字列なら何でもよい(検査しない)。衝突の記録に `kind` が残る |
| `sim.observe()` のレイ | 高さ 8 cm 以上の障害物だけ(外部の AI に送る 13 本) |
| RL の観測(features.mjs) | 高さに関係なく全部(レイ 19 本・`d_min`)。`prepareGeometry` は kind を問わない |
| カメラ・ToF | THREE のシーンのメッシュを描く(ToF は深度の描画)。**メッシュを足せば両方に見える**。カメラは床から 0.25 m、ToF は 0.20 m・俯角 16° なので、高さ 0.25 m 以上・半径 5 cm 以上にする |
| 元の RC 教師 | 形が v3 と完全一致しないと拒否(障害物があると使えない)。障害物ありは RL の教師だけ |

おすすめの形: 円柱(`disks`、kind `obstacle`、半径 0.05〜0.15 m、高さ 0.3 m)と、壁・箱は短いカプセル(`segments`、kind `obstacle`)を
「箱 + 両端の半円柱」で描く(見た目と当たりを合わせる)。色はコースに無い色(橙など)。テクスチャは使わない
(`DefaultLoadingManager.onLoad` にかかわらない)。

### 5.2 二つの入れ方

- **(a) アップロードのコースに固定の障害物**: `courses.validate_world()` の kind に `obstacle` を足し、`course-scene.js` に描き方を足すだけ。
  簡易描画のコースなので、カメラ画像は Default と同じにはならない(今の注意書きのまま)。RL モードは使えない(参照ループが Default 用)。
- **(b) Default の見た目のまま、シーンごとに配置を変える**(おすすめ): **仮想コース `default-obstacles`**(kind `overlay`、`base: default`)を登録簿に足す。
  このページだけ import map に `./scene.js` → `./_jf/scene-overlay.js`、`./world.js` → `./_jf/world-overlay.js`(scopes で元に戻す)を足す。
  - `scene-overlay.js`: 元の `buildCourse(scene)` を呼び、返ってきた `physics.disks`・`physics.segments` の**配列をそのまま持っておき**、障害物の
    THREE のグループをシーンに足す。`setLayout(layout)` で配列の中の kind `obstacle` のものだけを入れ替え、メッシュも作り直す。
  - アプリの `world = makeWorld(model.physics)` も、リセットのたびの `activeWorld = {...world, route, spawn}` も浅い写しなので、`disks` は同じ配列。
    次のリセットの `createSimulation`(world を写す)から、物理・センサー・RL の観測(teacher には `activeWorld` が渡る)に入る。**確かめた**(7.2 節)。
    リセットの後に配置を変えても、走行中の sim には入らない(sim は作るときに写す)→ 配置は必ずリセットの前に変える(実行係はシーンの前に変える)。
  - `world-overlay.js`: `export * from '../world.js'` + `export const WORLD_VERSION = '<元>+jf-obstacles-v1'`。障害物ありのデータは別の
    `world_version` になり、障害物なしのデータ(と元の RC 教師の条件)と混ざらない。配置が違ってもこの文字列は同じなので、同じモードのシーンは 1 つにまとまる。
  - 配置の記録: フレームの `info.jf.layout`(配置 ID)と、報告の `world_hash`(障害物を含めて計算される)。配置の一覧は
    `data/sim_rl/obstacle_sets/<set>.json`(`{id, base: 'default', layouts: [{id, obstacles: [{shape, x, z, r | length, yaw, h}]}]}`)。
    JetFormer が検査する(範囲・数・大きさ・開始位置や必須の黒シートの通り道をふさがない)。乱数で作る配置も、作るのは JetFormer(Python)で、
    ページは ID で受け取るだけにする(ページ側で作らない = 検査とテストがしやすい)。
  - `_features('default-obstacles')` に `speed-rc` を入れる(shim が選択肢を消さないように)。60 シーンの収集条件のボタンは隠す(RC 教師が使えないため)。
  - 外部の AI(JetFormer のモデル)もこのコースで走らせられる → 障害物を避けられるかの評価にも使える。

---

## 6. (5) シミュレーター タブの状態表示

### 6.1 登録簿(RL の学習側が書き、JetFormer は読むだけ)

```
data/sim_rl/modes/<id>/mode.json     {format: 'jetformer-sim-rl-mode/v1', id, name, description, order, course: 'default' | 'default-obstacles',
                                      features_version, reference_loop_sha256, policy_schema, policy_sha256, speed_cap_mps,
                                      start: {gates, left_offset_m, left_yaw_deg}, perturbation | null, route_variation | null,
                                      obstacle_set | null, scenarios, trained_at, trainer: {workflow, run, commit}}
data/sim_rl/modes/<id>/policy.json   重み(rl-bestlap-policy/v1 の形)
data/sim_rl/modes/<id>/eval.json     {evaluated_with: 'sim-core(元)', episodes, completed, collision_episodes, collisions,
                                      lap_from_stop_s, flying_lap_s, min_clearance_m: {min, p05}, mean_speed_mps, recover: {…} | null,
                                      start_distribution, obstacle_layouts, created_at}
data/sim_rl/modes/<id>/status.json   学習中だけ {state: 'training', progress, updated_at}(学習のスクリプトが書く)
data/sim_rl/obstacle_sets/<set>.json 障害物の配置
data/sim_rl/plans/<id>.json          収集プランと進み具合
```

置くときは「作業用の名前に書いて fsync → 置き換え」(`paths.write_durable` と同じ)。発行用のスクリプトを RL の段に用意する
(例 `experiments/20261013_sim_rl_modes/publish_mode.py`。重みの形・有限・大きさ・sha256 を確かめてから置く)。

### 6.2 API

| | |
|---|---|
| `GET /api/sim/rl/modes` | 一覧(状態・評価の要約・重みの sha256 と大きさ・集めたデータセット(タグ `sim-rl-<id>`)の数と分) |
| `GET /api/sim/rl/modes/{id}/policy.json` | 重み(検査済み。ETag・no-cache) |
| `GET /api/sim/rl/obstacles/{set}` | 配置の一覧 |
| `POST /api/sim/rl/plans`・`GET /api/sim/rl/plans/{id}`・`POST /api/sim/rl/plans/{id}/progress` | 収集プラン |
| `POST /api/sim/rl/import` | シーン ZIP をまとめて 1 つのデータセットに(ジョブ `import`。4 章の除外・3.3 のタグ込み) |

### 6.3 カード「強化学習モード」(右の列、「シミュレーターを開く」の下)

- 表: モード名(説明)/ 状態(学習済み・学習中 進み具合・未学習・評価なし)/ ラップ 停止から・走りながら / 衝突(例「0 回 / 200 試験」)/
  最小のすき間(cm)/ 完走率 / 学習日時 / 集めたデータ(分)/ ボタン「このモードで開く」(`/sim/c/<course>/?rl=<id>`)。
  「絶対にぶつからない」は言い切らず、「N 回の試験で衝突 0」と書く。
- 「収集プラン」: モードごとの分数(0 = 使わない)・train/val の割合・シード → 「プランを作って開く」(`?rlplan=<id>` で新しいタブ)。
  進み具合(`progress`)をここにも出す(何シーン目・衝突で外れた数)。
- 「シミュレーターで集めたデータセット」の一覧にモードの印(タグから)を出す。

---

## 7. 確かめたこと(小さな確認)

### 7.1 `action.info` が収集器と集約を通るか(ブラウザなし)

- `experiments/20261013_sim_rl_modes/probe_ui/recorder_info_probe.py`(ブリッジの venv の Python で `-B -I`。ユーザーのファイルは読むだけで
  `__pycache__` も作らないことを確認)。結果 `probe_ui/recorder_info/result.json`。
- 24 フレーム × 2 シーン(8〜11 フレーム目を `learn: false`)を記録し、`expected_frames: 24` で集約 → 集約は成功(48 フレーム・学習 28)。
  `simulation/scenes/<シーン>/frames.jsonl` の全行に `action_info.jf` が残り、`learn: false` は 8〜11 フレーム。乱れの舵 0.52 rad は PWM 2000 µs。
  `future_valid` は最後の 10 フレームだけ false(フレームごとの除外は JetFormer で行う必要がある、を確認)。`context.teacher` は契約にそのまま入る。
- 作った ZIP は消した(結果の JSON だけ残した)。

### 7.2 import map の差し替え・障害物の配列(Firefox 157、ヘッドレス。WebGL は不要)

- `experiments/20261013_sim_rl_modes/probe_ui/importmap_probe.py`、静的なファイルは `probe_ui/importmap/www/sim/c/default/`(ユーザーのファイルは使わない、
  同じ名前の小さな偽物)。`http://192.168.55.1:8801/sim/c/default/`(非ループバック)で開いた。`jf-heavy --mem 2500M --lock big` で実行。
  結果 `probe_ui/importmap/result.json`: `ok: true`。
  - `./rc-speed-controller.js` → `./_jf/rl/teacher.js`、teacher.js からの `../../rc-speed-controller.js` は元(`base: orig`)。`export *` で元の
    `RC_COLLECTION_PLAN`・`assertRcTeacherContract` がそのまま出る。`JETFORMER_RL.mode = 'rc'` で元の教師に戻る。
  - teacher.js のトップレベル await(fetch)を app.js が待つ。fetch の相対 URL はページ基準、`new URL(…, import.meta.url)` はモジュール基準。
  - `./scene.js` → `./_jf/scene-overlay.js`、重ね描き側からは元の scene.js、`three` は 1 つ(`three_same: true`)。
  - `setLayout()` で配列の中身を入れ替えると、`world.disks`・`{...world}` の `disks` に見える(`['post', 'obstacle']`)。
- Firefox・geckodriver・HTTP サーバー(ポート 4447・8801)は止めた。

### 7.3 動いている JetFormer の状態(GET だけ)

- `GET /api/sim/status`: ブリッジ 動作中(モデル未読み込み・収集なし)、学習・走行・記録なし。触っていない。
- 空き容量 24.05 GiB(3.4 節の計算に使用)。

(同じフォルダーの `probe/`(`obstacles.mjs`・`kicks.mjs` など)は並行して動いている RL 側の段のもの。この段は `probe_ui/` だけを使った。)

---

## 8. 落とし穴のまとめ

1. 非ループバックの http(`http://192.168.55.1:8080`)では `crypto.subtle`・`crypto.randomUUID` が無い。ページで sha256 を計算しない。確認は必ずこのアドレスで行う。
2. 連続周回の `demo` は `speed-rc` だけ → RL は必ず `speed-rc` の口から。Default 以外(アップロードのコース)では shim が `speed-rc` を消す。
3. 教師は 1 シーンで 2 回作られる・検証では回していない world と seed なしの config で呼ばれる・止めたときは同じフレームで 2 回 `action` が呼ばれうる。
4. `context.teacher`(metadata・options・reference の長さと時間)は全シーンで同じに。シーンごとの値は `info` へ。収集開始の要求は 2 MiB まで。
5. `#automated-labels` は計画の項目に無い → 実行係が印を付ける。無いと学習ラベルにならず、まとめも拒否。
6. 換算(PWM の中心・中立・最高速度)はブリッジに読み込み中のモデルで決まる。シミュレーターだけで学習したモデル(1375 / 1333 µs)が読み込まれたまま
   記録しない。データセットの中で換算をそろえる。
7. ブリッジの集約は空き 20 GiB + 元の 2 倍が要る(今は 20 分のモード 3 つ前後で尽きる)→ JetFormer 側でまとめて取り込む。
8. 収集のタブは表に出しておく(裏では止まる)。Jetson の Firefox では 20 分のデータに約 70 分。
9. 障害物の配置は、リセットの前に変える(sim は作るときに world を写す)。障害物ありのページは `world_version` を変える。
10. RL の方策は Default の参照ループで学習している → RL モードは Default と、その上の障害物の仮想コースだけ。
11. 学習除外は「正解に含まれる操作」で決める(観測 k の正解は k+1〜k+10)。乱れの直後の立て直しを消さない。

## 9. 統合の段で作る・直すもの(提案)

| ファイル | 変更 |
|---|---|
| `jetformer/sim/web.py` | import map の書き足し(JSON として)、`_jf/rl/*`・`_jf/scene-overlay.js`・`_jf/world-overlay.js` の配信(許可リスト)、4 行目 `sim-rl.js`、`/api/sim/rl/*` |
| `jetformer/sim/rl_modes.py`(新規) | 登録簿の読み込みと検査、プランの生成、配置の検査 |
| `jetformer/sim/rl_import.py`(新規)または `dataset_import.py` | シーン ZIP のまとめての取り込み、学習除外(4.2)、タグ(3.3) |
| `jetformer/sim/courses.py` | 仮想コース `overlay`(`default-obstacles`)、kind `obstacle`、`_features` |
| `jetformer/sim/static/rl/{teacher.js, rl-core.mjs, features.mjs, reference_loop.mjs, policy_forward.mjs}`・`static/scene-overlay.js`・`world-overlay.js`・`sim-rl.js`(新規)、`course-scene.js`(obstacle の描き方) | ページ側 |
| `jetformer/web/static/{index.html, app.js}` | カード「強化学習モード」・収集プラン・進み具合 |
| tests | `test_sim_web.py`(import map の中身・配信・API)、`test_sim_js.py`(node で teacher.js/rl-core.mjs をスタブの元モジュールで: 委ねる・RL・info・決定性・検証の world)、新規 `test_sim_rl.py`(プラン・登録簿の検査・まとめ取り込み・除外の範囲 [ps−10, pe−2]・タグ)、ブラウザの確認は `http://192.168.55.1:8080` で |
