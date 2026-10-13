# 教師と記録のしくみ — RL の方策でアプリを走らせ、同じ形式のデータを作る方法(理解の段)

作成 2026-10-13 ・ 対象 `/home/jetson/sim/extracted/course-simulator`(読むだけ。変更していません)
作業用のコピー `/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/sim/`(`dist` はユーザーの dist とバイト単位で同じ 33 ファイル + 追加の `rl.html`・`rl/`)
コード `/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/`

依頼: 「シミュレーターは強化学習でベストラップの走行を実現したいです。それをシミュレーターデータとして学習させたい」
この段の役目: 教師(参照軌跡と追従)・収集の流れ・センサー・PWM ラベルを読み解き、**RL の方策でアプリを走らせて、アプリ自身の収集器で
20 分データと同じ形式のデータを作る一番小さく確実な方法**を決め、ブラウザ(WebGL)で実際に動くかを確かめる。

16:08 の再起動の前に、この段の一部(コピー・WebGL の試験・RC 教師 1 シーンの記録と元データとの比較)はできていました。
壊れたファイルはありませんでした(0 バイトなし・JSON と ZIP はすべて読める)。その続きから進めました。

---

## 0. 結論

- **決めた方法**: 差し替えたページ `rl.html` の import map で、`app.js` が読む `./rc-speed-controller.js` を自分のモジュールに置き換える。
  アプリの「RC加速教師」(`teacher-profile: speed-rc`)の差し込み口に、RL の方策を JavaScript の関数として入れる。
  `app.js`・`sim-core.js`・収集器(`dataset_recorder.py`)・集約(`qa/combine_scene_datasets.py`)は元のまま使うので、
  **データの形式はアプリ本来のもの**。ユーザーのファイルは変えない。
- **確かめたこと**(自己テスト: RC 教師の操舵 + 速度指令 ×0.9、開始ずれを ±3 cm・±3° の外の 0.10 m・6° / −0.08 m・−5° に広げたもの)
  - ブラウザで 2 シーン(train 1・val 1、各 20 秒・600 フレーム)を収集し、アプリの「1つのZIPにまとめる」で集約 ZIP ができた(2 シーン・1,200 フレーム・21.1 MiB)。
  - ZIP の NPZ のキー・型・形・データカードの項目が元の 20 分データと同じ。PWM は記録した指令から計算し直した値と完全に一致(差 0 µs)。
  - 記録した操作を Node の元の物理で再生すると、全フレームの姿勢が一致(最大差 9e-15)。差し替えた方策を Node で計算し直しても一致(最大差 5e-14)。
- **WebGL**: ヘッドレス Firefox では WebGL が使えない(4 通り試して全部だめ)。**画面 :0 に窓を開くと WebGL 2 が動く**(snap の Firefox に入っている
  Mesa の llvmpipe = CPU 描画)。収集は **約 10 フレーム/秒**(20 秒 1 シーン ≈ 60 秒 + ZIP 作成 約 10 秒)→ 60 シーンで約 70 分。
- **注意が要る点**(後で詳しく):
  1. snap の Firefox は起動時に自分の systemd スコープへ移るので、**jf-heavy のメモリ上限が Firefox に効かない**。`browser_collect.py` は起動後にそのスコープにも上限を付ける。
  2. 元の 20 分データは、ブリッジにモデルを接続した状態で集めたので PWM の中心が **1442 / 1498 µs**。接続なしだと収集器の既定 **1441 / 1496 µs** になり 1〜2 µs ずれる。
     `run_bridge.py --policy calibration_only_policy.py`(走らないポリシー)で同じ換算にできる(元データとの差 0 を確認)。
  3. **シミュレーターのデータだけで JetFormer を学習すると、`model_config` の操舵中心・スロットル中立が 1375.3 / 1332.9 µs になる**
     (`train_data.action_reference()` が「止まっているフレームの中央値」を使い、シミュレーターでは各シーンの最初の 1 フレームしか止まっていないため)。
     実機データ(ai_mix)では 1442 / 1498。シミュレーターだけで学習したモデルは **1442 / 1498 に直す必要がある**(直さないと、まっすぐのつもりの出力が左に約 0.06 rad、4.5 m/s のつもりが約 3.1 m/s になる)。
  4. 速度の指令に上限を付けないと、RL は加速のために 7.24 m/s(PWM 1000 µs)を指令しがち。作者の教師は**指令 4.5 m/s まで**(PWM 1188.5 µs 以上)。実車の範囲に合わせるなら RL の行動も 0〜4.5 m/s にする。

---

## 1. 教師: 参照軌跡と追従

### 1.1 参照軌跡の点(`optimized-reference.js` の `points`、RC 教師の `RC_BASE_REFERENCE` も同じ形)

| 項目 | 意味 |
|---|---|
| `x, z` | 位置 [m]。前進は `x += v cos(yaw)`, `z −= v sin(yaw)`(左回りが yaw の正) |
| `s` | 始点からの道のり [m]。点の間隔は約 2.5 cm |
| `yaw` | 進む向き [rad] |
| `curvature` | 曲率 κ [1/m] = 前後 3 点の向きの差 / 道のり |
| `steering` | その曲率に要る舵角 = atan(ホイールベース 0.26 m × κ) [rad] |
| `mu` | その点の路面の摩擦係数(カーペット 0.8・黒シート 0.7・青マット 0.65・芝 0.6・PTFE 0.22)× `frictionMultiplier`(低摩擦の場面は 0.4) |
| `speed` | その点での**実際の速さ**の計画値 [m/s](指令ではない) |

速さの計画(`createSpeedReference` / `createRcSpeedReference`):

1. 各点の上限 = min(上限速度, √(tractionFraction · μ · g / |κ|), rateFraction · 操舵の速さ 1.2 rad/s / |dδ/ds|)
   - **横加速度の仮定は「v²·κ ≤ tractionFraction·μ·g」**(タイヤの滑り・摩擦円はない)。最適化参照は tractionFraction 1.0、RC 教師は 0.95。
   - 舵の速さの条件は rateFraction 2、つまり**計画上はサーボの 2 倍の速さで舵を切ってよい**ことになっている(実際の舵はシミュレーターで 1.2 rad/s に制限される)。
2. 後ろから: 減速の上限で間に合う速さに下げる(最適化参照は一定 1.2 m/s²、RC 教師は `brakeDeceleration` 1.2 m/s²)。
3. 前から: 停止から加速できる速さに下げる(最適化参照は一定 1.2 m/s²。RC 教師は「min(6 m/s², μg, (4.5 − v)/0.40 s)」の応答を 1 mm 刻みで積分)。

### 1.2 教師の種類

| 教師 | 参照軌跡 | 物理 | 追従のしかた | 1 周(停止から) |
|---|---|---|---|---|
| 最適化参照(`speed-optimized`) | `OPTIMIZED_REFERENCE`(1,092 点・26.56 m・最高 4.73 m/s・横加速度は μg ちょうどまで)。作者がゲート内の横のずらし(±2.5〜10 cm)を最適化した固定の軌跡 | **一定加減速専用**(RC 応答の物理では契約で拒否) | `createSpeedController`: 操舵 = 参照の舵角 + 2·向きの誤差 + atan2(2·横の誤差, |v|+0.35)、速度指令 = 参照の速さ(指令 = 計画の速さ) | 10.091 秒(一定加減速)。RC 応答の物理で無理に走らせると 11.12 秒 |
| **RC 教師**(`speed-rc`、今の 20 分データ) | `RC_BASE_REFERENCE`(1,092 点・1 周 26.12 m。埋め込みで export されていない)を RC 応答で速さを付け直す。連続周回では 1 周を 4 回つなぎ、つなぎ目(1.5 m)を 3 次エルミートでなめらかにする(104.5 m) | RC 応答(`real-drive-response-estimate-v1`) | 操舵は上と同じ式(ゲイン 2・2、先読み 0)。**速度指令 = clamp(参照の速さ + 0.40·max(0, 参照の加速度) + 2·(参照の速さ − 今の速さ), 0, 4.5)** | 単周 8.931 秒(計画 8.865 秒)。連続の 1 周目 8.71〜8.76 秒、2 周目 8.36〜8.41 秒 |

RC 教師の数字(`teacher_reference_stats.json`): 参照の最高 4.435 m/s・平均 3.27 m/s・横加速度の計画は最大 7.45 m/s²(= 0.95·0.8·g)。
走らせると指令は 4.5 m/s に張り付き、20 秒で横の頭打ち(ヨーレートを μg/|v| で切る小区間)が 1,635 回、横加速度は物理の上限 7.85 m/s² に達する。
**同じ RC 教師でも、開始のずれが 0.2 m・12° を超えると(速度 ×0.9 でも)ぶつかる**(`probe/selftest_precheck.mjs`)。
制御遅延 1 フレームで走らせても 16 ゲートでぶつかる(`probe_result.json`)。余裕の少ない教師です。

### 1.3 教師の契約(元の `assertRcTeacherContract`)

監査済み v3 コース・33 ゲート(形状が完全一致)、物理の値が監査値と同じ、雑音は 0 か 0.025、**開始はゲート 0・左右 ±3 cm・向き ±3° だけ**、
途中状態からの開始(initial-state)は不可、連続周回は 20 秒。この検査は、連続試験の計画を確かめるときと、リセットのたびに呼ばれます。

---

## 2. 収集の流れ(`app.js`)

1. (元の 20 分データでは)「接続を確認」でブリッジのモデル情報を読み込む → PWM の換算(1442/1498 µs・7.24 m/s・0.52 rad)が収集に使われる。
2. **「20分・60シーンの収集条件を用意」**: 計画欄に `RC_COLLECTION_PLAN`(= `qa/rc-collection-plan-60.json`、60 条件)を入れ、
   「収集する」に印、収集ジャンル 8、自動ラベル(教師の操作を学習に使う)を ON にする。
3. **「連続試験を開始」**: 計画を検査(使える項目・範囲・教師の契約)。1 条件ずつ:
   - 項目を画面に入れて `reset()`: コースの経路を開始ゲートで回転し、開始ずれ(左右 ±0.3 m・向き ±25° に丸める)を加えた開始姿勢でシミュレーションを作り、
     教師 `createRcTrainingTeacher(activeWorld, config, {gate, left_offset_m, left_yaw_deg})` を作る。連続周回では教師は RC 教師だけ。
   - 収集開始 `/api/datasets/start`(名前・controller `demo`・ジャンル 8・split・自動ラベル → `training_label_mode: synthetic_controller`・context(教師の情報・センサー設定)・config・PWM 換算)。
   - 1 フレームごと(`tick`): センサーを描く → 観測(カメラ PNG・ToF・IMU)→ **教師の操作 = `teacher.action(sim.snapshot())`** → `sim.step(操作)` → `/api/datasets/frame`(観測・操作・操作前の姿勢・操作後の状態)。
   - 20 秒(600 フレーム)で `collection_duration_complete`、またはぶつかって終了 → `/api/datasets/finish`(シーンの ZIP)。
     連続周回・600 フレーム・衝突 0 のシーンだけ集約の候補に入る。
   - **計画がちょうど 60 条件で全部きれいに終わったときだけ**、自動で `/api/datasets/collection`(1 つの ZIP にまとめる)。それ以外は「1つのZIPにまとめる」を手で押す。
4. 60 条件の中身: 開始はゲート 0、左右 −2.9〜+2.8 cm・向き −2.9〜+2.9°(重ならない低食い違い列)、seed 2026101200+i、
   **baseline と noise(ToF の各画素に 25 mm 相当の一様雑音)を交互**、train 0〜47(48 本)・val 48〜59(12 本)。
   雑音は ToF の描画にだけ入り、物理と教師には影響しない。連続周回は周の間にリセットしない(周の区切りは 33 ゲート + 黒シートの判定)。
5. 集約(`combine_scene_datasets.py`)が拒否するもの: controller が demo でない・合成ラベルでない・600 フレームでない・30 Hz が途切れる・
   姿勢が飛ぶ・連続周回でない・initial-state を使った・衝突あり・**シーン間で契約(コース版・経路・センサー・教師 `context.teacher`・物理・PWM 換算・モデル設定・ジャンル)が違う**・
   同じ開始姿勢が train と val にまたがる・5 GiB 超。ID8 は 9 ジャンルの枠が必要。

**RL の記録で効くこと**: `context.teacher`(教師の metadata)は全シーンで同じでなければまとめられない → シーンごとに変わる値を metadata に入れない。
途中状態からの開始(動的な立て直し)や単周のシーンは集約 ZIP に入れられない → 立て直しの多様さは「止まった状態からの、ずれた開始(ゲート・左右・向き)」の 20 秒連続シーンで作る。

---

## 3. センサー(`app.js` の `renderSensors`・`ego-sensors.js`・`sensors.js`)

| 入力 | 作り方 |
|---|---|
| カメラ | 320×180 で描いて 224×224 に引き伸ばし。縦画角 36.4°、床から 0.25 m(+ カーペット 1.8 cm)、俯角 11.8°、後ろへ 0.15 m。`measured-height-approx` の車体の一部(青いボンネット)を描く。記録は JPEG 品質 92 |
| ToF | 54×42、**放射距離** [mm](深度テクスチャから逆算)、縦 36.46°・横 56.16°、床から 0.20 m・俯角 16°。当たりなし = 0、上位ビットに信頼度(当たり 4、車体の自己反射 0)。noise の場面は画素ごとに一様雑音 |
| IMU | 合成。100 Hz の時刻に、そのフレームの値 [−v·ヨーレート, 前後の加速度, 9.80665, 0, 0, ヨーレート] を入れ、直近 10 個 |
| eRPM | 操作前の速さ × 2000 × (−1)(前進が負) |

時刻: フレーム k の観測(カメラ・ToF・IMU)は**操作前の状態 k**で作り、教師の操作 k も同じ状態 k から計算して、同じフレームで適用します。

ブラウザの違い: 作者の Mac(GPU)と Jetson(llvmpipe)で同じシーンを描くと、RGB は PSNR 37〜44 dB(ほぼ同じだが完全一致ではない)、
ToF は信頼度ビットが 99.9997% 一致・距離の差は中央値 0 mm・99% 点 1 mm(`webgl_probe/compare_*.json`)。

---

## 4. 操作から PWM ラベルへ(`dataset_recorder.py`)

PWM 換算(`action_to_pwm`、ゲインは掛けない):

- 操舵: f = clamp(δ / 0.52, −1, 1)、**操舵 µs = 中心 + f·(2000 − 中心)(f ≥ 0、左)/ 中心 + f·(中心 − 1000)(f < 0、右)**。中心 1442 なら左は 558 µs、右は 442 µs の幅(左右で非対称)
- スロットル(前進): **中立 − min(1, v/7.24)·(中立 − 1000)**。中立 1498 なら 4.5 m/s → 1188.5 µs、1 m/s → 1429.2 µs、7.24 m/s → 1000 µs。後退は中立 + min(1, −v/0.8)·(2000 − 中立)
- CH3 は常に 1500、eRPM は上のとおり
- 換算の値の出どころ: 接続したモデルの `calibration`・`model_config` があればそれ、無ければ既定(最高 = 画面の最高速度、中心 1441・中立 1496)

記録(`finish`): `vehicle[k]` = [操舵 µs, スロットル µs, 1500, eRPM]、**`future_actions[k][j] = vehicle[k+1+j]`(j = 0..9)**、
最後の 10 フレームは `future_valid = false`(1 シーン 600 フレームのうち学習に使うのは 590)、`mode` = 8、`synthetic` = true、`human_control` = false。
つまり**観測 k に対する正解は「次のフレームからの 10 個の操作」**で、操作 k 自体は正解に入りません(作者が走行時に「先読み 1・制御遅延 1 フレーム」を使うのはこのため)。

JetFormer との比較:

| | 操舵の中心 | スロットルの中立 | 出どころ |
|---|---|---|---|
| 実機 ai_mix | 1442.0 | 1498.0 | 止まっているフレーム 6,548 の中央値 |
| 実機 clean_driving | 1441.0 | 1497.0 | 同 1,153 フレーム |
| 元の 20 分データの換算 | 1442 | 1498 | 接続していたモデルの `model_config` |
| 収集器の既定(未接続) | 1441 | 1496 | `DEFAULT_MODEL` |
| **20 分データだけを JetFormer に学習させたときの値** | **1375.3** | **1332.9** | 各シーンの最初の 1 フレーム(計 60)の中央値 — 誤り |

---

## 5. RL の方策でアプリを走らせる方法

### 5.1 比べた案

| 案 | 内容 | 判定 |
|---|---|---|
| **A. 教師の差し込み口を差し替える**(採用) | `rl.html` の import map で `./rc-speed-controller.js` → `./rl/<自分のモジュール>.js`(`./rl/` の中からは元のファイルを読めるよう scopes で戻す)。自分のモジュールは元の名前をすべて再エクスポートし、`createRcTrainingTeacher` と `assertRcTeacherContract` だけ置き換える | 収集器・集約・ZIP は元のまま、controller は `demo`、ラベルは `synthetic_controller` で 20 分データと同じ。方策は正解の状態(snapshot)とコースの形を使える。**ブラウザで確認済み** |
| B. 「学習済みAI」(controller `external`)としてブリッジの `/api/predict` に Python の方策を置く | ブリッジの `--policy` の拡張点 | 集約は controller `demo` しか受け付けないので 1 つの ZIP にできない。観測に位置・向き・ゲート番号が無い(カメラ・ToF・IMU・速さ・舵角と 13 本の距離だけ)。HTTP の往復が増える。不採用 |
| C. 参照軌跡だけ差し替える | RL で「経路と速度の計画」を作り、元の `createPeriodicRcTeacher(world, config, base, options)`(export されている)に渡して元の追従器で走らせる | A の中でできる(同じ差し込み口)。ただし操作を決めるのは元の追従器で、RL の方策そのものではない。経路を RL で最適化する場合の選択肢として残す |
| D. Node で計算した操作列をそのまま流す | 開ループの再生 | ブラウザと Node の物理の差は 1e-13 程度で、20 秒なら一致するが、閉ループ(A)のほうが壊れにくい。A の確認(5.4)にこの再生を使う |

### 5.2 差し込み口の約束(`app.js` から読み取ったもの)

- `createRcTrainingTeacher(world, config, variation)` が **`{action, reference, options, metadata}`** を返すこと。
  - `world`: その回のコース(開始ゲートで回転した `route`・開始姿勢 `spawn`・線分 51・柱 94・路面・黒シート)。`config`: 正規化済みの物理設定。
  - `action(snapshot)` → `{steering [rad], speed [m/s]}`。`snapshot` は `{state: {x, z, yaw, speed, steering, sim_time, frame, surface, terminal, …}, metrics: {route_index, completed_laps, current_lap_passed_gates, …}, config}`。
    アプリはこの操作をそのフレームで `sim.step` に渡し、記録では ±0.52 rad・±7.24 m/s に丸めてから PWM にする。
  - `reference.length` と `reference.planned_time_s` が無いと、リセットで例外になる(表示・記録用。中身は任意)。
  - `metadata` は `context.teacher` に展開され、`session.json`・集約の契約に入る。**全シーンで同じ**にし、小さく(収集開始の要求は 2 MiB まで。元の RC 教師は 4 周ぶんの参照 約 1.1 MiB を入れている)。
  - リセットのたびに新しく作られる(1 シーンで 2 回: 計画の適用時と収集開始時)。方策の内部状態はここで初期化されるので、**決定的**(乱数・時刻を使わない)にすれば Node で同じ操作を再現できる。
  - 単周モードで initial-state(動的な開始)を使うなら `seek(snapshot)` も要る。連続周回では使えない。
- `assertRcTeacherContract(world, config, variation)`: 計画の検査とリセットで呼ばれる。自分のモジュールで、
  コース・物理などは元の検査に「標準の開始姿勢」で渡して確かめ、**開始の範囲だけ自分で広げる**(自己テストでは左右 ±0.3 m・±25°)。
- `RC_COLLECTION_PLAN` などその他の名前は元のまま(「20分・60シーンの収集条件を用意」はそのまま使える)。

### 5.3 RL の方策をどう入れるか(後段への提案)

- 方策は **JavaScript の関数**として入れる。重みは JS モジュール(`export const WEIGHTS = …`)にして、ブラウザと Node の両方で同じファイルを import する(fetch とファイル読み込みの違いを避ける)。
- 観測の特徴量(経路上の位置・先の曲率・障害物までの距離など)も**共有の ESM ファイル**で計算し、Node の RL 環境とブラウザで同じコードを使う。
  学習を Python で行うなら、環境(特徴量の計算を含む)を Node 側に置くか、Python に移す場合は一致の試験を付ける。
- 速度の指令は 0〜4.5 m/s(作者の教師と同じ上限)にするのがよい。舵は ±0.52 rad。
- 開始は、ゲート 0 以外・大きなずれからも走れるように鍛えると、立て直しのデータも作れる(連続周回はゲート k からも始められる。経路はアプリが回転する)。
- 物理を速く回すには `sim/fast-sim-core.mjs`(元のコアを読み込み時に文字列置換。近くの障害物だけを判定。101 例で軌跡が完全一致、約 13,000 ステップ/秒、元は 1,700)。
  最終の確認は必ず元の `sim-core.js` で行う。

### 5.4 自己テストの結果(`webgl_probe/selftest/`)

| シーン | 開始ずれ | 場面 | 結果 | 周回 | 再生の最大差 | 方策の再計算の最大差 |
|---|---|---|---|---|---|---|
| selftest-train-00 | +0.10 m・+6° | baseline | 600 フレーム・衝突 0 | 9.198 / 8.795 秒 | 8.9e-15 | 5.2e-14 |
| selftest-val-01 | −0.08 m・−5° | noise | 600 フレーム・衝突 0 | 9.198 / 8.794 秒 | 7.5e-15 | 3.0e-14 |

- 集約 ZIP `sim/runtime/datasets/7d1bf71091204a979434a6de635513ed/rl-wrapper-selftest_20261013-072333_7d1bf710.zip`(2 シーン・1,200 フレーム・学習 1,180・train 1 / val 1・21.1 MiB、集約は 5 秒)。
  `check_collection_zip.py` で: マニフェストの sha256 が全 2,429 ファイルで一致、NPZ のキー・型・形が元と同じ、ジャンル 8・num_modes 9、未来ラベル・最後の 10 フレームの除外・合成フラグが正しい、
  **PWM の再計算の差 0 µs**(中心 1442・中立 1498)、データカードの項目は元のダウンロード ZIP と同じ(取り込み済みフォルダーにある `scene_log`・`tags` は JetFormer の取り込みが足したもの)。
  集約の契約の教師は `rl-wrapper-selftest-v1` になっている(元は `rc-response-teacher-v1`)。**この ZIP は仕組みの確認用で、学習には使わない。JetFormer には取り込んでいない。**
- 元の RC 教師で 1 シーン(rc20-mode8-train-00 と同じ条件)を Jetson で記録し、取り込み済みの元データと比べると、`vehicle`・`future_actions`・IMU・時刻が**完全一致**(換算をそろえた場合)。

---

## 6. WebGL(ヘッドレス Firefox ほか)

| 方法 | WebGL | 備考 |
|---|---|---|
| ヘッドレス(既定) | × | 「WebGL 2を利用できません」 |
| ヘッドレス + `webgl.force-enabled` など | × | |
| ヘッドレス + `LIBGL_ALWAYS_SOFTWARE=1`・`GALLIUM_DRIVER=llvmpipe` | × | |
| ヘッドレス + ソフトウェア WebRender・`EGL_PLATFORM=surfaceless` | × | |
| **DISPLAY=:0 に窓を開く** | **○ WebGL 2**(レンダラー「llvmpipe, or similar」) | X は NVIDIA の OpenGL(glxinfo: Tegra Orin)だが、snap の Firefox は自分の Mesa(mesa-2404)を使うため CPU 描画になる |
| Xvfb | 未導入 | 入れても llvmpipe なので速くならない見込み(試していない) |

:0 での速さ(窓の中身 576×359): 画面の描画 17〜20 回/秒。収集は 600 フレームが約 60 秒(**約 10 フレーム/秒**)+ ZIP 作成 約 10 秒。
1 フレームの内訳: フレーム送信(`/api/datasets/frame`)の往復 中央値 14 ms(90% 点 17 ms)、カメラ画像の PNG 化 3.8 ms、残りの約 80 ms は CPU の WebGL 描画。
メモリ: Firefox のスコープ 約 1.25 GB、収集中のシステムの空きは 4.6〜4.7 GB。**60 シーン(20 分)で約 70 分**。収集中は Jetson の画面に Firefox の窓が出ます。

**メモリ上限の落とし穴**: `jf-heavy` は自分の systemd スコープに上限を付けるが、snap の Firefox(geckodriver 経由)は起動時に
`snap.firefox.geckodriver-<uuid>.scope` へ移るので**上限の外に出る**(確認済み)。`browser_collect.py` は起動直後にそのスコープへ
`systemctl --user set-property --runtime <scope> MemoryMax=2500M MemorySwapMax=0` を付ける(`--firefox-mem`)。ほかのワークフローの Firefox も同じです。

---

## 7. 作ったもの(後段で使う)

コード(`/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/`):

| ファイル | 役目 |
|---|---|
| `install_wrapper.py <module.js>` | コピーの dist が元とバイト単位で同じかを確かめ、`web/*.js` を `dist/rl/` に置き、`rl.html` を作る |
| `web/rl-teacher-selftest.js` | 差し込み口の見本(約束をコメントに記載)。元の RC 教師 + 速度 ×0.9、開始の範囲を広げる |
| `web/rc-teacher-probe.js` | 何もしない差し替え(読み込みの確認用) |
| `run_bridge.py --port 8791..8799` | コピーのブリッジを起動(`python -I`)。`--policy calibration_only_policy.py` で PWM 換算を元データと同じに |
| `calibration_only_policy.py` + `id8_original_calibration.json` | 走らないポリシー(元の 20 分データの収集時の換算 1442/1498・7.24 m/s・0.52 rad を渡すだけ) |
| `browser_collect.py` | Firefox(:0)で計画を連続試験し、収集・集約まで行う。時間の内訳も記録。snap のスコープにメモリ上限 |
| `webgl_probe.py` | WebGL の試験(上の表) |
| `verify_recording.mjs <シーン> [--teacher module.js]` | 記録した操作を元の物理で再生して照合、方策を計算し直して照合(どの教師でも使える) |
| `lib/app_emulation.mjs` | アプリの reset と同じ手順で Node にコース・設定・開始姿勢・教師を作る |
| `check_collection_zip.py <ZIP>` | 集約 ZIP が元の 20 分データと同じ形式か・PWM の再計算 |
| `compare_probe_scene.py` | 記録した 1 シーンと元データの同じシーンの比較 |
| `sim/fast-sim-core.mjs`・`probe/fast_equivalence.mjs` | RL 用の高速な物理(元と一致を確認) |
| `probe/teacher_reference_stats.mjs`・`probe/selftest_precheck.mjs`・`probe/teacher_splits.mjs`・`probe/headless_probe.mjs` | 教師の数字・自己テストの事前確認・ゲートごとの時刻・Node での一式の確認 |

結果(`/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/`): `probe_result.json`(Node の確認一式)、`fast_equivalence.json`、
`teacher_reference_stats.json`、`webgl_probe/<方法>/result.json`・スクリーンショット、`webgl_probe/selftest/`(`result.json`・`replay_*.json`・`collection_check.json`)、
`webgl_probe/compare_*.json`(元データとの比較)。

使い方(RL の方策ができたら):

```bash
source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
cd ~/jetformer/experiments/20261013_sim_rl_bestlap
python -I run_bridge.py --port 8791 --no-browser --policy calibration_only_policy.py &   # 終わったら止める
python -I install_wrapper.py rl-teacher-<名前>.js
jf-heavy --mem 2500M --lock big -- python -I browser_collect.py --plan <60条件.json> --tag <名前> --collection-name <データ名>
node verify_recording.mjs <シーンのフォルダー> --teacher ../../../jetformer_reports/sim_rl_bestlap_20261013/sim/dist/rl/rl-teacher-<名前>.js
python -I check_collection_zip.py <集約ZIP>
```

---

## 8. 後段への注意

- 学習: JetFormer に取り込むとき(`dataset_import.import_archive(zip, simulator=True)`)はジャンル 8・9 ジャンル。シミュレーターだけで学習したら、
  **`model_config` の `steer_center_us`/`throttle_neutral_us` を 1442/1498 に直す**(4 章)。シミュレーターで評価するなら作者と同じ「先読み 1・制御遅延 1」と `simulator_*` の項目を付ける。
- 比較の物差し: RL 教師のラップも学生のラップも、元の `sim-core.js` で再生して確かめる(`verify_recording.mjs`、学習済みモデルはユーザーの `qa/verify-learned-lap.mjs`)。
- 教師の metadata(`context.teacher`)に RL の識別(id・方策の sha256・行動の上限・学習条件の要約)を入れ、全シーンで同じにする。
- 収集は Firefox を 1 つずつ。ブリッジは 8791〜8799。収集が終わったら Firefox・geckodriver・ブリッジを止める(この段で起動したものは止めました)。
- このデータはすべて「この仮想コース・この暫定の物理」の中のもの。RGB は作者の環境の描画とわずかに違う(PSNR 37〜44 dB)。
