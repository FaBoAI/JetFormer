# 実装の段の報告 — 強化学習モードの仕組み(sim_rl_modes_20261013)

- 作成 2026-10-13 23:30。段の役目: `design.md` を `jetformer/experiments/20261013_sim_rl_modes/` に実装する(環境の拡張・モードの設定・学習の入口・
  モードごとの試験・ブラウザ向けの方策の書き出し・試験)。各モードの環境を短い試しの学習で動かし、試験のコードを最速の方策で確かめる。
- 依頼(ユーザー、21:20):「シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども作成できるように。
  強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意」
- 守ったこと: ベストラップの作業のファイル・ユーザーのシミュレーター・JetFormer 本体・ブリッジ・jetformer-web には触れていない(写すか import で読むだけ)。
  実車・PCA9685・I2C・GPU・Firefox は使っていない。RL の学習はすべて `jf-heavy --mem 2G --lock rl -- nice -n 15`、Node の試験は
  `jf-heavy --mem 1500M --lock none -- nice -n 15`、pytest は `jf-heavy --mem 2500M --lock big`。git にコミットしていない。
  `protocol.json` は変えていない(`sha256sum -c protocol.sha256` は OK)。決まりと違うことは `deviations.md` に 1 件書いた(5.4 節)。

---

## 0. 要点

1. **6 モードの仕組みを全部作った**。走行中の仕組みは 1 つの純粋な ESM の組 `rl/runtime/`(`mode_runtime.mjs` が入口)にまとめ、
   **Node の試験とブラウザの教師で同じファイル**を使う。観測の 79 次元は features.mjs(ベストラップとバイト単位で同じ写し)のまま、後ろに足す。
2. **試験した走り = 記録する走り**を確かめた: `mode_runtime.mjs` で最速の方策を元の sim-core.js で走らせると、停止から 1 周が
   **8.091953367107205 秒**(ベストラップの確認 r1 の値と同じ)、REC60 **60/60** 有効(F2・F3 合格。F4 はブラウザで判定する段)。
   600 フレームの snapshot の列(F-ID)を書き出し、Node 同士の照合は差 0。
3. **学習の入口**(`rl/train_modes.py` + `rl/run_mode.sh`)は 4 本(no_wall・recover・cond = 経路と速さ・obstacle)。100 万ステップごとのチェックポイント、
   50 万ステップごとの `last.pt`、止める合図で区切って保存、同じコマンドで続きから。初期化元の 79 次元の重みは 1 層目を 0 で広げる(初めは同じ走り。試験で確認)。
4. **試しの学習**(各 50 万ステップ、各 2〜3 分)で、4 本とも環境・学習・書き出し・検証の組の試験が通しで動いた。no_wall は 50 万ステップで
   20 秒を走り切る回が 0 → 62%、cond(速さ)は**検証の組で S3・S4・S5 がもう合格**(c_v 0.5〜1.0 で 1 周 8.2〜13.3 秒)。
   recover は lock rl の順番待ち(ベストラップの作業の r3 の学習)のあとに走らせ、途中で止めて続きから動くことを確かめた(7.3 節)。
5. **モードごとの試験**(`rl/eval_modes.mjs`)は protocol の試験の組をそのまま作る(no_wall 340 回・recover 100 + 148 + 1 + 60 回・line 60 + 6・speed 60 + 3・
   obstacle 100 + 1。回の数は試験で確認)。最速の方策で、protocol が「報告」に挙げた比べる値を出した(7.4 節): 離脱復活の試験で 有効 30/100・出来事の成功 75%、
   障害物よけの試験で 0/100(障害物はいつもの線をふさぐ位置にある)、壁にぶつからないの試験(シールドなし)で 衝突なし 205/340・ρ_req > 1 が 37% のフレーム。
   **学習で直す必要があることが数字で出ている**。
6. **ばらつきの物差し**(`rl/variation_metrics.mjs`)は VM1〜VM7。教師 REC60 で **VM1 = 7.58 µs**(protocol の確かめの値 7.58 ± 0.05)・VM3 0.0039 m を再現(試験)。
7. **試験**: Node 23 本・pytest 14 本、すべて合格。
8. **気をつけること**: 安全シールドは、壁から 5.7 cm を最高速で走る最速の方策と組むと、衝突・停止を防げない例がある(5.4 節)。no_wall の方策は
   すき間の約束で学習するので出番は少ない見込みだが、N1・N5・N7 は試験の段で判定する。no_wall の試験の組はシールドのぶん重い(8 章に時間の見込み)。

---

## 1. ファイル

コードはすべて `/home/jetson/jetformer/experiments/20261013_sim_rl_modes/` の下。

| ファイル | 中身 |
|---|---|
| `make_snapshot.py` → `snapshot/`(28 ファイル)・`snapshot/SNAPSHOT.json` | ベストラップの作業のファイルの写し。protocol の `snapshot_sources` の sha256 と**全部一致**(違えば止まる)。`env_core.mjs` の相対の置き場所は元と同じ |
| `rl/runtime/`(ブラウザにも配る組) | `mode_runtime.mjs`(入口)・`obs_modes.mjs`・`shield.mjs`・`perturb.mjs`・`line_profiles.mjs`・`layouts.mjs`・`rng.mjs`・`catalogue.mjs`(modes.json の写し)・`room_table.mjs/.json`・`features.mjs`・`reference_loop.mjs/.json`・`policy_forward.mjs`(最後の 4 つはベストラップとバイト単位で同じ) |
| `rl/gen_catalogue.mjs` | modes.json(sha256 を protocol と照合)→ `runtime/catalogue.mjs` |
| `rl/build_room_table.mjs` | 使える幅の表(`room_table.json`、sha256 `69335861…8d4a`)。作り直しで同じ sha256(試験) |
| `rl/fast_core_modes.mjs` | 高速版の sim-core(ベストラップと同じ 3 か所の置き換え + **摩擦円**。`frictionCircle: true` の回だけ) |
| `rl/env_modes.mjs` | 学習の環境 `ModeEnv`(no_wall・recover・cond・obstacle・base) |
| `rl/env_worker_modes.mjs`・`rl/vec_env_modes.py`・`rl/ppo_modes.py`・`rl/train_modes.py`・`rl/run_mode.sh` | 学習(ワーカー・Python 側・PPO・入口・jf-heavy の包み) |
| `rl/episode.mjs` | 1 回を「アプリと同じ形」で走らせる(元の sim-core + `mode_runtime.mjs` + 物理のずれ・キック・配置 + すき間の測り直し) |
| `rl/eval_modes.mjs` | モードごとの合否の試験(protocol の試験の組と判定。検証の組・代わりの方法・F-ID の書き出しも) |
| `rl/variation_metrics.mjs`・`rl/build_nominal_line.mjs` | VM1〜VM7・乱しなしの線 |
| `rl/build_obstacle_sets.mjs` → `rl/obstacle_sets/{validation,test,recording}.json` | 障害物の配置の組(30・50・30) |
| `rl/nominal_line_fastest.json` | 最速の方策の乱しなしの線(配置を置く位置) |
| `rl/make_web_bundle.mjs` → `web/rl/`(+ `manifest.json`) | ブラウザに配る組(統合の段が `jetformer/sim/static/rl/` に置く) |
| `rl/publish_mode.py` → `bundle/<key>/` | 登録の一式(mode.json・policy.json・eval.json・status.json)。`--registry` で `data/sim_rl/modes/<key>/` |
| `rl/check_fid.mjs` | F-ID の照合(保存した snapshot の列を runtime に与えて操作を比べる) |
| `rl/predict_scenes.mjs` | 記録のプランのシーンを前もって走らせる(走り切らない開始を替えるため。操作列の sha256 も出す) |
| `rl/masking.py` | 学習から外す決まりの Python の参照実装(統合の段の取り込みが使う形。JS と同じ結果を試験) |
| `rl/smoke_chain.sh`・`rl/smoke_recover_resume.sh` | この段の試しの学習 |
| `tests/*.test.mjs`(Node 23 本)・`tests/test_*.py`(pytest 14 本) | 試験(7.1 節) |

結果は `/home/jetson/jetformer_reports/sim_rl_modes_20261013/` の下: `rl/smoke-*/`(試しの学習)・`rl/logs/`・`eval/fastest/`(F の試験・F-ID の列)・
`eval/{recover,obstacle,speed,no_wall}/`(最速の方策の比べる値)。

---

## 2. 写し(snapshot)

- `python make_snapshot.py`: protocol の `snapshot_sources` 18 ファイルの sha256 を確かめてから写し、RC 教師などに要る `sim/dist` の残りと
  `lib/app_emulation.mjs` を足した(28 ファイル)。元のパスと sha256 は `snapshot/SNAPSHOT.json`。
- 写しの環境で、ベストラップの評価(`snapshot/rl/eval_sets.mjs`、元のコア)が STD 単周 8.091953367107205 秒・フライング 7.656318984241537 秒を出すことを確かめた。
- ベストラップの作業はこの段の間も動いていた(22:40 から r3 の学習)。写しは 22:10 の版で、protocol の sha256 と同じ。

---

## 3. 走行中の仕組み(`rl/runtime/`、Node とブラウザで同じファイル)

### 3.1 `mode_runtime.mjs`

```js
const t = createModeTeacher(world, config, variation, {mode, weights, params, policySha256, obstacleSet});
t.action(snapshot)   // → {steering, speed, info: {jf: {v: 1, m, f, ph, l, ev, cl, es?, sh?, p?}}}
t.metadata           // context.teacher(モードの中の全シーンで同じ)
t.reference          // {length: 26.12, planned_time_s}
```

- `world` はアプリの activeWorld(開始ゲートで回した経路)。v3 の経路を回したものでなければ例外(RL の観測は v3 専用)。障害物よけ以外は
  kind `obstacle` の障害物があると例外。
- 重みは `rl-bestlap-policy/v1`(最速)と `jf-sim-rl-mode-policy/v1`(この作業の書き出し)の両方を読む。policy_forward.mjs はそのまま使い、
  schema だけ読み替える。重みの `obs_dim` がモードの観測の包みと違えば例外。
- **同じフレームで 2 回呼ばれても同じ答え**(フレーム番号で覚える)。フレームが飛べば例外。乱数は `config.seed` からだけ。
- `info.jf`: `ph` = drive | perturb | shield、`l` = 学習に使う 1 / 使わない 0、`ev` = 乱しの番号、`cl` = d_min [mm]、`es` = 目標の横ずれ [mm](経路・速さ)、
  `sh` = シールドが替えた 1/0(no_wall)。フレーム 0 だけ `p`(シーンの条件の全部 + 方策の sha256 の先頭 12 桁 + runtime の版)。
  `steering_pwm`・`throttle_pwm` の鍵と `/` で始まる文字列は使わない(試験)。1 フレーム 400 バイト未満(試験)。
- `metadata` = `{schema: 'jetformer-sim-rl-teacher/v1', id: 'jf-rl-<key>-v1', mode, policy_sha256, obs_wrapper, features_version,
  reference_loop_sha256, speed_cap_mps, runtime_version, catalogue_sha256, room_table_sha256, obstacle_set}`(+ no_wall はシールドの値)。
  2 つの種で同じことを全モードで試験した。
- シーンの条件 `sceneParams(key, seed, params)`: プランの指定があればそのまま、無ければ種から決定的に(経路いろいろは family_plan_15 の割合、速さいろいろは 5 段)。
- Node の試験だけの上書き `overrides`: `{shield: false}`(NW-OFF)・`{perturbation: false}`(乱しなしの線・RC-KICK)・`{allow_obstacles: true}`(OB-BASE)。
  使ったときは `metadata.overrides_node_test_only` に残る。ブラウザでは使わない。
- 速さいろいろの代わりの方法(学習しない上限)は `params.speed_limit`(どのモードでも)。

### 3.2 観測の包み(`obs_modes.mjs`)

- base79(最速・壁にぶつからない)/ base79v(離脱復活)/ cond97(経路・速さ)/ obst95(障害物よけ)。79 の部分は features.mjs そのもの(試験で一致)。
- 仮想ゲート(base79v): R 上の s が目標ゲートの s + 0.30 m を越えても sim の route_index が進まなければ「外した」とし、s から決めた次のゲートを観測に使う。
  **戻り方(設計に書かれていなかったので決めた)**: 外したあと、sim の route_index が進んで仮想の目標と同じゲートになったら sim の値に戻る。外していなければ base79 と同じ値(試験)。
- cond97: 先読み 16 点の e*/0.5、(e_y − e*)/0.5、c_v。obst95: 足した障害物の近い 4 個(3 m・±100°)の 前/3・左/3・半径/0.2・有無。箱は芯の線分の一番近い点。

### 3.3 安全シールド(`shield.mjs`)

- 車の式は sim-core の step() の写し(小区間・舵の速さ・加減速・ヨーレートの頭打ち)。**600 フレームの操作列で sim-core との差 ≤ 1e-6 m**(試験。実際は 0 に近い)。
- 判定は modes.json のとおり(提案 1 フレーム + 予備の操縦 24 フレーム、標準 ≥ 7 cm・悲観 ≥ 5 cm、候補 11 個)。すき間は features.mjs の正確な距離。
  計算を減らす工夫: フレームの両端のすき間と車体の点の動ける距離の上限から、しきい値を割りえないフレームは小区間を調べない(判定は変わらない)。
- 決定性(同じ入力に同じ答え・作り直しても同じ)を試験。止まって広い所のふつうの指令は替えない(試験)。

### 3.4 乱し(`perturb.mjs`)

- 記録の決まり(舵のパルス 45%・わざと外す 40%・ブレーキ/踏み込み 15%、弱・中・強、最初 2〜4 秒・あけて 2〜3.5 秒・18.5 秒以降は始めない)と
  学習の決まり(+ 状態のキック 25%、最初 1〜4 秒・あけて 1.5〜4 秒、幅の条件 0.45 m・黒シートだけ除く、強さの倍率)。
- 出来事ごとに一様乱数を 12 個(いつも同じ数)引くので、同じ snapshot の列なら Node とブラウザで同じ予定になる(試験: 同じ種で同じ、違う種で違う)。
- 始めてよい位置(読み方): s から先 2.0 m の**区間の全部**が「すき間 5 cm で動ける幅 ≥ 0.60 m」で、黒シート(s 5.9〜8.0)・奥の直線(s 14.6〜19.3)に**かからない**。
  R の上の 59% の位置で始められる。試験で、黒シート・奥の直線の中で始まらないことを確かめた。
- 強さの指定は `auto`(出来事ごとに 3 つから等確率)・1 つ・配列(`['弱', '中']` = protocol の代わりの方法)。
- 学習から外す決まり `maskFromLearn`(JS)と `rl/masking.py`(Python)は同じ結果(試験)。区間 [ps, pe) で k ∈ [ps − 10, pe − 2](試験)。

### 3.5 経路の形と使える幅の表(`line_profiles.mjs`・`room_table`)

- 使える幅の表: 522 点(5.0 cm おき)。すき間 ≥ 8 cm・ゲートの窓(±0.53 m)・黒シートの車幅で切り、±0.5 m の最小フィルター → σ 0.4 m → 3 cm 引く。
  取れる最大のすき間の最小は **0.1417 m**(理解の段の 14 cm と同じ)。R が自由でない(すき間 8 cm 未満の)点が 24 か所あり、そこでは区間が R の片側だけになる。
- e*(s) は設計の式のあと、**使える区間 [−ρ_R, ρ_L] に切る**(R が自由でない所で c(s)·ρ が区間の外になるのを防ぐ。設計の式の端の扱いを決めたもの)。
- 形: const・inside・outside・late_apex・early_apex・random(自然 3 次スプライン、開始から 2.5 周分)。どの形でも e* が区間の中・|c| ≤ 1・イン = −アウト(試験)。

### 3.6 障害物の配置(`layouts.mjs`)

- 柱(半径 0.05〜0.10 m)と箱(カプセル、芯の長さ 0.15〜0.35 m・半径 0.05 m)を 1〜4 個、s で 2 m 以上離して、最速の方策のふだんの線 ±0.15 m に置く。
  黒シート・奥の直線(±0.3 m の余白)・開始の前後 1.5 m(+ 0.3 m)・ゲートの柱から 0.3 m 以内・コースの障害物から 5 cm 以内には置かない。
- 通り道の検査器: 障害物の s ±0.6 m の全部の点で、すき間 ≥ 8 cm で通れる横の幅 ≥ 0.10 m がゲートの窓(R から ±0.65 m)の中に残ること。
  道をふさぐ配置を拒否し、ふつうの配置を通すことを試験した。組(validation 30・test 50・recording 30)は全部、検査器(格子を使わない遅い方)でも通る。
- 形は `prepareGeometry(worldWithObstacles(...))` と同じ中身をキャッシュを通さずに作る(試験で一致)。回ごとに配置を変えてもメモリが増えない。
- test・validation の組は STD の開始の前後 1.5 m にも置かない(OB-T100 はどの配置も STD から走るため)。

---

## 4. 学習

### 4.1 環境(`env_modes.mjs` の `ModeEnv`)

| run | 観測 | 報酬・終了(modes.json の値。tuning で 1 回だけ変えられる) | 回ごとに引くもの |
|---|---|---|---|
| base | 79 | ベストラップの v2-gatecap(**RlEnv と観測・報酬・終了がビット単位で同じ**。試験) | 開始 |
| no_wall | 79 | 進み + 終了の罰、なめらかさ 0.06・0.03、−0.2·max(0, ρ_req − 0.9)、−0.2·max(0, ρ_comb − 0.9)、すき間の壁(12 cm から、上限 4·0.25)、**d_min < 7.5 cm で終了 −10** | DR-NW-train(標準 15%・遅延 0/1/2 = 4:4.5:1.5・摩擦円 30%・舵の白色雑音 σ 0.03 rad 50%)・開始 2 成分(半分は開始のすき間 ≥ 12 cm) |
| recover | 79(仮想ゲート) | v2-gatecap、ゲートを外すと −3 で続ける、衝突・停止・逆走・黒シート違反は −10 で終了 | 乱し(学習の決まり)。強さは 0.5 → 1.2 倍(最初の 300 万ステップ)。**差し込んだ行動のステップは injected = 1** |
| cond | 97 | v2-gatecap − 0.08·min(1, ((e_y − e*)/0.10)²)、速さの上限 4.5·c_v | 経路の形(families_p)・c_v(40% は 1.0、60% は U(0.5, 1.0)) |
| obstacle | 95 | v2-gatecap − 0.1·min(4, ((0.08 − d_obs)/0.04)²) | 配置(10% は無し。開始の s を避ける) |

- 物理のずれの約束は r2 と同じ: 観測は標準の物理の値、遅延 n は n フレーム前の指令(最初は中立)、観測の「前回の行動」は適用した行動、ゲインは適用する指令に掛ける。
  摩擦円は `fast_core_modes.mjs` の置き換えで、**own_physics_r1 の frictionCircle と 1e-9 以内で同じ**(ふつうの操作と乱暴な操作の 2 本で試験)。
  摩擦円なしは、ベストラップの高速版・元のコアと軌跡・集計がビット単位で同じ(試験)。
- 状態のキック(recover の学習だけ): 高速版の `peek().state` を書き換える。キックのあとのすき間が 5 cm 未満なら戻す。
- 1 環境の速さ(1 プロセス、方策の計算込み): base・no_wall・recover・cond 約 2,800〜3,300 steps/s、obstacle は配置の生成のぶん遅い(下の学習の速さ)。

### 4.2 学習の入口(`train_modes.py`・`run_mode.sh`)

```bash
cd ~/jetformer/experiments/20261013_sim_rl_modes
rl/run_mode.sh <名前> <no_wall|recover|cond|obstacle> <ステップ数> [--eval] [--tuning '{"clearance_barrier_w": 0.3}'] [--seed N] [--init r2_dr_5M|main_s1_10M|<.pt>]
#   = jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 python -u rl/train_modes.py --run … --out ~/jetformer_reports/sim_rl_modes_20261013/rl/<名前>
```

- 既定: 種(no_wall 21・recover 31・cond 41・obstacle 61)、初期化元(no_wall は r2 の 500 万 `step_005013504.pt`、ほかは main 種 1 の 1,000 万)、
  lr 1e-4 → 3e-5、log σ は初期化元のまま、PPO の値はベストラップと同じ。
- 79 次元の重みを 95・97 次元へ広げるとき、actor・critic の 1 層目の新しい列は 0、観測の正規化は新しい入力を平均 0・分散 1(**初期化元と出力が同じ**。pytest)。
- 差し込んだ行動のステップは方策の損失・エントロピー・KL・アドバンテージの正規化から外し、価値の学習には使う(pytest: 外したステップのアドバンテージを
  10⁶ にしても方策の勾配が変わらない)。
- tuning は modes.json の `tuning_ranges` の中だけ(範囲の外・そのモードで変えられない値は止まる。pytest)。使った値は `run.json` に残る。続きのときに違う値を渡すと止まる。
- 出力: `run.json`・`progress.jsonl`・`last.pt`(50 万ごと・終わり・止める合図)・`checkpoints/step_N.pt` と `policies/step_N.json`(100 万ごと)・`done.json`。
  `--eval` を付けると、チェックポイントごとに検証の組(protocol selection、高速版)を裏で 1 本ずつ評価して `evals/<mode>/step_N.json` に書く(cond は line と speed の両方)。
- 書き出しの重み `jf-sim-rl-mode-policy/v1`: `obs_dim`・`obs_wrapper`・`mode`・正規化(固定値)・層・`meta`(run・種・ステップ・初期化元・tuning・protocol と modes.json の sha256)。
  JS(`mode_runtime.mjs` の読み方)と numpy の行動の差 < 1e-4(pytest)。

---

## 5. 試験(`eval_modes.mjs`)

### 5.1 使い方

```bash
node rl/eval_modes.mjs --mode <key> --policy <policy.json> [--set test|validation] [--core original|fast] [--jobs N] [--only 試験,…] --out <結果.json>
#   代わりの方法: --shield-c 0.09(no_wall)・--classes 弱,中(recover)・--speed-caps --run-mode fastest|no_wall(speed)
#                 --run-mode fastest --policy-list a.json,b.json,…(line の線の混ぜ合わせ。L1・L2・L5 だけ判定)
#   報告用: --run-mode fastest(別の方策で同じ組。例 OB-BASE)・--no-shield(NW-OFF の形)・--fid <出力>(F-ID の列)
```

- 判定の数字は元の sim-core.js(`--core original`、試験の組の既定)。摩擦円は高速版の写し、状態のキック(RC-KICK)は高速版(protocol の例外のとおり)。
- すき間は、保存した操作列を lap_audit.mjs の写し(摩擦円は own_physics_r1 の写し)で小区間ごとに測り直した値。自前の積分と sim の軌跡の差も残す(`audit_err`。最速の F の試験で 0)。
- `--jobs N` は回を N 個の子プロセスに分ける(回の並びは決まっていて、結果は 1 本と同じ)。結果の JSON は判定(judged)・報告(reported)・回ごとの行・物差しの参考値。

### 5.2 試験の組(回の数は試験で確認)

| モード | 組 | 回 |
|---|---|---|
| fastest | F-STD(単周・連続)・F-REC60 | 2 + 60(+ F-ID の列) |
| no_wall | NW-N60(REC60 の偶数 30 + Halton 30)・NW-STD・NW-G(範囲の中 34 行 × 2)・NW-R200・NW-FC10 = **340**、報告: NW-G の範囲の外 12・NW-OFF 260 | 612 |
| recover | RC-P100・RC-KICK(37 時刻 × 左右 × 2 種類 = 148、判定は始めてよい位置の回)・RC-STD・RC-REC60 | 309 |
| line | LN-P60(family_plan_15 × 4)・LN-FAM6 | 66 |
| speed | SP-P60(c_v 5 段 × 12)・SP-STD3 | 63 |
| obstacle | OB-T100(試験の 50 配置 × {STD, Halton})・OB-EMPTY | 101 |

検証の組(チェックポイントを選ぶ): 種 validation_base・Halton 10,001〜・回の数は試験の 1/4(切り上げ)。試験の組と開始・種が重ならない(試験)。

### 5.3 protocol の読み方(実装で決めたこと。決まりに書かれていない所)

- NW-N60 の Halton 30 は no_wall の記録の開始の分布(開始のすき間 ≥ 12 cm)の試験の Halton の**有効な番号の最初の 30**、NW-R200 は**その次の 200**。
  種は N60 が test_base + i、R200 の物理は `mix32(test_base + 100 + i)` から DR-T を引く(R200 では舵の白色雑音は使わない。DR-T に無いため)。
- NW-FC10 は REC60 の 0〜9 番に摩擦円(物理は標準の値)。検証の組は REC60 の奇数番・11・13・15 番など、試験と重ならないものを使う。
- RC-P100 の乱しの予定は記録と同じ決まりで、種 = test_base + i(= シーンの `config.seed`)。「戻った」は同じ方策の乱しなしの線(8 回、乱しを止めて走らせる)との比べ。
  停止・逆走は学習の環境と同じ決まり(1 秒以降に 0.2 m/s 未満が 60 フレーム、s が最大から 1 m 戻る)。
- LN-P60 の形は family_plan_15 の並び(左 2・右 2・中央 1・イン 2・アウト 2・レイト 2・アーリー 1・ランダム 3)を 4 回、強さ・c0・c_v は種から。
- L4 の「6 回ともそろう 5 cm のマス」は、開始から 3 m 以降の最初の通過のマス(絶対の s)。「1 m 以上続けて」は 20 マスの移動平均の最大。
- ばらつきの物差しは試験の結果に参考として付けた(判定は記録の段)。VM2 は舵の PWM [µs]。

### 5.4 安全シールドで分かったこと(と 1 件の逸脱)

- 最速の方策(壁から 5.7 cm を最高速で走る、no_wall 用に学習していない方策)とシールドを組んで STD を走らせると、決まりのままでは
  (1) 全部の候補が不合格の場面で、予備の操縦つきの予想だけで選んだ指令が方策より悪く、衝突した、(2) 壁から 8 cm で止まった車が「速さ 0」の候補を選び続け、
  20 秒止まったままになった。→ `deviations.md` 1 のとおり、全部不合格のときは方策の提案も同じ物差しで並べ、止まってすくむときは予備の操縦の舵に近い候補を選ぶようにした。
- 直したあとも、最速の方策では開始 4 つ(ゲート 0・8・20・27)のうち 2 回で衝突、残りの 2 回も 222・364 フレーム止まっていた(すくみは減ったが無くならない)。理由は物理: ブレーキが 1.2 m/s² しかないので、4 m/s 以上で角に入ると
  0.8 秒先までの予備の操縦(ブレーキ + 中点の線を 0.6 m 先で追う)では曲がり切れない。最速の方策の 336 の状態から予備の操縦だけで走らせると、
  25 フレームすき間 ≥ 7 cm を保てたのは 307(速さを 0.75 倍にした状態からなら 333)。
  → シールドは「遅めに走る no_wall の方策の、まれなはみ出し」を止める仕組みで、速い方策を安全にする仕組みではない。no_wall の試験(N1・N5・N7)で判定する。
- シールドの計算: 1 フレーム 1〜25 回の先読み(各 25 フレーム × 2 模型)。最速の方策では 1 回(600 フレーム)2〜25 秒(Node)。no_wall の方策は出番が少ない見込み。

---

## 6. ばらつきの物差し(`variation_metrics.mjs`)

- VM1(時刻でそろえた舵 PWM)・VM2〜VM4(絶対の s の 5 cm のマス、開始から 3 m 以降の最初の通過で、舵 PWM・横ずれ・速さ)・VM5(すき間)・VM6(乱しの数・外す割合・
  立て直しの時間)・VM7(s 0.25 m × e_y 5 cm × 向き 0.1 rad × 速さ 0.5 m/s のマスの数。9,000 フレームを種 12345 で間引く)。乱しなしの線 `nominalLine`。
- Node の走り(`sceneFromEpisode`)と、記録の姿勢の列(`sceneFromPoses`。R に射影)の両方から作れる。
- **教師 REC60 で VM1 = 7.58 µs(± 0.05 の中)、VM3 = 0.0039 m 前後**を Node の試験で確かめた(protocol variation_metrics.teacher_reference の確かめ)。

---

## 7. 確かめたこと

### 7.1 試験(全部合格)

- Node(`node --test tests/*.test.mjs`、23 本): 摩擦円なしの高速版 = ベストラップの高速版 = 元のコア(ビット単位)/ 摩擦円 = own_physics_r1(1e-9)/
  base の環境 = RlEnv(ビット単位)/ 観測の次元・差し込みの印・DR の割合・引く数が一定 / recover はゲートを外しても終わらない /
  使える幅の表の作り直しが同じ sha256 / 経路の形の範囲・決定性・イン = −アウト / 障害物の形・往復・入力・通り道の検査器・配置の決定性 /
  VM1 = 7.58 µs / 試験の組の回の数・検証と試験が重ならない / 乱しの決定性・始めてよい位置・学習除外の範囲・カリキュラム /
  最速 8.091953367107205 秒 / base79・base79v = features.mjs / runtime の約束(2 回呼び・p・禁止の鍵・大きさ・metadata が全モードで一定)/
  シーンの条件の決定性 / ブラウザの組がバイト単位で同じ・純粋な ESM / 車の式 ≤ 1e-6 m / シールドの決定性。
- pytest(14 本): 1 層目の 0 広げで出力が同じ(79・95・97)/ 書き出しの重みを JS と numpy で比べて < 1e-4 / 差し込んだステップが方策の勾配に効かない /
  tuning の範囲 / ワーカーと Python の形(4 run)/ 学習除外の Python と JS が同じ。

### 7.2 試しの学習(各 50 万ステップ、lock rl・nice 15・メモリ 2G)

| run | 初期化元 | 速さ(全体) | はじめ → 終わり(その更新の間に終わった回) | 検証の組(高速版、50 万の方策) |
|---|---|---|---|---|
| no_wall | r2 の 500 万 | 5,402 steps/s | 20 秒を走り切る 0% → 62%、終わりの理由 clearance 91 → 14、1 周目 9.41 秒、すき間の平均 8.4 cm | N4 合格(9.08 秒)、ほかは不合格(衝突なし 36/88・シールド 10%)。学習が足りないのは想定どおり |
| cond | main 種 1 | 5,213 | 走り切り 100%、1 周目 9.31 秒、追従の誤差の平均 0.15 m | line: L1 15/15・L6 合格、L2〜L5 不合格(追従 中央値 12.8 cm)。**speed: S2〜S5 合格**(S1 14/15 で 15 に 1 足りない) |
| obstacle | main 種 1 | 3,105 | 走り切り 0% → 12%、ほとんど足した障害物に衝突 | O1 0/26・O4 合格(8.16 秒) |
| recover | main 種 1 | 5,638(続きの回) | 20 秒を走り切る 29〜43%(乱しの強さ 0.5 → 0.61 倍に上がりながら)、出来事 3.5/回、差し込んだ行動のステップ 8%、ゲートの外し 0.58/回(終わらずに続く) | C3 合格(中央値 0.25 秒)・C5 向き 9/10・横 8/10・C6 合格(8.056 秒・REC60 15/15・6.2 cm)、C1 8/25・C2 76% |

- どの run も 25 万・50 万ステップのチェックポイントと方策の JSON を書き、`done.json` で終わった。obstacle の学習は 1 秒 3,100 ステップで、
  1,500 万ステップには約 80 分かかる(設計の見込み 55 分より長い)。

### 7.3 recover の試しの学習と、途中で止めて続ける確かめ

- recover は、ベストラップの作業の r3 の学習(lock rl)のあとに走らせた(23:05〜)。`rl/smoke_recover_resume.sh` で、15 回の更新(245,760 ステップ)のところで
  止める合図(SIGTERM)を送り、学習は区切りで `last.pt` を書いて終了コード 0 で終わった。同じコマンドで出し直すと(lock rl の順番待ちのあと 23:27)
  **「last.pt から再開します: 245,760 ステップ」**で続き、507,904 ステップで `done.json` を書いた。`run.json` の `resumes` に
  `{at: 23:27:05, steps: 245760, env_reset_seed: 31245853}` が残る。
- 検証の組(高速版)で、有効な回の出来事は平均 4.4 回・外すフレームの割合 0.18(決まりの 3.5 回以上・0.10〜0.40 の中)。有効でない回は乱しのあとの衝突で早く終わるので、
  全部の回の平均は 2.84 回になる(C4 は学習した方策で回が最後まで走れば満たす見込み)。最速の方策の試験の組でも、有効な回は 4.4 回・0.18。

### 7.4 最速の方策での試験(protocol が「報告」に挙げた比べる値。元のコア)

| 試験 | 結果 | 時間(--jobs 2) |
|---|---|---|
| fastest(F の試験、`eval/fastest/eval_test_build.json`) | **F2 合格(8.091953367107205 秒)・F3 合格(60/60)**、F4 はブラウザの段。フライング 7.656 秒 | 41 秒 |
| recover の試験の組(`eval/recover/baseline_fastest_test.json`) | C1 30/100・C2 成功 75.2%・C3 合格(中央値 0.40 秒・90% 点 1.02 秒)・C4 出来事 2.86/回・外す割合 0.112・C5 向き 33/40・横 35/40・C6 合格(8.092 秒・REC60 60/60・すき間 5.7 cm) | 119 秒 |
| obstacle の試験の組 = OB-BASE(`eval/obstacle/baseline_fastest_OB-BASE.json`) | O1 **0/100**(100 回とも足した障害物に衝突、中央値 56 フレーム目)・O4 合格(8.092 秒) | 12 秒 |
| speed の代わりの方法(最速 + 上限 4.5〜2.5 m/s、`eval/speed/fallback_caps_fastest_test.json`) | S1 60/60・S3・S4・S5 合格(VM4 0.42 m/s)、**S2 不合格(すき間 0.16 cm)** | 51 秒 |
| no_wall の組をシールドなし(`eval/no_wall/baseline_fastest_noshield_test.json`) | 衝突なし 205/340・N60 60/60・R200 95/200・すき間(標準)2.6 cm・ρ_req > 1 37%・ρ_comb > 1 37% | 176 秒 |

→ 最速の方策は「ぶつからない」「戻る」「よける」を学習していない(数字はそのとおり)。各モードの学習で直す対象がはっきりした。
速さいろいろは、学習しない代わりの方法ではすき間(S2)が足りない(理解の段の probe と同じ)。

代わりの方法の道も動かして確かめた(検証の組。判定には使わない):
- recover の強さ「弱・中」だけ(`--classes 弱,中`): 出来事 71 回(弱 34・中 37)、成功 弱 27・中 24(最速の方策)。
- line の線の混ぜ合わせ(`--run-mode fastest --policy-list` にベストラップのチェックポイント 3 つ): L1 14/15・L2 5.4 cm・**L5(VM3)0.054 m < 0.08**。
  L3・L4・L6 は「当てはまらない」になる。チェックポイントの混ぜ合わせでは線のばらつきが足りない見込みで、cond の学習が要る。

---

## 8. 次の段(学習・試験)へ

- 学習の順番(protocol budget): no_wall → recover → cond → obstacle、それぞれパイロット 100 万 → (1 回だけの調整)→ 本番。例:
  `rl/run_mode.sh no_wall-pilot no_wall 1000000 --eval` → `rl/run_mode.sh no_wall-main no_wall 10000000 --eval`。
- チェックポイントの選び方は protocol selection(検証の組で判定を全部満たす一番後のもの)。`evals/<mode>/step_N.json` の `all_judged_pass`。
- 試験の組は選んだ方策だけに使う: `node rl/eval_modes.mjs --mode no_wall --policy <選んだ方策> --set test --jobs 2 --out …`。
  no_wall の組(340 + 報告 272 回)はシールドのぶん重い(試しの方策の検証の組は高速版・--jobs 2 で 88 回 247 秒 = 1 回 約 5.6 秒の CPU。
  元のコアの試験の組では 1 回 5〜8 秒と見て、--jobs 2 で 30〜40 分)。Firefox や RL の学習と重ならない時間に回すのがよい。
- 登録の一式: `python rl/publish_mode.py --mode <key> --policy … --eval <試験の組の結果>`(既定は `bundle/<key>/`、`--registry` で登録簿)。
  最速は `bundle/fastest/` を作った(状態は「確認待ち」= F4 のブラウザの照合だけが残り)。
- 統合の段へ: `web/rl/`(manifest.json に sha256)を `jetformer/sim/static/rl/` に置き、teacher.js から `createModeTeacher` を呼ぶ。
  F-ID の列は `eval/fastest/fid_std.json`(`rl/check_fid.mjs` と同じ比べ方をブラウザで)。記録の前の予想は `rl/predict_scenes.mjs`、
  取り込みの学習除外は `rl/masking.py` の形(`frames.jsonl` の `action_info.jf.l`)。障害物の配置は `rl/obstacle_sets/*.json`(kind `obstacle` で world に入れる)。

---

## 9. 起動したプロセス

- 試しの学習(no_wall・cond・obstacle・recover)、検証の組と最速の方策の試験、Node の試験、pytest。**すべて終わっている**(段の終わりに `ps` で確認。
  次の段が続ける長い学習は無い)。ベストラップの作業のプロセス(r3 の学習)には触れていない(lock rl で順番待ちをしただけ)。
- この段のコードの sha256 は `build_sha256.txt`(`rl/`・`rl/runtime/`・`tests/` の全部)。
