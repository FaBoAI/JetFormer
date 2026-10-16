# 較正した物理をシミュレーターに当てた — ユーザーのファイルを変えずに(S2、2026-10-16)

- 依頼(10/16 18:00 ごろ):「今週は走行できないので、ほかにできることは?」のうち **「7 シミュレーターを実車に近づける」** の実装の段です。
  - S1 で固めたプロファイル(`profiles/P_cal.json`、sha256 `4b0e9348…`)を、シミュレーターに当てています。
- 実車は走らせていません。PCA9685・I2C には触れていません。drive.py は変えていません。
- ユーザーのシミュレーター(`data/sim/releases/…`・`~/sim`)は読むだけです。git commit はしていません。この段で起動したプロセスは、全部終わっています。
- 確かめの結果は次の 2 つです。
  - `fidelity.md`: 保留のシーンで、較正の前と後を比べた。
  - `g1_replay.md`: 安全層 v1 を、記録の再生で確かめた。

## 0. 結論

1. **較正版のコアを、読み込むときに差分を当てる形で作りました。** ユーザーの `sim-core.js`(sha256 `aaeb2379…549e`)は変えていません。
   - Node でもブラウザでも、同じ差分の関数・同じ実行部(`calib_runtime.mjs`)を使います。
   - **プロファイルが無ければ、元とビット単位で同じ動き**です。教師の REC60 の 60 条件・STD 2 本・ランダム操作 20 本で、report の JSON が完全に一致しました(試験 T1)。
2. **ブラウザ**: レポートのフォルダーにシミュレーターの写しを作り、`sim/dist/calib.html` を足しました。
   - 写しは、ユーザーの展開物の dist 33 ファイルとブリッジなど 5 ファイルが、sha256 で同じです。
   - index.html との差は、`<title>` と import map の 1 行(`"./sim-core.js": "./_calib/sim-core-jf-calib-v1-real-20261010.js"`)だけです。
   - app.js だけでなく、先生のモジュールも同じ較正版のコアを読みます。
3. **忠実度**: 保留のシーンで、速さ・ヨー・向き・位置の誤差が 27〜34% 減りました。ただし、偏り(C3)で手順の判定は「不合格」でした(fidelity.md)。
   - 設定欄だけの版 P_cfg も不合格です。成果は **P_cal(calib.html)** です。
4. **安全層 v1**: 純粋な関数 `jetformer/safety_caps.py` として作りました(drive.py にはつないでいません)。
   - 10/10 の記録の再生では、G1 の 6 つの決まりのうち 4 つが通り、2 つが落ちました(g1_replay.md)。
   - シミュレーター用の写し `governorStep`(JS)は、Python と同じ出力です(試験 T6)。
5. **試験はすべて通りました**: node 15 本(T1・T2・T4・T5・T6)、T3(Python と JS の一致、差 3e-13)、T4b、pytest 32 本(安全層 16 + S1 16)。

## 1. 使い方

**Node(上限の試験・評価のスクリプトから)**

```js
import {loadCalibratedCore, loadProfile} from '/home/jetson/jetformer/experiments/20261016_sim_calibration/calib_core.mjs';
const {profile, sha256} = loadProfile('P_cal');                       // sha256 を確かめて読む(違えば止まる)
const core = await loadCalibratedCore(profile, {fast: true, profileSha256: sha256});   // fast = 10/13 の高速化も重ねる
const sim = core.createSimulation(world, config);                     // 使い方は元の sim-core.js と同じ
// ブレーキの弱め・強め: loadCalibratedCore(profile, {brakeScenario: 'low' | 'high'})
// 電池の係数・上限の層: sim.calibControl({kBatt: 0.8}), sim.calibControl({filter: fn})
```

- 10/13 の RL の環境(`rl/env_core.mjs` の RlEnv)には、`core` としてそのまま渡せます。fast なら coreKind 'fast'、そうでなければ 'original' です。
- `report().model.calibration` に、プロファイルの名前・sha256・ブレーキのシナリオ・遅れが残ります。`snapshot().config` は元と同じ形です。

**ブラウザ(画面で見る・S-DIV を走らせる)**

- `/home/jetson/jetformer_reports/sim_calibration_20261016/sim/dist/calib.html` です。10/15 と同じ写しの bridge.py か、静的なサーバーで開きます。
  - `index.html` は元の物理のままです。並べて比べられます。
- 作り直すとき: `node build_browser_core.mjs`。写しの確かめ・較正版のコア・ページ・`_calib/MANIFEST.json` を書きます。
- この段では、ブラウザでの表示は試していません。ブラウザの S-DIV の確認は S5 です。
  - ブラウザ用のファイルそのものを Node で読み込み、教師の REC60 の 1 条件で、Node の較正版と同じ report になることは確かめました(試験 T4b)。

**JetFormer の画面に入れる道(製品にするとき。今回はしていない)**

- `jetformer/sim/web.py` の import map に、同じ 1 行を足す方法です。
- jetformer/sim の変更と jetformer-web の再起動が要るので、ユーザーの了承のあとにします。

**設定欄だけで近づける場合(P_cfg。合格はしていない)**

| 設定 | 今の値 | P_cfg |
|---|---|---|
| driveTau(駆動の時定数) | 0.40 | 1.03 |
| driveAccelerationCap(加速の上限) | 6.0 | 0.74 m/s²(速さの頭打ちの代わり。発進が遅くなりすぎる) |
| brakeDeceleration(惰性とブレーキの共通) | 1.2 | 0.71 |
| steerRate | 1.2 | 4.56 rad/s |
| maxSteer | 0.52 | 0.398 rad |
| frictionMultiplier | 1.0 | 1.09 |
| アプリの遅れ | 1 フレーム | 2 フレーム |

## 2. 仕組み

- **`calib_runtime.mjs`**(純粋な ESM、import なし)
  - 較正の物理 jf-calib-v1 を、step の小区間 1 つ分として計算します。中身は P_cal.json の semantics のとおりです。
    - 舵: 遅れ 50 ms → 左右で別の対応表(左いっぱい δ 0.402・右 0.440 rad)→ 舵の速さ 8 rad/s → μ g/|v| の頭打ち。
    - 前後: 速さの対応 T(u)(不感帯 0.43、頭打ち 5.0 m/s)× 電池 → 駆動(τ_up 0.32 秒、a_cap 4.1)/ 惰性 0.58 + こすれ 0.40 × |v r| / ESC のブレーキ 0.0105 × (off − 65 µs) / 止まったあとの後退の手順。
    - 路面の μ はプロファイルの値です(カーペット 0.82・ショートカット 0.70・芝 0.58・PTFE 0.21・青マット 0.66)。
  - 「元の式 + 純粋な遅れ」も持っています。忠実度の再生で P_default・P_cfg に使います(deviations D13)。
  - 上限の層 `governorStep`(jetformer/safety_caps.py の cap_step と同じ)も持っています。
- **`calib_core.mjs` の `patchSimCore(元の文字列, プロファイル, オプション)`**
  - 元の sha256 を確かめます。違えば止まります。
  - 次の 7 か所を、1 か所ずつ置き換えます。置き換える場所が 1 か所でなければ止まります。
    1. フレームの始まりで、指令を実行部へ渡す(遅れの履歴・フィルター)。
    2. 小区間の速さ・舵・ヨーの計算(プロファイルが無ければ元の式のまま)。
    3. 実行部を作る。
    4. reset で実行部も戻す。
    5. report().model.calibration。
    6. 再生・試験の操作 calibControl。
    7. 返す関数の一覧に calibControl を足す。
  - 実行部の文字列は、先頭に埋め込みます。data: URL の中からは、相対の import ができないためです。
  - fast を付けると、10/13 の高速化の 5 か所も先に当てます。
- **当て方を選んだ理由**(design 7 章)
  - 設定欄では、次の効きを表せません: 左右で違う頭打ち・純粋な遅れ・惰性とブレーキの区別・こすれ・速さの頭打ち・ESC の後退の手順。
  - 差分を当てる形なら、元を変えずに全部入れられます。ユーザーが sim-core.js を更新すれば、sha256 で気づけます。

| 物理の値 | 今のシミュレーター | 較正(P_cal) |
|---|---|---|
| 舵いっぱいの曲率 | 左右とも 2.20 /m | 左 1.63・右 1.81 /m |
| 舵の遅れ・速さ | 0・1.2 rad/s | 50 ms・8 rad/s |
| 惰性(直線) | 1.2 m/s²(ブレーキ・後退と共通) | 0.58 m/s² + 曲がりながらのこすれ 0.40 × 横加速度 |
| ブレーキ | 1.2 m/s² のまま後退へ | 中立 + 200 µs で惰性 + 1.42 m/s²。走っている間は後退しない |
| 速さの頭打ち | 7.24 m/s(指令どおり) | 5.0 m/s(人の操作の対応) |
| 駆動 | τ 0.40・上限 6 m/s² | τ 0.32・上限 4.1 m/s²・不感帯 |
| 路面の μ | 0.8 / 0.7 / 0.6 / 0.22 / 0.65 | 0.82 / 0.70 / 0.58 / 0.21 / 0.66 |

## 3. 試験

| | 確かめたこと | 結果 |
|---|---|---|
| T1 | プロファイル無しの較正版(fast なし・あり)が、元と同じ動き(教師の REC60 60 条件・STD 単周と連続・ランダム操作 20 本。ランダムは衝突・ブレーキ・後退・設定違い・dt 1/60・低 μ・雑音を含む) | 62 本・20 本とも report の JSON が完全に一致 |
| T1b | 「元の式 + 遅れ 0」の実行部(P_default の再生の道具)も元と同じ | 完全に一致(model.calibration を除く) |
| T2 | 元の sha256 が違う・差分の場所が 1 か所でない・プロファイルが単調でない・知らない形式 → 止まる | 止まる |
| T3 | Python の参照(calib_model.replay)と JS のコアが、当てはめの窓 50 個(ブレーキ 6 を含む)で一致 | 速さの差 0、ヨーレートの差 3.3e-13(基準 1e-6)。P_cfg は小区間の長さの違いで最大 0.055 rad/s(参考) |
| T4 | ブラウザ用のファイル = patchSimCore の出力(バイト列)。calib.html と index.html の差は title と import map の 1 行だけ | 一致 |
| T4b | ブラウザ用のファイルそのものを読み込み、教師で REC60 の 1 条件 | Node の較正版と report が完全に一致 |
| T5 | 手計算の値: 惰性・ブレーキ(+200 µs で 2.00 m/s²、+60 µs は惰性と同じ)・ESC の後退の手順(中立 0.10 秒では下がらず 0.15 秒で下がる)・遅れ 50 ms と舵の速さ 8 rad/s・速さの遅れ・左右いっぱいの曲率 1.634 / 1.812 /m・アンダーステア・横の限界(カーペット・芝)・速さの対応と頭打ち・不感帯・電池 0.8・こすれ・元の式の遅れ 1/30 秒 | 全部合う(誤差 ≤ 1e-9。こすれは 1 フレームの近似で ≤ 0.06 m/s²) |
| T6 | 上限の層の JS(governorStep)が、Python の cap_step と同じ出力(4 つの設定 × 3,000 フレーム) | 12,000 フレームで、出力・区分・上限・ブレーキの始まりがすべて一致 |
| pytest | 安全層 16 本(tests/test_safety_caps.py)+ S1 の 16 本 | 32 本すべて合格 |

ログは `tests/` にあります: `t1_identity.json`・`s2_node_tests.log`・`t3_python_js.json`・`s2_t3.log`・`t4b_browser_file.log`・`s2_pytest.log`・`s2_build_browser.log`。

## 4. 上限の試験(S4)への引き継ぎ

- 上限の層は、2 つの形で使えます。
  - コアの指令のフィルター: `sim.calibControl({filter: ({frame, steering, speed, state, config}) => ({speed: …})})`。
  - 外から `governorStep` を呼ぶ形。
  - 位置の区間(参照ループ R への射影)は S4 で足します。`calib_cap.html` も S4・S5 で作ります(deviations D14)。
- 上限の層の既定は v1.1 です(`GOVERNOR_V1_SIM`)。
  - ブレーキをやめたあと 0.5 秒は再び始めません。舵が右へ 30% 以上ならブレーキしません。
  - protocol との違いは deviations D15・D17 にあります。
- 物理は、手順の決まりのとおり P_default と P_cal の両方で走らせます(P_cal は C3 で不合格のため)。
  - 電池の係数は 1 にします(P_cal_k1 は「条件つき」)。
  - AI の速さの偏りの感度として、k_batt 1.12 も走らせることを勧めます。
- 気づいたこと: 元の物理で作られた RC 教師を、較正の物理で走らせると遅くなりました。REC60 の 1 条件で、1 周が 8.76 秒から 18.2 秒になっています(衝突なし)。
  - 舵いっぱいが小さくなり、速さが頭打ちになったためとみられます。
  - 「元の物理で学んだ運転手」をそのまま使う試験(RL の先生・S-DIV)では、この遅さが結果に効きます。

## 5. 限界

- P_cal は、偏り(AI の窓で −0.45 m/s、区間の向き ±5 度)が残っていて、手順の判定は「不合格」です。上限の試験の結果は「仮説」です。
- ブレーキの表は仮です(16 回、強いもの 3 回)。計画項目 8 の実測で作り直します(v2)。
- ブラウザでの表示と S-DIV の走行は、まだ確かめていません(S5)。
- 実車の記録は 10/10 の 1 日だけで、電池の電圧の記録はありません。

## 6. ファイル

| ファイル | 中身 |
|---|---|
| `experiments/20261016_sim_calibration/calib_runtime.mjs` | 較正の物理の実行部・上限の層(純粋な ESM) |
| `experiments/20261016_sim_calibration/calib_core.mjs` | patchSimCore・loadCalibratedCore・loadProfile(sha256 の確かめつき) |
| `experiments/20261016_sim_calibration/build_browser_core.mjs` | 写しの確かめと、ブラウザ用のファイル・calib.html |
| `experiments/20261016_sim_calibration/replay_fidelity.mjs`・`fidelity.py` | 忠実度の再生(JS)と指標・判定(S3)、試験 T3 |
| `experiments/20261016_sim_calibration/g1_replay.py` | 安全層の再生と G1 の判定 |
| `experiments/20261016_sim_calibration/figures_s23.py` | 図 |
| `experiments/20261016_sim_calibration/tests/` | calib_core.test.mjs(T1・T2・T4・T5)・governor.test.mjs(T6)・t6_make_vectors.py・t4b_browser_file.mjs |
| `jetformer/safety_caps.py`・`tests/test_safety_caps.py` | 安全層 v1(v1.1)と試験 |
| `sim/`(このフォルダー) | ブラウザの写し。`dist/calib.html`・`dist/_calib/`(較正版のコアと MANIFEST.json) |
| `fidelity.md`・`fidelity.json`・`g1_replay.md`・`g1_replay.json` | 確かめの結果 |
| `deviations.md` | 手順との違い(D13〜D19 がこの段) |
