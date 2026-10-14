# 学習の段の報告 — 強化学習モード(絶対壁にぶつからない)`no_wall` 第 2 回 = 作り直し(sim_rl_modes_20261013)

- 作成 2026-10-14 05:40。段の役目: 確認 r1(`check_no_wall_r1.md`、不合格)の指摘を直し、protocol の 1 回だけの作り直し(種 22)で学習し、試験でない組で確かめてから
  最後の試験(selection.test_runs の 2 回目)を走らせ、ブラウザ向けに書き出し、図と報告を作る。第 1 回の報告は `train_no_wall_r1.md` に写して残した。
- 依頼(ユーザー、21:20):「シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども作成できるように。
  強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意」
- 守ったこと: ベストラップの作業のファイル・ユーザーのシミュレーター・JetFormer 本体(jetformer/ の下)・ブリッジ・jetformer-web に触れていない。実車・PCA9685・I2C・GPU・
  Firefox は使っていない。RL の学習は `jf-heavy --mem 2G --lock rl -- nice -n 15`(lock rl の順番待ち 57 分)、Node は `jf-heavy --lock none -- nice -n 15`。
  git にコミットしていない。`protocol.json` は変えていない(`sha256sum -c protocol.sha256` OK)。決まりと違うことは `deviations.md` の 4〜7 に書いた。

---

## 0. 要点

1. **作り直しの試験(元のコア、340 回、最後の 1 回)で N1〜N7 全部合格**。登録の一式 `bundle/no_wall/` は **`accepted`「合格」**。
   衝突 **0/340**・最小すき間 **11.75 cm(標準の物理、N2)/ 7.00 cm(物理のずれ、N3。r1 は 4.69 cm で不合格)**・停止から 1 周 **9.797 秒**・
   N60 60/60・STD・R200 **200/200**・ρ_req > 1 0.07%・ρ_comb > 1 0.19%・シールドの作動(標準の物理)**0%**。
   画面の文言: **「340 回の試験で衝突 0・最小すき間 11.7 cm(標準の物理)/ 7.0 cm(物理のずれ)」**。
2. 確認 r1 が見つけた範囲の中の衝突(ゲート 16・舵 0.8〜0.85・遅延 1 で 15 回中 11 回)は、**確認 r1 の道具そのもの**(`rl/check_nw/probe_g16.mjs`・
   `verify_nw.mjs --sets ABL,CORNER`)を作り直しの方策に当てて、試験の前に確かめた: **G16 衝突 0/40(最小 13.2 cm)・ABL 0/330・CORNER 6/460**
   (r1 + シールド v2: 12/40・7/330・80/460)。CORNER でシールドが**起こした衝突 0**(v2 は 13)、救った 161、シールドが 1 フレームも替えずに衝突 0(v2 は 19)。
3. 直したことは 3 つ: (a) **安全シールド v3**: 走りながら舵の速さ・遅延・ゲイン・ブレーキを実際の舵角から見積もって検査する + 遅い舵・弱いブレーキと分かったら
   速さの指令に上限をかける + 悲観の模型を DR-T の悪い端に(標準の物理では v2 と全く同じ判定)。(b) **作り直しの学習**: 種 22、r1 から続けて、遅い舵 + 遅延の回を
   学習の 0.1% → 約 25% に増やし、同じ速さの上限をかけて学ぶ。(c) **選び方**: 検証の組(標準の物理 15 回)では見えない標準の物理の衝突が 600 万の点にあったので、
   新しい種の 300 回で衝突 0 の点(400 万)を選んだ(deviations.md 7)。
4. 残っていること(判定ではない): DR-T の 4 隅を全部同時に悪くした CORNER で 6/460 衝突(ゲート 16 から右寄り・右向き 15° の開始 × 4 隅、摩擦円 2)。
   アプリの「低摩擦」シナリオ(μ × 0.4、範囲の外)では **10 回中 10 回衝突** → mode.json の `scenario_blocklist` に入れた(統合の段で選べないようにする)。
   標準の物理の新しい開始の 1.3% で 7.1〜7.9 cm(N2 の 8 cm 未満)になる → 文言を弱め、記録の前の予想に「すき間 8 cm 以上」を足した。
5. データセットのばらつき: no_wall は最速と線が平均 6.8 cm(90% 点 13.6 cm)違い、遅い(走っているとき 2.75 vs 3.30 m/s)。ただしモードの中の線のばらつきは
   0.65 cm(最速 2.4 cm)で、「新しい 1 本の線」を足すモード(r1 と同じ。VM1 118 µs・VM3 0.37 cm)。

---

## 1. 確認 r1 の指摘への答え

| 指摘(check_no_wall_r1) | 何をしたか | 結果 |
|---|---|---|
| 1. N3 不合格(R58 4.694 cm)。作り直し(種 22・最後の試験)が通らなければ「不合格(参考)」のまま。作り直しは試験の前に `probe_g16.mjs`・`verify_nw.mjs --sets ABL,CORNER` で調べること | 種 22 で作り直し(5 章)、確認 r1 の道具で調べてから(6 章)試験(7 章) | 試験 N3 **7.00 cm**(R58 は 8.93 cm)・全部合格。probe_g16 0/40、ABL 0/330、CORNER 6/460 |
| 2. 範囲の中で衝突(ゲート 16・舵 0.80〜0.85・遅延 1 で 11/15)。CORNER 80 衝突、うち 19 はシールド 0 フレーム、13 はシールドが起こした。r1 の提案(すき間の壁 0.4)は引き金を狙っていない(学習の 0.1%) | シールド v3(見積もり + 速さの見張り)と、遅い舵 + 遅延の回を増やした学習(hard_dr 30%・速さの見張りつき) | r2 + v3: G16 0/40、CORNER 6/460・起こした 0・0 フレームで衝突 0。シールドなしの r2 の方策だけでも G16 0/40(r1 は 4/40) |
| 3. v2 の keep_nominal が衝突を通す(ABL の衝突 7 回すべてで先に proposal_nominal)。悲観の模型(舵 0.85)が DR-T の下限 0.8 より甘い。シールドを変えるなら deviations.md に書き、起こす衝突と救う衝突を両方数えること | keep_nominal の「標準」を**見積もった模型**にした(遅い舵 + 遅延なら、それで検査する)。悲観の模型 = 舵 min(0.8, 見積もり)・遅延 max(1, 見積もり)・ブレーキ min(0.7, 見積もり)・μ × 0.85。deviations.md 4 | 起こした / 救った(シールドなしの同じ回と比べて): r1 方策 CORNER 0 / 132、ABL 0 / 5、G16 0 / 4。r2 方策 CORNER 0 / 161、ABL 0 / 3。標準の物理(FNOM 150 回)では v3 の軌跡は v2 と**ビット単位で同じ** |
| 4. 画面の文言に N2 と N3 の両方を | `eval_modes.mjs` の ui_text を「N 回の試験で衝突 k・最小すき間 X cm(標準の物理)/ Y cm(物理のずれ)」に。不合格なら publish_mode.py が「・不合格(参考)」を足す(deviations.md 6) | 「340 回の試験で衝突 0・最小すき間 11.7 cm(標準の物理)/ 7.0 cm(物理のずれ)」(合格なので後ろは付かない)。`reported.ui_numbers` に数字も |
| 5. short_ja「8 cm 以上はなれて走る」は記録の条件で言い切れない(150 回中 2 回 7.10・7.54 cm) | 両方した: short_ja を弱めた + mode.json `recording.prefilter_min_clearance_m` 0.08 と `rl/predict_scenes.mjs --min-clearance`(8 cm 未満のシーンも替える)。deviations.md 6 | r2 でも FNOM 150 回中 2 回 7.08・7.88 cm(ゲート 10・23 の斜めの開始)= 前もって走らせて替えれば、記録のシーンは 8 cm 以上 |
| 6. 統合の段: 「低摩擦」(μ × 0.4)を強化学習モードでは止めるか警告(r1 は 9/10 衝突) | mode.json に `scenario_blocklist: ["low-friction"]` と `scenario_note_ja`(統合の段が読む)。シールド v3 でも μ は見積もらない | r2 + v3 でも **10/10 衝突**(範囲の外)。統合の段で必ず止めること |

---

## 2. 安全シールド v3(`rl/runtime/shield.mjs`、jf-shield/v3、sha256 d682a6b5…)

- **走りながらの見積もり**: sim の舵は 1 フレームで高々「舵の速さ × dt」しか動かず、目標に届かなかったフレームではちょうどその量だけ動く。そこで snapshot の実際の舵角 δ と、
  シールドが返した指令の列から、遅延 0/1/2 の仮説ごとに「δ の動きが指令と合うか」を調べ、合わない仮説を捨てる。残った一番遅い仮説で、舵の速さ(届かなかったフレームの
  |Δδ|/dt、正確な値)・舵のゲイン(届いたフレームの δ/指令)・ブレーキ(目標より上で止まった減速の最大)を使う。見積もりの前(走り出しの 1〜2 フレーム)は config の値。
  標準の物理なら見積もり = 標準の値なので、**記録(標準の物理)のシールドの判定は v2 と同じ**。元のコアでの試験で、舵の速さ 0.8〜1.5・遅延 0/1/2・ゲイン 0.94〜1.06・
  ブレーキ 0.7〜2.0 が 1e-6 以内で当たる(`tests/shield_v3.test.mjs`)。no_wall の方策は 1〜2 フレーム目で舵を速さいっぱいに切るので、2 フレームで見積もりが決まる。
- **速さの見張り**: 見積もりが確かになったら(3 フレーム目から)、舵の応答の時間 T = 遅延·dt + 0.35/舵の速さ が標準(0.35/1.2)より長いぶん、またブレーキが標準より弱いぶん
  √(b/1.2)、速さの指令に上限をかける(上限 = 4.5 m/s × T_標準/T × √(min(1, b/1.2))。例: 舵 0.8・遅延 1 → 2.79 m/s)。0.8 秒の先読みでは、遅い舵で 3.7 m/s まで
  出てからではどの指令でも間に合わない(ゲート 16 の回)ため。上限で替えたフレームも sh = 1 に数える。
- **悲観の模型**: DR-T の悪い端(舵 min(0.8, 見積もり)・遅延 max(1, 見積もり)・ブレーキ min(0.7, 見積もり)・μ × 0.85)。v2 は 0.85・1・0.8・μ そのまま。
- 変えていないもの: しきい値(見積もった模型 ≥ 7 cm・悲観 ≥ 5 cm)・先読み 24 フレーム・予備の操縦の組・候補・「見積もった模型で合格なら替えない」。
- 版の選び(r1 の方策で、試験でない組だけ。deviations.md 4 の表): v3 の 3 つの版のうち、ブレーキの見積もりを「減速の最大」にした版を採った
  (「最小」にすると摩擦円で減速が小さく出て、検証の組の R15 がのろくなり 1 周できなかった)。

| r1 の方策 | G16 衝突(40) | ABL 衝突(330) | CORNER 衝突(460) | CORNER 救った / 起こした / 0 フレームで衝突 | CORNER 完走 + 1 周 | FDR(400、DR-T 一様) | FNOM(150、標準) |
|---|---|---|---|---|---|---|---|
| シールドなし | 4 | 5 | 141 | — | 315 | 衝突 0・5.77 cm | — |
| v2(確認 r1) | 12 | 7 | 80 | 74 / 13 / 19 | 328 | 衝突 0・6.75 cm | 衝突 0・7.10 cm |
| **v3(採用)** | **0** | **0** | **9** | **132 / 0 / 0** | **418** | **衝突 0・7.11 cm**(作動 17.6%) | v2 と同じ軌跡 |

- ブラウザの組 `web/rl/` を作り直した(manifest 1f0ba8f1…。変わったのは shield.mjs と、mode_runtime.mjs のシールドの記録 `shield.id` だけ)。
  Node の試験: shield・shield_v2(版の名前だけ直した)・shield_v3(5 本)・runtime・env・cond_gate_miss・core・perturb・metrics_eval の 30 本、全部合格。

---

## 3. 作り直しの学習(`train/no_wall/r2/`、deviations.md 5)

| 項目 | 値 |
|---|---|
| 種・ステップ | **22**(= 21 + 1)・**600 万**(retry_steps_max 1,000 万の中)。04:05〜04:42、平均 2,744 steps/s(ベストラップの GPU の学習と重なった) |
| 初期化 | r1 の選んだ点 `train/no_wall/main/checkpoints/step_010010624.pt`(方策 d2f9c262…)。lr 5e-5 → 3e-5 |
| 範囲の中の調整 | r1 と同じ(traction_req_w 0.5・friction_circle_w 0.5・terminal_clearance 0.08・barrier_start 0.14)+ **clearance_barrier_w 0.25 → 0.4** |
| 足した条件(範囲の外の鍵) | `hard_dr`: 物理をずらす回の 30% を 舵 0.75〜0.95 rad/s + 遅延 1(70%)/2(30%)に(値は DR-NW-train の中)。`governor`: シールド v3 と同じ速さの上限を本当の物理の値でかける |
| 学習の終わり(最後の 30 回の更新、835 回) | 20 秒を走り切る 97.5%(すき間の約束で終了 18・ゲート外し 2・黒シート 1、**衝突 0**)、1 周目 10.35 秒、回の最小すき間の平均 12.8 cm、ρ_comb > 1 0.9%、σ 0.079・0.162 |

### 3.1 チェックポイントの選び(`train/no_wall/r2/selection.json`)

| 点 | 検証の組(88 回、シールド v3) | N2 / N3 [cm] | N4 [秒] | 選びの追加の組 SELN + SELD(各 150 回)衝突 |
|---|---|---|---|---|
| 100 万 | N2 不合格 | 7.24 / 9.63 | 9.75 | — |
| 200 万 | N2 不合格 | 7.14 / 8.55 | 9.69 | — |
| 300 万 | **7/7** | 8.00 / 9.75 | 9.81 | 0 + 0 |
| **400 万** | **7/7** | **8.09 / 10.03** | **9.80** | **0 + 0** |
| 500 万 | N2 不合格 | 7.49 / 7.03 | 9.69 | 0 + 0 |
| 600 万 | 7/7 | 8.01 / 9.38 | 9.60 | **1 + 1** |

- 検証の組だけの決まりなら 600 万。試験の前の確認(確認 r1 の FNOM・FDR と同じ作り方)で、600 万は**標準の物理で 150 回中 3 回衝突**(ゲート 2 からの開始で、
  1 周目の終わりの右の大きな曲がりに 4 m/s で入り、10 秒でカーテン。シールドは 1 フレーム前に「どの候補も間に合わない」)、FDR で 4/400 だった。検証の組の標準の物理は
  15 回しかなく 2% 前後の衝突は見えないので、確認の組とは別の種の SELN・SELD を作って「衝突 0」を足して選んだ → **400 万(`step_004014080`、sha256 153c709c…)**。
  600 万の確認の結果は `train/no_wall/r2/probes_6M_*/` に残した。
- 図: `fig/no_wall_r2_learning_curve.png`(上 = 学習中、下 = チェックポイントごとの検証の組と追加の組。点線 = 選んだ点)。

---

## 4. 試験でない組での確認(試験の前、選んだ 400 万の方策 + シールド v3)

確認 r1 の道具そのもの(`rl/check_nw/`: ユーザーの sim-core.js・ユーザーのコースの読み込み・ブラウザの組 `web/rl/`・自前のすき間)を、`bundle/no_wall/` に
「未試験」で置いた方策で走らせた(`train/no_wall/r2/probes_checker_tools/`)。そのほかは同じ走らせ方の写し `rl/nw2/verify_nw2.mjs`(`probes_r2policy/`)。

| 組 | 回 | 衝突 | 最小すき間 | 5 cm 未満 / 8 cm 未満 | 完走 + 1 周 | シールドの作動 | シールドなしの衝突 | 救った / 起こした |
|---|---|---|---|---|---|---|---|---|
| G16(probe_g16.mjs) | 40 | **0** | 13.17 cm | 0 / 0 | 40 | 55.8% | 0 | 0 / 0 |
| ABL(verify_nw.mjs) | 330 | **0** | 7.00 cm | 0 / 2 | 322 | 57.1% | 3 | 3 / 0 |
| CORNER(verify_nw.mjs) | 460 | **6** | 0 | 10 / 47 | 423 | 73.0% | 167 | 161 / **0** |
| FDR(DR-T 一様・新しい開始) | 400 | **0** | 7.08 cm | 0 / 5 | 400 | 18.2% | 0(最小 5.2 cm) | 0 / 0 |
| FNOM(標準の物理・新しい開始) | 150 | **0** | 7.08 cm | 0 / 2 | 150 | 0% | — | — |
| 検証の組 | 88 | 0 | 8.09 cm | 0 / 0 | 78/78(連続) | 13.8% | — | — |
| LOWMU(μ × 0.4、範囲の外) | 10 | **10** | 0 | — | 0 | 2.9% | — | — |

- CORNER の残りの 6 回: ゲート 16 から右寄り 0.2 m・右向き 15° の開始 × 4 隅(1.9〜2.1 秒、外側のレール)と、摩擦円の隅 C3 の 2 回(15〜17 秒)。
  CORNER の衝突のうちシールドが 1 フレームも替えなかった回は 0。衝突しないが 20 秒で 1 周できない回が 31(4 隅とも悪い物理で、上限のかかった遅い走り)。
- FNOM の 8 cm 未満の 2 回(7.08 cm ゲート 10・7.88 cm ゲート 23)は、r1 と同じ形(斜めの開始の 0.5 秒後)。記録の前の予想の 8 cm の決まりで替わる。
- 図: `fig/no_wall_r2_probe_sets.png`(r1 + v2 / r1 + v3 / r2 + v3 の衝突と、救った・起こした)、`fig/no_wall_r2_topdown.png`(ゲート 16・舵 0.8・遅延 1 の線と標準の物理の STD)。

---

## 5. 試験の組の結果(`eval/no_wall/no_wall_test_r2.json`、元のコア、selection.test_runs の 2 回目 = 最後)

| 判定 | 基準 | r1(参考) | **r2** | 合否 |
|---|---|---|---|---|
| N1 衝突 | 340 回すべて 0 | 340/340 | **340/340** | 合格 |
| N2 すき間(標準の物理) | ≥ 8 cm | 12.07 | **11.75 cm** | 合格 |
| N3 すき間(ずらした物理) | ≥ 5 cm | 4.69(不合格) | **7.00 cm**(R16。R58 は 8.93 cm) | **合格** |
| N4 停止から 1 周 | ≤ 12 秒 | 9.772 | **9.797 秒** | 合格 |
| N5 完走 | N60 60/60・STD・R200 ≥ 190/200 | 60/60・STD・200/200 | **60/60・STD・200/200** | 合格 |
| N6 摩擦 | ≤ 2% | 0.02%・0.20% | **0.07%・0.19%** | 合格 |
| N7 シールド(N60) | ≤ 5% | 0% | **0%** | 合格 |

- 組ごと(最小・中央値): N60 11.75・13.62 cm / STD 13.17 / G(68)12.97・13.74 / R200 7.00・13.87(8 cm 未満 4 回: R16・R167・R31 は舵 1.4〜1.6・遅延 0、R76)/ FC10 13.18。
- **報告の値**: NW-OFF(シールドなし 260 回)衝突 **1**(R58。方策だけでは遅い舵 + 遅延 + ゲート 16 の斜めの開始をまだ越えられない。シールドありでは 8.93 cm)、5 cm 未満 2・8 cm 未満 5。
  R200 のシールドの作動 17.7%(ほとんどが速さの見張り)。NW-G の範囲の外 12 回: 衝突 0・13.0〜14.3 cm。1 周: N60 の 120 周 9.40〜9.90 秒(中央値 9.52)、
  R200 の 342 周 中央値 9.77・最大 12.14 秒。ρ の最大 ρ_req 1.52・ρ_comb 1.31。走っているときの平均の速さ 2.72 m/s。
- 判定の外の物差し(N60、参考): VM1 118 µs・VM2 11.7 µs・VM3 0.37 cm・VM4 0.015 m/s・VM7 269 マス(r1 122・15.8・0.46・0.019・292)。

---

## 6. 書き出し(`experiments/20261013_sim_rl_modes/bundle/no_wall/`)

- `policy.json`(jf-sim-rl-mode-policy/v1、obs 79、sha256 **153c709c…**)・`eval.json`(試験の組、シールドの版と shield.mjs の sha256 入り)・
  `status.json` = **`accepted`「合格」**・`mode.json`(上書き: short_ja・runtime.shield_params・recording.prefilter_min_clearance_m 0.08・scenario_blocklist ["low-friction"]・
  scenario_note_ja。`overrides_from_modes_json` に鍵の一覧)。r1 の一式は `train/no_wall/r2/diag/bundle_r1/`。
- 登録簿(`data/sim_rl/modes/no_wall/`)には書いていない(統合の段が `--registry` で置く)。ブラウザの組は **`web/rl/`(manifest 1f0ba8f1…)を使うこと**(v2 の組は古い)。
- F-ID: `eval/no_wall/fid_std_no_wall_r2.json`(標準の物理・STD、シールドの作動 0)と **`eval/no_wall/fid_slow_no_wall_r2.json`**(ゲート 16・舵 0.8・遅延 1 の snapshot の列。
  シールドの見積もり・速さの見張りが 400 フレーム働く)。どちらも `rl/check_fid.mjs` で Node と `web/rl/mode_runtime.mjs` の差 0。ブラウザ(8080)の teacher.js での照合は統合の段
  (U5「シールドも同じ」は fid_slow で確かめられる)。
- mode.json の short_ja: 「壁・柱からはなれて遅めに走る(記録するシーンは前もって走らせ、すき間が 8 cm 未満になるものは替える)。物理のずれ(…)でもぶつからないよう学習し、
  走行中は安全シールドが舵の遅れを見積もって見張る。遅くてよい」。

### 統合の段・記録の段へ

1. 「低摩擦」シナリオは強化学習モードで選べないようにする(mode.json `scenario_blocklist`。r2 + v3 で 10/10 衝突)。
2. 記録の前の予想は `node rl/predict_scenes.mjs --plan … --policy bundle/no_wall/policy.json --min-clearance 0.08`(mode.json `recording.prefilter_min_clearance_m`)。
3. ブラウザの組は今の `web/rl/`(shield.mjs d682a6b5…)。F-ID は 2 本(上)。

---

## 7. データセットのばらつき(FNOM の 150 の開始、標準の物理。`train/no_wall/r2/line_compare.json`)

| 物差し | r2 | r1 | 最速 |
|---|---|---|---|
| 線の違い(最速と、5 cm のマスの平均 e_y の差) | 平均 6.8 cm・90% 点 13.6・最大 21.4 cm | 7.2・18.7・27.0 cm | — |
| r2 と r1 の線の違い | 平均 3.6 cm・90% 点 6.1 cm | — | — |
| モードの中の線のばらつき | 0.65 cm | 0.53 cm | 2.41 cm |
| 走っているときの平均の速さ | 2.75 m/s | 2.67 m/s | 3.30 m/s |
| 1 周目の中央値 | 9.59 秒 | 9.76 秒 | 8.10 秒 |
| 回の最小すき間の中央値 | 12.3 cm | 12.8 cm | 6.2 cm |

no_wall は最速とは別の「遅くて真ん中寄りの 1 本の線」を足す。モードの中のばらつきは小さい(シーンを増やしても線は増えない)。線のばらつきは経路いろいろ(line)・速さいろいろ(speed)の役目。

---

## 8. ファイル

| 置き場所 | 中身 |
|---|---|
| `train_no_wall.md`(この報告)・`train_no_wall_r1.md`(第 1 回の報告の写し) | |
| `train/no_wall/r2/` | 作り直しの学習(`run.json`・`progress.jsonl`・`checkpoints/`・`policies/`・`evals/no_wall/`・`done.json`)、`selection.json`(と `selection_validation_only.json`)、`sel/`(選びの追加の組)、`probes_checker_tools/`・`probes_r2policy/`(試験の前の確認)・`probes_6M_*`(600 万の確認)、`shield_v3*_r1policy/`・`r1policy_off/`(シールドの版の比べ)、`sets_summary.json`・`summary_r2.json`・`line_compare.json`、`fig_traces/`、`diag/`(変える前の写し・v2 の shield.mjs・r1 の一式)、`sha256_r2.txt`、`mode_overrides.json` |
| `train/no_wall/tuning_r2.json`・`run_r2.sh` | 作り直しの調整と起動 |
| `train/no_wall/logs/` | `r2.log`(学習)・`eval_checkpoints_r2.log`・`after_r2.log`(選び・確認・試験・書き出し) |
| `eval/no_wall/no_wall_test_r2.json`(.log)・`fid_std_no_wall_r2.json`・`fid_slow_no_wall_r2.json` | 試験の組・F-ID |
| `fig/no_wall_r2_learning_curve.png`・`no_wall_r2_probe_sets.png`・`no_wall_r2_topdown.png` | 図 |
| コード `experiments/20261013_sim_rl_modes/` | `rl/runtime/shield.mjs`(v3)・`rl/runtime/mode_runtime.mjs`(シールドの記録)・`web/rl/`・`bundle/no_wall/`・`tests/shield_v3.test.mjs`・`rl/nw2/`(この段のスクリプト: `run_nw2.mjs`・`verify_nw2.mjs` = 確認 r1 の写しに VAL・G16・SELN・SELD・*OFF を足し試験の組を外したもの、`compare_sets.py`・`build_sets_json.py`・`select_r2.py`・`after_r2.sh`・`write_fid_slow.mjs`・`line_compare.py`・`make_figures_r2.py`・`traces_fig.mjs`・診断の `dump*.mjs`・`ident_probe.mjs`)。共有のファイルの変更: `rl/env_modes.mjs`(hard_dr・governor)・`rl/train_modes.py`(鍵 2 つ)・`rl/eval_modes.mjs`(ui_text・シールドの記録)・`rl/publish_mode.py`(--overrides・不合格の文言)・`rl/predict_scenes.mjs`(--min-clearance)。どれも鍵・引数が無ければ今までと同じ。sha256 は `train/no_wall/r2/sha256_r2.txt`(env_modes.mjs・train_modes.py・eval_modes.mjs・predict_scenes.mjs はこのあとほかの段(speed r2)も直している) |

やり直し方(要点):

```bash
cd ~/jetformer/experiments/20261013_sim_rl_modes
/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/no_wall/run_r2.sh             # 学習(lock rl、last.pt から続く)
rl/nw/eval_checkpoints.sh /home/jetson/jetformer_reports/sim_rl_modes_20261013/train/no_wall/r2 2   # 検証の組(シールド v3)
NW2_POLICY=<policy> node rl/nw2/verify_nw2.mjs --sets SELN,SELD --jobs 3 --out <dir>     # 選びの追加の組
rl/nw2/after_r2.sh select | probes | test | publish                                       # 選び・確認・試験(1 回だけ)・書き出し
```

---

## 9. 起動したプロセス

- 学習(r2)・検証の評価のループ・確認の組・試験・F-ID はすべて終わっている(05:31 に `ps` で確認)。次の段が続ける長い学習は無い。
- ほかの作業(cond/line・speed・obstacle の学習、ベストラップの GPU の学習と記録の確かめ、ユーザーのブリッジ)には触れていない。lock rl は line の作り直しのあとに取り、
  学習が終わって返した(そのあと obstacle が取った)。
