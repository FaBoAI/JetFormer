# 審査の段の報告(第 3〜4 回)— シミュレーターの強化学習モード(sim_rl_modes_20261013 / r3)

- 作成 2026-10-15 04:40。段の役目: 第 3〜4 回(r3 の計画 → no_wall r4/r4b/r4c・recover r4・line r4/r4b の学び直し → 確認 r4 → 昇格)を、
  生の記録(試験の回ごとの行・JSON・データセット・API)に照らして疑いながら端から端まで確かめ、間違いを直し、ユーザー向けの `README.md` と
  1 分動画の絵コンテ `video/storyboard.json` を最終の状態に書き直す。
- 依頼(ユーザー、2026-10-13 21:20):「シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども
  作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意」、
  2026-10-14 08:30「作業を引き続き実施」。
- 守ったこと: 実車・PCA9685・I2C・GPU に触れていない。RL の学習はしていない。pytest 一式は `jf-heavy --mem 2500M --lock big`、ほかは `jf-heavy --lock none` + nice 15。
  jetformer-web は再起動していない。ブリッジは見つけたときのまま(停止中・モデルなし)。ユーザーのシミュレーター・ベストラップの作業のファイルは読むだけ。
  共有のコード・bundle・登録簿・JetFormer 本体・データセットは変えていない(変えたのは報告の 3 ファイルと図だけ。8 章)。git にはコミットしていない。
- 前の試みの有無: この段の前の試み(電源断で止まったもの)は無かった(`r3/review*` が無かった)。昇格の段は 04:10 に終わっていて、動いている処理は無かった。

---

## 0. 結論

1. **最終の状態は正しい**: 選べる強化学習モードは **5 つ(最速・絶対壁にぶつからない・離脱復活・速さいろいろ・経路いろいろ)で全部「合格」**、障害物よけは「出していない」。
   bundle・登録簿・画面の API・候補の status が全部そろい(`review/bundles.json`)、第 3 回の失敗した候補 3 つは not_accepted。
2. **固めた決まりは変わっていない**: protocol.json・r2 の追補 4 つ・r3/r4 の追補 8 つ・立て直せる表 2 つの **18 ファイルの sha256 が全部、固めたときと同じ**。
   protocol.json の files(ユーザーのアプリ・元の sim-core.js・ベストラップの方策など 18)も同じ。順番も「追補を固めた → 学習 → 選び → 試験(1 回)」(1 章)。
3. **数字は回ごとの記録までたどれた**: status.json の test の値は試験の JSON の judged と 26 項目すべて同じ。主な数字(衝突・最小すき間・走り切り・成功の割合など)を
   回ごとの行から判定のコードを使わずに数え直して一致(2 章)。確認の段の数字も status.json・カードと一致。
4. **データセットは正しく一覧・タグ・学習除外されている**: sim-rl-* 11 個すべてで、タグ = mode.json、ジャンルのフレーム・学習除外の数・シーンタブの展開(mode_id 8)がそろう。
   前の版の方策の 4 つには `rl_superseded`、今の方策の 7 つには無い。JetFormer の学習の読み込みで、乱した操作が入力・正解に入る観測は **0**、
   正解が k+1〜k+10 のフレームでないものも **0**(シーンタブから作り直す道・ジャンルを付け直す道も 0。昇格の段の「フレームの番号で引く」直しが効いている)(3 章)。
5. **テスト一式 679 passed・3 skipped**(4 分 13 秒)。前後で作業ツリーの git status は同じ(4 章)。
6. **ユーザーのシミュレーターは変わっていない**: 元の展開 851/851・JetFormer に入れた版 849/849(2 つはインストールで省いたもの)が COMPLETE-MANIFEST の sha256 と一致。
   ベストラップの作業のファイルは完了のコミット 9363a19 から差分 0・そのあと書き換わったファイル 0(5 章)。
7. **直したもの**: `deviations.md` に line r4 の段の決まりの外のこと(1〜9)が写されていなかった → 写した。数字の訂正 2 つ(前の教師の VM7 の基準 278・前の版の離脱復活のゲートの外し 6/15)。
   `README.md` を最終の状態に書き直し、絵コンテと図を第 4 回の数字で作り直した(7〜8 章)。
8. **残る注意**(判定の外。README の限界にも書いた): 絶対壁にぶつからないの 8.1 cm はシールドのしきい値(方策だけでは 4.54 cm)・線はほぼ 1 本、
   離脱復活は余裕が小さい(8.85 秒 / 9.0、3.50 cm / 3 cm)・止まる開始がある、経路いろいろは方策だけでは角で崩れる(攻めの組 23/360)。小さな見た目の問題 3 つ(6 章)。

---

## 1. 固めた決まり(protocol と追補)— `review/protocols.json`(`r3/review/check_protocols.py`)

| ファイル | 固めた sha256 | 今 | 固めた時刻(.sha256) |
|---|---|---|---|
| protocol.json | 21dba877… | 同じ | 10-13 22:05:47 |
| protocol_speed_r2・line_r2・obstacle_r2・obstacle_r2b | 5550adcf…・82de57e2…・9b591db7…・0c5fc6eb… | 同じ | 10-14 05:19〜07:44 |
| r3/protocol_{no_wall,recover,line}_r3 | 73a5a719…・0a58b11c…・25c853d7… | 同じ | 09:52:58 |
| r3/protocol_recover_r3_table(表 v1・recoverable_table.mjs) | 410f18c2…・921e1514… | 同じ | 10:37:41 |
| r3/protocol_line_r4 | f77e844a… | 同じ | 15:52:32 |
| r3/protocol_no_wall_r4・r4b | ad90c926…・e6a77b7f… | 同じ | 15:17:20(r4 のファイル)・16:09:58 |
| r3/protocol_line_r4b | c7a1f677… | 同じ | 21:16:33 |
| r3/protocol_no_wall_r4c | 26d890ad… | 同じ | 20:56:38 |
| r3/protocol_recover_r4 | 1b6bd8ad… | 同じ | 22:14:43 |
| r3/protocol_recover_r4_table(表 v2・recoverable_table.mjs) | 8735365c…・4ac6b682… | 同じ | 23:06:24 |

- protocol.json の files 18 個(design.md・modes.json・ユーザーのアプリ・元の sim-core.js・ベストラップの方策など)も全部同じ。
- 順番(ファイルの時刻と run.json・selection・試験の JSON の時刻):

| モード | 追補を固めた | 学習の始まり | 選び | 試験(1 回) | 試験の JSON の追補の sha256・方策 |
|---|---|---|---|---|---|
| no_wall | r4 15:17・r4b 16:09・r4c 20:56 | r4 15:39(16:03 に 106 万で止めて r4b へ)・r4b 20:18(停電のあと)・r4c 20:56:58 | 22:24:27 | 22:24〜23:28 | ad90c926…・e632d96b… |
| recover | r4 22:14・表 v2 23:06:24 | 23:06:34 | 00:07:20 | 00:07〜00:16 | 1b6bd8ad…・表 8735365c…・68e321e4… |
| line | r4 15:52・r4b 21:16:33 | r4b 21:16:58(r4 の 401 万から) | 22:34:55 | 22:34〜22:50 | f77e844a…・c7a1f677…・5c197a2b… |

  no_wall r4c の追補(20:56:38)は、r4b の最後の点の検証(20:51:34 開始・20:59:54 に結果)より前に固めている(学習の段の「検証の結果を読む前」と合う)。

## 2. bundle・登録簿・画面の数字 — `review/bundles.json`・`review/recount_tests.json`

### 2.1 一式がそろっているか(`r3/review/check_bundles.py`)

| 確かめたこと | no_wall | recover | line |
|---|---|---|---|
| 方策の sha256: 試験の JSON = 候補 = bundle = mode.json = 登録簿 = 画面の API | e632d96b…(7 つ同じ) | 68e321e4…(同じ) | 5c197a2b…(同じ) |
| bundle の sha256.json = 今のファイル / 候補と違うファイル | 一致 / 無し | 一致 / 無し | 一致 / 無し |
| status.json の test = 試験の JSON の judged(項目ごと) | 7/7 同じ | 8/8 同じ | 11/11 同じ |
| 状態(bundle・登録簿・API・候補) | accepted × 4 | accepted × 4 | accepted × 4 |
| 登録簿 = bundle(mode・policy・eval) | 同じ | 同じ(eval は末尾の改行だけ違う・中身同じ) | mode・policy 同じ、eval は回ごとの行を除いた版(昇格の段の決まりの外 3) |

- 配るファイル: `rl/runtime` = `web/rl` = `jetformer/sim/static/rl` の 19 ファイルが全部同じ、manifest(5e6cf478…)の sha256 = ファイル。shield.mjs 130ed5b2・shield_v4p.mjs 29c9d111・mode_runtime.mjs f59b1955。
  **ページが実際に読むもの** も http://192.168.55.1:8080 から取って比べた: `/sim/c/default/_jf/rl/` の 21 ファイル(teacher.js・manifest を含む)と `/sim/static/sim-rl.js` がディスクと同じ。
  登録簿の line の `runtime`(shield true・select v5b・4 cm)が API から出る。jetformer-web は 03:02:27 に起動、Python の最後の変更は 03:01:51(今のコードで動いている)。
- 確認の段の数字 → status.json の check: no_wall(新しい組 6,429・範囲の中 5,749 で衝突 0・標準 8.10 cm・物理のずれ 7.03 cm・止まり 0/3,303・4/2,411・方策だけ 4.54 cm・範囲の外 6/200)、
  recover(新しい開始 200/200・C2 0.977・Firefox 100/100・キック 244/244・すき間 3〜5 cm で 9/60・「強」固定 0.957/0.941)、line(新 1,717・攻め 360・衝突 0・最小 3.995 cm・シールドなしの攻め 23/360)が、確認の段の出力と全部同じ。
- 第 3 回の候補 `r3/candidates/{no_wall,recover,line}_r3/status.json` は 3 つとも not_accepted(昇格の段が line_r3 を正した)。画面の API の状態: 最速・壁・離脱復活・経路・速さ = accepted、障害物よけ = not_published。

### 2.2 回ごとの行から数え直す(`r3/review/recount_tests.py`。判定のコードは使わない)

| モード | 数え直し | 報告・画面の値 |
|---|---|---|
| no_wall(rows_test.jsonl 4,010 行) | 判定の組 3,038 回で衝突 0・標準の物理の最小 8.1044 cm(F990)・物理のずれ 864 回の最小 8.1562 cm(VX223)・8〜9 cm の標準 65 回・N300 のシールド 0.146%・回の最大 1.5% | 3,038/3,038・8.104・8.156 cm・65 回・0.15%・1.5% |
| recover(eval_test.json 609 回) | RC-P100 走り切り 99/100(外れは P27 の止まり)・衝突 0・乱し 408 回のうち成功 405(0.9926)・1 回 4.08 回・外す 18.27%・有効な回のゲートの外し 0・U60 60/60 最小 3.5035 cm・REC60 60/60 | C1 99/100・C2 0.993・C4 4.08/0.183・C8 0/99・C6 3.50 cm |
| line(eval_test_r4b.json 542 回) | 衝突 0・止まり 0・抜け出し 0・後ろへのフレーム 0・有効 60/60/60・115/116(C46 = ゲート 11 の黒シート)・120/120・120/120・最小 4.1268 cm・P60 のシールド 0.111% | L1・L2 4.13 cm・L7 0.11%・L8〜L11 0 |

- no_wall の行の「物理」で数えると標準は 2,176 回になるが、決まり(N2 = N300・FS1000・LOWC・EDGE・STD・LONG の標準)では NW-G の標準 2 回は N3 側に入るので 2,174 回が正しい(最小は同じ)。
- recover の RC-KICK-R4(288)・RC-KICKR-R4(100)には衝突 25・17 があるが、C5 はキックの種類ごとの生き残りの割合(83/84・84/84・24/24・26/26)で判定する組で、決まりどおり。

## 3. JetFormer のデータセットと学習除外 — `review/datasets.json`・`review/mask_e2e_r4.json`・`review/frame_indexed_raw.json`

### 3.1 一覧・タグ・ジャンル・展開(`/api/datasets`・`/api/scenes`、`r3/review/check_datasets.py`)

| データセット | 方策 | シーン | 学習に使える | ジャンルのフレーム | 学習除外 | シーンタブ | カード |
|---|---|---|---|---|---|---|---|
| sim-rl-no_wall_20261015-0302 | e632d96b(今) | 15(12/3) | 8,850 | 8: 9,000 | 0 | 15・mode_id 8 | — |
| sim-rl-recover_20261015-0302 | 68e321e4(今) | 15(12/3) | 7,227 | 8: 7,377・0: 1,623 | 1,623 | 15・mode_id 8 | — |
| sim-rl-line_20261015-0302 | 5c197a2b(今) | 15(12/3) | 8,850 | 8: 9,000 | 0 | 15・mode_id 8 | — |
| sim-rl-fastest_20261014-{1025,1125}・sim-rl-speed_20261014-{1025,1125} | 今 | 3・15 | 1,770・8,850 | 8 だけ | 0 | 3・15 | — |
| sim-rl-no_wall_20261014-{1025,1125} | 153c709c(前の版) | 3・15 | 1,770・8,850 | 8 だけ | 0 | 3・15 | rl_superseded |
| sim-rl-recover_20261014-{1025,1125} | 7720f275(前の版) | 3・15 | 1,455・7,156 | 0 が 315・1,694 | 315・1,694 | 3・15 | rl_superseded(ゲートの外し 2・6 シーン) |

- タグは 11 個とも登録簿の mode.json の tags と同じ(simulator・sim-rl・sim-rl-<モード> + sim-safe / sim-recover / sim-line / sim-speed)。ジャンルの表 `data/modes.json` は 10-12 23:43 のまま(この作業の前)。
- 一覧(28 データセット)に sim-rl-* は 11 個。今の方策のものに rl_superseded は無く、前の版のものには全部ある。

### 3.2 乱した操作が学習に入らないこと・正解のフレーム(`r3/review/mask_e2e_r4.py`。書き込みなし)

| 道 | データセット | 学習に使う観測 | 入力に l = 0 | 正解に l = 0 | 正解が k+1〜k+10 でない |
|---|---|---|---|---|---|
| A. JetFormerDataset(tools/train.py と同じ読み込み)train / val | recover_0302 | 5,802 / 1,425 | 0 / 0 | 0 / 0 | 0 / 0 |
| 〃 | no_wall_0302・line_0302 | 7,080 / 1,770 ずつ | 0 | 0 | 0 |
| 〃 | recover_1125(前の版) | 5,743 / 1,413 | 0 | 0 | 0 |
| B. シーンタブのシーンから作り直す(今の dataset.py の計算) | recover_0302・no_wall_0302・line_0302・recover_1125 | 7,227・8,850・8,850・7,156 | 0 | 0 | **0**(15 シーンとも「フレームの番号で引く」道) |
| C. B + シーンタブでジャンルを付け直す | 同じ | 同じ | 0 | 0 | 0 |

- 前の審査(1 回目)では道 B・C で正解の 1 歩目が 1 フレーム早い観測が 33% あった。昇格の段の `_frame_indexed` でこれが 0 になった(値で照合。VEHICLE.npz の整数の PWM と float の丸めの差 0.5 µs を同じとみなした)。
- `_frame_indexed` が実機のシーンで働かないことも data/raw の全部で確かめた: **実機のシーン 138 は 0 個**(今までどおり時刻で引く = 前と同じ計算)、展開したシミュレーターのシーン 267 は 267 個。
  data/raw には中身が 0 バイトの実機のシーンが 2 つある(`20261010-163247-0021`・`-0022`、10-10 16:39 のもの。この作業の前から。読めないので数えから外した)。

### 3.3 見本の記録とばらつき(昇格の段の数字)

- 記録の結果 `r3/promote/record/`: 45/45 シーン・外れ 0、開ループの再生・r4 の決まり・ユーザーの検査と同じ検査・記録そのものの数え直しが 3 モードとも 15/15。
  録ったシーンの教師の印は no_wall jf-shield/v5 8.1 cm・recover v4-proto 3.5 cm・line jf-shield/v5 **4 cm**(経路いろいろのシールドが本当に入っている。sim-rl.js が 03:04:57 に変わったが、表示だけの差で、line の記録は 03:39 以降)。
- `r3/variation.json` の表(5 モードの VM2・VM3・VM4・VM7・開始ゲート・1 周目)は promote.md 8 章の表と全部一致。
- **別のやり方で数え直した**(`r3/review/indep_variation_r4.py`。RL の参照ループを使わず、前の教師の 1 周を物差しにした横ずれ、x・z・向き・速さのマス):

| 群 | 線のばらつき(自前) | 状態のマス(自前) | 報告の VM3 / VM7 |
|---|---|---|---|
| 前の教師(15) | 0.28 cm | 435 | 0.35 cm / 278 |
| 最速 | 1.73 cm(6.2 倍) | 889(2.0 倍) | 1.82 cm / 768 |
| 絶対壁にぶつからない(r4) | 0.32 cm(1.1 倍) | 434(1.0 倍) | 0.41 cm / 299 |
| 離脱復活(r4) | 3.56 cm(12.8 倍) | 1,655(3.8 倍) | 4.11 cm / 1,466 |
| 速さいろいろ | 2.24 cm(8.0 倍) | 1,757(4.0 倍) | 2.17 cm / 1,250 |
| 経路いろいろ(r4b) | 9.72 cm(34.8 倍) | 2,060(4.7 倍) | 9.97 cm / 1,656 |
| 5 モード合わせ | 7.91 cm(28.4 倍) | 2,911(6.7 倍) | 7.96 cm / 2,635 |

  物差しが違うので倍率は違うが、順番(経路いろいろ > 離脱復活 > 速さ・最速 > 壁 ≈ 教師)と「壁にぶつからないは 1 本の線」は同じ。前の教師の舵のずれ VM1 7.562 µs(15 シーン)も再現した。

## 4. テスト一式 — `review/pytest_full.log`

- `jf-heavy --mem 2500M --lock big -- python -m pytest -q -p no:cacheprovider --basetemp=<作業用> tests`: **679 passed・3 skipped・86 warnings**(253 秒、exit 0)。
  昇格の段の 2 回目(679 passed・3 skipped)と同じ。新しい試験 `tests/test_sim_modes_r4.py`・`test_sim_label_frame_index.py`・`test_sim_label_mask_e2e.py` を含む。
- 前後の `git status --porcelain`(45 行)が同じ(試験がリポジトリを書き換えていない)。作業用の一時フォルダー(1.5 GB)は消した。

## 5. ユーザーのシミュレーター・ベストラップの作業のファイル — `review/sim_files_check.json`

- `data/sim/releases/20261013-144013-6f610dd5/`: COMPLETE-MANIFEST(sha256 6f610dd5…)の 851 のうち 849 一致・2 つはインストールで省いたもの(release の JSON の skipped)・不一致 0・欠け 0。
  いちばん新しいファイルの時刻 10-13 16:41:11(インストールのとき)。
- `/home/jetson/sim/extracted/course-simulator/`(元の展開): 851/851 一致。いちばん新しいファイル 10-13 14:51。
- `experiments/20261013_sim_rl_bestlap/` と `jetformer_reports/sim_rl_bestlap_20261013/`: 完了のコミット 9363a19(10-14 12:11)からの git の差分 0・未コミットの変更 0・そのあと書き換わったファイル 0。

## 6. 見つけたこと(判定・状態には関係しないもの)

| # | 見つけたこと | どうしたか |
|---|---|---|
| 1 | `deviations.md` に **line r4 の段の決まりの外のこと(`r3/line_r4/deviations_line_r4.md` の 1〜9)が写されていない**。line r4b の写しは参照だけで、項目 8 は「昇格の段が写す」と書いていた(確認 line r3 の手順の指摘 = r3 の学習の段の乱数の種・0.1 mm の丸め・nice 19 もここにしか無い) | **直した**: deviations.md の末尾「審査の段 第 3〜4 回」に全文を写した |
| 2 | `r3/promote.md` 8 章の表の「前の教師 VM7 279(1)」: 比の基準は 15 シーンの 278(`teacher.first15`)。比の値(9.5 倍など)は 278 で計算されていて正しい | deviations.md に訂正を書いた(promote.md は昇格の段の報告なので変えない)。README は 278 にした |
| 3 | 前の版の離脱復活の見本のゲートの外しを、記録の段(record.md 3 章)と審査 1 回目(review.md 8 章)が「2/15」としていた。正しくは 1125 が 6/15・1025 が 2/3(確認 recover r4 の数え直し、昇格の段が `rl_superseded` に記録) | deviations.md に訂正、README に正しい値 |
| 4 | 登録簿の line の `ui_note_ja`「経路の形はシーンの種から決まります」は手で走らせるときの話で、収集プランでは family_plan_15 の割合で回す(昇格の段で変えた) | 変えていない(手で走らせるときの注意としては正しい)。README に両方を書いた |
| 5 | 離脱復活の mode.json の dataset_card に「替えた出来事の割合 0.6764705882352942」と生の小数が出る(取り込んだデータセットのカードにも写る) | 変えていない(中身は正しい。68%)。次に mode.json を書くときに丸めるとよい |
| 6 | no_wall の status.json の注意「ブラウザの 1 フレームの時間は昇格の段で測る」は、昇格の段が測った(抜け出しの計画 115 ms・33 ms 超は 1 フレーム)あとも残っている | 変えていない(promote.md 6 章に値がある) |
| 7 | `r3/line_r4b/ff_profile/`(line r4b の段の Firefox のプロファイル、71 MB)が残っている | 変えていない(ほかの段のファイル。要らなければ消せる) |
| 8 | data/raw の実機のシーン 2 つ(`20261010-163247-0021`・`-0022`)の scene.json などが 0 バイト(10-10 の記録。この作業の前から) | この作業の外。触っていない |

## 7. 書き直した報告

- `README.md`(ユーザー向けのモードの案内): **最終の状態** で書き直した。結論(5 つ合格・障害物よけは出していない)、モードの表(状態・方策・タグ・手で走らせるときの注意)、
  使い方(1 シーン約 75 秒〜2 分、記録の前の確かめの決まり、replay_check_r4)、いま入っているデータセット 11 個(前の版の 4 つの扱い)、各モードの結果(試験と確認の数字)、
  ばらつき(5 モード、審査の数え直し)、限界(シールドに頼っていること・余裕)、次の手、経緯(何回目で合格したか)、ファイル。数字は全部、この段で確かめた JSON の値。前の版は `r3/review/before/README.md`。
- `video/storyboard.json`(1 分動画の絵コンテ): 7 場面・読み上げ 312 字・約 64 秒、時刻・予定のセリフなし、最後は「今日のまとめ」と「課題」。
  `check_storyboard.py` で OK。図は新しく `video/fig/r4_v1_modes.png`(6 モードの状態と実際のカードの画面)・`r4_v2_results.png`(学び直した 3 つの前 → 今)・
  `r4_v3_variation.png`(線のばらつきと軌跡)を、登録簿・昇格の前の写し・確認の段・variation.json・審査の数え直しの JSON から作った
  (`experiments/…/r3/review/video/make_slides_r4.py`・`make_storyboard_r4.py`)。前の絵コンテは `r3/review/before/storyboard.json`、前の図 v1〜v3 は残した。

## 8. 変えたファイル(変える前の写し `r3/review/before/`、sha256 は `before/sha256_before.txt`)

| ファイル | 変えたこと |
|---|---|
| `deviations.md` | 末尾に「審査の段 第 3〜4 回」(line r4 の 1〜9 の写し・数字の訂正 2 つ・この段の変更) |
| `README.md` | 最終の状態で書き直し |
| `video/storyboard.json` | 第 4 回の最終の状態で書き直し |
| `video/fig/r4_v{1,2,3}_*.png` | 新しく足した |

共有のコード(bundle・web/rl・rl/runtime・登録簿・JetFormer 本体・試験)・データセット・ユーザーのシミュレーターは変えていない。

## 9. ファイル・やり直し方

- コード `experiments/20261013_sim_rl_modes/r3/review/`(sha256 は `review/code_sha256.txt`): `check_protocols.py`・`check_bundles.py`・`recount_tests.py`・`check_datasets.py`・
  `mask_e2e_r4.py`(審査 1 回目の mask_e2e.py の写し)・`check_frame_indexed.py`・`check_sim_files.py`・`indep_variation_r4.py`(審査 1 回目の indep_variation.py の写し)・
  `video/make_slides_r4.py`・`video/make_storyboard_r4.py`。
- 結果 `jetformer_reports/sim_rl_modes_20261013/r3/review/`: `protocols.json`・`bundles.json`・`recount_tests.json`・`datasets.json`・`mask_e2e_r4.json`・`frame_indexed_raw.json`・
  `sim_files_check.json`・`indep_variation_r4.json`・`pytest_full.log`・`git_status_{before,after}_pytest.txt`・`api_{modes,datasets,scenes}.json`(API の写し)・`progress.log`・`before/`。

```bash
cd ~/jetformer/experiments/20261013_sim_rl_modes/r3/review
O=~/jetformer_reports/sim_rl_modes_20261013/r3/review; PY=/home/jetson/miniconda/envs/jetracer/bin/python
python3 -I check_protocols.py > $O/protocols.json; python3 -I check_bundles.py > $O/bundles.json; python3 -I recount_tests.py > $O/recount_tests.json
python3 -I check_datasets.py > $O/datasets.json; python3 -I check_sim_files.py > $O/sim_files_check.json
cd ~/jetformer && $PY experiments/20261013_sim_rl_modes/r3/review/mask_e2e_r4.py sim-rl-recover_20261015-0302 sim-rl-no_wall_20261015-0302 sim-rl-line_20261015-0302 sim-rl-recover_20261014-1125 > $O/mask_e2e_r4.json
$PY experiments/20261013_sim_rl_modes/r3/review/check_frame_indexed.py > $O/frame_indexed_raw.json; $PY -I experiments/20261013_sim_rl_modes/r3/review/indep_variation_r4.py > $O/indep_variation_r4.json
/home/jetson/bin/jf-heavy --mem 2500M --lock big -- $PY -m pytest -q -p no:cacheprovider tests
$PY -I experiments/20261013_sim_rl_modes/r3/review/video/make_slides_r4.py && python3 -I experiments/20261013_sim_rl_modes/r3/review/video/make_storyboard_r4.py
python3 ~/jetformer_reports/sim_cotrain_20261013/video/src/check_storyboard.py ~/jetformer_reports/sim_rl_modes_20261013/video/storyboard.json
```

## 10. 状態(段の終わり)

- jetformer-web: 03:02 に昇格の段が再起動した版で動作中(この段は再起動していない)。ブリッジ: 停止中・モデルなし(見つけたときと同じ)。学習・走行・記録・ジョブなし。
- この段が起動したプロセス(pytest・Python の確かめ)はすべて終わっている。git にはコミットしていない。
