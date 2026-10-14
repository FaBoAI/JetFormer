# 統合の段の報告 — 強化学習モードを JetFormer のシミュレーターで選べるようにした(sim_rl_modes_20261013)

- 作成 2026-10-14 11:20。段の役目: `design.md` のとおり、学習した強化学習モードを JetFormer が配るシミュレーターのページ(`/sim/c/default/`)で
  選べるようにし、モード × 分の収集プラン・取り込み(学習除外・タグ)・シミュレーター タブの状態表示・README・試験を作り、
  **http://192.168.55.1:8080(非ループバック。Mac から開くときと同じ)** の Firefox で確かめる。
- 依頼(ユーザー、10-13 21:20):「シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども
  作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意」
- 守ったこと: 実車・PCA9685・I2C・GPU に触れていない。ベストラップの作業のファイル・ユーザーのシミュレーターのファイルは変えていない(読むか写すだけ)。
  重い処理は jf-heavy(Firefox・pytest 一式は `--lock big`、Node は `--lock none`・nice 15)。git にコミットしていない。
  jetformer-web は 3 回再起動した(09:26・09:31・11:12。どれも学習・走行・収集・セッション・ジョブが無いことを API で確かめてから)。
  再起動で止まるブリッジは、そのたびに起動し直した(最初に見たときと同じ「動作中・モデル未読み込み・収集なし」)。

---

## 0. 要点

1. **4 つのモードを選べるようにした**(シミュレーターのページの「教師の走行」の下の「強化学習モード」、または `?rl=<key>`)。

   | モード | 状態(登録簿) | 選べる | 収集プラン | 主な数字(元の sim-core.js の試験) |
   |---|---|---|---|---|
   | 強化学習モード(最速) `fastest` | **合格**(F4 = ブラウザの同一性をこの段で確かめて合格に) | ○ | ○(シーンの組 90) | 停止から 1 周 8.09 秒・走りながら 7.66 秒・REC60 60/60 |
   | 強化学習モード(絶対壁にぶつからない) `no_wall` | **不合格(参考)** | ○ | ○(90) | 試験 340 回で衝突 0・新しい開始 600 回の最小すき間 7.0 cm・停止から 1 周 9.80 秒 |
   | 強化学習モード(離脱復活) `recover` | **不合格(参考)** | ○ | ○(75) | 乱した 100 回のうち 38 回が走り切る・立て直しの成功 76%・中央値 0.47 秒 |
   | 強化学習モード(速さいろいろ) `speed` | **合格**(r2) | ○ | ○(90) | 倍率 100/75/50% で 1 周 8.33/9.46/12.71 秒・60/60・最小すき間 8.4 cm |
   | 経路いろいろ `line`・障害物よけ `obstacle` | 出していない(優先 2・3 で不合格) | × | × | カードに理由と数字だけ(L2 1.33 cm・O1 66/100) |

   no_wall は学習の段では accepted だったが、確認 r2 が反証した(物理の範囲の中の角で 33 回中 30 回衝突・N2 の 8 cm は試験の組の引き)ので、
   protocol publication のとおり **不合格(参考)** で出し、画面の最小すき間は新しい開始の 7.0 cm を出した(deviations.md に記録)。
2. **ブラウザと Node の操作が同じ**(U5): 保存した snapshot の列 7 本(最速・no_wall の標準とシールドが働く遅い舵・離脱復活の乱しつき 2 本・速さ 100%/50%)を、
   8080 のページの teacher.js(import map で差し替えた教師)に与えると、Node の操作との差は **最大 8.2e-15**、info.jf の違い **0 フレーム**。
   no_wall のシールド v3 の「走りながらの見積もり」(フレーム 0 から連続・snapshot の舵角そのもの)もブラウザで同じに動いた。
3. **各モードがページで走る**(U1): `?rl=<key>` で開いて「▶ 評価を開始」→ sim の 10.2〜10.5 秒(305〜314 フレーム)・衝突 0・1 周。
   **モードを選ばなければ元の RC加速教師のまま**(U2): RL の教師は 0 回作られ、元の教師で 316 フレーム走った。
4. **収集プランが通しで動いた**(U3・U4): タブのカードで 4 モード × 1 分のプランを作り、新しいタブの「開始」だけで 13 シーン(1 つは離脱復活の衝突で替えた)を
   記録し、モードごとに 1 つのデータセットに取り込んだ(ジャンル 8・タグ・学習除外込み)。取り込んだ 12 シーンは全部
   **開ループの再生で姿勢の差 ≤ 7e-14 m・Node の教師の操作との差 ≤ 2.2e-12・info.jf の違い 0**。離脱復活の学習除外は 15 出来事・315 フレームで、決まり(k ∈ [j − 10, j])と全フレーム一致。
5. **pytest 一式 662 passed・3 skipped**(jf-heavy --lock big、4 分 47 秒)。新しい試験 `tests/test_sim_modes.py`(10)・`tests/test_sim_modes_js.py`(2)。
6. **分かった大事なこと**: ブラウザ(Firefox)と Node(V8)は Math.tanh などの最後の 1 桁が違うことがあり、**閉ループでは 54〜223 フレームで記録と Node の予想が分かれる**
   (最後には数十 cm 違う走りになる)。教師の操作そのものは同じ(上の 2・4)なので記録は「方策の走り」そのものだが、
   **「前もって Node で確かめたシーン」が記録でも同じ結果になるとは限らない**(離脱復活で 1 回、予想では走り切るシーンが記録で衝突した)。
   → 記録の段では、取り込んだシーンを `integrate/replay_check.mjs` で確かめ、モードの決まり(no_wall のすき間 8 cm など)を満たさないシーンを外すこと(9 章)。

---

## 1. 作ったもの・直したもの

### 1.1 JetFormer(リポジトリ)

| ファイル | 中身 |
|---|---|
| `jetformer/sim/rl_modes.py`(新規) | 登録簿 `data/sim_rl/modes/<key>/` の読み込みと確かめ(mode.json の形・status・`policy.json` の sha256 = mode.json の policy_sha256・シーンの組が同じ方策で作られたこと)、集めたデータセットの数(タグ `sim-rl-<key>`)、収集プラン(作る・進み具合・外れたシーンの替え・取り込みの状態)、`JETFORMER_SIM_RL=0` で全体を止める |
| `jetformer/sim/rl_import.py`(新規) | プランのシーン ZIP(1 シーン = 1 ZIP)をブリッジの集約と同じ形の 1 つのデータセットにまとめて取り込む。集約と同じ確かめ(1 シーン・600 フレーム・30 Hz・連続周回・衝突 0・合成ラベル・future_valid は最後の 10 だけ・manifest の sha256・契約(コースの版・経路・センサー・context.teacher・物理・換算・ジャンル表)がシーン間で同じ・同じ開始が 2 つない)。そのあと dataset_import と同じ道 |
| `jetformer/sim/web.py` | ① index.html の import map を **JSON として**書き足す(`./rc-speed-controller.js` → `./_jf/rl/teacher.js`、scopes `./_jf/` で元のファイル)② `sim-rl.js`・`sim-rl.css` の 2 行を差し込む ③ `/sim/c/<cid>/_jf/rl/<名前>` を `static/rl/manifest.json` の一覧 + teacher.js だけ配る ④ `_jf/config.js` に `rl`(使えるコースか・API)⑤ API `GET /api/sim/rl/modes`・`/modes/<key>`・`/modes/<key>/policy.json`(sha256 を確かめて `X-Policy-SHA256` 付き)・`GET|POST /api/sim/rl/plans`・`GET /plans/<id>`・`POST /plans/<id>/progress`・`POST /api/sim/rl/import`(ジョブ import。取り込んだら元のシーン ZIP と外れたシーンの ZIP を消せる) |
| `jetformer/sim/static/rl/teacher.js`(新規) | ページの教師の差し込み口。元の名前を全部再エクスポートし、`createRcTrainingTeacher`・`assertRcTeacherContract` だけを置き換える。モード未選択 → 元の関数そのもの。選択 → 開始の範囲(ゲート 0〜32・±0.3 m・±25°・経路の回し方・開始姿勢)を確かめ、元の検査に「ゲート 0 に回し戻した world」を渡して物理・雑音・20 秒を確かめ(低摩擦は拒否)、Node の試験と同じ `mode_runtime.mjs` の `createModeTeacher` で走る |
| `jetformer/sim/static/rl/*.mjs・*.json`(16 ファイル + manifest) | `experiments/20261013_sim_rl_modes/web/rl/` の写し(= rl/runtime/。sha256 は manifest どおり。試験で照合) |
| `jetformer/sim/static/sim-rl.js`・`sim-rl.css`(新規) | ページの「強化学習モード」の欄(状態・説明・速さの倍率・乱しの強さ・走行中のすき間/シールド/乱し)、上の帯の「RL: <モード>」、`?rl=`、収集プランの実行係(`?rlplan=`)。連続試験・収集の間は選べない |
| `jetformer/sim/label_mask.py` | 学習除外を **k ∈ [j − 10, j]**(乱しの区間なら [ps − 10, pe − 1])に広げた。**1 シーンの ZIP の形(`simulation/frames.jsonl`)も読む**(それまでは集約の形だけで、1 シーンの ZIP の RL シーンは除外されずに通っていた) |
| `jetformer/dataset_import.py` | 取り込みの後半を `check_staged`・`after_import` に分けた(まとめての取り込みと共有。動きは同じ)。RL のシーンには `sim-rl`・`sim-rl-<モード>`・登録簿のタグを自動で付ける(「JetFormerへ送る」の道も) |
| `jetformer/scene_import.py`・`jetformer/dataset.py` | 展開したシーンの scene.json に `label_mask.unset_segments` を残し、`update_scene` でジャンルを付け直しても「未設定」の区間を残す(確認 recover r2 の must-fix) |
| `jetformer/web/static/{index.html, app.js, style.css}` | シミュレーター タブのカード「🤖 強化学習モード」(状態・停止から 1 周・衝突・最小すき間・主な数字・試験の結果と注意・集めたデータ・「このモードで開く」・収集プランの表(モード × 分)・val の割合・種・元のシーンを消す・「プランを作って開く」・最近のプランと取り込み) |
| `README.md` | 「強化学習モード(データセットのばらつき)」の節(使い方・データセット・しくみ・API・止め方) |
| `tests/test_sim_modes.py`(新規 10)・`tests/test_sim_modes_js.py`(新規 2)・`tests/test_sim_label_mask.py`(数を新しい決まりに) | 登録簿の状態と確かめ・方策の配信・無効化・import map・RL のファイルの配信と sha256・config・プラン(シーン数・split・開始が重ならない・項目・場面の回し方・替え)・拒否・進み具合・まとめての取り込み(学習除外・タグ・表の重ね・元/外れた ZIP の削除・契約違い/衝突/同じ開始の拒否・半端なフォルダーを残さない)・ジャンルの付け直しで未設定が残る / teacher.js(委ねる・再エクスポート・元の検査に回し戻した world・2 回呼び・p・禁止の鍵・metadata がシーン間で同じ・フレームの飛び・検査の拒否 6 種・速さの上限)/ sim-rl.js(要素が無くても止まる) |

### 1.2 RL の側(`experiments/20261013_sim_rl_modes/integrate/`)

| ファイル | 中身 |
|---|---|
| `make_scene_pool.mjs`・`run_pools.sh` | 記録に使う「前もって Node で確かめたシーンの組」(4 章) |
| `publish_registry.py` | web/rl の写しを `jetformer/sim/static/rl/` へ(sha256 を照合)、登録簿 `data/sim_rl/modes/<key>/` を作る(学習の書き出し + 確認の段の判定 VERDICTS + カードの要約 + シーンの組)。`--browser-fid` で最速の F4 を判定。**後の段でモードを作り直したら、これを走らせ直せばよい** |
| `browser_check.py`・`run_checks.sh`・`run_checks2.sh` | 8080 の Firefox での確認(F-ID・各モードの走行・モード未選択・プラン)と pytest 一式 |
| `replay_check.mjs` | 取り込んだデータセットを Node で確かめる(開ループの再生・教師の同一性・正確なすき間と決まり・予想との違い) |

結果: `integrate/`(pools/・browser/result.json・browser2/result.json・browser_fid.json・replay/*.json・pytest_full.log・publish_last.json・sha256.txt)、画面 `screens/`。

---

## 2. 登録簿(`data/sim_rl/modes/<key>/`)

- 中身: `mode.json`(学習の段の書き出しのまま)・`status.json`(確認の段の判定で上書き。学習の段の値は `from_training_status` に残す)・`policy.json`・`eval.json`・
  `card.json`(画面の要約。数字は eval.json と確認の段の報告から。no_wall の最小すき間は新しい開始の 7.0 cm、試験の組の 11.7 / 7.0 cm も併記)・`scenes.json`(シーンの組)。
- 方策の sha256: 最速 `7c43cb53…53ee`(ベストラップの確認 r1・r3 で本物。その作業にこれより新しい「最速」の方策は無い — r3 の 2 本は 8.5〜8.6 秒で遅い)・
  no_wall `153c709c…`(r2、シールド v3)・recover `7720f275…`・speed `6f4d5e8d…`(r2。r1 は登録しない、PUBLISH.json のとおり)。
- 状態の決め方は protocol publication: 合格 → 合格、優先 1 で不合格 → 不合格(参考)、優先 2・3 で不合格 → 出さない(方策を置かない。カードには理由だけ)。
- サーバーは `policy.json` の sha256 が mode.json と違えば配らない(差し替えの途中・壊れたとき)。シーンの組が別の方策のものならプランを作らない(モードは選べる)。

---

## 3. ページ(teacher.js・sim-rl.js)

- import map は文字列の置き換えでなく JSON として書き足し、元の `three` などの対応はそのまま。`JETFORMER_SIM_RL=0` で差し込みも API も止まる(試験)。
- 強化学習モードを選ぶと: 方策を読み(ヘッダーの sha256 が登録簿と同じか確かめる。非ループバックの http では crypto.subtle が無いので、ページでは sha256 を計算しない)→
  コントローラー = 経路追従デモ・教師 = RC加速教師の口(選択肢の文字を「強化学習: <モード名>」に)・物理とセンサー = `RC_COLLECTION_PLAN[0]`・連続周回・20 秒 → リセット。
  開始ゲート・ずれ・種・場面は利用者の値を残す(低摩擦なら baseline に戻して知らせる)。
- 教師の `metadata`(context.teacher)はモードの中の全シーンで同じ(シーンごとの条件はフレーム 0 の `info.jf.p` にだけ入る)。試験で確かめた。
- 走行中の表示: フレーム・走行/乱し中/シールド作動・すき間(最小)・シールドのフレーム数・乱しの数と学習から外す操作の数・速さの倍率・種。
- 強化学習モードは Default コースだけ(ほかのコースでは欄が使えず理由を出す)。障害物よけの仮想コース `default-obstacles` は、モードを出していないので作っていない。

---

## 4. 収集プランとシーンの組

- プランはサーバー(Python)が作る: モードごとに 分 × 3 シーン、val は等間隔(5 分なら 15 シーンのうち 3)、場面は baseline・baseline・noise・dim の順、
  シーン名 `rl-<key>-<train|val>-<NN>`、データセット名 `sim-rl-<key>_<日時>`(足りないまま取り込むと `_<数>of<予定>`)、替えのシーンは 25% + 2。
  アプリに渡す項目は `name・dataset-split・seed・start-gate・spawn-offset・spawn-yaw・scenario・controller・teacher-profile・episode-mode・limit・dataset-mode` だけ
  (センサー・物理はページが RC_COLLECTION_PLAN[0] から足す)。
- 開始・種・条件は **シーンの組** から選ぶ。組は統合の段で Node(元の sim-core.js + ブラウザと同じ mode_runtime.mjs、`rl/episode.mjs`)で前もって走らせ、
  20 秒・衝突 0・有効な周 ≥ 1・止まる(0.2 m/s 未満)のは 1 秒まで・ゲートの外し 0(recover だけは許す)・すき間 ≥ mode.json の prefilter_min_clearance_m
  (no_wall 8 cm・speed 3 cm)を満たしたもの:

  | モード | 候補 | 合格 | 落ちた理由 | 組の最小すき間 | 時間 |
  |---|---|---|---|---|---|
  | fastest | 98 | 91(組 90) | not_valid 5・周 < 1 が 6・ゲートの外し 4・止まる 2 | 1.44 cm | 115 秒 |
  | no_wall | 92 | 91(90) | すき間 < 8 cm が 1 | 10.0 cm | 134 秒 |
  | recover | 246 | 75 | not_valid 171・周 < 1 が 105 | 0.06 cm | 228 秒 |
  | speed | 91 | 91(90、倍率 5 段を順に) | なし | 7.3 cm | 87 秒 |

- ページの実行係: 「開始」→ ブリッジの換算を読む(モードを読み込んでいれば操舵の中心 1442 ±5・スロットルの中立 1498 ±5 µs でなければ止める。途中で変わっても止める)→
  自動ラベルの印・ジャンル 8 → シーンごとに モードの方策を読み・条件を置き・計画欄に 1 条件だけ入れてアプリの「連続試験を開始」を押す → 結果(600 フレーム・衝突 0・session_id)を
  JetFormer に送る → 外れたら(衝突など)替えのシーンを足す(サーバー)→ モードのシーンが全部終わったらそのモードを取り込ませる。2 シーン続けて記録できなければ止まる。
- 取り込み: 学習除外(下)・タグ(mode.json の tags = `simulator・sim-rl・sim-rl-<key>`(+ `sim-safe`・`sim-recover`・`sim-speed`))・カードの `rl`
  (モード・状態・方策・プラン・シーン数・分・換算)と `rl_notes`(recover の dataset_card の注意)・`simulation/collection.json`(シーンごとの開始・種・条件・予想の操作列の sha256)・
  シーンタブへの展開。プランで「元のシーンを消す」を選ぶと(既定)、取り込んだシーンと外れたシーンのブリッジの ZIP を消す。

---

## 5. 学習除外・ジャンル

- ジャンルは **全部 8(シミュレーター)のまま**(design D11・8 章。収集器は 0〜5・8 だけ、シミュレーターの推論は num_modes ≤ 9)。**ジャンルの登録はしていない**
  (`data/modes.json` は変えていない)。モードはタグとデータセット名で分かれる。
- 乱した操作(`info.jf.l = 0`)を正解に含む観測と、その操作のフレーム自身を `future_valid = False`・`mode = 0` にする(k ∈ [j − 10, j])。jf の無い行・読めない frames.jsonl は外す(安全側)。
  設計の [ps − 10, pe − 2] から 1 つ広げた理由は deviations.md(シーンからのデータセットの作り直しで正解が 1 フレーム前にずれるため)。
- シーンタブでジャンルを付け直しても、外した区間は「未設定」のまま残る(試験: 付け直したあと 10〜24 フレームだけ 0、ほかは新しいジャンル)。

---

## 6. 確かめたこと

### 6.1 試験

- **pytest 一式**(jf-heavy --mem 2500M --lock big、11:00〜11:05): **662 passed・3 skipped**(`integrate/pytest_full.log`)。最後の小さな画面の文言の直しのあとに、
  シミュレーターと取り込みの試験 7 ファイル(70)をもう一度通した。
- Node: 実験の runtime・shield v3 の試験 10/10(rl/runtime を変えていないことの確かめ)。

### 6.2 ブラウザ(http://192.168.55.1:8080、DISPLAY :0 の snap Firefox 157.0.1、geckodriver 4447、プロファイルは reports の下)

ページは非ループバックの http(`isSecureContext` = false、`crypto.subtle` なし、`crypto.randomUUID` は shim が補う)で動いた。

- **F-ID(U5)**(`integrate/browser2/result.json`・`browser_fid.json`):

  | モード | 列 | フレーム | 舵の差(最大) | 速さの差(最大) | info.jf の違い |
  |---|---|---|---|---|---|
  | fastest | eval/fastest/fid_std.json | 600 | 3.9e-16 | 2.4e-15 | 0 |
  | no_wall | fid_std_no_wall_r2.json | 600 | 3.3e-16 | 2.7e-15 | 0 |
  | no_wall | fid_slow_no_wall_r2.json(舵の速さ 0.8・遅延 1、シールド v3 が見積もりで働く) | 600 | 2.2e-16 | 1.3e-15 | 0 |
  | recover | fid_scene_P5.json(乱しつき) | 600 | 3.9e-16 | 6.2e-15 | 0 |
  | recover | train/recover/fid_std.json | 118 | 2.8e-16 | 1.3e-15 | 0 |
  | speed | fid_std_speed_r2_cv1.json | 600 | 5.0e-16 | 8.2e-15 | 0 |
  | speed | fid_std_speed_r2_cv0.5.json | 600 | 3.3e-16 | 1.1e-15 | 0 |

  比べのため Node でも teacher.js(本物の rc-speed-controller.js)を通して同じ 7 本を与え、差 0・info の違い 0。metadata も Node の試験と同じ。
- **各モードの走行(U1)**(実行速度「高速」、sim の約 10 秒。`screens/drive_<key>_{window,panel,camera}.png`):

  | `?rl=` | フレーム | sim 秒 | 衝突 | 周 | 走行中の様子 |
  |---|---|---|---|---|---|
  | fastest | 314 | 10.47 | 0 | 1 | 最小すき間 6.3 cm |
  | no_wall | 313 | 10.43 | 0 | 1 | 最小すき間 13.5 cm・シールドの置き換え 0 |
  | recover | 305 | 10.17 | 0 | 1 | 乱し 2 回・学習から外す操作 51 フレーム・最小すき間 7.6 cm |
  | speed | 313 | 10.43 | 0 | 1 | 倍率 87.5%(種から)・最小すき間 10.0 cm |
  | rc(モード未選択、U2) | 316 | 10.53 | 0 | 1 | RL の教師は作られていない(元の RC加速教師) |

  1 回目の確認(実行速度「通常」、壁時計の 10 秒 = sim の約 1.8 秒)も 4 モードとも衝突 0。1 回目の「モード未選択」は、確認の手順が物理・センサーを
  記録の条件にそろえていなかったため元の RC 教師の検査で止まっていた(機能の不具合ではない)ので、2 回目でそろえて確かめた。
- **カード(U3)**: `screens/tab_rl_card.png`・`tab_rl_card_after.png`・`tab_rl_plan_form.png`。6 モードの状態・数字・注意、「N 回の試験で衝突 0」の書き方。
- **収集プラン(U4)**(`integrate/browser/result.json`、10:26〜10:49): カードで最速・壁にぶつからない・離脱復活・速さいろいろ を各 1 分 →「プランを作って開く」→
  新しいタブで「開始」(`screens/plan_before_start.png`・`plan_recording.png`・`plan_panel_after.png`・`tab_rl_after_plan.png`)。

  | シーン | モード | 結果 | 周(秒) |
  |---|---|---|---|
  | s000〜s002 | fastest | 600・衝突 0 ×3 | 8.31/7.81・7.93/7.68・8.24/7.69 |
  | s003〜s005 | no_wall | 600・衝突 0 ×3 | 9.78/9.50・9.54/9.47・9.85/9.49 |
  | s006・s007・s008 | recover | 600・衝突 0 ×2、**s008 は 474 フレームで衝突** → 替え s008r1 が 600・衝突 0 | 17.67・8.52/8.32・(s008r1)8.64 |
  | s009〜s011 | speed | 600・衝突 0 ×3 | 8.44/7.98・8.30/8.29・9.40/8.99 |

  1 シーン約 1 分 40 秒〜2 分(Jetson の Firefox、記録と ZIP 込み)。モードごとに自動で取り込み、データセット 4 つ(各 3 シーン・1,800 フレーム・ジャンル 8・
  タグどおり)。換算はモード未読み込み(既定 1441/1496 µs)。離脱復活は mode 0(未設定)315 フレーム = カードの label_mask 315、学習に使えるフレーム 1,455/1,800。
  ほかの 3 つは学習除外 0(乱しが無い)。取り込んだあと、元のシーン ZIP は消えた(外れた s008 の ZIP は手で消し、そのあと自動で消すようにした)。
- **取り込んだ記録の確かめ**(`integrate/replay/<key>.json`、Node): 12 シーン全部で
  開ループの再生の姿勢の差 ≤ **6.8e-14 m**、同じ snapshot に対する Node の教師の操作の差 ≤ **2.2e-12**、info.jf の違い **0**(全 7,200 フレーム)、
  決まり(衝突 0・有効な周 ≥ 1・止まらない・no_wall のすき間 ≥ 8 cm(11.9・13.4・13.3 cm))**全部満たす**。離脱復活の学習除外は 3 シーン・15 出来事で
  決まり k ∈ [j − 10, j] と全フレーム一致、シーンタブの 3 シーンに `label_mask.unset_segments`(6・4・5 区間)。

### 6.3 閉ループの予想と記録の違い(大事)

- 同じ開始・種で Node が閉ループで走らせた走り(シーンの組を作ったときの予想)と記録は、**54〜223 フレーム目から操作が分かれ**、最後には 0.1〜1.4 m 違う走りになった。
  最初の違いは方策の出力の最後の 1 桁(8e-17 など)で、Firefox と Node(V8)の Math.tanh などの実装の差。閉ループで育つ。
- 教師の操作そのもの(同じ snapshot → 同じ操作)と sim-core の再生は 1e-12 以下で同じ(上)。つまり**記録は「その方策の走り」そのもの**で、ラベル・印・すき間は
  Node で作り直せる。ただし「Node で確かめたシーン」が記録でも同じ結果になる保証は無い(今回 13 シーン中 1 つ、離脱復活が記録で衝突し、替えのシーンを使った)。
- 衝突したシーンは 600 フレームにならないので取り込まれない(実行係が替える)。**すき間・止まる・周の決まりは記録では確かめていない**ので、記録の段で 9 章の確かめをする。

---

## 7. 確認の段の指摘への対応(統合の段で受け持つもの)

| 指摘 | 対応 |
|---|---|
| no_wall: 物理の範囲の端でぶつかる → 直すか「不合格(参考)」 | 不合格(参考)で出し、カードに理由と注意。シールドは外せない(画面にも外す道は無い) |
| no_wall: 画面の 11.7 cm は代表的でない | カードの最小すき間を新しい開始 600 回の 7.0 cm にし、試験の組の値を併記 |
| no_wall・speed・recover・obstacle: 止まったまま・周 0 のシーンが記録に通る | シーンの組の決まりに 有効な周 ≥ 1・止まる ≤ 1 秒 を全モードに入れた。記録したシーンも replay_check.mjs で確かめられる |
| no_wall: 記録の前の予想にすき間 8 cm を必ず使う | シーンの組で使った(落ちたのは 1)。記録したシーンは 11.9〜13.4 cm |
| no_wall: ブラウザでシールド v3 の見積もりが働くか | fid_slow_no_wall_r2.json(見積もりが働く列)を含め F-ID が差 1e-15 以下・info 一致 |
| low-friction を止める | teacher.js が拒否(元の検査も μ 1 を求める)。ページは選んでいたら baseline に戻す。プランは場面に使わない |
| recover: ジャンルの付け直しで未設定が消える | `update_scene` が scene.json の `label_mask.unset_segments` を残す(試験) |
| recover: シーンからの作り直しで正解が 1 フレームずれる | 学習除外を [ps − 10, pe − 1] に広げた(試験) |
| recover: mode.json の「約3割」→ 約4割 | カードの注意は「約 4 割」と書いた(mode.json そのものは学習の段のファイルなので変えていない) |
| speed: r2 だけを登録 | train/speed/r2/export を登録(r1 は登録しない) |
| obstacle・line: 登録しない | 方策を置かず、状態 not_published(カードに理由だけ) |
| 1 シーンの ZIP の RL シーンが除外されずに通る(この段で見つけた) | label_mask.py が `simulation/frames.jsonl` も読む |

---

## 8. 逸脱(deviations.md に追記)

1. 学習除外の範囲を [ps − 10, pe − 1] に広げた。1 シーンの ZIP の形も読むようにした。
2. 記録の前の予想を「前もって確かめたシーンの組」にした(サーバーで Node を動かさない)。種の式 `700000 + 1000·m + 候補の番号`、プランの通し番号は選ぶ窓の位置に使う。
   決まりに 有効な周 ≥ 1・止まる ≤ 1 秒・ゲートの外し 0(recover 以外)を全モードに足した。recover は候補 246 本(プランの 3 倍を超える)。
3. no_wall を不合格(参考)で出した(学習の段は accepted)。
4. 08:57 に `rl/make_web_bundle.mjs` をうっかり走らせ、`web/rl/manifest.json` の created_at だけが変わった(16 ファイルの中身は同じ。manifest の sha256 は affabdf4…)。

---

## 9. 次の段(記録)へ

- 見本の記録は、タブのカードで分を入れて「プランを作って開く」→ 新しいタブで「開始」だけでできる(ユーザーが Mac から同じことをできる)。
  Jetson の Firefox で 1 シーン約 1 分 40 秒〜2 分(1 回目の見積もり 70 秒より長い。DISPLAY :0 の画面で記録したため)。5 分 × 4 モード = 60 シーンで約 2 時間。
- **取り込んだら `node integrate/replay_check.mjs --dataset data/datasets/<名前>` で確かめる**(開ループの再生・教師の同一性・決まり)。
  決まりを満たさないシーン(特に no_wall のすき間 8 cm 未満、止まる、周 0)があれば、そのシーンを外してデータセットを作り直す(または外れとして替えを記録する)。
  閉ループでは予想と記録が分かれるので、シーンの組の予想だけでは足りない(6.3)。
- この段の確認で作ったデータセット 4 つ(各 1 分 = 3 シーン、`sim-rl-{fastest,no_wall,recover,speed}_20261014-1025`、展開したシーン 12)と
  プラン `20261014-102558-e09caf` は残してある(本物の記録で、全部確かめ済み)。要らなければ データセットタブ・シーンタブ(ログ `ds-sim-rl-*_20261014-1025`)から消せる。
- 学習の段(r3 など)がモードを作り直したら、`python integrate/publish_registry.py`(ブラウザの組の写しと登録簿)→ 必要ならシーンの組
  (`integrate/run_pools.sh` の該当モード)→ `publish_registry.py` をもう一度。rl/runtime を変えたら `rl/make_web_bundle.mjs` で web/rl を作り直してから。
  ブラウザの確認は `jf-heavy --mem 2500M --lock big -- python -u integrate/browser_check.py --skip-plan --drive-sim-seconds 10`。

---

## 10. 状態(段の終わり)

- jetformer-web: 11:12 に再起動した版(この段のコード)で動作中。ブリッジ: 動作中・モデル未読み込み・収集なし(最初と同じ)。ブリッジの作業場所にシーン ZIP は残っていない。
- 起動したプロセス(シーンの組の Node・ブラウザの確認・Firefox・geckodriver・pytest)はすべて終わっている。Firefox のプロファイル(137 MB)は消した。
- 空き容量 約 16 GB(データセット 4 つと展開したシーンで約 0.24 GB 増えた)。git にはコミットしていない。
