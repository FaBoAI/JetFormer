# 確認の段(review)— sim_rl_bestlap_20261013 の報告を生の記録から数え直す

- 作成 2026-10-14。対象: `README.md`・`evaluation.md`・`evaluation.json`(報告の段 11:10〜11:15 の版。写しは `review/*.before_review.*`)。
- やり方: 報告の数字を 1 つずつ生の記録(`eval/closed_loop/*.json`・`rl/final/records/*.json`・`recording/summary.json`・`train/offline_l1.json`・
  `eval/takeover.json`)に当てた。周の正しさは、報告の段の道具を使わず、**Python で書き直した積分と判定**(`experiments/…/review/independent_lap_check.py`。
  JS のコードを使わない)と、ユーザーの確認器で確かめた。重い処理は `jf-heavy` を通した。実車・GPU・Firefox は使っていない。git にはコミットしていない。
- コードは `/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/review/`、出力は `review/`。

## 結論

| 確かめたこと | 結果 |
|---|---|
| 判定(H1 成立・H2 できなかった) | **そのまま正しい**。H2 の基準(a)(b)(c)を protocol.json のとおりに当て直しても「できなかった」(S-R の完走 0/12) |
| 数字の出どころ | 主な数字はすべて生の記録で再現した。ただし **README の「失敗の形」の 6 か所に言い過ぎ・書き違い**があり、直した(下の表)。図から読んだだけで JSON に無かった数字は、`review/derived_numbers.json` に計算の形で残し、evaluation.md 5.2 節に出した |
| 手順の固定(sha256) | **変えていない**。protocol.json の sha256 `1f249211…b8b6` は protocol.sha256 と同じで、パイロット(10/13 17:45)・本番の RL・記録・学習・評価のどの記録にも同じ値が残っている |
| 手順と違ったこと | deviations.md に全部ある。抜けていた注意 2 つ(描画の違い・シーンタブからの作り直し)を README 7 章に足した |
| 周の正しさ(ランダムな開始 2 つ) | **有効**。P1 = ゲート 0・+2.0 cm・−2.92°: 停止から 8.082 秒、連続 8.081 / 7.810 秒。P2 = ゲート 6・−4.6 cm・+7.00°: 7.940 秒、連続 7.939 / 7.789 秒。すべて衝突 0・33 ゲート・黒シート 1 回 |
| 閉ループの 59 走行 | 自前の積分で **59/59 が sim と一致**(軌跡の差 ≤ 5.8e-13)。ゲートの外し方・ぶつかった相手・時刻も evaluation.json と同じ |
| データセット | **取り込める・一覧に出る・シーンに展開できる・ジャンル 8・タグ正しい**。今のコードで別の置き場に取り込み直すと、本物と 72,378 ファイルが同じ(違うのは取り込みの時刻の 1 ファイル)。注意: シーンタブから作り直すと正解の 1/3 が 1 フレームずれる(下) |
| テスト一式 | 703 passed・5 failed・3 skipped。**失敗 5 件はほかの作業のもの**で、この作業の変更によるものではない(この作業は jetformer/ のコードを変えていない。この作業の試験 13 件は合格) |
| ユーザーのシミュレーター | **変わっていない**。COMPLETE-MANIFEST.json の 851 ファイルすべてと 20 分データの ZIP が sha256 で一致 |

## 1. 数字の数え直し

### 1.1 そのまま正しかった数字(生の記録で再現)

| 数字(README) | 出どころ | 確かめ |
|---|---|---|
| RL 8.092 秒・フライング 7.656 秒、教師 8.931・8.370 秒、差 0.839 秒(9.4%) | `rl/final/records/*.json` | 自前の積分で 8.091953・7.656319・8.930784・8.370092(sim と差 ≤ 6e-13) |
| E0〜E5 の RL 8.085〜8.131 秒・教師 8.931〜8.982 秒 | 同上 | 18 本すべて一致 |
| 選べた種 3 本(8.522・8.092・8.401) | `rl/selection.json` | 一致 |
| 記録: 1 周目 8.092・2 周目 7.731 秒(教師のデータより 0.65・0.64 秒速い)、舵が端 18.8%(教師 3.7%)、変化 58 対 18 µs、すき間 5.7 cm | `recording/summary.json`・`label_stats.json` | 一致。ランダムな 2 シーン(種 2708978707)も自前の判定で 2 周とも有効 |
| 学習 L1(17.5・2.3 µs など)・オフライン L1 の表 | `train/summary.json`・`train/offline_l1.json` | 一致 |
| 閉ループの表(完走 0/12・2/12・2/6、8.830・8.840・9.022・9.033 秒、ゲートの外し方 1.9〜33.4 cm、衝突 0.87〜1.35 秒) | `eval/closed_loop/*.json`・`evaluation.json` | 自前の積分・判定で 59/59 一致 |
| フライングの無効な周(17.25〜17.39 秒) | 同上 | 33 ゲートだが黒シートの有効な横断が 2 回(自前でも同じ) |
| 参考の追加条件の表(遅延 0・先読み 2・ゲイン 0.8〜1.01) | `eval/runs.csv` | 一致 |
| 点 4 のゲート: 教師 0.628 m(端まで 2.2 cm)、S-T 種 0 の E1 0.644 m | 自前の積分の副ステップで計算 | 0.6281・0.6443 m で一致(30 Hz の姿勢をつなぐと 0.630・0.645 になるので、副ステップで数える必要がある) |
| 引き継ぎ: S-R は衝突の 4〜10 フレーム(0.13〜0.33 秒)前まで RL が立て直せる | `eval/takeover.json` | 12 本すべて一致 |
| 1 フレーム先の舵 29.6 対 4.1 µs ≒ 0.03 対 0.004 rad | PWM の換算(中心 1442 µs、0.52 rad = 442〜558 µs) | 0.028〜0.035 対 0.004〜0.005 rad。言い方の範囲 |

### 1.2 直したこと(README。数字の出どころは evaluation.md 5.2 節・`review/derived_numbers.json`)

| # | 直す前 | 生の記録 | 直した後 |
|---|---|---|---|
| 1 | 「RL の線は最初の直線で**レール**まで約 10 cm(教師は**同じ所**で 29 cm)」 | いちばん近いのは**幕**(segment-43)10.4 cm。ぶつかったレール(segment-11)までは 13.9 cm。教師の 29.0 cm は同じ区間の最小で相手は柱の脚(場所も違う)、レールまでは 34.1 cm | 「右の壁(z = 2.8 m の幕・レール・柱の脚)まで約 10 cm(教師は同じ区間で最小 29 cm)」 |
| 2 | 「**速い線ほど余裕がない**」 | 周全体の最小すき間は RL 7.1 cm・教師 6.7 cm で同じくらい(REC60 では RL 5.7 cm・教師 3.2 cm)。点 4 のゲートの端までは RL 13.8 cm・教師 2.2 cm | 「どちらの線にも余裕の小さい所が 1 か所ある(場所が違う)」 |
| 3 | 「RL の線からのずれは **0.5 秒で 0〜6 cm**、ぶつかる時点で 10〜18 cm」 | 0〜6 cm は E0 の線からの値で、E1〜E4 の開始のずらし(±3〜8 cm)を含む。同じ開始の RL の走りからは 0.1〜2.8 cm。ぶつかる時点は 10〜18 cm(E0 の線から)・11〜19 cm(同じ開始から) | 「同じ開始の RL の走りからのずれは 0.5 秒で 0〜3 cm、ぶつかる時点で 11〜19 cm(E0 の線からは 10〜18 cm)」 |
| 4 | 「毎回 1 秒前後で**レールか幕**にぶつかる」 | レール 8・幕 2・柱の脚 2(S-R 種 0 の E5、S-R 種 1 の E2) | 「右の壁(レール 8 本・幕 2 本・柱の脚 2 本)」 |
| 5 | 「RL のデータは**約 7 倍**まねしにくい」、表の「10 ステップの平均」 | 学習中の検証 7.6 倍・1 フレーム先の舵 7.2 倍だが、全フレームの舵とスロットルの 10 ステップの平均では 19.6 対 4.7 µs で 4.2 倍。表の値は舵とスロットルの平均 | 「4〜7 倍」と物差しごとの値。表の見出しを「舵とスロットルの 10 ステップの平均」 |
| 6 | 引き継ぎ: S-T は「1.5〜1.7 秒から任せると RL もぶつかる」「S-T 種 1 の E0 では 1.5 秒より後のどこからでも完走できない」 | S-T の 10 本とも 1.5 秒(S-T 種 1 の E2 は 1.4 秒)から任せるとぶつかる。完走しなかった 8 本はそれより後のどこからでも完走できない。完走した 2 本は 1.5〜1.7 秒のときだけ | そのとおりに書き直した |

- 判定・表・図・数値の結果は変わらない。evaluation.md の表はもともと正しかった(evaluation.md は生成なので、5.1・5.2 節を足して作り直した)。
- 報告の段の StructuredOutput(段のまとめ)にも 1〜3 の言い方が残っているが、後から直せないので、この review.md と README が正。

### 1.3 足した注意(README 7 章)

1. **描画の違い**: S-T と作者の ONNX は作者の環境(Mac の GPU)で描いた画像で学習したモデル。評価はこの Jetson の llvmpipe で描いた画像(作者のデータとの差は PSNR の中央値 42.8 dB、
   `webgl_probe/compare_probe-*.json`)。S-R は同じ llvmpipe の画像で学習した。S-T にわずかに不利だが、S-R が完走しなかったので H2 の判定は変わらない。
2. **シーンタブからの作り直し**: 取り込んだデータセット(学習に使ったもの)の正解は 100% 正しい(`future_actions[k]` = k+1〜k+10 の操作)。ところが、シーンタブの 60 シーンから
   データセットを作り直すと、今の `dataset._aligned_arrays` は正解を時刻(33,333.33 µs おき)で引くので、sim のフレームの間隔(33,333 / 33,334 µs)のため
   **正解の 33% が 1 フレームずれる**(舵の正解の差 平均 19.5 µs。S-R の検証 L1 17.5 µs と同じくらいの大きさ。`review/dataset_check/rebuild_label_shift.json`)。
   ほかの作業(sim_rl_modes)が直し方(`dataset_frame_index.patch`)を用意している(テスト一式の失敗のうち 3 件が、この直しを前提にした試験)。それまでは、取り込んだデータセットをそのまま使う。
3. **シーンのタグ**: データセットのタグは `simulator` + `rl-bestlap` だが、シーンタブの 60 シーンのタグは `simulator` だけ(取り込みの仕組みどおり。protocol もデータセットのタグだけを決めていた)。
   シーンタブで RL のシーンを探すときは、ログ名 `ds-jetformer-simulator-rl-bestlap-20min_…` で探す。

## 2. 手順の固定(protocol.json)

- `sha256sum protocol.json` = `1f249211704c0065b87645d5bf93d959fcd65205fd4c2f4e535a1c310e81b8b6` = `protocol.sha256`。ファイルの時刻は 10/13 16:47:26(固定の時刻)のまま。
- 同じ値が、パイロット(`rl/pilot-main-s100/run.json`、17:45 開始)・本番(`rl/main-s1/run.json`、19:04)・記録(`recording/install.json`)・学習(`train/state.json`)・
  評価(`evaluation.json`)に残っている。design.md の時刻も 10/13 16:45 のまま。
- H2 の基準(a)(b)(c)・Δ・予備の基準・判定の言葉、評価の設定(遅延 1・先読み 1・ゲイン 1.0・ジャンル 8、外部の参照はジャンル 4・前進ゲイン 1.01)・開始 E0〜E5・
  単周の上限 15 秒は、protocol.json と evaluation.* で同じ。

## 3. 周の正しさ(ランダムな開始 2 つ、別に書いた道具)

- 開始は `secrets` の乱数(種 2713733778、`review/random_starts/draw.json`)で引いた。P1 は記録の 60 条件の範囲(ゲート 0・±3 cm・±3°)、P2 は RL の学習の開始の範囲(ゲート 1〜32・±12 cm・±8°)。
- 走らせ方(`review/recheck_runner.mjs`): ユーザーの元の `dist/sim-core.js`・`qa/load-course-world.mjs` をそのまま読み、開始は app.js reset() の式を書き直し、方策は `policy.json`
  (sha256 `7c43cb53…`)を自前の MLP で計算(観測の定義の `rl/features.mjs` だけ使う)。確かめに、標準の開始で RL の 8.091953367 秒・7.656318984 秒がビット単位で出た。
- 判定(`review/independent_lap_check.py`): Python で物理を積分し直し(副ステップ・前後の応答・μg の頭打ち)、自前の幾何で衝突(車体の長方形と半径つき線分・円・脚の箱・外枠)、
  ゲート(順番・半幅 0.65 m)、黒シート(車幅がすべて開口の中・シートの上、周の中でちょうど 1 回)、周の時間(補間)を数えた。

| 走行 | 自前の判定 | 周の時間 | ベストラップの定義 | 最小すき間 | sim と一致 |
|---|---|---|---|---:|---|
| P1 単周 | 完走 | 8.082 秒 | 有効 | 7.3 cm | ○ |
| P1 連続 20 秒 | 20 秒 | 8.081 / 7.810 秒 | 2 周とも有効 | 6.2 cm | ○ |
| P2 単周(ゲート 6 から) | 完走 | 7.940 秒 | 有効 | 5.3 cm | ○ |
| P2 連続 20 秒 | 20 秒 | 7.939 / 7.789 秒 | 2 周とも有効 | 5.3 cm | ○ |

- ユーザーの `qa/verify-learned-lap.mjs` の `verifyLearnedReport` を当てると、周の項目(完走・衝突 0・33 ゲート・黒シート ちょうど 1 回・物理の再生)は 3 本(標準・P1・P2 の単周)とも合格。
  ONNX・開始の約束の項目は RL の周には当てはまらないので除いた(protocol best_lap_definition のとおり)。
- 同じ道具で、閉ループの 59 走行・RL と教師の記録 18 本・データセットのランダムな 2 シーンも数え直し、すべて sim と一致した。

## 4. データセット

| 項目 | 結果 |
|---|---|
| 一覧に出る | 動いている jetformer-web の `GET /api/datasets` に出る(60 シーン・36,000 フレーム・学習 35,400・ジャンル 8 だけ・タグ simulator + rl-bestlap) |
| シーンに展開 | `GET /api/scenes` にログ `ds-jetformer-simulator-rl-bestlap-20min_…` の 60 シーン(すべてジャンル 8・カメラ 36,000 枚) |
| 取り込める | 記録した ZIP(sha256 `8942243f…`、rl_dataset.json と同じ)を、**今の** JetFormer のコード(ほかの作業が足した学習除外 label_mask を含む)で別の置き場(`JETFORMER_DATA`)に取り込み直した。123 秒・60 シーン・学習 35,400・学習除外は当たらない(RL モードのシーンではないため)・学習の読み込みで 28,320 サンプル。本物の data/ には書いていない(置き場は確かめたあと消した) |
| 本物と同じか | 取り込み直したものと本物で 72,379 ファイル中 72,378 が同じ(違うのは取り込みの時刻を書く import_source.json だけ)。本物は取り込んだときのまま |
| ジャンル | data/modes.json の sha256 `f40f64bc…` は記録の前後と同じ。ジャンル 8 だけに 36,000 フレーム |

## 5. テスト一式

`jf-heavy --mem 2500M --lock big -- CUDA_VISIBLE_DEVICES= nice python -m pytest -q --deselect tests/test_session.py::test_run_lasts_exactly_from_start_beep_to_end_beep`
(`review/pytest_full.log`、5 分 47 秒): **703 passed・5 failed・3 skipped・1 deselected**。

| 失敗 | だれのもの | 理由 |
|---|---|---|
| `experiments/20261013_sim_rl_modes/r3/rc/promote/test_sim_label_frame_index.py` の 2 件 | ほかの作業(sim_rl_modes) | まだ当てていない直し(`dataset_frame_index.patch`)を前提にした試験。上の 1.3-2 と同じ問題 |
| `…/promote/test_sim_label_mask_e2e.py::test_e2e_import_expand_rebuild_relabel` | 同上 | 同上 |
| `experiments/20261013_sim_rl_modes/tests/test_masking.py::test_same_as_jetformer_import` | 同上 | 統合の段で jetformer/sim/label_mask.py の外す範囲を 1 つ広げたのに、試験の写しが古い |
| `tests/test_sim_runner.py::test_runner_sigint_stops` | シミュレーターの統合(jetformer/sim) | 単独で走らせても同じ(SIGINT で終わりのコードが −2)。コミット済みのコードのままの失敗 |

- この作業の試験(`experiments/20261013_sim_rl_bestlap`)は 13 件すべて合格。この作業は jetformer/・tests/ のコードを変えていない(git の差分にあるコアの変更はほかの作業のもの)。
  よって、この作業による後退は無い。失敗 5 件はそれぞれの作業に伝える。

## 6. ユーザーのシミュレーター

- `/home/jetson/sim/extracted/course-simulator` の 851 ファイルが COMPLETE-MANIFEST.json の大きさ・sha256 と一致(`review/sim_manifest_check.json`)。20 分データの ZIP も一致。
  マニフェストに無いのは `最初にお読みください.txt` だけで、これは元の ZIP に入っているもの(sha256 が ZIP の中と同じ)。`__pycache__` も無い。
- 作業用の写し(`sim/`)の 42 ファイルも元とバイト単位で同じ(写しにだけあるのは記録の段の差し込み `dist/rl.html`・`dist/rl/*` とログ)。

## 7. ユーザーの質問への答え(確認の段から見て)

「今後、速くぶつからないようにするには、どういうアップデートが有望ですか？」— README 8 章の順番は変えない。確認で分かったことを足すと:

1. **線の余裕を「いちばん狭い所」で約束させる**(いちばん有望)。周全体の最小すき間は RL 7.1 cm・教師 6.7 cm で同じくらいで、どちらも余裕の小さい 1 か所(RL は最初の直線の壁 10 cm、
   教師は点 4 のゲートの端 2.2 cm)で失敗した。平均ではなく、壁とゲートの端までの最小を 10〜20 cm(JetFormer のずれ 10〜19 cm に合わせる)にする。速さの損はまだ測っていない。
2. **ずれたときの戻し方を教える(DART・DAgger)**。S-R がぶつかった状態は、衝突の 0.13〜0.33 秒前まで RL が立て直せた。記録した状態から RL の操作を計算し直す仕組みは確かめ済み。
3. **RL の操作をなめらかにする**。まねの誤差は物差しで 4〜7 倍で、差はほとんど舵。
4. **RL を、まねする側の誤差(遅延 1・舵の乱れ)に強くする**。
- そのとき、シミュレーターのデータをシーンタブから作り直して混ぜるなら、先にフレームの番号で正解を引く直し(1.3-2)が要る。

## 8. この段でしたこと・ファイル

- 直したもの: `README.md`(1.2・1.3 の内容、8 章に 1 項目、9 章に確認の手順、10 章にファイル)・`evaluation.json`(`review_20261014` を足した。ほかの欄は同じ)・
  `evaluation.md`(`eval/render_md.py` に 5.1・5.2 節を足して作り直した)・`deviations.md`(この段の項目を追記)。直す前の写し: `review/*.before_review.*`。
- 新しいコード(`experiments/20261013_sim_rl_bestlap/review/`): `check_sim_manifest.py`・`recheck_runner.mjs`・`independent_lap_check.py`・`user_verifier_lap_items.mjs`・
  `attach_world.mjs`・`derived_numbers.py`・`reimport_check.py`・`attach_review.py`・`video/make_slides.py`。
- 出力(`review/`): `own_check_closed_loop.json`・`own_check_rl_teacher_records.json`・`random_starts/`・`recorded_scenes/`・`derived_numbers.json`・`dataset_check/`・
  `sim_manifest_check.json`・`pytest_full.log`。
- 動画: `video/storyboard.json`(約 1 分・7 場面・読み上げ 363 字。sim_cotrain の `check_storyboard.py` で OK)、図 `video/fig/v1_rl_vs_teacher.png`・`v2_dataset.png`・
  `v3_closed_loop.png`・`v4_why.png`(すべて生の記録と実際の画面から)。
- 確かめていないこと: ブラウザの閉ループを走らせ直してはいない(記録した操作の再生と数え直しだけ)。学習もやり直していない。RL の段の頑健さの表(r2・r3)は確認 r1〜r3 に任せ、ここでは数え直していない。
- 起動したものは全部終わっている(ブリッジ・Firefox・geckodriver は起動していない。jf-heavy の pytest・取り込み直しは終了済み)。
