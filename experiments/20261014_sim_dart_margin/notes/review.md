# 確認の段(崩しにかかる見直し)— sim_dart_20261014

- 作成 2026-10-14(19:44〜)。見たもの: evaluation.md / evaluation.json のすべての主張、protocol の sha256、deviations.md、JetFormer に取り込んだデータセット、
  D1 の正解の付け替え、ほかの作業のファイル。評価の段の道具(eval/*.py・*.mjs・runs.csv)は使わず、走行の **生の記録**(ブラウザのアプリが保存した JSON)から
  自前のコードで数え直した。道具は `/home/jetson/jetformer/experiments/20261014_sim_dart_margin/review/`、結果は `review/`。
- 実車・PCA9685・I2C には触れていない。GPU・Firefox は使っていない(CPU だけ、`jf-heavy --mem 1500M --lock none -- nice -n 10`)。ユーザーのファイル・ほかの作業のファイルは読むだけ。

## 0. 結論

**評価の段の結論は正しい。判定(Q1・Q2 とも「一部効いた」)・完走の数(A0 0/12・A1 0/12・A2 0/12)・表の数は、生の記録から数え直して同じだった。**
直したのは 3 か所(どれも結論は変わらない):

| # | どこ | 何が違ったか | 直したあと |
|---|---|---|---|
| 1 | evaluation.md 0 章・5 章、評価の段の報告 | 先生が点 4 のゲートを通る位置を「sim が数えた最初のフレームの位置」で測っていたため、線を越えてから最大 1 フレーム(約 13 cm)内側へ進んだ所の値になり、ゲートの端までの余裕が **2〜6 cm 小さく** 出ていた(T_fast 5.7〜9.9 cm・T_margin 4.9〜10.7 cm) | 線を越えた位置の線形補間で測り直した: **T_fast 11.0〜13.8 cm、T_margin 9.5〜11.8 cm**(先生をユーザーの sim-core.js で E0〜E9 から走らせ直した。周回時間は評価の段の値と同じ)。記録したデータでも同じ範囲(A0 のデータ 9.6〜14.3 cm、D2 9.2〜11.9 cm)。「先生も内側の端の近くを通る」「余裕の線はこのゲートの余裕を増やしていない」は変わらない(余裕の線のほうが 1〜3 cm 小さい) |
| 2 | evaluation.md 6 章・evaluation.json `user_verifier` | 今日の陽性対照(作者の ONNX の E0)のユーザーの確認器は **実行されて PASS** だった(`eval/closed_loop/user_verifier/EXT-E0-single.json`)。ところが evaluation.json の `user_verifier` は走行の名前で引くので、ベストラップの同じ名前 `EXT-E0-single` の行で上書きされ、6 章には「PASS(ベストラップ)」しか出ていなかった | `EXT-E0-single@today` を足し、6 章に今日の PASS を書いた。`positive_control_today.user_verifier = true` は元から今日のファイルを見ていて正しかった |
| 3 | evaluation.md 4 章の注 | 「A0-s0 の E0〜E5 の距離はベストラップの 10〜18 cm と同じ範囲」は、E5(ゲート 20 から)が 26.2 cm で範囲の外。ベストラップの 10〜18 cm は E0〜E4 の値 | 「E0〜E4 で 10.1〜18.8 cm(ベストラップの E0〜E4 の 10〜18 cm とほぼ同じ)、E5 は 26.2 cm」に直した |

- 直す前の写し: `review/evaluation.before_review.md`・`review/evaluation.before_review.json`・`review/deviations.before_review.md`。deviations.md に 31・32 を足した。
- evaluation.json には `review_20261014`(直したところ・測り直した値)を足した。`eval_sha256.txt` は評価の段の値のまま。直したあとの sha256 は `review/review_sha256.txt`。

## 1. 生の記録からの数え直し(`review/recount.py` → `review/recount.json`)

方法: 走行の JSON の `outcome`・`metrics`・`trace`・`world` だけを読む。完走 = route_complete・衝突 0・33 ゲート・黒シート 1 回。
最初の直線を抜けた = ゲート 0 の開始で、衝突の前に x > 2.6 m(protocol eval.per_run_metrics)。点 4 のゲート = 点 3 → 点 4 の向きの線を前へ越えた位置(線形補間)。
判定は protocol.json の verdicts の文をそのままコードにした。

| 腕 | 走行 | 完走(数え直し) | evaluation.md | 止まった時刻の中央値 | 最初の直線を抜けた(ゲート 0 の開始) | 点 4 を外した開始 |
|---|---:|---:|---:|---:|---:|---|
| A0-s0(S-R 種 0) | 12 | 0 | 0 / 12 | 1.091 秒 | 0 / 9 | — |
| A1-s0(A0 + DART) | 12 | 0 | 0 / 12 | 4.438 秒 | 8 / 9 | E0・E1・E3・E6・E8・E9(6 本) |
| A2-s0(余裕の線) | 12 | 0 | 0 / 12 | 6.557 秒 | 9 / 9 | E0・E1・E2・E4・E6・E7・E8・E9(8 本) |
| A0 種 1 | 6 | 0 | 0 / 6 | 1.118 秒 | 0 / 5 | — |
| S-T 種 0 | 6 | 2(E1 8.830・E3 8.840 秒) | 2 / 6 | — | 5 / 5 | E0・E2・E4 |
| S-T 種 1 | 6 | 0 | 0 / 6 | — | 5 / 5 | E0〜E4 |
| 作者の ONNX(ベストラップ) | 6 | 2(E0 9.022・E4 9.033 秒) | 2 / 6 | — | 5 / 5 | — |
| 作者の ONNX(今日の陽性対照 E0) | 1 | 1(9.022 秒) | 9.022 秒 | — | — | — |

- **判定の数え直し**: Q1 = 一部効いた(c = 0、差 0、最初の直線 8 / 9 ≥ 8)、Q2 = 一部効いた(9 / 9)。evaluation.md と同じ。
- 走行ごとの終わり方・ゲートの数・止まった時刻・ぶつかった物(レール・柱の脚・幕)・位置は、2 章・4 章の表と全部同じ。
  「A1 の 4 本(E0・E3・E6・E9)が x ≈ −0.2, z ≈ −2.2 の柱の脚に 5.63〜5.74 秒」「A2 の 6 本(E1・E2・E4・E6・E8・E9)が x ≈ −2.1, z ≈ −2.3 の柱の脚に 6.54〜6.75 秒」も同じ。
- 点 4 の「外し」で線の延長を越えた横の位置: A1 −2.62〜−2.84 m、A2 −2.94〜−3.26 m(表の値と 0.03 m 以内。0 章の「2.6〜3.2 m」は「2.6〜3.3 m」に直した)。
- 設定: 34 走行すべてで制御遅延 1・先読み 1・舵ゲイン 1.0・前進ゲイン 1.0(作者の ONNX は 1.01)・ジャンル 8(作者の ONNX は 4)・単周の上限 15 秒・種 1・baseline。
  モデルの sha256 は一式の ONNX と同じ(A1 `7bb9b4ad…`・A2 `5737fe92…`。ZIP の中の ONNX と data/runs の ONNX の両方で確かめた)。
- 決定性: A1 の E0 の 2 回は 169 フレームとも同じ(数え直しでも同じ終わり方・時刻)。

## 2. ユーザーの sim-core.js での再生

- ユーザーの有効な版(`data/sim/active.json` → `releases/20261013-144013-6f610dd5`)の **`replay-report.mjs`**(ユーザーの道具そのもの。`dist/sim-core.js` を import し、
  保存した操作を再生して world_hash・outcome・metrics・final_state・actions・trace を差 1e-9 で比べる)を、元のフォルダーのまま実行した(`review/replay_all.sh`)。
- 指定の 2 本: **A1-s0-E0-single PASS**(169 フレーム・衝突 5.626 秒)、**A2-s0-E7-single PASS**(323 フレーム・幕に 10.758 秒・差の最大 3.3e-15)。
- 念のため今日の 34 走行すべても再生した: **34 / 34 PASS**(`review/replay_user_tool/`)。sim-core.js の sha256 `aaeb2379…` はこの作業・ベストラップの写しと同じ。

## 3. 手順の決まり・違えたこと

- protocol.json の sha256 = `c423c30f3a01ea705e07ad0ceae1175ff78c08dada85b803151e207323680aae` = protocol.sha256 = evaluation.json の記録の値。protocol.json は 12:45 から変わっていない。
- record_sha256.txt・train_sha256.txt・eval_sha256.txt(確認の段で直す前)は全部一致。build_sha256.txt は 4 ファイル(browser_collect.py・run_record.sh・verify_recording.mjs・
  verify_dart_collection.py)が違うが、記録の段で直したもの(deviations 9・10、元は `*.build_1405.bak`)で、record_sha256.txt の値と一致する。
- deviations.md の 1〜30 は、記録・学習・評価の段の報告にある変更(A3 を削った・名前の取り込み直し・電源断と時間割・A4 と種 1 を削った・workers・FP16 を作らない・
  オフライン L1・評価の道具の写し・陽性対照の走らせ直し・「10 本」が 9 本・測り方・図)をすべて含む。確認の段の分を 31・32 に足した。
- 削ったもの: A3(D3 を 15:30 までに始められなかった。cut 3)、A4(DAgger)と種 1(電源断のあとの時間割。cut 1・2)。評価の本数は削っていない。never_cut の E0〜E5 は全部の腕で走った。

## 4. JetFormer のデータセット

JetFormer 自身の `jetformer.dataset.list_datasets()`(読むだけ)で確かめた:

| データセット | タグ | シーン | フレーム | 学習に使える | train / val | ジャンル 8 | シーンのタブ |
|---|---|---:|---:|---:|---|---:|---|
| `jetformer-simulator-rl-dart-fast-10min_20261014-062216_4446c73d`(D1) | simulator・rl-dart | 30 | 18,000 | 17,700 | 24 / 6 | 18,000 | 展開しない(protocol: 付け替えた正解がシーンから作り直すと消えるため) |
| `jetformer-simulator-rl-margin-line-10min_20261014-070934_48d04e6b`(D2) | simulator・rl-margin | 30 | 18,000 | 17,700 | 24 / 6 | 18,000 | 30 シーン展開済み |

- 名前を直す前の最初の D1 の取り込み(`rl-dart-fast-10min_20261014-061747_6b97ee7a`)は一覧に無い(消されている)。D1 の dataset_card に付け替えの記録(`relabel`、変わったフレーム 5,447)がある。
- 学習の結果: `data/runs/simdart_20261014-A1-s0`(30 エポック・最良の検証 19.0 µs・30 エポック目)と `-A2-s0`(16.1 µs・29 エポック目)。train.md と同じ。

## 5. D1 の正解の付け替え(`review/relabel_check.py` → `review/relabel_check.json`)

付け替えの道具(relabel/*)を使わず、取り込んだ NPZ・付け替える前の写し(relabel/backup/)・ブラウザの記録(frames.jsonl の command_si・dart.clean・dart.noise)だけで 30 シーンを見た。
PWM の式は記録から当てはめた(舵: 左 850 µs/rad・右 1073 µs/rad・中心 1442、前進: −68.8 µs/(m/s)・中立 1498。当てはめの差 ≤ 2e-10 µs)。

| 確かめたこと | 結果 |
|---|---|
| 付け替える前の正解 = 記録の操作の 1〜10 先 | 17,670 / 17,670(差の最大 6e-5 µs = float32 の丸め) |
| 次のフレームが乱されていない → 最初の正解は付け替える前と同じ | 14,740 / 14,740(差 0.0) |
| 次のフレームが乱された → 最初の正解 = 先生の乱す前の操作(記録の dart.clean)| 2,930 / 2,930(差の最大 6e-5 µs)。そのうち 2,841 は付け替える前(乱し入りの操作)と違う |
| t〜t+10 のどれも乱されていない → 10 個の正解すべてが付け替える前と同じ | 11,862 / 11,862(差 0.0) |
| 変わったフレームの数 | 5,447(record.md・dataset_card と同じ) |
| future_valid | 30 シーンとも付け替えの前後で同じ |

## 6. DART の乱しの場所(参考の測り)の数え直し(`review/coverage_check.py`)

D1 の frames.jsonl から自前の区間の当て方で数え直した: 乱しのあった区間 **18 / 33**、乱しの無い区間は点 6〜12・21〜26・29・30 から次の点まで(15 区間)、
乱しのフレーム 2,936・乱しの始まり 292、点 3 → 4 の区間の始まり **0 回**(続いてきた乱しのフレーム 97)。evaluation.md・dart_coverage.json と全部同じ。
先生は点 4 のゲートの中心から 約 51〜56 cm の所を通る(0 章の「55〜60 cm」を直した)ので、見張りの決まり(30 cm 以内のときだけ始める)で始まらない、という説明は正しい。

## 7. ほかの作業のファイル

- ベストラップの作業(experiments/20261013_sim_rl_bestlap・jetformer_reports/sim_rl_bestlap_20261013): 今日 12:10:02 より後に変わったファイルは無い
  (12:10:02 の notes/ への写しは、この作業が始まる前のベストラップの作業の最後の段のもの)。
- modes の作業(experiments/20261013_sim_rl_modes・jetformer_reports/sim_rl_modes_20261013): 今日変わったファイル(r3・notes・README・review・deviations)は modes の作業自身のもの。
  この作業の道具は modes のファイルを読むだけ(書き込み先はこの作業の 2 つのフォルダーだけ。install_dart.py も同じ)。
- ユーザーのシミュレーター(`~/sim/extracted/course-simulator`・`data/sim/releases`): 今日変わったファイルは無い。この作業の写し(dist・bridge.py・jetformer_policy.py)は
  元とバイト単位で同じ(足したのは dist/dart/ と d1〜d3.html だけ)。
- ユーザーのブリッジ(data/sim)のログに今日 12:11〜13:27 の 2 分のシーンの記録があるが、この作業のものではない(この作業の記録は 20 秒のシーンで、口 8811 の自前のブリッジ)。
- JetFormer の git: 追跡しているファイルの変更は無い(`experiments/20261014_sim_dart_margin/` は未追跡)。コミットはしていない。jetformer-web は 16:36 の起動(電源断のあと)のまま、再起動 0。

## 8. 崩しにかかって分かったこと(結論は変えないが、読むときに気をつけること)

1. **「一部効いた」の根拠は「最初の直線を抜けた」だけ**。完走は 3 つの腕とも 0 で、差は無い。A2 の先生の線は最初の直線で壁から約 40 cm 離れる(A0 は約 11 cm)ので、
   A2 がそこを抜けるのは線の形からほぼ決まっている。「余裕の線は最初の直線の衝突を防いだが、1 周を走り切らせる効果はまだ見えない」と読むのが正しい。
2. **点 4 のゲートは前から分かっていた難所**。作者の先生のデータで学んだモデル(S-T 種 0・種 1)も、ゲート 0 から始めた 10 本中 8 本が点 4 を外している(数え直し)。ベストラップの報告も、作者の先生の線は点 4 の端まで 2.2 cm と書いている。
   A1・A2 は点 4 の手前で内側へ曲がって外した。物の上ではそのあともコースに沿って 1 周の 60〜70% ほど進んでいる(衝突したのは柱の脚)。
3. **A1 の改善が DART のおかげかは分けられない**(A1 は 30 分、A0 は 20 分。種も 1 つ)。evaluation.md・protocol にも書いてある交絡。
4. **先生の線からの距離**は、先生をユーザーの sim-core.js で 12 の開始から走らせ直した軌跡(`review/teacher_traces.json`。周回時間は評価の段の表と 24 本とも同じ)と、
   自前の距離の計算(`review/video/make_slides.py` の path_dist)で数え直した: 1 秒の時点の中央値 **A0 12.45・A1 4.35・A2 1.11 cm**(evaluation.md と同じ。`review/path_dist_check.json`)。
   ゲート 0 から始めた A0 の 9 本の衝突の時点の距離は 10.7〜19.0 cm。**自前で計算し直していない数**は「止まるまでに進んだ道のり」と「先生への引き継ぎ」(評価の段の道具の値)で、
   どちらも判定には使っていない。
5. 走行はすべてシミュレーターの中(実車では確かめていない)。

## 9. ファイル

- 道具: `/home/jetson/jetformer/experiments/20261014_sim_dart_margin/review/`(recount.py・replay_all.sh・relabel_check.py・coverage_check.py・gate4_teacher_check.py・teacher_gate4.mjs・teacher_traces.mjs・video/make_slides.py)
- 結果: `review/recount.json`・`review/replay_user_tool/`・`review/relabel_check.json`・`review/coverage_check.json`・`review/gate4_teacher_check.json`・`review/teacher_gate4.json`・`review/teacher_traces.json`・`review/path_dist_check.json`・`review/review_sha256.txt`
- 動画: 絵コンテ `video/storyboard.json`(ユーザーの決まりの確かめ `check_storyboard.py` は OK、読み上げ 313 字・9 場面)、図 `video/fig/v1〜v6`(`review/video/make_slides.py`。数は `video/fig/slide_numbers.json`)
- 直す前: `review/evaluation.before_review.md`・`review/evaluation.before_review.json`・`review/deviations.before_review.md`
