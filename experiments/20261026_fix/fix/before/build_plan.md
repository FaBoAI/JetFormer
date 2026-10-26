# 段 plan の作業の記録(10/26)

- 依頼(ユーザー、10/26 10:0x):「それでは、今日の課題の整理と、それを修正する作業を開始してください。」
- この段の担当: 課題 6(11/3 の並び・手順書・車の確認・ファームの確かめ)。`fix_plan.md` 1 章の課題 6・5 章。
- 守ったこと: 車・PCA9685・I2C・カメラ・ToF・基板に触れていない。走行・保存・センサーの開始・較正の API を呼んでいない(GET だけ)。jetformer-web は再起動していない。既定値・推奨モデル・`drive_settings.json`・protocol_v2/v3・lock2・data/raw・labels を変えていない(protocol_v3 は `sha256sum -c` で 7 つとも OK)。学習・GPU を使っていない。git commit していない。

## 1. 作ったもの

| ファイル | 中身 |
|---|---|
| `data/bench_candidates.json` | days に **2026-11-03** を足した(23 ブロック)。changes に 1 行。今の日・候補・基準は同じ(下の確かめ) |
| `data/bench_candidates.json.bak-20261026-plan` | 足す前の控え(sha256 `aee854d6…3163b`) |
| `runbook_20261103.md` | 11/3 の手順書(時間割・並び・1 本の手順・入れ方・決め方・止める条件・夜の集計) |
| `todo_user_1026.md` | ユーザーが 11/3 までにすること・決めてほしいこと |
| `tools/fw_vbat_check.py` | ファームの電圧の確かめ(GET を数秒読むだけ。装置は開かない) |
| `experiments/20261026_fix/plan/add_day_1103.py` | 11/3 の日を足す台本(控えが今のファイルと同じときだけ書く・書いたあとに確かめる) |
| `experiments/20261026_fix/plan/check_bench_day.py` | 画面と同じ読み方(`web/bench.load_candidates`)で確かめる。上書きで使える項目の表 |
| `experiments/20261026_fix/plan/runs_1103.py` | 夜: 説明欄の本と log.json の実際の設定を並べ、計画と違う本に印 |
| `experiments/20261026_fix/plan/test_fw_vbat_check.py` | fw_vbat_check の試験 13 個(偽の管理画面) |

## 2. 11/3 の並び(bench_candidates.json)

| 番号 | 本 | 押したときの上書き | 押したあとに入れるもの |
|---|---|---|---|
| 1〜5 | 回 1: 1-1 確認・1-2/1-3 床・1-4 天井 4.1・1-5 P1 | 速度 0.90 | 1-4 だけ天井 ON |
| 6〜11 | 回 2: 2-1 天井 4.1・2-2 P1・2-3/2-4 天井 + 自動 8.12・2-5 左 4.2(条件)・2-6 左 4.1 | 速度 0.90 | 天井・自動合わせのスイッチ、2-5・2-6 は API で左の値 |
| 12〜19 | 回 3: 3-1〜3-8 発進の前進 OFF/ON 交互 | 速度 0.90 | 回 2 のよい形・API で `launch_boost` |
| 20〜23 | 回 4: 4-1〜4-3 本番の形(止まって 3 周)・4-4 予備 | 速度 0.90 | その日いちばんよい形 |

- どれもモデル B。押したときの速度 0.90 = 電圧の表のいちばん低い値(満充電で安全な方)。走る前に表・目安で入れ直す。
- 各ブロックに `run_id`(本)・`form`・`gain_rule`・`expect`(実際の設定の計画。null = その場で決める)・`after_press`・`if`(条件)を書いた。画面と bench_report はこれらを読まない(block・id・params・mode_id・tune・counted_ai_s だけ)。`runs_1103.py` が expect を照らす。

## 3. ブロックの上書きで使える項目(確かめた)

| 項目 | サーバー(`_block_overrides`・`PARAM_LIMITS`) | 画面(`app.js`) | 判定 |
|---|---|---|---|
| throttle_gain・steer_gain・steer_trim_us・lookahead・start_kick・ensemble_j・tof_guard | 受ける | 欄に入れて走行開始で送る | **使える** |
| speed_ceiling・speed_auto・zone_caps | スイッチなので捨てる(警告) | 押すと OFF にする | 使えない(押したあとスイッチで) |
| ceil_left_mps・ceil_all_mps・auto_target_mps・auto_vref_v | 受けて説明欄に `ceil_left_mps=4.2` と書く | 欄が無く、送らない | **使えない**(説明欄だけ違う値になる)。止まっているときに API |
| launch_boost・launch_us・launch_mps・launch_max_s | 今は PARAM_LIMITS に無いので捨てる | 欄が無い | 使えない。段 launch のあとも欄が無ければ同じ。API |

## 4. 確かめ

| 確かめ | 結果 |
|---|---|
| `check_bench_day.py`(jf-heavy --lock none --mem 1500M) | **OK**: 23 ブロック・走れる 23・説明欄 23・11/3 の警告 0。今の日・候補・基準・今の警告が控えと同じ |
| 本物の画面の GET `/api/bench/candidates` | ok・警告 0・11/3 の 23 ブロックが全部走れる。説明欄 `… day=2026-11-03 block=4 cand=B … tune=1-4・天井4.1・同じゲイン・4〜6周 throttle_gain=0.9 memo=` |
| `plan_check`(走らせずに) | 速度 0.95 → `cand=EXP`・official 0。天井 ON → EXP。速度 1.00・天井 OFF → B のまま |
| 夜の道具 | `bench_nightly.sh` の 1 日目の自動は 20261023(レポートは 2 日分)。`bench_report.planned_block(11/3, 4)` = B |
| `test_fw_vbat_check.py` | **13 / 13 合格**(古いファーム・よいファーム・範囲の外・揺れ・センサー停止・電圧計の差・読みの失敗・走っている車・途切れ・低い電池・GET だけ・偽の HTTP・届かない) |
| 本物の画面で `fw_vbat_check.py` | 番号 3「センサーが止まっている」(今はセンサー停止中。開始はしない)。待たずに 1 回の GET で返る |
| `runs_1103.py --day 20261024` | 20 本を並べた。違いの印は 1 本(午前の ENS。あとでブロック 3 を B に書き替えたため。読み方は正しい) |

## 5. 残ったこと

| # | こと | 次 |
|---|---|---|
| 1 | 速度 0.90 か天井 ON の本は全部 `cand=EXP`(plan_check は基準と比べる)。quick_eval の段ごとの表で天井 ON/OFF が混ざる | 夜は `runs_1103.py` の実際の設定で分ける。quick_eval を実際の設定で分ける台本は未担当 |
| 2 | 「止まってから 3 周」の時間を記録から出す台本が無い | 未担当。当日はストップウォッチ + 動画 |
| 3 | 回 3 の測り(`diag_launch/collect.py`)は 10/23・10/24 に固定 | 11/3 を読める版が要る(未担当) |
| 4 | 左 4.2・発進の前進は画面に欄が無い。説明欄の tune・throttle_gain は押したときの値 | API の手順を手順書 5 章に。本当の値は log.json |
| 5 | 発進の前進が段 launch でまだ入っていない(今の drive.py に `launch_*` 無し) | 入らなければ回 3 は OFF だけ |
| 6 | 予選の決まりが分からない(スタートの線・計時) | ユーザーに聞く(todo 2 章 5) |
