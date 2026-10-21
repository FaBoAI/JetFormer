# 10/23(金)の候補 5 本と順番(段 Finalists、2026-10-21 13:00)

- 決まり: `plan/protocol.json`(sha256 `1dbb279e…0a2bb`、`sha256sum -c protocol.sha256` は OK)と `plan/protocol_amendments.md` の A1。
- 机上の値: `desk/desk_bench.json`(12:54、sha256 `26b7a0f4…2fca`。protocol_sha256 は今の protocol.json と同じ)。
- 決まりをそのまま当てました。決まりの外の判断はしていません。
- 書いたもの: `/home/jetson/jetformer/data/bench_candidates.json`(画面の「ベンチマークの候補」・「基準の設定」と、夜の道具が読む)。
- 計算: `/home/jetson/jetformer/experiments/20261021_bench_prep/desk/finalists.py`(標準ライブラリだけ)。結果の全部は同じ場所の `logs/finalists_select.json`。

## 1. 結果

| ブロック | id | 表示名 | 枠 | 決まりの行 | risk |
|---|---|---|---|---|---|
| 1 | B | モデルB(基準) | 基準 | pool.baseline | low |
| 2 | SLOW | 遅く入る型 | C1 | selection.step2_slow → order 1 | low |
| 3 | ENS | 2本の平均 | C2 | selection.step3(ens)→ order 2 | mid |
| 4 | B | モデルB(基準) | 基準 | block_layout n5 | low |
| 5 | AUG | ずれに強い型 | C3 | selection.step3(robust)→ order 3 | low |
| 6 | ACT | ACT型 | C4 | selection.step3(arch)→ order 3 | low |
| 7 | WGT | 苦手区間重み型 | C5 | selection.step3(weight)→ order 3 | low |
| 8 | B | モデルB(基準) | 基準 | block_layout n5 | low |

- 10/24 は空のままです。note は「10/23 の夜に決める」です(ranking の day1_top2 で T1・T2 を選び、B・T1・T2・B・T2・T1・B で走る)。
- 入らなかったのは MSF・TCP・SIM の 3 本です。3 本とも門には合格しています。同じ家族の中で desk_key が下だったので入りませんでした(4 章)。

## 2. 先に確かめたこと

- B は門に掛けません(gates.baseline)。B の engine_ok・sanity_ok はどちらも true でした。なので、10/23 は予定どおり走れます。
- 9 本とも、engine_check.json は今のエンジンのもの(大きさと mtime が一致)で、精度は合格でした(actions_max_us 0.86〜2.09)。読んだだけです。
- 9 本とも drivable true で、ジャンル 5 も学習済みです。
  - 選んだ 6 本(B を含む)は、画面の読み込み(`jetformer/web/bench.py` の load_candidates)で確かめました。
  - 残りの 3 本(MSF・TCP・SIM)は、drive.model_info で確かめました。

## 3. 門(selection.step1、gates + A1)

- 門は desk_bench.json の値から、式で計算し直しました(gates.recompute)。
- 家族 label_r1(SLOW)だけ A1 を当てました。門 5 を l1_steer_best_us で比べ、brake_lowspeed_frac ≤ 0.01 を足しています。
- B の l1_best_us は 49.144 µs、l1_steer_best_us は 50.890 µs です。

| id | 家族 | 門 5 の欄 | B との差 [µs] | trt_p99_ms | steer_amp_ratio | engine・sanity | brake_lowspeed_frac(A1) | 門 |
|---|---|---|---|---|---|---|---|---|
| MSF | arch | l1_best_us | +0.155 | 3.489 | 1.0158 | ok・ok | — | 合格 |
| TCP | arch | l1_best_us | +0.556 | 2.954 | 0.9987 | ok・ok | — | 合格 |
| ACT | arch | l1_best_us | +1.668 | 2.879 | 0.9892 | ok・ok | — | 合格 |
| SIM | robust | l1_best_us | +0.943 | 2.606 | 1.0026 | ok・ok | — | 合格 |
| AUG | robust | l1_best_us | −0.324 | 2.428 | 0.9822 | ok・ok | — | 合格 |
| ENS | ens | l1_best_us | −2.718 | 4.261 | 0.9707 | ok・ok | — | 合格 |
| SLOW | label_r1 | l1_steer_best_us | +3.704 | 2.480 | 1.0184 | ok・ok | 0.00353 | 合格(A1) |
| WGT | weight | l1_best_us | +1.287 | 2.523 | 0.9936 | ok・ok | — | 合格 |

- 8 本とも合格です。
- desk_bench.json の gates_pass と、8 本とも一致しました。食い違いが無いので、deviations.md に書くことはありません(gates.recompute)。
- SLOW は、A1 が無ければ落ちていました。全体の L1 の差が +14.221 µs で、10 µs 以上だからです。
  - A1 は、desk_bench.json ができる前の 11:30 に書かれています(protocol_amendments.md の「書く前に見たもの」)。

## 4. 選び方(selection.step2〜5)

- **step2**: SLOW(bench_20261021-slow-s0)は門に合格しました。なので必ず入れます。slow-s1 は作っていません(AUG ができたため)。
- **step3**: 残りの 4 家族から、desk_key がいちばん良い 1 本ずつを入れます。
  - desk_key の 1 番目は l1_interv_us です(小数第 1 位に四捨五入)。どの家族も 1 番目の鍵で決まり、同点はありませんでした。

| 家族 | 候補(l1_interv_us の丸め) | 入れたもの | 入らなかったもの |
|---|---|---|---|
| arch | ACT 125.4・MSF 129.7・TCP 134.0 | ACT | MSF・TCP |
| robust | AUG 123.6・SIM 124.7 | AUG | SIM |
| ens | ENS 117.3 | ENS | — |
| weight | WGT 128.2 | WGT | — |

- **step4**: step3 までで 5 本そろったので、使いません。そのため、同じ家族の 2 本目(SIM・MSF・TCP)は入りません。
- **step5**: 門に通ったのは 8 本で、5 本以上あります。枠は詰めません。

## 5. 順番(order)

- **order 1**: SLOW の枠を C1 にしました。
- **order 2**: ENS が選ばれているので、C2 にしました。
- **order 3**: 残りの AUG・ACT・WGT は、3 本とも risk が low です。なので desk_key の順にしました。AUG(123.6)→ ACT(125.4)→ WGT(128.2)で、C3・C4・C5 です。
- **block_layout n5**: 並びは B・C1・C2・B・C3・C4・C5・B です。

### risk(risk_rule)

| id | steer_amp_ratio | r1_app_thr_diff_us | trt_p99_ms | 新しい方策のファイルを通るか | risk |
|---|---|---|---|---|---|
| B | 0.9964 | −8.333 | 2.487 | いいえ(jetformer_v0) | low |
| SLOW | 1.0184 | +31.859 | 2.480 | いいえ(jetformer_v0) | low |
| ENS | 0.9707 | −4.996 | 4.261 | **はい(ens_average)** | **mid** |
| AUG | 0.9822 | −9.722 | 2.428 | いいえ(jetformer_v0) | low |
| ACT | 0.9892 | +0.979 | 2.879 | いいえ(act) | low |
| WGT | 0.9936 | −1.299 | 2.523 | いいえ(jetformer_v0) | low |
| (MSF) | 1.0158 | −2.147 | 3.489 | いいえ(msfusion) | low |
| (TCP) | 0.9987 | −2.916 | 2.954 | いいえ(tcp) | low |
| (SIM) | 1.0026 | −8.187 | 2.606 | いいえ(jetformer_v0) | low |

- 「新しい方策のコード」は、2026-10-21 に jetformer/policies に足したファイル(ens_average.py・ens_config.py)です。どれを通るかは、学習結果の model_config.json の policy で見ました。
  - policies/__init__.py も 10/21 に直されていますが、新しいファイルではありません。決まりの文のとおり、数えていません。全部の候補がこのファイルを通ります。
- AUG の r1_app_thr_diff_us は −9.722 です。low の境(≥ −10)の内側です。決まりには丸めの指定が無いので、生の値で判定しました(丸めても −9.7 で同じです)。

## 6. 当日の段取りへの当てはめ(protocol day1 から。新しく決めたことはありません)

- 慣らし(warmup_anchors): ブロック 1 の B と、SLOW・ENS・AUG・ACT・WGT は、実車で初めてです。なので K = 3 です。ブロック 4・8 の B は K = 2 です。
- 電池(day1.batteries): #1 = B・SLOW、#2 = ENS・B、#3 = AUG・ACT、#4 = WGT・B、#5 = ブレーキの測定、#6 = 予備。
- 削る順(day1.cut_order): 4 番目で C5 = **WGT** を外します。5 番目で C4 = **ACT** を外します。最後の B は削りません。
- 「説明」欄: 画面が description_template から作ります。例は `bench=real_bench_20261023 day=2026-10-23 block=2 cand=SLOW model=bench_20261021-slow-s0 battery=#__ V_before=__ V_after=__ check=__ memo=` です。
- 基準の設定(baseline): mode_id 5。値は steer_gain 1.0・throttle_gain 1.0・lookahead 3・tof_guard 1・ensemble_j 0・speed_cap 0・zone_caps 0 です。ほかは drive.py の DEFAULT_PARAMS の値です(steer_trim_us 0・osc_percent 0・esc_reverse 1・recovery 1・rev_us 300・rev_s 0.4・speed_scale 1.0・tcp_mode 0・ensemble_m 0.5)。
  - protocol の baseline_params に無いのは rev_us・rev_s の 2 つだけです。「ほかは DEFAULT_PARAMS」に従い、DEFAULT_PARAMS の値を書きました。
  - 画面の読み込みで組み立てた値は、DEFAULT_PARAMS に 3.4 の値を重ねたものと一致しました。

## 7. 決まりで決まらなかったところと、その扱い

- **B の risk**: risk_rule は「bench_candidates.json の risk はこの規則で付ける」と書いています。なので B にも同じ式を当て、low としました。B は順番には入りません(毎日の基準です)。
- **B の hypothesis**: pool に文がありません。依頼の表のとおり「基準」と書きました。
- **10/24 の note**: 「10/23 の夜に決める」に、決め方(ranking の day1_top2・day2.sequence)を書き足しました。
- **bench_candidates.json に足した欄**: 候補ごとの family・slot・selected_by と、上の階層の selection(落ちた候補と理由・desk_bench.json の sha256)です。
  - 画面(bench.py)は、これらの欄を読みません。夜の道具(bench_report.py)が family を読みますが、run の名前の規則と同じ値です。
- **deviations.md**: 門の食い違いは無く、この段で手順から外れたこともありません。なので、作っていません。
  - A1 は protocol_amendments.md にあります。実車のレポートでは、bench_report.py の候補の表に「(A1)」の印が出ます。

## 8. 照合(どれも一致)

- `plan/src/select_ref.py`(段 Design の参照の実装。A1 は入っていない)
  - 門が違うのは SLOW だけでした。ほかの 7 本の門と、9 本の risk は一致しました。
  - 参考: A1 が無いと、選ばれるのは ENS・AUG・SIM・ACT・WGT になります。
- select_ref.py の門だけを A1 入りに差し替えた版でも、選んだ 5 本・順番・8 ブロックの並びが同じになりました。
- `tools/bench_report.py` の recompute_gates(A1 入り、段 S3)も、8 本とも同じ判定でした。SLOW には A1 の印が付きました。
- `jetformer/web/bench.py` の load_candidates で、書いたファイルを読みました。
  - ok true、warnings は 0 件でした。
  - 6 つの学習結果は、すべて drivable true でした。
  - 8 ブロックとも、説明欄の文が作れました。
- bench_report.py の planned_block と family_of で、8 ブロックの id と家族が読めました。
- 試験: `experiments/20261021_bench_prep/desk/test_finalists.py` の 5 件は、すべて合格しました。中身は次のとおりです。
  - 門の境の値と、欠けた欄の扱い
  - A1 は label_r1 だけに当たること
  - 家族を run の名前で決めること
  - risk の境の値
  - desk_key の丸め・null・id による決着

## 9. 当日の注意(情報だけ。選び方と順番は変えていません)

- **WGT(C5)**: 低速でブレーキ側の出力を出すことがあります。brake_lowspeed_frac は 0.055 で、10/10 の AI 走行で止まった場面の 6.0% です。
  - この門は A1 で label_r1 だけに掛かるので、WGT は落ちません。
  - 走行は後退を受け付ける設定(esc_reverse 1)です。なので、止まったときに後ろへ動きうります。
  - 止まる衝突は、すぐ CH3 で取り返します(5.1 の決まりのまま)。
  - 入らなかった SIM は 0.080 です。
- ACT(C4)は 0.020、AUG(C3)は 0.0102 です。どちらも label_r1 ではないので、門には掛かりません。
- ENS(C2)は、新しい方策のコード(2 本の平均)を通る唯一の候補です(risk mid)。決まりのとおり前半に置き、B で挟みます。
- B(−8.3 µs)と AUG(−9.7 µs)は、R1 の手前で人より強く入る向きの予測です。SLOW だけが +31.9 µs で緩めます。
- 画面のパネル: 今動いている jetformer-web(:8080)には、まだ `/api/bench/candidates` がありません。GET は Not Found でした。
  - パネルが出るのは、決まった段で jetformer-web を再起動したあとです。この段では再起動していません。

## 10. 変えていないもの

- protocol.json(sha256 は同じ)・protocol_amendments.md・desk_bench.json
- data/drive_recommended.json(10/12 のまま)・data/drive_settings.json(10/10 のまま)・drive.py の DEFAULT_PARAMS
- 学習結果・エンジン・engine_check.json(9 本とも既にあったので、作っていません)
- 車・装置・jetformer-web には触れていません。GPU は使っていません。git commit はしていません。
