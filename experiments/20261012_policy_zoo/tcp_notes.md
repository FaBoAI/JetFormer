# 方策の比較: `tcp`(TCP / TransFuser++ 型)の実装メモ(P-tcp)

作成 2026-10-12 ・ 対象 `/home/jetson/jetformer`(main の作業ツリー。commit はしていない)・ CPU だけ(GPU は使っていない)

spec.md §3.2 の `tcp` を実装した。仮のモジュール(`POLICY_STUB = True`)は消したので、登録簿の `ready()` は `(True, '')` になり、
管理画面の学習タブ・`tools/train.py --policy tcp` で学習できる。

## 1. 作ったもの(すべて P-tcp の持ち物。ほかの持ち主のファイルは変えていない)

| ファイル | 中身 |
|---|---|
| `jetformer/policies/tcp.py`(torch) | モデル `TCPPolicy`(v0 と同じトークン → Encoder 4層 → クエリ 21 の Decoder 2層 → `heads_wp`・`heads_v`・`heads`)。`EXPORT_OUTPUTS = ('actions', 'waypoints', 'speed_logits')`、`policy_loss`(l1 + 0.5·wp + 0.1·ce、wp・ce は `wp_valid` だけ)、`twohot`。モジュールの関数 `build`・`prepare`・`extra_targets`・`finalize`・`postprocess_from` |
| `jetformer/policies/tcp_labels.py`(numpy) | 正解(通過点・目標速度・有効の印)の計算とキャッシュ、制御器の較正(ステア → 曲率の表・スロットルの前向きの写像) |
| `jetformer/policies/controller.py`(numpy) | 制御器 `TCPPost`(`post.KINDS['tcp_controller']` が読む)。走行(`drive._loop` の `post.Stage`)と評価(`post.replay`)が **同じこのコード** を呼ぶ |
| `tools/make_tcp_labels.py` | 正解と較正を作る・確かめる CLI(`tools/train.py --policy tcp` も学習の前に同じものを作る) |
| `tests/test_policy_tcp.py` | モデル・損失・ONNX・正解・較正・学習の通し(18 本、CPU 約15秒) |
| `tests/test_policy_tcp_controller.py` | 制御器(21 本、CPU 約1秒) |

## 2. 制御器をどこで動かすか(決定)

**ONNX の外、numpy(`controller.py`)で動かす**(spec.md §0.1 の 5・§8「後処理を ONNX の中に入れない」)。

- ONNX(TensorRT)の出力は `actions [1,10,2]`(操作の枝)・`waypoints [1,10,2]`・`speed_logits [1,10]` の3つ。`actions` は v0 と同じ意味なので、
  走行の約束(`check_io`)・先読み・ゲイン・ToF ガード・補助・CH3・センサーの途切れで中立は、そのまま効く。
- 速度の上限 `speed_cap`・倍率 `speed_scale`・出し方 `tcp_mode` は走行のパラメータで、エンジンを作り直さずに走行タブで変えられる。
- 走行の既定は `tcp_mode = 0`(操作の枝)。**`tcp_mode = 0` かつ `speed_cap = 0` なら出力は `actions` の µs とビット単位で同じ**
  (テスト `test_mode0_without_cap_is_bit_identical_to_actions`。`post.Stage` を通して確かめた)。比較の主指標(§5.2)はこの出力。
- 速度の上限は `tcp_mode = 0` でも効く(スロットルを `max(操作の枝, thr_ff(上限))` に。前進を弱めるだけで速くしない)。
  上限が上がる・外れるときは 1.5 m/s² で上げ、5.0 m/s を超えたら「なし」。下げるのはすぐ。
- 1 回の時間: 0.3〜0.4 ms(CPU の中央値。ほかの作業と CPU を分け合った状態)。spec の目安 0.1〜0.3 ms より少し長いが、1 ms の線より十分短い。

## 3. 正解(`data/derived/clean_driving_20261011/tcp_labels_v1.npz`、CPU 4〜6 秒)

| 項目 | 値 |
|---|---|
| フレーム | 27,502(split の 52 シーンの全フレーム) |
| 通過点が有効 | 19,491(70.9%)。学習に使うフレーム(future_valid & vehicle_valid、2.5G の前後を除く)では ジャンル 4: **90.9%**・5: **84.2%** |
| 無効の理由 | 窓の中で衝突の前後に掛かる 4,426・記録の外(シーンの終わりの 1 秒など)1,556・後退 1,360・IMU の切れ目 669・ジャイロの飽和 0 |
| 目標速度 p10 / 50 / 90 / 99 | 1.55 / 2.50 / 3.73 / 4.44 m/s(spec の確認値 1.42 / 2.49 / 3.75 / 4.46) |
| ジャイロのバイアス | ログごと。−0.0016・−0.00106 rad/s ほか(research と同じ) |
| 符号の確認 | ステアとヨーレートの相関の中央値 +0.92、左回りのシーン 50 / 52、ステアが左(+150 µs)で走るフレームの 1 秒先の y > 0 が 92.8%、右で y < 0 が 86.4%、前進で x > 0 が 98.3% |

有効の割合が research(96.1%・93.3%)より低いのは、spec の規則で「窓が衝突の前 1 秒〜後 2 秒に掛かる」(= 衝突の 2 秒前から)と「窓の中で後退」を
足したため。無効のフレームも `actions` の L1 は今までどおり学習する(通過点・速度の損失だけ 0)。

## 4. 較正(`tcp_calib_v1.json`、学習の 41 シーンの人の操作だけ、CPU 2 秒)

| 項目 | 値 |
|---|---|
| ステアの遅れ | **40 ms**(R² 0 ms 0.681・40 ms **0.804**・80 ms 0.767・120 ms 0.627) |
| 中心 | 1441.0 µs(モデル B の `steer_center_us` と同じ計算) |
| 小さい切れ角の傾き | 左 0.00390・右 0.00399 /m/µs(research 0.00393・0.00401) |
| いっぱいに切った曲率(右・左)[1/m] | v < 2: −1.83・+1.56 / 2〜3: −1.39・+1.20 / 3 以上: −1.69・+0.95 |
| スロットル | thr = 1490.7 − 72.1·v − 13.6·(v_tgt − v)/0.33、前進の上限 361 µs、学習の誤差 31.4 µs(61,091 サンプル) |

**スロットルの係数が spec の例(c0 1530.2・c1 −82.7・c2 −25.4)と違う理由**: spec の例は research の値で、research は「前進のフレームだけ」の条件なしに
全フレームで当てはめていた。同じ計算を全フレーム・v(t+0.33) でやり直すと 1529.1・−82.5・−24.7 になり、research と合う。
この実装は spec の手順の文(「前進のフレーム thr < 中立 − 10」)に従った。定常 2.75 m/s での差は約 10 µs(前進 204 µs 対 194 µs)。

3 m/s 以上の帯の右いっぱい(−1.69)は、その帯の −475 µs の区切りが 4 サンプルしかなく、spec の規則(30 個未満は全速度の表の値)で全速度の値が入ったもの。
表示用の `v_curv` で右の速い帯を甘く見るおそれがある(判定には使わない)。

## 5. 正解を入れたときの制御器の誤差(`tcp_check/oracle.json`、制御器と較正だけで生じる誤差の下限)

正解の通過点(0.1 秒刻み)・目標速度を `TCPPost` に入れ、人の操作(`future_actions`)と比べた。stride 3、2.5G の前後を除く、通過点が有効なフレーム。
定数は spec のまま(T_la 0.25 s・L_min 0.3 m。調整していない)。

| | ステア k1 / k3 / k10 [µs] | ステア 平均 | スロットル(hybrid)平均 |
|---|---|---|---|
| 学習 5,084 フレーム・**頂点の向き(実装)** | 54.6 / 57.1 / 56.6 | **56.2** | 58.3 |
| 学習・弦の向き(spec の文字どおり) | 64.7 / 88.4 / 65.0 | 69.8 | 58.3 |
| 検証 1,396 フレーム・**頂点の向き(実装)** | 50.1 / 53.2 / 52.3 | **52.2** | 71.4 |
| 検証・弦の向き | 58.9 / 81.5 / 59.0 | 65.1 | 71.4 |
| 参考: モデル B(compare.json) | 42.7 / 48.7 / 56.1 | — | 41.5〜50.7 |

research の下限(ステア 約40 µs @ 0.25 s)より大きいのは、research が 0.02 秒刻みの細かい道と本当の向きを使ったのに対し、ここはモデルと同じ 0.1 秒刻みの通過点
だけを使うため。**通過点からステアを作っても L1 ではモデル B に勝てない見込み**(research §4.6 と同じ結論)。走行の既定が操作の枝なのはそのため。

## 6. spec からの変更・追加(M の判断用)

1. **pure pursuit の向き θ_k**: spec §3.2.4 は「その区間(弦)の向き」。弦の向きのままだと、円の上でも τ_k が区間のどこにあるかで追う曲率が 0.6〜1.4 倍に揺れる
   (0.1 秒で 0.2 m 進む区間)。上の表のとおり k3 が 88 µs に跳ねる。**頂点の向き**(原点 = 0 = 自車の向き、途中の頂点 = 隣り合う弦の向きの平均、
   最後は折り返し)を時間で補間する形にした。円の上で ±4% 以内(テスト)。道の上の位置 P_k・先の点 G_k・見る距離は spec のまま。
2. 較正の表で、**どの速さにもサンプルが無い区切り**(表の端 ±525・±575 µs)は、全速度の表の値ではなく、その速度帯の端の値をそのまま延ばす(= いっぱい)。
   全速度の表にもデータが無いので、spec の規則のままだと別の帯の外挿値が入って「いっぱいの曲率」が速い帯でも大きくなっていた。
3. 制御器の `branch`: 道が短い(0.2 m 未満)・通過点が数値でないとき、trajectory はステアだけ操作の枝にして `branch = 'fallback'`(スロットルは制御器のまま)。
   目標速度・今の速さが無いときは mode 1〜3 でも操作の枝(+ 上限)で `fallback`。どの場合も例外は出さない。
4. 今の速さが分からない(NaN)ときの上限: 止まっているとして `thr_ff(上限, 0)`(上限は弱くしか効かない。走行はセンサーが古ければその前に中立で止める)。
5. fusion の履歴(過去 1 秒の出力のステア)は、走行の先読みの行(`params['lookahead']`)の値を残す。時刻が戻ったら空にし、長さは 256 で切る。
6. `finalize` は学習の後に呼ばれるので止まらない: 較正を読めなければ警告して `{}`(後処理なし = 走行は操作の枝だけ。走行タブに「後処理を使えない」と出る)。
7. npz に `wp_steps`・`wp_dt`・`scenes`・`scene_frames`・`summary` を足した。キャッシュは splits.json の sha1 に加えて、各シーンのフレーム数が変わっても作り直す。
   `wp_steps`・`wp_dt` を既定から変えたときは別のファイル名(既定のキャッシュを壊さない)。
8. `postprocess` に `steer_table.center_us`・`throttle_ff.neutral_us`・`calib_source` の数字(R²・学習の誤差)を足した(表示・記録用。制御器は ref の中心・中立を使う)。

## 7. 他の持ち主へ

- **E(`tools/policy_zoo.py`)**: tcp の追加の評価(spec §5.5)は `post.replay(cfg, ref, outputs, erpm, t_ns, params, scene=...)` で、走行と同じ `TCPPost` を通る。
  ADE/FDE・目標速度の誤差の正解は `tcp_labels.labels_by_scene(data_root, keys=('waypoints', 'v_tgt', 'wp_valid'))`(検証のシーン、`wp_valid` のフレーム)。
  `v_hat` は `controller.softmax_expectation(speed_logits, postprocess['speed_bins'])`。
- **I(走行・画面)**: `info` は `mode`・`branch`('control'・'hybrid'・'fusion:ctrl'・'fusion:traj'・'trajectory'・'fallback')・`v_hat`・`v_cmd`・`v_curv`(nan 可)・`cap`(なしは 0)。
- **M**: `data/derived/clean_driving_20261011/` に正解と較正がある(CPU の試し学習で作った。何度作っても同じ中身)。

## 8. 確かめたこと・確かめていないこと

- 既定の大きさ **15.59M**(spec 15.6 ± 0.3)。CPU の試し学習(spec §7 のコマンド、2 ステップ)で `jetformer.onnx`(746 ノード、`CumSum` 1、ループ・分岐なし)・
  `model_config.json`(`postprocess` つき)ができ、`runtime.load_run` → `post.Stage` で 4 つの `tcp_mode` が動いた。約 20 秒。
- 関係するテスト: `test_policy_tcp.py` 18・`test_policy_tcp_controller.py` 21・`test_policies.py` ほか枠組み・走行のテストが通る。全体(ほかの方策の担当のテストを含む今の作業ツリー)**359 passed, 2 skipped**(CPU 約100秒)。途中で1本、実行の順番で落ちるテストがあったが、自分のテストの作り方の誤り(LayerNorm の後の特徴に定数の重みを足しても出力が変わらない)で、直した。
- **GPU が要るので確かめていない**: TensorRT FP16 で `waypoints` の差がゲート(0.02 m)に入るか(bench の乱数の重みで「通過点の大きさの 0.7%」= 3.7 m で約 0.026 m の見込み。
  ぎりぎり)、AMP での学習、30 エポックの時間(bench の見込み 18 分)。入らなければ、その学習結果の `engine.precision` を bf16 → fp32 にする(dinov2 と同じ手順)。
- 学習した `tcp` の通過点・速度の頭の精度、`hybrid` の L1・R1 の進入の速さは、段階4の比較(GPU)で出る。
