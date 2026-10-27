# 机上の台本 desk_v3・eval_v3(係 desk、2026-10-26)

- 依頼(ユーザー、10/26 10:0x):「それでは、今日の課題の整理と、それを修正する作業を開始してください。」
- 依頼(ユーザー、10/26 13:4x):「ここまでの課題の整理、修正コードの作成、時間はたっぷりあります」
- 目標: ぶつからずに、止まってから 3 周 25 秒(1 周 約 8.12 秒)。次の実車は 11/3。
- この係の仕事: 段 1 の前に要る机上の台本(protocol_v3 の 6 章 8・7 章 欄・8 章 閾値・9 章 拒否権と門・10 章 E1〜E3)。
- GPU・torch は使っていない。車・基板・走行の API・jetformer-web に触れていない。キュー・学習の台本・protocol・既定値は変えていない。git commit していない。

## 0. 要点

| 項目 | 答え |
|---|---|
| 台本 | `c3/desk_v3.py`(欄を出す)・`c3/eval_v3.py`(判定・GPU の予測のまとめ・段 0 の 8)。名前は指示のとおり |
| 段 0 の 8 | **合格**。保存した 5 本・v2 の 22 本で protocol_numbers_v3.json と **1,170 個とも同じ**(最大の差 0.005 µs・0.00005。許す差は ±0.01 µs・±0.0001) |
| 自己試験 | protocol の数え方で、5 本は新しい門を全部通る。v2 の 19 本は全部落ちる。B-FAST の 3 本は後ろ側の門を通る。protocol の self_test と 22 本とも同じ |
| 全部の門 | 27 本に gates_v3 を全部当てると、合格は **B だけ**。A_s0・A_s1・B-FAST 3 本は v2 の門「低速のブレーキ ≤ 0.01」で落ちる(9 章 2) |
| 足りない値 | low10 が 20 本・probe_last が 27 本・TensorRT の欄が 2 本(v0-s2・v0-s3)。どれも段 0 の 8 には要らない(5 章) |
| キュー | 学習の出力・物差しの名前・engine_check.json・predicted の約束は gpu_queue_v3 と同じ(試験で照らした)。**キューは `desk_v3/eval_v3.py` を見る** → symlink 1 本で動く(6 章) |
| 試験 | `fix2/test_desk_v3.py` **31 個 合格**(7 秒)。GPU の手順は作り物の torch・TensorRT で通した |
| GPU の予測 | コマンドだけ用意(2.2)。1 本 約 7 分、12 本で約 1.5 時間(protocol の机上 約 2 時間の中) |

## 1. 置き場所

| ファイル | 中身 | sha256(lock2_v3 に入れる) |
|---|---|---|
| `/home/jetson/jetformer/experiments/20261026_fix/c3/desk_v3.py` | 欄(直線の回ごと・後ろ側・物差し・TensorRT)。GPU の予測(tab・trt) | `3221880bf09dba6957eaee6c93016651dfdb8a0540ef4215b1a67a51c63f014c` |
| `/home/jetson/jetformer/experiments/20261026_fix/c3/eval_v3.py` | H1・介入の欄・拒否権・門・E1〜E3・段 0 の 8・キューの predict/predicted | `91044fa2f2735bc0d1cf7d9e7896c42d7f8a65782d03c9d6e95a9d6ca0c48f57` |
| `/home/jetson/jetformer/experiments/20261026_fix/fix2/test_desk_v3.py` | 試験(pytest) | `bb73156bd0b4bac633b6fb878f31a1628a2155361b26f691811126b1ba07a405` |
| `/home/jetson/jetformer_reports/fix_20261026/fix2/check_v3.json`・`.md` | 段 0 の 8 の答え(`all_ok: true`・1,170 個・足りない値・27 本の門) | lock2 の `--desk-check` に渡す |

- 台本が読む 10/21 の部品(変えていない): `experiments/20261021_bench_prep/desk/desk_lib.py`・`desk_gpu.py`、`tools/policy_zoo.py`・`rl_evaluate.py`・`rl_policies.py`。lock2 に記録だけ入れるとよい。
- 学習のあとの出力(protocol files_out): `fix_20261026/desk_v3.json`・`.md`、`fix_20261026/desk_v3/`(pred・trt)、`fix_20261026/eval_v3/`(pred・eval_v3.json・md・decisions_v3_proposed.json)。**decisions_v3.json は書かない**(案を main session が写す)。

## 2. 使い方

```
cd ~/jetformer
P=/home/jetson/miniconda/envs/jetracer/bin/python
L="/home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 --"
G="/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 $P -u"
```

### 2.1 段 1 の前(今。CPU・3 秒)

| 手順 | コマンド | 出るもの |
|---|---|---|
| 段 0 の 8 | `$L $P -I -B experiments/20261026_fix/c3/eval_v3.py check` | `fix2/check_v3.json`(all_ok)・`check_v3.md` |
| lock2_v3 | `write_lock2_v3.py --desk-check /home/jetson/jetformer_reports/fix_20261026/fix2/check_v3.json …` | lock2 の係の台本。all_ok を読む |

- 台本を直したら、check を流し直す(check_v3.json に台本の sha256 が入っている)。

### 2.2 段 1 のあいだ(GPU。1 本ずつ)

| 手順 | コマンド | 中身・時間の見込み |
|---|---|---|
| まとめて | `$G experiments/20261026_fix/c3/eval_v3.py predict --runs <名前>` | ① H1・介入の best・last(約 1.5 分)② AI の記録の集合 16,621 行(約 1 分)③ TensorRT B + 候補(約 4 分)。そろった部分は飛ばす。中は別のプロセスで順に(中で jf-heavy を呼ばない) |
| 初回だけ | (自動) | B を予測し直して保存と照らす(H1 は 10/12 の B.npz、tab は v2 の B の tab。どちらも ≤ 0.01 µs)。train_data.py が変わったら、もう一度 |
| そろったか | `$P -I eval_v3.py predicted <名前>` | 0 = そろった・1 = まだ(今の best.pth・last.pth・エンジンのものか、大きさと更新時刻で見る) |
| 一覧 | `$P -I eval_v3.py commands` | 12 本の GPU のコマンドと、5 本・v2 の low10 を足すコマンド |

- lock2_v3 が無いと v3 の予測はしない(protocol lock.second_lock)。`--allow-before-lock2` は deviations_v3.md に書くときだけ。
- 走行・収集・管理画面の学習が動いているときは始めない(GET の状態だけ見る)。

### 2.3 腕の種がそろったら(10/27。CPU)

| 手順 | コマンド | 出るもの |
|---|---|---|
| 欄 | `$L $P -I -B experiments/20261026_fix/c3/desk_v3.py report` | `fix_20261026/desk_v3.json`・`.md`(v3・5 本・v2 の行) |
| 判定 | `$L $P -I -B experiments/20261026_fix/c3/eval_v3.py judge` | `fix_20261026/eval_v3/` の eval_v3.json・eval_v3.md・decisions_v3_proposed.json(E1〜E3 の案) |
| 空回し | `… eval_v3.py judge --dry-run-b` | v3 の 12 本の枠を全部 B で埋めて道筋を通す(`eval_v3/dryrun_B/`)。今の答え: E1 3 腕・E2 v3n → v3d → v3b・V1 v3n-s0・V2 v3n-s1 |

- desk_v3.json が予測より古いと、judge が注に書く。先に report を流す。

## 3. 段 0 の 8 の確かめ(10/26 14:32、`check_v3.md`)

| 部分 | 個数 | 外れ | 最大の差 [µs] | 最大の差 [割合] |
|---|---|---|---|---|
| 1 5 本の H1・介入(介入のステア・スロットル・全体、H1。best・last) | 40 | 0 | 0.000000 | 0 |
| 1 5 本の平均・sd・閾値(n=1〜4・k=2〜4) | 54 | 0 | 0.000049 | 0 |
| 2 新しい欄(5 本。直線・10/10・10/23・後ろ側・止まって出る・行の数・tab の sha256) | 35 | 0 | 0.000000 | 0 |
| 2 新しい欄(v2 の 22 本) | 155 | 0 | 0.000000 | 0 |
| 2 5 本の平均・sd | 20 | 0 | 0.000050 | 0.0000484 |
| 3 物差し(保存した予測から。27 本 × 4 欄) | 109 | 0 | 0.004987 | 0.0000495 |
| 3 物差し 対 GPU の probe_check(4 本) | 16 | 0 | 0.004558 | 0.0000441 |
| 3 low10: G1 の保存した予測 対 GPU の probe_check(4 本) | 4 | 0 | 0 | 0.0000454 |
| 3 5 本の平均・sd | 6 | 0 | 0.001098 | 0.0000183 |
| 4 v2 の 22 本の介入のステア・スロットル(best・last) | 89 | 0 | 0.000496 | 0 |
| 8 章 閾値の数え直し(介入のステア・直線・後ろ側・probe・直しが効いた・途中でやめる・人の走り) | 26 | 0 | 0.000351 | 0.0000445 |
| v2 の desk_v2.json と同じ(v2 の欄・低速のブレーキ・sanity・p99・エンジン) | 298 | 0 | 0.000005 | 0.0000050 |
| v2 の eval_v2.json と同じ(H1・介入・偏り・振れ幅) | 270 | 0 | 0.000050 | 0.0000489 |
| 自己試験(protocol の数え方) | 29 | 0 | — | — |
| 自己試験(全部の門で v2 の 19 本が落ちる) | 19 | 0 | — | — |

- 差の 0.005 µs は protocol_numbers_v3.json の丸め(物差しは 0.01 µs・0.0001)。
- probe の閾値は protocol と同じく、5 本の値を丸めてから数え直した。丸めない値からだと stand_brake_side の門は 0.0639(protocol は 0.0640。門は protocol の値を使う)。

## 4. 自己試験

| モデル | protocol の数え方(回ごとの直線・後ろ側・物差し st10・st23・stand) | 全部の門(gates_v3 の 16 個) |
|---|---|---|
| 5 本 | **5 本とも通る** | B だけ合格。A_s0 = 低速のブレーキ 0.023・low10 0.0206 で落ちる。A_s1 = 低速のブレーキ 0.208 で落ちる。v0-s2・v0-s3 = エンジンが無く「未」 |
| v2 の 19 本 | **19 本とも落ちる**(protocol の self_test の落ちた門の並びと 19 本とも同じ) | 19 本とも落ちる |
| B-FAST の 3 本 | 後ろ側 0.013〜0.020 で通る | 3 本とも低速のブレーキ(0.041〜0.050)で落ちる |

## 5. 足りない値(GPU で予測し直すのは学習のあと。待たない)

| 欄 | 足りない学習 | 理由 | 出し方 |
|---|---|---|---|
| 物差しの low10(10/10 の低速 582 フレーム) | 20 本: A_s1・v0-s2・v0-s3・B-FAST s1・s2・v2 の 15 本(v2h-aug2-e60 s0・s2・s3、e100-s0、e30 s2・s3、e60 s0・s1・s3、msf-aug2 s0・s1、v2m-e30 s0〜s3) | C3 の集合に 10/10 の低速が無い。FP32 の保存した予測は G1 の 7 本だけ(B・A_s0・v2h-e30-s0・s1・v2h-e60-s2・v2h-aug2-e60-s1・bfast-e15-s0) | `eval_v3.py commands` の最後の行(probe_gates.py run --ckpt best、1 本 約 20 秒)。出力 `fix2/probe_low10_ref.json` を desk_v3 が自動で読む |
| probe_last(last.pth の物差し) | 27 本 | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last。門には使わない(印「終わりで緩む」だけ) |
| sanity・推論・低速のブレーキ・エンジン | v0-s2・v0-s3 | エンジンが無い | 要らない(5 本の判定は H1・介入だけ使う) |

- v3 の 12 本は、学習のあとに result.json(probe_best・probe_last)とキューの予測でそろう。段 0 の 8 にはどれも要らない。

## 6. キューとの照らし合わせ(gpu_queue_v3.py・.sh・.tsv。変えていない)

| もの | キュー | この台本 | 確かめ |
|---|---|---|---|
| 学習の出力 | tsv のコマンドの `--out` = `data/runs/<名前>` | protocol の runs のコマンドの `--out` | 12 本とも同じ(試験) |
| 物差し | `logs/probe/<名前>_best.json`・`_last.json`(1 つだけの値) | 同じ読み方。段 0 の abl-same の本物の出力で照らした | 同じ |
| 学習中の物差し | train.py が result.json に probe_best・probe_last・probe_history | **result.json を先に使う**。キューの物差しと ±0.01 µs・±0.0001 で照らし、違えば注に書く | 試験 |
| エンジンの確認 | `<run>/engine_check.json`(accuracy_ok・actions_max_us・engine の大きさと時刻) | 同じ規則で engine_ok | 同じ |
| 削った種 | tsv の行の頭に `#` | 同じ読み方で「要る種」から外す | 試験 |
| 予測の約束 | `predicted <名前>`(0 なら飛ばす)・`predict --runs <名前>`(jf-heavy --lock big の中) | 同じ形 | 試験 |
| **台本の場所** | **`experiments/20261026_fix/desk_v3/eval_v3.py`** | `experiments/20261026_fix/c3/eval_v3.py` | **違う**。このままだとキューは予測を黙って飛ばす |

- 案(main session が決める。この係は作っていない): lock2_v3 のあとに symlink を 1 本。
  `mkdir -p /home/jetson/jetformer/experiments/20261026_fix/desk_v3 && ln -s ../c3/eval_v3.py /home/jetson/jetformer/experiments/20261026_fix/desk_v3/eval_v3.py`
  - symlink から `predicted` を流して動くのを確かめた(tmp の写しで)。lock2 の台本は中身が同じなら通す。
  - しないなら、学習のあとに 2.2 のコマンドを 12 本、手で流す。
- 途中でやめる決まりの数: キューは 42.4 µs・0.192(protocol の文)、protocol_v3.json は 42.369・0.1919。あいだに入ると分かれる。eval_v3 は protocol_v3.json で「要る種」を決め、分かれたら注に書く。

## 7. 決まりの中身(protocol → 関数)

| protocol | 中身 | 関数 |
|---|---|---|
| 7 章 欄 | 介入のステア・スロットルの L1(best・last)、直線の 10/10・10/23、後ろ側、動いている低速のブレーキ(参考)、物差し | `eval_v3.metrics`・`desk_v3.row_fields`・`back_frac`・`brake_parts`・`pick_probe` |
| 8 章 閾値 | protocol_v3.json thresholds_us の値をそのまま(書き直さない) | `eval_v3.thresholds` |
| 9 章 拒否権 | 介入のステア(平均 − 158.385 > 閾値(n) かつ全部の種が上)・偏り(v2 と同じ) | `veto_interv_steer`・`veto_bias` |
| 9 章 門 | v2 の門 9 個 + 回ごとの直線 2 個 + 後ろ側 + 物差し 4 個 = 16 個。欄が無い門は「未」で合格にしない | `gates_v3`・`gates_pass` |
| 9 章 途中でやめる・削った種 | 種 0・1 が両方外れ → 要る種は 0・1(n = 2)。削った種も外す。要る種の外の種は数えない | `futility`・`expected_seeds` |
| 9 章 印 | 長すぎ・上がり・終わりで緩む(参考) | `h1_fields`・`rise_mark`・`loosen_at_end` |
| 10 章 E1 | 拒否権なし・5 本に「悪い」でない腕。種がそろうまで「未」 | `decide_e1` |
| 10 章 E2 | 門に全部合格した種の数の多い順 → 同じなら H1 の L1 の平均が小さい順。門が「未」の種があれば仮 | `decide_e2` |
| 10 章 E3 | 1 番目の腕の種 0 → 3 で最初の合格 = V1・次 = V2。1 本だけなら V2 = 2 番目の腕の最初 | `decide_e3` |
| 10 章 直しが効いた | 同じ種の v2h-aug2-e60 との d(10/23 の直線 ≤ −7.835・後ろ側 ≤ −0.0241、全部 < 0)。選びには使わない | `fix_confirmed` |

## 8. 試験(`test_desk_v3.py`、31 個 合格・7 秒)

| まとまり | 中身 |
|---|---|
| 式 | 車に出る値 = drive.apply_gains(本物の関数を AST で)・後ろ側と前の割合・回で分けた直線・物差しの式・low10 |
| 閾値 | protocol_v3.json の値と protocol_v3.md の太字の数・v2 の門の文の数 |
| 門・拒否権 | ちょうど閾値は合格・少し超えると不合格(16 個とも)・欄が無いと「未」・止まった場面 300 未満は「参考」・n = 2 の閾値 |
| E1〜E3 | 順・同じ数のとき L1・V2 を 2 番目の腕から・候補なし・種がそろわない・途中でやめた腕(n = 2)・削った種・作り物の 10 本の通し |
| キュー | run_dir・物差しの名前・削った種の読み方・predict/predicted の約束(本物の gpu_queue_v3.py を読むだけで照らす) |
| 予測の印 | 重みを変えると「予測し直す」になる・0 バイトの npz を飛ばす・封の記録で止まる |
| GPU の手順 | 作り物の torch・TensorRT で predict(B の照らし → 候補)・predict-h1・trt(0 バイトの aligned を飛ばす)・まとめの predict が足りない部分だけ回す |
| 本物 | 段 0 の 8(1,170 個)・空回し(B で E3 まで)・守るファイルの sha256 が変わらない |

- 流し方: `jf-heavy --lock none --mem 1500M -- $P -I -B -m pytest -q -p no:cacheprovider --basetemp=<fix2/tmp_desk> test_desk_v3.py`。tmp は消した。pytest 一式は流していない。

## 9. 課題(main session へ)

| # | 課題 | 案 |
|---|---|---|
| 1 | **キューが机上の台本を見る場所が違う**(6 章)。このままだと段 1 のあいだ予測を飛ばす | lock2_v3 のあとに symlink 1 本。しないなら 2.2 を手で |
| 2 | **v2 の門「低速のブレーキ ≤ 0.01」は、B と同じ作り方の 5 本のうち A_s0(0.023)・A_s1(0.208)も落とす**。B-FAST 3 本・v2 の 19 本も全部落ちる。v3 の候補も落ちやすく、V1・V2 が無い(11/3 の 1 枠は B)見込みが高い | protocol は固定なので変えていない。結果を見る前に知っておく。11/3 のあと(v4)で門の決め方(2 × 標準誤差か)を見直す |
| 3 | 物差しの low10 ≤ 0.01 も A_s0(0.0206)・B-FAST s0(0.055)を落とす(2 と同じ心配) | 同じ |
| 4 | 5 本・v2 の low10 が 20 本足りない(5 章) | 学習のあとに 1 本 20 秒 × 20(参考。v3 の判定には要らない) |
| 5 | desk_v3 report・eval_v3 judge はキューが流さない | 腕がそろったら 2.3 を流し、decisions_v3_proposed.json を見て decisions_v3.json に写す |
| 6 | 「直しが効いた」の閾値は protocol に k = 4 しか無い。途中でやめた腕(k = 2)は同じ式で出して注に書く | 結論の言い方だけ(選びに使わない) |
| 7 | E4(B-FAST2)・封の T・V・E5(11/1)はこの台本の外 | BF2-0 が「学習しない」なので E4 は要らない見込み。封は 11/1 の手順で |
| 8 | lock2_v3 のあとに台本を直すと sha256 が合わなくなる | 直したら check を流し直し、deviations_v3.md に書く |
