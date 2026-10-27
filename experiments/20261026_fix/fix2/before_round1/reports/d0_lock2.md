# 係 d0: 段 0 のあとの決め(D0)と lock2_v3 の準備

- 作成: 2026-10-26 13:5x〜14:1x。係 d0(workflow のサブエージェント。main session に頼まれた)。
- 依頼(ユーザー、10/26 10:0x):「それでは、今日の課題の整理と、それを修正する作業を開始してください。」(13:4x「時間はたっぷりあります」→ 急がず、正しく、全部)
- 目標: ぶつからずに、止まってから 3 周 25 秒(1 周 約 8.12 秒)。次の実車は 11/3。
- この係の仕事: ① D0 を決める台本 ② lock2_v3 を書く台本 ③ deviations_v3.md に main session の代わりに書き足す。
- 守ったこと:
  - 車・PCA9685・I2C・カメラ・ToF・基板に触れていない。走行・保存・params の API は呼んでいない。jetformer-web は再起動していない。
  - 学習が使うもの(train.py・train_data.py・dataset.py・gpu_queue_v3.*・protocol_v3*・dataset_v3 の表・probe・data/datasets)は読むだけ。protocol_v2・lock2・既定値・data/raw・labels も変えていない。
  - GPU・torch は使っていない。pytest は自分の 2 ファイルだけ(`jf-heavy --lock none --mem 1500M`、作業場所は `fix2/tmp_d0` に置いて消した)。
  - **本物の decisions_v3.json・protocol_v3_lock2.json は書いていない**(段 0 がまだ終わっていない。書くのは main session)。git commit していない。

## 0. 要点

| 項目 | 結果 |
|---|---|
| D0 の台本 | `fix2/d0_decide.py`。段 0 の 4 本の best.pth の物差しを読み、D0 の判定と order を決まりどおりに出す。BF2-0(学習しない)も数え直して入れる。`--write` で decisions_v3.json に 1 回だけ書く(あれば上書きしない) |
| lock2 の台本 | `fix2/write_lock2_v3.py`。表・コード・probe・キュー・机上の台本の sha256、段 0 の答え 1〜9、D0・BF2-0、時刻を集める。書いたら chmod a-w と `.sha256`。`--dry-run` は出すだけ |
| キューとの形 | キューは lock2 が「ある」ことだけを見る(中身のキーは読まない)。decisions_v3.json は `D0.order`(腕 3 つ)と `"BF2-0".train` を読む。**本物の gpu_queue_v3.py で、この 2 つを読んで段 1 が RUN になり D0 の順に並ぶことを確かめた** |
| 試験 | 作り物の入力で **36 通った**(d0 25・lock2 11) |
| deviations_v3.md | 4 つ足した(2 段 launch の a〜i・3 区間の上限 184 µs・4 手順書の直し・5 D0 の読み方)。書ける状態だった(chmod は変えていない) |
| 今の段 0(14:1x) | abl-same は学習済み(一度メモリの上限 137 → キューが `--resume` で書き出し)。物差しはまだ。残り 3 本 |
| 今の lock2 の dry-run | 止まる理由 3 つ: decisions_v3.json が無い・段 0 の 8 の json が無い・`c3/eval_v3.py` が無い(`c3/desk_v3.py` は 14:08 にできた) |
| 一番の注意 | **キューの机上の予測は `experiments/20261026_fix/desk_v3/eval_v3.py` を見る**。eval_v3.py が `c3/` だけにあると、キューは予測を黙って飛ばす(5 章 1) |

## 1. D0 の台本 `d0_decide.py`

### 1.1 読むもの

| もの | 場所 | 確かめ |
|---|---|---|
| 段 0 の 4 本 | `diag_data/out/ablation/runs/abl-{same,noTK,noTK_noLB,tkB}-e15-s0/result.json`・`best.pth` | state done・15 エポック・種 0・学習 6,506 / 6,142 / 6,105 / 6,506・検証 1,559・drop の表の名前・tkB の差し替え 3,640 行・best.pth が 0 バイトでない |
| best.pth の物差し | (a) キューの `logs/probe/abl-*_best.json` (b) ablation_short.sh の `diag_data/out/ablation/probe_best.json` (c) result.json の `probe_best` | あるものは全部、best.pth より新しく、同じ値(±0.01 µs・±0.0001)。フレームは 441・581・372・582。数でない値は止まる |
| B の物差し | `diag_data/out/probe_check.json` の B | st10・st23 が 0(B 自身) |
| BF2-0 | `data/out/bf2_0.json` | 台本 2 つの sha256、決まりの式(上限 ≥ +10 かつ 上限 × 0.35 ≥ +10)で数え直して同じ |
| protocol_v3.json | 固定のまま | sha256 が `protocol_v3.sha256` と同じ・D0 の文が台本の決まりと同じ |

- 段 0 のコマンドには `--probe-npz` が無い。だから result.json に probe_best は無く、ふつうは (a) だけ。
- どれかがそろっていない・壊れている・食い違うときは、何も出さずに止まる(rc 2)。

### 1.2 決まり(protocol_v3.md 10 章のまま)と読み方

| 決まり | 式 |
|---|---|
| 確かめになるか | same の st23 − B ≥ +20 µs |
| 案 1 を確かめた | noTK の st23 − B ≤ same の 1/3、かつ noTK の stand_brake_side ≤ 0.10 |
| 案 D を確かめた | tkB で同じ式 |
| 案 2 を確かめた | noTK_noLB の low10 ≤ 0.01 |

| 確かめた案 | order(段 1 の腕の順) |
|---|---|
| 案 1 と案 D | v3d → v3n → v3b(protocol) |
| 案 1 だけ | v3n → v3d → v3b(読み方) |
| 案 D だけ | v3d → v3n → v3b(読み方) |
| どちらも無い・確かめにならない | v3b → v3d → v3n(protocol) |

- protocol に無いところの読み方(deviations_v3.md 5 に、答えを見る前に書いた):
  - 案 1 = v3n、案 D = v3d。片方だけのときは、その腕が先で、残りは protocol の runs の順。
  - 案 2 は 3 つの腕に全部入る(37 を外す)ので、順に効かない。答えは書く。
  - 「確かめにならない」は st23 を使う案 1・案 D に掛ける。
- 決まりは別の書き方(`recount_d0`。割り算なし・表で引く)でも数え、違えば止まる(protocol 16 章)。
- protocol stage0 6 の if_off(止まったら same・noTK だけで D0)は、`--same-noTK-only "<理由>"` を付けたときだけ。4 本そろっていれば、付けても止まる。

### 1.3 出す形(decisions_v3.json)

```
{"schema": "jf-train-v3-decisions/v1", "protocol_v3_sha256": "97c677d6…",
 "D0": {"order": ["v3d", "v3n", "v3b"], "status": "決まった", "rule": […], "why": "…",
        "confirmed": {"valid": true, "案 1(noTK → v3n)": true, "案 D(tkB → v3d)": true, "案 2(…)": true},
        "values": {"st23_minus_b_us": {…}, "probe_best": {…}}, "interpretation": […], "runs": {…sha256…}},
 "BF2-0": {"train": false, "values": {"bound_g090_us": {"A": -0.049, "B": 2.42}, "need_us": 28.571, …}},
 "pending": {"E1": …, "E5": …}}
```

- キューが読むのは `D0.order` と `"BF2-0".train` だけ(gpu_queue_v3.py の d0_order・bf2_answer と同じ読み方で確かめてから出す)。
- 今は decisions_v3.json が無いので、キューは BF2-0 を `data/out/bf2_0.json` から読んでいる(同じ「学習しない」)。

### 1.4 使い方(main session)

```bash
# 段 0 の 4 本の物差しがそろったら(logs/probe/abl-*_best.json が 4 つ)。読むだけ
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I /home/jetson/jetformer/experiments/20261026_fix/fix2/d0_decide.py
# 書く(1 回だけ。もうあれば止まる)
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I /home/jetson/jetformer/experiments/20261026_fix/fix2/d0_decide.py --write
```

## 2. lock2_v3 の台本 `write_lock2_v3.py`

### 2.1 中身(protocol_v3 lock.second_lock・build_data.md 10 章)

| 欄 | 中身 | 合わないとき |
|---|---|---|
| `files.v3_tables`・`train_samples` | 表 4 つ + tables_v3.json の sha256(build_data 10 章の値と tables_v3.json の記録)、学習 6,105・6,469・5,875 | 止まる |
| `files.stage0_tables`・`sets_v2`・`dataset_v2` | 段 0 の表 4 つ(protocol の値)、sets_v2.npz、dataset_card・splits・modes | 止まる |
| `files.code` | train.py `e04f7be8…`・train_data.py `b01513bf…`・dataset.py `e63e9673…`(build_data 10 章)。build_engine.py・trt_runtime.py・jetformer_v0.py は記録だけ | 止まる |
| `files.probe` | probe_gates.py `abcfc533…`・probe_set.npz `90050004…`(protocol)・probe_check.json | 止まる |
| `files.queue_v3` | gpu_queue_v3.sh `9befa63d…`・.py `baf9050c…`・.tsv `ab25aac3…`・試験・動いている写し | 止まる(写しの違いは警告) |
| `files.desk(机上の台本)` | `c3/desk_v3.py`・`c3/eval_v3.py`、キューの見る場所 `desk_v3/eval_v3.py` | 無ければ止まる。キューの場所に無ければ警告、中身が違えば止まる |
| `files` のほか | protocol_v3 の 5 つ・protocol_v2・lock2(v2)・decisions_v3.json・bf2_0.json・この台本と d0_decide.py | 記録 |
| `stage0.1` | `sha256sum -c` を 3 つ(v3・v2・v2 の lock2)と check_seal_v2.py を、書くときに流す | 止まる |
| `stage0.2〜5` | check_v3_data.json(全部 OK・封 0・3,640 行)・作用の基準 1441/1496 と 1433/1500・probe_onoff.json・probe_equiv.json | 止まる |
| `stage0.6・7` | decisions_v3.json の D0・BF2-0。d0_decide.py で数え直して同じか、段 0 の result.json と物差しが D0 のあとに変わっていないか | 止まる |
| `stage0.8` | 机上の台本の確かめの json(`--desk-check`。`all_ok`・`ok`・`pass`・`passed` のどれかが true) | 止まる(protocol の if_off「止める」) |
| `stage0.9` | 空き(5 GB 以上)・電源(`--power` の文) | 空きは止まる。電源が無ければ「未(ユーザー)」と書いて警告 |
| `when` | 書いた時刻・段 1 の学習 0 本(`data/runs/v3_20261026-*` があれば止まる) | 止まる |
| `answers`・`queue_contract`・`warnings` | D0.order・BF2-0.train、キューが読むもの、警告 | — |

- もう lock2 か `.sha256` があれば止まる(1 回だけ)。書くのは `os.link` で、あれば失敗する(上書きしない)。
- 書いたら json と `.sha256` を chmod a-w。最後に `sha256sum -c` を流して結果を出す。

### 2.2 使い方(main session)

```bash
# 下書き(書かない。止まる理由を出す)
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I /home/jetson/jetformer/experiments/20261026_fix/fix2/write_lock2_v3.py --dry-run \
  --desk-check <段 0 の 8 の json> --power "<ユーザーが AC か満充電を確かめた時刻と文>"
# 書く(段 1 の前に 1 回)
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I /home/jetson/jetformer/experiments/20261026_fix/fix2/write_lock2_v3.py \
  --desk-check <段 0 の 8 の json> --power "<…>"
cd /home/jetson/jetformer_reports/fix_20261026 && sha256sum -c protocol_v3_lock2.sha256
```

- 机上の台本が `c3/` 以外にあるときは `--desk`・`--eval` で場所を渡す。

### 2.3 今の dry-run(14:11)

| 段 0 | 今 |
|---|---|
| 1 固定・封 | OK(protocol_v3 の 7 つ・v2・v2 の lock2・check_seal_v2) |
| 2〜5 | OK(check_v3_data・作用の基準・ON/OFF・照らし合わせ) |
| 6・7 | decisions_v3.json が無い(段 0 の物差し待ち) |
| 8 | 確かめの json が無い・`c3/eval_v3.py` が無い |
| 9 | 空き 11 GB。電源は未(警告) |
| 表・コード・probe・キューの sha256 | 全部、学習の前の値と同じ |

## 3. deviations_v3.md に足したこと(14:1x)

| # | 見出し | 判定への影響 |
|---|---|---|
| 2 | 段 launch が仕様に足したこと a〜i(とくに d「走行の始めから AI 側の区間では出さない」) | 決まりは変えない。d のため 11/3 の ON の本は「CH3 を人側のまま走行開始 → AI 側」でないと始まらない。ON で始まらなかった本の数え方は protocol に無い → 11/3 の前に決める(案: ON の数に入れず印で見つけてやり直す) |
| 3 | 「launch_mps ≤ 3.0 なら区間の上限に当たらない」が違った(184 µs) | なし。11/3 は zone_caps OFF。文のもとは diag_launch.md 5.2(protocol 14 章の文そのものには無い。14 章の順番「あとの 2 つは前進を弱めるだけ」は合っている) |
| 4 | 1 回目のワークフローの fix が手順書を直した(スイッチで入れる・API で入れない・車輪を浮かせて実際の出力で・前 1.5 m・zone_caps OFF・確認の文の行) | なし。直す前は ON の本が OFF のまま数えられる恐れがあった |
| 5 | D0 の決まりの読み方(protocol に無いところ) | 段 1 の順だけ。段 0 の答えを見る前に書いた |

- もとの 3 つ(1 キューの行の順・キューが自動で書いた 2 つ)は消していない。キューは同じファイルに書き足すので、追記だけにした。

## 4. 試験(作り物の入力)

```bash
cd /home/jetson/jetformer/experiments/20261026_fix/fix2 && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
  /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/home/jetson/jetformer/experiments/20261026_fix/fix2/tmp_d0/bt --rootdir=. test_d0_decide.py test_write_lock2_v3.py
# → 36 passed in 4.4 s(終わったら tmp_d0 を消す)
```

| 試験 | 中身 |
|---|---|
| `test_d0_decide.py`(25) | 順の 4 通り + 後ろ側が 0.10 を超える形・same が +20 未満・境目(1/3 ちょうど・0.10 ちょうど・same 20 ちょうど)・B の値を引く・案 2 は順に効かない・4 本の 1 本が無い・done でない・サンプルの数・best.pth が 0 バイト・result.json が壊れている・物差しが無い・古い・フレームの数・NaN・2 つの出どころが食い違う・1 回だけ書く(上書きしない)・same・noTK だけ・BF2-0 の数え直し(境目 28.571)・本物の bf2_0.json・**本物の gpu_queue_v3.py がこの decisions を読む**(templates の段 1 の順・BF2 の SKIP の出どころ・lock2 が無ければ STOP) |
| `test_write_lock2_v3.py`(11) | 書く → chmod a-w・`sha256sum -c` OK・段 0 の 1〜9・sha256 の値・キューの場所の警告・2 回目は止まる・**本物の gpu_queue_v3.py がこの lock2 で段 1 を RUN にし D0 の順に並べる**・dry-run は書かない・机上の台本が無い・段 0 の 8 が無い/不合格・decisions が数え直しと違う/無い・段 0 の result.json や物差しが D0 のあとに変わった・キューの EVAL_V3 の場所 |

- 試験は本物の protocol・表・コード・キューを読むだけ。書くのは pytest の作業場所だけ。本物の decisions_v3.json・lock2 は作らない(試験で確かめる)。
- 固定の確かめ(`sha256sum -c`・check_seal_v2.py)は本物を流す。

## 5. main session への注意

| # | こと | どうする |
|---|---|---|
| 1 | **キューの机上の予測は `experiments/20261026_fix/desk_v3/eval_v3.py` を見る**(gpu_queue_v3.py・.sh の EVAL_V3)。机上の係は `c3/` に作っている。このままだと段 1 のあいだ予測を飛ばす(ログに「まだ無いか lock2_v3 の前」と出るだけ) | キューは変えない。案: `desk_v3/eval_v3.py` を `../c3/eval_v3.py` への symlink にする(lock2 は中身が同じなら通る)か、予測を main session があとで `eval_v3.py predict --runs <名前>` で回す |
| 2 | lock2 は **机上の台本と段 0 の 8 の確かめ**が要る(protocol の決まり)。机上が遅れると段 1(約 12 時間)の起動も遅れる | 段 0 の 8 の json(`all_ok` などが true)を机上の係に出してもらう。形は 2.1 |
| 3 | 電源(段 0 の 9)は台本では確かめられない | 夜の前にユーザーに聞き、`--power` に書く |
| 4 | D0 は 1 本・15 エポックの読み。変えるのは段 1 の順だけ | `d0_decide.py` を見てから `--write`。読み方は deviations_v3.md 5 |
| 5 | abl-same は 1 回目がメモリの上限(137)で止まり、キューが上限 4608M・`--resume` でやり直した(15 エポックのあとなので、学習の中身は 1 回目と同じ)。キューが deviations_v3.md に 2 つ書いた | そのまま。d0_decide は result.json の 15 エポック・done を見る |
| 6 | lock2 は train.py などの sha256 を学習の前の値と比べる。誰かが変えると止まる | 止まったら変えた係に聞く(決まりは変えない) |
| 7 | fix2 のほかの係が同じ時間に `jetformer/assist.py`(13:57)・`runbook_20261103.md`(14:03)を変えている | その分の deviations は各係が書く。この係の 2〜4 は 1 回目のワークフローの分だけ |

## 6. ファイル

| ファイル | sha256 | 中身 |
|---|---|---|
| `/home/jetson/jetformer/experiments/20261026_fix/fix2/d0_decide.py` | `bd5512f9…` | D0・BF2-0 → decisions_v3.json |
| `/home/jetson/jetformer/experiments/20261026_fix/fix2/write_lock2_v3.py` | `fceba903…` | lock2_v3 |
| `/home/jetson/jetformer/experiments/20261026_fix/fix2/test_d0_decide.py` | `beec23ab…` | 試験 25 |
| `/home/jetson/jetformer/experiments/20261026_fix/fix2/test_write_lock2_v3.py` | `b0be594b…` | 試験 11 |
| `/home/jetson/jetformer_reports/fix_20261026/deviations_v3.md` | — | 2〜5 を書き足した |
| この報告 `/home/jetson/jetformer_reports/fix_20261026/fix2/d0_lock2.md` | — | — |

- lock2 は書くときに `d0_decide.py`・`write_lock2_v3.py` の sha256 も入れる。書く前に台本を直したら、試験を流し直す。
