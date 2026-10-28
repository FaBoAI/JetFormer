# 係 disk: 11/3 の空きディスクの見積もりと、消してよい候補(了承のための 1 枚。2026-10-26 20:5x)

- 依頼(ユーザー、10/26 10:0x):「それでは、今日の課題の整理と、それを修正する作業を開始してください。」
- 依頼(ユーザー、10/26 13:4x):「ここまでの課題の整理、修正コードの作成、時間はたっぷりあります」→ 急がず、正しく、全部。
- 目標: ぶつからずに、止まってから 3 周 25 秒。次の実車は 11/3。
- この係の仕事: critic.md 4 章 8・README 6.2 の 16(11/3 の朝の「空き 10 GB 以上」に届かない見込み)の手当て。11/3 の朝と夜の空きを見積もり、消してよい候補を並べる。**消すかはユーザーが決める。**
- 守ったこと:
  - **何も消していない・動かしていない・書き換えていない。** 測っただけ(df・du・stat・cmp・find・snap list・journalctl --disk-usage・conda clean --dry-run)。
  - GPU・torch は使っていない。numpy で予測の npz を読んだだけ(`jf-heavy --lock none --mem 1500M`)。pytest は流していない。
  - 書いたのはこの 1 枚と、読むだけの台本 `experiments/20261026_fix/fix3/disk/disk_check.py` だけ。lock2 の 55 本・protocol_v2・lock2(v2)・decisions_v3.json・既定値・data/raw・labels には触れていない。git commit していない。車・装置・jetformer-web には触れていない。

## 0. 要点

| 項目 | 中身 |
|---|---|
| 今の空き | **10.42 GB**(10/26 20:55。段 1 は 1 本目 v3d-s0 が済み、予測の途中) |
| 段 1 のあと(10/27 朝〜昼) | **6.9 GB**(残り 11 本 × 0.312 GB + 予測 12 本 × 0.01 GB = 3.55 GB) |
| pytest 一式の最中 | **3.0 GB**(作業場所 3.9 GB)。終わって消せば 6.9 GB に戻る |
| 11/3 の朝(何も消さない) | **5.7 GB**(幅 4.7〜6.1)。手順書の目安 10 GB に **4.3 GB 足りない** |
| 11/3 の夜(何も消さない) | **4.5 GB**(幅 3.0〜5.5)。1 日の記録は約 1.0 GB |
| 危ないこと | pytest の作業場所を消し忘れると 11/3 の朝 1.8 GB・夜 0.6 GB(多めなら走行中にいっぱい)。記録の係(recorder)は空きを見ずに書く(いっぱいになると記録が書けずに失敗する。前もって止める仕組みは無い) |
| 勧め | **A1 + A2(v2・B-FAST の last.pth 22 本、3.84 GB)+ B1(同じ ONNX をハードリンクにまとめる、0.57 GB。何も失わない)** → 11/3 の朝 **10.1 GB**・夜 8.9 GB |
| 消すとき | 段 1 と予測が全部終わってから・pytest 一式の前(10/27 昼〜)。main session がユーザーの了承のあとに |
| 消さないもの | 学習 v3 の 12 本と段 0 の 4 本(消すと数から外れる・学習し直す。4 章)。ユーザーのもの・決まりで外すもの |

## 1. 見積もり(GB。中 = 下の 2 章の中の値)

| 時点 | 何も消さない | A1 + A2 | A1 + A2 + B1 | + C1(ゴミ箱) | pytest の作業場所を残す(何も消さない) |
|---|---|---|---|---|---|
| 今(10/26 20:55) | 10.4 | — | — | — | — |
| 段 1 のあと(10/27 朝〜昼) | 6.9 | 10.7 | 11.3 | 14.1 | — |
| pytest 一式の最中 | 3.0 | 6.8 | 7.4 | 10.2 | — |
| **11/3 の朝** | **5.7**(4.7〜6.1) | 9.6(8.5〜9.9) | **10.1**(9.1〜10.5) | 13.0 | **1.8**(0.8〜2.2) |
| **11/3 の夜** | **4.5**(3.0〜5.5) | 8.4 | 8.9 | 11.8 | **0.6**(−0.9〜1.6) |

- 段 1 で「途中でやめる決まり」(同じ腕の種 0・1 が両方とも悪い → 種 2・3 を始めない)が効くと、腕 1 つにつき +0.62 GB。
- 増えが「多め」だと A1 + A2 + B1 でも朝 9.1 GB。10 GB を守るなら C1 か C2 も要る。
- 空きは管理画面の「空き X GB」の表示(app.js の `#disk`。`/api/status` の disk.free_gb)で見られる。

## 2. 見積もりの元(測った値)

| もの | 値 | どう測ったか |
|---|---|---|
| 学習 1 本 | 0.312 GB(best 56.5・last 169.4・ONNX 56.5・エンジン 29.2 MB) | `du -B1`。v3d-s0(できたて)と v2 の 1 本が同じ。保存の途中に last.pth.tmp が 0.17 GB 増える(一時) |
| 机上の予測 1 本 | 約 0.01 GB | v2 の eval/pred(1.4 MB)・desk/pred(3.3 MB)・desk/trt(4.5 MB)から |
| 段 1 の残り | 11 本 + 予測 12 本 = 3.55 GB | キューのログ(v3d-s0 は 20:42 にエンジン、20:45 に予測を開始) |
| pytest 一式 | 3.9 GB(作業場所) | verify.md。既定の場所だと `/tmp/pytest-of-jetson` に **3 回分残る**(2 回流すと 7.8 GB) |
| 10/27〜11/2 のほかの増え | 0.7 / 1.0 / 2.0 GB(少なめ / 中 / 多め) | 10/26 は学習のほかに約 0.6 GB(会話の記録 0.32・v2 の机上 0.21・報告や動画)。毎日がこの重さではない |
| 11/1 の封(E5) | 0.1〜0.2 GB | 集合と 7 本の予測 |
| 11/3 の記録 | 0.5 / 1.0 / 1.25 GB | 10/24: 20 本・94 区切り・27.5 分で 0.836 GB(1 本 41.8 MB)。11/3 は 23 本 → 0.96 GB。時間で数えると約 16 分で 0.5 GB。やり直し 3 割で 1.25 GB |
| 11/3 の休み・夜の道具・報告・動画 | 0.1 / 0.2 / 0.4 GB | 10/24 の報告(ui_check 59 MB・動画の作業 91 MB) |

## 3. 消してよい候補(大きい順。どれもユーザーの了承が要る)

| # | 大きさ | 何か | 作った | 消すと何ができなくなるか | 控え | 勧め |
|---|---|---|---|---|---|---|
| **A1** | **3.34 GB** | v2 の 19 本の `last.pth`(`data/runs/v2_20261025-*/last.pth`。17 本 169 MB・msf 2 本 228 MB) | 学習 v2 のキュー(gpu_queue_v2、10/25 12:3x〜10/26 08:xx。ワークフロー) | 続きからの学習(どれも済み)・last の重みで新しい物差しや予測を作ること(check_v3 の「probe_last が無い」の行は埋まらない。門には使わない)。**gpu_queue_v2.sh を流し直すと、この本を最初から学習し直す**(流さない) | 重みの控えは無い。last の予測は `train_v2_20261025/eval/pred/<名前>_last.npz` に **19 本とも保存**(20:5x に読めることを確かめた)。best.pth・ONNX・エンジンは残る | ◎ protocol_v2 の決まり(start_conditions.disk「予測を保存して確かめたあと消す。候補に残るものは消さない」)どおり。v2 は候補 0 |
| C1 | 2.84 GB | ゴミ箱 `~/.local/share/Trash`(JupyterLab で 8/16〜8/26 に消した `jetracer/notebooks` の camera・dataset・model・model_trt・video の 6 回分と .bak) | ユーザー | ゴミ箱から戻すこと | 今の `notebooks/camera` 1.7 GB などは別にある。ゴミ箱の古い版の控えは無い | △ ユーザーのデータ。10/15 の整理(B1)でも残した |
| C2 | 1.20 GB | `jetformer_reports/sim_rl_modes_20261013` の `*.jsonl` 383 個(progress.jsonl 29 個は残す) | 10/13〜10/14 の強化学習モードの作業(ワークフロー) | 確かめの走りの 1 行ずつの記録から数え直すこと | 無い。まとめ(check_*.md・json)・方策・`data/sim_rl/modes` が指す json・md は残る | ○ 10/15 の B9 と同じ |
| S1 | 1.18 GB [sudo] | 無効な snap の古い版 5 つ(firefox 9035・gnome-46-2404 154・thunderbird 1281・core24 1644・snapd 27407) | Ubuntu の自動更新 | snap を前の版に戻すこと | 今の版は残る | ○ ハードリンクが 2 つあり、すぐには空かないことがある。更新でまた増える |
| **B1** | **0.57 GB(消さない)** | bundle の ONNX 9 個(`sim_rl_bestlap`・`sim_dart`・`sim_diverse_train` の `bundles/<名前>/jetformer.onnx`。`data/runs/<名前>/jetformer.onnx` とバイト単位で同じ)と `shortcut-rc-finetuned.zip`(2 か所で同じ)を**ハードリンクにまとめる** | 10/13〜10/15 のシミュレーターの bundle(ワークフロー) | 何も失わない(パスも中身も同じ)。片方をその場で上書きすると両方変わる(どちらも書き直す予定は無い) | 同じ中身が data/runs と ZIP にある | ◎ |
| **A2** | **0.51 GB** | B-FAST の 3 本の `last.pth`(`data/runs/bfast_20261025-e15-s*`) | 学習 v2 のキュー(10/25 11:3x〜12:3x) | A1 と同じ | A1 と同じ(3 本とも予測あり) | ◎ |
| S2 | 約 0.4〜0.5 GB [sudo] | journal の 4 週より古い分(8 月まで。今 623 MB) | システム | 8 月までの起動のログを読むこと(カメラの CRC の調べに使う 10 月分は残る) | 無い | ○ |
| S3 | 約 0.24 GB [sudo] | `/var/log/syslog`(268 MB。logrotate が無く 7 月から増え続け)を gzip に写して空にする | システム | 無い(中身は gz に残る) | gz | ○ |
| S4 | 0.14 GB [sudo] | apt のキャッシュ | apt | 無い(`apt update` で戻る) | — | ○ |

- A1・A2 を消しても困らないことを、台本で確かめた:
  - E5(11/1)は best.pth と model_config.json だけを使う(`fix3/e5_open_seal.py` の 304〜308・861〜867・1477 行)。予測するのは 5 本と v3 の候補。
  - eval_v2.py は保存した予測があれば重みを読まない(919〜926 行)。desk_v3・eval_v3 が v2 の本で照らすのは best.pth の大きさ・時刻だけ(desk_v3.py 810 行)。
  - lock2(v3)の 55 本・protocol_v2・lock2(v2)・check_seal_v2.py に、どの last.pth も入っていない。
- 今の時点でもう無いもの: `/tmp/pytest-of-jetson`(無い)・fix2/fix3 の `tmp_*`(無い)・古い学習の last.pth(10/15〜10/25 に消してある)・pip のキャッシュ・conda の要らない分(`conda clean --all --dry-run` で 0)。

## 4. 今は消さないもの・候補に入れないもの

| もの | 大きさ | なぜ |
|---|---|---|
| **学習 v3 の段 1 の 12 本**(last.pth を含む全部) | 3.7 GB(できたあと) | **eval_v3.py は、last の予測があるのに last.pth が無い本を「数に入れない」**(`v3_exclusion` → `h1_current`。大きさ・時刻で照らす)。キュー v3 は last.pth・ONNX が無い本を学習し直す。11/3 のあと、V1・V2 にならなかった本の last.pth で約 1.7〜2.0 GB 空けられる(そのときは eval_v3 を流し直さないと決めてから) |
| **段 0 の 4 本**(`diag_data/out/ablation/runs/abl-*`)の last.pth・ONNX | 0.68 + 0.23 GB | キュー v3(lock2 で固定)の `is_trained` が last.pth・ONNX を見る。電源断などでキューを流し直すと段 0 を学習し直し、decisions_v3.json が指す best.pth を書き換える。段 1 のあと、キューを閉じると決めたら候補にできる |
| 5 本(A_s0・A_s1・v0-s2・v0-s3・B)と bench_candidates.json の 10 本(重なりを除いて 13 本) | 2.1 GB | E5 の基準・推奨モデル B・11/3 のブロックの候補。5 本の last.pth はもう無い |
| 10/10〜10/22 の古い学習 47 本 | 6.2 GB | 走行タブの選択肢・方策の比べの出どころ。消すなら 11/3 のあとにユーザーと 1 本ずつ |
| 古い予測のキャッシュ(train_v2 の desk・eval、data/zoo、bench_prep など) | 合わせて約 0.5 GB | protocol_v2 が sha256 で固定する予測(reference_predictions_sha256)を含む。小さい |
| experiments の npz(.gitignore で git に入らない分) | 約 0.15 GB | lock2 が sha256 で固定する表(ablation の drop_*.npz・probe_set.npz)を含む。小さい。experiments の大きい分は段 0 の 4 本(上) |
| `/tmp`・`~/.cache` | 0.06・0.08 GB | 小さい。/tmp は動いている Claude の作業場所。`~/.cache/torch` は jetracer の授業の resnet18 |
| 決まりで外すもの | datasets 9.3・raw 2.0・labels・data/sim/releases・~/sim 1.2・封の記録(train_v2_20261025)・推奨モデル B | 依頼のとおり |
| 使っているもの | miniconda の環境 17.8・スワップ `/var/swap-jrbuild` 4.0(10/16 から使用中)・`~/.nv` 1.1(CUDA のキャッシュ。消してもまた増える)・`~/.claude` 2.3(会話の記録) | 消さない |
| ユーザーのプロジェクト | jetformer_board の Mac のビルド出力と .venv 1.44・lerobot/models 2.0・jetracer/notebooks 1.8 | 10/15 も残した。決めるならユーザー |

## 5. 了承の欄と、消すときの手順(main session が了承のあとに)

了承(ユーザーが選ぶ): [x] A1 [x] A2 [x] B1(10/27 ユーザー。済み) [ ] C1 [ ] C2 [ ] S1 [ ] S2 [ ] S3 [ ] S4(勧めは A1・A2・B1)

```bash
# 0. 前に見ること: 段 1 と予測が全部済み・学習や予測が動いていない・走行中でない
tail -5 /home/jetson/jetformer/experiments/20261026_fix/logs/gpu_queue_v3.log
pgrep -af 'tools/train.py|build_engine|probe_gates|eval_v3.py|desk_v3.py|e5_open_seal' || echo '動いていない'
df -h /

# A1・A2: v2 の決まりの道具で(学習が済み・best と last の予測が読める本だけ消す。名前を全部書く)
cd /home/jetson/jetformer/experiments/20261025_train_v2
P=/home/jetson/miniconda/envs/jetracer/bin/python
$P -I gpu_queue_v2.py prune-last          # 一覧だけ(22 本・3.8 GB と出るはず)
$P -I gpu_queue_v2.py prune-last --yes $(cd /home/jetson/jetformer/data/runs && ls -d v2_20261025-* bfast_20261025-*)

# B1: 同じ中身だけハードリンクにまとめる(data/runs の側は変わらない)
R=/home/jetson/jetformer_reports; D=/home/jetson/jetformer/data/runs
for f in $R/sim_rl_bestlap_20261013/bundles/*/jetformer.onnx $R/sim_dart_20261014/bundles/*/jetformer.onnx $R/sim_diverse_train_20261015/bundles/*/jetformer.onnx; do
  r=$(basename "$(dirname "$f")"); cmp -s "$D/$r/jetformer.onnx" "$f" && ln -f "$D/$r/jetformer.onnx" "$f"; done
Z1=$R/sim_rl_bestlap_20261013/eval/bundles/shortcut-rc-finetuned.zip; Z2=$R/sim_dart_20261014/eval/bundles/shortcut-rc-finetuned.zip
cmp -s "$Z1" "$Z2" && ln -f "$Z1" "$Z2"

# C1: ゴミ箱(ユーザーのデータ)
rm -rf /home/jetson/.local/share/Trash/files /home/jetson/.local/share/Trash/info
# C2: 強化学習モードの確かめの行(progress.jsonl は残す)
find /home/jetson/jetformer_reports/sim_rl_modes_20261013 -name '*.jsonl' ! -name progress.jsonl -delete

# S1〜S4(sudo)
sudo snap remove --revision=9035 firefox; sudo snap remove --revision=154 gnome-46-2404
sudo snap remove --revision=1281 thunderbird; sudo snap remove --revision=1644 core24; sudo snap remove --revision=27407 snapd
sudo journalctl --vacuum-time=4weeks
sudo sh -c 'gzip -c /var/log/syslog > /var/log/syslog.20261027.gz && truncate -s 0 /var/log/syslog'
sudo apt clean
```

- **pytest 一式(決まっている片付け)**: `--basetemp=<空の場所>` で流し、終わったらその場所を消す。`/tmp/pytest-of-jetson` があれば消す。既定の場所のまま 2 回流さない(3 回分残る)。
- 消したあと: `df -h /` と、`cd /home/jetson/jetformer_reports/fix_20261026 && sha256sum -c protocol_v3.sha256 protocol_v3_lock2.sha256`・`cd /home/jetson/jetformer_reports/train_v2_20261025 && sha256sum -c protocol_v2.sha256 protocol_v2_lock2.sha256`・`python3 -I /home/jetson/jetformer/experiments/20261025_train_v2/check_seal_v2.py` が全部 OK のままか。
- 11/2 の見直し: 下の台本でもう一度測る(読むだけ・1 分ほど)。

```bash
cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
  /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix3/disk/disk_check.py
```

## 6. 残る課題

| # | 課題 | 誰が | いつ |
|---|---|---|---|
| 1 | 3 章の了承(勧め A1・A2・B1。10 GB を守るなら C1 か C2 も) | ユーザー | 10/27〜11/2(消すのは段 1 のあと) |
| 2 | `gpu_queue_v2.py prune-last` の一覧はこの係では流せなかった(自動の許可で止められた)。同じ条件を `disk_check.py` で確かめた(22 本とも予測あり)。消す前に一覧を一度見る | main session | 消す前 |
| 3 | v2 の probe_last を参考に埋めたいなら、A1・A2 の前に GPU で(`probe_gates.py run --ckpt last`、1 本 25 秒ほど) | main session | 段 1 のあと。要らなければ飛ばす |
| 4 | 11/3 のあと: v3 の V1・V2 以外の last.pth(約 1.7〜2.0 GB)・段 0 の 4 本・古い学習 47 本の整理 | main session・ユーザー | 11/4〜 |
| 5 | 記録の係(recorder)は空きを見ずに書く。空きが 2 GB を切ったら走行を始めない、を手順書の朝の点検に足すか | main session | 11/2 |
