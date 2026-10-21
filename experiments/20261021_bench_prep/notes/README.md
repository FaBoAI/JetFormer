# 実車のベンチマーク(10/23・10/24)の準備(2026-10-21)

- **依頼**(ユーザー、10/21 10:00 ごろ):「現在の課題を解決したいですね。あとは、テスト走行に向けて、いろいろなぱたーんの学習済みモデルを作成し、実際の走行で、ベストモデルを見つけ出したい」「ベンチマーキング的に、レポートを作成したいですね。その準備もおこなってください」。
- 実車は走らせていません。PCA9685・I2C・カメラ・ToF(STEVAL-VL53L9)・コントローラ基板には触れていません。推奨のモデル(モデル B)・走行の保存値・DEFAULT_PARAMS の既存の値は変えていません。git commit はしていません。
- まとめの本文は Jetson の `/home/jetson/jetformer_reports/bench_prep_20261021/README.md` です(このフォルダは、その作業に使ったコード・記録)。
- その写し(まとめ・当日の手順書・事前登録の protocol.json と直し・机上のベンチの表と図・ベンチマークレポートの雛形)は [`report/`](report/README.md) にあります。

## 結論

**10/23 の 1 日目は B・SLOW・ENS・B・AUG・ACT・WGT・B の 8 ブロックで走ります。机の上では順位を決められないので、実車で決めます。**

- 候補 9 本(B と 8 本)の机上のベンチを作り、走る前に固定した決まり(protocol.json + 直し A1)で 5 本を選びました。
  - 新しく 3 本を学習しました: **SLOW**(上限 v1.1 を記録に再生し、スロットルの正解を上限の出力に付け替え → R1 の手前で自分で緩める)・**WGT**(R1・GM・DE の損失の重み 3 倍)・**AUG**(カメラの明るさ・ずれ・ToF の雑音の拡張)。
  - **ENS**(10/15 の 2 本の平均の 1 つのエンジン)を、走行だけの方策 `ens_average` で走れるようにしました。
- 保留の L1 は B 49.1 µs、候補は 46.4〜50.8 µs(SLOW を除く)。ENS(−2.7 µs)以外は、種の違いの幅(1.2 µs)と同じくらいです。
- SLOW は R1 の手前 1 秒のスロットルを B より 40 µs 緩めます(B −8.3 → SLOW +31.9 µs)。ただし全体にも遅くなります(L1 63.4 µs。ステアだけなら 54.6 µs)。
- 区間の速度上限(v1.1: R1 2.5・GM 2.2・DE 3.0 m/s)を drive.py に `zone_caps` で **既定 OFF** でつなぎました。OFF の出力は変更前とビット単位で同じです(10/10 の 8 本の再生)。
- 走行タブに「走る日の準備」(チェック・基準の設定)・「ベンチマークの候補」・上限のスイッチ・「AI の時間」を足しました。
- 夜の 1 コマンド `tools/bench_nightly.sh`(ラベル → 成績表 → G1 → ブレーキの表 → レポート)を作り、10/10 の記録で通しました。
- 最後の pytest 一式: **878 passed・3 skipped・失敗 0**。

## フォルダ

| 場所 | 中身 |
|---|---|
| `data/make_bench_tables.py` | SLOW の付け替えの表・WGT の重みの表を作る(data/derived/bench_20261021/) |
| `train/` | 既定のままなら train.py が変える前と同じかの確かめ(check_default_same)・順番の道具の試験 |
| `gpu_queue.sh`・`gpu_queue.py`・`gpu_queue.tsv` | 3 本の学習 → エンジン → 確認を GPU 1 つで順に回した台本 |
| `logs/` | 学習・エンジン・確認の記録、順番の記録(gpu_queue_summary.md) |
| `ens/` | ENS の配置(place_ens.py)と確認の記録 |
| `desk/` | 机上のベンチ(desk_lib・desk_eval・desk_gpu・数え直しの recount.py)・候補の選び(finalists.py)・試験 |
| `ui_s2/` | 画面の確かめ(偽のサーバーとヘッドレス Firefox) |
| `tools_check/` | 道具を 10/10 の記録で回した出力(成績表・G1・ブレーキの表) |
| `verify_drive/`・`verify_models/`・`verify_ui/` | 確認の段のコード(速度上限の再生・候補 6 本・本物の画面) |
| `fix_drive/`・`fix_all/` | 直しの段の確かめ(画面・止める条件 A2 の 10/10 への当てはめ) |
| `notes/` | この文書・ベンチマークの雛形を作るスクリプト・最後の試験の記録(logs/) |

- リポジトリに入れた道具: `tools/section_scorecard.py`・`cap_replay.py`・`brake_table.py`・`bench_report.py`・`bench_nightly.sh`・`bench_day2.py`・`tof_floor_ref.py`、方策 `jetformer/policies/ens_average.py`、画面の API `jetformer/web/bench.py`。

## Jetson の上の報告(リポジトリの外)

| 場所 | 中身 |
|---|---|
| `jetformer_reports/bench_prep_20261021/README.md` | 10/21 のまとめ(課題 → 対策 → 結果・候補・確認・決めてほしいこと) |
| `…/runbook_20261023.md`・`runbook_20261024.md`・`todo_20261022.md` | 当日の手順書・前の日にすること |
| `…/plan/` | 比べ方の事前登録 protocol.json(sha256 `1dbb279e…0a2bb`)・直し A1〜A9・レポートの形・選んだ結果 |
| `…/desk/` | 机上のベンチ(desk_bench.json・desk_bench.md・図) |
| `…/benchmark/` | ベンチマークレポートの雛形(机上の部分だけ埋まっている) |
| `…/build/`・`…/verify/` | 段ごとの記録と確認の報告 |
| `data/bench_candidates.json` | 候補・基準の値・10/23 の並び(画面と道具が読む) |

## 作り直し

```bash
cd ~/jetformer
PY=/home/jetson/miniconda/envs/jetracer/bin/python
# 机上のベンチの数え直し(GPU なし。保存した予測から)
$PY experiments/20261021_bench_prep/desk/recount.py
# ベンチマークの雛形(bench_report.py の出力に、実車の部分の空の表を足す)
$PY -I experiments/20261021_bench_prep/notes/make_benchmark_template.py
# 試験
/home/jetson/bin/jf-heavy --mem 3G --lock big -- $PY -m pytest -q tests experiments/20261016_b2_curation experiments/20261016_sim_calibration
```

## 注意

- 学習結果・エンジン・予測・付け替えの表(data/)は git の外です。
- 机上の結果は、実車での改善を示すものではありません。実車は 10/23・10/24 です。
- 区間の速度上限を ON で走るのは、10/23 の B の記録で G1 に合格し、ユーザーがもう一度了承したあとです(4 つめの上限 = 位相が不確かなときの 3.0 m/s も含めて)。
