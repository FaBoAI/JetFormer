# 検証中のトライアル(experiments)

JetFormer の実車走行・学習・解析の検証の記録。作業で使った解析スクリプト・レポート(Markdown)・図をそのまま置いている。
ツールとして整えたものは `tools/` にある(`label_events.py`・`build_ai_dataset.py`・`make_intervention_set.py`・
`offline_rl.py`・`rl_*.py`・`course_*.py` など)。

| フォルダ | 中身 |
|---|---|
| [`20261010_drive_reports/`](20261010_drive_reports/) | 2026-10-10 の学習・走行の評価レポート(`tools/evaluate.py`・`tools/drive_report.py` の出力)。AI走行 006/007 の解析(パラメータ・衝突の場所・人の立て直し・ToF による衝突の予告)、モデルの精度比較 |
| [`20261011_overnight/`](20261011_overnight/README.md) | 10/10 夜〜10/11 朝: 衝突のない走行データでの学習(モデルA: 操作値あり / モデルB: 操作値なし)と精度比較、AI走行 009/010/101 の解析、course.html からのコース図、ロードマップとオフライン強化学習・事前学習の調査(`research/`) |
| [`20261011_day_summary/`](20261011_day_summary/fast_and_safe_plan.md) | 10/10 の1日のまとめ(`facts.md`)と「ぶつからずに速く走る車への道筋」(データ・制御・学習の3つの探索 `explore_*.md` と計画 `fast_and_safe_plan.md`)。構成図・Transformer の図(`fig/`) |
| [`20261011_offline_rl/`](20261011_offline_rl/README.md) | オフライン強化学習(RECAP 型の最小形)のチャレンジ: 価値・アドバンテージ・方策(filtered BC / AWR / アドバンテージ条件付き / 陰性対照)の学習と机上の判定(ゲートC)。結果は「机上では走行の精度は上がらなかった。走行はモデルBのまま」 |

## 注意

- 生データ・データセット・学習済みモデル・TensorRT エンジン・動画・音声は入れていない(`data/` は git の外)。図は幅 1600px・256色に縮小している。
- レポートの中のパス(`/home/jetson/jetformer_reports/...`・`data/...`)は、Jetson 上で作業したときの場所。
- 数字は各レポートを書いた時点のもの。後のレポートで直したものがある。たとえば 006 の周回数は `20261011_day_summary/facts.md`、
  オフライン強化学習の結論は `20261011_offline_rl/README.md`(レビュー後の最終値)を正とする。
- 実車での確認(A/B/A)はまだしていないものが多い。机上の結果は実車での改善を示すものではない。
- `tools/course_fit.py`・`course_map.py`・`course_landmarks.py`(走行ログをコース図に当てはめる)は検証中で、保留検証が終わっていない。
