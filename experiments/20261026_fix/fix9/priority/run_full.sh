#!/bin/bash
# 試験一式(10/29 優先のモデルの直しのあと。10/28 fix8 の一式と同じ形: 実験の試験 21 本 + 台本の形の試験 2 本 + tests)
# 使い方: bash experiments/20261026_fix/fix9/priority/run_full.sh   (中で jf-heavy --lock big を使う)
set -u
cd /home/jetson/jetformer || exit 1
PY=/home/jetson/miniconda/envs/jetracer/bin/python
OUT=${OUT:-/home/jetson/jetformer_reports/fix_20261026/fix9/pytest_full_20261029.txt}
BT=${BT:-experiments/20261026_fix/tmp_pytest_full_20261029}          # /tmp や隠しフォルダでない所(snap の Firefox が読めるように)
EXP=$(cat experiments/20261026_fix/fix9/priority/exp_tests.txt)
mkdir -p "$BT"
{
  echo "開始 $(date '+%m/%d %H:%M:%S')"
  df -h / | tail -1
  echo "実験の試験(pytest の形) $(echo "$EXP" | wc -l) 本"
  echo "$EXP"
  echo "--- 台本の形の試験 2 本(python -I で流す)"
  for t in experiments/20261025_train_v2/c3/test_desk_v2.py experiments/20261025_train_v2/verify_d/test_c1_default_off.py; do
    "$PY" -I "$t" 2>&1 | tail -3
    echo "$t rc=${PIPESTATUS[0]}"
  done
  echo "--- pytest(終わった試験の一時ファイルは消す: tmp_path_retention_policy=failed)"
  PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock big --mem 3872M -- \
    "$PY" -m pytest -q -p no:cacheprovider -rfEs -o tmp_path_retention_policy=failed --basetemp="$BT/bt" tests $EXP
  echo "rc=$?"
  echo "終了 $(date '+%m/%d %H:%M:%S')"
  df -h / | tail -1
  du -sh "$BT"
} > "$OUT" 2>&1
