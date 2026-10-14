#!/bin/bash
# 学習の段の後片付け(sim_dart_20261014): dart_train.py が終わるのを待ってから
#   1. 各一式の ZIP を、ユーザーのブリッジ(写し)に HTTP で読み込ませる(http_load_check.py → bundles/<名前>/http_load_check.json)
#   2. オフライン L1(offline_eval.py、GPU・lock big)
#   3. 図と表(train_report.py)
# 使い方: post_train.sh <dart_train.py の PID>
set -u
HERE=/home/jetson/jetformer/experiments/20261014_sim_dart_margin/train
REPORT=/home/jetson/jetformer_reports/sim_dart_20261014
PY=/home/jetson/miniconda/envs/jetracer/bin/python
SIM_PY=/home/jetson/jetformer/data/sim/venv/bin/python
JF=/home/jetson/bin/jf-heavy
PID=${1:-}
log() { echo "[$(date '+%F %T')] $*"; }
if [ -n "$PID" ]; then
  log "dart_train.py(PID $PID)の終わりを待ちます"
  while kill -0 "$PID" 2>/dev/null; do sleep 15; done
fi
log "dart_train.py が終わりました"
cd /home/jetson/jetformer || exit 1
for run in simdart_20261014-A1-s0 simdart_20261014-A2-s0; do
  z=$REPORT/bundles/$run.zip
  if [ ! -f "$z" ]; then log "$run: 一式の ZIP がありません。飛ばします"; continue; fi
  log "$run: ブリッジ(HTTP)の読み込み"
  $JF --mem 1300M --lock none --need-mb 1000 -- nice -n 10 "$SIM_PY" -I -B "$HERE/http_load_check.py" "$z" \
      > "$REPORT/bundles/$run/http_load_check.json.tmp" 2> "$REPORT/train/logs/$run.http_load.log"
  code=$?
  mv "$REPORT/bundles/$run/http_load_check.json.tmp" "$REPORT/bundles/$run/http_load_check.json"
  log "  → 終了コード $code"
done
log "オフライン L1(GPU)"
$JF --mem 2500M --lock big --need-mb 2500 -- nice -n 10 "$PY" -u "$HERE/offline_eval.py" > "$REPORT/train/logs/offline_eval.log" 2>&1
log "  → 終了コード $?"
log "図と表"
$JF --mem 800M --lock none --need-mb 800 -- nice -n 10 "$PY" -I "$HERE/train_report.py" > "$REPORT/train/logs/train_report.log" 2>&1
log "  → 終了コード $?"
log "済み"
