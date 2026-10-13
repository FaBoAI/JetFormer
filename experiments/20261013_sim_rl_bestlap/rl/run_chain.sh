#!/bin/bash
# 本番の RL を順番に 1 本ずつ動かす(protocol budget.order)。2026-10-13、RL の学習の段
#   rl/run_chain.sh <B ステップ> [名前:variant:種 ...]
#   既定: main-s0 main-s1 main-s2 open-s0。終わった run(done.json あり)は飛ばし、途中の run は last.pt から続ける(電源断のあとも同じコマンド)
set -u
STEPS=${1:-10000000}; shift || true
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=(main-s0:main:0 main-s1:main:1 main-s2:main:2 open-s0:open:0)
HERE=/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/rl
OUTROOT=/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl
mkdir -p "$OUTROOT/logs"
echo "[$(date -Is)] 連続実行の開始 B=$STEPS ${RUNS[*]} (pid $$)" >> "$OUTROOT/logs/chain.log"
for r in "${RUNS[@]}"; do
  IFS=: read -r NAME VARIANT SEED <<< "$r"
  if [ -f "$OUTROOT/$NAME/done.json" ] && [ -n "$(ls "$OUTROOT/$NAME/evals/"*.json 2>/dev/null)" ]; then
    echo "[$(date -Is)] $NAME はもう終わっています" >> "$OUTROOT/logs/chain.log"; continue
  fi
  echo "[$(date -Is)] $NAME を始めます" >> "$OUTROOT/logs/chain.log"
  "$HERE/run_rl.sh" "$NAME" "$VARIANT" "$SEED" "$STEPS"
  code=$?
  echo "[$(date -Is)] $NAME 終了コード $code" >> "$OUTROOT/logs/chain.log"
  [ -f "$OUTROOT/STOP_CHAIN" ] && { echo "[$(date -Is)] STOP_CHAIN があるので止めます" >> "$OUTROOT/logs/chain.log"; exit 0; }
done
echo "[$(date -Is)] 連続実行の終わり" >> "$OUTROOT/logs/chain.log"
