#!/bin/bash
# no_wall の本番の学習のチェックポイントを、出るたびに検証の組(protocol selection、高速版・シールド v2)で評価する(2026-10-14)。
#   rl/nw/eval_checkpoints.sh <run の出力> [jobs]
# 学習の --eval の代わり(シールドを直したので、全部のチェックポイントを同じシールドで評価するため)。done.json があり、全部の方策を評価したら終わる。
set -u
RUN=$1; JOBS=${2:-1}
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
mkdir -p "$RUN/evals/no_wall"
cd "$EXP"
while :; do
  todo=0
  for p in "$RUN"/policies/step_*.json; do
    [ -e "$p" ] || continue
    out="$RUN/evals/no_wall/$(basename "$p")"
    [ -s "$out" ] && continue
    todo=1
    /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 \
      node rl/eval_modes.mjs --mode no_wall --policy "$p" --set validation --core fast --jobs "$JOBS" --out "$out" > "${out%.json}.log" 2>&1
    echo "[$(date -Is)] 評価 $(basename "$p"): $(tail -1 "${out%.json}.log" | cut -c1-400)"
  done
  if [ -e "$RUN/done.json" ] && [ $todo -eq 0 ]; then break; fi
  sleep 20
done
echo "[$(date -Is)] 全部のチェックポイントを評価しました"
