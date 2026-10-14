#!/bin/bash
# r3(確認 r2 のあと。手順の外の参考の追加実験): 角・辺を多めに引くランダム化 + すき間の約束で、選んだ方策から 500 万ステップ学習を続ける。
# 種 21 と 22 を 1 本ずつ順に(deviations.md 2026-10-13 22:40 の事前登録のとおり)。電源断のあとも同じコマンドで last.pt から続く。
#   rl/r3/run_r3.sh
set -u
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap
INIT=/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/main-s1/checkpoints/step_010010624.pt
for SEED in 21 22; do
  NAME=r3-s$SEED
  if [ -f /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/$NAME/done.json ]; then echo "$NAME は終わっています"; continue; fi
  "$EXP/rl/run_rl.sh" "$NAME" main "$SEED" 5000000 --dr r3 --reward r3 --init-ckpt "$INIT" --lr-start 1e-4 --ckpt-every 1000000 \
    --tag "r3-corner-dr-clear3-from-main-s1-10M" || exit $?
done
