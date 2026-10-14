#!/bin/bash
# 実装の段の試しの学習(no_wall・cond・obstacle を各 50 万ステップ、数分)。lock rl・nice 15・メモリ 2G(rl/run_mode.sh)。2026-10-13
# recover は rl/smoke_recover_resume.sh(途中で止めて last.pt から続くことも確かめる)。
set -u
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
cd "$EXP"
for run in no_wall cond obstacle; do
  rl/run_mode.sh smoke-$run $run 500000 --ckpt-every 250000 --save-every 100000 --tag smoke
done
