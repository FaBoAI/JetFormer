#!/bin/bash
# 速さいろいろ(speed)の確認 r1: 全部の組を走らせる(2026-10-14、確認の段)。メモリが少ない(スワップなし・ほかの学習が動いている)ので 1 本ずつ
set -u
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_sp
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_speed_r1/runs
mkdir -p $OUT
for S in ${SETS:-REPLAY REPLAY_FB STDV FRESH FRESH_CAP FRESH_FAST DR}; do
  /home/jetson/bin/jf-heavy --mem 1G --lock none --need-mb 450 -- nice -n 15 node $D/batch_sp.mjs --set $S --shard 0/1 --out $OUT/$S.jsonl
done
echo 全部終わり $(date)
