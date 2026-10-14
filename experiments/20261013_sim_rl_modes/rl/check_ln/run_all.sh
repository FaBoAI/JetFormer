#!/bin/bash
# 経路いろいろ(line)の確認 r1: 組を走らせる(2026-10-14、確認の段)。メモリが少ない(スワップなし・ほかの学習が動いている)ので 2 本まで
#   SETS="REPLAY FAM6V" bash run_all.sh
set -u
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_ln
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_line_r1/runs
mkdir -p $OUT
for S in ${SETS:-REPLAY FAM6V FB_MIX}; do
  /home/jetson/bin/jf-heavy --mem 1G --lock none --need-mb 400 -- nice -n 15 node $D/batch_ln.mjs --set $S --shard 0/1 --out $OUT/$S.jsonl
done
echo 全部終わり $(date)
