#!/bin/bash
# 障害物よけ(obstacle)の確認 r1: 組を走らせる(2026-10-14、確認の段)。メモリが少ない(スワップなし・ほかの学習が動いている)ので 2 本まで
set -u
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_ob
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r1/runs
mkdir -p $OUT
H="/home/jetson/bin/jf-heavy --mem 1G --lock none --need-mb 400 -- nice -n 15 node $D/batch_ob.mjs"
( $H --set VREPLAY --out $OUT/VREPLAY.jsonl; $H --set FRESH --shard 0/2 --out $OUT/FRESH_0.jsonl; $H --set EMPTY --out $OUT/EMPTY.jsonl; $H --set REC --out $OUT/REC.jsonl ) &
( $H --set FRESH --shard 1/2 --out $OUT/FRESH_1.jsonl; $H --set FRESH_FAST --out $OUT/FRESH_FAST.jsonl; $H --set NOOB --out $OUT/NOOB.jsonl ) &
wait
echo 全部終わり $(date)
