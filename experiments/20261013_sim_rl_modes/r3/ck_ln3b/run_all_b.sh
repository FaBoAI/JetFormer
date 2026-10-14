#!/bin/bash
# 確認 line r3(再実行): 新しい組をシールドありで 3 本の働き手で走らせる
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3b
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b
for k in 0 1 2; do
  /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- nice -n 15 node batch_b.mjs $OUT/jobs.json $k 3 $OUT/runs/cand_$k.jsonl shield > $OUT/runs/cand_$k.log 2>&1 &
done
wait
echo done > $OUT/runs/cand.done
