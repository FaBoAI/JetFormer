#!/bin/bash
# 確認 line r3 の全部の走り(4 ワーカー)。jf-heavy --lock none の中で nice 15 で動かす。続きから再開できる
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3
for v in cand nosh web fast; do
  for w in 0 1 2 3; do node ck_batch.mjs $v $w 4 >> $OUT/run_$w.log 2>&1 & done
  wait
done
echo ALL_DONE >> $OUT/run_0.log
