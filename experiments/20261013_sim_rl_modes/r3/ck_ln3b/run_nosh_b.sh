#!/bin/bash
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3b
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b
python3 noshjobs_b.py
for k in 0 1 2; do
  /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- nice -n 15 node batch_b.mjs $OUT/noshjobs.json $k 3 $OUT/runs/nosh_$k.jsonl noshield > $OUT/runs/nosh_$k.log 2>&1 &
done
wait
echo done > $OUT/runs/nosh.done
