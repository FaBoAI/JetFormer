#!/bin/bash
# 確認 line r4: 追いの組(最速 120・シールドなし 1,717)を 3 本で(続きから再開できる)。run_all4.sh のあとに
C=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln4
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4
for k in 0 1 2; do
  ( /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 node $C/batch4.mjs $O/fastjobs4.json $O/runs/fast_$k.jsonl $k 3 \
    && /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 node $C/batch4.mjs $O/noshjobs4.json $O/runs/nosh_$k.jsonl $k 3 ) > $O/logs/more_$k.log 2>&1 &
done
wait
echo "追いの組 終わり $(date)" >> $O/logs/run_all.log
