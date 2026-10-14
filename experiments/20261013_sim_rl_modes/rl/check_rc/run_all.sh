#!/bin/bash
# 離脱復活の確認 r1: 一覧の全部を 3 つに分けて走らせる(jf-heavy・nice 15・lock なし)
set -u
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_recover_r1
N=${N:-3}
for i in $(seq 0 $((N-1))); do
  /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1200 -- nice -n 15 node rl/check_rc/batch_rc.mjs $R/jobs.json $i $N $R/runs/rows_$i.jsonl > $R/runs/shard_$i.log 2>&1 &
done
wait
echo done > $R/runs/ALL_DONE
