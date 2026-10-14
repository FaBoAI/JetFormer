#!/bin/bash
# 確認 line r4: 試験の組の作り直し(542)と新しい組(1,717)を 3 本で走らせる(続きから再開できる)
C=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln4
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4
mkdir -p $O/runs $O/logs
for k in 0 1 2; do
  ( /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 node $C/batch4.mjs $O/testjobs4.json $O/runs/test_$k.jsonl $k 3 \
    && /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 node $C/batch4.mjs $O/jobs4.json $O/runs/fresh_$k.jsonl $k 3 ) > $O/logs/run_$k.log 2>&1 &
done
wait
echo "全部終わり $(date)" >> $O/logs/run_all.log
