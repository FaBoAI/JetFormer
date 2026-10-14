#!/bin/bash
# 確認 line r4: 攻めの組(360)を 2 本で(続きから再開できる)。シールドあり、あとで同じ開始のシールドなしも
C=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln4
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4
for k in 0 1; do
  ( /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 node $C/batch4.mjs $O/advjobs4.json $O/runs/adv_$k.jsonl $k 2 \
    && /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 node $C/batch4.mjs $O/advjobs4.json $O/runs/advnosh_$k.jsonl $k 2 '{"shield":false}' ) > $O/logs/adv_$k.log 2>&1 &
done
wait
echo "攻めの組 終わり $(date)" >> $O/logs/run_all.log
