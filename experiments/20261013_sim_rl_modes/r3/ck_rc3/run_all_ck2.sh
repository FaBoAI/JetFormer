#!/bin/bash
# 離脱復活 r3 の確認(停電のあとの続き、20:30): jobs.json の残り(3 本、出力にある回は飛ばす)+ 足した回 jobs_extra.json(2 本)
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_rc3
H="/home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15"
for i in 0 1 2; do $H node $D/ck_batch.mjs $R/jobs.json $i 3 $R/runs/rows_$i.jsonl >> $R/runs/log_$i.txt 2>&1 & done
for i in 0 1; do $H node $D/ck_batch.mjs $R/jobs_extra.json $i 2 $R/runs/rows_x$i.jsonl >> $R/runs/log_x$i.txt 2>&1 & done
wait
echo done > $R/runs/ALL_DONE2
