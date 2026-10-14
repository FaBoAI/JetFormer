#!/bin/bash
# 離脱復活 r3 の確認(21:10): 止まりの開始・試験のキックをシールドなしで(報告だけ。止まりとキックへのシールドの効き)
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_rc3
H="/home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15"
for i in 0 1 2; do $H node $D/ck_batch.mjs $R/jobs_extra3.json $i 3 $R/runs/rows_z$i.jsonl >> $R/runs/log_z$i.txt 2>&1 & done
wait
