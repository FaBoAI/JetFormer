#!/bin/bash
# 離脱復活 r3 の確認: 全部の回を 3 本 + 強さの上限の試し 1 本で走らせる(jf-heavy --lock none・nice 15)
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_rc3
H="/home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15"
for i in 0 1 2; do $H node $D/ck_batch.mjs $R/jobs.json $i 3 $R/runs/rows_$i.jsonl > $R/runs/log_$i.txt 2>&1 & done
CK_RTDIR=$D/rt_forcemax $H node $D/ck_batch.mjs $R/jobs_fmx.json 0 1 $R/runs/rows_fmx.jsonl > $R/runs/log_fmx.txt 2>&1 &
wait
echo done > $R/runs/ALL_DONE
