#!/bin/bash
# 離脱復活 r4 の確認: jobs.json を 5 本 + jobs_fmx.json(強さの中身を上限に固定した写しの runtime)を 1 本。出力にある回は飛ばす(停電のあとの続きはこのまま走らせ直す)
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r4
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_rc4
H="/home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15"
mkdir -p $R/runs
for i in 0 1 2 3 4; do $H node $D/ck_batch.mjs $R/jobs.json $i 5 $R/runs/rows_$i.jsonl >> $R/runs/log_$i.txt 2>&1 & done
wait
CK_RTDIR=$D/rt_forcemax $H node $D/ck_batch.mjs $R/jobs_fmx.json 0 1 $R/runs/rows_fmx.jsonl >> $R/runs/log_fmx.txt 2>&1
echo done > $R/runs/ALL_DONE
