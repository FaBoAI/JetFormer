#!/bin/bash
# 離脱復活 r4 の確認: 足した回(jobs_extra.json)を 2 本。出力にある回は飛ばす
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r4
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_rc4
H="/home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15"
for i in 0 1; do $H node $D/ck_batch.mjs $R/jobs_extra.json $i 2 $R/runs/rows_x$i.jsonl >> $R/runs/log_x$i.txt 2>&1 & done
wait
echo done > $R/runs/EXTRA_DONE
