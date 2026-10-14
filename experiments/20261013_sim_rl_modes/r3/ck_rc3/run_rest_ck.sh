#!/bin/bash
# 離脱復活 r3 の確認(20:40): jobs.json の残り(jobs_rest.json、まだ行の無い回だけ)を 5 本で
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_rc3
H="/home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15"
for i in 0 1 2 3 4; do $H node $D/ck_batch.mjs $R/jobs_rest.json $i 5 $R/runs/rows_r$i.jsonl >> $R/runs/log_r$i.txt 2>&1 & done
wait
echo done > $R/runs/ALL_DONE_REST
