#!/bin/bash
# 離脱復活の確認 r2: 新しい開始の一覧を 3 つに分けて走らせる(jf-heavy・nice 15・lock なし)。走らせる本体は確認 r1 の batch_rc.mjs(自前の作り方)
set -u
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_recover_r2
N=${N:-3}
for i in $(seq 0 $((N-1))); do
  /home/jetson/bin/jf-heavy --mem 1200M --lock none --need-mb 900 -- nice -n 15 node rl/check_rc/batch_rc.mjs $R/jobs.json $i $N $R/runs/rows_$i.jsonl > $R/runs/shard_$i.log 2>&1 &
done
wait
echo done > $R/runs/ALL_DONE
