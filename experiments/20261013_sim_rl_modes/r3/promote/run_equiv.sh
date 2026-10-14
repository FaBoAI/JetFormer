#!/bin/bash
# 昇格の段: rt_merged と候補の runtime の比べ(3 本並べて。jf-heavy --lock none・nice 15)
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/promote/equiv
for k in a b c; do
  /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 600 -- nice -n 15 node r3/promote/equiv_runtime.mjs --cases $O/cases_$k.json --out $O/equiv_$k.json > $O/equiv_$k.log 2>&1 &
done
wait
echo "$(date -Is) equiv done" >> $O/equiv.status
