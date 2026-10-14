#!/bin/bash
# 確認 line r4: 昇格の取り違えの試し(shield.mjs だけを別の版に替えた写しで、むずかしい組 180 回ずつ)。シールドなし・攻めの組が終わってから
C=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln4
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4
S=/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad
until grep -q "シールドなし 終わり" $O/logs/run_all.log 2>/dev/null; do sleep 20; done
for v in nw4v5 rc4v4 shared; do
  /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 800 -- nice -n 15 env CK4_RT=$S/rt_$v node $C/batch4.mjs $O/altjobs4_$v.json $O/runs/alt_$v.jsonl 0 1 > $O/logs/alt_$v.log 2>&1 &
done
wait
echo "取り違えの試し 終わり $(date)" >> $O/logs/run_all.log
