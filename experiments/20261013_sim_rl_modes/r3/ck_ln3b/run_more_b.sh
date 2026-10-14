#!/bin/bash
# 確認 line r3(再実行): 試験の組 186 回の照らし合わせ(候補 + シールド)、ゲート 30 の角の組 218 回、最速 120 回を 3 本の働き手で
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3b
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b
for k in 0 1 2; do
  ( /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- nice -n 15 node batch_b.mjs $OUT/testjobs.json $k 3 $OUT/runs/test_$k.jsonl shield
    /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- nice -n 15 node batch_b.mjs $OUT/probejobs.json $k 3 $OUT/runs/cand_probe_$k.jsonl shield
    /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 1000 -- nice -n 15 node batch_b.mjs $OUT/fastjobs.json $k 3 $OUT/runs/fast_$k.jsonl fastest ) > $OUT/runs/more_$k.log 2>&1 &
done
wait
echo done > $OUT/runs/more.done
