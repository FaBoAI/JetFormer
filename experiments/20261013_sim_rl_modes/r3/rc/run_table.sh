#!/bin/bash
# 離脱復活 r3: 立て直せる表を作る(4 本を並べて。lock rl = 学習と重ねない。nice 15)。終わったら merge_recoverable_table.py で表にする。
#   r3/rc/run_table.sh   (中で jf-heavy --mem 2G --lock rl -- nice -n 15 を使う)
set -u
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc3/table
mkdir -p $O
cd $E
N=4
echo "[$(date -Is)] 表づくりを始めます(shard $N 本)" >> $O/run_table.log
/home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 800 -- nice -n 15 bash -c "
  for k in \$(seq 0 $((N-1))); do
    if [ -f $O/rows_\$k.jsonl ] && tail -1 $O/rows_\$k.jsonl | grep -q '\"done\":true'; then echo \"shard \$k はもう終わっています\"; continue; fi
    node r3/rc/build_recoverable_table.mjs --shard \$k/$N --out $O/rows_\$k.jsonl 2>> $O/shard_\$k.log &
  done
  wait" >> $O/run_table.log 2>&1
echo "[$(date -Is)] 表づくり終わり(終了コード $?)" >> $O/run_table.log
