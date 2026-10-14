#!/bin/bash
# r3 の下調べ(離脱復活の oracle の試作)を 4 つの開始で並べて走らせる(1 本ずつ jf-heavy --lock none -- nice -n 15)
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc
H="/home/jetson/bin/jf-heavy --mem 900M --lock none --need-mb 400 -- nice -n 15"
for i in 0 1 2 3; do $H node r3/rc/oracle_proto.mjs $O/oracle_s$i.jsonl =$i 3 & done
wait
echo "[$(date +%T)] 全部終わり"
