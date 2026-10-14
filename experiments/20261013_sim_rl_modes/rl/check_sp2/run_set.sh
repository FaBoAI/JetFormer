#!/bin/bash
# 速さいろいろ(speed)の確認 r2: 1 つの組を N 本に分けて走らせる(2026-10-14、確認の段)。メモリが少ない(スワップなし・ほかの学習が動いている)ので 1 本 1G・nice 15・
# 空きメモリ 900 MB 以上で始める(JetFormer の学習が 4.5 GB まで使うことがあるため)
#   ./run_set.sh REPLAY 3        (CHK_RT=web ./run_set.sh REPLAY_WEB 3)
set -u
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_sp2
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_speed_r2/runs
S=$1; N=${2:-3}
mkdir -p $OUT
pids=()
for ((k=0;k<N;k++)); do
  /home/jetson/bin/jf-heavy --mem 1G --lock none --need-mb ${NEED:-900} -- nice -n 15 node $D/batch_sp2.mjs --set $S --shard $k/$N --out $OUT/$S.$k.jsonl &
  pids+=($!)
  sleep 4   # 1 本ずつ始めて、次の空きメモリの確かめに前の分が入るように
done
for p in "${pids[@]}"; do wait $p; done
cat $OUT/$S.*.jsonl > $OUT/$S.jsonl && rm -f $OUT/$S.[0-9]*.jsonl
echo "$S 終わり $(date -Is) $(wc -l < $OUT/$S.jsonl) 回"
