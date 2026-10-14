#!/bin/bash
# 速さいろいろ(speed)の確認 r2: 全部の組を走らせる(2026-10-14、確認の段)。組ごとに 3 本(1 本 1G・nice 15)
set -u
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_sp2
LOG=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_speed_r2/run_all.log
echo "始め $(date -Is)" >> $LOG
for S in ${SETS:-FRESH GRID FRESH_CAP FRESH_FAST STDV REPLAY_R1 REPLAY_FB DR SEEDCHK}; do
  $D/run_set.sh $S ${NSH:-2} >> $LOG 2>&1
done
CHK_RT=web $D/run_set.sh REPLAY_WEB ${NSH:-2} >> $LOG 2>&1
echo "全部終わり $(date -Is)" >> $LOG
