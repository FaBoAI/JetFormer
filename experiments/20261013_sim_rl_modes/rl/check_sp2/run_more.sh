#!/bin/bash
# 速さいろいろ(speed)の確認 r2: 近い所のまわりの細かい組と 2 つ目の新しい開始(2026-10-14、確認の段)。run_all.sh のあとに走らせる
set -u
D=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_sp2
LOG=/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_speed_r2/run_all.log
echo "追加の始め $(date -Is)" >> $LOG
HOT_GATES=4,5 HOT_CV=1,0.875,0.75,0.625,0.5 $D/run_set.sh HOT_G45 ${NSH:-3} >> $LOG 2>&1
HOT_GATES=16,17,18 HOT_CV=1 $D/run_set.sh HOT_G17 ${NSH:-3} >> $LOG 2>&1
HOT_GATES=7,8 HOT_CV=0.625,0.5 $D/run_set.sh HOT_G78 ${NSH:-3} >> $LOG 2>&1
$D/run_set.sh FRESH2 ${NSH:-3} >> $LOG 2>&1
HOT_GATES=4,5 HOT_CV=1 HOT_POLICY=speed_r1 $D/run_set.sh HOT_G45_R1 ${NSH:-3} >> $LOG 2>&1
echo "追加の終わり $(date -Is)" >> $LOG
