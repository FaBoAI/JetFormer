#!/bin/bash
# 確認 line r4: run_all4.sh が終わったら run_more4.sh を始める
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4
until grep -q "全部終わり" $O/logs/run_all.log 2>/dev/null; do sleep 20; done
/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln4/run_more4.sh
