#!/bin/bash
# r3 の下調べ(離脱復活 + シールド): 元の仕組み(web)とシールドあり(r3/runtime)で REC60 + FP100 を走らせる。3 本ずつ
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc
H="/home/jetson/bin/jf-heavy --mem 1G --lock none --need-mb 500 -- nice -n 15"
SH='{"modeShield":{"recover":{"c_nominal":0.035,"c_pessimistic":-1,"governor":{"enabled":false},"identify":{"enabled":false},"unstick":{"enabled":true,"c":0.03}}},"shieldSkipPerturb":true}'
mkdir -p $O/rc_web $O/rc_sh
for k in 0 1 2; do JF_RT=web $H node r3/rc/batch_rc3.mjs $O/jobs_rc3.json $k 3 $O/rc_web/rows_$k.jsonl & done; wait
for k in 0 1 2; do JF_R3="$SH" $H node r3/rc/batch_rc3.mjs $O/jobs_rc3.json $k 3 $O/rc_sh/rows_$k.jsonl & done; wait
echo "[$(date +%T)] 全部終わり"
