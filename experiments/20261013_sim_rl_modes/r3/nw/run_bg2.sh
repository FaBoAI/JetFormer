#!/bin/bash
# r3 の下調べ: 固定したシールド v4 の値(c_nominal 0.08・Q c 0.07・weak_extra 0.92・unstick なし)で FN3(標準の物理 600 回)と VX(480 回)
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/nw
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 700 -- nice -n 15"
V4C='{"shield":{"c_nominal":0.08,"worst":{"enabled":true,"c":0.07},"governor":{"weak_extra":0.92}}}'
for S in FN3 VX; do echo "[$(date +%T)] v4c $S"; JF_R3="$V4C" $H node verify_nw3.mjs --sets $S --jobs 3 --out $O/v4c; done
echo "[$(date +%T)] 全部終わり"
