#!/bin/bash
# r3 の下調べ(no_wall のシールド v4 の試作)をまとめて走らせる。jf-heavy --lock none -- nice -n 15、3 本ずつ
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/nw
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 700 -- nice -n 15"
V4B='{"shield":{"c_nominal":0.09,"worst":{"enabled":true,"c":0.07},"governor":{"weak_extra":0.92}}}'
V4A='{"shield":{"worst":{"enabled":true,"c":0.07},"governor":{"weak_extra":0.92}}}'
mkdir -p $O/v4b $O/v4a
for S in VX FN3 ABX NG; do echo "[$(date +%T)] v4b $S"; JF_R3="$V4B" $H node verify_nw3.mjs --sets $S --jobs 3 --out $O/v4b; done
echo "[$(date +%T)] v4a FN3"; JF_R3="$V4A" $H node verify_nw3.mjs --sets FN3 --jobs 3 --out $O/v4a
echo "[$(date +%T)] 全部終わり"
