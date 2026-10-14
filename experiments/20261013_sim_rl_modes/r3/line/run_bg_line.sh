#!/bin/bash
# r3 の下調べ(経路いろいろ): 開始のそろえ・シールドの試作を、確認 line r2 の組(HEAD・EXT・REPLAY・FRESH)で走らせる。2 本ずつ
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line
H="/home/jetson/bin/jf-heavy --mem 1G --lock none --need-mb 500 -- nice -n 15"
declare -A V
V[la1]='{"lineAlign":{"v_cap":1.5,"dpsi_on":0.6,"dpsi_off":0.3,"min_ds":2.5,"max_frames":150}}'
V[sh1]='{"modeShield":{"line":{"c_nominal":0.04,"c_pessimistic":-1,"governor":{"enabled":false}}}}'
V[lash1]='{"lineAlign":{"v_cap":1.5,"dpsi_on":0.6,"dpsi_off":0.3,"min_ds":2.5,"max_frames":150},"modeShield":{"line":{"c_nominal":0.04,"c_pessimistic":-1,"governor":{"enabled":false}}}}'
V[lapp1]='{"lineAlign":{"mode":"pp","dpsi_on":0.6,"dpsi_off":0.3,"ey_off":0.15,"min_ds":0.3,"v":1.2,"learn":0,"max_frames":120},"modeShield":{"line":{"c_nominal":0.04,"c_pessimistic":-1,"governor":{"enabled":false}}}}'
V[lapp0]='{"lineAlign":{"mode":"pp","dpsi_on":0.6,"dpsi_off":0.3,"ey_off":0.15,"min_ds":0.3,"v":1.2,"learn":0,"max_frames":120}}'
V[shu1]='{"modeShield":{"line":{"c_nominal":0.04,"c_pessimistic":-1,"governor":{"enabled":false},"unstick":{"enabled":true,"c":0.032}}}}'
for v in ${VARS:-la1 sh1 lash1}; do
  mkdir -p $O/$v
  for S in ${SETS:-HEAD EXT REPLAY}; do
    echo "[$(date +%T)] $v $S"
    JF_R3="${V[$v]}" $H node batch_ln3.mjs --set $S --shard 0/2 --out $O/$v/${S}_0.jsonl &
    JF_R3="${V[$v]}" $H node batch_ln3.mjs --set $S --shard 1/2 --out $O/$v/${S}_1.jsonl &
    wait
  done
done
echo "[$(date +%T)] 全部終わり"
