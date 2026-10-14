#!/bin/bash
# line r4b の下調べ(固める前): 確認 line r3 の再確認で崩れた 16 回・ゲート 29〜32 の角 × 右寄りの形・ゲート 16/17 の箱(新しい種)を、
# 方策(r3・r4 の 300 万・400 万)× シールド(r3 = v4 の動き・r4 = v5・r4b = v5b・なし)で走らせる。選び・試験には使わない。
#   bash r3/line4b/dev/run_dev4b.sh   (jf-heavy --lock none、nice 15、2 本)
set -u
L=/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line4b
O=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r4b/dev
POL=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r4/train/main/policies
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 600 -- nice -n 15"
export JF_TMP=/tmp
mkdir -p $O
run() { # 名前 方策 シールド(r3|v5|v5b|none) [--only ...]
  local name=$1 pol=$2 sh=$3; shift 3
  [ -s "$O/$name.json" ] && { echo "$(date -Is) skip $name" >> $O/chain.log; return; }
  local a=()
  case $sh in r3) a=(--shield-config $L/dev_cfg/sh_r3.json);; v5) a=(--shield-config $L/dev_cfg/sh_r4_v5.json);; v5b) a=(--shield-config $L/dev_cfg/sh_r4b_v5b.json);; none) a=(--no-shield);; esac
  echo "$(date -Is) start $name" >> $O/chain.log
  $H node $L/eval_line_r4b.mjs --set dev4b --core original --jobs 2 --policy $pol "${a[@]}" "$@" --out $O/$name.json > $O/$name.log 2>&1
  echo "$(date -Is) end $name $?" >> $O/chain.log
}
cd $L
run r3pol_r3sh_fail16 $POL/step_000000000.json r3 --only DEV-R3B-FAIL16
run r3pol_v5_fail16 $POL/step_000000000.json v5 --only DEV-R3B-FAIL16
run r3pol_v5b_fail16 $POL/step_000000000.json v5b --only DEV-R3B-FAIL16
run r4_3M_v5b $POL/step_003014656.json v5b
run r4_4M_v5b $POL/step_004014080.json v5b
run r4_3M_none $POL/step_003014656.json none
run r3pol_none $POL/step_000000000.json none --only DEV-G30R60,DEV-G30W40,DEV-R3B-FAIL16,LN-FAM6
run r4_4M_none $POL/step_004014080.json none
run r4_3M_v5 $POL/step_003014656.json v5
echo "$(date -Is) all done" >> $O/chain.log
