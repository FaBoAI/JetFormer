#!/bin/bash
# fix8 係 safety 直す 8-1 回目: exp_q12.py の場面の組を 2 つずつ流す(out/exp_q12_*.json・.log)
set -u
JF=/home/jetson/jetformer
D=$JF/experiments/20261026_fix/fix8/proposal_r1
cd $JF
printf '%s\n' "rd2 0/1" "rd 0/2" "rd 1/2" "hm 0/1" "sl 0/2" "sl 1/2" "lw 0/4" "lw 1/4" "lw 2/4" "lw 3/4" \
  "fz 0/4 8801 900" "fz 1/4 8801 900" "fz 2/4 8801 900" "fz 3/4 8801 900" | \
  xargs -P 2 -I{} bash -c 'set -- {}; n=$(echo "$1_$2" | tr / o); PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B '"$D"'/exp_q12.py {} > '"$D"'/out/exp_q12_$n.log 2>&1'
echo all done
