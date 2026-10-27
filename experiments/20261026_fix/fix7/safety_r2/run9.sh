#!/bin/bash
# 直す 7-2 回目・係 safety: exp9.py のくみを 3 つずつ並べて流す(引数 = くみの並び)。出力 out/exp9_<くみ>.json・.log
cd /home/jetson/jetformer
D=experiments/20261026_fix/fix7/safety_r2
PY=/home/jetson/miniconda/envs/jetracer/bin/python
n=0
for k in "$@"; do
  /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $PY -I $D/exp9.py $k > $D/out/exp9_$k.log 2>&1 &
  n=$((n + 1))
  if [ $((n % 3)) -eq 0 ]; then wait; fi
done
wait
echo all done "$@"
