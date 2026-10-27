#!/bin/bash
# 反証 7-2 回目・safety: exp8_hold.py を 4 つ並べて回す
cd /home/jetson/jetformer
D=experiments/20261026_fix/fix7/review_safety_r2
PY=/home/jetson/miniconda/envs/jetracer/bin/python
for args in "orig 50" "human 50" "orig 100" "human 100"; do
  set -- $args
  /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $PY -I $D/exp8_hold.py $1 $2 > $D/out/exp8_hold_$1_$2.log 2>&1 &
done
wait
echo all done
