#!/bin/bash
# 直す 7-1 回目・係 safety: exp7.py のくみを順に流す(引数 = くみの並び)。出力 out/exp7_<くみ>.json・.log
cd /home/jetson/jetformer
for k in "$@"; do
  /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
    experiments/20261026_fix/fix7/safety/exp7.py "$k" > experiments/20261026_fix/fix7/safety/out/exp7_$k.log 2>&1
done
