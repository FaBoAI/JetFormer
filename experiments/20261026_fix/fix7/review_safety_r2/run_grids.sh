#!/bin/bash
# 反証 7-2 回目・safety: 反証 7-1 回目の grid_onoff と同じグリッドを今の版(fix7 r1 のあと)で流し直す(g8.py)。4 つずつ並べる
cd /home/jetson/jetformer
D=experiments/20261026_fix/fix7/review_safety_r2
PY=/home/jetson/miniconda/envs/jetracer/bin/python
run() { RV8_TAG="g_$1_$2" /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $PY -I $D/g8.py $1 $2 orig > $D/out/grid_$1_$2.log 2>&1; }
run fr_rev None & run fr_rev 1.3 & run fr_brake 1.5 & run fr_brake None & wait
run w61_rev 1.3 & run w61_brake 1.5 & run sv_rev None & run sv_brake None & wait
run cs_brake 1.45 & run cs_rev 1.45 & run w61_rev None & run w61_brake None & wait
echo all done
