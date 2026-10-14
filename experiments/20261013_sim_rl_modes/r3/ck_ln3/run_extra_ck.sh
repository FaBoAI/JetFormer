#!/bin/bash
# 確認 line r3 の追加の調べ(角の近く NB・ゲート 11 の開始 G11)。4 ワーカー
cd /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3
for w in 0 1 2 3; do node nb_probe.mjs cand $w 4 >> $OUT/extra_$w.log 2>&1 & done; wait
for w in 0 1 2 3; do node nb_probe.mjs nosh $w 4 >> $OUT/extra_$w.log 2>&1 & done; wait
for w in 0 1 2 3; do node g11_probe.mjs $w 4 >> $OUT/extra_$w.log 2>&1 & done; wait
echo EXTRA_DONE >> $OUT/extra_0.log
