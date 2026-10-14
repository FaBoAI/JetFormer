#!/bin/bash
# 1 モードの取り込んだデータセットを確かめる(再生・教師の同一性・決まり / センサー・ラベル)。bash record/check_one.sh <key> <データセット名>
set -u
k=$1; n=$2
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/record
DS=/home/jetson/jetformer/data/datasets
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 900 --"
cd $EXP
echo "$(date -Is) replay_check $k $n"
$H nice -n 15 node integrate/replay_check.mjs --dataset $DS/$n --out $OUT/replay/$k.json; echo "replay $k exit $?"
echo "$(date -Is) verify $k"
$H nice -n 15 $PY -I record/verify_mode_dataset.py $DS/$n --out $OUT/verify/$k.json; echo "verify $k exit $?"
echo "$(date -Is) 終わり $k"
