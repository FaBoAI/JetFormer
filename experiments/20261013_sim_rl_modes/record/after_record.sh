#!/bin/bash
# 記録の段: 取り込んだデータセットの確かめ(再生・教師の同一性・決まり / ユーザーの検査と同じセンサー・ラベルの検査)と、ばらつきの物差し・図。
#   bash record/after_record.sh <プランの ID>
# Node・Python とも jf-heavy --lock none・nice 15(Firefox・学習の邪魔をしない)。データは変えない。
set -u
PID=${1:?プランの ID}
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/record
DS=/home/jetson/jetformer/data/datasets
TEACHER=$DS/jetformer-simulator-id8-20min_20261012-150956_b55616cf
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 900 --"
cd $EXP
mkdir -p $OUT/replay $OUT/verify
NAMES=$($PY -I -c "
import json,sys
p=json.load(open('/home/jetson/jetformer/data/sim_rl/plans/$PID.json'))
for k,v in p['imports'].items():
    if v.get('state')=='done': print(k+'='+v['dataset'])
")
echo "$(date -Is) データセット: $NAMES"
VARGS=""; RARGS=""
for kv in $NAMES; do
  k=${kv%%=*}; n=${kv#*=}
  echo "$(date -Is) replay_check $k $n"
  $H nice -n 15 node integrate/replay_check.mjs --dataset $DS/$n --out $OUT/replay/$k.json || echo "replay $k exit $?"
  echo "$(date -Is) verify $k"
  $H nice -n 15 $PY -I record/verify_mode_dataset.py $DS/$n --out $OUT/verify/$k.json || echo "verify $k exit $?"
  VARGS="$VARGS --dataset $k=$DS/$n --replay $k=$OUT/replay/$k.json"
  RARGS="$RARGS --replay $k=$OUT/replay/$k.json"
done
echo "$(date -Is) variation"
$H nice -n 15 node record/variation.mjs $VARGS --teacher $TEACHER --nominal-recover $OUT/nominal_line_recover.json \
  --out $OUT/variation_raw.json --figdata $OUT/figdata.json
echo "$(date -Is) figs"
$H nice -n 15 $PY -I record/record_figs.py --figdata $OUT/figdata.json --variation $OUT/variation_raw.json \
  --outdir /home/jetson/jetformer_reports/sim_rl_modes_20261013/fig $RARGS
echo "$(date -Is) 終わり"
