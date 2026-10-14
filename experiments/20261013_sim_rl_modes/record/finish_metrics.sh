#!/bin/bash
# 記録の段: 4 つのデータセットのばらつきの物差しと図(確かめ(check_one.sh)のあと)。bash record/finish_metrics.sh <プランの ID>
set -u
PID=${1:?プランの ID}
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/record
DS=/home/jetson/jetformer/data/datasets
TEACHER=$DS/jetformer-simulator-id8-20min_20261012-150956_b55616cf
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 900 --"
cd $EXP
VARGS=""; RARGS=""
for kv in $($PY -I -c "
import json
p=json.load(open('/home/jetson/jetformer/data/sim_rl/plans/$PID.json'))
for k in p['modes']:
    v=p['imports'].get(k) or {}
    if v.get('state')=='done': print(k+'='+v['dataset'])
"); do
  k=${kv%%=*}; n=${kv#*=}
  VARGS="$VARGS --dataset $k=$DS/$n --replay $k=$OUT/replay/$k.json"
  RARGS="$RARGS --replay $k=$OUT/replay/$k.json"
done
echo "$(date -Is) variation $VARGS"
$H nice -n 15 node record/variation.mjs $VARGS --teacher $TEACHER --nominal-recover $OUT/nominal_line_recover.json \
  --out $OUT/variation_raw.json --figdata $OUT/figdata.json; echo "variation exit $?"
echo "$(date -Is) figs"
$H nice -n 15 $PY -I record/record_figs.py --figdata $OUT/figdata.json --variation $OUT/variation_raw.json \
  --outdir /home/jetson/jetformer_reports/sim_rl_modes_20261013/fig $RARGS; echo "figs exit $?"
echo "$(date -Is) 終わり"
