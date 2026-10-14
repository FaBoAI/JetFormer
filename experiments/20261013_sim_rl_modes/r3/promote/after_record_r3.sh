#!/bin/bash
# 昇格の段(2026-10-15): 記録した見本(no_wall・recover・line の r4)の確かめと、ばらつきの物差し・図。
#   bash r3/promote/after_record_r3.sh <プランの ID>
# 確かめ: replay_check_r4.mjs(開ループの再生・教師の同一性・r4 の決まり)・record/verify_mode_dataset.py(ユーザーの検査と同じセンサー・ラベル)・
#         r3/rc4/promote/check_recorded_scenes_rc4.mjs(記録そのものから止まり・進み・ゲートの外しを数え直す。3 モードとも)
# 物差し: record/variation.mjs を 2 回(5 モード = fastest・speed は記録の段の 1125 + r4 の 3 つ / 記録の段と同じ 4 モードの組み合わせ)
# Node・Python とも jf-heavy --lock none・nice 15。データは変えない。
set -u
PID=${1:?プランの ID}
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
R3=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3
OUT=$R3/promote/record
DS=/home/jetson/jetformer/data/datasets
TEACHER=$DS/jetformer-simulator-id8-20min_20261012-150956_b55616cf
OLD=/home/jetson/jetformer_reports/sim_rl_modes_20261013/record
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 900 --"
cd $EXP
mkdir -p $OUT/replay $OUT/verify $OUT/check_recorded $R3/promote/fig_tmp
NAMES=$($PY -I -c "
import json
p=json.load(open('/home/jetson/jetformer/data/sim_rl/plans/$PID.json'))
for k,v in p['imports'].items():
    if v.get('state')=='done': print(k+'='+v['dataset'])
")
echo "$(date -Is) データセット: $NAMES"
declare -A NAME
for kv in $NAMES; do
  k=${kv%%=*}; n=${kv#*=}; NAME[$k]=$n
  if [ -s $OUT/replay/$k.json ] && [ -s $OUT/verify/$k.json ] && [ -s $OUT/check_recorded/$k.json ] && [ "${REDO:-0}" != 1 ]; then
    echo "$(date -Is) $k は確かめ済み(REDO=1 でやり直す)"; continue
  fi
  echo "$(date -Is) replay_check_r4 $k $n"
  $H nice -n 15 node r3/promote/replay_check_r4.mjs --dataset $DS/$n --out $OUT/replay/$k.json || echo "replay $k exit $?"
  echo "$(date -Is) verify $k"
  $H nice -n 15 $PY -I record/verify_mode_dataset.py $DS/$n --out $OUT/verify/$k.json || echo "verify $k exit $?"
  echo "$(date -Is) check_recorded $k"
  $H nice -n 15 node r3/rc4/promote/check_recorded_scenes_rc4.mjs $DS/$n --out $OUT/check_recorded/$k.json || echo "check_recorded $k exit $?"
done
FA=sim-rl-fastest_20261014-1125; SP=sim-rl-speed_20261014-1125
V5="--dataset fastest=$DS/$FA --replay fastest=$OLD/replay/fastest.json"
V4="$V5"
for k in no_wall recover; do
  [ -n "${NAME[$k]:-}" ] && V5="$V5 --dataset $k=$DS/${NAME[$k]} --replay $k=$OUT/replay/$k.json" && V4="$V4 --dataset $k=$DS/${NAME[$k]} --replay $k=$OUT/replay/$k.json"
done
V5="$V5 --dataset speed=$DS/$SP --replay speed=$OLD/replay/speed.json"; V4="$V4 --dataset speed=$DS/$SP --replay speed=$OLD/replay/speed.json"
[ -n "${NAME[line]:-}" ] && V5="$V5 --dataset line=$DS/${NAME[line]} --replay line=$OUT/replay/line.json"
echo "$(date -Is) variation 5 modes: $V5"
$H nice -n 15 node record/variation.mjs $V5 --teacher $TEACHER --nominal-recover $OUT/nominal_line_recover_r4.json \
  --out $OUT/variation_raw_5modes.json --figdata $OUT/figdata_5modes.json; echo "variation5 exit $?"
echo "$(date -Is) variation 4 modes: $V4"
$H nice -n 15 node record/variation.mjs $V4 --teacher $TEACHER --nominal-recover $OUT/nominal_line_recover_r4.json \
  --out $OUT/variation_raw_4modes.json; echo "variation4 exit $?"
RARGS=""; for f in $OUT/replay/*.json; do k=$(basename $f .json); RARGS="$RARGS --replay $k=$f"; done
RARGS="$RARGS --replay fastest=$OLD/replay/fastest.json --replay speed=$OLD/replay/speed.json"
echo "$(date -Is) figs"
$H nice -n 15 $PY -I r3/promote/record_figs_r3.py --figdata $OUT/figdata_5modes.json --variation $OUT/variation_raw_5modes.json \
  --outdir $R3/promote/fig_tmp $RARGS; echo "figs exit $?"
for f in $R3/promote/fig_tmp/record_*.png; do [ -f "$f" ] && cp -p "$f" "$R3/fig/promote_$(basename $f)"; done
echo "$(date -Is) variation.json"
$PY -I r3/promote/make_variation_json_r3.py $PID; echo "make_variation exit $?"
echo "$(date -Is) 終わり"
